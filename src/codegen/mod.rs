//! Renaming pass + splicing into the `functions{}` block: the core
//! "compiler" step that turns a parsed `.laplace` file plus its resolved
//! packages into final `.stan` text.
//!
//! # Output shape
//!
//! Two modes, chosen by [`CodegenOptions::split_functions`]:
//!
//! - **inline** (default): every imported package's renamed functions are
//!   spliced into the top of the compiled file's `functions { }` block, so
//!   the `.stan` file is entirely self-contained. This is the portability
//!   guarantee and stays the default.
//! - **split**: each *directly imported* package gets its own
//!   `<pkg>.stanfunctions` file, and the `functions { }` block gets one
//!   `#include "<pkg>.stanfunctions"` line per package instead. Easier to
//!   read at scale, at the cost of shipping more than one file.
//!
//! # Transitive dependencies are flattened, never chained
//!
//! A package's own dependencies are emitted *alongside* it, into the same
//! `.stanfunctions` file -- laplace never generates an `#include` of one
//! `.stanfunctions` file from another. Each transitively resolved package is
//! emitted exactly once across the whole build (a diamond does not
//! duplicate the shared package), assigned to the first directly-imported
//! package that needs it, in dependency order. That keeps "which files must
//! ship next to my `.stan` file" answerable by looking at the `library { }`
//! block alone.
//!
//! # Encapsulation: imports are private
//!
//! A package may call `dep::func()` only for packages *it* declares. The
//! top-level project may call `pkg::func()` only for packages in its own
//! `library { }` block. A transitive dependency is in the build but is not
//! in scope for anyone that did not ask for it directly -- there is no
//! re-export mechanism in v1.
//!
//! # Mangling
//!
//! An exported function `foo` of package `pkg` becomes `pkg__foo`,
//! transitive dependencies included -- no hash, no version fragment. That is
//! safe (and stays readable) precisely because dependency resolution
//! guarantees at most one version of any package name per build, so a
//! package name is a unique prefix. Two versions of one package can never
//! coexist; see `resolve::graph`.
//!
//! A package's *non*-exported functions keep their original names -- they
//! are internal to a package but share the compiled file's single Stan
//! namespace, so two packages defining the same private helper are a hard
//! error ([`CodegenError::PrivateFunctionCollision`]) rather than a silent
//! clobber.

pub mod rename;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::Range;

use thiserror::Error;

use crate::parser::brace_match::CodeMask;
use crate::parser::library_block::{ImportStatement, LibraryBlock};
use crate::parser::signatures::FunctionSig;
use rename::{find_qualified_calls, mangle, rename_identifier_calls};

/// A resolved, installed package ready for codegen: its raw source (every
/// function it defines, exported or private, with any `library { }` block
/// already stripped by the loader), the extracted signatures, which of
/// those are exported (from the package's own `laplace.toml`), and which
/// other packages it imports.
#[derive(Debug, Clone)]
pub struct InstalledPackage {
    pub name: String,
    pub version: String,
    pub source: String,
    pub signatures: Vec<FunctionSig>,
    pub exported: Vec<String>,
    /// Names of the packages this one imports. Every entry must itself
    /// appear in the `installed` slice handed to codegen.
    pub dependencies: Vec<String>,
}

impl InstalledPackage {
    /// A leaf package: no dependencies of its own. Convenience for tests
    /// and for the common single-level case.
    pub fn leaf(
        name: impl Into<String>,
        version: impl Into<String>,
        source: impl Into<String>,
        exported: Vec<String>,
    ) -> Self {
        let source = source.into();
        let signatures = crate::parser::signatures::extract_signatures(&source);
        InstalledPackage {
            name: name.into(),
            version: version.into(),
            source,
            signatures,
            exported,
            dependencies: Vec::new(),
        }
    }

    /// The name this package's function `func` ends up with in compiled
    /// output: mangled if exported, unchanged if private.
    fn output_symbol(&self, func: &str) -> String {
        if self.exported.iter().any(|e| e == func) {
            mangle(&self.name, func)
        } else {
            func.to_string()
        }
    }
}

/// Knobs for [`generate_with_options`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodegenOptions {
    /// Emit `<pkg>.stanfunctions` files and `#include` them, instead of
    /// inlining imported functions into the `functions { }` block.
    pub split_functions: bool,
}

impl CodegenOptions {
    pub fn inline() -> Self {
        CodegenOptions {
            split_functions: false,
        }
    }

    pub fn split() -> Self {
        CodegenOptions {
            split_functions: true,
        }
    }
}

/// The extension laplace writes package function files with. Stan resolves
/// `#include` relative to the including file's directory (and any
/// `--include-paths`), and `stanc` parses a `.stanfunctions` file as a bare
/// sequence of function definitions -- no `functions { }` wrapper -- which
/// is exactly what these files contain.
pub const STANFUNCTIONS_EXTENSION: &str = "stanfunctions";

/// One `<pkg>.stanfunctions` file to write next to the compiled `.stan`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionFile {
    /// File name only, no directory: the compiled `.stan` file `#include`s
    /// it by this exact string.
    pub file_name: String,
    pub contents: String,
    /// Every package flattened into this file, in emission order. The first
    /// is the directly-imported package the file is named after; the rest
    /// are its private transitive dependencies.
    pub packages: Vec<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CodegenError {
    #[error(
        "`library` imports `{package}`, but no installed package by that name was provided \
         (run `laplace install`?)"
    )]
    MissingImport { package: String },

    #[error(
        "`{package}` depends on `{dependency}`, but no installed package by that name was \
         provided -- laplace.lock may be stale (run `laplace install`?)"
    )]
    MissingDependency { package: String, dependency: String },

    #[error("`{package}::{func}(` is called, but `{func}` is not in `{package}`'s exports")]
    FunctionNotExported { package: String, func: String },

    #[error(
        "`{package}::{func}(` is called, but `{package}` was not imported in the `library {{ }}` block"
    )]
    UnknownPackageReference { package: String, func: String },

    #[error(
        "package `{in_package}` calls `{package}::{func}(`, but `{in_package}` does not import \
         `{package}` -- a package's imports are private to it, so `{in_package}` cannot borrow \
         someone else's dependency"
    )]
    UndeclaredPackageReference {
        in_package: String,
        package: String,
        func: String,
    },

    #[error(
        "package `{package}` declares export `{func}`, but no such function was found in its source"
    )]
    ExportedFunctionMissing { package: String, func: String },

    #[error(
        "packages `{first_package}` and `{second_package}` both mangle to `{mangled}` -- this \
         should be impossible given the package-name prefix and indicates a naming collision \
         between the two packages"
    )]
    DuplicateMangledName {
        mangled: String,
        first_package: String,
        second_package: String,
    },

    #[error(
        "packages `{first_package}` and `{second_package}` both define a non-exported function \
         `{func}`, which would collide in the compiled model (Stan has one flat function \
         namespace) -- export it from one of them so it gets a `pkg__` prefix, or rename it"
    )]
    PrivateFunctionCollision {
        func: String,
        first_package: String,
        second_package: String,
    },

    #[error(
        "packages `{first_package}` and `{second_package}` would both be written to \
         `{file_name}` -- one would silently overwrite the other"
    )]
    FunctionFileCollision {
        file_name: String,
        first_package: String,
        second_package: String,
    },
}

/// Which imported package (if any) contributed a given range of lines in
/// the compiled `.stan` output -- a best-effort splice map, used by the
/// (optional) `stanc` validation pass to guess which package an error near
/// a given line likely came from. 1-indexed, inclusive on both ends,
/// matching how `stanc` reports line numbers.
///
/// Empty in `--split-functions` mode: package code lives in its own files
/// there, so a line in the compiled `.stan` never belongs to a package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageLineRange {
    pub package: String,
    pub lines: std::ops::RangeInclusive<usize>,
}

/// The result of [`generate_with_options`]: the compiled `.stan` text, the
/// `.stanfunctions` files that must be written alongside it (empty unless
/// split mode is on), and where each imported package's spliced-in code
/// ended up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedStan {
    pub source: String,
    pub package_line_ranges: Vec<PackageLineRange>,
    pub function_files: Vec<FunctionFile>,
}

/// Compile a `.laplace` source file's text into final `.stan` text.
///
/// `library_block` is the parsed `library { }` block, if the source has one.
/// `installed` are the resolved packages available for those imports *plus
/// every transitive dependency* -- codegen does not itself consult the
/// lockfile or filesystem, and does not re-parse the library block.
///
/// Packages are emitted in dependency-first order (a package always follows
/// everything it depends on); packages with no ordering relation between
/// them keep the order they appear in `installed`, so output stays
/// byte-identical run to run.
pub fn generate(
    source: &str,
    library_block: Option<&LibraryBlock>,
    installed: &[InstalledPackage],
) -> Result<String, CodegenError> {
    generate_impl(source, library_block, installed, &CodegenOptions::inline())
        .map(|(generated, _)| generated.source)
}

/// Same as [`generate`], but also reports the `.stanfunctions` files and
/// which output lines came from which imported package.
pub fn generate_with_package_lines(
    source: &str,
    library_block: Option<&LibraryBlock>,
    installed: &[InstalledPackage],
) -> Result<GeneratedStan, CodegenError> {
    generate_with_options(source, library_block, installed, &CodegenOptions::inline())
}

/// Full-control entry point: pick inline or split output.
pub fn generate_with_options(
    source: &str,
    library_block: Option<&LibraryBlock>,
    installed: &[InstalledPackage],
    options: &CodegenOptions,
) -> Result<GeneratedStan, CodegenError> {
    generate_impl(source, library_block, installed, options).map(|(generated, _)| generated)
}

/// Same as [`generate_with_options`], but also returns a [`SourceMap`] from
/// byte offsets in the compiled output back to the original `.laplace`
/// source -- for relocating live `stanc` diagnostics onto the file the user
/// is actually editing.
pub fn generate_with_source_map(
    source: &str,
    library_block: Option<&LibraryBlock>,
    installed: &[InstalledPackage],
    options: &CodegenOptions,
) -> Result<(GeneratedStan, SourceMap), CodegenError> {
    generate_impl(source, library_block, installed, options)
}

/// Maps a byte offset in generated `.stan` output back to the corresponding
/// byte offset in the original `.laplace` source. `None` for output text
/// that has no single corresponding source position: the boilerplate
/// `functions { }` wrapper codegen synthesizes, and code spliced in from an
/// imported package (that byte belongs to the package's own source, not
/// this file). A renamed `pkg::func` call site maps its whole span back to
/// the start of the original call -- approximate, but exact enough to place
/// a diagnostic on the right line.
#[derive(Debug, Clone)]
pub struct SourceMap {
    // Sorted, contiguous, non-overlapping output ranges, in increasing
    // order -- built in one pass over the same edits `apply_edits` applies.
    segments: Vec<(Range<usize>, SegmentKind)>,
}

#[derive(Debug, Clone, Copy)]
enum SegmentKind {
    /// Exact 1:1 copy from source starting at this original offset -- add
    /// the in-segment delta to get the exact original offset.
    Copied(usize),
    /// Every offset in this output range maps to this same single original
    /// offset (used for edited spans, e.g. a renamed `pkg::func` call,
    /// where column-for-column mapping isn't meaningful).
    Approx(usize),
    /// No corresponding position in the original file.
    Unmapped,
}

impl SourceMap {
    pub fn map(&self, output_offset: usize) -> Option<usize> {
        let idx = self.segments.partition_point(|(r, _)| r.end <= output_offset);
        let (range, kind) = self.segments.get(idx)?;
        if output_offset < range.start || output_offset > range.end {
            return None;
        }
        match *kind {
            SegmentKind::Copied(orig_start) => Some(orig_start + (output_offset - range.start)),
            SegmentKind::Approx(orig) => Some(orig),
            SegmentKind::Unmapped => None,
        }
    }
}

fn generate_impl(
    source: &str,
    library_block: Option<&LibraryBlock>,
    installed: &[InstalledPackage],
    options: &CodegenOptions,
) -> Result<(GeneratedStan, SourceMap), CodegenError> {
    let imports: &[ImportStatement] = library_block.map(|b| b.imports.as_slice()).unwrap_or(&[]);

    let by_name: BTreeMap<&str, &InstalledPackage> =
        installed.iter().map(|p| (p.name.as_str(), p)).collect();

    for import in imports {
        if !by_name.contains_key(import.name.as_str()) {
            return Err(CodegenError::MissingImport {
                package: import.name.clone(),
            });
        }
    }
    // The packages this build actually needs: everything reachable from the
    // `library { }` block's imports. Anything else in `installed` (a stale
    // cache entry, say) is ignored rather than silently compiled in -- and is
    // not validated either, since its problems can't reach this output.
    let direct: BTreeSet<&str> = imports.iter().map(|i| i.name.as_str()).collect();
    let reachable = reachable_from(&direct, &by_name);
    let ordered: Vec<&InstalledPackage> = topological_order(installed, &reachable, &by_name);

    for pkg in &ordered {
        for dep in &pkg.dependencies {
            if !by_name.contains_key(dep.as_str()) {
                return Err(CodegenError::MissingDependency {
                    package: pkg.name.clone(),
                    dependency: dep.clone(),
                });
            }
        }

        let sig_names: HashSet<&str> = pkg.signatures.iter().map(|s| s.name.as_str()).collect();
        for export in &pkg.exported {
            if !sig_names.contains(export.as_str()) {
                return Err(CodegenError::ExportedFunctionMissing {
                    package: pkg.name.clone(),
                    func: export.clone(),
                });
            }
        }
    }

    check_symbol_collisions(&ordered)?;

    // Rewrite each package's own source: first its `dep::func(` call sites
    // (validated against *its* declared imports -- imports are private),
    // then its own exported names into `pkg__name`.
    let mut rewritten: BTreeMap<&str, String> = BTreeMap::new();
    for pkg in &ordered {
        let mut text = pkg.source.clone();

        let mut call_edits: Vec<Edit> = Vec::new();
        for call in find_qualified_calls(&text) {
            if !pkg.dependencies.iter().any(|d| d == &call.package) {
                return Err(CodegenError::UndeclaredPackageReference {
                    in_package: pkg.name.clone(),
                    package: call.package,
                    func: call.func,
                });
            }
            let target = by_name
                .get(call.package.as_str())
                .expect("dependency presence was checked above");
            if !target.exported.iter().any(|e| e == &call.func) {
                return Err(CodegenError::FunctionNotExported {
                    package: call.package,
                    func: call.func,
                });
            }
            call_edits.push(Edit {
                range: call.range.clone(),
                replacement: mangle(&call.package, &call.func),
                splice_offset: None,
            });
        }
        text = apply_edits(&text, call_edits).0;

        let mut exported_sorted = pkg.exported.clone();
        exported_sorted.sort();
        for export in &exported_sorted {
            text = rename_identifier_calls(&text, export, &mangle(&pkg.name, export));
        }

        rewritten.insert(pkg.name.as_str(), text);
    }

    let mut edits: Vec<Edit> = Vec::new();

    if let Some(block) = library_block {
        edits.push(Edit {
            range: block.byte_range.clone(),
            replacement: String::new(),
            splice_offset: None,
        });
    }

    // The project's own `pkg::func(` call sites. Only packages named in the
    // `library { }` block are in scope here -- a transitive dependency the
    // project never imported is not callable from the project.
    for call in find_qualified_calls(source) {
        if let Some(block) = library_block {
            if block.byte_range.start <= call.range.start && call.range.end <= block.byte_range.end
            {
                continue;
            }
        }

        if !direct.contains(call.package.as_str()) {
            return Err(CodegenError::UnknownPackageReference {
                package: call.package,
                func: call.func,
            });
        }
        let target = by_name
            .get(call.package.as_str())
            .expect("direct imports were checked above");
        if !target.exported.iter().any(|e| e == &call.func) {
            return Err(CodegenError::FunctionNotExported {
                package: call.package,
                func: call.func,
            });
        }

        edits.push(Edit {
            range: call.range.clone(),
            replacement: mangle(&call.package, &call.func),
            splice_offset: None,
        });
    }

    let (assembled, package_ranges_in_assembled, function_files) = if options.split_functions {
        let files = build_function_files(&ordered, &direct, &rewritten)?;
        let includes: String = files
            .iter()
            .map(|f| format!("#include \"{}\"\n", f.file_name))
            .collect();
        (includes, Vec::new(), files)
    } else {
        let (text, ranges) = assemble_inline(&ordered, &rewritten);
        (text, ranges, Vec::new())
    };

    if !ordered.is_empty() {
        match find_functions_block(source) {
            Some(fb) => {
                let prefix = "\n";
                edits.push(Edit {
                    range: fb.open_brace + 1..fb.open_brace + 1,
                    replacement: format!("{prefix}{assembled}\n"),
                    splice_offset: Some(prefix.len()),
                });
            }
            None => {
                let prefix = "functions {\n";
                edits.push(Edit {
                    range: 0..0,
                    replacement: format!("{prefix}{assembled}\n}}\n"),
                    splice_offset: Some(prefix.len()),
                });
            }
        }
    }

    let (text, splice_start_in_output, source_map) = apply_edits(source, edits);

    let package_line_ranges = splice_start_in_output
        .map(|splice_start| {
            package_ranges_in_assembled
                .into_iter()
                .map(|(package, rel_range)| {
                    let abs_start = splice_start + rel_range.start;
                    let abs_end = splice_start + rel_range.end;
                    PackageLineRange {
                        package,
                        lines: line_at(&text, abs_start)..=line_at(&text, abs_end.saturating_sub(1)),
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    Ok((
        GeneratedStan {
            source: text,
            package_line_ranges,
            function_files,
        },
        source_map,
    ))
}

/// Everything reachable from the directly-imported packages, following each
/// package's own `dependencies`. Iterative, so a lockfile that somehow
/// describes a cycle can't blow the stack here (`resolve::graph` rejects
/// cycles long before codegen sees them).
fn reachable_from<'a>(
    direct: &BTreeSet<&'a str>,
    by_name: &BTreeMap<&'a str, &'a InstalledPackage>,
) -> BTreeSet<&'a str> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut queue: Vec<&str> = direct.iter().copied().collect();
    while let Some(name) = queue.pop() {
        if !seen.insert(name) {
            continue;
        }
        if let Some(pkg) = by_name.get(name) {
            for dep in &pkg.dependencies {
                queue.push(dep.as_str());
            }
        }
    }
    seen
}

/// Dependency-first ordering of the reachable packages: a package always
/// comes after everything it depends on, and packages with no ordering
/// relation keep their `installed` order. That tie-break is what keeps a
/// project with no transitive dependencies byte-identical to what earlier
/// laplace versions produced.
fn topological_order<'a>(
    installed: &'a [InstalledPackage],
    reachable: &BTreeSet<&str>,
    by_name: &BTreeMap<&'a str, &'a InstalledPackage>,
) -> Vec<&'a InstalledPackage> {
    let candidates: Vec<&InstalledPackage> = installed
        .iter()
        .filter(|p| reachable.contains(p.name.as_str()))
        .collect();

    let mut emitted: BTreeSet<&str> = BTreeSet::new();
    let mut out: Vec<&InstalledPackage> = Vec::new();

    loop {
        let mut progressed = false;
        for pkg in &candidates {
            if emitted.contains(pkg.name.as_str()) {
                continue;
            }
            let ready = pkg.dependencies.iter().all(|d| {
                emitted.contains(d.as_str()) || !by_name.contains_key(d.as_str())
            });
            if ready {
                emitted.insert(pkg.name.as_str());
                out.push(pkg);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    // A cycle would leave stragglers; append them in `installed` order so
    // output stays deterministic instead of silently dropping code.
    for pkg in candidates {
        if !emitted.contains(pkg.name.as_str()) {
            out.push(pkg);
        }
    }
    out
}

/// Every top-level function of every package ends up in one flat Stan
/// namespace. Exported functions are prefixed with their package name and
/// so can only collide pathologically; non-exported ones keep their source
/// names and collide easily. Either way, catch it here rather than emitting
/// a `.stan` file with two definitions of the same function.
fn check_symbol_collisions(ordered: &[&InstalledPackage]) -> Result<(), CodegenError> {
    let mut owner: BTreeMap<String, &str> = BTreeMap::new();

    for pkg in ordered {
        // Sorted, so which of two colliding packages is named "first" in
        // the error is stable across runs.
        let mut names: Vec<&str> = pkg.signatures.iter().map(|s| s.name.as_str()).collect();
        names.sort();
        names.dedup();

        for name in names {
            let symbol = pkg.output_symbol(name);
            let exported = symbol != name;
            match owner.get(&symbol) {
                Some(first) if *first != pkg.name.as_str() => {
                    return Err(if exported {
                        CodegenError::DuplicateMangledName {
                            mangled: symbol,
                            first_package: first.to_string(),
                            second_package: pkg.name.clone(),
                        }
                    } else {
                        CodegenError::PrivateFunctionCollision {
                            func: symbol,
                            first_package: first.to_string(),
                            second_package: pkg.name.clone(),
                        }
                    });
                }
                _ => {
                    owner.insert(symbol, pkg.name.as_str());
                }
            }
        }
    }
    Ok(())
}

/// Inline mode: one blob of every package's renamed source, in dependency
/// order, plus each package's byte range within that blob.
fn assemble_inline(
    ordered: &[&InstalledPackage],
    rewritten: &BTreeMap<&str, String>,
) -> (String, Vec<(String, Range<usize>)>) {
    let mut assembled = String::new();
    let mut ranges = Vec::new();
    for (i, pkg) in ordered.iter().enumerate() {
        if i > 0 {
            assembled.push('\n');
        }
        let start = assembled.len();
        assembled.push_str(&rewritten[pkg.name.as_str()]);
        ranges.push((pkg.name.clone(), start..assembled.len()));
    }
    (assembled, ranges)
}

/// Split mode: one `.stanfunctions` file per *directly imported* package,
/// with each transitive dependency flattened into the file of the first
/// direct import that needs it -- emitted exactly once across the build, so
/// a diamond dependency is not duplicated.
///
/// Files are ordered dependency-first, and so are the `#include` lines the
/// caller generates from them, so every function is defined before the file
/// that calls it.
fn build_function_files(
    ordered: &[&InstalledPackage],
    direct: &BTreeSet<&str>,
    rewritten: &BTreeMap<&str, String>,
) -> Result<Vec<FunctionFile>, CodegenError> {
    // Direct imports, dependency-first. Each owns a file named after it.
    let roots: Vec<&InstalledPackage> = ordered
        .iter()
        .copied()
        .filter(|p| direct.contains(p.name.as_str()))
        .collect();

    let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
    for root in &roots {
        owner.insert(root.name.as_str(), root.name.as_str());
    }
    // Walking roots dependency-first and claiming each unclaimed transitive
    // dependency means a package shared by two roots lands in the *earlier*
    // root's file -- which is included first, so its definitions precede
    // every use.
    let by_name: BTreeMap<&str, &InstalledPackage> =
        ordered.iter().map(|p| (p.name.as_str(), *p)).collect();
    for root in &roots {
        let mut root_set = BTreeSet::new();
        root_set.insert(root.name.as_str());
        for name in reachable_from(&root_set, &by_name) {
            owner.entry(name).or_insert(root.name.as_str());
        }
    }

    let mut files: Vec<FunctionFile> = Vec::new();
    let mut file_names: BTreeMap<String, String> = BTreeMap::new();

    for root in &roots {
        let file_name = format!("{}.{STANFUNCTIONS_EXTENSION}", root.name);
        if let Some(first) = file_names.get(&file_name) {
            return Err(CodegenError::FunctionFileCollision {
                file_name,
                first_package: first.clone(),
                second_package: root.name.clone(),
            });
        }
        file_names.insert(file_name.clone(), root.name.clone());

        let members: Vec<&InstalledPackage> = ordered
            .iter()
            .copied()
            .filter(|p| owner.get(p.name.as_str()) == Some(&root.name.as_str()))
            .collect();

        let mut contents = String::new();
        contents.push_str(&function_file_header(root, &members));
        for (i, pkg) in members.iter().enumerate() {
            if i > 0 {
                contents.push('\n');
            }
            contents.push_str(&rewritten[pkg.name.as_str()]);
        }

        files.push(FunctionFile {
            file_name,
            contents,
            packages: members.iter().map(|p| p.name.clone()).collect(),
        });
    }

    Ok(files)
}

/// A short, deterministic (no timestamps) provenance header for a generated
/// `.stanfunctions` file, so someone reading it knows where it came from
/// and that editing it is pointless.
fn function_file_header(root: &InstalledPackage, members: &[&InstalledPackage]) -> String {
    let mut header = format!(
        "// Generated by laplace from package `{}` {}. Do not edit.\n",
        root.name, root.version
    );
    let bundled: Vec<String> = members
        .iter()
        .filter(|p| p.name != root.name)
        .map(|p| format!("{} {}", p.name, p.version))
        .collect();
    if !bundled.is_empty() {
        header.push_str(&format!(
            "// Bundled dependencies of `{}`: {}.\n",
            root.name,
            bundled.join(", ")
        ));
    }
    header.push('\n');
    header
}

/// 1-indexed line number containing byte offset `at` in `text`.
fn line_at(text: &str, at: usize) -> usize {
    text.as_bytes()[..at].iter().filter(|&&b| b == b'\n').count() + 1
}

struct Edit {
    range: Range<usize>,
    replacement: String,
    /// If this edit splices in the assembled imported-function text, the
    /// byte offset within `replacement` where that text starts.
    splice_offset: Option<usize>,
}

/// Apply a set of non-overlapping edits (in original-source byte offsets) in
/// one linear pass, so offsets computed up front never need adjusting for
/// earlier edits. Returns the compiled text, the byte offset in that text
/// where the spliced package text begins (if one of the edits was the
/// import-splice insertion), and a [`SourceMap`] back to `source`.
fn apply_edits(source: &str, mut edits: Vec<Edit>) -> (String, Option<usize>, SourceMap) {
    edits.sort_by_key(|e| (e.range.start, e.range.end));

    let mut out = String::with_capacity(source.len());
    let mut cursor = 0usize;
    let mut splice_start_in_output = None;
    let mut segments: Vec<(Range<usize>, SegmentKind)> = Vec::new();

    for edit in edits {
        if edit.range.start < cursor {
            // Overlapping edits shouldn't occur given how callers build
            // them; skip defensively rather than corrupt the output.
            continue;
        }

        if edit.range.start > cursor {
            let copy_start = out.len();
            out.push_str(&source[cursor..edit.range.start]);
            segments.push((copy_start..out.len(), SegmentKind::Copied(cursor)));
        }

        let replacement_start = out.len();
        if let Some(offset) = edit.splice_offset {
            splice_start_in_output = Some(out.len() + offset);
        }
        out.push_str(&edit.replacement);
        if !edit.replacement.is_empty() {
            // Synthesized text (the import-splice insertion) has no source
            // position to map back to; everything else -- a renamed
            // `pkg::func` call being the only other kind of replacement --
            // maps approximately to where the original text it replaced
            // started.
            let kind = if edit.splice_offset.is_some() {
                SegmentKind::Unmapped
            } else {
                SegmentKind::Approx(edit.range.start)
            };
            segments.push((replacement_start..out.len(), kind));
        }

        cursor = edit.range.end;
    }

    if cursor < source.len() {
        let copy_start = out.len();
        out.push_str(&source[cursor..]);
        segments.push((copy_start..out.len(), SegmentKind::Copied(cursor)));
    }

    (out, splice_start_in_output, SourceMap { segments })
}

struct FunctionsBlock {
    open_brace: usize,
}

/// Find the user's own `functions { }` block, if present, using the same
/// comment/string-aware brace matching as the signature scanner.
fn find_functions_block(source: &str) -> Option<FunctionsBlock> {
    let mask = CodeMask::new(source);
    let bytes = source.as_bytes();
    let mut search_from = 0;

    while let Some(rel) = source[search_from..].find("functions") {
        let start = search_from + rel;
        let end = start + "functions".len();
        let word_ok = mask.is_real(start)
            && (start == 0 || !crate::parser::brace_match::is_ident_char(bytes[start - 1]))
            && (end == bytes.len() || !crate::parser::brace_match::is_ident_char(bytes[end]));

        if word_ok {
            let after = &source[end..];
            let trimmed = after.trim_start();
            if trimmed.starts_with('{') {
                let open_brace = end + (after.len() - trimmed.len());
                if mask.is_real(open_brace) && mask.match_closing_brace(source, open_brace).is_some()
                {
                    return Some(FunctionsBlock { open_brace });
                }
            }
        }

        search_from = end;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::library_block::parse_library_block;
    use crate::parser::signatures::extract_signatures;

    const GPS_SOURCE: &str = r#"// @laplace
// @brief Squared exponential (RBF) covariance matrix.
// @param x Vector of input locations.
// @param alpha Marginal standard deviation of the GP.
// @param rho Length-scale of the GP.
// @return An N x N positive semi-definite covariance matrix.
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}
"#;

    fn gps_package() -> InstalledPackage {
        InstalledPackage::leaf("gps", "1.0.0", GPS_SOURCE, vec!["rbf_cov".to_string()])
    }

    /// A package that imports another one: `source` may contain
    /// `dep::func(...)` calls.
    fn lib_package(
        name: &str,
        source: &str,
        exported: &[&str],
        dependencies: &[&str],
    ) -> InstalledPackage {
        InstalledPackage {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            source: source.to_string(),
            signatures: extract_signatures(source),
            exported: exported.iter().map(|s| s.to_string()).collect(),
            dependencies: dependencies.iter().map(|s| s.to_string()).collect(),
        }
    }

    // ---- existing single-level behaviour (must not regress) -------------

    #[test]
    fn end_to_end_gps_rbf_cov_example_synthesizes_functions_block() {
        let laplace_source = r#"library {
  import gps@1.0.0
}

data {
  int<lower=1> N;
  vector[N] x;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> rho;
}

model {
  matrix[N, N] K = gps::rbf_cov(x, alpha, rho);
  x ~ multi_normal(rep_vector(0, N), K);
}
"#;

        let block = parse_library_block(laplace_source).unwrap();
        let installed = vec![gps_package()];

        let output_1 = generate(laplace_source, block.as_ref(), &installed).unwrap();
        let output_2 = generate(laplace_source, block.as_ref(), &installed).unwrap();
        assert_eq!(output_1, output_2, "codegen must be deterministic");

        assert!(!output_1.contains("library"));
        assert!(!output_1.contains("gps::rbf_cov"));
        assert!(output_1.contains("gps__rbf_cov"));

        let byte_range = block.unwrap().byte_range;
        let expected_functions_block = format!(
            "functions {{\n{}\n}}\n",
            GPS_SOURCE.replace("rbf_cov", "gps__rbf_cov")
        );
        let expected_tail = format!(
            "{}{}",
            &laplace_source[..byte_range.start],
            &laplace_source[byte_range.end..]
        )
        .replace("gps::rbf_cov", "gps__rbf_cov");
        assert_eq!(output_1, format!("{expected_functions_block}{expected_tail}"));
    }

    #[test]
    fn prepends_into_an_existing_functions_block_in_installed_order() {
        let laplace_source = r#"library {
  import beta
  import alpha
}

functions {
  real user_fn(real x) {
    return x;
  }
}

model {
}
"#;

        let alpha = InstalledPackage::leaf(
            "alpha",
            "1.0.0",
            "real a_fn(real x) {\n  return x;\n}\n",
            vec!["a_fn".to_string()],
        );
        let beta = InstalledPackage::leaf(
            "beta",
            "1.0.0",
            "real b_fn(real x) {\n  return x;\n}\n",
            vec!["b_fn".to_string()],
        );

        // `installed` is given in lockfile (alphabetical) order: alpha, beta
        // -- even though the library block imports beta first.
        let block = parse_library_block(laplace_source).unwrap();
        let output = generate(laplace_source, block.as_ref(), &[alpha, beta]).unwrap();

        let alpha_pos = output.find("alpha__a_fn").unwrap();
        let beta_pos = output.find("beta__b_fn").unwrap();
        let user_pos = output.find("user_fn").unwrap();
        assert!(
            alpha_pos < beta_pos && beta_pos < user_pos,
            "expected alpha before beta before the user's own function, got:\n{output}"
        );
    }

    #[test]
    fn package_line_ranges_point_at_each_packages_spliced_lines() {
        let laplace_source = r#"library {
  import beta
  import alpha
}

functions {
  real user_fn(real x) {
    return x;
  }
}

model {
}
"#;

        let alpha = InstalledPackage::leaf(
            "alpha",
            "1.0.0",
            "real a_fn(real x) {\n  return x;\n}\n",
            vec!["a_fn".to_string()],
        );
        let beta = InstalledPackage::leaf(
            "beta",
            "1.0.0",
            "real b_fn(real x) {\n  return x;\n}\n",
            vec!["b_fn".to_string()],
        );

        let block = parse_library_block(laplace_source).unwrap();
        let generated =
            generate_with_package_lines(laplace_source, block.as_ref(), &[alpha, beta]).unwrap();
        assert_eq!(generated.package_line_ranges.len(), 2);

        let lines: Vec<&str> = generated.source.lines().collect();
        let alpha_range = generated
            .package_line_ranges
            .iter()
            .find(|r| r.package == "alpha")
            .unwrap();
        let beta_range = generated
            .package_line_ranges
            .iter()
            .find(|r| r.package == "beta")
            .unwrap();

        assert!(alpha_range.lines.end() < beta_range.lines.start());

        for line_no in alpha_range.lines.clone() {
            let line = lines[line_no - 1];
            assert!(!line.contains("beta__") && !line.contains("user_fn"), "{line}");
        }
        for line_no in beta_range.lines.clone() {
            let line = lines[line_no - 1];
            assert!(!line.contains("alpha__") && !line.contains("user_fn"), "{line}");
        }
        assert!(lines[*alpha_range.lines.start() - 1].contains("alpha__a_fn"));
        assert!(lines[*beta_range.lines.start() - 1].contains("beta__b_fn"));
    }

    #[test]
    fn package_line_ranges_empty_when_nothing_is_imported() {
        let source = "data {\n  int n;\n}\nmodel {\n}\n";
        let generated = generate_with_package_lines(source, None, &[]).unwrap();
        assert_eq!(generated.source, source);
        assert!(generated.package_line_ranges.is_empty());
    }

    #[test]
    fn no_library_block_and_no_imports_is_a_pure_passthrough() {
        let source = "data {\n  int n;\n}\nmodel {\n}\n";
        let output = generate(source, None, &[]).unwrap();
        assert_eq!(output, source);
    }

    #[test]
    fn empty_library_block_just_gets_deleted() {
        let source = "library {}\ndata {\n  int n;\n}\n";
        let block = parse_library_block(source).unwrap();
        let output = generate(source, block.as_ref(), &[]).unwrap();
        assert_eq!(output, "\ndata {\n  int n;\n}\n");
    }

    #[test]
    fn missing_import_is_an_error() {
        let source = "library {\n  import gps\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::MissingImport {
                package: "gps".to_string()
            }
        );
    }

    #[test]
    fn calling_a_non_exported_function_is_an_error() {
        let source = r#"library {
  import gps
}
model {
  real y = gps::secret_helper(1.0);
}
"#;
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[gps_package()]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::FunctionNotExported {
                package: "gps".to_string(),
                func: "secret_helper".to_string(),
            }
        );
    }

    #[test]
    fn calling_an_unimported_package_is_an_error() {
        let source = "model {\n  real y = gps::rbf_cov(1.0);\n}\n";
        let err = generate(source, None, &[gps_package()]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::UnknownPackageReference {
                package: "gps".to_string(),
                func: "rbf_cov".to_string(),
            }
        );
    }

    #[test]
    fn declared_export_missing_from_source_is_an_error() {
        let pkg = InstalledPackage::leaf(
            "gps",
            "1.0.0",
            "real rbf_cov(real x) {\n  return x;\n}\n",
            vec!["rbf_cov".to_string(), "matern_cov".to_string()],
        );
        let source = "library {\n  import gps\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[pkg]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::ExportedFunctionMissing {
                package: "gps".to_string(),
                func: "matern_cov".to_string(),
            }
        );
    }

    #[test]
    fn duplicate_mangled_name_across_packages_is_an_error() {
        // "a" exporting "b__c" and "a__b" exporting "c" both mangle to
        // "a__b__c" -- a contrived but real collision the sanity check must
        // catch.
        let pkg_a = InstalledPackage::leaf(
            "a",
            "1.0.0",
            "real b__c(real x) {\n  return x;\n}\n",
            vec!["b__c".to_string()],
        );
        let pkg_a_b = InstalledPackage::leaf(
            "a__b",
            "1.0.0",
            "real c(real x) {\n  return x;\n}\n",
            vec!["c".to_string()],
        );

        let source = "library {\n  import a\n  import a__b\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[pkg_a, pkg_a_b]).unwrap_err();
        assert!(matches!(err, CodegenError::DuplicateMangledName { .. }));
    }

    #[test]
    fn two_packages_with_the_same_private_helper_are_a_loud_error() {
        // Neither `_scale` is exported, so neither gets a `pkg__` prefix and
        // both would be defined in the one flat Stan namespace.
        let left = InstalledPackage::leaf(
            "left",
            "1.0.0",
            "real scale_(real x) {\n  return x;\n}\nreal l(real x) {\n  return scale_(x);\n}\n",
            vec!["l".to_string()],
        );
        let right = InstalledPackage::leaf(
            "right",
            "1.0.0",
            "real scale_(real x) {\n  return 2 * x;\n}\nreal r(real x) {\n  return scale_(x);\n}\n",
            vec!["r".to_string()],
        );

        let source = "library {\n  import left\n  import right\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[left, right]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::PrivateFunctionCollision {
                func: "scale_".to_string(),
                first_package: "left".to_string(),
                second_package: "right".to_string(),
            }
        );
    }

    #[test]
    fn one_package_may_define_two_overloads_of_the_same_private_name() {
        let pkg = InstalledPackage::leaf(
            "over",
            "1.0.0",
            "real h(real x) {\n  return x;\n}\nreal h(vector x) {\n  return x[1];\n}\nreal f(real x) {\n  return h(x);\n}\n",
            vec!["f".to_string()],
        );
        let source = "library {\n  import over\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        assert!(generate(source, block.as_ref(), &[pkg]).is_ok());
    }

    #[test]
    fn qualified_calls_inside_the_users_functions_block_are_rewritten_too() {
        let source = r#"library {
  import gps
}
functions {
  real wrapper(vector x, real alpha, real rho) {
    return gps::rbf_cov(x, alpha, rho)[1, 1];
  }
}
model {
}
"#;
        let block = parse_library_block(source).unwrap();
        let output = generate(source, block.as_ref(), &[gps_package()]).unwrap();
        assert!(output.contains("return gps__rbf_cov(x, alpha, rho)[1, 1];"));
    }

    #[test]
    fn source_map_is_identity_for_a_pure_passthrough() {
        let source = "data {\n  int n;\n}\nmodel {\n}\n";
        let (generated, source_map) =
            generate_with_source_map(source, None, &[], &CodegenOptions::inline()).unwrap();
        assert_eq!(generated.source, source);
        for offset in 0..source.len() {
            assert_eq!(source_map.map(offset), Some(offset));
        }
    }

    #[test]
    fn source_map_relocates_user_code_after_a_synthesized_functions_block() {
        let laplace_source = r#"library {
  import gps@1.0.0
}

model {
  matrix[1, 1] K = gps::rbf_cov([1.0], 1.0, 1.0);
}
"#;
        let block = parse_library_block(laplace_source).unwrap();
        let installed = vec![gps_package()];
        let (generated, source_map) = generate_with_source_map(
            laplace_source,
            block.as_ref(),
            &installed,
            &CodegenOptions::inline(),
        )
        .unwrap();

        let gen_model_at = generated.source.find("matrix[1, 1] K").unwrap();
        let orig_model_at = laplace_source.find("matrix[1, 1] K").unwrap();
        assert_eq!(source_map.map(gen_model_at), Some(orig_model_at));

        let gen_pkg_at = generated.source.find("gp_exp_quad_cov").unwrap();
        assert_eq!(source_map.map(gen_pkg_at), None);

        let gen_call_at = generated.source.find("gps__rbf_cov([1.0]").unwrap();
        let orig_call_at = laplace_source.find("gps::rbf_cov(").unwrap();
        assert_eq!(source_map.map(gen_call_at), Some(orig_call_at));
    }

    // ---- Feature A: --split-functions ----------------------------------

    const SPLIT_MODEL: &str = r#"library {
  import gps
}

data {
  int<lower=1> N;
  vector[N] x;
}

model {
  matrix[N, N] K = gps::rbf_cov(x, 1.0, 1.0);
}
"#;

    #[test]
    fn split_mode_emits_an_include_instead_of_the_function_bodies() {
        let block = parse_library_block(SPLIT_MODEL).unwrap();
        let generated = generate_with_options(
            SPLIT_MODEL,
            block.as_ref(),
            &[gps_package()],
            &CodegenOptions::split(),
        )
        .unwrap();

        assert!(generated.source.contains("#include \"gps.stanfunctions\""));
        // The bodies are gone from the .stan file...
        assert!(!generated.source.contains("gp_exp_quad_cov"));
        // ...but the call site is still rewritten.
        assert!(generated.source.contains("gps__rbf_cov(x, 1.0, 1.0)"));

        assert_eq!(generated.function_files.len(), 1);
        let file = &generated.function_files[0];
        assert_eq!(file.file_name, "gps.stanfunctions");
        assert_eq!(file.packages, vec!["gps"]);
        assert!(file.contents.contains("matrix gps__rbf_cov(vector x"));
        // A .stanfunctions file is bare definitions -- no wrapper block.
        assert!(!file.contents.contains("functions {"));
        assert!(file.contents.starts_with("// Generated by laplace"));
    }

    #[test]
    fn split_mode_is_deterministic() {
        let block = parse_library_block(SPLIT_MODEL).unwrap();
        let a = generate_with_options(
            SPLIT_MODEL,
            block.as_ref(),
            &[gps_package()],
            &CodegenOptions::split(),
        )
        .unwrap();
        let b = generate_with_options(
            SPLIT_MODEL,
            block.as_ref(),
            &[gps_package()],
            &CodegenOptions::split(),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn split_mode_is_a_no_op_for_a_project_with_no_imports() {
        let source = "data {\n  int n;\n}\nmodel {\n}\n";
        let inline =
            generate_with_options(source, None, &[], &CodegenOptions::inline()).unwrap();
        let split = generate_with_options(source, None, &[], &CodegenOptions::split()).unwrap();
        assert_eq!(inline.source, source);
        assert_eq!(split.source, source);
        assert!(split.function_files.is_empty());
    }

    #[test]
    fn split_mode_keeps_the_users_own_functions_inline() {
        let source = r#"library {
  import gps
}
functions {
  real user_fn(real x) {
    return x;
  }
}
model {
}
"#;
        let block = parse_library_block(source).unwrap();
        let generated = generate_with_options(
            source,
            block.as_ref(),
            &[gps_package()],
            &CodegenOptions::split(),
        )
        .unwrap();

        assert!(generated.source.contains("real user_fn(real x)"));
        assert!(generated.source.contains("#include \"gps.stanfunctions\""));
        let include_at = generated.source.find("#include").unwrap();
        let user_at = generated.source.find("user_fn").unwrap();
        assert!(include_at < user_at, "include must precede the user's functions");
    }

    #[test]
    fn a_multi_file_package_still_lands_in_exactly_one_stanfunctions_file() {
        // Multi-file packages are concatenated by the loader before codegen
        // ever sees them, so this is really asserting that concatenation
        // survives the trip into a split file unchanged.
        let concatenated = "real one() {\n  return 1;\n}\n\nreal two() {\n  return 2;\n}\n";
        let pkg = InstalledPackage::leaf(
            "multi",
            "1.0.0",
            concatenated,
            vec!["one".to_string(), "two".to_string()],
        );
        let source = "library {\n  import multi\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let generated =
            generate_with_options(source, block.as_ref(), &[pkg], &CodegenOptions::split()).unwrap();

        assert_eq!(generated.function_files.len(), 1);
        let contents = &generated.function_files[0].contents;
        assert!(contents.contains("real multi__one()"));
        assert!(contents.contains("real multi__two()"));
    }

    #[test]
    fn split_mode_reports_no_package_line_ranges() {
        // Package code isn't in the .stan file at all, so there is nothing
        // for the stanc annotator to point at.
        let block = parse_library_block(SPLIT_MODEL).unwrap();
        let generated = generate_with_options(
            SPLIT_MODEL,
            block.as_ref(),
            &[gps_package()],
            &CodegenOptions::split(),
        )
        .unwrap();
        assert!(generated.package_line_ranges.is_empty());
    }

    // ---- Feature B: transitive dependencies -----------------------------

    /// `stats` (leaf) <- `regression` (imports stats). The project imports
    /// only `regression`.
    fn chain_fixture() -> (InstalledPackage, InstalledPackage) {
        let stats = InstalledPackage::leaf(
            "stats",
            "1.0.0",
            "// @laplace\n// @brief Mean.\nreal mean_(vector x) {\n  return sum(x) / num_elements(x);\n}\n",
            vec!["mean_".to_string()],
        );
        let regression = lib_package(
            "regression",
            "// @laplace\n// @brief Centre a vector.\nvector centre(vector x) {\n  return x - stats::mean_(x);\n}\n",
            &["centre"],
            &["stats"],
        );
        (stats, regression)
    }

    const CHAIN_MODEL: &str = r#"library {
  import regression
}

data {
  int<lower=1> N;
  vector[N] y;
}

model {
  vector[N] c = regression::centre(y);
  c ~ std_normal();
}
"#;

    #[test]
    fn a_transitive_dependency_is_emitted_and_its_call_sites_mangled() {
        let (stats, regression) = chain_fixture();
        let block = parse_library_block(CHAIN_MODEL).unwrap();
        let output = generate(CHAIN_MODEL, block.as_ref(), &[regression, stats]).unwrap();

        assert!(output.contains("real stats__mean_(vector x)"));
        assert!(output.contains("vector regression__centre(vector x)"));
        // `regression`'s internal call to `stats` is mangled...
        assert!(output.contains("return x - stats__mean_(x);"));
        assert!(!output.contains("stats::"));
        // ...and the project's own call to `regression` too.
        assert!(output.contains("regression__centre(y)"));

        // Dependency-first: `stats` is defined before `regression` uses it,
        // regardless of the order `installed` came in.
        assert!(output.find("stats__mean_").unwrap() < output.find("regression__centre").unwrap());
    }

    #[test]
    fn transitive_ordering_does_not_depend_on_installed_order() {
        let (stats, regression) = chain_fixture();
        let block = parse_library_block(CHAIN_MODEL).unwrap();
        let a = generate(CHAIN_MODEL, block.as_ref(), &[stats.clone(), regression.clone()]).unwrap();
        let b = generate(CHAIN_MODEL, block.as_ref(), &[regression, stats]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn the_project_cannot_call_a_package_it_only_depends_on_transitively() {
        let (stats, regression) = chain_fixture();
        let source = r#"library {
  import regression
}
model {
  real m = stats::mean_([1.0]);
}
"#;
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[regression, stats]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::UnknownPackageReference {
                package: "stats".to_string(),
                func: "mean_".to_string(),
            }
        );
    }

    #[test]
    fn a_package_cannot_call_a_package_it_does_not_import() {
        let stats = InstalledPackage::leaf(
            "stats",
            "1.0.0",
            "real mean_(vector x) {\n  return sum(x);\n}\n",
            vec!["mean_".to_string()],
        );
        // `sneaky` calls stats:: without declaring it as a dependency.
        let sneaky = lib_package(
            "sneaky",
            "real f(vector x) {\n  return stats::mean_(x);\n}\n",
            &["f"],
            &[],
        );
        let source = "library {\n  import sneaky\n  import stats\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[sneaky, stats]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::UndeclaredPackageReference {
                in_package: "sneaky".to_string(),
                package: "stats".to_string(),
                func: "mean_".to_string(),
            }
        );
    }

    #[test]
    fn a_dependency_missing_from_installed_is_a_clear_error() {
        let (_, regression) = chain_fixture();
        let source = "library {\n  import regression\n}\nmodel {\n}\n";
        let block = parse_library_block(source).unwrap();
        let err = generate(source, block.as_ref(), &[regression]).unwrap_err();
        assert_eq!(
            err,
            CodegenError::MissingDependency {
                package: "regression".to_string(),
                dependency: "stats".to_string(),
            }
        );
    }

    #[test]
    fn a_diamond_emits_the_shared_package_exactly_once() {
        let (stats, regression) = chain_fixture();
        let source = r#"library {
  import regression
  import stats
}
model {
  real m = stats::mean_([1.0]);
  vector[1] c = regression::centre([1.0]);
}
"#;
        let block = parse_library_block(source).unwrap();
        let output = generate(source, block.as_ref(), &[regression, stats]).unwrap();

        assert_eq!(
            output.matches("real stats__mean_(vector x)").count(),
            1,
            "the shared package must be emitted once, got:\n{output}"
        );
    }

    #[test]
    fn a_diamond_in_split_mode_puts_the_shared_package_in_exactly_one_file() {
        let (stats, regression) = chain_fixture();
        let source = r#"library {
  import regression
  import stats
}
model {
  real m = stats::mean_([1.0]);
  vector[1] c = regression::centre([1.0]);
}
"#;
        let block = parse_library_block(source).unwrap();
        let generated = generate_with_options(
            source,
            block.as_ref(),
            &[regression, stats],
            &CodegenOptions::split(),
        )
        .unwrap();

        // One file per *direct* import.
        let names: Vec<&str> = generated
            .function_files
            .iter()
            .map(|f| f.file_name.as_str())
            .collect();
        assert_eq!(names, vec!["stats.stanfunctions", "regression.stanfunctions"]);

        let defined_in: Vec<&str> = generated
            .function_files
            .iter()
            .filter(|f| f.contents.contains("real stats__mean_(vector x)"))
            .map(|f| f.file_name.as_str())
            .collect();
        assert_eq!(defined_in, vec!["stats.stanfunctions"]);

        // `stats` is included before `regression`, which calls into it.
        let stats_at = generated.source.find("#include \"stats.").unwrap();
        let reg_at = generated.source.find("#include \"regression.").unwrap();
        assert!(stats_at < reg_at);
    }

    #[test]
    fn split_mode_flattens_a_private_transitive_dependency_into_its_parents_file() {
        let (stats, regression) = chain_fixture();
        // The project imports `regression` only, so `stats` has no file of
        // its own and is flattened into regression.stanfunctions.
        let block = parse_library_block(CHAIN_MODEL).unwrap();
        let generated = generate_with_options(
            CHAIN_MODEL,
            block.as_ref(),
            &[regression, stats],
            &CodegenOptions::split(),
        )
        .unwrap();

        assert_eq!(generated.function_files.len(), 1);
        let file = &generated.function_files[0];
        assert_eq!(file.file_name, "regression.stanfunctions");
        assert_eq!(file.packages, vec!["stats", "regression"]);
        assert!(file.contents.contains("real stats__mean_(vector x)"));
        assert!(file.contents.contains("vector regression__centre(vector x)"));
        assert!(file.contents.contains("Bundled dependencies of `regression`: stats 1.0.0."));
        // Dependency-first inside the file, too.
        assert!(
            file.contents.find("stats__mean_").unwrap()
                < file.contents.find("regression__centre").unwrap()
        );
    }

    #[test]
    fn a_deep_chain_orders_every_level_dependency_first() {
        let base = InstalledPackage::leaf(
            "base",
            "1.0.0",
            "real b() {\n  return 1;\n}\n",
            vec!["b".to_string()],
        );
        let mid = lib_package(
            "mid",
            "real m() {\n  return base::b();\n}\n",
            &["m"],
            &["base"],
        );
        let top = lib_package("top", "real t() {\n  return mid::m();\n}\n", &["t"], &["mid"]);

        let source = "library {\n  import top\n}\nmodel {\n  real y = top::t();\n}\n";
        let block = parse_library_block(source).unwrap();
        let output = generate(source, block.as_ref(), &[top, mid, base]).unwrap();

        let base_at = output.find("real base__b()").unwrap();
        let mid_at = output.find("real mid__m()").unwrap();
        let top_at = output.find("real top__t()").unwrap();
        assert!(base_at < mid_at && mid_at < top_at, "got:\n{output}");
        assert!(output.contains("return base__b();"));
        assert!(output.contains("return mid__m();"));
    }

    #[test]
    fn a_transitive_package_that_is_not_reachable_is_not_compiled_in() {
        let (stats, regression) = chain_fixture();
        let unrelated = InstalledPackage::leaf(
            "unrelated",
            "1.0.0",
            "real u() {\n  return 0;\n}\n",
            vec!["u".to_string()],
        );
        let block = parse_library_block(CHAIN_MODEL).unwrap();
        let output =
            generate(CHAIN_MODEL, block.as_ref(), &[regression, stats, unrelated]).unwrap();
        assert!(!output.contains("unrelated__u"));
    }
}
