//! The `laplace` CLI. All of the actual work lives in the library crate
//! (`src/lib.rs` and below); this file is argument parsing, filesystem
//! paths, and terminal output.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use laplace::codegen::{self, CodegenOptions};
use laplace::parser::library_block::{parse_library_block, ImportStatement};
use laplace::resolve::{self, Registry};
use laplace::{docs, init, manifest, package, validate};

#[derive(Parser)]
#[command(
    name = "laplace",
    about = "Source-to-source preprocessor for Stan: package manager + namespaces + doc lookup"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan the current directory's .stan files and generate a starter
    /// laplace.toml, guessing `exports` from @laplace-documented functions
    Init,
    /// Compile a .laplace file to .stan
    Build {
        file: PathBuf,
        /// Output file (default: build/<name>.stan). An existing directory,
        /// or a path ending in `/`, means `<dir>/<name>.stan`.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Diff against the existing build output instead of writing; exits
        /// non-zero if they differ.
        #[arg(long)]
        check: bool,
        /// After writing, type-check the output with `stanc` (must be on
        /// PATH). Off by default so build never requires stanc installed.
        #[arg(long)]
        validate: bool,
        /// Write each imported package's functions to its own
        /// `<pkg>.stanfunctions` file next to the output and `#include` it,
        /// instead of inlining every function into the `.stan` file. Off by
        /// default, so a plain build still produces one self-contained file.
        #[arg(long)]
        split_functions: bool,
    },
    /// Install every package pinned in laplace.lock
    Install,
    /// Add a dependency (optionally pinned to `<pkg>@<version>`), resolving
    /// it and updating laplace.toml + laplace.lock. Pass `--git <url>` with
    /// `--tag <tag>` or `--rev <rev>` to add a git dependency instead of
    /// resolving from the local registry, and `--subdir <path>` if that
    /// repository keeps the package below its top level.
    Add {
        package: String,
        #[arg(long)]
        git: Option<String>,
        #[arg(long)]
        tag: Option<String>,
        #[arg(long)]
        rev: Option<String>,
        /// Directory inside the git repository holding the package's
        /// laplace.toml (default: the repository root)
        #[arg(long)]
        subdir: Option<String>,
    },
    /// Re-resolve a dependency to the latest version matching its existing
    /// range in laplace.toml
    Update { package: String },
    /// Print a package function's signature and doc comment
    Doc {
        /// `<package>::<function>`
        spec: String,
        /// Render as a standalone HTML file instead of printing to the
        /// terminal: brief/params/return as plain text, the example in a
        /// <pre><code> block, and the math field (if any) rendered with
        /// KaTeX. Requires -o/--output.
        #[arg(long)]
        html: bool,
        /// Output path for --html
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), CliError> {
    match Cli::parse().command {
        Command::Init => cmd_init(),
        Command::Build {
            file,
            output,
            check,
            validate,
            split_functions,
        } => cmd_build(&file, output, check, validate, split_functions),
        Command::Install => cmd_install(),
        Command::Add {
            package,
            git,
            tag,
            rev,
            subdir,
        } => cmd_add(
            &package,
            git.as_deref(),
            tag.as_deref(),
            rev.as_deref(),
            subdir.as_deref(),
        ),
        Command::Update { package } => cmd_update(&package),
        Command::Doc { spec, html, output } => cmd_doc(&spec, html, output),
    }
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Init(#[from] init::InitError),
    #[error(transparent)]
    LibraryBlock(#[from] laplace::parser::library_block::LibraryBlockError),
    #[error(transparent)]
    Package(Box<package::PackageError>),
    #[error(transparent)]
    Codegen(#[from] codegen::CodegenError),
    #[error(transparent)]
    Resolve(Box<resolve::ResolveError>),
    #[error(transparent)]
    Manifest(#[from] manifest::ManifestError),
    #[error(transparent)]
    Lockfile(#[from] resolve::lockfile::LockfileError),
    #[error(transparent)]
    Docs(Box<docs::DocsError>),
    #[error(transparent)]
    Validate(Box<validate::ValidateError>),
    #[error("{0}")]
    Message(String),
    #[error(
        "{} is out of date with `laplace build` (run without --check to update it)",
        .0.display()
    )]
    CheckFailed(PathBuf),
}

// ResolveError is boxed above (clippy::result_large_err) -- write the
// conversion by hand so `?` still works directly on a plain ResolveError.
impl From<resolve::ResolveError> for CliError {
    fn from(err: resolve::ResolveError) -> Self {
        CliError::Resolve(Box::new(err))
    }
}

impl From<package::PackageError> for CliError {
    fn from(err: package::PackageError) -> Self {
        CliError::Package(Box::new(err))
    }
}

impl From<docs::DocsError> for CliError {
    fn from(err: docs::DocsError) -> Self {
        CliError::Docs(Box::new(err))
    }
}

impl From<validate::ValidateError> for CliError {
    fn from(err: validate::ValidateError) -> Self {
        CliError::Validate(Box::new(err))
    }
}

fn cmd_init() -> Result<(), CliError> {
    let dir = env::current_dir()?;
    let summary = init::init(&dir)?;

    println!("wrote laplace.toml for `{}`", summary.name);
    if let Some(dir_name) = &summary.renamed_from {
        println!(
            "note: the directory name `{dir_name}` is not a valid package name, so `{}` was used \
             instead -- {}",
            summary.name,
            manifest::PACKAGE_NAME_RULE,
        );
    }
    if summary.source_files == 0 {
        println!(
            "no .stan or .laplacelib files found in this directory -- exports is empty, add \
             entries by hand"
        );
    } else if summary.included.is_empty() {
        println!(
            "none of the functions in the {} source {} has a `// @laplace` doc comment -- \
             exports is empty, add entries by hand",
            summary.source_files,
            pluralize(summary.source_files, "file", "files"),
        );
    } else {
        println!(
            "included {} exported {}: {}",
            summary.included.len(),
            pluralize(summary.included.len(), "function", "functions"),
            summary.included.join(", "),
        );
    }
    if !summary.excluded.is_empty() {
        println!(
            "note: {} undocumented {} left out of exports (add manually if this guess is wrong): {}",
            summary.excluded.len(),
            pluralize(summary.excluded.len(), "function", "functions"),
            summary.excluded.join(", "),
        );
    }
    if !summary.imports.is_empty() {
        println!(
            "note: the sources import {} -- add {} under [dependencies] in laplace.toml",
            summary.imports.join(", "),
            pluralize(summary.imports.len(), "it", "them"),
        );
    }

    Ok(())
}

fn cmd_build(
    file: &Path,
    output: Option<PathBuf>,
    check: bool,
    validate: bool,
    split_functions: bool,
) -> Result<(), CliError> {
    let source = fs::read_to_string(file)?;
    let library_block = parse_library_block(&source)?;
    let imports: &[ImportStatement] = library_block
        .as_ref()
        .map(|b| b.imports.as_slice())
        .unwrap_or(&[]);

    let lockfile_path = PathBuf::from("laplace.lock");
    let cache_root = default_cache_root();
    let lock = resolve::lockfile::read_lockfile(&lockfile_path)?;
    let roots = lock.root_names();

    // Direct imports first, in `library { }` order, then every transitive
    // dependency sorted by name. Codegen re-orders dependency-first and
    // falls back to this order for packages with no ordering relation, so a
    // project with no transitive dependencies compiles exactly as it always
    // has.
    let mut order: Vec<String> = Vec::new();
    for import in imports {
        if lock.get(&import.name).is_none() {
            return Err(CliError::Message(format!(
                "`{}` is imported but not recorded in laplace.lock -- run `laplace add {}` first",
                import.name, import.name
            )));
        }
        if !roots.iter().any(|r| r == &import.name) {
            return Err(CliError::Message(format!(
                "`{}` is imported but is only a transitive dependency (some other package pulls \
                 it in) -- a package's imports are private, so run `laplace add {}` to depend on \
                 it directly",
                import.name, import.name
            )));
        }
        order.push(import.name.clone());
    }
    let direct_count = order.len();
    for locked in lock.closure(&order.clone()) {
        if !order.iter().any(|n| n == &locked.name) {
            order.push(locked.name.clone());
        }
    }

    let mut installed = Vec::with_capacity(order.len());
    for name in &order {
        let locked = lock.get(name).ok_or_else(|| {
            CliError::Message(format!(
                "`{name}` is required by another locked package but has no laplace.lock entry -- \
                 the lockfile is inconsistent; re-run `laplace add`"
            ))
        })?;
        let package_dir = cache_root.join(&locked.name).join(&locked.version);
        if !package_dir.is_dir() {
            return Err(CliError::Message(format!(
                "`{}@{}` is in laplace.lock but not installed -- run `laplace install` first",
                locked.name, locked.version
            )));
        }
        installed.push(package::load(&package_dir, &locked.name)?);
    }

    let options = CodegenOptions { split_functions };
    let generated =
        codegen::generate_with_options(&source, library_block.as_ref(), &installed, &options)?;
    let output_path = resolve_output_path(file, output);
    let output_dir = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    if check {
        let existing = fs::read_to_string(&output_path).unwrap_or_default();
        if existing != generated.source {
            return Err(CliError::CheckFailed(output_path));
        }
        for file in &generated.function_files {
            let path = output_dir.join(&file.file_name);
            if fs::read_to_string(&path).unwrap_or_default() != file.contents {
                return Err(CliError::CheckFailed(path));
            }
        }
        println!("{} is up to date", output_path.display());
        return Ok(());
    }

    fs::create_dir_all(&output_dir)?;
    fs::write(&output_path, &generated.source)?;
    for file in &generated.function_files {
        fs::write(output_dir.join(&file.file_name), &file.contents)?;
    }

    let lines = generated.source.lines().count();
    println!(
        "wrote {} ({} {}, {} {}{})",
        output_path.display(),
        lines,
        pluralize(lines, "line", "lines"),
        direct_count,
        pluralize(direct_count, "dependency", "dependencies"),
        match order.len() - direct_count {
            0 => String::new(),
            n => format!(" + {n} transitive"),
        },
    );

    for file in &generated.function_files {
        let path = output_dir.join(&file.file_name);
        println!(
            "wrote {} ({})",
            path.display(),
            file.packages.join(", "),
        );
    }
    if !generated.function_files.is_empty() {
        println!(
            "note: keep the .stanfunctions files next to {} and compile with the include path \
             set to that directory -- `stanc --include-paths={}`, or \
             `cmdstan_model(..., include_paths = \"{}\")` in cmdstanr",
            output_path.display(),
            output_dir.display(),
            output_dir.display(),
        );
    }

    if validate {
        validate::validate(&stanc_command(), &output_path, &generated)?;
        println!("stanc: OK");
    }

    Ok(())
}

/// Which `stanc` binary/command to invoke for `--validate`. Overridable via
/// `LAPLACE_STANC` so tests never depend on a real stanc install.
fn stanc_command() -> String {
    env::var("LAPLACE_STANC").unwrap_or_else(|_| "stanc".to_string())
}

fn default_output_path(file: &Path) -> PathBuf {
    PathBuf::from("build").join(output_file_name(file))
}

/// `-o/--output` names a file, but a directory is accepted too: an existing
/// directory, or a path spelled with a trailing separator, gets
/// `<name>.stan` appended rather than failing with a bare "Is a directory".
fn resolve_output_path(file: &Path, output: Option<PathBuf>) -> PathBuf {
    match output {
        None => default_output_path(file),
        Some(path) => {
            let names_a_dir = path.is_dir()
                || path
                    .as_os_str()
                    .to_str()
                    .is_some_and(|s| s.ends_with(std::path::MAIN_SEPARATOR) || s.ends_with('/'));
            if names_a_dir {
                path.join(output_file_name(file))
            } else {
                path
            }
        }
    }
}

fn output_file_name(file: &Path) -> String {
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    format!("{stem}.stan")
}

fn pluralize(count: usize, singular: &'static str, plural: &'static str) -> &'static str {
    if count == 1 {
        singular
    } else {
        plural
    }
}

fn cmd_install() -> Result<(), CliError> {
    let lockfile_path = PathBuf::from("laplace.lock");
    let registry = Registry::new(registry_root());
    let cache_root = default_cache_root();

    let installed = resolve::install(&lockfile_path, &registry, &cache_root)?;
    println!(
        "installed {} {}",
        installed.len(),
        pluralize(installed.len(), "package", "packages"),
    );
    Ok(())
}

fn cmd_add(
    spec: &str,
    git: Option<&str>,
    tag: Option<&str>,
    rev: Option<&str>,
    subdir: Option<&str>,
) -> Result<(), CliError> {
    let project_manifest_path = PathBuf::from("laplace.toml");
    let lockfile_path = PathBuf::from("laplace.lock");
    let cache_root = default_cache_root();

    let (name, _) = parse_package_spec(spec);
    if !manifest::is_valid_package_name(name) {
        return Err(CliError::Message(format!(
            "invalid package name `{name}`: {}",
            manifest::PACKAGE_NAME_RULE
        )));
    }

    if let Some(url) = git {
        let registry = Registry::new(registry_root());
        let locked = resolve::add_git(
            &project_manifest_path,
            &lockfile_path,
            &registry,
            &cache_root,
            spec,
            url,
            tag,
            rev,
            subdir,
        )?;
        println!("added {}@{} (git)", locked.name, locked.version);
        return Ok(());
    }

    if tag.is_some() || rev.is_some() || subdir.is_some() {
        return Err(CliError::Message(
            "--tag/--rev/--subdir only apply together with --git".to_string(),
        ));
    }

    let (name, version) = parse_package_spec(spec);
    let registry = Registry::new(registry_root());
    let locked = resolve::add(
        &project_manifest_path,
        &lockfile_path,
        &registry,
        &cache_root,
        name,
        version,
    )?;
    println!("added {}@{}", locked.name, locked.version);
    Ok(())
}

fn cmd_update(package: &str) -> Result<(), CliError> {
    let project_manifest_path = PathBuf::from("laplace.toml");
    let lockfile_path = PathBuf::from("laplace.lock");
    let registry = Registry::new(registry_root());
    let cache_root = default_cache_root();

    let locked = resolve::update(
        &project_manifest_path,
        &lockfile_path,
        &registry,
        &cache_root,
        package,
    )?;
    println!("updated {} to {}", locked.name, locked.version);
    Ok(())
}

fn cmd_doc(spec: &str, html: bool, output: Option<PathBuf>) -> Result<(), CliError> {
    let Some((package, func)) = spec.split_once("::") else {
        return Err(CliError::Message(format!(
            "expected `<package>::<function>`, got `{spec}`"
        )));
    };

    let lockfile_path = PathBuf::from("laplace.lock");
    let cache_root = default_cache_root();

    let overloads = docs::lookup(&lockfile_path, &cache_root, package, func)?;

    if html {
        let output = output
            .ok_or_else(|| CliError::Message("--html requires -o/--output <path>".to_string()))?;
        fs::write(&output, docs::render_html_overloads(package, &overloads))?;
        println!("wrote {}", output.display());
    } else {
        print!("{}", docs::render_overloads(package, &overloads));
    }
    Ok(())
}

/// Split `<pkg>` or `<pkg>@<version>` into its parts.
fn parse_package_spec(spec: &str) -> (&str, Option<&str>) {
    match spec.split_once('@') {
        Some((name, version)) => (name, Some(version)),
        None => (spec, None),
    }
}

fn laplace_home() -> PathBuf {
    let home = env::var_os("HOME").expect("HOME environment variable must be set");
    PathBuf::from(home).join(".laplace")
}

/// Where `laplace install`/`add`/`update` copy resolved packages, per the
/// design doc: `~/.laplace/packages/<name>/<version>/`.
fn default_cache_root() -> PathBuf {
    laplace_home().join("packages")
}

/// The local filesystem "registry" packages are resolved from (real
/// git/http fetching is a later task). Defaults to `~/.laplace/registry`,
/// overridable with `LAPLACE_REGISTRY` -- mainly so tests don't have to
/// touch the real home directory.
fn registry_root() -> PathBuf {
    env::var_os("LAPLACE_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|| laplace_home().join("registry"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_and_pinned_package_specs() {
        assert_eq!(parse_package_spec("gps"), ("gps", None));
        assert_eq!(
            parse_package_spec("gps@1.0.0"),
            ("gps", Some("1.0.0"))
        );
    }

    #[test]
    fn default_output_path_uses_the_input_file_stem() {
        assert_eq!(
            default_output_path(Path::new("model.laplace")),
            PathBuf::from("build/model.stan")
        );
        assert_eq!(
            default_output_path(Path::new("models/gp.laplace")),
            PathBuf::from("build/gp.stan")
        );
    }

    #[test]
    fn pluralize_picks_singular_only_for_exactly_one() {
        assert_eq!(pluralize(0, "line", "lines"), "lines");
        assert_eq!(pluralize(1, "line", "lines"), "line");
        assert_eq!(pluralize(2, "line", "lines"), "lines");
    }
}
