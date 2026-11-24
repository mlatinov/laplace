//! What an editor shows about the names in sampler files: the built-ins of
//! a `.lapkernel` ([`builtins`]), the combinators of a `.lapsampler`
//! ([`combinators`]), and every kernel a `.lapsampler` can name
//! ([`kernel_index`]). Each is read from the table the checker uses, so it
//! cannot drift from what is checked.
//!
//! Also the kernel documentation `laplace sampler doc` prints ([`doc`]):
//! [`kernel_index`] gives each kernel the same text.

use std::path::{Component, Path, PathBuf};

use laplace::package::{KERNELS_DIR, LAPKERNEL_EXTENSION};

use crate::builtin_kernels::BuiltIn;
use crate::lapkernel::builtins::{Kind, Registry};
use crate::lapkernel::doc::{kernel_doc, render, KernelDoc};
use crate::lapkernel::file::LapKernelFile;
use crate::libraries::{display_path, installed, resolve_import, KernelPackage, Project};

// ---- .lapkernel built-ins ------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinKind {
    /// A context value: `q`, `lp`, `grad`, `dim`, `metric`.
    Value,
    /// A function.
    Function,
    /// A terminal statement, which ends a step: `propose`, `move`.
    Terminal,
}

/// A name a `.lapkernel` can use without declaring it.
#[derive(Clone, Debug, PartialEq)]
pub struct Builtin {
    pub name: &'static str,
    pub kind: BuiltinKind,
    /// One per overload, with types: `real log_density(vector[dim] x)`.
    pub signatures: &'static [&'static str],
    pub doc: &'static str,
    /// Random numbers it draws from the chain's stream ("none" if none).
    pub rng: &'static str,
    /// Needs a running chain, so it cannot appear in a tuning default.
    pub needs_chain: bool,
}

/// Every `.lapkernel` built-in, from the checker's registry, in its order:
/// context values, context functions, terminals, then the library.
pub fn builtins() -> Vec<Builtin> {
    Registry::standard()
        .entries()
        .iter()
        .map(|e| Builtin {
            name: e.name,
            kind: match e.kind {
                Kind::Value { .. } => BuiltinKind::Value,
                Kind::Function { terminal: true, .. } => BuiltinKind::Terminal,
                Kind::Function { .. } => BuiltinKind::Function,
            },
            signatures: e.signatures,
            doc: e.doc,
            rng: e.rng,
            needs_chain: e.needs_chain,
        })
        .collect()
}

// ---- .lapsampler combinators ---------------------------------------------------

/// A combinator of `.lapsampler` kernel expressions.
#[derive(Clone, Debug, PartialEq)]
pub struct Combinator {
    pub name: &'static str,
    /// As written: `seq(k1, k2, ...)`.
    pub signature: &'static str,
    pub doc: &'static str,
    /// Its named arguments, after the kernel (`target_accept` for adapt).
    pub settings: &'static [&'static str],
    /// `false` for `on`: a kernel line already is `on(block, kernel)`, so
    /// writing `on(...)` in one is an error.
    pub in_kernel_lines: bool,
    /// A use that checks (tests parse it).
    pub example: &'static str,
}

const COMBINATORS: &[Combinator] = &[
    Combinator {
        name: "seq",
        signature: "seq(k1, k2, ...)",
        doc: "run each kernel in turn, every iteration",
        settings: &[],
        in_kernel_lines: true,
        example: "seq(mala(), rwm())",
    },
    Combinator {
        name: "mix",
        signature: "mix(w1 : k1, w2 : k2, ...)",
        doc:
            "each iteration, run one kernel, picked at random with probability proportional to its \
              positive weight. nuts() cannot sit under mix()",
        settings: &[],
        in_kernel_lines: true,
        example: "mix(1 : mala(), 3 : rwm())",
    },
    Combinator {
        name: "repeat",
        signature: "repeat(n, k)",
        doc: "run k n times each iteration (n a whole number >= 1)",
        settings: &[],
        in_kernel_lines: true,
        example: "repeat(3, rwm())",
    },
    Combinator {
        name: "adapt",
        signature: "adapt(k, target_accept = ...)",
        doc: "adapt k's step size and metric during warmup, then freeze them. k is one kernel: a \
              built-in other than nuts(), or a .lapkernel with @adapt tags. target_accept \
              (strictly between 0 and 1) overrides the kernel's default",
        settings: &["target_accept"],
        in_kernel_lines: true,
        example: "adapt(mala(), target_accept = 0.6)",
    },
    Combinator {
        name: "on",
        signature: "on(block(p1, p2, ...), k)",
        doc: "run k on the listed parameters only. A kernel line `block(...) : k` means \
              on(block(...), k), so on() is not written inside a kernel line",
        settings: &[],
        in_kernel_lines: false,
        example: "on(block(theta), rwm())",
    },
];

/// The combinators: `seq`, `mix`, `repeat`, `adapt`, `on`.
pub fn combinators() -> &'static [Combinator] {
    COMBINATORS
}

// ---- kernels -------------------------------------------------------------------

/// Where a kernel comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum KernelOrigin {
    /// `nuts`, `rwm`, `mala`, `hmc`: implemented in laplace-sample.
    BuiltIn,
    /// A kernel package in the project's laplace.lock.
    Package { package: String, version: String },
    /// A `.lapkernel` file in the project.
    Local,
}

/// A value that can be set after a kernel's name: a built-in's setting or
/// a `.lapkernel` tuning value.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSetting {
    pub name: String,
    /// The declaration as written, without its tag
    /// (`real<lower=0> eps = 1`); `None` for a built-in's setting.
    pub decl: Option<String>,
    pub default: Option<String>,
    /// `@adapt(step_size, target_accept = 0.574)`, if tagged.
    pub adapt: Option<String>,
    /// Its `//@param` text.
    pub doc: Option<String>,
}

/// A kernel a `.lapsampler` can name.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelEntry {
    /// As written in a `.lapsampler`: `mala`, `langevin::mala`, or a path
    /// relative to the sampler's directory (`./kernels/mine.lapkernel`).
    pub name: String,
    pub origin: KernelOrigin,
    /// The `.lapkernel` file (`None` for a built-in; see
    /// [`BuiltIn::source`]). For a path dependency, the file in its own
    /// directory, not the cache copy builds read.
    pub file: Option<PathBuf>,
    pub brief: Option<String>,
    pub settings: Vec<KernelSetting>,
    /// The terminals its step can end in (`propose`, `move`), in source
    /// order. Empty for a built-in.
    pub ends_in: Vec<&'static str>,
    /// Its documentation, as `laplace sampler doc` prints it.
    pub doc: String,
    /// Why the file does not load or check; then only `name`, `origin` and
    /// `file` are filled in.
    pub error: Option<String>,
}

/// Every kernel a `.lapsampler` in `from` can name: the built-ins, the
/// kernels of each kernel package in `project`'s laplace.lock (installed,
/// as `laplace sample` finds them), and local `.lapkernel` files, named by
/// their path relative to `from`. A package that does not resolve is
/// skipped; its message is in the second list.
///
/// Local files: in a project (a laplace.lock in `project.dir`), every one
/// under `project.dir`. Without a lock, `project.dir` is just wherever the
/// sampler is, possibly `~`, so only the nearby files are listed: those in
/// the directory itself and under its `kernels/`. The same goes for `from`
/// when it is outside the project. A file elsewhere can still be named by
/// its path.
pub fn kernel_index(project: &Project, from: &Path) -> (Vec<KernelEntry>, Vec<String>) {
    let mut kernels: Vec<KernelEntry> = BuiltIn::ALL.into_iter().map(builtin_entry).collect();
    let mut problems = vec![];
    for package in installed(project) {
        match package {
            Err(e) => problems.push(e),
            Ok(p) => {
                for name in p.kernels() {
                    kernels.push(package_entry(&p, &name));
                }
            }
        }
    }
    let mut files = vec![];
    if project.lockfile().is_file() {
        find_kernel_files(&project.dir, 0, &mut files);
    } else {
        find_nearby_kernel_files(&project.dir, &mut files);
    }
    // The sampler's own directory, when it is outside the project.
    if !from.starts_with(&project.dir) {
        find_nearby_kernel_files(from, &mut files);
    }
    files.sort();
    files.dedup();
    for file in files {
        let name = relative_name(from, &file);
        kernels.push(match LapKernelFile::load(&file) {
            Ok(f) => file_entry(&f, name.clone(), KernelOrigin::Local, file, "local file"),
            Err(e) => failed(name, KernelOrigin::Local, file, e),
        });
    }
    (kernels, problems)
}

fn builtin_entry(b: BuiltIn) -> KernelEntry {
    KernelEntry {
        name: b.name().to_string(),
        origin: KernelOrigin::BuiltIn,
        file: None,
        brief: Some(b.brief().to_string()),
        settings: b
            .sampler_settings()
            .iter()
            .map(|s| KernelSetting {
                name: s.to_string(),
                decl: None,
                default: b
                    .defaults()
                    .iter()
                    .find(|(k, _)| k == s)
                    .map(|(_, v)| v.to_string()),
                adapt: None,
                doc: None,
            })
            .collect(),
        ends_in: vec![],
        doc: b.doc(),
        error: None,
    }
}

fn package_entry(p: &KernelPackage, name: &str) -> KernelEntry {
    let spec = format!("{}::{name}", p.name);
    let origin = KernelOrigin::Package {
        package: p.name.clone(),
        version: p.version.clone(),
    };
    // A path dependency's own file, not the cache copy that is overwritten
    // on the next resolve: that is the one to open and edit.
    let file = p
        .source_dir
        .as_ref()
        .unwrap_or(&p.dir)
        .join(KERNELS_DIR)
        .join(format!("{name}.{LAPKERNEL_EXTENSION}"));
    match load_package_kernel(p, name) {
        Ok(f) => file_entry(&f, spec, origin, file, &package_origin(p, name)),
        Err(e) => failed(spec, origin, file, e),
    }
}

fn file_entry(
    f: &LapKernelFile,
    name: String,
    origin: KernelOrigin,
    file: PathBuf,
    shown_origin: &str,
) -> KernelEntry {
    let d: KernelDoc = kernel_doc(f, &name, shown_origin);
    KernelEntry {
        settings: d
            .tuning
            .iter()
            .map(|t| KernelSetting {
                name: t.name.clone(),
                default: t.decl.split_once(" = ").map(|(_, v)| v.to_string()),
                decl: Some(t.decl.clone()),
                adapt: t.adapt.clone(),
                doc: t.doc.clone(),
            })
            .collect(),
        brief: d.brief.clone(),
        ends_in: d.ends_in.clone(),
        doc: render(&d),
        name,
        origin,
        file: Some(file),
        error: None,
    }
}

fn failed(name: String, origin: KernelOrigin, file: PathBuf, e: anyhow::Error) -> KernelEntry {
    KernelEntry {
        name,
        origin,
        file: Some(file),
        brief: None,
        settings: vec![],
        ends_in: vec![],
        doc: String::new(),
        error: Some(e.to_string()),
    }
}

/// `.lapkernel` files under `dir`, skipping hidden directories and build
/// output, at most 8 levels down.
/// The `.lapkernel` files directly in `dir`, and under `dir/kernels`.
fn find_nearby_kernel_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path.extension().and_then(|e| e.to_str()) == Some(LAPKERNEL_EXTENSION)
            {
                out.push(path);
            }
        }
    }
    find_kernel_files(&dir.join("kernels"), 0, out);
}

fn find_kernel_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 8 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if !name.starts_with('.') && name != "target" && name != "node_modules" {
                find_kernel_files(&path, depth + 1, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some(LAPKERNEL_EXTENSION) {
            out.push(path);
        }
    }
}

/// `file` as a `.lapsampler` in `from` writes it: `./x.lapkernel`,
/// `../k/x.lapkernel`; the absolute path if they share no root.
fn relative_name(from: &Path, file: &Path) -> String {
    let a: Vec<Component> = from.components().collect();
    let b: Vec<Component> = file.components().collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    if common == 0 {
        return file.display().to_string();
    }
    let mut parts: Vec<String> = vec![];
    for _ in common..a.len() {
        parts.push("..".into());
    }
    for c in &b[common..] {
        parts.push(c.as_os_str().to_string_lossy().into_owned());
    }
    let joined = parts.join("/");
    if joined.starts_with("../") {
        joined
    } else {
        format!("./{joined}")
    }
}

// ---- documentation -------------------------------------------------------------

/// A kernel of an imported package, loaded and checked.
pub fn load_package_kernel(p: &KernelPackage, name: &str) -> anyhow::Result<LapKernelFile> {
    let file = p.kernel_file(name).map_err(anyhow::Error::msg)?;
    let src = std::fs::read_to_string(&file)?;
    LapKernelFile::from_source(&display_path(&p.name, name), src)
}

fn package_origin(p: &KernelPackage, name: &str) -> String {
    format!(
        "{} {}, {KERNELS_DIR}/{name}.{LAPKERNEL_EXTENSION}",
        p.name, p.version
    )
}

/// `laplace sampler doc <spec>`: `pkg::kernel` (a kernel package of
/// `project`), a `.lapkernel` path, or a built-in.
pub fn doc(spec: &str, project: &Project) -> anyhow::Result<String> {
    if let Some((package, name)) = spec.split_once("::") {
        let p = resolve_import(project, package).map_err(anyhow::Error::msg)?;
        let file = load_package_kernel(&p, name)?;
        return Ok(render(&kernel_doc(&file, spec, &package_origin(&p, name))));
    }
    if spec.ends_with(&format!(".{LAPKERNEL_EXTENSION}")) {
        let file = LapKernelFile::load(Path::new(spec))?;
        return Ok(render(&kernel_doc(&file, spec, "local file")));
    }
    if let Ok(b) = BuiltIn::parse(spec) {
        return Ok(b.doc());
    }
    anyhow::bail!("expected `<package>::<kernel>` or a .lapkernel file, got `{spec}`")
}

#[cfg(test)]
mod tests;
