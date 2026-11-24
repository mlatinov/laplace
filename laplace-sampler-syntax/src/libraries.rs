//! Installed kernel packages (Phase 4.2): what `library { import langevin }`
//! in a `.lapsampler` refers to.
//!
//! Versions come only from the project's `laplace.toml` and `laplace.lock`,
//! through the compiler's own lockfile and installed-package code: an import
//! names a package, never a version, and resolves to exactly what `laplace
//! add` / `install` put in the cache (a path dependency is re-synced first,
//! as `laplace build` does).

use std::path::PathBuf;

use laplace::manifest::PackageKind;
use laplace::package::{kernel_names, KERNELS_DIR, LAPKERNEL_EXTENSION};
use laplace::resolve::{self, lockfile};

/// The project a `.lapsampler` is used in: the current directory, as for
/// every `laplace` command, and the package cache.
#[derive(Clone, Debug)]
pub struct Project {
    pub dir: PathBuf,
    pub cache_root: PathBuf,
}

impl Project {
    pub fn current() -> Self {
        Self {
            dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            cache_root: resolve::default_cache_root(),
        }
    }

    pub(crate) fn lockfile(&self) -> PathBuf {
        self.dir.join("laplace.lock")
    }
}

/// One imported kernel package, as locked and installed.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelPackage {
    pub name: String,
    pub version: String,
    /// Where its files are read from: the installed copy (for a path
    /// dependency, the cache entry it is synced into on every resolve).
    pub dir: PathBuf,
    /// A path dependency's own directory, where its files are edited
    /// (`None` for a registry or git package).
    pub source_dir: Option<PathBuf>,
}

impl KernelPackage {
    pub fn kernels(&self) -> Vec<String> {
        kernel_names(&self.dir).unwrap_or_default()
    }

    /// The file of `name`, or the reason there is none.
    pub fn kernel_file(&self, name: &str) -> Result<PathBuf, String> {
        let file = self
            .dir
            .join(KERNELS_DIR)
            .join(format!("{name}.{LAPKERNEL_EXTENSION}"));
        if file.is_file() {
            return Ok(file);
        }
        Err(format!(
            "`{}` {} has no kernel `{name}`; its kernels are: {}",
            self.name,
            self.version,
            self.kernels().join(", ")
        ))
    }
}

/// Resolve `import <name>`. `Err` is the message, for the import's span.
pub fn resolve_import(project: &Project, name: &str) -> Result<KernelPackage, String> {
    let lock_path = project.lockfile();
    let add_hint =
        format!("add it with `laplace add {name} --path <dir>` (or `--git <url> --tag <tag>`)");
    if !lock_path.is_file() {
        return Err(format!(
            "`import {name}`: there is no laplace.lock in {} -- run from the project directory; {add_hint}",
            project.dir.display()
        ));
    }
    let lock = lockfile::read_lockfile(&lock_path).map_err(|e| e.to_string())?;
    let Some(locked) = lock.get(name) else {
        return Err(format!("`{name}` is not in laplace.lock -- {add_hint}"));
    };
    if locked.kind != PackageKind::Kernels {
        return Err(format!(
            "`{name}` is a function library: its functions are used from a model's library {{ }} \
             block; a .lapsampler imports kernel packages (kind = \"kernels\")"
        ));
    }
    if !lock.root_names().iter().any(|r| r == name) {
        return Err(format!(
            "`{name}` is only a transitive dependency -- imports are private; {add_hint}"
        ));
    }
    let (dir, _note) = resolve::installed_package_dir(&lock_path, &project.cache_root, locked)
        .map_err(|e| e.to_string())?;
    if !dir.is_dir() {
        return Err(format!(
            "`{name}@{}` is in laplace.lock but not installed -- run `laplace install`",
            locked.version
        ));
    }
    let source_dir = resolve::parse_path_source(&locked.source)
        .and_then(|relative| project.dir.join(relative).canonicalize().ok());
    Ok(KernelPackage {
        name: name.to_string(),
        version: locked.version.clone(),
        dir,
        source_dir,
    })
}

/// Every kernel package the project depends on directly (for `laplace
/// sampler list`). Empty without a lock.
pub fn installed(project: &Project) -> Vec<Result<KernelPackage, String>> {
    let Ok(lock) = lockfile::read_lockfile(&project.lockfile()) else {
        return vec![];
    };
    lock.root_names()
        .iter()
        .filter(|n| lock.get(n).is_some_and(|p| p.kind == PackageKind::Kernels))
        .map(|n| resolve_import(project, n))
        .collect()
}

/// A kernel's display path: `<package>/kernels/<name>.lapkernel`, never the
/// machine-specific cache path (messages, provenance).
pub fn display_path(package: &str, kernel: &str) -> String {
    format!("{package}/{KERNELS_DIR}/{kernel}.{LAPKERNEL_EXTENSION}")
}
