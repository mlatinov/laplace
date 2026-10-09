//! Dependency resolution: a local filesystem "registry" of package folders,
//! `laplace add` (resolve + pin + fetch), and `laplace install` (restore
//! from `laplace.lock` alone, verified by checksum).
//!
//! Packages can also come from a git repository (`git = "..."` table form
//! in `laplace.toml`) -- see `git.rs` for the fetch itself. Either way the
//! result is a plain package directory (`laplace.toml` + `.stan` files),
//! and from that point on registry- and git-sourced packages are handled
//! identically: same checksum, same cache layout, same `install_one`.

mod git;
pub mod graph;
pub mod lockfile;
mod path;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use semver::{Version, VersionReq};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::manifest::{
    self, Dependency, GitDependency, ManifestError, PackageManifest, PathDependency,
};
use graph::{DepRequirement, GraphError, PackageProvider};
use lockfile::{LockedPackage, Lockfile, LockfileError};

/// The `source` value recorded in `laplace.lock` for packages resolved from
/// the local filesystem registry.
const REGISTRY_SOURCE: &str = "registry";

#[derive(Debug, Error)]
pub enum ResolveError {
    #[error(transparent)]
    Manifest(#[from] ManifestError),

    #[error(transparent)]
    Lockfile(#[from] LockfileError),

    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("`{value}` is not a valid version: {source}")]
    InvalidVersion {
        value: String,
        #[source]
        source: semver::Error,
    },

    #[error("`{value}` is not a valid version requirement: {source}")]
    InvalidVersionReq {
        value: String,
        #[source]
        source: semver::Error,
    },

    #[error("package `{package}` was not found in the registry at {registry_root}")]
    PackageNotInRegistry {
        package: String,
        registry_root: PathBuf,
    },

    #[error("`{package}@{version}` is not available in the registry")]
    PinnedVersionNotAvailable { package: String, version: String },

    #[error("no version of `{package}` in the registry satisfies `{range}`")]
    NoMatchingVersion { package: String, range: String },

    #[error("registry package at {path} declares name `{found}`, expected `{expected}`")]
    PackageNameMismatch {
        path: PathBuf,
        expected: String,
        found: String,
    },

    #[error(
        "laplace.lock references `{name}@{version}`, which was not found in the registry at {registry_root}"
    )]
    LockedVersionNotInRegistry {
        name: String,
        version: String,
        registry_root: PathBuf,
    },

    #[error(transparent)]
    ChecksumMismatch(Box<ChecksumMismatch>),

    #[error("`{git_ref}` does not exist in {url}{available}")]
    GitRefNotFound {
        url: String,
        git_ref: String,
        /// Rendered list of the tags that do exist, plus a suggestion.
        available: String,
    },

    #[error(
        "`{package}` is not a dependency of this project yet -- run `laplace add {package}` first"
    )]
    NotADependency { package: String },

    #[error(
        "`{package}` is a git or path dependency in laplace.toml -- use `laplace update \
         {package}` to refresh it, not `laplace add {package}`"
    )]
    NotARegistryDependency { package: String },

    #[error("failed to create a temporary directory: {0}")]
    TempDir(#[source] io::Error),

    #[error(
        "`{pkg_source}` has no laplace.toml at {location} -- a git package's \
         laplace.toml must sit at the root of the directory the dependency points at{hint}"
    )]
    GitManifestMissing {
        pkg_source: String,
        location: String,
        hint: String,
    },

    #[error(
        "laplace.lock source `{pkg_source}` has an invalid subdirectory (`{subdir}`): {reason}"
    )]
    InvalidGitSubdir {
        pkg_source: String,
        subdir: String,
        reason: &'static str,
    },

    #[error(transparent)]
    Git(#[from] git::GitError),

    #[error("laplace.lock has an unrecognized source `{pkg_source}` for `{name}@{version}`")]
    UnknownSource {
        name: String,
        version: String,
        pkg_source: String,
    },

    #[error(
        "path dependency `{name}` points at {path}, which {problem} (relative paths are \
         resolved against the directory of the laplace.toml that declares them)"
    )]
    PathDependencyMissing {
        name: String,
        path: PathBuf,
        problem: &'static str,
    },

    #[error(
        "`{package}` declares `{dependency}` as a path dependency, but `{package}` itself comes \
         from {origin} -- only the project, or another path package, may use a path dependency"
    )]
    PathDependencyNotAllowed {
        package: String,
        dependency: String,
        origin: String,
    },

    #[error(
        "laplace.lock has path {}: {} -- a path dependency only exists on this machine, so the \
         lock cannot reproduce it elsewhere\n  help: depend on a released version (registry or \
         `--git ... --tag ...`) before relying on --locked",
        if .0.len() == 1 { "dependency" } else { "dependencies" },
        .0.join(", ")
    )]
    LockedPathDependencies(Vec<String>),

    #[error(transparent)]
    Docs(Box<crate::docs::DocsError>),

    #[error(transparent)]
    Graph(Box<GraphError>),
}

/// Something worth telling the user that is not an error, collected while
/// resolving or installing and handed back to the caller to print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Note {
    /// The cached copy of `name@version` differed from its source and was
    /// replaced -- the package changed without a version bump.
    Refreshed { name: String, version: String },
    /// A git tag that looks like a version names a different version than
    /// the package's own `laplace.toml` -- usually a tag pushed before the
    /// manifest was bumped.
    TagVersionMismatch {
        name: String,
        tag: String,
        version: String,
    },
    /// A locked path dependency, which another machine will not have.
    PathDependency { name: String, source: String },
}

impl Note {
    /// Whether this note is a warning (something probably wrong) rather
    /// than plain information.
    pub fn is_warning(&self) -> bool {
        matches!(
            self,
            Note::TagVersionMismatch { .. } | Note::PathDependency { .. }
        )
    }
}

impl std::fmt::Display for Note {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Note::Refreshed { name, version } => {
                write!(f, "refreshed {name}@{version} (source changed)")
            }
            Note::TagVersionMismatch { name, tag, version } => write!(
                f,
                "tag {tag} of `{name}` contains version {version} in laplace.toml -- the \
                 package was probably tagged before its version was bumped"
            ),
            Note::PathDependency { name, source } => write!(
                f,
                "`{name}` is a path dependency ({source}) -- it is re-read from that directory \
                 on every install/build/doc, and another machine will not have it"
            ),
        }
    }
}

/// The result of `add`/`update`: the requested package's new pin, plus
/// any [`Note`]s about the rest of the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub locked: LockedPackage,
    pub notes: Vec<Note>,
}

/// The result of `install`: every package restored, plus any [`Note`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Installed {
    pub packages: Vec<LockedPackage>,
    pub notes: Vec<Note>,
}

/// What [`install_one`] found in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheOutcome {
    /// Nothing was cached for this version yet.
    Fresh,
    /// The cached copy already matched the source.
    Unchanged,
    /// The cached copy differed from the source and was replaced.
    Refreshed,
}

/// A locked package whose source no longer hashes to the checksum
/// `laplace.lock` recorded for it.
#[derive(Debug, Error)]
#[error(
    "checksum mismatch for `{name}@{version}`: laplace.lock records {expected}, but its source \
     ({pkg_source}) now hashes to {actual}\n  the package changed without a version bump{moved}, \
     so neither copy is installed\n  help: if the new contents are intended, run \
     `laplace update {name}` to re-pin them in laplace.lock; otherwise ask the author to restore \
     {version} and publish the change as a new version"
)]
pub struct ChecksumMismatch {
    pub name: String,
    pub version: String,
    pub expected: String,
    pub actual: String,
    pub pkg_source: String,
    /// `" (or its tag was moved)"` for a git source, empty otherwise.
    pub moved: &'static str,
}

/// A local filesystem package registry: `<root>/<name>/<version>/` folders,
/// each holding a package `laplace.toml` plus its `.stan` file(s).
pub struct Registry {
    root: PathBuf,
}

impl Registry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Registry { root: root.into() }
    }

    pub fn package_dir(&self, name: &str, version: &Version) -> PathBuf {
        self.root.join(name).join(version.to_string())
    }

    /// Every version of `name` available in the registry, unsorted. Entries
    /// that aren't valid semver versions are ignored, and a package with no
    /// directory at all yields an empty list rather than an error.
    pub fn available_versions(&self, name: &str) -> Result<Vec<Version>, ResolveError> {
        let package_root = self.root.join(name);
        let entries = match fs::read_dir(&package_root) {
            Ok(entries) => entries,
            Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(ResolveError::Io {
                    path: package_root,
                    source,
                })
            }
        };

        let mut versions = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| ResolveError::Io {
                path: package_root.clone(),
                source,
            })?;
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            if let Some(dir_name) = entry.file_name().to_str() {
                if let Ok(version) = Version::parse(dir_name) {
                    versions.push(version);
                }
            }
        }
        Ok(versions)
    }
}

/// Turn a manifest dependency entry into the requirement the graph
/// resolver speaks. `base` is the directory relative path dependencies are
/// resolved against -- the declaring manifest's own directory -- or `Err`
/// with where the declarer came from when it may not have path
/// dependencies at all.
fn dep_requirement(
    name: &str,
    dep: &Dependency,
    base: Result<&Path, (&str, &str)>,
) -> Result<DepRequirement, ResolveError> {
    match dep {
        Dependency::Range(range) => {
            let req =
                VersionReq::parse(range).map_err(|source| ResolveError::InvalidVersionReq {
                    value: range.clone(),
                    source,
                })?;
            Ok(DepRequirement::Range(req))
        }
        Dependency::Git(git_dep) => Ok(DepRequirement::Git {
            url: git_dep.git.clone(),
            git_ref: git_dep.git_ref()?.to_string(),
            subdir: git_dep.subdir()?.map(str::to_string),
        }),
        Dependency::Path(path_dep) => {
            let base =
                base.map_err(|(package, origin)| ResolveError::PathDependencyNotAllowed {
                    package: package.to_string(),
                    dependency: name.to_string(),
                    origin: origin.to_string(),
                })?;
            Ok(DepRequirement::Path {
                root: path_package_root(name, path_dep, base)?,
            })
        }
    }
}

/// The absolute, canonical root of a path package: `base/path[/subdir]`,
/// which must be a directory holding a `laplace.toml`.
fn path_package_root(
    name: &str,
    dep: &PathDependency,
    base: &Path,
) -> Result<PathBuf, ResolveError> {
    let mut root = base.join(&dep.path);
    if let Some(subdir) = dep.subdir()? {
        root.push(subdir);
    }
    let missing = |problem| ResolveError::PathDependencyMissing {
        name: name.to_string(),
        path: root.clone(),
        problem,
    };
    let root = root.canonicalize().map_err(|_| missing("does not exist"))?;
    if !root.join("laplace.toml").is_file() {
        return Err(ResolveError::PathDependencyMissing {
            name: name.to_string(),
            path: root,
            problem: "has no laplace.toml",
        });
    }
    Ok(root)
}

fn root_requirements(
    manifest: &manifest::ProjectManifest,
    project_dir: &Path,
) -> Result<Vec<(String, DepRequirement)>, ResolveError> {
    manifest
        .dependencies
        .iter()
        .map(|(name, dep)| Ok((name.clone(), dep_requirement(name, dep, Ok(project_dir))?)))
        .collect()
}

/// The directory `laplace.toml`/`laplace.lock` live in, absolute: what path
/// dependencies and lock `path+` sources are relative to.
fn project_dir(lockfile_path: &Path) -> Result<PathBuf, ResolveError> {
    let dir = match lockfile_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    dir.canonicalize().map_err(|source| ResolveError::Io {
        path: dir.to_path_buf(),
        source,
    })
}

/// Where a git package's files start inside a fresh clone at `clone`.
fn package_root(clone: &Path, subdir: Option<&str>) -> PathBuf {
    match subdir {
        Some(subdir) => clone.join(subdir),
        None => clone.to_path_buf(),
    }
}

/// Read the `laplace.toml` of a freshly cloned git package, turning a
/// missing one into an error that says *where* laplace looked -- the
/// symptom is otherwise just a bare "No such file or directory" naming a
/// temporary directory the user never sees. When the manifest is somewhere
/// else in the clone, the error suggests the `subdir` that would find it.
fn read_git_package_manifest(
    root: &Path,
    source: &str,
    subdir: Option<&str>,
) -> Result<PackageManifest, ResolveError> {
    let manifest_path = root.join("laplace.toml");
    if manifest_path.is_file() {
        return manifest::read_package_manifest(&manifest_path).map_err(ResolveError::from);
    }

    let location = match subdir {
        Some(subdir) => format!("`{subdir}/laplace.toml`"),
        None => "its top level".to_string(),
    };
    // Only look one level up from the package root when the clone itself is
    // the package root -- a wrong `subdir` is the user's own typo, and the
    // clone root is not ours to rummage through beyond a single hint.
    let hint = match subdir {
        Some(_) => String::new(),
        None => match find_manifest_dirs(root) {
            candidates if candidates.is_empty() => String::new(),
            candidates => format!(
                ". Did you mean `subdir = \"{}\"`?",
                candidates.join("` or `subdir = \"")
            ),
        },
    };
    Err(ResolveError::GitManifestMissing {
        pkg_source: source.to_string(),
        location,
        hint,
    })
}

/// Immediate subdirectories of `dir` that hold a `laplace.toml`, sorted, so
/// the `subdir = "..."` hint is deterministic.
fn find_manifest_dirs(dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        if !entry.path().join("laplace.toml").is_file() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            found.push(name.to_string());
        }
    }
    found.sort();
    found
}

/// A checked-out git package, kept alive for the duration of a resolve so
/// the same ref is cloned at most once and the resolved copy can be
/// installed straight from it afterwards.
struct GitCheckout {
    /// The clone's `TempDir` guard. Never read -- it is held purely so the
    /// directory `root` points into survives until the resolve is over.
    /// Deleting this field deletes the checkout out from under `root`.
    #[allow(dead_code)]
    dir: tempfile::TempDir,
    /// The package root inside `dir`: the clone itself, or the `subdir` of
    /// it the dependency named.
    root: PathBuf,
    version: Version,
    source: String,
}

/// A path package found during a resolve.
struct PathCheckout {
    /// Absolute package root.
    root: PathBuf,
    source: String,
}

/// The [`PackageProvider`] laplace actually resolves against: the local
/// filesystem registry, plus on-demand git checkouts and local directories.
struct RegistryProvider<'a> {
    registry: &'a Registry,
    /// What `path+` lock sources are written relative to.
    project_dir: PathBuf,
    git: RefCell<BTreeMap<String, GitCheckout>>,
    paths: RefCell<BTreeMap<String, PathCheckout>>,
    /// The last structured error a provider callback hit. `GraphError` is a
    /// plain-data type (it has to be, so the resolver stays unit-testable
    /// against synthetic graphs), so it can only carry a provider failure as
    /// a string. Stashing the real error here lets `resolve` hand the caller
    /// back the precise `ResolveError` -- a name mismatch, a git failure --
    /// instead of a flattened message.
    last_error: RefCell<Option<ResolveError>>,
    /// Notes gathered while fetching (e.g. a tag/version mismatch).
    notes: RefCell<Vec<Note>>,
}

impl<'a> RegistryProvider<'a> {
    fn new(registry: &'a Registry, project_dir: PathBuf) -> Self {
        RegistryProvider {
            registry,
            project_dir,
            git: RefCell::new(BTreeMap::new()),
            paths: RefCell::new(BTreeMap::new()),
            last_error: RefCell::new(None),
            notes: RefCell::new(Vec::new()),
        }
    }

    /// Record `err` and wrap it for the resolver.
    fn fail(&self, err: ResolveError) -> GraphError {
        let message = err.to_string();
        *self.last_error.borrow_mut() = Some(err);
        GraphError::Provider(message)
    }

    fn take_error(&self) -> Option<ResolveError> {
        self.last_error.borrow_mut().take()
    }

    /// Where a resolved package's files live, and what `source` string the
    /// lockfile should record for it.
    fn locate(&self, name: &str, version: &Version) -> Result<(PathBuf, String), ResolveError> {
        if let Some(checkout) = self.git.borrow().get(name) {
            return Ok((checkout.root.clone(), checkout.source.clone()));
        }
        if let Some(checkout) = self.paths.borrow().get(name) {
            return Ok((checkout.root.clone(), checkout.source.clone()));
        }
        Ok((
            self.registry.package_dir(name, version),
            REGISTRY_SOURCE.to_string(),
        ))
    }

    fn manifest_of(&self, name: &str, version: &Version) -> Result<PackageManifest, ResolveError> {
        let (dir, _) = self.locate(name, version)?;
        manifest::read_package_manifest(&dir.join("laplace.toml")).map_err(ResolveError::from)
    }
}

impl PackageProvider for RegistryProvider<'_> {
    fn available_versions(&self, name: &str) -> Result<Vec<Version>, GraphError> {
        self.registry
            .available_versions(name)
            .map_err(|e| self.fail(e))
    }

    fn git_version(
        &self,
        name: &str,
        url: &str,
        git_ref: &str,
        subdir: Option<&str>,
    ) -> Result<Version, GraphError> {
        let source = git::git_source(url, git_ref, subdir);
        if let Some(checkout) = self.git.borrow().get(name) {
            if checkout.source == source {
                return Ok(checkout.version.clone());
            }
        }

        let tmp = tempfile::tempdir().map_err(|e| self.fail(ResolveError::TempDir(e)))?;
        fetch_git(url, git_ref, tmp.path()).map_err(|e| self.fail(e))?;

        let root = package_root(tmp.path(), subdir);
        let pkg_manifest =
            read_git_package_manifest(&root, &source, subdir).map_err(|e| self.fail(e))?;
        if pkg_manifest.name != name {
            return Err(self.fail(ResolveError::PackageNameMismatch {
                path: root.join("laplace.toml"),
                expected: name.to_string(),
                found: pkg_manifest.name,
            }));
        }
        let version = Version::parse(&pkg_manifest.version).map_err(|source| {
            self.fail(ResolveError::InvalidVersion {
                value: pkg_manifest.version.clone(),
                source,
            })
        })?;
        if let Some(tagged) = git::version_in_tag(git_ref) {
            if tagged != version {
                self.notes.borrow_mut().push(Note::TagVersionMismatch {
                    name: name.to_string(),
                    tag: git_ref.to_string(),
                    version: version.to_string(),
                });
            }
        }

        self.git.borrow_mut().insert(
            name.to_string(),
            GitCheckout {
                dir: tmp,
                root,
                version: version.clone(),
                source,
            },
        );
        Ok(version)
    }

    fn dependencies_of(
        &self,
        name: &str,
        version: &Version,
    ) -> Result<Vec<(String, DepRequirement)>, GraphError> {
        let pkg_manifest = self.manifest_of(name, version).map_err(|e| self.fail(e))?;
        // A path package's own path dependencies are relative to its
        // directory. Anything else has no stable directory to be relative
        // to -- a git clone is temporary, a registry copy is not the
        // author's tree -- so it may not have any.
        let path_root = self.paths.borrow().get(name).map(|c| c.root.clone());
        let origin = if self.git.borrow().contains_key(name) {
            "a git repository"
        } else {
            "the registry"
        };
        let base = match &path_root {
            Some(root) => Ok(root.as_path()),
            None => Err((name, origin)),
        };
        pkg_manifest
            .dependencies
            .iter()
            .map(|(dep_name, dep)| {
                Ok((
                    dep_name.clone(),
                    dep_requirement(dep_name, dep, base).map_err(|e| self.fail(e))?,
                ))
            })
            .collect()
    }

    fn path_version(&self, name: &str, root: &Path) -> Result<Version, GraphError> {
        let manifest_path = root.join("laplace.toml");
        let pkg_manifest =
            manifest::read_package_manifest(&manifest_path).map_err(|e| self.fail(e.into()))?;
        if pkg_manifest.name != name {
            return Err(self.fail(ResolveError::PackageNameMismatch {
                path: manifest_path,
                expected: name.to_string(),
                found: pkg_manifest.name,
            }));
        }
        let version = Version::parse(&pkg_manifest.version).map_err(|source| {
            self.fail(ResolveError::InvalidVersion {
                value: pkg_manifest.version.clone(),
                source,
            })
        })?;
        let source = path::path_source(&path::relative_to(root, &self.project_dir));
        self.paths.borrow_mut().insert(
            name.to_string(),
            PathCheckout {
                root: root.to_path_buf(),
                source,
            },
        );
        Ok(version)
    }
}

/// Resolve the project's entire dependency graph, install every package in
/// it, and write the full graph to `laplace.lock`.
///
/// `free` names the one package (if any) allowed to move off its currently
/// locked version -- the target of `add`/`update`. Every other already-
/// locked package is preferred at its locked version, so one `add` does not
/// silently bump the rest of the graph. `pinned` overrides that for an
/// explicit `add pkg@version`.
fn resolve_install_and_lock(
    project_manifest: &manifest::ProjectManifest,
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    free: Option<&str>,
    pinned: &BTreeMap<String, Version>,
) -> Result<(Lockfile, Vec<Note>), ResolveError> {
    let project_dir = project_dir(lockfile_path)?;
    let roots = root_requirements(project_manifest, &project_dir)?;

    let previous = lockfile::read_lockfile(lockfile_path)?;
    let mut preferred: BTreeMap<String, Version> = BTreeMap::new();
    for pkg in &previous.packages {
        if Some(pkg.name.as_str()) == free {
            continue;
        }
        if let Ok(version) = Version::parse(&pkg.version) {
            preferred.insert(pkg.name.clone(), version);
        }
    }
    preferred.extend(pinned.iter().map(|(k, v)| (k.clone(), v.clone())));

    let provider = RegistryProvider::new(registry, project_dir);
    let graph =
        graph::resolve_graph_with_preferences(&roots, &provider, &preferred).map_err(|err| {
            match provider.take_error() {
                // A provider callback failed; report the precise error it hit
                // rather than the resolver's stringified copy of it.
                Some(original) if matches!(err, GraphError::Provider(_)) => original,
                _ => map_graph_error(err, registry),
            }
        })?;

    let mut notes = provider.notes.borrow().clone();
    let mut packages = Vec::with_capacity(graph.packages.len());
    for resolved in graph.packages.values() {
        let (package_dir, source) = provider.locate(&resolved.name, &resolved.version)?;
        let pkg_manifest = manifest::read_package_manifest(&package_dir.join("laplace.toml"))?;
        if pkg_manifest.name != resolved.name {
            return Err(ResolveError::PackageNameMismatch {
                path: package_dir.join("laplace.toml"),
                expected: resolved.name.clone(),
                found: pkg_manifest.name,
            });
        }

        let locked = LockedPackage {
            name: resolved.name.clone(),
            version: resolved.version.to_string(),
            checksum: checksum_dir(&package_dir)?,
            source,
            dependencies: resolved.dependencies.clone(),
        };
        // Same version, different contents than the lock recorded: the
        // package was edited without a version bump. Say so even when there
        // was no cached copy to compare against.
        let repinned = previous
            .get(&locked.name)
            .is_some_and(|old| old.version == locked.version && old.checksum != locked.checksum);
        let outcome = if path::parse_path_source(&locked.source).is_some() {
            install_into(
                &package_dir,
                &path::cache_dir(cache_root, &locked.name, &package_dir),
                &locked,
            )?
        } else {
            install_one(&package_dir, cache_root, &locked)?
        };
        if outcome == CacheOutcome::Refreshed || repinned {
            notes.push(Note::Refreshed {
                name: locked.name.clone(),
                version: locked.version.clone(),
            });
        }
        packages.push(locked);
    }

    let lock = Lockfile {
        root: graph.roots.clone(),
        packages,
    };
    lockfile::write_lockfile(lockfile_path, &lock)?;
    Ok((lock, notes))
}

/// Translate the pure resolver's errors into laplace's user-facing ones,
/// keeping the registry-specific wording where it is more helpful.
fn map_graph_error(err: GraphError, registry: &Registry) -> ResolveError {
    match err {
        // A package nothing can supply. This is usually a *transitive*
        // dependency, so name the registry it was looked for in -- the user
        // never wrote this name down anywhere and needs the hint.
        GraphError::PackageUnavailable { package } => ResolveError::PackageNotInRegistry {
            package,
            registry_root: registry.root.clone(),
        },
        other => ResolveError::Graph(Box::new(other)),
    }
}

fn locked_for<'a>(lock: &'a Lockfile, package: &str) -> Result<&'a LockedPackage, ResolveError> {
    lock.get(package)
        .ok_or_else(|| ResolveError::NotADependency {
            package: package.to_string(),
        })
}

/// `laplace add <pkg>[@version]`.
///
/// - With an explicit `@version`: that exact version must exist in the
///   registry, and `laplace.toml`'s range for `pkg` is (re)written to
///   `^<version>`.
/// - Without a version: reuses `pkg`'s existing range in `laplace.toml` if
///   there is one, resolving to the latest version satisfying it; if `pkg`
///   is new to the manifest, picks the latest available version and writes
///   `^<version>` as its range.
///
/// Either way it re-resolves the *whole* dependency graph (packages can
/// have dependencies of their own), installs every package in it, and
/// writes the full graph to `laplace.lock`. Packages already pinned in the
/// lock stay where they are unless a constraint forces them to move.
pub fn add(
    project_manifest_path: &Path,
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    package: &str,
    pinned_version: Option<&str>,
) -> Result<Resolved, ResolveError> {
    let mut project_manifest = manifest::read_project_manifest(project_manifest_path)?;
    let mut pinned: BTreeMap<String, Version> = BTreeMap::new();

    match pinned_version {
        Some(raw) => {
            let version = Version::parse(raw).map_err(|source| ResolveError::InvalidVersion {
                value: raw.to_string(),
                source,
            })?;
            let available = registry.available_versions(package)?;
            if available.is_empty() {
                return Err(ResolveError::PackageNotInRegistry {
                    package: package.to_string(),
                    registry_root: registry.root.clone(),
                });
            }
            if !available.contains(&version) {
                return Err(ResolveError::PinnedVersionNotAvailable {
                    package: package.to_string(),
                    version: version.to_string(),
                });
            }
            project_manifest.dependencies.insert(
                package.to_string(),
                Dependency::Range(format!("^{version}")),
            );
            pinned.insert(package.to_string(), version);
        }
        None => {
            match project_manifest.dependencies.get(package) {
                Some(Dependency::Git(_)) | Some(Dependency::Path(_)) => {
                    return Err(ResolveError::NotARegistryDependency {
                        package: package.to_string(),
                    })
                }
                Some(Dependency::Range(_)) => {}
                None => {
                    // New to the manifest: pick the latest available version
                    // and record `^<version>` as the range to accept.
                    let latest = registry
                        .available_versions(package)?
                        .into_iter()
                        .max()
                        .ok_or_else(|| ResolveError::PackageNotInRegistry {
                            package: package.to_string(),
                            registry_root: registry.root.clone(),
                        })?;
                    project_manifest
                        .dependencies
                        .insert(package.to_string(), Dependency::Range(format!("^{latest}")));
                }
            }
        }
    }

    let (lock, notes) = resolve_install_and_lock(
        &project_manifest,
        lockfile_path,
        registry,
        cache_root,
        Some(package),
        &pinned,
    )?;
    manifest::write_project_manifest(project_manifest_path, &project_manifest)?;

    Ok(Resolved {
        locked: locked_for(&lock, package)?.clone(),
        notes,
    })
}

/// `laplace update <pkg>`: re-resolve `pkg` against its *existing* entry in
/// `laplace.toml` and refresh the lock. For a registry range, picks the
/// latest version that still satisfies it. For a git source, refetches the
/// pinned tag/rev (useful if a tag moved, or just to refresh the cache).
/// Never creates a new dependency or changes the manifest entry -- `pkg`
/// must already be in `laplace.toml` (use `add` for that).
///
/// Only `pkg` is free to move; every other locked package keeps its pin
/// unless `pkg`'s new version forces it.
pub fn update(
    project_manifest_path: &Path,
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    package: &str,
) -> Result<Resolved, ResolveError> {
    let project_manifest = manifest::read_project_manifest(project_manifest_path)?;
    if !project_manifest.dependencies.contains_key(package) {
        return Err(ResolveError::NotADependency {
            package: package.to_string(),
        });
    }

    let (lock, notes) = resolve_install_and_lock(
        &project_manifest,
        lockfile_path,
        registry,
        cache_root,
        Some(package),
        &BTreeMap::new(),
    )?;
    Ok(Resolved {
        locked: locked_for(&lock, package)?.clone(),
        notes,
    })
}

/// `laplace add <pkg> --git <url> --tag <tag>` (or `--rev <rev>`): record
/// the git source in `laplace.toml`, then resolve, fetch and lock the whole
/// graph exactly as a registry `add` does. `subdir` points at the directory
/// inside the repository holding the package's `laplace.toml`, for repos
/// that keep it below the top level.
#[allow(clippy::too_many_arguments)]
pub fn add_git(
    project_manifest_path: &Path,
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    package: &str,
    url: &str,
    tag: Option<&str>,
    rev: Option<&str>,
    subdir: Option<&str>,
) -> Result<Resolved, ResolveError> {
    let git_dep = GitDependency {
        git: url.to_string(),
        tag: tag.map(str::to_string),
        rev: rev.map(str::to_string),
        subdir: subdir.map(str::to_string),
    };
    // Validate the tag/rev pair and the subdir before touching anything on
    // disk.
    git_dep.git_ref()?;
    git_dep.subdir()?;

    let mut project_manifest = manifest::read_project_manifest(project_manifest_path)?;
    project_manifest
        .dependencies
        .insert(package.to_string(), Dependency::Git(git_dep));

    let (lock, notes) = resolve_install_and_lock(
        &project_manifest,
        lockfile_path,
        registry,
        cache_root,
        Some(package),
        &BTreeMap::new(),
    )?;
    manifest::write_project_manifest(project_manifest_path, &project_manifest)?;

    Ok(Resolved {
        locked: locked_for(&lock, package)?.clone(),
        notes,
    })
}

/// `laplace add <pkg> --path <dir> [--subdir <sub>]`: depend on a package
/// in a local directory. The lock records `path+<dir relative to the
/// project>`, and the package is re-read from that directory on every
/// install, build and doc -- edits are picked up without a version bump or
/// a tag.
pub fn add_path(
    project_manifest_path: &Path,
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    package: &str,
    dir: &str,
    subdir: Option<&str>,
) -> Result<Resolved, ResolveError> {
    let path_dep = PathDependency {
        path: dir.to_string(),
        subdir: subdir.map(str::to_string),
    };
    path_dep.subdir()?;

    let mut project_manifest = manifest::read_project_manifest(project_manifest_path)?;
    project_manifest
        .dependencies
        .insert(package.to_string(), Dependency::Path(path_dep));

    let (lock, notes) = resolve_install_and_lock(
        &project_manifest,
        lockfile_path,
        registry,
        cache_root,
        Some(package),
        &BTreeMap::new(),
    )?;
    manifest::write_project_manifest(project_manifest_path, &project_manifest)?;

    Ok(Resolved {
        locked: locked_for(&lock, package)?.clone(),
        notes,
    })
}

/// Where the files of a locked package are to be read from for a build or
/// a doc lookup: its versioned cache entry, or -- for a path dependency --
/// a fresh sync of its directory into its own cache entry, so an edit is
/// seen without any version bump. The [`Note`] says when that sync found
/// the directory changed.
pub fn installed_package_dir(
    lockfile_path: &Path,
    cache_root: &Path,
    locked: &LockedPackage,
) -> Result<(PathBuf, Option<Note>), ResolveError> {
    let Some(relative) = path::parse_path_source(&locked.source) else {
        return Ok((cache_root.join(&locked.name).join(&locked.version), None));
    };
    let (root, synced) =
        sync_path_package(&project_dir(lockfile_path)?, cache_root, locked, relative)?;
    let note = (synced == CacheOutcome::Refreshed).then(|| Note::Refreshed {
        name: locked.name.clone(),
        version: locked.version.clone(),
    });
    Ok((root, note))
}

/// Copy a path package's directory into its cache entry if it changed.
/// Returns the cache entry and what the sync did.
fn sync_path_package(
    project_dir: &Path,
    cache_root: &Path,
    locked: &LockedPackage,
    relative: &str,
) -> Result<(PathBuf, CacheOutcome), ResolveError> {
    let root = project_dir.join(relative).canonicalize().map_err(|_| {
        ResolveError::PathDependencyMissing {
            name: locked.name.clone(),
            path: project_dir.join(relative),
            problem: "does not exist",
        }
    })?;
    if !root.join("laplace.toml").is_file() {
        return Err(ResolveError::PathDependencyMissing {
            name: locked.name.clone(),
            path: root,
            problem: "has no laplace.toml",
        });
    }
    // The lock's checksum is only what the directory held when it was
    // locked; the whole point of a path dependency is that it moves on.
    // Sync against what is there now.
    let current = LockedPackage {
        checksum: checksum_dir(&root)?,
        ..locked.clone()
    };
    let dest = path::cache_dir(cache_root, &locked.name, &root);
    let outcome = install_into(&root, &dest, &current)?;
    Ok((dest, outcome))
}

/// `laplace install`: read `laplace.lock` only -- never `laplace.toml` -- and
/// restore every pinned package into `cache_root/<name>/<version>/`,
/// verifying each one's checksum first. Registry-sourced packages are
/// verified against the registry copy on disk; git-sourced packages are
/// refetched from their recorded `source` (so a fresh machine with no
/// registry can still restore them) and verified the same way.
///
/// Path dependencies are the exception to "verified by checksum": they are
/// synced from their directory as it is now, with a warning that the lock
/// cannot reproduce them. With `locked` (the CI form) they are an error
/// instead, before anything is installed.
pub fn install(
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    locked: bool,
) -> Result<Installed, ResolveError> {
    let lock = lockfile::read_lockfile(lockfile_path)?;
    let path_deps: Vec<String> = lock
        .packages
        .iter()
        .filter(|p| path::parse_path_source(&p.source).is_some())
        .map(|p| format!("{} ({})", p.name, p.source))
        .collect();
    if locked && !path_deps.is_empty() {
        return Err(ResolveError::LockedPathDependencies(path_deps));
    }
    let mut notes = Vec::new();
    let mut note_refresh = |outcome: CacheOutcome, pkg: &LockedPackage| {
        if outcome == CacheOutcome::Refreshed {
            notes.push(Note::Refreshed {
                name: pkg.name.clone(),
                version: pkg.version.clone(),
            });
        }
    };

    let mut path_notes = Vec::new();
    for pkg in &lock.packages {
        if let Some((url, git_ref, subdir)) = git::parse_git_source(&pkg.source) {
            note_refresh(install_git_one(url, git_ref, subdir, cache_root, pkg)?, pkg);
            continue;
        }
        if let Some(relative) = path::parse_path_source(&pkg.source) {
            let (_, outcome) =
                sync_path_package(&project_dir(lockfile_path)?, cache_root, pkg, relative)?;
            note_refresh(outcome, pkg);
            path_notes.push(Note::PathDependency {
                name: pkg.name.clone(),
                source: pkg.source.clone(),
            });
            continue;
        }

        if pkg.source != REGISTRY_SOURCE {
            return Err(ResolveError::UnknownSource {
                name: pkg.name.clone(),
                version: pkg.version.clone(),
                pkg_source: pkg.source.clone(),
            });
        }

        let version =
            Version::parse(&pkg.version).map_err(|source| ResolveError::InvalidVersion {
                value: pkg.version.clone(),
                source,
            })?;
        let package_dir = registry.package_dir(&pkg.name, &version);
        if !package_dir.is_dir() {
            return Err(ResolveError::LockedVersionNotInRegistry {
                name: pkg.name.clone(),
                version: pkg.version.clone(),
                registry_root: registry.root.clone(),
            });
        }

        let actual = checksum_dir(&package_dir)?;
        if actual != pkg.checksum {
            return Err(ResolveError::ChecksumMismatch(Box::new(ChecksumMismatch {
                name: pkg.name.clone(),
                version: pkg.version.clone(),
                expected: pkg.checksum.clone(),
                actual,
                pkg_source: format!("registry at {}", package_dir.display()),
                moved: "",
            })));
        }

        note_refresh(install_one(&package_dir, cache_root, pkg)?, pkg);
    }

    notes.extend(path_notes);
    Ok(Installed {
        packages: lock.packages,
        notes,
    })
}

/// Refetch a git-sourced lock entry into a scratch directory, verify its
/// checksum against the lock, and install it into the cache.
fn install_git_one(
    url: &str,
    git_ref: &str,
    subdir: Option<&str>,
    cache_root: &Path,
    pkg: &LockedPackage,
) -> Result<CacheOutcome, ResolveError> {
    let source = git::git_source(url, git_ref, subdir);
    if let Some(subdir) = subdir {
        // The subdir arrives from laplace.lock, which is a file like any
        // other; re-validate before it reaches the filesystem so a hand-
        // edited lock cannot install from outside the checkout.
        manifest::validate_subdir(subdir).map_err(|reason| ResolveError::InvalidGitSubdir {
            pkg_source: source.clone(),
            subdir: subdir.to_string(),
            reason,
        })?;
    }

    let tmp = tempfile::tempdir().map_err(ResolveError::TempDir)?;
    fetch_git(url, git_ref, tmp.path())?;

    let root = package_root(tmp.path(), subdir);
    // Not read for its contents -- this is the check that turns a missing
    // subdirectory into an explanation rather than an empty-checksum
    // mismatch further down.
    read_git_package_manifest(&root, &source, subdir)?;

    let actual = checksum_dir(&root)?;
    if actual != pkg.checksum {
        return Err(ResolveError::ChecksumMismatch(Box::new(ChecksumMismatch {
            name: pkg.name.clone(),
            version: pkg.version.clone(),
            expected: pkg.checksum.clone(),
            actual,
            pkg_source: source,
            moved: " (or its tag was moved)",
        })));
    }

    install_one(&root, cache_root, pkg)
}

/// [`git::fetch`], but a ref the remote does not have is reported with the
/// tags it *does* have and the newest of them suggested, instead of git's
/// raw "Remote branch not found" text.
fn fetch_git(url: &str, git_ref: &str, dest: &Path) -> Result<(), ResolveError> {
    let Err(err) = git::fetch(url, git_ref, dest) else {
        return Ok(());
    };
    // A commit sha cannot be checked against the tag list; only explain
    // refs that are not one.
    if git::looks_like_commit(git_ref) {
        return Err(err.into());
    }
    let Ok(remote) = git::remote_refs(url) else {
        return Err(err.into());
    };
    if remote.has(git_ref) {
        return Err(err.into());
    }
    let available = if remote.tags.is_empty() {
        " -- the repository has no tags at all; tag a release first (`laplace release`) or \
         use `--rev <commit>`"
            .to_string()
    } else {
        let mut text = format!("\n  available tags: {}", remote.tags.join(", "));
        if let Some(newest) = git::newest_version_tag(&remote.tags) {
            text.push_str(&format!("\n  help: did you mean `--tag {newest}`?"));
        }
        text
    };
    Err(ResolveError::GitRefNotFound {
        url: url.to_string(),
        git_ref: git_ref.to_string(),
        available,
    })
}

/// The file in a cache entry recording the checksum of the source it was
/// copied from. It is what lets a re-install tell "already cached" from
/// "same version, different contents" without trusting the version number.
const CACHE_CHECKSUM_FILE: &str = ".laplace-checksum";

/// Files laplace itself adds to a cache entry, which the source never had.
const CACHE_ONLY_FILES: &[&str] = &["docs.json", CACHE_CHECKSUM_FILE];

/// Make `cache_root/<name>/<version>/` an exact copy of `package_dir`
/// (whose checksum is `pkg.checksum`), plus its `docs.json`.
///
/// A cache entry whose recorded checksum already matches is left alone. One
/// that differs -- the package was edited without a version bump -- is
/// replaced and its docs regenerated, and the caller is told so.
fn install_one(
    package_dir: &Path,
    cache_root: &Path,
    pkg: &LockedPackage,
) -> Result<CacheOutcome, ResolveError> {
    install_into(
        package_dir,
        &cache_root.join(&pkg.name).join(&pkg.version),
        pkg,
    )
}

/// [`install_one`] with an explicit destination: the versioned cache entry,
/// or a path package's own entry under `path-packages/`.
fn install_into(
    package_dir: &Path,
    dest: &Path,
    pkg: &LockedPackage,
) -> Result<CacheOutcome, ResolveError> {
    // Refuse a package written for a newer compiler before copying it
    // anywhere: parsing its sources for docs.json would otherwise fail on
    // syntax this compiler does not know, with a far less useful message.
    manifest::read_package_manifest(&package_dir.join("laplace.toml"))?;

    let dest = dest.to_path_buf();
    let outcome = if dest.is_dir() {
        if cached_checksum(&dest)?.as_deref() == Some(pkg.checksum.as_str())
            && dest.join("docs.json").is_file()
        {
            return Ok(CacheOutcome::Unchanged);
        }
        let outcome = match cached_checksum(&dest)? {
            Some(cached) if cached != pkg.checksum => CacheOutcome::Refreshed,
            _ => CacheOutcome::Unchanged,
        };
        fs::remove_dir_all(&dest).map_err(|source| ResolveError::Io {
            path: dest.clone(),
            source,
        })?;
        outcome
    } else {
        CacheOutcome::Fresh
    };
    copy_dir_recursive(package_dir, &dest).map_err(|source| ResolveError::Io {
        path: dest.clone(),
        source,
    })?;
    let marker = dest.join(CACHE_CHECKSUM_FILE);
    fs::write(&marker, format!("{}\n", pkg.checksum)).map_err(|source| ResolveError::Io {
        path: marker,
        source,
    })?;

    crate::docs::write_sidecar(&dest, &pkg.name, &pkg.version)
        .map_err(|source| ResolveError::Docs(Box::new(source)))?;
    Ok(outcome)
}

/// The checksum of the source a cache entry was copied from: the recorded
/// one, or -- for an entry written before the record existed -- the entry's
/// own contents minus the files laplace added.
fn cached_checksum(dest: &Path) -> Result<Option<String>, ResolveError> {
    match fs::read_to_string(dest.join(CACHE_CHECKSUM_FILE)) {
        Ok(text) => Ok(Some(text.trim().to_string())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            checksum_dir_skipping(dest, CACHE_ONLY_FILES).map(Some)
        }
        Err(source) => Err(ResolveError::Io {
            path: dest.join(CACHE_CHECKSUM_FILE),
            source,
        }),
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

/// Deterministic sha256 over a package directory's full contents: every
/// regular file, sorted by its path relative to `dir`, so directory-walk
/// order never affects the result.
fn checksum_dir(dir: &Path) -> Result<String, ResolveError> {
    checksum_dir_skipping(dir, &[])
}

/// [`checksum_dir`], ignoring the top-level files named in `skip`.
fn checksum_dir_skipping(dir: &Path, skip: &[&str]) -> Result<String, ResolveError> {
    let mut files = Vec::new();
    collect_files_relative(dir, dir, &mut files).map_err(|source| ResolveError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    files.retain(|rel| !skip.iter().any(|name| rel == Path::new(name)));
    files.sort();

    let mut hasher = Sha256::new();
    for rel in &files {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        hasher.update(rel_str.as_bytes());
        hasher.update([0u8]);
        let contents = fs::read(dir.join(rel)).map_err(|source| ResolveError::Io {
            path: dir.join(rel),
            source,
        })?;
        hasher.update(&contents);
        hasher.update([0u8]);
    }

    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn collect_files_relative(root: &Path, current: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files_relative(root, &path, out)?;
        } else if file_type.is_file() {
            out.push(path.strip_prefix(root).unwrap().to_path_buf());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lockfile::Lockfile;

    fn write_package(registry_root: &Path, name: &str, version: &str, exports: &[&str]) -> PathBuf {
        let dir = registry_root.join(name).join(version);
        fs::create_dir_all(&dir).unwrap();
        let exports_toml = exports
            .iter()
            .map(|e| format!("\"{e}\""))
            .collect::<Vec<_>>()
            .join(", ");
        fs::write(
            dir.join("laplace.toml"),
            format!("name = \"{name}\"\nversion = \"{version}\"\nexports = [{exports_toml}]\n"),
        )
        .unwrap();
        fs::write(
            dir.join(format!("{name}.stan")),
            format!(
                "real {}() {{\n  return 1;\n}}\n",
                exports.first().copied().unwrap_or("noop")
            ),
        )
        .unwrap();
        dir
    }

    struct Fixture {
        _tmp: tempfile::TempDir,
        registry: Registry,
        registry_root: PathBuf,
        project_manifest_path: PathBuf,
        lockfile_path: PathBuf,
        cache_root: PathBuf,
    }

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let registry_root = tmp.path().join("registry");
        fs::create_dir_all(&registry_root).unwrap();
        Fixture {
            registry: Registry::new(registry_root.clone()),
            registry_root,
            project_manifest_path: tmp.path().join("laplace.toml"),
            lockfile_path: tmp.path().join("laplace.lock"),
            cache_root: tmp.path().join("cache"),
            _tmp: tmp,
        }
    }

    #[test]
    fn available_versions_ignores_non_semver_dirs_and_missing_packages() {
        let f = fixture();
        assert_eq!(f.registry.available_versions("gps").unwrap(), Vec::new());

        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        fs::create_dir_all(f.registry_root.join("gps").join("not-a-version")).unwrap();

        let versions = f.registry.available_versions("gps").unwrap();
        assert_eq!(versions, vec![Version::parse("1.0.0").unwrap()]);
    }

    #[test]
    fn add_with_no_range_picks_latest_and_writes_caret_range() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        write_package(&f.registry_root, "gps", "1.2.0", &["rbf_cov"]);

        let locked = add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            None,
        )
        .unwrap()
        .locked;

        assert_eq!(locked.name, "gps");
        assert_eq!(locked.version, "1.2.0");

        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap().as_range(),
            Some("^1.2.0")
        );

        let lock = lockfile::read_lockfile(&f.lockfile_path).unwrap();
        assert_eq!(lock.packages, vec![locked.clone()]);

        let installed = f.cache_root.join("gps").join("1.2.0").join("laplace.toml");
        assert!(installed.is_file());
    }

    #[test]
    fn add_reuses_existing_range_and_respects_it() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        write_package(&f.registry_root, "gps", "2.0.0", &["rbf_cov"]);

        let mut manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        manifest
            .dependencies
            .insert("gps".to_string(), Dependency::Range("^1.0".to_string()));
        manifest::write_project_manifest(&f.project_manifest_path, &manifest).unwrap();

        let locked = add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            None,
        )
        .unwrap()
        .locked;

        // ^1.0 must not pick the incompatible 2.0.0.
        assert_eq!(locked.version, "1.0.0");
        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap().as_range(),
            Some("^1.0")
        );
    }

    #[test]
    fn add_with_explicit_version_pins_exactly_that_version() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        write_package(&f.registry_root, "gps", "1.2.0", &["rbf_cov"]);

        let locked = add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            Some("1.0.0"),
        )
        .unwrap()
        .locked;

        assert_eq!(locked.version, "1.0.0");
        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap().as_range(),
            Some("^1.0.0")
        );
    }

    #[test]
    fn add_with_unavailable_pinned_version_errors() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);

        let err = add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            Some("9.9.9"),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ResolveError::PinnedVersionNotAvailable { .. }
        ));
    }

    #[test]
    fn add_errors_when_package_absent_from_registry() {
        let f = fixture();
        let err = add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "nonexistent",
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ResolveError::PackageNotInRegistry { .. }));
    }

    #[test]
    fn add_errors_on_package_name_mismatch() {
        let f = fixture();
        let dir = f.registry_root.join("gps").join("1.0.0");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("laplace.toml"),
            "name = \"not-gps\"\nversion = \"1.0.0\"\nexports = []\n",
        )
        .unwrap();

        let err = add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ResolveError::PackageNameMismatch { .. }));
    }

    #[test]
    fn fresh_install_from_a_lock_restores_into_cache() {
        let f = fixture();
        let pkg_dir = write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        let checksum = checksum_dir(&pkg_dir).unwrap();

        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    &checksum,
                    REGISTRY_SOURCE,
                )],
            },
        )
        .unwrap();

        let installed = install(&f.lockfile_path, &f.registry, &f.cache_root, false)
            .unwrap()
            .packages;
        assert_eq!(installed.len(), 1);

        let restored = f.cache_root.join("gps").join("1.0.0").join("laplace.toml");
        assert!(restored.is_file());
        assert_eq!(
            fs::read_to_string(restored).unwrap(),
            fs::read_to_string(pkg_dir.join("laplace.toml")).unwrap()
        );
    }

    #[test]
    fn install_does_not_consult_project_manifest_ranges() {
        // Write a laplace.toml with a range that would exclude 1.0.0, and
        // confirm install() -- driven only by the lock -- ignores it.
        let f = fixture();
        let pkg_dir = write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        let checksum = checksum_dir(&pkg_dir).unwrap();

        let mut manifest = manifest::ProjectManifest::default();
        manifest
            .dependencies
            .insert("gps".to_string(), Dependency::Range("^2.0".to_string()));
        manifest::write_project_manifest(&f.project_manifest_path, &manifest).unwrap();

        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    &checksum,
                    REGISTRY_SOURCE,
                )],
            },
        )
        .unwrap();

        let installed = install(&f.lockfile_path, &f.registry, &f.cache_root, false)
            .unwrap()
            .packages;
        assert_eq!(installed[0].version, "1.0.0");
    }

    #[test]
    fn checksum_mismatch_errors() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);

        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string().to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                    REGISTRY_SOURCE,
                )],
            },
        )
        .unwrap();

        let err = install(&f.lockfile_path, &f.registry, &f.cache_root, false).unwrap_err();
        assert!(matches!(err, ResolveError::ChecksumMismatch(_)));
    }

    #[test]
    fn lock_referencing_package_absent_from_registry_errors_clearly() {
        let f = fixture();
        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                root: vec!["ghost".to_string().to_string()],
                packages: vec![LockedPackage::leaf(
                    "ghost",
                    "1.0.0",
                    "sha256:doesnotmatter",
                    REGISTRY_SOURCE,
                )],
            },
        )
        .unwrap();

        let err = install(&f.lockfile_path, &f.registry, &f.cache_root, false).unwrap_err();
        match &err {
            ResolveError::LockedVersionNotInRegistry { name, version, .. } => {
                assert_eq!(name, "ghost");
                assert_eq!(version, "1.0.0");
            }
            other => panic!("expected LockedVersionNotInRegistry, got {other:?}"),
        }
        // The error message should clearly name the missing package.
        assert!(err.to_string().contains("ghost@1.0.0"));
    }

    #[test]
    fn checksum_dir_is_deterministic() {
        let f = fixture();
        let pkg_dir = write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        let a = checksum_dir(&pkg_dir).unwrap();
        let b = checksum_dir(&pkg_dir).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("sha256:"));
    }

    #[test]
    fn update_picks_latest_version_matching_the_existing_range() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);

        add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            None,
        )
        .unwrap();

        // A new compatible version lands in the registry after the initial add.
        write_package(&f.registry_root, "gps", "1.1.0", &["rbf_cov"]);

        let updated = update(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
        )
        .unwrap()
        .locked;

        assert_eq!(updated.version, "1.1.0");
        // The range itself is untouched by update.
        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap().as_range(),
            Some("^1.0.0")
        );

        let lock = lockfile::read_lockfile(&f.lockfile_path).unwrap();
        assert_eq!(lock.packages, vec![updated]);
    }

    #[test]
    fn update_respects_the_range_and_wont_jump_to_an_incompatible_version() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);
        add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            None,
        )
        .unwrap();

        write_package(&f.registry_root, "gps", "2.0.0", &["rbf_cov"]);

        let updated = update(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
        )
        .unwrap()
        .locked;
        assert_eq!(updated.version, "1.0.0");
    }

    #[test]
    fn update_on_a_package_that_was_never_added_errors() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);

        let err = update(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
        )
        .unwrap_err();
        assert!(matches!(err, ResolveError::NotADependency { .. }));
    }

    // -- git source tests --------------------------------------------------
    //
    // These never touch the network: the "remote" is a `git init --bare`
    // repo in a tempdir, populated by pushing a tagged commit from a throwaway
    // working clone.

    fn run_git(args: &[&str], cwd: Option<&Path>) {
        let mut cmd = std::process::Command::new("git");
        cmd.args(args);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let output = cmd.output().expect("failed to spawn git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn run_git_capture(args: &[&str], cwd: Option<&Path>) -> String {
        let mut cmd = std::process::Command::new("git");
        cmd.args(args);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let output = cmd.output().expect("failed to spawn git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    /// Set up a bare git repo under `tmp_root` containing a single tagged
    /// commit with a package `laplace.toml` + `.stan` file at its root (the
    /// repo root *is* the package root, same as a registry package
    /// directory). Returns `(bare_repo_path, commit_sha)`.
    fn init_git_package_repo(
        tmp_root: &Path,
        name: &str,
        version: &str,
        tag: &str,
        exports: &[&str],
    ) -> (PathBuf, String) {
        init_git_package_repo_in(tmp_root, name, version, tag, exports, None)
    }

    /// Same, but with the package's files committed under `subdir` instead
    /// of at the repository root -- the layout a repo uses when it holds a
    /// package alongside a README, a LICENSE, or several packages at once.
    fn init_git_package_repo_in(
        tmp_root: &Path,
        name: &str,
        version: &str,
        tag: &str,
        exports: &[&str],
        subdir: Option<&str>,
    ) -> (PathBuf, String) {
        let bare = tmp_root.join(format!("{name}-bare.git"));
        let work = tmp_root.join(format!("{name}-work"));

        run_git(&["init", "--quiet", "--bare", bare.to_str().unwrap()], None);
        run_git(
            &["init", "--quiet", "-b", "main", work.to_str().unwrap()],
            None,
        );

        let exports_toml = exports
            .iter()
            .map(|e| format!("\"{e}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let pkg_dir = match subdir {
            Some(subdir) => {
                let dir = work.join(subdir);
                fs::create_dir_all(&dir).unwrap();
                // Something at the top level that is *not* the package, so
                // a checkout rooted at the repo root is visibly wrong.
                fs::write(work.join("README.md"), "# repo\n").unwrap();
                dir
            }
            None => work.clone(),
        };
        fs::write(
            pkg_dir.join("laplace.toml"),
            format!("name = \"{name}\"\nversion = \"{version}\"\nexports = [{exports_toml}]\n"),
        )
        .unwrap();
        fs::write(
            pkg_dir.join(format!("{name}.stan")),
            format!(
                "real {}() {{\n  return 1;\n}}\n",
                exports.first().copied().unwrap_or("noop")
            ),
        )
        .unwrap();

        run_git(
            &[
                "-c",
                "user.email=test@example.com",
                "-c",
                "user.name=Test",
                "add",
                ".",
            ],
            Some(&work),
        );
        run_git(
            &[
                "-c",
                "user.email=test@example.com",
                "-c",
                "user.name=Test",
                "commit",
                "--quiet",
                "-m",
                "init",
            ],
            Some(&work),
        );
        run_git(&["tag", tag], Some(&work));
        run_git(
            &["remote", "add", "origin", bare.to_str().unwrap()],
            Some(&work),
        );
        run_git(&["push", "--quiet", "origin", "main"], Some(&work));
        run_git(&["push", "--quiet", "origin", tag], Some(&work));

        let rev = run_git_capture(&["rev-parse", "HEAD"], Some(&work));
        (bare, rev)
    }

    #[test]
    fn add_git_with_tag_records_git_source_and_installs() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
        );
        let url = bare.to_str().unwrap();

        let locked = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            None,
        )
        .unwrap()
        .locked;

        assert_eq!(locked.name, "gps");
        assert_eq!(locked.version, "1.0.0");
        assert_eq!(locked.source, format!("git+{url}@0.1.0"));

        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap(),
            &Dependency::Git(GitDependency {
                git: url.to_string(),
                tag: Some("0.1.0".to_string()),
                rev: None,
                subdir: None,
            })
        );

        let lock = lockfile::read_lockfile(&f.lockfile_path).unwrap();
        assert_eq!(lock.packages, vec![locked.clone()]);

        let installed = f.cache_root.join("gps").join("1.0.0").join("laplace.toml");
        assert!(installed.is_file());
        // The cached copy must not carry .git metadata along with it.
        assert!(!f.cache_root.join("gps").join("1.0.0").join(".git").exists());
    }

    #[test]
    fn add_git_with_rev_records_git_source_and_installs() {
        let f = fixture();
        let (bare, rev) = init_git_package_repo(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
        );
        let url = bare.to_str().unwrap();

        let locked = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            None,
            Some(&rev),
            None,
        )
        .unwrap()
        .locked;

        assert_eq!(locked.version, "1.0.0");
        assert_eq!(locked.source, format!("git+{url}@{rev}"));

        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap(),
            &Dependency::Git(GitDependency {
                git: url.to_string(),
                tag: None,
                rev: Some(rev),
                subdir: None,
            })
        );
    }

    /// The whole clone -> checkout -> manifest-read -> install sequence for
    /// a package that lives in a subdirectory of its repository, which is
    /// the layout that used to fail with a bare "no such file or directory"
    /// naming a temporary path. Also pins the tempdir lifetime: the
    /// `TempDir` guard has to outlive every read below, so a refactor that
    /// returns only a path from the checkout fails here.
    #[test]
    fn add_git_reads_the_manifest_from_a_subdirectory_and_installs_it() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo_in(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
            Some("laplace"),
        );
        let url = bare.to_str().unwrap();

        let locked = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            Some("laplace"),
        )
        .unwrap()
        .locked;

        assert_eq!(locked.version, "1.0.0");
        assert_eq!(locked.source, format!("git+{url}@0.1.0#laplace"));

        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap(),
            &Dependency::Git(GitDependency {
                git: url.to_string(),
                tag: Some("0.1.0".to_string()),
                rev: None,
                subdir: Some("laplace".to_string()),
            })
        );

        // The cache holds the *package*, not the repository: the package
        // files land at the top of the cache entry and the repo's own
        // top-level files are not dragged along.
        let cached = f.cache_root.join("gps").join("1.0.0");
        assert!(cached.join("laplace.toml").is_file());
        assert!(cached.join("gps.stan").is_file());
        assert!(!cached.join("README.md").exists());
        assert!(!cached.join("laplace").exists());
    }

    /// `install` reconstructs the subdirectory from the lock's `source`
    /// alone -- the fresh-machine path, with no laplace.toml in sight.
    #[test]
    fn install_restores_a_subdirectory_git_package_from_the_lock_alone() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo_in(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
            Some("pkgs/gps"),
        );
        let url = bare.to_str().unwrap();

        let locked = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            Some("pkgs/gps"),
        )
        .unwrap()
        .locked;
        assert_eq!(locked.source, format!("git+{url}@0.1.0#pkgs/gps"));

        fs::remove_dir_all(&f.cache_root).unwrap();
        fs::remove_file(&f.project_manifest_path).unwrap();

        let installed = install(&f.lockfile_path, &f.registry, &f.cache_root, false)
            .unwrap()
            .packages;
        assert_eq!(installed, vec![locked]);
        assert!(f
            .cache_root
            .join("gps")
            .join("1.0.0")
            .join("laplace.toml")
            .is_file());
    }

    /// The reported failure, as a test: a repo whose package sits in a
    /// subdirectory, added *without* `--subdir`. It must say what is wrong
    /// and where the manifest actually is, not fail on a temporary path.
    #[test]
    fn add_git_without_subdir_explains_a_manifest_below_the_repo_root() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo_in(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
            Some("laplace"),
        );
        let url = bare.to_str().unwrap();

        let err = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            None,
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(
            matches!(err, ResolveError::GitManifestMissing { .. }),
            "{message}"
        );
        assert!(message.contains("its top level"), "{message}");
        assert!(message.contains(r#"subdir = "laplace""#), "{message}");
        // Nothing was written on the way to the error.
        assert!(!f.lockfile_path.exists());
    }

    /// A wrong `--subdir` names the directory the user asked for rather
    /// than guessing a different one.
    #[test]
    fn add_git_with_a_wrong_subdir_names_the_subdir_it_looked_in() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo_in(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
            Some("laplace"),
        );
        let url = bare.to_str().unwrap();

        let err = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            Some("lib"),
        )
        .unwrap_err();

        let message = err.to_string();
        assert!(message.contains("`lib/laplace.toml`"), "{message}");
        assert!(!message.contains("Did you mean"), "{message}");
    }

    #[test]
    fn add_git_rejects_a_subdir_that_escapes_the_checkout() {
        let f = fixture();
        let err = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            "https://example.com/repo",
            Some("0.1.0"),
            None,
            Some("../../etc"),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                ResolveError::Manifest(ManifestError::GitSubdirInvalid { .. })
            ),
            "{err:?}"
        );
    }

    /// The same guard on the other entry point: a hand-edited lock cannot
    /// talk `install` into copying from outside the clone.
    #[test]
    fn install_rejects_a_lock_subdir_that_escapes_the_checkout() {
        let f = fixture();
        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                root: vec!["gps".to_string()],
                packages: vec![LockedPackage::leaf(
                    "gps",
                    "1.0.0",
                    "sha256:doesnotmatter",
                    "git+https://example.com/repo@0.1.0#../../etc",
                )],
            },
        )
        .unwrap();

        let err = install(&f.lockfile_path, &f.registry, &f.cache_root, false).unwrap_err();
        assert!(
            matches!(err, ResolveError::InvalidGitSubdir { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn add_git_errors_on_package_name_mismatch() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo(
            f.registry_root.parent().unwrap(),
            "not-gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
        );
        let url = bare.to_str().unwrap();

        let err = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ResolveError::PackageNameMismatch { .. }));
    }

    #[test]
    fn install_refetches_git_sourced_package_on_a_fresh_machine() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
        );
        let url = bare.to_str().unwrap();

        let locked = add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            None,
        )
        .unwrap()
        .locked;

        // Simulate a fresh machine: no cache, and (since this package was
        // never in the local registry to begin with) no registry entry
        // either -- `install` must restore purely from the lock's `source`.
        fs::remove_dir_all(&f.cache_root).unwrap();
        assert!(f.registry.available_versions("gps").unwrap().is_empty());

        let installed = install(&f.lockfile_path, &f.registry, &f.cache_root, false)
            .unwrap()
            .packages;
        assert_eq!(installed, vec![locked]);

        let restored = f.cache_root.join("gps").join("1.0.0").join("laplace.toml");
        assert!(restored.is_file());
    }

    #[test]
    fn install_detects_checksum_mismatch_for_git_source() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
        );
        let url = bare.to_str().unwrap();

        add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            None,
        )
        .unwrap();

        // Tamper with the lock's checksum after the fact.
        let mut lock = lockfile::read_lockfile(&f.lockfile_path).unwrap();
        lock.packages[0].checksum =
            "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_string();
        lockfile::write_lockfile(&f.lockfile_path, &lock).unwrap();

        fs::remove_dir_all(&f.cache_root).unwrap();
        let err = install(&f.lockfile_path, &f.registry, &f.cache_root, false).unwrap_err();
        assert!(matches!(err, ResolveError::ChecksumMismatch(_)));
    }

    #[test]
    fn update_on_a_git_dependency_refetches_from_the_pinned_ref() {
        let f = fixture();
        let (bare, _rev) = init_git_package_repo(
            f.registry_root.parent().unwrap(),
            "gps",
            "1.0.0",
            "0.1.0",
            &["rbf_cov"],
        );
        let url = bare.to_str().unwrap();

        add_git(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            url,
            Some("0.1.0"),
            None,
            None,
        )
        .unwrap();

        fs::remove_dir_all(&f.cache_root).unwrap();

        let updated = update(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
        )
        .unwrap()
        .locked;

        assert_eq!(updated.version, "1.0.0");
        assert!(f
            .cache_root
            .join("gps")
            .join("1.0.0")
            .join("laplace.toml")
            .is_file());
    }
}
