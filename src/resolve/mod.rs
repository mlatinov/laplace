//! Dependency resolution: a local filesystem "registry" of package folders,
//! `laplace add` (resolve + pin + fetch), and `laplace install` (restore
//! from `laplace.lock` alone, verified by checksum).
//!
//! Real git/http fetching is a later task -- for now the registry is just a
//! directory of `<name>/<version>/` package folders on disk.

pub mod lockfile;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use semver::{Version, VersionReq};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::manifest::{self, ManifestError};
use lockfile::{LockedPackage, LockfileError};

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

    #[error(
        "checksum mismatch for `{name}@{version}`: laplace.lock says {expected}, registry \
         contents hash to {actual} -- the registry copy has changed since the lock was written"
    )]
    ChecksumMismatch {
        name: String,
        version: String,
        expected: String,
        actual: String,
    },

    #[error("`{package}` is not a dependency of this project yet -- run `laplace add {package}` first")]
    NotADependency { package: String },
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
/// Either way, resolves to exactly one version, records its checksum in
/// `laplace.lock`, and copies it into `cache_root/<name>/<version>/`.
pub fn add(
    project_manifest_path: &Path,
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    package: &str,
    pinned_version: Option<&str>,
) -> Result<LockedPackage, ResolveError> {
    let mut project_manifest = manifest::read_project_manifest(project_manifest_path)?;

    let resolved_version = match pinned_version {
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
            project_manifest
                .dependencies
                .insert(package.to_string(), format!("^{version}"));
            version
        }
        None => {
            let existing_range = project_manifest.dependencies.get(package).cloned();
            let req = match &existing_range {
                Some(range) => VersionReq::parse(range).map_err(|source| {
                    ResolveError::InvalidVersionReq {
                        value: range.clone(),
                        source,
                    }
                })?,
                None => VersionReq::STAR,
            };

            let available = registry.available_versions(package)?;
            if available.is_empty() {
                return Err(ResolveError::PackageNotInRegistry {
                    package: package.to_string(),
                    registry_root: registry.root.clone(),
                });
            }
            let latest = available
                .into_iter()
                .filter(|v| req.matches(v))
                .max()
                .ok_or_else(|| ResolveError::NoMatchingVersion {
                    package: package.to_string(),
                    range: existing_range.clone().unwrap_or_else(|| "*".to_string()),
                })?;

            project_manifest
                .dependencies
                .entry(package.to_string())
                .or_insert_with(|| format!("^{latest}"));
            latest
        }
    };

    let locked = fetch_and_lock(lockfile_path, registry, cache_root, package, &resolved_version)?;
    manifest::write_project_manifest(project_manifest_path, &project_manifest)?;

    Ok(locked)
}

/// `laplace update <pkg>`: re-resolve `pkg` against its *existing* range in
/// `laplace.toml`, picking the latest version that still satisfies it, and
/// refresh its lockfile entry + cache copy. Never creates a new dependency
/// or changes the range -- `pkg` must already be in `laplace.toml` (use
/// `add` for that).
pub fn update(
    project_manifest_path: &Path,
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    package: &str,
) -> Result<LockedPackage, ResolveError> {
    let project_manifest = manifest::read_project_manifest(project_manifest_path)?;
    let Some(range) = project_manifest.dependencies.get(package) else {
        return Err(ResolveError::NotADependency {
            package: package.to_string(),
        });
    };

    let req = VersionReq::parse(range).map_err(|source| ResolveError::InvalidVersionReq {
        value: range.clone(),
        source,
    })?;

    let available = registry.available_versions(package)?;
    if available.is_empty() {
        return Err(ResolveError::PackageNotInRegistry {
            package: package.to_string(),
            registry_root: registry.root.clone(),
        });
    }
    let latest = available
        .into_iter()
        .filter(|v| req.matches(v))
        .max()
        .ok_or_else(|| ResolveError::NoMatchingVersion {
            package: package.to_string(),
            range: range.clone(),
        })?;

    fetch_and_lock(lockfile_path, registry, cache_root, package, &latest)
}

/// Shared tail of `add`/`update`: verify the resolved version's package
/// manifest actually declares this name, hash its contents, copy it into
/// the cache, and record it in `laplace.lock`.
fn fetch_and_lock(
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
    package: &str,
    version: &Version,
) -> Result<LockedPackage, ResolveError> {
    let package_dir = registry.package_dir(package, version);
    let pkg_manifest = manifest::read_package_manifest(&package_dir.join("laplace.toml"))?;
    if pkg_manifest.name != package {
        return Err(ResolveError::PackageNameMismatch {
            path: package_dir.join("laplace.toml"),
            expected: package.to_string(),
            found: pkg_manifest.name,
        });
    }

    let checksum = checksum_dir(&package_dir)?;
    let locked = LockedPackage {
        name: package.to_string(),
        version: version.to_string(),
        checksum,
    };

    install_one(&package_dir, cache_root, &locked)?;

    let mut lock = lockfile::read_lockfile(lockfile_path)?;
    lock.packages.retain(|p| p.name != locked.name);
    lock.packages.push(locked.clone());
    lockfile::write_lockfile(lockfile_path, &lock)?;

    Ok(locked)
}

/// `laplace install`: read `laplace.lock` only -- never `laplace.toml` -- and
/// restore every pinned package into `cache_root/<name>/<version>/`,
/// verifying each one's checksum against the registry copy first.
pub fn install(
    lockfile_path: &Path,
    registry: &Registry,
    cache_root: &Path,
) -> Result<Vec<LockedPackage>, ResolveError> {
    let lock = lockfile::read_lockfile(lockfile_path)?;

    for pkg in &lock.packages {
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
            return Err(ResolveError::ChecksumMismatch {
                name: pkg.name.clone(),
                version: pkg.version.clone(),
                expected: pkg.checksum.clone(),
                actual,
            });
        }

        install_one(&package_dir, cache_root, pkg)?;
    }

    Ok(lock.packages)
}

fn install_one(
    package_dir: &Path,
    cache_root: &Path,
    pkg: &LockedPackage,
) -> Result<(), ResolveError> {
    let dest = cache_root.join(&pkg.name).join(&pkg.version);
    if dest.is_dir() {
        fs::remove_dir_all(&dest).map_err(|source| ResolveError::Io {
            path: dest.clone(),
            source,
        })?;
    }
    copy_dir_recursive(package_dir, &dest).map_err(|source| ResolveError::Io {
        path: dest,
        source,
    })
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
    let mut files = Vec::new();
    collect_files_relative(dir, dir, &mut files).map_err(|source| ResolveError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
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
        .unwrap();

        assert_eq!(locked.name, "gps");
        assert_eq!(locked.version, "1.2.0");

        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(manifest.dependencies.get("gps").unwrap(), "^1.2.0");

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
            .insert("gps".to_string(), "^1.0".to_string());
        manifest::write_project_manifest(&f.project_manifest_path, &manifest).unwrap();

        let locked = add(
            &f.project_manifest_path,
            &f.lockfile_path,
            &f.registry,
            &f.cache_root,
            "gps",
            None,
        )
        .unwrap();

        // ^1.0 must not pick the incompatible 2.0.0.
        assert_eq!(locked.version, "1.0.0");
        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(manifest.dependencies.get("gps").unwrap(), "^1.0");
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
        .unwrap();

        assert_eq!(locked.version, "1.0.0");
        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(manifest.dependencies.get("gps").unwrap(), "^1.0.0");
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
                packages: vec![LockedPackage {
                    name: "gps".to_string(),
                    version: "1.0.0".to_string(),
                    checksum,
                }],
            },
        )
        .unwrap();

        let installed = install(&f.lockfile_path, &f.registry, &f.cache_root).unwrap();
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
            .insert("gps".to_string(), "^2.0".to_string());
        manifest::write_project_manifest(&f.project_manifest_path, &manifest).unwrap();

        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                packages: vec![LockedPackage {
                    name: "gps".to_string(),
                    version: "1.0.0".to_string(),
                    checksum,
                }],
            },
        )
        .unwrap();

        let installed = install(&f.lockfile_path, &f.registry, &f.cache_root).unwrap();
        assert_eq!(installed[0].version, "1.0.0");
    }

    #[test]
    fn checksum_mismatch_errors() {
        let f = fixture();
        write_package(&f.registry_root, "gps", "1.0.0", &["rbf_cov"]);

        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                packages: vec![LockedPackage {
                    name: "gps".to_string(),
                    version: "1.0.0".to_string(),
                    checksum: "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_string(),
                }],
            },
        )
        .unwrap();

        let err = install(&f.lockfile_path, &f.registry, &f.cache_root).unwrap_err();
        assert!(matches!(err, ResolveError::ChecksumMismatch { .. }));
    }

    #[test]
    fn lock_referencing_package_absent_from_registry_errors_clearly() {
        let f = fixture();
        lockfile::write_lockfile(
            &f.lockfile_path,
            &Lockfile {
                packages: vec![LockedPackage {
                    name: "ghost".to_string(),
                    version: "1.0.0".to_string(),
                    checksum: "sha256:doesnotmatter".to_string(),
                }],
            },
        )
        .unwrap();

        let err = install(&f.lockfile_path, &f.registry, &f.cache_root).unwrap_err();
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
        .unwrap();

        assert_eq!(updated.version, "1.1.0");
        // The range itself is untouched by update.
        let manifest = manifest::read_project_manifest(&f.project_manifest_path).unwrap();
        assert_eq!(manifest.dependencies.get("gps").unwrap(), "^1.0.0");

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
        .unwrap();
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
}
