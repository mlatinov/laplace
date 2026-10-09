//! `laplace.toml` (project + package) parsing.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Project-level `laplace.toml`: dependency entries, either a semver range
/// (resolved via the local filesystem registry) or a git source.
///
/// ```toml
/// [dependencies]
/// gps = "^1.0"
/// gps2 = { git = "https://github.com/user/repo", tag = "0.1.0" }
/// gps3 = { git = "https://github.com/user/repo", rev = "abc123" }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectManifest {
    #[serde(default)]
    pub dependencies: BTreeMap<String, Dependency>,
}

/// A single dependency entry in the project manifest: either the plain
/// string form (a semver range, resolved via the local registry) or the
/// table form (a git source).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    Range(String),
    Git(GitDependency),
}

impl Dependency {
    pub fn as_range(&self) -> Option<&str> {
        match self {
            Dependency::Range(range) => Some(range),
            Dependency::Git(_) => None,
        }
    }
}

/// The table form of a dependency entry: a git repository pinned to either
/// a tag or a commit rev (exactly one of the two), optionally with the
/// package rooted in a subdirectory of the repository rather than at its
/// top level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitDependency {
    pub git: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rev: Option<String>,
    /// Where the package's `laplace.toml` lives relative to the repository
    /// root. `None` means the repository root itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subdir: Option<String>,
}

impl GitDependency {
    /// The single ref to check out. Exactly one of `tag`/`rev` must be set;
    /// anything else is a malformed manifest entry.
    pub fn git_ref(&self) -> Result<&str, ManifestError> {
        match (&self.tag, &self.rev) {
            (Some(tag), None) => Ok(tag),
            (None, Some(rev)) => Ok(rev),
            (None, None) => Err(ManifestError::GitRefMissing {
                git: self.git.clone(),
            }),
            (Some(_), Some(_)) => Err(ManifestError::GitRefAmbiguous {
                git: self.git.clone(),
            }),
        }
    }

    /// The validated subdirectory the package is rooted in, if any.
    pub fn subdir(&self) -> Result<Option<&str>, ManifestError> {
        match self.subdir.as_deref() {
            None => Ok(None),
            Some(subdir) => {
                validate_subdir(subdir).map_err(|reason| ManifestError::GitSubdirInvalid {
                    git: self.git.clone(),
                    subdir: subdir.to_string(),
                    reason,
                })?;
                Ok(Some(subdir))
            }
        }
    }
}

/// Whether `name` can be a package name. It becomes the prefix of Stan
/// identifiers (`pkg::func` -> `pkg__func`) and must survive the call-site
/// scanner, so it follows Stan's identifier rules: a letter, then letters,
/// digits and `_`. Notably no `-`: `laplace-splines::f(` would be read as
/// `laplace - splines::f(`.
pub fn is_valid_package_name(name: &str) -> bool {
    // `__` is reserved for generated names: the whole point of `pkg__func`
    // is that the two halves can be told apart, which a package name
    // containing `__` would defeat.
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.contains(crate::parser::identifiers::RESERVED_SEPARATOR)
}

/// Why `name` is not a valid package name, for error messages.
pub const PACKAGE_NAME_RULE: &str =
    "a package name must start with a letter and contain only letters, digits and single `_` \
     (it becomes the `pkg__` prefix of Stan identifiers, so `__` is reserved)";

/// Reject anything that is not a plain relative path *inside* the clone.
/// A `subdir` reaches the filesystem straight from `laplace.toml` and from
/// `laplace.lock`'s `source` string, so an absolute path or a `..`
/// component would let a manifest read and install files from outside the
/// checkout it is supposed to be confined to.
pub(crate) fn validate_subdir(subdir: &str) -> Result<(), &'static str> {
    use std::path::Component;

    if subdir.is_empty() {
        return Err("it is empty -- omit `subdir` for a package at the repository root");
    }
    if subdir.contains('#') {
        return Err(
            "it contains `#`, which laplace.lock uses to separate the subdirectory \
                    from the git ref",
        );
    }
    for component in Path::new(subdir).components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            Component::ParentDir => return Err("it contains a `..` component"),
            Component::RootDir | Component::Prefix(_) => return Err("it is an absolute path"),
        }
    }
    Ok(())
}

/// Package-level `laplace.toml`, shipped alongside a package's `.stan` /
/// `.laplacelib` file(s) in the registry.
///
/// ```toml
/// name = "regression"
/// version = "1.0.0"
/// exports = ["fit"]
///
/// [dependencies]
/// stats = "^1.0"
/// ```
///
/// `[dependencies]` uses exactly the same syntax as a project's
/// `laplace.toml` -- that is what lets a `.laplacelib` library depend on
/// another library. A package with no `[dependencies]` table (every package
/// written before libraries could depend on libraries) reads as a leaf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageManifest {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub exports: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dependencies: BTreeMap<String, Dependency>,
}

impl PackageManifest {
    /// A leaf package manifest: no dependencies of its own.
    pub fn new(name: impl Into<String>, version: impl Into<String>, exports: Vec<String>) -> Self {
        PackageManifest {
            name: name.into(),
            version: version.into(),
            exports,
            dependencies: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("failed to serialize manifest for {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: toml::ser::Error,
    },

    #[error("git dependency `{git}` needs a `tag` or a `rev`")]
    GitRefMissing { git: String },

    #[error("git dependency `{git}` has an invalid `subdir` (`{subdir}`): {reason}")]
    GitSubdirInvalid {
        git: String,
        subdir: String,
        reason: &'static str,
    },

    #[error("git dependency `{git}` cannot have both `tag` and `rev` set")]
    GitRefAmbiguous { git: String },
}

/// Read a project's `laplace.toml`. A missing file is treated as an empty
/// manifest (no dependencies yet) rather than an error, so a fresh project
/// can run `laplace add` before `laplace.toml` exists.
pub fn read_project_manifest(path: &Path) -> Result<ProjectManifest, ManifestError> {
    match fs::read_to_string(path) {
        Ok(text) => toml::from_str(&text).map_err(|source| ManifestError::Parse {
            path: path.to_path_buf(),
            source,
        }),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(ProjectManifest::default())
        }
        Err(source) => Err(ManifestError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

pub fn write_project_manifest(
    path: &Path,
    manifest: &ProjectManifest,
) -> Result<(), ManifestError> {
    let text = toml::to_string_pretty(manifest).map_err(|source| ManifestError::Serialize {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, text).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Read a package's own `laplace.toml`. Unlike the project manifest, this
/// must exist -- every package in the registry ships one.
pub fn read_package_manifest(path: &Path) -> Result<PackageManifest, ManifestError> {
    let text = fs::read_to_string(path).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&text).map_err(|source| ManifestError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

pub fn write_package_manifest(
    path: &Path,
    manifest: &PackageManifest,
) -> Result<(), ManifestError> {
    let text = toml::to_string_pretty(manifest).map_err(|source| ManifestError::Serialize {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, text).map_err(|source| ManifestError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_project_manifest_reads_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        assert_eq!(
            read_project_manifest(&path).unwrap(),
            ProjectManifest::default()
        );
    }

    #[test]
    fn project_manifest_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        let mut manifest = ProjectManifest::default();
        manifest
            .dependencies
            .insert("gps".to_string(), Dependency::Range("^1.0".to_string()));

        write_project_manifest(&path, &manifest).unwrap();
        assert_eq!(read_project_manifest(&path).unwrap(), manifest);
    }

    #[test]
    fn project_manifest_parses_git_table_form() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        fs::write(
            &path,
            concat!(
                "[dependencies]\n",
                "gps = { git = \"https://github.com/user/repo\", tag = \"0.1.0\" }\n",
                "gps2 = { git = \"https://github.com/user/repo\", rev = \"abc123\" }\n",
            ),
        )
        .unwrap();

        let manifest = read_project_manifest(&path).unwrap();
        assert_eq!(
            manifest.dependencies.get("gps").unwrap(),
            &Dependency::Git(GitDependency {
                git: "https://github.com/user/repo".to_string(),
                tag: Some("0.1.0".to_string()),
                rev: None,
                subdir: None,
            })
        );
        assert_eq!(manifest.dependencies.get("gps").unwrap().as_range(), None);
        assert_eq!(
            manifest.dependencies.get("gps2").unwrap(),
            &Dependency::Git(GitDependency {
                git: "https://github.com/user/repo".to_string(),
                tag: None,
                rev: Some("abc123".to_string()),
                subdir: None,
            })
        );
    }

    #[test]
    fn git_dependency_parses_and_round_trips_a_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        fs::write(
            &path,
            concat!(
                "[dependencies]\n",
                "gps = { git = \"https://github.com/user/repo\", tag = \"0.1.0\", \
                 subdir = \"laplace\" }\n",
            ),
        )
        .unwrap();

        let manifest = read_project_manifest(&path).unwrap();
        let dep = GitDependency {
            git: "https://github.com/user/repo".to_string(),
            tag: Some("0.1.0".to_string()),
            rev: None,
            subdir: Some("laplace".to_string()),
        };
        assert_eq!(
            manifest.dependencies.get("gps").unwrap(),
            &Dependency::Git(dep.clone())
        );
        assert_eq!(dep.subdir().unwrap(), Some("laplace"));

        // Writing it back and reading it again keeps the subdir.
        let out = dir.path().join("out.toml");
        write_project_manifest(&out, &manifest).unwrap();
        assert_eq!(read_project_manifest(&out).unwrap(), manifest);
    }

    #[test]
    fn a_git_dependency_without_a_subdir_serializes_without_the_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        let mut manifest = ProjectManifest::default();
        manifest.dependencies.insert(
            "gps".to_string(),
            Dependency::Git(GitDependency {
                git: "https://example.com/repo".to_string(),
                tag: Some("0.1.0".to_string()),
                rev: None,
                subdir: None,
            }),
        );
        write_project_manifest(&path, &manifest).unwrap();
        assert!(!fs::read_to_string(&path).unwrap().contains("subdir"));
    }

    #[test]
    fn a_subdir_that_escapes_the_repository_is_rejected() {
        for bad in ["../elsewhere", "/etc", "", "lib#x", "a/../../b"] {
            let dep = GitDependency {
                git: "https://example.com/repo".to_string(),
                tag: Some("0.1.0".to_string()),
                rev: None,
                subdir: Some(bad.to_string()),
            };
            assert!(
                matches!(dep.subdir(), Err(ManifestError::GitSubdirInvalid { .. })),
                "expected `{bad}` to be rejected"
            );
        }
    }

    #[test]
    fn a_nested_subdir_is_accepted() {
        let dep = GitDependency {
            git: "https://example.com/repo".to_string(),
            tag: Some("0.1.0".to_string()),
            rev: None,
            subdir: Some("pkgs/stats".to_string()),
        };
        assert_eq!(dep.subdir().unwrap(), Some("pkgs/stats"));
    }

    #[test]
    fn git_dependency_requires_exactly_one_of_tag_or_rev() {
        let neither = GitDependency {
            git: "https://example.com/repo".to_string(),
            tag: None,
            rev: None,
            subdir: None,
        };
        assert!(matches!(
            neither.git_ref(),
            Err(ManifestError::GitRefMissing { .. })
        ));

        let both = GitDependency {
            git: "https://example.com/repo".to_string(),
            tag: Some("0.1.0".to_string()),
            rev: Some("abc123".to_string()),
            subdir: None,
        };
        assert!(matches!(
            both.git_ref(),
            Err(ManifestError::GitRefAmbiguous { .. })
        ));

        let tag_only = GitDependency {
            git: "https://example.com/repo".to_string(),
            tag: Some("0.1.0".to_string()),
            rev: None,
            subdir: None,
        };
        assert_eq!(tag_only.git_ref().unwrap(), "0.1.0");
    }

    #[test]
    fn package_manifest_parses_name_version_exports() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        fs::write(
            &path,
            "name = \"gps\"\nversion = \"1.0.0\"\nexports = [\"rbf_cov\", \"matern_cov\"]\n",
        )
        .unwrap();

        let manifest = read_package_manifest(&path).unwrap();
        assert_eq!(manifest.name, "gps");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.exports, vec!["rbf_cov", "matern_cov"]);
        assert!(manifest.dependencies.is_empty());
    }

    #[test]
    fn package_manifest_parses_its_own_dependencies() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        fs::write(
            &path,
            concat!(
                "name = \"regression\"\n",
                "version = \"1.0.0\"\n",
                "exports = [\"fit\"]\n",
                "\n[dependencies]\n",
                "stats = \"^1.0\"\n",
                "priors = { git = \"https://example.com/priors\", tag = \"0.2.0\" }\n",
            ),
        )
        .unwrap();

        let manifest = read_package_manifest(&path).unwrap();
        assert_eq!(
            manifest.dependencies.get("stats").unwrap().as_range(),
            Some("^1.0")
        );
        assert!(matches!(
            manifest.dependencies.get("priors").unwrap(),
            Dependency::Git(_)
        ));
    }

    #[test]
    fn a_package_manifest_with_dependencies_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        let mut manifest = PackageManifest::new("regression", "1.0.0", vec!["fit".to_string()]);
        manifest
            .dependencies
            .insert("stats".to_string(), Dependency::Range("^1.0".to_string()));

        write_package_manifest(&path, &manifest).unwrap();
        assert_eq!(read_package_manifest(&path).unwrap(), manifest);
    }

    #[test]
    fn missing_package_manifest_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        assert!(read_package_manifest(&path).is_err());
    }

    #[test]
    fn package_manifest_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.toml");
        let manifest = PackageManifest::new("gps", "0.1.0", vec!["rbf_cov".to_string()]);

        write_package_manifest(&path, &manifest).unwrap();
        assert_eq!(read_package_manifest(&path).unwrap(), manifest);
    }
}
