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
/// a tag or a commit rev (exactly one of the two).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitDependency {
    pub git: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rev: Option<String>,
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
}

/// Package-level `laplace.toml`, shipped alongside a package's `.stan`
/// file(s) in the registry.
///
/// ```toml
/// name = "gps"
/// version = "1.0.0"
/// exports = ["rbf_cov"]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageManifest {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub exports: Vec<String>,
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

/// Read and concatenate every `.stan` file directly inside `package_dir`, in
/// filename order, so codegen and doc extraction always agree on what a
/// package's source is.
pub fn read_package_stan_source(package_dir: &Path) -> std::io::Result<String> {
    let mut stan_files: Vec<PathBuf> = fs::read_dir(package_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("stan"))
        .collect();
    stan_files.sort();

    let mut source = String::new();
    for path in &stan_files {
        source.push_str(&fs::read_to_string(path)?);
        source.push('\n');
    }
    Ok(source)
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
            })
        );
        assert_eq!(
            manifest
                .dependencies
                .get("gps")
                .unwrap()
                .as_range(),
            None
        );
        assert_eq!(
            manifest.dependencies.get("gps2").unwrap(),
            &Dependency::Git(GitDependency {
                git: "https://github.com/user/repo".to_string(),
                tag: None,
                rev: Some("abc123".to_string()),
            })
        );
    }

    #[test]
    fn git_dependency_requires_exactly_one_of_tag_or_rev() {
        let neither = GitDependency {
            git: "https://example.com/repo".to_string(),
            tag: None,
            rev: None,
        };
        assert!(matches!(
            neither.git_ref(),
            Err(ManifestError::GitRefMissing { .. })
        ));

        let both = GitDependency {
            git: "https://example.com/repo".to_string(),
            tag: Some("0.1.0".to_string()),
            rev: Some("abc123".to_string()),
        };
        assert!(matches!(
            both.git_ref(),
            Err(ManifestError::GitRefAmbiguous { .. })
        ));

        let tag_only = GitDependency {
            git: "https://example.com/repo".to_string(),
            tag: Some("0.1.0".to_string()),
            rev: None,
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
        let manifest = PackageManifest {
            name: "gps".to_string(),
            version: "0.1.0".to_string(),
            exports: vec!["rbf_cov".to_string()],
        };

        write_package_manifest(&path, &manifest).unwrap();
        assert_eq!(read_package_manifest(&path).unwrap(), manifest);
    }
}
