//! Read/write `laplace.lock`: exact pins + checksums, machine-written.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockedPackage {
    pub name: String,
    pub version: String,
    pub checksum: String,
    /// Where this package came from, so `laplace install` knows how to
    /// refetch it on a fresh machine without re-resolving: `"registry"` for
    /// the local filesystem registry, or `"git+<url>@<tag-or-rev>"` for a
    /// git source. Defaults to `"registry"` when reading an older lockfile
    /// written before this field existed.
    #[serde(default = "default_source")]
    pub source: String,
}

fn default_source() -> String {
    "registry".to_string()
}

/// `laplace.lock`: an array of exact package pins. Always written sorted by
/// name, so the file is stable across writes regardless of insertion order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lockfile {
    #[serde(rename = "package", default)]
    pub packages: Vec<LockedPackage>,
}

#[derive(Debug, Error)]
pub enum LockfileError {
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

    #[error("failed to serialize lockfile for {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: toml::ser::Error,
    },
}

/// Read `laplace.lock`. A missing file is treated as an empty lock (nothing
/// installed yet), not an error.
pub fn read_lockfile(path: &Path) -> Result<Lockfile, LockfileError> {
    match fs::read_to_string(path) {
        Ok(text) => toml::from_str(&text).map_err(|source| LockfileError::Parse {
            path: path.to_path_buf(),
            source,
        }),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(Lockfile::default()),
        Err(source) => Err(LockfileError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

pub fn write_lockfile(path: &Path, lock: &Lockfile) -> Result<(), LockfileError> {
    let mut sorted = lock.clone();
    sorted.packages.sort_by(|a, b| a.name.cmp(&b.name));

    let text = toml::to_string_pretty(&sorted).map_err(|source| LockfileError::Serialize {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, text).map_err(|source| LockfileError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_lockfile_reads_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.lock");
        assert_eq!(read_lockfile(&path).unwrap(), Lockfile::default());
    }

    #[test]
    fn write_then_read_round_trips_sorted_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.lock");
        let lock = Lockfile {
            packages: vec![
                LockedPackage {
                    name: "zzz".into(),
                    version: "1.0.0".into(),
                    checksum: "sha256:aaa".into(),
                    source: "registry".into(),
                },
                LockedPackage {
                    name: "aaa".into(),
                    version: "2.0.0".into(),
                    checksum: "sha256:bbb".into(),
                    source: "git+https://github.com/user/repo@0.1.0".into(),
                },
            ],
        };

        write_lockfile(&path, &lock).unwrap();
        let read_back = read_lockfile(&path).unwrap();
        assert_eq!(
            read_back
                .packages
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec!["aaa", "zzz"]
        );
        assert_eq!(read_back.packages[0].checksum, "sha256:bbb");
    }

    #[test]
    fn lockfile_without_source_field_defaults_to_registry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.lock");
        fs::write(
            &path,
            "[[package]]\nname = \"gps\"\nversion = \"1.0.0\"\nchecksum = \"sha256:aaa\"\n",
        )
        .unwrap();

        let lock = read_lockfile(&path).unwrap();
        assert_eq!(lock.packages[0].source, "registry");
    }

    #[test]
    fn malformed_lockfile_is_a_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.lock");
        fs::write(&path, "this is not valid toml {{{").unwrap();
        assert!(matches!(
            read_lockfile(&path),
            Err(LockfileError::Parse { .. })
        ));
    }
}
