//! Read/write `laplace.lock`: exact pins + checksums, machine-written.
//!
//! # The lock is a graph, not a list
//!
//! Since a package can declare its own `[dependencies]` (see the
//! `.laplacelib` dialect), the lock has to record the *whole* dependency
//! DAG, not just the project's direct pins:
//!
//! - `root` lists the package names the project itself depends on. Only
//!   these may be imported in a `.laplace` file's `library { }` block.
//! - each `[[package]]` carries its own `dependencies` -- the names of the
//!   packages it depends on, whose exact versions are themselves
//!   `[[package]]` entries in the same file.
//!
//! That is what lets `laplace install` reconstruct the full graph without
//! re-resolving any version range, which is the whole point of the lock.
//!
//! Older locks written before this format existed have neither field. They
//! still read fine: a missing `dependencies` means "no transitive deps",
//! and a missing/empty `root` means "every locked package is a direct
//! dependency" (see [`Lockfile::root_names`]), which is exactly what those
//! locks meant when they were written.

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
    /// Names of the packages this one depends on, sorted. Each is itself a
    /// `[[package]]` entry in this same lockfile.
    #[serde(default)]
    pub dependencies: Vec<String>,
}

impl LockedPackage {
    /// A leaf pin: no dependencies of its own. Most packages, and every
    /// package written by a pre-DAG version of laplace.
    pub fn leaf(name: &str, version: &str, checksum: &str, source: &str) -> Self {
        LockedPackage {
            name: name.to_string(),
            version: version.to_string(),
            checksum: checksum.to_string(),
            source: source.to_string(),
            dependencies: Vec::new(),
        }
    }
}

fn default_source() -> String {
    "registry".to_string()
}

/// `laplace.lock`: the project's direct dependency names plus an array of
/// exact package pins covering the full transitive graph. Always written
/// sorted by name, so the file is stable across writes regardless of
/// insertion order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lockfile {
    /// The names the top-level project depends on directly. Sorted.
    #[serde(default)]
    pub root: Vec<String>,
    #[serde(rename = "package", default)]
    pub packages: Vec<LockedPackage>,
}

impl Lockfile {
    /// The project's direct dependencies. Falls back to "every locked
    /// package" for a lockfile written before `root` existed, which had no
    /// way to express transitive pins and so listed direct deps only.
    pub fn root_names(&self) -> Vec<String> {
        if self.root.is_empty() {
            let mut names: Vec<String> = self.packages.iter().map(|p| p.name.clone()).collect();
            names.sort();
            names.dedup();
            names
        } else {
            self.root.clone()
        }
    }

    pub fn get(&self, name: &str) -> Option<&LockedPackage> {
        self.packages.iter().find(|p| p.name == name)
    }

    /// `name` plus everything it transitively depends on, sorted and
    /// deduplicated. Tolerates a cyclic or dangling lock without looping.
    pub fn closure(&self, names: &[String]) -> Vec<&LockedPackage> {
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        let mut queue: Vec<&str> = names.iter().map(String::as_str).collect();

        while let Some(name) = queue.pop() {
            if !seen.insert(name) {
                continue;
            }
            if let Some(pkg) = self.get(name) {
                queue.extend(pkg.dependencies.iter().map(String::as_str));
            }
        }

        let mut out: Vec<&LockedPackage> = seen.iter().filter_map(|n| self.get(n)).collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
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
    for pkg in &mut sorted.packages {
        pkg.dependencies.sort();
        pkg.dependencies.dedup();
    }
    sorted.root.sort();
    sorted.root.dedup();

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

    fn leaf(name: &str, version: &str) -> LockedPackage {
        LockedPackage::leaf(name, version, "sha256:aaa", "registry")
    }

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
            root: vec!["zzz".into(), "aaa".into()],
            packages: vec![
                LockedPackage::leaf("zzz", "1.0.0", "sha256:aaa", "registry"),
                LockedPackage::leaf(
                    "aaa",
                    "2.0.0",
                    "sha256:bbb",
                    "git+https://github.com/user/repo@0.1.0",
                ),
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
        assert_eq!(read_back.root, vec!["aaa", "zzz"]);
    }

    #[test]
    fn the_full_dependency_graph_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.lock");
        let lock = Lockfile {
            root: vec!["regression".into()],
            packages: vec![
                LockedPackage {
                    dependencies: vec!["stats".into()],
                    ..leaf("regression", "1.0.0")
                },
                leaf("stats", "2.1.0"),
            ],
        };

        write_lockfile(&path, &lock).unwrap();
        let read_back = read_lockfile(&path).unwrap();

        assert_eq!(read_back.root, vec!["regression"]);
        assert_eq!(
            read_back.get("regression").unwrap().dependencies,
            vec!["stats"]
        );
        assert!(read_back.get("stats").unwrap().dependencies.is_empty());

        // ...and writing what we read back is a fixpoint, so a committed
        // lock never churns just from being rewritten.
        let again = path.with_extension("lock2");
        write_lockfile(&again, &read_back).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            fs::read_to_string(&again).unwrap()
        );
    }

    #[test]
    fn closure_walks_transitive_dependencies() {
        let lock = Lockfile {
            root: vec!["app".into()],
            packages: vec![
                LockedPackage {
                    dependencies: vec!["mid".into()],
                    ..leaf("app", "1.0.0")
                },
                LockedPackage {
                    dependencies: vec!["base".into()],
                    ..leaf("mid", "1.0.0")
                },
                leaf("base", "1.0.0"),
                leaf("unrelated", "1.0.0"),
            ],
        };

        let names: Vec<&str> = lock
            .closure(&lock.root_names())
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["app", "base", "mid"]);
    }

    #[test]
    fn closure_terminates_on_a_corrupt_cyclic_lock() {
        let lock = Lockfile {
            root: vec!["a".into()],
            packages: vec![
                LockedPackage {
                    dependencies: vec!["b".into()],
                    ..leaf("a", "1.0.0")
                },
                LockedPackage {
                    dependencies: vec!["a".into(), "ghost".into()],
                    ..leaf("b", "1.0.0")
                },
            ],
        };
        let names: Vec<&str> = lock
            .closure(&["a".to_string()])
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b"]);
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
    fn a_pre_dag_lockfile_reads_every_package_as_a_direct_dependency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("laplace.lock");
        fs::write(
            &path,
            concat!(
                "[[package]]\nname = \"gps\"\nversion = \"1.0.0\"\nchecksum = \"sha256:a\"\n",
                "[[package]]\nname = \"abc\"\nversion = \"2.0.0\"\nchecksum = \"sha256:b\"\n",
            ),
        )
        .unwrap();

        let lock = read_lockfile(&path).unwrap();
        assert!(lock.root.is_empty());
        assert_eq!(lock.root_names(), vec!["abc", "gps"]);
        assert!(lock.packages.iter().all(|p| p.dependencies.is_empty()));
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
