//! Path dependencies: a package taken straight from a local directory
//! (see `crate::manifest::PathDependency`).
//!
//! The lock records where the package root is *relative to the project*
//! (`path+../my-lib`; a `subdir` is already joined on, `path+../repo/lib`),
//! never an absolute path, so a committed lock does not name one
//! developer's home directory. Path
//! packages are cached apart from versioned ones -- under
//! `path-packages/<name>/<hash of the directory>/`, beside the shared
//! `packages/` cache -- because a local checkout claiming `gps@1.0.0` must
//! never overwrite the real `gps@1.0.0` other projects build against.

use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

/// The `laplace.lock` `source` string for a path package rooted at
/// `relative_root` (relative to the project directory).
pub fn path_source(relative_root: &str) -> String {
    format!("path+{relative_root}")
}

/// The project-relative package root of a `path+<dir>` source, or `None` if
/// it is not a path source.
pub fn parse_path_source(source: &str) -> Option<&str> {
    source.strip_prefix("path+")
}

/// `target` relative to `base`, both absolute, with `/` separators.
/// Falls back to `target` itself when the two share no root (different
/// Windows drives).
pub fn relative_to(target: &Path, base: &Path) -> String {
    let target: Vec<Component> = target.components().collect();
    let base: Vec<Component> = base.components().collect();
    if target.first() != base.first() {
        return slashes(&target.iter().collect::<PathBuf>());
    }
    let common = target.iter().zip(&base).take_while(|(a, b)| a == b).count();
    let mut parts: Vec<String> = Vec::new();
    for _ in common..base.len() {
        parts.push("..".to_string());
    }
    for component in &target[common..] {
        parts.push(component.as_os_str().to_string_lossy().into_owned());
    }
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join("/")
    }
}

fn slashes(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Where the path package rooted at `root` (absolute) is cached:
/// `path-packages/<name>/<16 hex digits of sha256(root)>/`, beside
/// `cache_root`.
pub fn cache_dir(cache_root: &Path, name: &str, root: &Path) -> PathBuf {
    let digest = Sha256::digest(root.to_string_lossy().as_bytes());
    let hash: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    cache_root
        .with_file_name("path-packages")
        .join(name)
        .join(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_source_round_trips() {
        assert_eq!(path_source("../gps"), "path+../gps");
        assert_eq!(parse_path_source("path+../repo/lib"), Some("../repo/lib"));
        assert_eq!(parse_path_source("registry"), None);
        assert_eq!(parse_path_source("git+https://x@1"), None);
    }

    #[test]
    fn relative_paths_climb_out_of_the_base_and_back_down() {
        assert_eq!(
            relative_to(Path::new("/work/libs/gps"), Path::new("/work/project")),
            "../libs/gps"
        );
        assert_eq!(
            relative_to(
                Path::new("/work/project/vendor/gps"),
                Path::new("/work/project")
            ),
            "vendor/gps"
        );
        assert_eq!(relative_to(Path::new("/work"), Path::new("/work")), ".");
    }

    #[test]
    fn different_directories_get_different_cache_entries() {
        let cache = Path::new("/home/u/.laplace/packages");
        let a = cache_dir(cache, "gps", Path::new("/work/a/gps"));
        let b = cache_dir(cache, "gps", Path::new("/work/b/gps"));
        assert_ne!(a, b);
        assert!(a.starts_with("/home/u/.laplace/path-packages/gps"));
        assert_eq!(a, cache_dir(cache, "gps", Path::new("/work/a/gps")));
    }
}
