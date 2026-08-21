//! Fetching a package from a git repository, for the `git = "..."` table
//! form of a dependency (see `crate::manifest::GitDependency`).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GitError {
    #[error("failed to run `git {args}`: {source}")]
    Spawn {
        args: String,
        #[source]
        source: io::Error,
    },

    #[error("`git {args}` failed: {stderr}")]
    CommandFailed { args: String, stderr: String },

    #[error("failed to prepare git checkout at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

fn run(args: &[&str], cwd: Option<&Path>) -> Result<(), GitError> {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let joined = args.join(" ");
    let output = cmd.output().map_err(|source| GitError::Spawn {
        args: joined.clone(),
        source,
    })?;
    if !output.status.success() {
        return Err(GitError::CommandFailed {
            args: joined,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(())
}

/// Fetch `git_ref` (a tag, branch, or commit) from `url` into `dest`, an
/// already-existing empty directory. Tries a fast shallow clone first
/// (`git clone --depth 1 --branch <ref>`), which works for tags and
/// branches; if that fails -- e.g. `git_ref` is an arbitrary commit, which
/// `--branch` can't shallow-clone -- falls back to a full clone plus an
/// explicit `git checkout <ref>`. Strips the `.git` directory from the
/// result before returning, since only the tracked files are the package.
pub fn fetch(url: &str, git_ref: &str, dest: &Path) -> Result<(), GitError> {
    let dest_str = dest.to_string_lossy().into_owned();

    let shallow = run(
        &[
            "clone", "--quiet", "--depth", "1", "--branch", git_ref, url, &dest_str,
        ],
        None,
    );

    if shallow.is_err() {
        if dest.is_dir() {
            fs::remove_dir_all(dest).map_err(|source| GitError::Io {
                path: dest.to_path_buf(),
                source,
            })?;
        }
        run(&["clone", "--quiet", url, &dest_str], None)?;
        run(&["checkout", "--quiet", git_ref], Some(dest))?;
    }

    let git_dir = dest.join(".git");
    if git_dir.is_dir() {
        fs::remove_dir_all(&git_dir).map_err(|source| GitError::Io {
            path: git_dir.clone(),
            source,
        })?;
    }

    Ok(())
}

/// The `laplace.lock` `source` string for a git-sourced package.
pub fn git_source(url: &str, git_ref: &str) -> String {
    format!("git+{url}@{git_ref}")
}

/// Parse a `laplace.lock` `source` string back into `(url, git_ref)`, or
/// `None` if it isn't a git source (e.g. `"registry"`). Splits on the last
/// `@` so scp-style urls containing their own `@` (`git@host:user/repo`)
/// still parse correctly.
pub fn parse_git_source(source: &str) -> Option<(&str, &str)> {
    source.strip_prefix("git+")?.rsplit_once('@')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_source_round_trips_through_parse() {
        let source = git_source("https://github.com/user/repo", "0.1.0");
        assert_eq!(source, "git+https://github.com/user/repo@0.1.0");
        assert_eq!(
            parse_git_source(&source),
            Some(("https://github.com/user/repo", "0.1.0"))
        );
    }

    #[test]
    fn parse_git_source_handles_scp_style_urls_with_embedded_at() {
        let source = git_source("git@github.com:user/repo", "abc123");
        assert_eq!(
            parse_git_source(&source),
            Some(("git@github.com:user/repo", "abc123"))
        );
    }

    #[test]
    fn parse_git_source_rejects_non_git_sources() {
        assert_eq!(parse_git_source("registry"), None);
    }
}
