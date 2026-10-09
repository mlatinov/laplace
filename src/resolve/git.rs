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

/// Run `git` and return its stdout.
fn output(args: &[&str], cwd: Option<&Path>) -> Result<String, GitError> {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let joined = args.join(" ");
    let out = cmd.output().map_err(|source| GitError::Spawn {
        args: joined.clone(),
        source,
    })?;
    if !out.status.success() {
        return Err(GitError::CommandFailed {
            args: joined,
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The tag and branch names a remote advertises.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RemoteRefs {
    /// Tag names, sorted (peeled `^{}` duplicates removed).
    pub tags: Vec<String>,
    /// Branch names, sorted.
    pub branches: Vec<String>,
}

impl RemoteRefs {
    /// Whether `name` is a tag or a branch on the remote.
    pub fn has(&self, name: &str) -> bool {
        self.tags.iter().any(|t| t == name) || self.branches.iter().any(|b| b == name)
    }

    /// Parse `git ls-remote --tags --heads` output.
    pub fn parse(listing: &str) -> RemoteRefs {
        let mut refs = RemoteRefs::default();
        for line in listing.lines() {
            let Some((_, name)) = line.split_once('\t') else {
                continue;
            };
            if let Some(tag) = name.strip_prefix("refs/tags/") {
                let tag = tag.strip_suffix("^{}").unwrap_or(tag);
                refs.tags.push(tag.to_string());
            } else if let Some(branch) = name.strip_prefix("refs/heads/") {
                refs.branches.push(branch.to_string());
            }
        }
        refs.tags.sort();
        refs.tags.dedup();
        refs.branches.sort();
        refs
    }
}

/// List a remote's tags and branches without cloning it.
pub fn remote_refs(url: &str) -> Result<RemoteRefs, GitError> {
    output(&["ls-remote", "--tags", "--heads", url], None).map(|text| RemoteRefs::parse(&text))
}

/// The version a tag names, if it names one: `1.2.0` or `v1.2.0`.
pub fn version_in_tag(tag: &str) -> Option<semver::Version> {
    let bare = tag.strip_prefix('v').unwrap_or(tag);
    semver::Version::parse(bare).ok()
}

/// The tag naming the highest version, if any tag names a version at all.
pub fn newest_version_tag(tags: &[String]) -> Option<&str> {
    tags.iter()
        .filter_map(|tag| version_in_tag(tag).map(|v| (v, tag)))
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, tag)| tag.as_str())
}

/// Whether `git_ref` could be a commit sha (abbreviated or full), which a
/// tag listing cannot confirm or deny.
pub fn looks_like_commit(git_ref: &str) -> bool {
    (7..=40).contains(&git_ref.len()) && git_ref.chars().all(|c| c.is_ascii_hexdigit())
}

/// The `laplace.lock` `source` string for a git-sourced package:
/// `git+<url>@<ref>`, with `#<subdir>` appended when the package is rooted
/// in a subdirectory of the repository rather than at its top level.
pub fn git_source(url: &str, git_ref: &str, subdir: Option<&str>) -> String {
    match subdir {
        Some(subdir) => format!("git+{url}@{git_ref}#{subdir}"),
        None => format!("git+{url}@{git_ref}"),
    }
}

/// Parse a `laplace.lock` `source` string back into `(url, git_ref,
/// subdir)`, or `None` if it isn't a git source (e.g. `"registry"`). The
/// `#<subdir>` fragment is split off first, then the remainder is split on
/// its *last* `@` so scp-style urls containing their own `@`
/// (`git@host:user/repo`) still parse correctly.
pub fn parse_git_source(source: &str) -> Option<(&str, &str, Option<&str>)> {
    let rest = source.strip_prefix("git+")?;
    let (rest, subdir) = match rest.split_once('#') {
        Some((rest, subdir)) => (rest, Some(subdir)),
        None => (rest, None),
    };
    let (url, git_ref) = rest.rsplit_once('@')?;
    Some((url, git_ref, subdir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_source_round_trips_through_parse() {
        let source = git_source("https://github.com/user/repo", "0.1.0", None);
        assert_eq!(source, "git+https://github.com/user/repo@0.1.0");
        assert_eq!(
            parse_git_source(&source),
            Some(("https://github.com/user/repo", "0.1.0", None))
        );
    }

    #[test]
    fn git_source_round_trips_with_a_subdir() {
        let source = git_source("https://github.com/user/repo", "0.1.0", Some("laplace"));
        assert_eq!(source, "git+https://github.com/user/repo@0.1.0#laplace");
        assert_eq!(
            parse_git_source(&source),
            Some(("https://github.com/user/repo", "0.1.0", Some("laplace")))
        );
    }

    #[test]
    fn nested_subdir_survives_the_round_trip() {
        let source = git_source("https://example.com/repo", "v2", Some("pkgs/stats"));
        assert_eq!(
            parse_git_source(&source),
            Some(("https://example.com/repo", "v2", Some("pkgs/stats")))
        );
    }

    #[test]
    fn parse_git_source_handles_scp_style_urls_with_embedded_at() {
        let source = git_source("git@github.com:user/repo", "abc123", None);
        assert_eq!(
            parse_git_source(&source),
            Some(("git@github.com:user/repo", "abc123", None))
        );
        let with_subdir = git_source("git@github.com:user/repo", "abc123", Some("lib"));
        assert_eq!(
            parse_git_source(&with_subdir),
            Some(("git@github.com:user/repo", "abc123", Some("lib")))
        );
    }

    #[test]
    fn remote_refs_parse_tags_and_branches_and_drop_peeled_duplicates() {
        let listing = "aaa\trefs/heads/main\nbbb\trefs/tags/0.1.0\nccc\trefs/tags/0.1.0^{}\n\
                       ddd\trefs/tags/v0.2.0\n";
        let refs = RemoteRefs::parse(listing);
        assert_eq!(refs.tags, vec!["0.1.0", "v0.2.0"]);
        assert_eq!(refs.branches, vec!["main"]);
        assert!(refs.has("main") && refs.has("v0.2.0") && !refs.has("0.3.0"));
    }

    #[test]
    fn the_newest_tag_is_chosen_by_version_not_by_name() {
        let tags: Vec<String> = ["0.9.0", "0.10.0", "v0.2.0", "latest"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(newest_version_tag(&tags), Some("0.10.0"));
        assert_eq!(newest_version_tag(&["latest".to_string()]), None);
    }

    #[test]
    fn version_in_tag_accepts_an_optional_v_prefix() {
        assert_eq!(
            version_in_tag("v1.2.3"),
            semver::Version::parse("1.2.3").ok()
        );
        assert_eq!(
            version_in_tag("1.2.3"),
            semver::Version::parse("1.2.3").ok()
        );
        assert_eq!(version_in_tag("abc1234"), None);
    }

    #[test]
    fn commit_shas_are_recognised() {
        assert!(looks_like_commit("abc1234"));
        assert!(looks_like_commit(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(!looks_like_commit("0.1.0"));
        assert!(!looks_like_commit("main"));
    }

    #[test]
    fn parse_git_source_rejects_non_git_sources() {
        assert_eq!(parse_git_source("registry"), None);
    }
}
