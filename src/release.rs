//! `laplace release <version|patch|minor|major>`: tag a *package* release.
//!
//! Not to be confused with a *compiler* release (shipping the `laplace`
//! binary, see RELEASING.md). This is for a library author: bump the version
//! in the package's `laplace.toml`, commit, tag, push -- the tag being what
//! `laplace add <pkg> --git <url> --tag <tag>` points at.
//!
//! Split in two so `--dry-run` is honest: [`plan`] runs every check and
//! computes every step but changes nothing; [`execute`] carries the steps
//! out. A dry run is exactly `plan` plus printing.

use std::path::{Path, PathBuf};
use std::process::Command;

use semver::Version;
use thiserror::Error;
use toml_edit::DocumentMut;

use crate::manifest::{self, Dependency, ManifestError};
use crate::package::{self, PackageError};
use crate::parser::signatures::extract_signatures;

#[derive(Debug, Error)]
pub enum ReleaseError {
    #[error(
        "no laplace.toml in {} -- run `laplace release` from the package directory, or pass \
         `--path <dir>`{hint}",
        .dir.display()
    )]
    NoManifest { dir: PathBuf, hint: String },

    #[error(transparent)]
    Manifest(#[from] ManifestError),

    #[error(transparent)]
    Package(Box<PackageError>),

    #[error("`{0}` is not a version or one of patch/minor/major")]
    BadBump(String),

    #[error(
        "{new} is not newer than the current version {current} -- a release must move the \
         version forward"
    )]
    NotNewer { current: String, new: String },

    #[error("{} is not inside a git repository", .0.display())]
    NotARepo(PathBuf),

    #[error("tracked files have uncommitted changes -- commit or stash them first:\n{0}")]
    Dirty(String),

    #[error("HEAD is detached -- check out the branch you want to release from")]
    DetachedHead,

    #[error(
        "branch `{branch}` has no upstream to push to\n  help: push it once with `git push -u \
         <remote> {branch}`"
    )]
    NoUpstream { branch: String },

    #[error(
        "branch `{branch}` is {behind} commit(s) behind `{upstream}` -- pull (and re-test) \
         before releasing"
    )]
    Behind {
        branch: String,
        upstream: String,
        behind: usize,
    },

    #[error("tag `{tag}` already exists {location} -- pick another version")]
    TagExists { tag: String, location: &'static str },

    #[error("the package is not ready to release:\n{}", .0.iter().map(|p| format!("  - {p}")).collect::<Vec<_>>().join("\n"))]
    Invalid(Vec<String>),

    #[error("`git {args}` failed: {stderr}")]
    Git { args: String, stderr: String },

    #[error("failed to run git: {0}")]
    GitSpawn(#[source] std::io::Error),

    #[error("failed to write {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Everything [`execute`] will do, computed and checked up front.
#[derive(Debug)]
pub struct ReleasePlan {
    pub package: String,
    pub manifest_path: PathBuf,
    /// The git work tree root.
    pub repo: PathBuf,
    pub current: Version,
    pub new: Version,
    pub tag: String,
    /// Why this tag spelling was chosen.
    pub tag_reason: String,
    pub remote: String,
    pub branch: String,
    /// Non-fatal observations, printed before the steps.
    pub warnings: Vec<String>,
    /// Human-readable steps, in order.
    pub steps: Vec<String>,
}

/// Find the package directory: `dir` itself if it holds a `laplace.toml`,
/// otherwise its only immediate subdirectory that does.
pub fn find_package_dir(dir: &Path) -> Result<PathBuf, ReleaseError> {
    if dir.join("laplace.toml").is_file() {
        return Ok(dir.to_path_buf());
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.join("laplace.toml").is_file())
                .collect()
        })
        .unwrap_or_default();
    candidates.sort();
    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => Err(ReleaseError::NoManifest {
            dir: dir.to_path_buf(),
            hint: String::new(),
        }),
        _ => Err(ReleaseError::NoManifest {
            dir: dir.to_path_buf(),
            hint: format!(
                " (candidates: {})",
                candidates
                    .iter()
                    .filter_map(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}

/// Check everything and work out the steps. Changes nothing -- except that
/// it fetches the remote's tags, read-only, so "is it up to date" and "does
/// the tag exist" are answered against the remote rather than a stale view.
pub fn plan(package_dir: &Path, bump: &str) -> Result<ReleasePlan, ReleaseError> {
    let package_dir = &package_dir
        .canonicalize()
        .map_err(|source| ReleaseError::Io {
            path: package_dir.to_path_buf(),
            source,
        })?;
    let manifest_path = package_dir.join("laplace.toml");
    let pkg_manifest = manifest::read_package_manifest(&manifest_path)?;
    let current = Version::parse(&pkg_manifest.version).map_err(|_| {
        ReleaseError::Invalid(vec![format!(
            "laplace.toml's version `{}` is not a semver version",
            pkg_manifest.version
        )])
    })?;
    let new = bumped(&current, bump)?;
    if new <= current {
        return Err(ReleaseError::NotNewer {
            current: current.to_string(),
            new: new.to_string(),
        });
    }

    let mut warnings = Vec::new();
    validate_package(package_dir, &pkg_manifest, &mut warnings)?;

    // -- git state ------------------------------------------------------
    let repo = PathBuf::from(
        git(package_dir, &["rev-parse", "--show-toplevel"])
            .map_err(|_| ReleaseError::NotARepo(package_dir.to_path_buf()))?,
    );
    let dirty = git(&repo, &["status", "--porcelain", "--untracked-files=no"])?;
    if !dirty.is_empty() {
        return Err(ReleaseError::Dirty(dirty));
    }
    let branch = git(&repo, &["symbolic-ref", "--quiet", "--short", "HEAD"])
        .map_err(|_| ReleaseError::DetachedHead)?;
    let upstream = git(
        &repo,
        &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
    )
    .map_err(|_| ReleaseError::NoUpstream {
        branch: branch.clone(),
    })?;
    let remote = git(&repo, &["config", &format!("branch.{branch}.remote")])
        .unwrap_or_else(|_| "origin".to_string());
    git(&repo, &["fetch", "--quiet", "--tags", &remote])?;
    let counts = git(
        &repo,
        &["rev-list", "--left-right", "--count", "HEAD...@{u}"],
    )?;
    let behind: usize = counts
        .split_whitespace()
        .nth(1)
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    if behind > 0 {
        return Err(ReleaseError::Behind {
            branch,
            upstream,
            behind,
        });
    }

    // -- tag ------------------------------------------------------------
    let local_tags: Vec<String> = git(&repo, &["tag", "--list"])?
        .lines()
        .map(str::to_string)
        .collect();
    let (prefix, tag_reason) = tag_prefix(&local_tags);
    let tag = format!("{prefix}{new}");
    if local_tags.contains(&tag) {
        return Err(ReleaseError::TagExists {
            tag,
            location: "locally",
        });
    }
    let on_remote = git(
        &repo,
        &["ls-remote", "--tags", &remote, &format!("refs/tags/{tag}")],
    )?;
    if !on_remote.is_empty() {
        return Err(ReleaseError::TagExists {
            tag,
            location: "on the remote",
        });
    }

    let relative_manifest = manifest_path
        .canonicalize()
        .ok()
        .and_then(|m| {
            let root = repo.canonicalize().ok()?;
            m.strip_prefix(root).ok().map(Path::to_path_buf)
        })
        .unwrap_or_else(|| manifest_path.clone());
    let steps = vec![
        format!(
            "set version = \"{new}\" in {} (was {current})",
            relative_manifest.display()
        ),
        format!("commit it: \"release {new}\""),
        format!("tag the commit `{tag}`"),
        format!("push branch `{branch}` to `{remote}`"),
        format!("push tag `{tag}` to `{remote}`"),
    ];

    Ok(ReleasePlan {
        package: pkg_manifest.name,
        manifest_path,
        repo,
        current,
        new,
        tag,
        tag_reason,
        remote,
        branch,
        warnings,
        steps,
    })
}

/// Carry out a [`plan`].
pub fn execute(plan: &ReleasePlan) -> Result<(), ReleaseError> {
    let text = std::fs::read_to_string(&plan.manifest_path).map_err(|source| ReleaseError::Io {
        path: plan.manifest_path.clone(),
        source,
    })?;
    let mut doc: DocumentMut = text.parse().map_err(|e: toml_edit::TomlError| {
        ReleaseError::Invalid(vec![format!("laplace.toml does not parse: {e}")])
    })?;
    doc["version"] = toml_edit::value(plan.new.to_string());
    std::fs::write(&plan.manifest_path, doc.to_string()).map_err(|source| ReleaseError::Io {
        path: plan.manifest_path.clone(),
        source,
    })?;

    let manifest = plan.manifest_path.to_string_lossy().into_owned();
    let message = format!("release {}", plan.new);
    git(&plan.repo, &["add", "--", &manifest])?;
    git(&plan.repo, &["commit", "--quiet", "-m", &message])?;
    git(&plan.repo, &["tag", "-a", &plan.tag, "-m", &message])?;
    git(&plan.repo, &["push", "--quiet", &plan.remote, &plan.branch])?;
    git(
        &plan.repo,
        &[
            "push",
            "--quiet",
            &plan.remote,
            &format!("refs/tags/{}", plan.tag),
        ],
    )?;
    Ok(())
}

/// The version `bump` asks for, relative to `current`.
fn bumped(current: &Version, bump: &str) -> Result<Version, ReleaseError> {
    let mut next = Version::new(current.major, current.minor, current.patch);
    match bump {
        "patch" => {
            // A pre-release of x.y.z releases as x.y.z itself.
            if current.pre.is_empty() {
                next.patch += 1;
            }
        }
        "minor" => {
            next.minor += 1;
            next.patch = 0;
        }
        "major" => {
            next.major += 1;
            next.minor = 0;
            next.patch = 0;
        }
        explicit => {
            let bare = explicit.strip_prefix('v').unwrap_or(explicit);
            next = Version::parse(bare).map_err(|_| ReleaseError::BadBump(bump.to_string()))?;
        }
    }
    Ok(next)
}

/// Match the repository's existing tag style: `v1.2.3` or bare `1.2.3`,
/// whichever its version tags mostly use. With no version tags at all, the
/// bare version -- what laplace's own docs write after `--tag`.
fn tag_prefix(tags: &[String]) -> (&'static str, String) {
    let with_v: Vec<&String> = tags
        .iter()
        .filter(|t| {
            t.strip_prefix('v')
                .is_some_and(|b| Version::parse(b).is_ok())
        })
        .collect();
    let bare: Vec<&String> = tags.iter().filter(|t| Version::parse(t).is_ok()).collect();
    let sample = |list: &[&String]| {
        list.iter()
            .rev()
            .take(3)
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if with_v.is_empty() && bare.is_empty() {
        (
            "",
            "no version tags yet, so the bare version is used (laplace's convention)".to_string(),
        )
    } else if with_v.len() > bare.len() {
        (
            "v",
            format!("existing tags use the `v` prefix ({})", sample(&with_v)),
        )
    } else {
        (
            "",
            format!("existing tags have no prefix ({})", sample(&bare)),
        )
    }
}

/// What makes a package releasable, beyond its manifest parsing: it loads
/// the way a consumer would load it, every `// @laplace` block documents
/// something, a `.laplacelib` package exports something, and nothing it
/// depends on is a local path. Every problem is collected, not just the
/// first.
fn validate_package(
    dir: &Path,
    pkg_manifest: &manifest::PackageManifest,
    warnings: &mut Vec<String>,
) -> Result<(), ReleaseError> {
    let mut problems = Vec::new();

    if let Err(err) = package::load_with_manifest(dir, &pkg_manifest.name, pkg_manifest) {
        problems.push(err.to_string());
    }

    let sources =
        package::read_package_sources(dir).map_err(|e| ReleaseError::Package(Box::new(e)))?;
    if !sources.laplacelib_items.is_empty() && sources.public_items.is_empty() {
        problems.push(
            "no `.laplacelib` item is marked `pub`, so the package would export nothing -- \
             write `pub` in front of its public functions"
                .to_string(),
        );
    }

    for (file, line) in unattached_doc_blocks(dir) {
        problems.push(format!(
            "{file}:{line}: this `// @laplace` block is not directly above a declaration, so \
             `laplace doc` will never show it"
        ));
    }

    for (name, dep) in &pkg_manifest.dependencies {
        if let Dependency::Path(path_dep) = dep {
            problems.push(format!(
                "dependency `{name}` is a path dependency (`{}`), which no consumer of the \
                 release can resolve -- depend on a released version instead",
                path_dep.path
            ));
        }
    }

    // A lock next to the package (its own test project) may use path
    // dependencies; that does not leak into the release, but it means the
    // tests that passed ran against unreleased code.
    if let Ok(lock) = crate::resolve::lockfile::read_lockfile(&dir.join("laplace.lock")) {
        for pkg in lock
            .packages
            .iter()
            .filter(|p| p.source.starts_with("path+"))
        {
            warnings.push(format!(
                "laplace.lock here uses `{}` from {} -- what you tested against is not a \
                 release, and nobody else can reproduce it",
                pkg.name, pkg.source
            ));
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(ReleaseError::Invalid(problems))
    }
}

/// `// @laplace` marker lines that do not start the doc comment of a
/// function, template or macro, as `(file name, line)`.
fn unattached_doc_blocks(dir: &Path) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("stan") | Some("laplacelib")
            )
        })
        .collect();
    paths.sort();

    for path in paths {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let file = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Blank out a leading `pub` (same length, so offsets survive) so the
        // signature scanner sees an ordinary Stan header.
        let scannable: String = text
            .split_inclusive('\n')
            .map(|line| {
                let indent = line.len() - line.trim_start().len();
                if line[indent..].starts_with("pub ") {
                    format!("{}    {}", &line[..indent], &line[indent + 4..])
                } else {
                    line.to_string()
                }
            })
            .collect();
        let documented: Vec<(usize, usize)> = extract_signatures(&scannable)
            .iter()
            .filter(|s| s.doc.is_some())
            .map(|s| (s.item_offset, s.header_offset))
            .collect();

        let lines: Vec<&str> = text.split_inclusive('\n').collect();
        let mut offset = 0;
        for (index, line) in lines.iter().enumerate() {
            let is_marker = line
                .trim()
                .strip_prefix("//")
                .is_some_and(|rest| rest.trim() == "@laplace");
            if is_marker {
                let attached_to_function = documented
                    .iter()
                    .any(|&(start, header)| (start..header).contains(&offset));
                let next_code = lines[index + 1..]
                    .iter()
                    .map(|l| l.trim())
                    .find(|l| !l.is_empty() && !l.starts_with("//"));
                let attached_to_construct = next_code.is_some_and(|l| {
                    let l = l.strip_prefix("pub ").unwrap_or(l).trim_start();
                    l.starts_with("@template") || l.starts_with("@macro")
                });
                if !attached_to_function && !attached_to_construct {
                    found.push((file.clone(), index + 1));
                }
            }
            offset += line.len();
        }
    }
    found
}

/// Run git in `cwd`, returning trimmed stdout.
fn git(cwd: &Path, args: &[&str]) -> Result<String, ReleaseError> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(ReleaseError::GitSpawn)?;
    if !out.status.success() {
        return Err(ReleaseError::Git {
            args: args.join(" "),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap()
    }

    #[test]
    fn bumps_follow_semver() {
        assert_eq!(bumped(&v("1.2.3"), "patch").unwrap(), v("1.2.4"));
        assert_eq!(bumped(&v("1.2.3"), "minor").unwrap(), v("1.3.0"));
        assert_eq!(bumped(&v("1.2.3"), "major").unwrap(), v("2.0.0"));
        assert_eq!(bumped(&v("1.2.3-rc.1"), "patch").unwrap(), v("1.2.3"));
        assert_eq!(bumped(&v("1.2.3"), "2.0.0").unwrap(), v("2.0.0"));
        assert_eq!(bumped(&v("1.2.3"), "v2.0.0").unwrap(), v("2.0.0"));
        assert!(matches!(
            bumped(&v("1.2.3"), "huge"),
            Err(ReleaseError::BadBump(_))
        ));
    }

    #[test]
    fn the_tag_prefix_follows_the_existing_majority() {
        let tags = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(tag_prefix(&tags(&[])).0, "");
        assert_eq!(tag_prefix(&tags(&["v0.1.0", "v0.2.0", "docs"])).0, "v");
        assert_eq!(tag_prefix(&tags(&["0.1.0", "0.2.0", "v0.0.1"])).0, "");
        assert!(tag_prefix(&tags(&["v0.1.0"])).1.contains("`v` prefix"));
    }

    #[test]
    fn a_doc_block_above_nothing_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("a.laplacelib"),
            "// @laplace\n// @brief Fine.\npub real f() {\n  return 1;\n}\n\n\
             // @laplace\n// @brief Orphan.\n\nreal g() {\n  return 1;\n}\n\n\
             // @laplace\n// @brief Template.\npub @template t(ident x) {\n}\n",
        )
        .unwrap();
        assert_eq!(
            unattached_doc_blocks(tmp.path()),
            vec![("a.laplacelib".to_string(), 7)]
        );
    }
}
