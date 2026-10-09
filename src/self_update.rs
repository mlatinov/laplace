//! `laplace self-update`: replace the running *compiler* with a newer
//! release. (A *package* release is `laplace release`, unrelated.)
//!
//! The only place laplace talks to the network on its own initiative, and
//! only when asked. Release metadata comes from the GitHub Releases API (the
//! base URL is overridable with `LAPLACE_RELEASES_URL`, which is how the
//! tests point it at a local fake server). Downloads go through `curl`, so
//! the binary carries no HTTP or TLS stack of its own.
//!
//! The release assets this expects are what `.github/workflows/release.yml`
//! publishes: `laplace-<target>.tar.gz` holding the binary, and
//! `laplace-<target>.tar.gz.sha256` holding its checksum.
//!
//! Replacement is atomic: the new binary is checksum-verified, extracted,
//! smoke-tested with `--version`, written *beside* the old one, and renamed
//! over it. Every failure before that rename leaves the old binary exactly
//! as it was.

use std::path::{Path, PathBuf};
use std::process::Command;

use semver::Version;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Where releases are published.
pub const DEFAULT_RELEASES_URL: &str = "https://api.github.com/repos/mlatinov/laplace";

/// The repository, for `cargo install --git`.
pub const REPO_URL: &str = "https://github.com/mlatinov/laplace";

#[derive(Debug, Error)]
pub enum SelfUpdateError {
    #[error(
        "self-update needs `curl` on PATH to download releases ({0}) -- install curl, or \
         download the release by hand from {REPO_URL}/releases"
    )]
    NoCurl(#[source] std::io::Error),

    #[error("could not fetch {url}: {message}")]
    Fetch { url: String, message: String },

    #[error("the release metadata from {url} did not parse: {message}")]
    BadMetadata { url: String, message: String },

    #[error("release `{tag}` has no build for this platform ({target}); available: {available}")]
    NoAsset {
        tag: String,
        target: String,
        available: String,
    },

    #[error(
        "checksum mismatch for {asset}: the published checksum is {expected}, the download \
         hashes to {actual} -- nothing was changed"
    )]
    ChecksumMismatch {
        asset: String,
        expected: String,
        actual: String,
    },

    #[error("could not extract {asset}: {message} -- nothing was changed")]
    Extract { asset: String, message: String },

    #[error("the downloaded binary did not run (`--version` failed: {0}) -- nothing was changed")]
    SmokeTest(String),

    #[error(
        "laplace at {} was installed by {manager}, which owns that file -- update it the same \
         way:\n  {command}",
        .exe.display()
    )]
    PackageManaged {
        exe: PathBuf,
        manager: &'static str,
        command: String,
    },

    #[error("could not replace {}: {source} -- the old binary is unchanged", .path.display())]
    Replace {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("`{command}` failed")]
    CargoInstall { command: String },

    #[error("I/O error at {}: {source}", .path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// The bits of a GitHub release laplace needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Release {
    pub tag_name: String,
    /// Release notes (Markdown), shown as "what changed".
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
}

impl Release {
    /// The version the tag names (`v0.3.0` -> `0.3.0`), if it names one.
    pub fn version(&self) -> Option<Version> {
        let bare = self.tag_name.strip_prefix('v').unwrap_or(&self.tag_name);
        Version::parse(bare).ok()
    }

    fn asset(&self, name: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.name == name)
    }
}

/// The archive name for `target`.
pub fn asset_name(target: &str) -> String {
    format!("laplace-{target}.tar.gz")
}

/// Fetch the latest release, or the release tagged `v<version>`.
pub fn fetch_release(base_url: &str, version: Option<&str>) -> Result<Release, SelfUpdateError> {
    let url = match version {
        Some(v) => format!(
            "{}/releases/tags/v{}",
            base_url.trim_end_matches('/'),
            v.strip_prefix('v').unwrap_or(v)
        ),
        None => format!("{}/releases/latest", base_url.trim_end_matches('/')),
    };
    let bytes = curl(&url)?;
    serde_json::from_slice(&bytes).map_err(|e| SelfUpdateError::BadMetadata {
        url,
        message: e.to_string(),
    })
}

/// How the running binary was installed, which decides who may replace it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallMethod {
    /// A system package manager owns the file; laplace must not touch it.
    PackageManager {
        manager: &'static str,
        command: String,
    },
    /// `cargo install`: rebuild it the same way.
    Cargo { command: String },
    /// A downloaded binary somewhere the user put it: replace in place.
    Standalone,
}

/// Classify `exe` (already canonical). `cargo_home` is `$CARGO_HOME` or
/// `~/.cargo`; `has` reports whether a program exists, so the package
/// manager can be named precisely.
pub fn install_method(
    exe: &Path,
    cargo_home: Option<&Path>,
    has: impl Fn(&str) -> bool,
) -> InstallMethod {
    let text = exe.to_string_lossy();
    if text.contains("/Cellar/")
        || text.starts_with("/opt/homebrew/")
        || text.contains("/linuxbrew/")
    {
        return InstallMethod::PackageManager {
            manager: "Homebrew",
            command: "brew upgrade laplace".to_string(),
        };
    }
    if text.starts_with("/nix/store/") {
        return InstallMethod::PackageManager {
            manager: "Nix",
            command: "nix profile upgrade laplace (or update your Nix configuration)".to_string(),
        };
    }
    let system = ["/usr/", "/bin/", "/sbin/"]
        .iter()
        .any(|prefix| text.starts_with(prefix))
        && !text.starts_with("/usr/local/");
    if system {
        let (manager, command) = if has("pacman") {
            (
                "pacman",
                "yay -Syu laplace-bin   (or: paru -Syu laplace-bin; sudo pacman -Syu after \
                 updating the AUR package)",
            )
        } else if has("apt") {
            (
                "apt",
                "sudo apt update && sudo apt install --only-upgrade laplace",
            )
        } else if has("dnf") {
            ("dnf", "sudo dnf upgrade laplace")
        } else {
            (
                "the system package manager",
                "update the `laplace` package with your system's package manager",
            )
        };
        return InstallMethod::PackageManager {
            manager,
            command: command.to_string(),
        };
    }
    if let Some(cargo_home) = cargo_home {
        if exe.starts_with(cargo_home.join("bin")) {
            return InstallMethod::Cargo {
                command: format!("cargo install --locked --force --git {REPO_URL}"),
            };
        }
    }
    InstallMethod::Standalone
}

/// Download the release asset for `target`, verify it, and atomically
/// replace `exe` with the binary inside.
pub fn replace_binary(release: &Release, target: &str, exe: &Path) -> Result<(), SelfUpdateError> {
    let name = asset_name(target);
    let asset = release
        .asset(&name)
        .ok_or_else(|| SelfUpdateError::NoAsset {
            tag: release.tag_name.clone(),
            target: target.to_string(),
            available: release
                .assets
                .iter()
                .map(|a| a.name.as_str())
                .filter(|n| n.ends_with(".tar.gz"))
                .collect::<Vec<_>>()
                .join(", "),
        })?;
    let checksum_asset =
        release
            .asset(&format!("{name}.sha256"))
            .ok_or_else(|| SelfUpdateError::NoAsset {
                tag: release.tag_name.clone(),
                target: format!("{target} (checksum file {name}.sha256)"),
                available: String::new(),
            })?;

    let archive = curl(&asset.browser_download_url)?;
    let published = String::from_utf8_lossy(&curl(&checksum_asset.browser_download_url)?)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_lowercase();
    let actual = format!("{:x}", Sha256::digest(&archive));
    if published != actual {
        return Err(SelfUpdateError::ChecksumMismatch {
            asset: name,
            expected: published,
            actual,
        });
    }

    let scratch = tempfile::tempdir().map_err(|source| SelfUpdateError::Io {
        path: std::env::temp_dir(),
        source,
    })?;
    let archive_path = scratch.path().join(&name);
    std::fs::write(&archive_path, &archive).map_err(|source| SelfUpdateError::Io {
        path: archive_path.clone(),
        source,
    })?;
    let unpacked = scratch.path().join("unpacked");
    std::fs::create_dir_all(&unpacked).map_err(|source| SelfUpdateError::Io {
        path: unpacked.clone(),
        source,
    })?;
    let tar = Command::new("tar")
        .arg("-xzf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&unpacked)
        .output()
        .map_err(|e| SelfUpdateError::Extract {
            asset: name.clone(),
            message: format!("could not run tar: {e}"),
        })?;
    if !tar.status.success() {
        return Err(SelfUpdateError::Extract {
            asset: name,
            message: String::from_utf8_lossy(&tar.stderr).trim().to_string(),
        });
    }
    let binary_name = if cfg!(windows) {
        "laplace.exe"
    } else {
        "laplace"
    };
    let new_binary = find_file(&unpacked, binary_name).ok_or_else(|| SelfUpdateError::Extract {
        asset: name.clone(),
        message: format!("the archive holds no `{binary_name}`"),
    })?;
    make_executable(&new_binary)?;

    let smoke = Command::new(&new_binary).arg("--version").output();
    match smoke {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            return Err(SelfUpdateError::SmokeTest(
                String::from_utf8_lossy(&out.stderr).trim().to_string(),
            ))
        }
        Err(e) => return Err(SelfUpdateError::SmokeTest(e.to_string())),
    }

    // Stage beside the target so the final rename never crosses a
    // filesystem boundary, which is what makes it atomic.
    let dir = exe.parent().unwrap_or(Path::new("."));
    let staged = dir.join(format!(".laplace-update-{}", std::process::id()));
    let replace_err = |source| SelfUpdateError::Replace {
        path: exe.to_path_buf(),
        source,
    };
    std::fs::copy(&new_binary, &staged).map_err(replace_err)?;
    if let Err(e) = make_executable(&staged) {
        let _ = std::fs::remove_file(&staged);
        return Err(e);
    }
    if cfg!(windows) {
        // A running .exe cannot be overwritten, but it can be renamed.
        let old = exe.with_extension("old.exe");
        let _ = std::fs::remove_file(&old);
        std::fs::rename(exe, &old).map_err(replace_err)?;
    }
    if let Err(e) = std::fs::rename(&staged, exe) {
        let _ = std::fs::remove_file(&staged);
        return Err(replace_err(e));
    }
    Ok(())
}

/// Run `cargo install` for the cargo install method.
pub fn cargo_install(command: &str) -> Result<(), SelfUpdateError> {
    let mut parts = command.split_whitespace();
    let program = parts.next().unwrap_or("cargo");
    let status =
        Command::new(program)
            .args(parts)
            .status()
            .map_err(|_| SelfUpdateError::CargoInstall {
                command: command.to_string(),
            })?;
    if status.success() {
        Ok(())
    } else {
        Err(SelfUpdateError::CargoInstall {
            command: command.to_string(),
        })
    }
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in &entries {
        if path.is_file() && path.file_name().is_some_and(|n| n == name) {
            return Some(path.clone());
        }
    }
    entries
        .iter()
        .filter(|p| p.is_dir())
        .find_map(|p| find_file(p, name))
}

fn make_executable(path: &Path) -> Result<(), SelfUpdateError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).map_err(
            |source| SelfUpdateError::Io {
                path: path.to_path_buf(),
                source,
            },
        )?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// GET `url` with curl, following redirects (release assets redirect to a
/// CDN), failing on HTTP errors.
fn curl(url: &str) -> Result<Vec<u8>, SelfUpdateError> {
    let out = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--proto",
            "=https,http",
            "--header",
            "Accept: application/vnd.github+json",
            "--user-agent",
            concat!("laplace/", env!("CARGO_PKG_VERSION")),
            url,
        ])
        .output()
        .map_err(SelfUpdateError::NoCurl)?;
    if !out.status.success() {
        return Err(SelfUpdateError::Fetch {
            url: url.to_string(),
            message: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_none(_: &str) -> bool {
        false
    }

    #[test]
    fn system_paths_belong_to_the_package_manager() {
        let pacman = install_method(Path::new("/usr/bin/laplace"), None, |p| p == "pacman");
        match pacman {
            InstallMethod::PackageManager { manager, command } => {
                assert_eq!(manager, "pacman");
                assert!(command.contains("laplace-bin"), "{command}");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            install_method(Path::new("/usr/bin/laplace"), None, |p| p == "apt"),
            InstallMethod::PackageManager { manager: "apt", .. }
        ));
        assert!(matches!(
            install_method(
                Path::new("/opt/homebrew/Cellar/laplace/0.2.0/bin/laplace"),
                None,
                has_none
            ),
            InstallMethod::PackageManager {
                manager: "Homebrew",
                ..
            }
        ));
    }

    #[test]
    fn cargo_bin_rebuilds_with_cargo_install() {
        let method = install_method(
            Path::new("/home/u/.cargo/bin/laplace"),
            Some(Path::new("/home/u/.cargo")),
            has_none,
        );
        match method {
            InstallMethod::Cargo { command } => {
                assert!(command.starts_with("cargo install --locked --force --git"))
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn anything_else_is_replaced_in_place() {
        for path in ["/home/u/.local/bin/laplace", "/usr/local/bin/laplace"] {
            assert_eq!(
                install_method(Path::new(path), Some(Path::new("/home/u/.cargo")), has_none),
                InstallMethod::Standalone,
                "{path}"
            );
        }
    }

    #[test]
    fn a_release_tag_reads_with_or_without_v() {
        let release = |tag: &str| Release {
            tag_name: tag.to_string(),
            body: None,
            assets: Vec::new(),
        };
        assert_eq!(release("v0.3.0").version(), Version::parse("0.3.0").ok());
        assert_eq!(release("0.3.0").version(), Version::parse("0.3.0").ok());
        assert_eq!(release("nightly").version(), None);
    }
}
