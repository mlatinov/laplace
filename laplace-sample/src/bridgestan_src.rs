//! The BridgeStan source tree (D9): pinned version, fetched on first use to
//! `~/.laplace/bridgestan/bridgestan-<version>/`, overridable with
//! `LAPLACE_BRIDGESTAN`.
//!
//! Fetched with `curl` and `tar` (the same approach as `laplace self-update`;
//! no HTTP or TLS crates) and verified against a pinned SHA-256 before it is
//! unpacked. The first model build additionally downloads the matching
//! `stanc` into the tree (BridgeStan's Makefile does that); set `STANC` to an
//! existing stanc to avoid it.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::sha256_hex;

/// Must equal the `bridgestan` crate version in Cargo.toml.
pub const BRIDGESTAN_VERSION: &str = "2.9.0";
pub const BRIDGESTAN_SHA256: &str =
    "b6034f83fb2ffb71ce7be9525b7dded1a225e9a4eaf4b75d4c7e97e3ade05a4c";

pub fn url() -> String {
    format!(
        "https://github.com/roualdes/bridgestan/releases/download/v{v}/bridgestan-{v}.tar.gz",
        v = BRIDGESTAN_VERSION
    )
}

pub fn home() -> anyhow::Result<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("cannot find the home directory (HOME is not set)"))
}

/// `~/.laplace`
pub fn laplace_home() -> anyhow::Result<PathBuf> {
    Ok(home()?.join(".laplace"))
}

/// Where the source tree is (or will be).
pub fn location() -> anyhow::Result<PathBuf> {
    if let Some(p) = std::env::var_os("LAPLACE_BRIDGESTAN") {
        return Ok(PathBuf::from(p));
    }
    Ok(laplace_home()?
        .join("bridgestan")
        .join(format!("bridgestan-{BRIDGESTAN_VERSION}")))
}

/// The source tree, downloading it if needed.
pub fn ensure() -> anyhow::Result<PathBuf> {
    let dir = location()?;
    if dir.join("Makefile").is_file() {
        return Ok(dir);
    }
    if std::env::var_os("LAPLACE_BRIDGESTAN").is_some() {
        anyhow::bail!(
            "LAPLACE_BRIDGESTAN={} is not a BridgeStan source tree (no Makefile)",
            dir.display()
        );
    }
    download_to(&dir)?;
    Ok(dir)
}

fn download_to(dir: &Path) -> anyhow::Result<()> {
    let parent = dir.parent().expect("versioned dir has a parent");
    std::fs::create_dir_all(parent)?;
    let staging = parent.join(format!(".download-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    let tarball = staging.join("bridgestan.tar.gz");
    eprintln!(
        "Downloading BridgeStan {BRIDGESTAN_VERSION} (one time) from {}",
        url()
    );
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--output",
        ])
        .arg(&tarball)
        .arg(url())
        .status()
        .map_err(|e| anyhow::anyhow!("could not run curl: {e}"))?;
    anyhow::ensure!(status.success(), "downloading BridgeStan failed ({status})");
    let got = sha256_hex(&std::fs::read(&tarball)?);
    if got != BRIDGESTAN_SHA256 {
        let _ = std::fs::remove_dir_all(&staging);
        anyhow::bail!(
            "BridgeStan download has sha256 {got}, expected {BRIDGESTAN_SHA256}; refusing to use it"
        );
    }
    let status = Command::new("tar")
        .arg("xzf")
        .arg(&tarball)
        .arg("-C")
        .arg(&staging)
        .status()
        .map_err(|e| anyhow::anyhow!("could not run tar: {e}"))?;
    anyhow::ensure!(status.success(), "unpacking BridgeStan failed ({status})");
    let unpacked = staging.join(format!("bridgestan-{BRIDGESTAN_VERSION}"));
    anyhow::ensure!(
        unpacked.join("Makefile").is_file(),
        "unexpected BridgeStan archive layout"
    );
    if let Err(e) = std::fs::rename(&unpacked, dir) {
        // Another process may have finished first; theirs is as good as ours.
        if !dir.join("Makefile").is_file() {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e.into());
        }
    }
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_and_source_versions_agree() {
        // The bridgestan crate's FFI matches exactly one release.
        assert_eq!(bridgestan::VERSION, super::BRIDGESTAN_VERSION);
    }
}
