//! The running compiler's own version and build provenance, embedded by
//! `build.rs`.
//!
//! "Compiler release" here means the `laplace` binary itself. It is unrelated
//! to a *package* release, which is a tag on a Stan library repository.

/// The Cargo version, e.g. `0.2.0`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `0.2.0 (e8b84c9)`, or plain `0.2.0` when built outside a git checkout.
pub const LONG_VERSION: &str = env!("LAPLACE_VERSION_LONG");

/// Build date, `YYYY-MM-DD` UTC (honours `SOURCE_DATE_EPOCH`).
pub const BUILD_DATE: &str = env!("LAPLACE_BUILD_DATE");

/// The target triple the binary was compiled for, e.g.
/// `x86_64-unknown-linux-gnu`. `self-update` picks its download by it.
pub const TARGET: &str = env!("LAPLACE_TARGET");

/// The short git commit the binary was built from, if it was built from a
/// checkout.
pub fn commit() -> Option<&'static str> {
    match env!("LAPLACE_GIT_COMMIT") {
        "" => None,
        commit => Some(commit),
    }
}

/// [`VERSION`] as a parsed semver version.
pub fn current() -> semver::Version {
    semver::Version::parse(VERSION).expect("Cargo.toml version is valid semver")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_long_version_starts_with_the_cargo_version() {
        assert!(LONG_VERSION.starts_with(VERSION));
        if let Some(commit) = commit() {
            assert_eq!(LONG_VERSION, format!("{VERSION} ({commit})"));
        }
    }

    #[test]
    fn the_build_date_is_an_iso_date() {
        assert_eq!(BUILD_DATE.len(), 10, "{BUILD_DATE}");
        assert_eq!(&BUILD_DATE[4..5], "-");
        assert_eq!(&BUILD_DATE[7..8], "-");
    }
}
