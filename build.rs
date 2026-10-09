//! Embed build provenance into the binary: the short git commit and the
//! build date, for `laplace --version` and `laplace version --verbose`.
//!
//! Both fall back gracefully. Outside a git checkout (a crates.io tarball, a
//! distro source package) there is no commit, and `--version` prints the
//! plain Cargo version. `SOURCE_DATE_EPOCH` overrides the build date, so a
//! reproducible-build packager gets a reproducible binary.

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let commit = git_short_commit();
    let date = build_date();
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());

    println!(
        "cargo:rustc-env=LAPLACE_GIT_COMMIT={}",
        commit.as_deref().unwrap_or("")
    );
    println!("cargo:rustc-env=LAPLACE_BUILD_DATE={date}");
    println!("cargo:rustc-env=LAPLACE_TARGET={target}");

    let version = env!("CARGO_PKG_VERSION");
    let long = match &commit {
        Some(commit) => format!("{version} ({commit})"),
        None => version.to_string(),
    };
    println!("cargo:rustc-env=LAPLACE_VERSION_LONG={long}");

    // Re-run when HEAD moves (a commit, a checkout) rather than on every
    // build. Both paths may be missing outside a checkout; cargo treats a
    // missing path as "always changed" only if it later appears.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
}

fn git_short_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!commit.is_empty()).then_some(commit)
}

/// `YYYY-MM-DD`, UTC, from `SOURCE_DATE_EPOCH` or the clock.
fn build_date() -> String {
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days since 1970-01-01 to a proleptic Gregorian date (Howard Hinnant's
/// `civil_from_days`), so the build script needs no date crate.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}
