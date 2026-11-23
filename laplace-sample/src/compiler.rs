//! Which C++ compiler builds the BridgeStan model library.
//!
//! First match wins:
//! 1. `LAPLACE_CXX`, then `CXX`, from the environment;
//! 2. the `CXX` set in the detected CmdStan's `make/local` (CmdStan is found
//!    through `CMDSTAN`, else the newest `~/.cmdstan/cmdstan-<version>`, the
//!    places cmdstanr installs to and reads from);
//! 3. otherwise none is passed and `make` uses its default: BridgeStan's own
//!    `make/local` if it sets `CXX`, else the system default (`g++` on most
//!    Linux systems).
//!
//! Why step 2 exists: so laplace-sample and CmdStan build a model with the
//! same compiler, and timings compare like with like. It is not a speed
//! rule. The compiler changes gradient time a lot, in either direction,
//! depending on the model: clang++ against g++ was 1.37x faster on the gp
//! model but 1.73x slower on survival and 1.23x slower on ts (Phase 1
//! report, "Follow-up"). Set `LAPLACE_CXX` to choose. Only `CXX` is taken
//! from CmdStan's `make/local`, nothing else.
//!
//! The choice is printed on every run and is part of the build cache key.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    LaplaceCxx,
    Cxx,
    /// The `make/local` file it came from.
    CmdStan(PathBuf),
    /// Nothing passed; `make` decides.
    MakeDefault,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiler {
    /// Passed to `make` as `CXX=<cxx>`; `None` leaves it to `make`.
    pub cxx: Option<String>,
    pub source: Source,
}

impl Compiler {
    pub fn describe(&self) -> String {
        match (&self.cxx, &self.source) {
            (Some(c), Source::LaplaceCxx) => format!("{c} (from LAPLACE_CXX)"),
            (Some(c), Source::Cxx) => format!("{c} (from CXX)"),
            (Some(c), Source::CmdStan(p)) => format!("{c} (from CmdStan's {})", p.display()),
            _ => "make's default (BridgeStan's make/local if it sets CXX, else the system compiler); \
                  set LAPLACE_CXX to choose"
                .to_string(),
        }
    }

    /// What the cache key records.
    pub fn key(&self) -> String {
        self.cxx
            .clone()
            .unwrap_or_else(|| "<make default>".to_string())
    }
}

/// The `CXX` a make fragment assigns, if any (last assignment wins;
/// `CXX=`, `CXX =`, `CXX :=`, `CXX ?=` and `override CXX = …`).
pub fn make_local_cxx(text: &str) -> Option<String> {
    let mut found = None;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let line = line.strip_prefix("override").map(str::trim).unwrap_or(line);
        let Some(rest) = line.strip_prefix("CXX") else {
            continue;
        };
        let rest = rest.trim_start();
        let value = rest
            .strip_prefix(":=")
            .or_else(|| rest.strip_prefix("?="))
            .or_else(|| rest.strip_prefix('='));
        if let Some(v) = value {
            let v = v.trim();
            if !v.is_empty() {
                found = Some(v.to_string());
            }
        }
    }
    found
}

/// `(major, minor, patch)` of a `cmdstan-X.Y.Z` directory name.
fn cmdstan_version(name: &str) -> Option<(u64, u64, u64)> {
    let v = name.strip_prefix("cmdstan-")?;
    let mut it = v.split('.').map(|p| p.parse::<u64>().ok());
    Some((it.next()??, it.next()??, it.next().flatten().unwrap_or(0)))
}

/// The CmdStan installation cmdstanr would use: `CMDSTAN`, else the newest
/// `<home>/.cmdstan/cmdstan-<version>`.
pub fn detect_cmdstan(cmdstan_env: Option<String>, home: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = cmdstan_env.filter(|p| !p.is_empty()) {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let root = home?.join(".cmdstan");
    std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let v = cmdstan_version(&e.file_name().to_string_lossy())?;
            Some((v, e.path()))
        })
        .max_by_key(|(v, _)| *v)
        .map(|(_, p)| p)
}

/// The selection rule with its inputs injected (for tests).
pub fn select_with(env: impl Fn(&str) -> Option<String>, cmdstan: Option<PathBuf>) -> Compiler {
    let set = |k: &str| env(k).filter(|v| !v.trim().is_empty());
    if let Some(c) = set("LAPLACE_CXX") {
        return Compiler {
            cxx: Some(c),
            source: Source::LaplaceCxx,
        };
    }
    if let Some(c) = set("CXX") {
        return Compiler {
            cxx: Some(c),
            source: Source::Cxx,
        };
    }
    if let Some(dir) = cmdstan {
        let local = dir.join("make").join("local");
        if let Some(c) = std::fs::read_to_string(&local)
            .ok()
            .and_then(|t| make_local_cxx(&t))
        {
            return Compiler {
                cxx: Some(c),
                source: Source::CmdStan(local),
            };
        }
    }
    Compiler {
        cxx: None,
        source: Source::MakeDefault,
    }
}

/// The compiler for this process's environment.
pub fn select() -> Compiler {
    let home = crate::bridgestan_src::home().ok();
    let cmdstan = detect_cmdstan(std::env::var("CMDSTAN").ok(), home.as_deref());
    select_with(|k| std::env::var(k).ok(), cmdstan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    fn cmdstan_with_local(text: &str) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("cmdstan-2.39.0");
        std::fs::create_dir_all(dir.join("make")).unwrap();
        std::fs::write(dir.join("make").join("local"), text).unwrap();
        (tmp, dir)
    }

    #[test]
    fn precedence_laplace_cxx_then_cxx_then_cmdstan_then_default() {
        let (_t, cs) = cmdstan_with_local("CXX=clang++\n");
        let both = env(&[("LAPLACE_CXX", "g++-14"), ("CXX", "icpx")]);
        assert_eq!(
            select_with(&both, Some(cs.clone())).cxx.as_deref(),
            Some("g++-14")
        );
        let cxx = env(&[("CXX", "icpx")]);
        let c = select_with(cxx, Some(cs.clone()));
        assert_eq!((c.cxx.as_deref(), c.source), (Some("icpx"), Source::Cxx));
        let c = select_with(env(&[]), Some(cs.clone()));
        assert_eq!(c.cxx.as_deref(), Some("clang++"));
        assert_eq!(c.source, Source::CmdStan(cs.join("make").join("local")));
        let c = select_with(env(&[]), None);
        assert_eq!((c.cxx, c.source), (None, Source::MakeDefault));
    }

    #[test]
    fn empty_variables_do_not_count() {
        let c = select_with(env(&[("LAPLACE_CXX", " "), ("CXX", "")]), None);
        assert_eq!(c.source, Source::MakeDefault);
    }

    #[test]
    fn cmdstan_without_cxx_in_make_local_falls_through() {
        let (_t, cs) = cmdstan_with_local("STAN_THREADS=true\n");
        assert_eq!(select_with(env(&[]), Some(cs)).source, Source::MakeDefault);
    }

    #[test]
    fn make_local_forms() {
        assert_eq!(make_local_cxx("CXX=clang++"), Some("clang++".into()));
        assert_eq!(
            make_local_cxx("CXX = clang++-18  # fast"),
            Some("clang++-18".into())
        );
        assert_eq!(
            make_local_cxx("CXX := g++\nCXX ?= clang++"),
            Some("clang++".into())
        );
        assert_eq!(make_local_cxx("override CXX = icpx"), Some("icpx".into()));
        assert_eq!(make_local_cxx("# CXX=clang++\nCXXFLAGS=-O2"), None);
        assert_eq!(make_local_cxx("CXX_TYPE=gcc"), None);
    }

    #[test]
    fn newest_cmdstan_is_detected_and_cmdstan_env_wins() {
        let tmp = tempfile::tempdir().unwrap();
        for v in [
            "cmdstan-2.9.0",
            "cmdstan-2.38.0",
            "cmdstan-2.39.1",
            "cmdstan-notes",
        ] {
            std::fs::create_dir_all(tmp.path().join(".cmdstan").join(v)).unwrap();
        }
        assert_eq!(
            detect_cmdstan(None, Some(tmp.path())),
            Some(tmp.path().join(".cmdstan").join("cmdstan-2.39.1"))
        );
        let other = tmp.path().join(".cmdstan").join("cmdstan-2.38.0");
        assert_eq!(
            detect_cmdstan(Some(other.to_string_lossy().into()), Some(tmp.path())),
            Some(other)
        );
        assert_eq!(
            detect_cmdstan(None, Some(&tmp.path().join("nowhere"))),
            None
        );
    }
}
