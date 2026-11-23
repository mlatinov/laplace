//! Turning a `.laplace` or `.stan` file into a loaded BridgeStan library.
//!
//! - `.laplace`: `laplace build <file>` runs as a subprocess in the file's
//!   directory, so the compiled `.stan` lands where a normal build puts it
//!   (`build/<stem>.stan`) and stays on disk as the inspectable artifact.
//! - `.stan`: used as is.
//!
//! Compiled libraries are cached under `~/.laplace/cache/bridgestan/<key>/`
//! (override: `LAPLACE_SAMPLE_CACHE`). The key is a SHA-256 over the `.stan`
//! text, the text of every file it `#include`s (recursively), the BridgeStan
//! version, the build flags, and the toolchain (the compiler chosen by
//! [`crate::compiler`], `CXXFLAGS` from the environment, BridgeStan's
//! `make/local`), so a one-character change anywhere rebuilds and an
//! unchanged model never does. The toolchain is part of the key because it
//! matters: the compiler changes gradient time by up to 1.7x on this
//! project's library models (Phase 1 report).

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::bridgestan_src::{self, BRIDGESTAN_VERSION};
use crate::compiler::{self, Compiler};

/// Build flags that are part of the cache key. `STAN_THREADS=true` is set
/// by `bridgestan::compile_model` itself and enforced when a model loads.
pub const BUILD_FLAGS: &str = "STAN_THREADS=true";

/// Bumped when the cache layout or key recipe changes.
const CACHE_FORMAT: &str = "laplace-sample model cache v2";

#[derive(Clone, Debug)]
pub struct Prepared {
    /// The `.stan` file that was compiled (for a `.laplace` input, the
    /// `laplace build` output).
    pub stan: PathBuf,
    /// `<stem>`; Stan names the model `<stem>_model`.
    pub stem: String,
    pub library: PathBuf,
    pub cache_key: String,
    /// True when the library came from the cache without compiling.
    pub cache_hit: bool,
}

pub fn cache_root() -> anyhow::Result<PathBuf> {
    if let Some(p) = std::env::var_os("LAPLACE_SAMPLE_CACHE") {
        return Ok(PathBuf::from(p));
    }
    Ok(bridgestan_src::laplace_home()?
        .join("cache")
        .join("bridgestan"))
}

/// `.laplace` -> run `laplace build`; `.stan` -> itself.
pub fn stan_file_for(input: &Path) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(input.is_file(), "{} does not exist", input.display());
    match input.extension().and_then(|e| e.to_str()) {
        Some("stan") => Ok(input.to_path_buf()),
        Some("laplace") => laplace_build(input),
        _ => anyhow::bail!("{}: expected a .laplace or .stan file", input.display()),
    }
}

fn laplace_build(input: &Path) -> anyhow::Result<PathBuf> {
    let input = input.canonicalize()?;
    let dir = input.parent().expect("a file has a parent");
    let name = input.file_name().expect("a file has a name");
    let stem = input
        .file_stem()
        .expect("a file has a stem")
        .to_string_lossy();
    let laplace = std::env::var_os("LAPLACE_BIN").unwrap_or_else(|| "laplace".into());
    let out = Command::new(&laplace)
        .arg("build")
        .arg(name)
        .current_dir(dir)
        .output()
        .map_err(|e| {
            anyhow::anyhow!(
                "could not run `{} build` ({e}); is the laplace compiler on PATH?",
                laplace.to_string_lossy()
            )
        })?;
    if !out.status.success() {
        anyhow::bail!(
            "`laplace build {}` failed:\n{}{}",
            name.to_string_lossy(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let stan = dir.join("build").join(format!("{stem}.stan"));
    anyhow::ensure!(
        stan.is_file(),
        "`laplace build` did not produce {}",
        stan.display()
    );
    Ok(stan)
}

/// The file named by a `#include` line, if the line is one.
fn include_target(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#include")?;
    let rest = rest.trim();
    let rest = rest
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .or_else(|| rest.strip_prefix('<').and_then(|r| r.strip_suffix('>')))
        .unwrap_or(rest);
    (!rest.is_empty()).then_some(rest)
}

/// `(path as written, contents)` for the file and everything it includes,
/// depth first in source order, each file once. Includes resolve against
/// the directory of the top-level `.stan` file, as `stanc` does with
/// `--include-paths=<that dir>`.
pub fn sources(stan: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let root_dir = stan.parent().unwrap_or(Path::new("."));
    let mut out = vec![];
    let mut seen = std::collections::HashSet::new();
    fn walk(
        path: &Path,
        label: String,
        root_dir: &Path,
        out: &mut Vec<(String, String)>,
        seen: &mut std::collections::HashSet<PathBuf>,
    ) -> anyhow::Result<()> {
        let canon = path
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        if !seen.insert(canon) {
            return Ok(());
        }
        let text = std::fs::read_to_string(path)?;
        let includes: Vec<String> = text
            .lines()
            .filter_map(include_target)
            .map(str::to_owned)
            .collect();
        out.push((label, text));
        for inc in includes {
            walk(&root_dir.join(&inc), inc, root_dir, out, seen)?;
        }
        Ok(())
    }
    let label = stan.file_name().unwrap().to_string_lossy().into_owned();
    walk(stan, label, root_dir, &mut out, &mut seen)?;
    Ok(out)
}

/// The toolchain settings that shape the build: the chosen compiler,
/// `CXXFLAGS`, and the BridgeStan tree's `make/local` (empty strings when
/// unset or absent).
pub fn toolchain(compiler: &Compiler) -> Vec<String> {
    let local = bridgestan_src::location()
        .ok()
        .and_then(|d| std::fs::read_to_string(d.join("make").join("local")).ok())
        .unwrap_or_default();
    vec![
        compiler.key(),
        std::env::var("CXXFLAGS").unwrap_or_default(),
        local,
    ]
}

/// The cache key for a `.stan` file under the current toolchain.
pub fn cache_key(stan: &Path) -> anyhow::Result<String> {
    cache_key_with(stan, &toolchain(&compiler::select()))
}

/// The cache key for a `.stan` file (see the module docs).
pub fn cache_key_with(stan: &Path, toolchain: &[String]) -> anyhow::Result<String> {
    let mut h = Sha256::new();
    let mut field = |s: &str| {
        h.update((s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    };
    field(CACHE_FORMAT);
    field(BRIDGESTAN_VERSION);
    field(BUILD_FLAGS);
    for t in toolchain {
        field(t);
    }
    for (name, text) in sources(stan)? {
        field(&name);
        field(&text);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Find or build the library for `stan` with `compile`, which is given the
/// staged `.stan` (inside the cache) and the original source directory, and
/// returns the path of the built library. Separated from [`prepare`] so the
/// caching rules are testable without a C++ toolchain.
pub fn prepare_with(
    stan: &Path,
    cache: &Path,
    toolchain: &[String],
    compile: impl FnOnce(&Path, &Path) -> anyhow::Result<PathBuf>,
) -> anyhow::Result<Prepared> {
    let stem = stan
        .file_stem()
        .ok_or_else(|| anyhow::anyhow!("{}: no file name", stan.display()))?
        .to_string_lossy()
        .into_owned();
    let key = cache_key_with(stan, toolchain)?;
    let dir = cache.join(&key);
    let library = dir.join(format!("{stem}_model.so"));
    if library.is_file() {
        return Ok(Prepared {
            stan: stan.to_path_buf(),
            stem,
            library,
            cache_key: key,
            cache_hit: true,
        });
    }
    std::fs::create_dir_all(cache)?;
    let staging = cache.join(format!(".build-{key}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    let staged = staging.join(format!("{stem}.stan"));
    std::fs::copy(stan, &staged)?;
    // A bare file name's parent is the *empty* path, not None.
    let src_dir = match stan.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let built = match compile(&staged, &src_dir.canonicalize()?) {
        Ok(p) => p,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
    };
    anyhow::ensure!(
        built == staging.join(format!("{stem}_model.so")),
        "compiler wrote {} instead of the expected library",
        built.display()
    );
    if std::fs::rename(&staging, &dir).is_err() {
        // A concurrent build of the same key won; use its library.
        let _ = std::fs::remove_dir_all(&staging);
        anyhow::ensure!(
            library.is_file(),
            "could not place the compiled model in {}",
            dir.display()
        );
    }
    Ok(Prepared {
        stan: stan.to_path_buf(),
        stem,
        library,
        cache_key: key,
        cache_hit: false,
    })
}

/// Find or build the BridgeStan library for a `.stan` file.
pub fn prepare(stan: &Path) -> anyhow::Result<Prepared> {
    let cache = cache_root()?;
    let compiler = compiler::select();
    eprintln!("Compiler: {}", compiler.describe());
    prepare_with(stan, &cache, &toolchain(&compiler), |staged, src_dir| {
        let bs = bridgestan_src::ensure()?;
        eprintln!(
            "Compiling {} with BridgeStan {BRIDGESTAN_VERSION} (cached afterwards)",
            stan.display()
        );
        let include = format!("--include-paths={}", src_dir.display());
        let jobs = format!(
            "-j{}",
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        );
        let mut make_args = vec![jobs];
        if let Some(cxx) = &compiler.cxx {
            make_args.push(format!("CXX={cxx}"));
        }
        let make_args: Vec<&str> = make_args.iter().map(String::as_str).collect();
        Ok(bridgestan::compile_model(
            &bs,
            staged,
            &[include.as_str()],
            &make_args,
        )?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn fake_compile(
        calls: &Cell<u32>,
    ) -> impl FnOnce(&Path, &Path) -> anyhow::Result<PathBuf> + '_ {
        move |staged, _src| {
            calls.set(calls.get() + 1);
            let so = staged.with_file_name(format!(
                "{}_model.so",
                staged.file_stem().unwrap().to_string_lossy()
            ));
            std::fs::write(&so, b"fake")?;
            Ok(so)
        }
    }

    #[test]
    fn second_run_hits_and_one_character_change_rebuilds() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let stan = tmp.path().join("m.stan");
        std::fs::write(&stan, "parameters { real x; } model { x ~ normal(0, 1); }").unwrap();
        let calls = Cell::new(0);
        let a = prepare_with(&stan, &cache, &[], fake_compile(&calls)).unwrap();
        assert!(!a.cache_hit);
        let b = prepare_with(&stan, &cache, &[], fake_compile(&calls)).unwrap();
        assert!(b.cache_hit);
        assert_eq!(a.library, b.library);
        assert_eq!(calls.get(), 1);
        std::fs::write(&stan, "parameters { real x; } model { x ~ normal(0, 2); }").unwrap();
        let c = prepare_with(&stan, &cache, &[], fake_compile(&calls)).unwrap();
        assert!(!c.cache_hit);
        assert_ne!(c.cache_key, a.cache_key);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn changing_an_included_file_changes_the_key() {
        let tmp = tempfile::tempdir().unwrap();
        let stan = tmp.path().join("m.stan");
        std::fs::write(&stan, "functions {\n#include pkg.stanfunctions\n}\n").unwrap();
        std::fs::write(
            tmp.path().join("pkg.stanfunctions"),
            "real f(real x) { return x; }",
        )
        .unwrap();
        let k1 = cache_key(&stan).unwrap();
        std::fs::write(
            tmp.path().join("pkg.stanfunctions"),
            "real f(real x) { return 2*x; }",
        )
        .unwrap();
        assert_ne!(k1, cache_key(&stan).unwrap());
    }

    #[test]
    fn the_toolchain_is_part_of_the_key() {
        let tmp = tempfile::tempdir().unwrap();
        let stan = tmp.path().join("m.stan");
        std::fs::write(&stan, "model {}").unwrap();
        let gcc = vec!["g++".to_string(), String::new(), String::new()];
        let clang = vec!["clang++".to_string(), String::new(), String::new()];
        let local = vec![String::new(), String::new(), "CXX=clang++\n".to_string()];
        let k = |t: &Vec<String>| cache_key_with(&stan, t).unwrap();
        assert_ne!(k(&gcc), k(&clang));
        assert_ne!(k(&gcc), k(&local));
        assert_eq!(k(&gcc), k(&gcc));
    }

    #[test]
    fn a_bare_relative_file_name_builds() {
        // `laplace sampler check x.lapsampler --model m.stan` from m.stan's
        // own directory: the parent of "m.stan" is the empty path.
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        std::fs::write(tmp.path().join("m.stan"), "model {}").unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let calls = Cell::new(0);
        let r = prepare_with(Path::new("m.stan"), &cache, &[], fake_compile(&calls));
        std::env::set_current_dir(cwd).unwrap();
        r.unwrap();
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn a_missing_include_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let stan = tmp.path().join("m.stan");
        std::fs::write(&stan, "#include \"nope.stan\"\n").unwrap();
        assert!(cache_key(&stan).is_err());
    }

    #[test]
    fn include_forms() {
        assert_eq!(include_target("#include a.stan"), Some("a.stan"));
        assert_eq!(include_target("  #include \"b/c.stan\""), Some("b/c.stan"));
        assert_eq!(include_target("#include <d.stan>"), Some("d.stan"));
        assert_eq!(include_target("// #include x"), None);
    }

    #[test]
    fn a_failed_build_leaves_no_cache_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let stan = tmp.path().join("m.stan");
        std::fs::write(&stan, "model {}").unwrap();
        let r = prepare_with(&stan, &cache, &[], |_, _| anyhow::bail!("stanc error"));
        assert!(r.is_err());
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 0);
    }

    #[test]
    fn same_text_in_two_places_shares_one_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        for d in ["a", "b"] {
            std::fs::create_dir(tmp.path().join(d)).unwrap();
            std::fs::write(tmp.path().join(d).join("m.stan"), "model {}").unwrap();
        }
        let calls = Cell::new(0);
        prepare_with(
            &tmp.path().join("a/m.stan"),
            &cache,
            &[],
            fake_compile(&calls),
        )
        .unwrap();
        let b = prepare_with(
            &tmp.path().join("b/m.stan"),
            &cache,
            &[],
            fake_compile(&calls),
        )
        .unwrap();
        assert!(b.cache_hit);
        assert_eq!(calls.get(), 1);
    }
}
