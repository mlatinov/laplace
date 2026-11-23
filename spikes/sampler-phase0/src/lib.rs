//! Phase 0 spike helpers. Never imported by the real crate (ground rule 5).

use anyhow::{Context, Result};
use bridgestan::{Model, StanLibrary};
use std::ffi::CString;
use std::path::{Path, PathBuf};

/// Pinned BridgeStan version. Must equal the `bridgestan` crate version.
pub const BRIDGESTAN_VERSION: &str = "2.9.0";

/// D9: `$LAPLACE_BRIDGESTAN` if set, else `~/.laplace/bridgestan/bridgestan-<ver>`.
pub fn bridgestan_src() -> PathBuf {
    if let Ok(p) = std::env::var("LAPLACE_BRIDGESTAN") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    PathBuf::from(home)
        .join(".laplace/bridgestan")
        .join(format!("bridgestan-{BRIDGESTAN_VERSION}"))
}

pub fn spike_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn model_path(name: &str) -> PathBuf {
    spike_dir().join("models").join(format!("{name}.stan"))
}

/// Compile `<name>.stan` to `<name>_model.so` unless the .so is newer.
/// `bridgestan::compile_model` sets STAN_THREADS=true itself.
pub fn compile(stan: &Path) -> Result<PathBuf> {
    let so = stan.with_file_name(format!(
        "{}_model.so",
        stan.file_stem().unwrap().to_string_lossy()
    ));
    let fresh = so.exists()
        && so.metadata()?.modified()? >= stan.metadata()?.modified()?;
    if fresh {
        return Ok(so);
    }
    let out = bridgestan::compile_model(&bridgestan_src(), stan, &[], &["-j8"])
        .with_context(|| format!("compiling {}", stan.display()))?;
    Ok(out)
}

pub fn library(name: &str) -> Result<StanLibrary> {
    let so = compile(&model_path(name))?;
    Ok(bridgestan::open_library(so)?)
}

/// Data: `models/<name>.json` if it exists, else none.
pub fn model<'a>(lib: &'a StanLibrary, name: &str, seed: u32) -> Result<Model<&'a StanLibrary>> {
    let json = spike_dir().join("models").join(format!("{name}.json"));
    let data = if json.exists() {
        Some(CString::new(json.to_string_lossy().as_bytes())?)
    } else {
        None
    };
    Ok(Model::new(lib, data, seed)?)
}

pub fn names(s: &str) -> Vec<String> {
    if s.is_empty() {
        vec![]
    } else {
        s.split(',').map(str::to_owned).collect()
    }
}
