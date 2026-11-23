//! Library-author commands (Phase 4.3): `laplace sampler doc`, which
//! `laplace doc pkg::kernel` forwards to.

use std::path::Path;

use laplace::package::{KERNELS_DIR, LAPKERNEL_EXTENSION};

use crate::lapkernel::doc::{kernel_doc, render};
use crate::lapkernel::kernel::LapKernelFile;
use crate::libraries::{display_path, resolve_import, KernelPackage, Project};
use crate::runner::BuiltIn;

fn load_library_kernel(lib: &KernelPackage, name: &str) -> anyhow::Result<LapKernelFile> {
    let file = lib.kernel_file(name).map_err(anyhow::Error::msg)?;
    let src = std::fs::read_to_string(&file)?;
    LapKernelFile::from_source(&display_path(&lib.name, name), src)
}

/// `pkg::kernel` (an installed kernel package of the current project) or
/// a `.lapkernel` path.
pub fn doc(spec: &str, project: &Project) -> anyhow::Result<String> {
    if let Some((package, name)) = spec.split_once("::") {
        let lib = resolve_import(project, package).map_err(anyhow::Error::msg)?;
        let file = load_library_kernel(&lib, name)?;
        let origin = format!(
            "{} {}, {KERNELS_DIR}/{name}.{LAPKERNEL_EXTENSION}",
            lib.name, lib.version
        );
        return Ok(render(&kernel_doc(&file, spec, &origin)));
    }
    if spec.ends_with(&format!(".{LAPKERNEL_EXTENSION}")) {
        let file = LapKernelFile::load(Path::new(spec))?;
        return Ok(render(&kernel_doc(&file, spec, "local file")));
    }
    if let Ok(b) = BuiltIn::parse(spec) {
        return Ok(format!(
            "{spec} is a built-in kernel; `laplace sampler list` shows its settings.\n{}",
            builtin_line(b)
        ));
    }
    anyhow::bail!("expected `<package>::<kernel>` or a .lapkernel file, got `{spec}`")
}

fn builtin_line(b: BuiltIn) -> String {
    let (settings, adapts) = match b {
        BuiltIn::Nuts => (
            "step_size, target_accept, max_depth",
            "adapts itself (Stan's windowed adaptation)",
        ),
        BuiltIn::Rwm => (
            "step_size",
            "adapt(rwm()) tunes step size and metric (target 0.234)",
        ),
        BuiltIn::Mala => (
            "step_size",
            "adapt(mala()) tunes step size and metric (target 0.574)",
        ),
        BuiltIn::Hmc => (
            "step_size, steps, jitter",
            "adapt(hmc()) tunes step size and metric (target 0.8)",
        ),
    };
    format!("  {:<6} settings: {settings}; {adapts}", b.name())
}
