//! Library-author commands (Phase 4.3-4.4): `laplace sampler doc`,
//! `laplace new kernel`, and `laplace sampler check` on a single
//! `.lapkernel`.

use std::path::{Path, PathBuf};

use laplace::manifest::{self, PackageKind, KERNEL_PACKAGES_REQUIREMENT};
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

/// `laplace sampler check kernels/x.lapkernel`: parse and check one kernel.
pub fn check_kernel(path: &Path) -> anyhow::Result<String> {
    let f = LapKernelFile::load(path)?;
    let c = f.check(None)?;
    let tags: Vec<String> = c
        .tuning
        .iter()
        .filter_map(|t| {
            t.adapt.map(|a| {
                let what = match a {
                    crate::lapkernel::check::AdaptKind::StepSize => "step size",
                    crate::lapkernel::check::AdaptKind::Metric => "metric",
                };
                format!("{} ({what})", t.name)
            })
        })
        .collect();
    Ok(format!(
        "ok: {} (kernel `{}`; {} tuning value(s){})",
        path.display(),
        f.name(),
        c.tuning.len(),
        if tags.is_empty() {
            String::new()
        } else {
            format!("; adapted: {}", tags.join(", "))
        }
    ))
}

/// What `laplace new kernel` scaffolds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    /// A Metropolis-Hastings kernel ending in `propose()`.
    Proposal,
    /// A direct kernel ending in `move()`.
    Direct,
}

fn kernel_skeleton(name: &str, kind: KernelKind) -> String {
    match kind {
        KernelKind::Proposal => format!(
            "//@laplace
//@brief {name}: a one-line description of the kernel
//@param eps Step size (adapted toward 0.234 acceptance)
//@param m Diagonal inverse metric (adapted in warmup)
//
// A Metropolis-Hastings kernel: the step ends in propose(q_new, c), and
// laplace-sample accepts or rejects q_new with log probability
// lp(q_new) - lp(q) + c, where c is the log proposal correction
// log g(q | q_new) - log g(q_new | q) (0 for a symmetric proposal).
// Replace the random walk below with your proposal.
kernel {name} {{
  tuning {{
    real<lower=0> eps = 1 @adapt(step_size, target_accept = 0.234);
    vector<lower=0>[dim] m = rep_vector(1, dim) @adapt(metric);
  }}
  step {{
    vector[dim] q_new = q + eps * sqrt(m) .* std_normal_rng(dim);
    propose(q_new, 0);
  }}
}}
"
        ),
        KernelKind::Direct => format!(
            "//@laplace
//@brief {name}: a one-line description of the kernel
//
// A direct kernel: the step ends in move(q_new), which is always taken.
// Nothing is accepted or rejected for you, so the step itself must leave
// the target invariant (an exact conditional draw, say, or a kernel that
// does its own accept/reject). This skeleton stays where it is, which is
// trivially invariant: replace it.
kernel {name} {{
  step {{
    move(q);
  }}
}}
"
        ),
    }
}

fn test_skeleton(name: &str) -> String {
    format!(
        "# How `laplace sampler test` runs `{name}` on the reference targets
# (standard normal, correlated Gaussian, eight schools, ...). Read in
# Phase 5; every key is optional.
kernel = \"{name}\"
# Wrap it in adapt() when testing (it has @adapt tags).
adapt = true

[defaults]
warmup = 1000
draws = 1000
thin = 1

# Per-target overrides, e.g. a slow-mixing kernel on the correlated target:
# [targets.corr_gauss_10]
# draws = 10000
# thin = 10
"
    )
}

fn readme(package: &str, kernel: &str) -> String {
    format!(
        "# {package}

Sampler kernels for laplace models, used from a `.lapsampler` file.

```text
library {{ import {package} }}
kernel {{
  rest : adapt({package}::{kernel})
}}
```

Add it to a project with `laplace add {package} --path <dir>` (or `--git <url>
--tag <tag>` once released); `laplace doc {package}::{kernel}` shows a kernel's
tuning values.

## Kernels

- `{kernel}` -- describe it here.
"
    )
}

fn write_new(path: &Path, contents: &str, created: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    anyhow::ensure!(
        !path.exists(),
        "{} already exists; not overwriting it",
        path.display()
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)?;
    created.push(path.to_path_buf());
    Ok(())
}

/// `laplace new kernel <name>`. Inside a kernel package (`dir` has a
/// laplace.toml with `kind = "kernels"`): the kernel and its test file.
/// Anywhere else: a new kernel package `<dir>/<name>/` with laplace.toml,
/// the kernel, its test file and a README. Returns the files created.
pub fn new_kernel(name: &str, kind: KernelKind, dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    anyhow::ensure!(
        manifest::is_valid_package_name(name),
        "`{name}` cannot be a kernel name (it is used as `package::{name}`): {}",
        manifest::PACKAGE_NAME_RULE
    );
    let mut created = vec![];
    let here = dir.join("laplace.toml");
    let package_dir = match manifest::read_package_manifest(&here) {
        Ok(m) if m.kind == PackageKind::Kernels => dir.to_path_buf(),
        Ok(m) => anyhow::bail!(
            "{} is the function library `{}`; a kernel lives in a kernel package (`kind = \"kernels\"`). \
             Run `laplace new kernel {name}` outside it to create one",
            here.display(),
            m.name
        ),
        // No laplace.toml, or a project's (no `name`): a new package.
        Err(_) => {
            let pkg = dir.join(name);
            anyhow::ensure!(!pkg.exists(), "{} already exists", pkg.display());
            write_new(
                &pkg.join("laplace.toml"),
                &format!(
                    "name = \"{name}\"\nversion = \"0.1.0\"\nkind = \"kernels\"\n\
                     # Kernel packages need a compiler that knows them (older ones refuse it).\n\
                     laplace = \"{KERNEL_PACKAGES_REQUIREMENT}\"\n"
                ),
                &mut created,
            )?;
            write_new(&pkg.join("README.md"), &readme(name, name), &mut created)?;
            pkg
        }
    };
    write_new(
        &package_dir
            .join(KERNELS_DIR)
            .join(format!("{name}.{LAPKERNEL_EXTENSION}")),
        &kernel_skeleton(name, kind),
        &mut created,
    )?;
    write_new(
        &package_dir.join("tests").join(format!("{name}.test.toml")),
        &test_skeleton(name),
        &mut created,
    )?;
    created.sort();
    Ok(created)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_kernel_package_scaffolds_and_checks_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let files = new_kernel("pmala", KernelKind::Proposal, tmp.path()).unwrap();
        let rel: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(tmp.path()).unwrap().display().to_string())
            .collect();
        assert_eq!(
            rel,
            vec![
                "pmala/README.md",
                "pmala/kernels/pmala.lapkernel",
                "pmala/laplace.toml",
                "pmala/tests/pmala.test.toml"
            ]
        );
        let pkg = tmp.path().join("pmala");
        // The manifest is a valid kernel package for this compiler.
        let m = manifest::read_package_manifest(&pkg.join("laplace.toml")).unwrap();
        assert_eq!(m.kind, PackageKind::Kernels);
        assert_eq!(
            laplace::package::check_kernel_package(&pkg, &m),
            Ok(()),
            "the package shape the package manager requires"
        );
        // 4.4's check: `laplace sampler check` passes unchanged.
        let ok = check_kernel(&pkg.join("kernels/pmala.lapkernel")).unwrap();
        assert!(ok.starts_with("ok:"), "{ok}");
        assert!(ok.contains("adapted: eps (step size), m (metric)"), "{ok}");
        // The test file is valid TOML.
        let t: toml::Table =
            toml::from_str(&std::fs::read_to_string(pkg.join("tests/pmala.test.toml")).unwrap())
                .unwrap();
        assert_eq!(t["kernel"].as_str(), Some("pmala"));

        // Inside the package: only the kernel and its test file.
        let files = new_kernel("gibbs", KernelKind::Direct, &pkg).unwrap();
        let rel: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(&pkg).unwrap().display().to_string())
            .collect();
        assert_eq!(
            rel,
            vec!["kernels/gibbs.lapkernel", "tests/gibbs.test.toml"]
        );
        let src = std::fs::read_to_string(pkg.join("kernels/gibbs.lapkernel")).unwrap();
        assert!(src.contains("move(q);"));
        assert!(check_kernel(&pkg.join("kernels/gibbs.lapkernel"))
            .unwrap()
            .starts_with("ok:"));
        assert!(laplace::package::check_kernel_package(&pkg, &m).is_ok());

        // Never overwrites; refuses bad names and function libraries.
        assert!(new_kernel("gibbs", KernelKind::Direct, &pkg)
            .unwrap_err()
            .to_string()
            .contains("already exists"));
        assert!(new_kernel("my-kernel", KernelKind::Proposal, tmp.path()).is_err());
        let lib = tmp.path().join("fns");
        std::fs::create_dir(&lib).unwrap();
        std::fs::write(
            lib.join("laplace.toml"),
            "name = \"fns\"\nversion = \"1.0.0\"\n",
        )
        .unwrap();
        let e = new_kernel("k", KernelKind::Proposal, &lib)
            .unwrap_err()
            .to_string();
        assert!(e.contains("is the function library `fns`"), "{e}");
    }
}
