//! Phase 3.5: parsing, resolution and one failing-file test per check rule.

use std::path::Path;

use super::*;
use crate::blocks::parameters;

fn kernels_dir() -> String {
    format!("{}/kernels", env!("CARGO_MANIFEST_DIR"))
}

/// Load from text; relative kernel paths resolve against the shipped
/// kernels' parent directory (so ./kernels/mala.lapkernel works).
fn load(src: &str) -> Result<SamplerFile, Vec<String>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    SamplerFile::from_source("s.lapsampler", src.to_string(), dir)
        .map_err(|es| es.iter().map(|e| e.to_string()).collect())
}

fn one_error(src: &str) -> String {
    let es = load(src).expect_err("should fail");
    assert_eq!(es.len(), 1, "{es:#?}");
    es[0].trim_start_matches("s.lapsampler:").to_string()
}

/// Model with parameters beta (3), sigma, tau, theta (simplex[4]: 3).
fn model_params() -> Vec<Param> {
    let names: Vec<String> = [
        "beta.1", "beta.2", "beta.3", "sigma", "tau", "theta.1", "theta.2", "theta.3",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    parameters(&names).unwrap()
}

fn block_error(src: &str) -> String {
    let f = load(src).unwrap_or_else(|e| panic!("{e:#?}"));
    let es = f.blocks(&model_params()).expect_err("should fail");
    assert_eq!(es.len(), 1, "{es:#?}");
    es[0]
        .to_string()
        .trim_start_matches("s.lapsampler:")
        .to_string()
}

const PLAN: &str = r#"sampler hybrid {
  kernel {
    block(beta)              : nuts()
    block(sigma, tau) as sc  : adapt(mala(), target_accept = 0.57)
    rest                     : rwm()
  }
  warmup { iterations = 1000; }
  draws  { iterations = 1000; thin = 1; }
}
"#;

#[test]
fn the_plan_example_loads_and_covers_the_model() {
    let f = load(PLAN).unwrap();
    assert_eq!(f.ast.name, "hybrid");
    assert_eq!(
        (f.warmup(), f.draws(), f.thin()),
        (Some(1000), Some(1000), Some(1))
    );
    let blocks = f.blocks(&model_params()).unwrap();
    let labels: Vec<&str> = blocks.iter().map(|b| b.label.as_str()).collect();
    assert_eq!(labels, vec!["beta", "sc", "rest"]);
    assert_eq!(blocks[2].params, vec!["theta".to_string()]);
    assert_eq!(
        blocks[2].indices,
        vec![5, 6, 7],
        "simplex[4] owns 3 coordinates"
    );
    assert_eq!(
        render(&f.ast.lines[1].kernel),
        "adapt(mala(), target_accept = 0.57)"
    );
}

#[test]
fn every_combinator_and_local_kernels_parse() {
    let src = r#"
// a comment
sampler all {
  kernel {
    block(beta) : seq(adapt(./kernels/mala.lapkernel), repeat(3, rwm(step_size = 0.2)))
    block(theta) : mix(1: adapt(./kernels/rwm.lapkernel, target_accept = 0.3), 2.5: hmc(steps = 5, jitter = false))
    rest : ./kernels/mala.lapkernel(eps = 0.05)
  }
}"#;
    let f = load(src).unwrap_or_else(|e| panic!("{e:#?}"));
    assert_eq!(f.blocks(&model_params()).unwrap().len(), 3);
    let _ = kernels_dir();
}

// ---- syntax ------------------------------------------------------------------

#[test]
fn syntax_errors_point_at_line_and_column() {
    let e = one_error("sampler s {\n  kernel {\n    block(beta) nuts()\n  }\n}\n");
    assert!(e.starts_with("3:17: syntax error"), "{e}");
    let e = one_error("sampler s {\n  kernel {\n    rest : \n  }\n}\n");
    assert!(e.starts_with("4:3: syntax error: expected a kernel"), "{e}");
}

// ---- structure -----------------------------------------------------------------

#[test]
fn no_kernel_block() {
    assert_eq!(
        one_error("sampler s {\n  warmup { iterations = 10; }\n}\n"),
        "1:1: sampler `s` has no `kernel { }` block"
    );
    assert_eq!(
        one_error("sampler s {\n  kernel { }\n}\n"),
        "2:3: `kernel { }` is empty"
    );
}

#[test]
fn duplicate_sections() {
    assert_eq!(
        one_error("sampler s {\n  kernel { rest : rwm() }\n  kernel { rest : rwm() }\n}\n"),
        "3:3: duplicate `kernel` block"
    );
}

#[test]
fn bad_run_settings() {
    assert_eq!(
        one_error("sampler s {\n  kernel { rest : rwm() }\n  draws { iterations = 10; thinning = 2; }\n}\n"),
        "3:28: unknown setting `thinning`; expected iterations or thin"
    );
    assert_eq!(
        one_error("sampler s {\n  kernel { rest : rwm() }\n  warmup { iterations = 10.5; }\n}\n"),
        "3:12: `iterations` must be a whole number, got 10.5"
    );
}

#[test]
fn rest_must_be_last() {
    assert_eq!(
        one_error("sampler s {\n  kernel {\n    rest : rwm()\n    block(beta) : mala()\n  }\n}\n"),
        "3:5: `rest` must be the last kernel line: it takes whatever the lines above left"
    );
}

#[test]
fn labels_must_be_unique() {
    assert_eq!(
        one_error("sampler s {\n  kernel {\n    block(beta) as b : rwm()\n    block(sigma) as b : rwm()\n  }\n}\n"),
        "4:5: block label `b` is already used at 3:5; name one with `as <label>`"
    );
}

// ---- kernels -------------------------------------------------------------------

#[test]
fn unknown_kernel() {
    let e = one_error("sampler s {\n  kernel { rest : langevin() }\n}\n");
    assert!(
        e.starts_with(
            "2:19: unknown kernel `langevin`: the built-in kernels are nuts, rwm, mala and hmc"
        ),
        "{e}"
    );
}

#[test]
fn unknown_or_misplaced_settings() {
    let e = one_error("sampler s {\n  kernel { rest : mala(eps = 0.1) }\n}\n");
    assert_eq!(
        e,
        "2:24: mala() has no setting `eps`; its settings are: step_size"
    );
    let e = one_error("sampler s {\n  kernel { rest : mala(target_accept = 0.5) }\n}\n");
    assert!(
        e.ends_with("(adaptation is explicit here: adapt(kernel, target_accept = ...))"),
        "{e}"
    );
    let e = one_error("sampler s {\n  kernel { rest : ./kernels/mala.lapkernel(m = 1) }\n}\n");
    assert_eq!(
        e,
        "2:44: ./kernels/mala.lapkernel has no tuning value `m`; its tuning values are: eps"
    );
}

// ---- library kernels (Phase 4.2) --------------------------------------------------

/// A project whose laplace.lock has a kernel package `langevin` (a path
/// dependency: mala, broken) and a function library `gps`.
fn project() -> (tempfile::TempDir, Project) {
    use laplace::manifest::PackageKind;
    use laplace::resolve::lockfile::{write_lockfile, LockedPackage, Lockfile};
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let pkg = root.join("langevin");
    std::fs::create_dir_all(pkg.join("kernels")).unwrap();
    std::fs::write(
        pkg.join("laplace.toml"),
        "name = \"langevin\"\nversion = \"0.1.0\"\nkind = \"kernels\"\nlaplace = \">=0.3\"\n",
    )
    .unwrap();
    std::fs::copy(
        format!("{}/mala.lapkernel", kernels_dir()),
        pkg.join("kernels/mala.lapkernel"),
    )
    .unwrap();
    std::fs::write(
        pkg.join("kernels/broken.lapkernel"),
        "kernel broken {\n  step {\n    move(nope);\n  }\n}\n",
    )
    .unwrap();
    let dir = root.join("project");
    std::fs::create_dir_all(&dir).unwrap();
    let lock = Lockfile {
        root: vec!["gps".into(), "langevin".into()],
        packages: vec![
            LockedPackage::leaf("gps", "1.0.0", "sha256:0", "registry"),
            LockedPackage {
                kind: PackageKind::Kernels,
                ..LockedPackage::leaf("langevin", "0.1.0", "sha256:0", "path+../langevin")
            },
        ],
    };
    write_lockfile(&dir.join("laplace.lock"), &lock).unwrap();
    let project = Project {
        dir,
        cache_root: root.join("home/.laplace/packages"),
    };
    (tmp, project)
}

fn load_in(src: &str, project: &Project) -> Result<SamplerFile, Vec<String>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    SamplerFile::from_source_in("s.lapsampler", src.to_string(), dir, project)
        .map_err(|es| es.iter().map(|e| e.to_string()).collect())
}

fn one_error_in(src: &str, project: &Project) -> String {
    let es = load_in(src, project).expect_err("should fail");
    assert_eq!(es.len(), 1, "{es:#?}");
    es[0].trim_start_matches("s.lapsampler:").to_string()
}

#[test]
fn a_library_kernel_resolves_through_the_lock_with_provenance() {
    let (_tmp, p) = project();
    let f = load_in(
        "sampler s {\n  library { import langevin }\n  kernel {\n    block(beta) : adapt(langevin::mala(eps = 0.1))\n    \
         block(theta) : seq(langevin::mala, ./kernels/rwm.lapkernel)\n    rest : nuts()\n  }\n}\n",
        &p,
    )
    .unwrap_or_else(|e| panic!("{e:#?}"));
    let target: Arc<dyn Target> = Arc::new(crate::target::analytic::DiagGaussian {
        mu: vec![0.0; 8],
        sd: vec![1.0; 8],
    });
    let built = f.instantiate(&model_params(), target, 10, 10).unwrap();
    let v: Vec<(&str, &str)> = built
        .lines
        .iter()
        .map(|l| (l.name.as_str(), l.version.as_str()))
        .collect();
    assert_eq!(
        v,
        vec![
            ("adapt(langevin::mala(eps = 0.1))", "langevin@0.1.0"),
            (
                "seq(langevin::mala, ./kernels/rwm.lapkernel)",
                "langevin@0.1.0+local"
            ),
            ("nuts()", crate::VERSION),
        ]
    );
    // The hash covers the installed kernel's own source.
    let mala = std::fs::read_to_string(format!("{}/mala.lapkernel", kernels_dir())).unwrap();
    assert!(built.lines[0].source.contains(&mala));
}

#[test]
fn library_kernel_errors() {
    let (_tmp, p) = project();
    let src = |lib: &str, k: &str| {
        format!("sampler s {{\n  library {{ {lib} }}\n  kernel {{ rest : {k} }}\n}}\n")
    };
    let e = one_error_in(&src("", "langevin::mala"), &p);
    assert!(
        e.starts_with("3:19: `langevin::mala`: `langevin` is not imported; add `import langevin`"),
        "{e}"
    );
    let e = one_error_in(&src("import langevin@0.2", "langevin::mala"), &p);
    assert!(
        e.starts_with(
            "2:13: `import langevin@0.2`: versions come only from laplace.toml and laplace.lock"
        ),
        "{e}"
    );
    let e = one_error_in(&src("import langevin", "langevin::pmala"), &p);
    assert_eq!(
        e,
        "3:19: `langevin` 0.1.0 has no kernel `pmala`; its kernels are: broken, mala"
    );
    let e = one_error_in(&src("import langevin", "langevin::mala(m = 1)"), &p);
    assert!(
        e.starts_with("3:34: langevin::mala has no tuning value `m`; its tuning values are: eps"),
        "{e}"
    );
    let e = one_error_in(&src("import langevin", "langevin::broken"), &p);
    assert!(
        e.starts_with("3:19: langevin::broken does not check:"),
        "{e}"
    );
    assert!(
        e.contains("langevin/kernels/broken.lapkernel:3:10:"),
        "the kernel's own position, never the cache path: {e}"
    );
    let e = one_error_in(&src("import gps", "nuts()"), &p);
    assert!(e.starts_with("2:13: `gps` is a function library"), "{e}");
    let e = one_error_in(&src("import nothere", "nuts()"), &p);
    assert!(
        e.starts_with("2:13: `nothere` is not in laplace.lock -- add it with `laplace add nothere"),
        "{e}"
    );
    let e = one_error_in(&src("import langevin import langevin", "nuts()"), &p);
    assert!(e.starts_with("2:29: `langevin` is imported twice"), "{e}");
    // No lock at all.
    let elsewhere = Project {
        dir: p.dir.join("sub"),
        cache_root: p.cache_root.clone(),
    };
    let e = one_error_in(&src("import langevin", "langevin::mala"), &elsewhere);
    assert!(e.contains("there is no laplace.lock in"), "{e}");
}

#[test]
fn missing_or_failing_kernel_files() {
    let e = one_error("sampler s {\n  kernel { rest : ./kernels/nope.lapkernel }\n}\n");
    assert!(e.starts_with("2:19: cannot read"), "{e}");
}

/// Q14: a .lapkernel whose @adapt(step_size) has no target_accept is a
/// `laplace sampler check` error, reported at the line using it with the
/// kernel's own file:line:col.
#[test]
fn a_kernel_without_target_accept_is_a_check_error() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("k.lapkernel"),
        "kernel k {\n  tuning { real eps = 1 @adapt(step_size); }\n  step { move(q); }\n}\n",
    )
    .unwrap();
    let es = SamplerFile::from_source(
        "s.lapsampler",
        "sampler s {\n  kernel { rest : adapt(./k.lapkernel) }\n}\n".into(),
        tmp.path(),
    )
    .unwrap_err();
    let all = es
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        all.contains("s.lapsampler:2:25: ./k.lapkernel does not check:"),
        "{all}"
    );
    assert!(
        all.contains(
            "k.lapkernel:2:25: @adapt(step_size) needs the kernel's default acceptance target"
        ),
        "{all}"
    );
}

// ---- adapt contracts -------------------------------------------------------------

#[test]
fn adapt_needs_something_to_adapt() {
    let e = one_error("sampler s {\n  kernel { rest : adapt(nuts()) }\n}\n");
    assert_eq!(
        e,
        "2:19: adapt(): nuts() adapts its own step size and metric; use it without adapt()"
    );
    let e = one_error("sampler s {\n  kernel { rest : adapt(seq(rwm(), mala())) }\n}\n");
    assert!(
        e.starts_with("2:19: adapt(): adapt() wraps a single kernel, not seq(...)"),
        "{e}"
    );
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("f.lapkernel"),
        "kernel f {\n  tuning { real<lower=0> h = 0.1; }\n  step { propose(q + h * std_normal_rng(dim), 0); }\n}\n",
    )
    .unwrap();
    let es = SamplerFile::from_source(
        "s.lapsampler",
        "sampler s {\n  kernel { rest : adapt(./f.lapkernel) }\n}\n".into(),
        tmp.path(),
    )
    .unwrap_err();
    assert_eq!(es.len(), 1);
    assert!(
        es[0]
            .to_string()
            .contains("adapt(): ./f.lapkernel has nothing to adapt: tag its step size"),
        "{}",
        es[0]
    );
    let e = one_error("sampler s {\n  kernel { rest : adapt(mala(), target_accept = 1.2) }\n}\n");
    assert_eq!(
        e,
        "2:33: target_accept must be a number strictly between 0 and 1"
    );
}

#[test]
fn nuts_under_mix_and_on_inside_a_line() {
    let e = one_error("sampler s {\n  kernel { rest : mix(1: nuts(), 1: rwm()) }\n}\n");
    assert!(e.starts_with("2:26: nuts() cannot sit under mix()"), "{e}");
    let e = one_error("sampler s {\n  kernel { rest : on(block(beta), rwm()) }\n}\n");
    assert!(
        e.starts_with("2:19: on(block(...), ...) cannot be used inside a kernel line"),
        "{e}"
    );
    let e = one_error("sampler s {\n  kernel { rest : mix(0: rwm(), 1: mala()) }\n}\n");
    assert_eq!(e, "2:19: mix() weights must be positive numbers");
    let e = one_error("sampler s {\n  kernel { rest : repeat(0, rwm()) }\n}\n");
    assert_eq!(e, "2:19: repeat(n, ...) needs a whole number n >= 1, got 0");
}

// ---- blocks against the model (D2) ------------------------------------------------

#[test]
fn coverage_is_required() {
    let e = block_error("sampler s {\n  kernel {\n    block(beta) : nuts()\n  }\n}\n");
    assert_eq!(
        e,
        "2:3: parameter(s) `sigma`, `tau`, `theta` are in no block (D2); add them to a block or end with `rest : <kernel>`"
    );
}

#[test]
fn unknown_parameter() {
    let e = block_error(
        "sampler s {\n  kernel {\n    block(gamma) : nuts()\n    rest : rwm()\n  }\n}\n",
    );
    assert_eq!(
        e,
        "3:5: the model has no parameter `gamma`; its parameters are: beta, sigma, tau, theta"
    );
}

#[test]
fn a_parameter_in_two_blocks() {
    let e = block_error("sampler s {\n  kernel {\n    block(beta) : nuts()\n    block(beta, sigma) as x : rwm()\n    rest : rwm()\n  }\n}\n");
    assert_eq!(
        e,
        "4:5: `beta` is already in the block at 3:5; a parameter belongs to one block"
    );
}

#[test]
fn rest_that_covers_nothing() {
    let e = block_error(
        "sampler s {\n  kernel {\n    block(beta, sigma, tau, theta) as all : nuts()\n    rest : rwm()\n  }\n}\n",
    );
    assert_eq!(
        e,
        "4:5: `rest` covers no parameters: every parameter is already in a block"
    );
}

// ---- never re-tune after warmup, for every combinator (Phase 3 review) -----------

/// Six coordinates in two correlated blocks `a` and `b`.
fn coupled_model() -> (std::sync::Arc<dyn crate::target::Target>, Vec<Param>) {
    let mut p = vec![vec![0.0; 6]; 6];
    for (i, row) in p.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = if i == j {
                2.0
            } else {
                0.3 + 0.05 * (i + j) as f64
            };
        }
    }
    let names: Vec<String> = ["a.1", "a.2", "a.3", "b.1", "b.2", "b.3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    (
        std::sync::Arc::new(crate::target::analytic::Gaussian { precision: p }),
        parameters(&names).unwrap(),
    )
}

fn assert_frozen_after_warmup(src: &str) {
    use crate::runner::{chain_rng, initial_state, Init};
    use crate::state::Evaluator;
    let (target, params) = coupled_model();
    let f = load(src).unwrap_or_else(|e| panic!("{e:#?}"));
    let (warmup, draws) = (400, 300);
    let mut k = f
        .instantiate(&params, target.clone(), warmup, draws)
        .unwrap()
        .kernel;
    let mut ev = Evaluator::new(target.as_ref());
    let mut rng = chain_rng(9, 1);
    let mut st = initial_state(&Init::Radius(2.0), &mut ev, &mut rng).unwrap();
    k.init(&st, &mut ev, &mut rng).unwrap();
    let start = k.tunings();
    for _ in 0..warmup {
        k.step(&mut st, &mut ev, &mut rng).unwrap();
    }
    k.end_warmup().unwrap();
    let frozen = k.tunings();
    for ((path, before), (_, after)) in start.iter().zip(&frozen) {
        if path.contains("adapt/") || path.contains("nuts") {
            assert_ne!(before, after, "{path}: nothing adapted during warmup");
        }
    }
    for i in 0..draws {
        k.step(&mut st, &mut ev, &mut rng).unwrap();
        assert_eq!(
            k.tunings(),
            frozen,
            "{src}\nre-tuned at post-warmup step {i}"
        );
    }
}

#[test]
fn no_combinator_re_tunes_after_warmup() {
    // seq, repeat, adapt of Rust kernels, on (every line), mix with adapt()
    // inside (fewer warmup steps than the schedule expects), a .lapkernel.
    assert_frozen_after_warmup(
        "sampler s {\n  kernel {\n    block(a) : seq(adapt(mala()), repeat(2, adapt(hmc(steps = 4))))\n    \
         block(b) : mix(1: adapt(rwm()), 2: adapt(./kernels/mala.lapkernel))\n  }\n}\n",
    );
    // nuts() as a block, beside adapt(); and nuts() under repeat().
    assert_frozen_after_warmup(
        "sampler s {\n  kernel {\n    block(a) : nuts()\n    block(b) : repeat(3, nuts())\n  }\n}\n",
    );
    assert_frozen_after_warmup(
        "sampler s {\n  kernel {\n    block(a) : repeat(2, seq(nuts(), adapt(mala())))\n    rest : adapt(./kernels/rwm.lapkernel, target_accept = 0.3)\n  }\n}\n",
    );
}
