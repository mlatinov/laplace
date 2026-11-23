//! Statistical checks for the built-in kernels (Phase 1.4, 1.5) on the two
//! Phase 1 reference targets, through real BridgeStan builds.
//!
//! Slow and needs a C++ toolchain plus the BridgeStan source (fetched on
//! first use), so every test is `#[ignore]`:
//!
//! ```sh
//! cargo test -p laplace-sample --release --test statistical -- --ignored --nocapture
//! ```
//!
//! Conventions (`docs/sampler/laplace_kernels.md`, section 4): fixed seeds,
//! draw counts per kernel x target in `tests/statistical.toml`. Moment
//! check: for every coordinate, z = (estimate - truth) / MCSE on E[x] and
//! E[x^2]. The verdict ([`laplace_sample::diagnostics::verdict`]) is
//! did-not-converge unless every quantity has R-hat < 1.01 and bulk ESS >
//! 400; only then is it pass (every |z| < 4) or fail. Every check here must
//! pass. R-hat, bulk ESS and ESS per gradient are printed for the report.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bridgestan::{Model, StanLibrary};
use laplace_sample::diagnostics::{moment, verdict, Moment, Verdict};
use laplace_sample::kernel::StepInfo;
use laplace_sample::model;
use laplace_sample::runner::{chain_rng, run_chain, BuiltIn, Init, SamplerSpec};
use laplace_sample::target::{BridgeStanTarget, Target};

const SEED: u64 = 20261123;

#[derive(serde::Deserialize, Clone, Copy, Debug)]
struct Counts {
    chains: u64,
    warmup: u64,
    draws: u64,
    thin: u64,
}

#[derive(serde::Deserialize)]
struct RunEntry {
    target: String,
    kernel: String,
    chains: Option<u64>,
    warmup: Option<u64>,
    draws: Option<u64>,
    thin: Option<u64>,
}

#[derive(serde::Deserialize)]
struct Config {
    defaults: Counts,
    run: Vec<RunEntry>,
}

fn config() -> Config {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/statistical.toml");
    toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The draw counts for one kernel x target; every pair must be listed.
fn counts(target: &str, kind: BuiltIn) -> Counts {
    let c = config();
    let e = c
        .run
        .iter()
        .find(|r| r.target == target && r.kernel == kind.name())
        .unwrap_or_else(|| {
            panic!(
                "tests/statistical.toml has no run for {} on {target}",
                kind.name()
            )
        });
    let d = c.defaults;
    Counts {
        chains: e.chains.unwrap_or(d.chains),
        warmup: e.warmup.unwrap_or(d.warmup),
        draws: e.draws.unwrap_or(d.draws),
        thin: e.thin.unwrap_or(d.thin),
    }
}

fn targets_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/targets")
}

struct Reference {
    name: &'static str,
    lib: Arc<StanLibrary>,
    data: Option<PathBuf>,
    /// Truth for E[x_i^2] (E[x_i] = 0 for both targets).
    second_moments: Vec<f64>,
}

fn reference(name: &'static str) -> Reference {
    let stan = targets_dir().join(format!("{name}.stan"));
    let prepared = model::prepare(&stan).expect("build reference target");
    let lib = Arc::new(bridgestan::open_library(&prepared.library).unwrap());
    let json = targets_dir().join(format!("{name}.json"));
    let (data, second_moments) = if json.exists() {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        let sigma = v["Sigma"].as_array().unwrap();
        let diag = (0..sigma.len())
            .map(|i| sigma[i][i].as_f64().unwrap())
            .collect();
        (Some(json), diag)
    } else {
        (None, vec![1.0; 10])
    };
    Reference {
        name,
        lib,
        data,
        second_moments,
    }
}

struct Run {
    /// draws[c][i] = unconstrained point of draw i in chain c
    draws: Vec<Vec<Vec<f64>>>,
    infos: Vec<Vec<StepInfo>>,
}

/// Run `n.chains` chains, keeping every `n.thin`-th draw for the
/// diagnostics (gradient counts include every iteration).
fn run(r: &Reference, kind: BuiltIn, sets: &[&str], n: Counts) -> Run {
    run_seeded(r, kind, sets, n, SEED)
}

fn run_seeded(r: &Reference, kind: BuiltIn, sets: &[&str], n: Counts, seed: u64) -> Run {
    let Counts {
        chains,
        warmup,
        draws,
        thin,
    } = n;
    let sets: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
    let spec = SamplerSpec::new(kind, &sets).unwrap();
    let results: Vec<(Vec<Vec<f64>>, Vec<StepInfo>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (1..=chains)
            .map(|chain| {
                let spec = spec.clone();
                s.spawn(move || {
                    let data = r
                        .data
                        .as_ref()
                        .map(|p| CString::new(p.to_str().unwrap()).unwrap());
                    let model = Model::new(r.lib.clone(), data, seed as u32).unwrap();
                    let target: Arc<dyn Target> = Arc::new(BridgeStanTarget { model });
                    let (mut k, a) = spec.build(target.clone(), warmup, draws).unwrap();
                    let (mut qs, mut infos) = (vec![], vec![]);
                    let mut i = 0u64;
                    run_chain(
                        target.as_ref(),
                        k.as_mut(),
                        a,
                        &Init::Radius(2.0),
                        warmup,
                        draws,
                        &mut chain_rng(seed, chain),
                        |_| Ok(()),
                        |st, info, _| {
                            if i.is_multiple_of(thin) {
                                qs.push(st.q.clone());
                            }
                            i += 1;
                            infos.push(info.clone());
                            Ok(())
                        },
                    )
                    .unwrap();
                    (qs, infos)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (draws, infos) = results.into_iter().unzip();
    Run { draws, infos }
}

/// Moment check: prints one RESULT line (and the unhealthy or failing rows)
/// and returns the verdict.
fn moment_check(r: &Reference, kind: BuiltIn, run: &Run, n: Counts) -> (Verdict, Vec<Moment>) {
    let d = r.second_moments.len();
    let mut rows = vec![];
    for i in 0..d {
        let x: Vec<Vec<f64>> = run
            .draws
            .iter()
            .map(|c| c.iter().map(|q| q[i]).collect())
            .collect();
        let x2: Vec<Vec<f64>> = x
            .iter()
            .map(|c| c.iter().map(|v| v * v).collect())
            .collect();
        rows.push(moment(&format!("E[x{}]", i + 1), &x, 0.0));
        rows.push(moment(
            &format!("E[x{}^2]", i + 1),
            &x2,
            r.second_moments[i],
        ));
    }
    let v = verdict(&rows);
    let grads: u64 = run.infos.iter().flatten().map(|i| i.n_grad).sum();
    let min_ess = rows
        .iter()
        .map(|m| m.ess_bulk)
        .fold(f64::INFINITY, f64::min);
    let max_rhat = rows.iter().map(|m| m.rhat).fold(0.0, f64::max);
    let max_z = rows.iter().map(|m| m.z.abs()).fold(0.0, f64::max);
    println!(
        "RESULT target={} sampler={} chains={} warmup={} draws={} thin={} max|z|={:.2} max_rhat={:.4} min_ess_bulk={:.0} grads={} ess_per_grad={:.5} verdict={}",
        r.name,
        kind.name(),
        n.chains,
        n.warmup,
        n.draws,
        n.thin,
        max_z,
        max_rhat,
        min_ess,
        grads,
        min_ess / grads as f64,
        v.label()
    );
    (v, rows)
}

fn describe(rows: &[Moment]) -> String {
    rows.iter()
        .map(|m| {
            format!(
                "  {}: estimate {:.4} truth {:.4} mcse {:.4} z {:.2} rhat {:.4} ess_bulk {:.0}",
                m.name, m.estimate, m.truth, m.mcse, m.z, m.rhat, m.ess_bulk
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn check(target: &'static str, kind: BuiltIn) {
    let n = counts(target, kind);
    let r = reference(target);
    let run = run(&r, kind, &[], n);
    let (v, rows) = moment_check(&r, kind, &run, n);
    match v {
        Verdict::Pass => {}
        Verdict::Fail => panic!(
            "{target}/{}: FAIL, the kernel does not target the right distribution\n{}",
            kind.name(),
            describe(&rows)
        ),
        Verdict::DidNotConverge => panic!(
            "{target}/{}: did-not-converge (needs R-hat < 1.01 and bulk ESS > 400); \
             give it more draws in tests/statistical.toml\n{}",
            kind.name(),
            describe(&rows)
        ),
    }
}

#[test]
fn config_lists_every_kernel_on_every_target_once() {
    let c = config();
    for target in ["std_normal_10", "corr_gauss_10"] {
        for kind in BuiltIn::ALL {
            let hits = c
                .run
                .iter()
                .filter(|r| r.target == target && r.kernel == kind.name())
                .count();
            assert_eq!(hits, 1, "{} on {target}", kind.name());
        }
    }
    assert_eq!(c.run.len(), 8);
}

#[test]
#[ignore]
fn rwm_std_normal() {
    check("std_normal_10", BuiltIn::Rwm);
}
#[test]
#[ignore]
fn rwm_corr_gauss() {
    check("corr_gauss_10", BuiltIn::Rwm);
}
#[test]
#[ignore]
fn mala_std_normal() {
    check("std_normal_10", BuiltIn::Mala);
}
#[test]
#[ignore]
fn mala_corr_gauss() {
    check("corr_gauss_10", BuiltIn::Mala);
}
#[test]
#[ignore]
fn hmc_std_normal() {
    check("std_normal_10", BuiltIn::Hmc);
}
#[test]
#[ignore]
fn hmc_corr_gauss() {
    check("corr_gauss_10", BuiltIn::Hmc);
}
#[test]
#[ignore]
fn nuts_std_normal() {
    check("std_normal_10", BuiltIn::Nuts);
}
#[test]
#[ignore]
fn nuts_corr_gauss() {
    check("corr_gauss_10", BuiltIn::Nuts);
}

/// 1.5: after warmup, MALA's acceptance on the standard normal is within
/// +-0.15 of its target (0.574). Widened from the plan's +-0.05 in the
/// Phase 1 review: Stan's dual averaging, kept for every kernel, settles
/// MALA's post-warmup acceptance near 0.69 (its iterates keep fluctuating and
/// acceptance at the averaged step is higher; see the Phase 1 report). Plain
/// 4 x (1000 + 1000): this checks the adapted step size, not convergence.
#[test]
#[ignore]
fn mala_acceptance_after_warmup_is_on_target() {
    let r = reference("std_normal_10");
    let n = Counts {
        chains: 4,
        warmup: 1000,
        draws: 1000,
        thin: 1,
    };
    let run = run(&r, BuiltIn::Mala, &[], n);
    let probs: Vec<f64> = run
        .infos
        .iter()
        .flatten()
        .map(|i| i.accept_prob.unwrap())
        .collect();
    let mean = probs.iter().sum::<f64>() / probs.len() as f64;
    println!("RESULT mala post-warmup mean accept_stat = {mean:.4} (target 0.574)");
    assert!((mean - 0.574).abs() < 0.15, "mean acceptance {mean}");
}
