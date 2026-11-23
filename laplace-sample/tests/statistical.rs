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
//! Conventions (`docs/sampler/laplace_kernels.md`, section 4): 4 chains,
//! 1000 warmup, 1000 draws, fixed seeds. Moment check: for every coordinate,
//! z = (estimate - truth) / MCSE on E[x] and E[x^2]; pass if every |z| < 4.
//! R-hat, ESS and ESS per gradient are printed for the report.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bridgestan::{Model, StanLibrary};
use laplace_sample::diagnostics::{moment, Moment};
use laplace_sample::kernel::StepInfo;
use laplace_sample::model;
use laplace_sample::runner::{chain_rng, run_chain, BuiltIn, Init, SamplerSpec};
use laplace_sample::target::{BridgeStanTarget, Target};

const CHAINS: u64 = 4;
const WARMUP: u64 = 1000;
const DRAWS: u64 = 1000;
const SEED: u64 = 20261123;

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

fn run(r: &Reference, kind: BuiltIn, sets: &[&str]) -> Run {
    run_long(r, kind, sets, WARMUP, DRAWS, 1)
}

/// `draws` per chain, keeping every `thin`-th for the diagnostics (gradient
/// counts still include every iteration).
fn run_long(
    r: &Reference,
    kind: BuiltIn,
    sets: &[&str],
    warmup: u64,
    draws: u64,
    thin: u64,
) -> Run {
    let sets: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
    let spec = SamplerSpec::new(kind, &sets).unwrap();
    let results: Vec<(Vec<Vec<f64>>, Vec<StepInfo>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (1..=CHAINS)
            .map(|chain| {
                let spec = spec.clone();
                s.spawn(move || {
                    let data = r
                        .data
                        .as_ref()
                        .map(|p| CString::new(p.to_str().unwrap()).unwrap());
                    let model = Model::new(r.lib.clone(), data, SEED as u32).unwrap();
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
                        &mut chain_rng(SEED, chain),
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

/// Moment check; prints a summary line and returns the rows.
fn moment_check(r: &Reference, kind: BuiltIn, run: &Run) -> Vec<Moment> {
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
    let grads: u64 = run.infos.iter().flatten().map(|i| i.n_grad).sum();
    let min_ess = rows.iter().map(|m| m.ess).fold(f64::INFINITY, f64::min);
    let max_rhat = rows.iter().map(|m| m.rhat).fold(0.0, f64::max);
    let max_z = rows.iter().map(|m| m.z.abs()).fold(0.0, f64::max);
    println!(
        "RESULT target={} sampler={} max|z|={:.2} max_rhat={:.4} min_ess={:.0} grads={} ess_per_grad={:.5} {}",
        r.name,
        kind.name(),
        max_z,
        max_rhat,
        min_ess,
        grads,
        min_ess / grads as f64,
        if max_z < 4.0 { "PASS" } else { "FAIL" }
    );
    rows
}

fn assert_moments(rows: &[Moment], what: &str) {
    for m in rows {
        assert!(
            m.z.abs() < 4.0,
            "{what}: {} estimate {:.4} truth {:.4} mcse {:.4} z {:.2}",
            m.name,
            m.estimate,
            m.truth,
            m.mcse,
            m.z
        );
    }
}

fn check(target: &'static str, kind: BuiltIn) {
    let r = reference(target);
    let run = run(&r, kind, &[]);
    let rows = moment_check(&r, kind, &run);
    assert_moments(&rows, &format!("{target}/{}", kind.name()));
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

/// RWM and MALA mix slowly on the rotated condition-100 target: at 1000
/// draws the moment check passes only because the MCSE is large. Given
/// enough draws they converge: 4 x 50 000 draws, thinned 1:50 for the
/// diagnostics, must also meet the chain-health bar (R-hat < 1.01,
/// ESS > 400).
fn check_long(kind: BuiltIn) {
    let r = reference("corr_gauss_10");
    let run = run_long(&r, kind, &[], 5000, 50_000, 50);
    let rows = moment_check(&r, kind, &run);
    assert_moments(&rows, &format!("corr_gauss_10/{} long", kind.name()));
    for m in &rows {
        assert!(
            m.rhat < 1.01 && m.ess > 400.0,
            "{}: rhat {:.4} ess {:.0}",
            m.name,
            m.rhat,
            m.ess
        );
    }
}

#[test]
#[ignore]
fn rwm_corr_gauss_long() {
    check_long(BuiltIn::Rwm);
}
#[test]
#[ignore]
fn mala_corr_gauss_long() {
    check_long(BuiltIn::Mala);
}

/// 1.5: after warmup, MALA's acceptance on the standard normal is within
/// +-0.05 of its target (0.574).
#[test]
#[ignore]
fn mala_acceptance_after_warmup_is_on_target() {
    let r = reference("std_normal_10");
    let run = run(&r, BuiltIn::Mala, &[]);
    let probs: Vec<f64> = run
        .infos
        .iter()
        .flatten()
        .map(|i| i.accept_prob.unwrap())
        .collect();
    let mean = probs.iter().sum::<f64>() / probs.len() as f64;
    println!("RESULT mala post-warmup mean accept_stat = {mean:.4} (target 0.574)");
    assert!((mean - 0.574).abs() < 0.05, "mean acceptance {mean}");
}
