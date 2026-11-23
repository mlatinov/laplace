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
struct HybridEntry {
    sampler: String,
    target: String,
    chains: Option<u64>,
    warmup: Option<u64>,
    draws: Option<u64>,
    thin: Option<u64>,
}

#[derive(serde::Deserialize)]
struct Config {
    defaults: Counts,
    run: Vec<RunEntry>,
    #[serde(default)]
    hybrid: Vec<HybridEntry>,
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
    /// Sampling-phase wall time, summed over chains.
    secs: f64,
}

/// What a run cost in its sampling phase (cost accounting, Phase 4):
/// gradient and density-only evaluations counted apart, and wall time
/// summed over chains (chains run in parallel threads; the harness's own
/// per-draw work is included, the same for every sampler it compares).
#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    grads: u64,
    logps: u64,
    secs: f64,
}

impl Cost {
    fn add_info(&mut self, i: &StepInfo) {
        self.grads += i.n_grad;
        self.logps += i.n_logp;
    }
    fn merge(&mut self, o: Cost) {
        self.grads += o.grads;
        self.logps += o.logps;
        self.secs += o.secs;
    }
    /// `ess_per_grad` is n/a for a run with no gradient (rwm).
    fn report(&self, min_ess: f64) -> String {
        let per_grad = if self.grads > 0 {
            format!("{:.5}", min_ess / self.grads as f64)
        } else {
            "n/a".into()
        };
        format!(
            "grads={} logps={} ess_per_grad={per_grad} secs={:.3} ess_per_s={:.0}",
            self.grads,
            self.logps,
            self.secs,
            min_ess / self.secs
        )
    }
    fn ess_per_s(&self, min_ess: f64) -> f64 {
        min_ess / self.secs
    }
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
    let results: Vec<(Vec<Vec<f64>>, Vec<StepInfo>, f64)> = std::thread::scope(|s| {
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
                    let mut k = spec.build(target.clone(), warmup, draws).unwrap();
                    let (mut qs, mut infos) = (vec![], vec![]);
                    let mut i = 0u64;
                    let sampling = std::cell::Cell::new(None);
                    run_chain(
                        target.as_ref(),
                        k.as_mut(),
                        &Init::Radius(2.0),
                        warmup,
                        draws,
                        &mut chain_rng(seed, chain),
                        |_| {
                            sampling.set(Some(std::time::Instant::now()));
                            Ok(())
                        },
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
                    let secs = sampling.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
                    (qs, infos, secs)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let secs = results.iter().map(|r| r.2).sum();
    let (draws, infos) = results.into_iter().map(|(d, i, _)| (d, i)).unzip();
    Run { draws, infos, secs }
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
    let mut cost = Cost {
        secs: run.secs,
        ..Default::default()
    };
    run.infos.iter().flatten().for_each(|i| cost.add_info(i));
    let min_ess = rows
        .iter()
        .map(|m| m.ess_bulk)
        .fold(f64::INFINITY, f64::min);
    let max_rhat = rows.iter().map(|m| m.rhat).fold(0.0, f64::max);
    let max_z = rows.iter().map(|m| m.z.abs()).fold(0.0, f64::max);
    println!(
        "RESULT target={} sampler={} chains={} warmup={} draws={} thin={} max|z|={:.2} max_rhat={:.4} min_ess_bulk={:.0} {} verdict={}",
        r.name,
        kind.name(),
        n.chains,
        n.warmup,
        n.draws,
        n.thin,
        max_z,
        max_rhat,
        min_ess,
        cost.report(min_ess),
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

/// 1.5: after warmup, MALA's mean acceptance on the standard normal is
/// within +-0.15 of its target (0.574). Widened from the plan's +-0.05 in the
/// Phase 1 review: Stan's dual averaging, kept for every kernel, settles
/// MALA's post-warmup acceptance near 0.69 (its iterates keep fluctuating and
/// acceptance at the averaged step is higher; see the Phase 1 report).
///
/// Averaged over 5 seeds (each 4 x (1000 + 1000)) rather than one fixed
/// seed: a single seed's acceptance varies by several hundredths (0.718 on
/// the seed this test used before), which put one seed 0.006 from the
/// bound. This checks the adapted step size, not convergence.
#[test]
#[ignore]
fn mala_acceptance_after_warmup_is_on_target() {
    const SEEDS: [u64; 5] = [11, 22, 33, 44, 55];
    let r = reference("std_normal_10");
    let n = Counts {
        chains: 4,
        warmup: 1000,
        draws: 1000,
        thin: 1,
    };
    let per_seed: Vec<f64> = SEEDS
        .iter()
        .map(|&seed| {
            let run = run_seeded(&r, BuiltIn::Mala, &[], n, seed);
            let probs: Vec<f64> = run
                .infos
                .iter()
                .flatten()
                .map(|i| i.accept_prob.unwrap())
                .collect();
            probs.iter().sum::<f64>() / probs.len() as f64
        })
        .collect();
    let mean = per_seed.iter().sum::<f64>() / per_seed.len() as f64;
    println!(
        "RESULT mala post-warmup mean accept_stat over {} seeds = {mean:.4} (target 0.574, per seed {:?})",
        SEEDS.len(),
        per_seed.iter().map(|p| (p * 1e4).round() / 1e4).collect::<Vec<_>>()
    );
    assert!((mean - 0.574).abs() < 0.15, "mean acceptance {mean}");
}

// ---- Phase 2.5: .lapkernel twins on real BridgeStan builds ------------------------

fn lapkernel(name: &str) -> String {
    format!("{}/kernels/{name}.lapkernel", env!("CARGO_MANIFEST_DIR"))
}

/// Every saved (position, lp, acceptance, step size, metric) of `sampler`.
fn twin_trace(
    r: &Reference,
    sampler: &laplace_sample::runner::Sampler,
    chain: u64,
) -> Vec<Vec<f64>> {
    let data = r
        .data
        .as_ref()
        .map(|p| CString::new(p.to_str().unwrap()).unwrap());
    let model = Model::new(r.lib.clone(), data, SEED as u32).unwrap();
    let target: Arc<dyn Target> = Arc::new(BridgeStanTarget { model });
    let mut k = sampler.build(target.clone(), 1000, 1000).unwrap();
    let mut rows = vec![];
    run_chain(
        target.as_ref(),
        k.as_mut(),
        &Init::Radius(2.0),
        1000,
        1000,
        &mut chain_rng(SEED, chain),
        |_| Ok(()),
        |st, info, t| {
            let mut row = st.q.clone();
            row.push(st.lp);
            row.push(info.accept_prob.unwrap());
            row.push(info.n_grad as f64);
            row.push(t.step_size.unwrap());
            row.extend(
                t.inv_metric
                    .clone()
                    .unwrap_or_else(|| vec![1.0; st.q.len()]),
            );
            rows.push(row);
            Ok(())
        },
    )
    .unwrap();
    rows
}

fn twin_check(target: &'static str, kernel: &str, rust_sets: &[&str], lap_sets: &[&str]) {
    use laplace_sample::runner::Sampler;
    let s = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let rust = Sampler::from_arg(kernel, &s(rust_sets)).unwrap();
    let lap = Sampler::from_arg(&lapkernel(kernel), &s(lap_sets)).unwrap();
    let r = reference(target);
    let mut worst: f64 = 0.0;
    for chain in 1..=4 {
        let (a, b) = (twin_trace(&r, &rust, chain), twin_trace(&r, &lap, chain));
        for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
            let d = (x - y).abs();
            worst = worst.max(if d.is_nan() { f64::INFINITY } else { d });
        }
    }
    println!("RESULT equivalence {kernel} {rust_sets:?} on {target}: max |difference| = {worst:e}");
    assert!(worst <= 1e-10, "max difference {worst}");
}

#[test]
#[ignore]
fn twins_rwm_fixed_tuning() {
    for t in ["std_normal_10", "corr_gauss_10"] {
        twin_check(
            t,
            "rwm",
            &["step_size=0.3", "adapt=false"],
            &["eps=0.3", "adapt=false"],
        );
    }
}

#[test]
#[ignore]
fn twins_rwm_default_adaptation() {
    for t in ["std_normal_10", "corr_gauss_10"] {
        twin_check(t, "rwm", &[], &[]);
    }
}

#[test]
#[ignore]
fn twins_mala_fixed_tuning() {
    for t in ["std_normal_10", "corr_gauss_10"] {
        twin_check(
            t,
            "mala",
            &["step_size=0.3", "adapt=false"],
            &["eps=0.3", "adapt=false"],
        );
    }
}

#[test]
#[ignore]
fn twins_mala_default_adaptation() {
    for t in ["std_normal_10", "corr_gauss_10"] {
        twin_check(t, "mala", &[], &[]);
    }
}

/// Phase 2.5, report only: interpreter time per step against BridgeStan
/// gradient time, on the gp library model (validation/models/gp.stan).
/// MALA costs one gradient per step either way, so
/// (lapkernel step time - Rust step time) is the interpreter's own cost.
#[test]
#[ignore]
fn interpreter_overhead_on_the_gp_model() {
    use laplace_sample::runner::Sampler;
    use std::time::Instant;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/models");
    let prepared = model::prepare(&dir.join("gp.stan")).unwrap();
    let lib = Arc::new(bridgestan::open_library(&prepared.library).unwrap());
    let data = CString::new(dir.join("gp.json").to_str().unwrap()).unwrap();
    let target: Arc<dyn Target> = Arc::new(BridgeStanTarget {
        model: Model::new(lib, Some(data), 1).unwrap(),
    });
    let fixed = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let steps = 3000u64;
    let time = |sampler: &Sampler| -> (f64, u64) {
        let mut k = sampler.build(target.clone(), 0, steps).unwrap();
        let mut grads = 0;
        let t0 = Instant::now();
        run_chain(
            target.as_ref(),
            k.as_mut(),
            &Init::Radius(0.5),
            0,
            steps,
            &mut chain_rng(SEED, 1),
            |_| Ok(()),
            |_, info, _| {
                grads += info.n_grad;
                Ok(())
            },
        )
        .unwrap();
        (t0.elapsed().as_secs_f64() / steps as f64, grads)
    };
    let rust = Sampler::from_arg("mala", &fixed(&["step_size=0.3", "adapt=false"])).unwrap();
    let lap = Sampler::from_arg(&lapkernel("mala"), &fixed(&["eps=0.3", "adapt=false"])).unwrap();
    // Warm up caches and the CPU, then measure each twice and keep the faster.
    time(&rust);
    let (r1, g) = time(&rust);
    let (l1, g2) = time(&lap);
    let (r2, _) = time(&rust);
    let (l2, _) = time(&lap);
    assert_eq!(g, g2, "same gradient count");
    let (r, l) = (r1.min(r2), l1.min(l2));
    let grad = r * steps as f64 / g as f64;
    println!(
        "RESULT overhead gp: rust mala {:.1} us/step, lapkernel mala {:.1} us/step, interpreter {:.1} us/step = {:.1}% of a gradient ({:.1} us, {} grads)",
        r * 1e6,
        l * 1e6,
        (l - r) * 1e6,
        100.0 * (l - r) / grad,
        grad * 1e6,
        g
    );
}

// ---- Phase 3.7: hybrids on real models ----------------------------------------------

/// A model from tests/targets with its optional data, built once.
struct ModelRef {
    lib: Arc<StanLibrary>,
    data: Option<PathBuf>,
}

fn model_ref(name: &str) -> ModelRef {
    let stan = targets_dir().join(format!("{name}.stan"));
    let prepared = model::prepare(&stan).expect("build model");
    let json = targets_dir().join(format!("{name}.json"));
    ModelRef {
        lib: Arc::new(bridgestan::open_library(&prepared.library).unwrap()),
        data: json.exists().then_some(json),
    }
}

/// Constrained draws (parameters and transformed parameters) by name, per
/// chain, and the sampling-phase gradients.
struct Constrained {
    names: Vec<String>,
    /// cols[k][c][i]: quantity k, chain c, kept draw i
    cols: Vec<Vec<Vec<f64>>>,
    cost: Cost,
}

fn run_constrained(
    m: &ModelRef,
    sampler: &laplace_sample::runner::Sampler,
    n: Counts,
    seed: u64,
) -> Constrained {
    let per_chain: Vec<(Vec<String>, Vec<Vec<f64>>, Cost)> = std::thread::scope(|s| {
        let hs: Vec<_> = (1..=n.chains)
            .map(|chain| {
                s.spawn(move || {
                    let data = m
                        .data
                        .as_ref()
                        .map(|p| CString::new(p.to_str().unwrap()).unwrap());
                    let model = Model::new(m.lib.clone(), data, seed as u32).unwrap();
                    let names: Vec<String> = model
                        .param_names(true, false)
                        .split(',')
                        .map(str::to_owned)
                        .collect();
                    let params = laplace_sample::sample::model_params(&model).unwrap();
                    let target = Arc::new(BridgeStanTarget { model });
                    let dyn_t: Arc<dyn Target> = target.clone();
                    let mut built = sampler
                        .build_chain(dyn_t, &params, n.warmup, n.draws)
                        .unwrap();
                    let mut out = vec![0.0; names.len()];
                    let mut cols: Vec<Vec<f64>> = vec![vec![]; names.len()];
                    let (mut i, mut cost) = (0u64, Cost::default());
                    let sampling = std::cell::Cell::new(None);
                    run_chain(
                        target.as_ref(),
                        built.kernel.as_mut(),
                        &Init::Radius(2.0),
                        n.warmup,
                        n.draws,
                        &mut chain_rng(seed, chain),
                        |_| {
                            sampling.set(Some(std::time::Instant::now()));
                            Ok(())
                        },
                        |st, info, _| {
                            cost.add_info(info);
                            if i.is_multiple_of(n.thin) {
                                target.model.param_constrain::<&StanLibrary>(
                                    &st.q, true, false, &mut out, None,
                                )?;
                                for (k, v) in out.iter().enumerate() {
                                    cols[k].push(*v);
                                }
                            }
                            i += 1;
                            Ok(())
                        },
                    )
                    .unwrap();
                    cost.secs = sampling.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
                    (names, cols, cost)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let names = per_chain[0].0.clone();
    let mut cols: Vec<Vec<Vec<f64>>> = vec![vec![]; names.len()];
    let mut cost = Cost::default();
    for (_, c, g) in per_chain {
        for (k, col) in c.into_iter().enumerate() {
            cols[k].push(col);
        }
        cost.merge(g);
    }
    Constrained { names, cols, cost }
}

/// Truth for E[x] and E[x^2] by quantity name: exact, or a reference with
/// its MCSEs.
enum Truth {
    Exact(f64, f64),
    Reference { mean: (f64, f64), m2: (f64, f64) },
}

fn truth(target: &str, name: &str) -> Truth {
    match target {
        "corr_gauss_split" => {
            let v: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(targets_dir().join("corr_gauss_split.json")).unwrap(),
            )
            .unwrap();
            let (p, k) = name.split_once('.').unwrap();
            let i = k.parse::<usize>().unwrap() - 1 + if p == "b" { 5 } else { 0 };
            Truth::Exact(0.0, v["Sigma"][i][i].as_f64().unwrap())
        }
        "simplex_hybrid" => {
            let alpha = [2.0, 3.0, 4.0, 5.0];
            let total: f64 = alpha.iter().sum();
            let (p, k) = name.split_once('.').unwrap();
            let i = k.parse::<usize>().unwrap() - 1;
            let e = alpha[i] / total;
            let e2 = alpha[i] * (alpha[i] + 1.0) / (total * (total + 1.0));
            match p {
                "theta" => Truth::Exact(e, e2),
                _ => Truth::Exact(2.0 * e, 1.0 + 4.0 * e2),
            }
        }
        "eight_schools_nc" => {
            let v: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(targets_dir().join("eight_schools_nc.reference.json"))
                    .unwrap(),
            )
            .unwrap();
            let stan_name = match name.split_once('.') {
                Some((p, k)) => format!("{p}[{k}]"),
                None => name.to_string(),
            };
            let q = v["quantities"]
                .as_array()
                .unwrap()
                .iter()
                .find(|q| q["name"] == stan_name.as_str())
                .unwrap_or_else(|| panic!("no reference for {stan_name}"));
            let f = |k: &str| q[k].as_f64().unwrap();
            Truth::Reference {
                mean: (f("mean"), f("mcse_mean")),
                m2: (f("m2"), f("mcse_m2")),
            }
        }
        t => panic!("no truth for {t}"),
    }
}

fn hybrid_moments(target: &str, c: &Constrained) -> Vec<Moment> {
    use laplace_sample::diagnostics::moment_vs_reference;
    let mut rows = vec![];
    for (k, name) in c.names.iter().enumerate() {
        let x = &c.cols[k];
        let x2: Vec<Vec<f64>> = x
            .iter()
            .map(|ch| ch.iter().map(|v| v * v).collect())
            .collect();
        match truth(target, name) {
            Truth::Exact(m, m2) => {
                rows.push(moment(&format!("E[{name}]"), x, m));
                rows.push(moment(&format!("E[{name}^2]"), &x2, m2));
            }
            Truth::Reference { mean, m2 } => {
                rows.push(moment_vs_reference(
                    &format!("E[{name}]"),
                    x,
                    mean.0,
                    mean.1,
                ));
                rows.push(moment_vs_reference(
                    &format!("E[{name}^2]"),
                    &x2,
                    m2.0,
                    m2.1,
                ));
            }
        }
    }
    rows
}

fn hybrid_counts(sampler: &str) -> (String, Counts) {
    let c = config();
    let e = c
        .hybrid
        .iter()
        .find(|h| h.sampler == sampler)
        .unwrap_or_else(|| panic!("tests/statistical.toml has no [[hybrid]] for {sampler}"));
    let d = c.defaults;
    (
        e.target.clone(),
        Counts {
            chains: e.chains.unwrap_or(d.chains),
            warmup: e.warmup.unwrap_or(d.warmup),
            draws: e.draws.unwrap_or(d.draws),
            thin: e.thin.unwrap_or(d.thin),
        },
    )
}

/// Prints one RESULT line; returns the verdict, min bulk ESS and the cost.
fn summarize(label: &str, rows: &[Moment], cost: Cost) -> (Verdict, f64, Cost) {
    let v = verdict(rows);
    let min_ess = rows
        .iter()
        .map(|m| m.ess_bulk)
        .fold(f64::INFINITY, f64::min);
    let max_rhat = rows.iter().map(|m| m.rhat).fold(0.0, f64::max);
    let max_z = rows.iter().map(|m| m.z.abs()).fold(0.0, f64::max);
    println!(
        "RESULT {label} max|z|={max_z:.2} max_rhat={max_rhat:.4} min_ess_bulk={min_ess:.0} {} verdict={}",
        cost.report(min_ess),
        v.label()
    );
    (v, min_ess, cost)
}

/// 3.7: a .lapsampler hybrid on its model passes the health-gated moment
/// check; plain NUTS on the same model, same draw counts, for cost.
fn check_hybrid(sampler_file: &str) {
    use laplace_sample::runner::Sampler;
    let (target, n) = hybrid_counts(sampler_file);
    let m = model_ref(&target);
    let path = targets_dir().join(sampler_file);
    let hybrid = Sampler::from_arg(path.to_str().unwrap(), &[]).unwrap();
    let c = run_constrained(&m, &hybrid, n, SEED);
    let rows = hybrid_moments(&target, &c);
    let label = format!(
        "hybrid={sampler_file} target={target} chains={} warmup={} draws={} thin={}",
        n.chains, n.warmup, n.draws, n.thin
    );
    let (v, ess, cost) = summarize(&label, &rows, c.cost);
    // Plain NUTS on the whole model, for the comparison (Q15: reported,
    // never a threshold).
    let nuts = Sampler::from_arg("nuts", &[]).unwrap();
    let cn = run_constrained(&m, &nuts, n, SEED);
    let (_, ess_n, cost_n) = summarize(
        &format!("nuts target={target} (comparison)"),
        &hybrid_moments(&target, &cn),
        cn.cost,
    );
    let ratio = |a: f64, b: f64| {
        if a.is_finite() && b > 0.0 {
            format!("{:.2}", a / b)
        } else {
            "n/a".into()
        }
    };
    println!(
        "RESULT ratio {sampler_file} / nuts on {target}: ess_per_grad {} ess_per_s {}",
        ratio(ess / cost.grads as f64, ess_n / cost_n.grads as f64),
        ratio(cost.ess_per_s(ess), cost_n.ess_per_s(ess_n)),
    );
    assert_eq!(
        v,
        Verdict::Pass,
        "{sampler_file} on {target}: {}",
        describe(&rows)
    );
}

#[test]
#[ignore]
fn hybrid_nuts_mala_on_the_split_correlated_gaussian() {
    check_hybrid("hybrid_corr.lapsampler");
}

#[test]
#[ignore]
fn hybrid_nuts_mala_on_eight_schools() {
    check_hybrid("hybrid_eight_schools.lapsampler");
}

/// Extra check (Phase 3 review): a constrained parameter (simplex[4], 3
/// unconstrained coordinates) inside a block, run by a .lapkernel.
#[test]
#[ignore]
fn hybrid_with_a_simplex_block() {
    check_hybrid("hybrid_simplex.lapsampler");
}

// ---- Phase 4 cost accounting ------------------------------------------------------

/// Report only: BridgeStan time per density-only evaluation against per
/// gradient evaluation, on the shipped models and the gp library model.
/// Density-only with `propto = true` still runs Stan's forward pass on
/// autodiff variables and skips the reverse sweep, so the ratio is between
/// "about the same" and "much cheaper" depending on the model.
#[test]
#[ignore]
fn density_only_against_gradient_cost() {
    use std::time::Instant;
    let gp = Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/models");
    let models: Vec<(String, PathBuf, Option<PathBuf>)> = vec![
        (
            "corr_gauss_10".into(),
            targets_dir().join("corr_gauss_10.stan"),
            Some(targets_dir().join("corr_gauss_10.json")),
        ),
        (
            "eight_schools_nc".into(),
            targets_dir().join("eight_schools_nc.stan"),
            Some(targets_dir().join("eight_schools_nc.json")),
        ),
        ("gp".into(), gp.join("gp.stan"), Some(gp.join("gp.json"))),
    ];
    for (name, stan, data) in models {
        let prepared = model::prepare(&stan).unwrap();
        let lib = Arc::new(bridgestan::open_library(&prepared.library).unwrap());
        let data = data.map(|p| CString::new(p.to_str().unwrap()).unwrap());
        let t = BridgeStanTarget {
            model: Model::new(lib, data, 1).unwrap(),
        };
        let d = t.dim();
        let q: Vec<f64> = (0..d).map(|i| 0.1 * (i as f64 % 5.0) - 0.2).collect();
        let mut g = vec![0.0; d];
        let n = if name == "gp" { 2000 } else { 200_000 };
        let time = |f: &mut dyn FnMut()| {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..n {
                    f();
                }
                best = best.min(t0.elapsed().as_secs_f64() / n as f64);
            }
            best
        };
        let lp_a = t.log_density(&q).unwrap();
        let lp_b = t.log_density_gradient(&q, &mut g).unwrap();
        assert_eq!(lp_a.to_bits(), lp_b.to_bits(), "same value, bit for bit");
        let logp = time(&mut || {
            std::hint::black_box(t.log_density(&q).unwrap());
        });
        let grad = time(&mut || {
            std::hint::black_box(t.log_density_gradient(&q, &mut g).unwrap());
        });
        println!(
            "RESULT cost {name} (dim {d}): density-only {:.2} us, gradient {:.2} us, ratio {:.2}",
            logp * 1e6,
            grad * 1e6,
            logp / grad
        );
    }
}
