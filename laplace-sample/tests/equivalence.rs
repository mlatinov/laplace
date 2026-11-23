//! Phase 2.5: the `.lapkernel` twins of the built-in kernels
//! (`kernels/rwm.lapkernel`, `kernels/mala.lapkernel`) produce the same draws
//! as the Rust kernels to 1e-10, with the same seed. Two conditions each:
//!
//! (a) fixed tuning, no adaptation: Rust `--set step_size=0.3 adapt=false`,
//!     `.lapkernel` `--set eps=0.3 adapt=false`;
//! (b) Level 1's default adaptation on both (dual averaging + windowed
//!     diagonal metric, driven by the `@adapt` tags on the `.lapkernel` side).
//!
//! The Rust kernels' RNG call order (Phase 1.4) is the reference; the
//! `.lapkernel` files match it, never the other way round.
//!
//! These run on closed-form targets on every `cargo test`; the same checks
//! on real BridgeStan builds are in tests/statistical.rs (`--ignored`).

use std::sync::Arc;

use laplace_sample::runner::{chain_rng, run_chain, Init, Sampler};
use laplace_sample::target::analytic::DiagGaussian;
use laplace_sample::target::Target;

#[derive(Debug, Default)]
struct Trace {
    q: Vec<Vec<f64>>,
    lp: Vec<f64>,
    accept: Vec<f64>,
    n_grad: Vec<u64>,
    step_size: Vec<f64>,
    metric: Vec<Vec<f64>>,
}

fn kernel_path(name: &str) -> String {
    format!("{}/kernels/{name}.lapkernel", env!("CARGO_MANIFEST_DIR"))
}

fn trace(sampler: &Sampler, target: Arc<dyn Target>, seed: u64, chain: u64) -> Trace {
    let (warmup, draws) = (1000, 1000);
    let (mut k, a) = sampler.build(target.clone(), warmup, draws).unwrap();
    let mut t = Trace::default();
    run_chain(
        target.as_ref(),
        k.as_mut(),
        a,
        &Init::Radius(2.0),
        warmup,
        draws,
        &mut chain_rng(seed, chain),
        |_| Ok(()),
        |st, info, tuning| {
            t.q.push(st.q.clone());
            t.lp.push(st.lp);
            t.accept.push(info.accept_prob.unwrap());
            t.n_grad.push(info.n_grad);
            t.step_size.push(tuning.step_size.unwrap());
            t.metric.push(
                tuning
                    .inv_metric
                    .clone()
                    .unwrap_or_else(|| vec![1.0; st.q.len()]),
            );
            Ok(())
        },
    )
    .unwrap();
    t
}

/// Largest absolute difference over everything both chains report.
fn max_diff(a: &Trace, b: &Trace) -> f64 {
    assert_eq!(a.q.len(), b.q.len());
    assert_eq!(a.n_grad, b.n_grad, "gradient counts differ");
    let mut m: f64 = 0.0;
    let mut upd = |x: f64, y: f64| {
        let d = (x - y).abs();
        m = m.max(if d.is_nan() { f64::INFINITY } else { d });
    };
    for i in 0..a.q.len() {
        for (x, y) in a.q[i].iter().zip(&b.q[i]) {
            upd(*x, *y);
        }
        for (x, y) in a.metric[i].iter().zip(&b.metric[i]) {
            upd(*x, *y);
        }
        upd(a.lp[i], b.lp[i]);
        upd(a.accept[i], b.accept[i]);
        upd(a.step_size[i], b.step_size[i]);
    }
    m
}

/// Runs both samplers on 4 chains and returns the largest difference.
pub fn equivalence(rust: &Sampler, lap: &Sampler, target: Arc<dyn Target>) -> f64 {
    (1..=4)
        .map(|chain| {
            let a = trace(rust, target.clone(), 20261123, chain);
            let b = trace(lap, target.clone(), 20261123, chain);
            max_diff(&a, &b)
        })
        .fold(0.0, f64::max)
}

fn sets(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

fn targets() -> Vec<(&'static str, Arc<dyn Target>)> {
    vec![
        ("std normal 10", Arc::new(DiagGaussian::std_normal(10))),
        (
            "scaled 5",
            Arc::new(DiagGaussian {
                mu: vec![1.0, -2.0, 0.5, 3.0, 0.0],
                sd: vec![1.0, 5.0, 0.2, 0.5, 2.0],
            }),
        ),
    ]
}

fn check(name: &str, rust_sets: &[&str], lap_sets: &[&str]) {
    let rust = Sampler::from_arg(name, &sets(rust_sets)).unwrap();
    let lap = Sampler::from_arg(&kernel_path(name), &sets(lap_sets)).unwrap();
    for (label, t) in targets() {
        let d = equivalence(&rust, &lap, t);
        println!("EQUIV {name} {rust_sets:?} on {label}: max |difference| = {d:e}");
        assert!(d <= 1e-10, "{name} on {label}: max difference {d}");
    }
}

#[test]
fn rwm_fixed_tuning() {
    check(
        "rwm",
        &["step_size=0.3", "adapt=false"],
        &["eps=0.3", "adapt=false"],
    );
}

#[test]
fn rwm_default_adaptation() {
    check("rwm", &[], &[]);
}

#[test]
fn mala_fixed_tuning() {
    check(
        "mala",
        &["step_size=0.3", "adapt=false"],
        &["eps=0.3", "adapt=false"],
    );
}

#[test]
fn mala_default_adaptation() {
    check("mala", &[], &[]);
}
