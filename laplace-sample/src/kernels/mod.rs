//! Built-in Rust kernels: the fast path (D3). Each documents its RNG call
//! order, because the Phase 2 `.lapkernel` versions must reproduce it draw for
//! draw (equivalence to 1e-10).
//!
//! Shared conventions:
//! - "normal draw" = `rng.sample(rand_distr::StandardNormal)`, one `f64`.
//! - The metric is diagonal; `m[i]` is the *inverse* metric (a variance
//!   scale). With `m = 1` every formula reduces to the textbook one.
//! - Every density evaluation goes through the [`Evaluator`], so `n_grad`
//!   is the number of real gradient evaluations in the step.

pub mod hmc;
pub mod mala;
pub mod nuts;
pub mod rwm;

use rand::RngExt;
use rand_distr::StandardNormal;

use crate::kernel::ChainRng;
use crate::state::Evaluator;

/// `dim` normal draws, in index order.
pub(crate) fn normals(rng: &mut ChainRng, dim: usize) -> Vec<f64> {
    (0..dim).map(|_| rng.sample(StandardNormal)).collect()
}

/// Inverse metric for `dim`, defaulting to the identity.
pub(crate) fn metric_or_unit(m: &Option<Vec<f64>>, dim: usize) -> Vec<f64> {
    match m {
        Some(v) => v.clone(),
        None => vec![1.0; dim],
    }
}

pub(crate) fn check_step_size(eps: f64) -> anyhow::Result<()> {
    anyhow::ensure!(
        eps.is_finite() && eps > 0.0,
        "step_size must be positive and finite, got {eps}"
    );
    Ok(())
}

pub(crate) fn check_metric(m: &[f64], dim: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        m.len() == dim,
        "inverse metric has length {}, model has {dim} unconstrained parameters",
        m.len()
    );
    anyhow::ensure!(
        m.iter().all(|x| x.is_finite() && *x > 0.0),
        "inverse metric entries must be positive and finite"
    );
    Ok(())
}

/// Gradient evaluations made since `before`.
pub(crate) fn grads_since(ev: &Evaluator, before: u64) -> u64 {
    ev.n_grad() - before
}
