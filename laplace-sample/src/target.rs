//! The density being sampled: log density and gradient on the unconstrained
//! scale.
//!
//! One convention everywhere: `propto = true, jacobian = true`, i.e. exactly
//! what Stan reports as `lp__`. A chain must never mix two conventions,
//! because a Metropolis ratio between an lp from one and an lp from the other
//! is off by an unknown constant (Phase 0 report, "Risks").

use std::sync::Arc;

use bridgestan::{Model, StanLibrary};

/// A log density on `R^dim`.
///
/// `Err` means the density could not be evaluated at all (a bug or a broken
/// model). A point outside the support, or a Stan `reject()`, is *not* an
/// error: it is `Ok(f64::NEG_INFINITY)`, which every kernel treats as a
/// rejected proposal.
pub trait Target: Send + Sync {
    fn dim(&self) -> usize;
    /// Writes the gradient into `grad` (length `dim`) and returns the log
    /// density. When the log density is `-inf` the gradient is unspecified.
    fn log_density_gradient(&self, q: &[f64], grad: &mut [f64]) -> anyhow::Result<f64>;
    /// The log density alone, for kernels that use no gradient. Must equal
    /// the value [`Target::log_density_gradient`] returns, bit for bit. The
    /// default computes the gradient and drops it.
    fn log_density(&self, q: &[f64]) -> anyhow::Result<f64> {
        let mut g = vec![0.0; self.dim()];
        self.log_density_gradient(q, &mut g)
    }
}

/// BridgeStan's error convention -> ours: domain errors and `reject()` mean
/// "this point has zero density", and so does a NaN.
fn bridgestan_lp(r: Result<f64, bridgestan::BridgeStanError>) -> anyhow::Result<f64> {
    match r {
        Ok(lp) if lp.is_nan() => Ok(f64::NEG_INFINITY),
        Ok(lp) => Ok(lp),
        Err(bridgestan::BridgeStanError::EvaluationFailed(_)) => Ok(f64::NEG_INFINITY),
        Err(e) => Err(e.into()),
    }
}

/// A BridgeStan model instance. Each chain gets its own instance built from
/// one shared library (Phase 0.3: safe and bitwise reproducible).
pub struct BridgeStanTarget {
    pub model: Model<Arc<StanLibrary>>,
}

impl Target for BridgeStanTarget {
    fn dim(&self) -> usize {
        self.model.param_unc_num()
    }

    fn log_density_gradient(&self, q: &[f64], grad: &mut [f64]) -> anyhow::Result<f64> {
        bridgestan_lp(self.model.log_density_gradient(q, true, true, grad))
    }

    /// BridgeStan's density-only call. With `propto = true` it still runs
    /// Stan's forward pass on autodiff variables (constants can only be
    /// dropped that way) but skips the reverse sweep: cheaper than a
    /// gradient, not free, and the same value.
    fn log_density(&self, q: &[f64]) -> anyhow::Result<f64> {
        bridgestan_lp(self.model.log_density(q, true, true))
    }
}

/// The log density of one block of a full target, the other coordinates
/// held at `rest` (Phase 3.1). Used where a kernel needs a target object of
/// its own (nuts-rs); kernels that go through the chain's [`Evaluator`] use
/// a [`crate::state::View`] instead.
///
/// `rest` is shared: whoever moves the other blocks writes the new full
/// position there before the block kernel runs again.
pub struct BlockTarget {
    pub full: Arc<dyn Target>,
    pub indices: Vec<usize>,
    pub rest: Arc<std::sync::Mutex<Vec<f64>>>,
    /// Real gradient evaluations through this block target.
    pub n_grad: Arc<std::sync::atomic::AtomicU64>,
}

impl BlockTarget {
    pub fn new(full: Arc<dyn Target>, indices: Vec<usize>, rest: Vec<f64>) -> Self {
        Self {
            full,
            indices,
            rest: Arc::new(std::sync::Mutex::new(rest)),
            n_grad: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
}

impl Target for BlockTarget {
    fn dim(&self) -> usize {
        self.indices.len()
    }

    fn log_density_gradient(&self, q: &[f64], grad: &mut [f64]) -> anyhow::Result<f64> {
        let mut full = self.rest.lock().expect("rest lock").clone();
        for (i, &j) in self.indices.iter().enumerate() {
            full[j] = q[i];
        }
        let mut g = vec![0.0; full.len()];
        let lp = self.full.log_density_gradient(&full, &mut g)?;
        self.n_grad
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for (i, &j) in self.indices.iter().enumerate() {
            grad[i] = g[j];
        }
        Ok(lp)
    }
}

/// Closed-form targets for fast unit tests (no BridgeStan, no compiler).
pub mod analytic {
    use super::Target;

    /// A zero-mean Gaussian given by its precision matrix (row-major):
    /// `lp = -x^T P x / 2`, `grad = -P x`. With off-diagonal entries between
    /// blocks, moving one block changes the other's gradient: the target
    /// the stale-gradient tests need (Phase 3.1).
    pub struct Gaussian {
        pub precision: Vec<Vec<f64>>,
    }

    impl Target for Gaussian {
        fn dim(&self) -> usize {
            self.precision.len()
        }
        fn log_density_gradient(&self, q: &[f64], grad: &mut [f64]) -> anyhow::Result<f64> {
            let mut lp = 0.0;
            for (i, row) in self.precision.iter().enumerate() {
                let px: f64 = row.iter().zip(q).map(|(p, x)| p * x).sum();
                grad[i] = -px;
                lp -= 0.5 * q[i] * px;
            }
            Ok(lp)
        }
    }

    /// Independent normals with means `mu` and standard deviations `sd`.
    pub struct DiagGaussian {
        pub mu: Vec<f64>,
        pub sd: Vec<f64>,
    }

    impl DiagGaussian {
        pub fn std_normal(dim: usize) -> Self {
            Self {
                mu: vec![0.0; dim],
                sd: vec![1.0; dim],
            }
        }
    }

    impl Target for DiagGaussian {
        fn dim(&self) -> usize {
            self.mu.len()
        }
        fn log_density_gradient(&self, q: &[f64], grad: &mut [f64]) -> anyhow::Result<f64> {
            let mut lp = 0.0;
            for i in 0..q.len() {
                let z = (q[i] - self.mu[i]) / self.sd[i];
                lp -= 0.5 * z * z;
                grad[i] = -z / self.sd[i];
            }
            Ok(lp)
        }
    }
}
