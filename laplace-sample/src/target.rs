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
        match self.model.log_density_gradient(q, true, true, grad) {
            Ok(lp) if lp.is_nan() => Ok(f64::NEG_INFINITY),
            Ok(lp) => Ok(lp),
            // BridgeStan reports domain errors and `reject()` as evaluation
            // failures; both mean "this point has zero density".
            Err(bridgestan::BridgeStanError::EvaluationFailed(_)) => Ok(f64::NEG_INFINITY),
            Err(e) => Err(e.into()),
        }
    }
}

/// Closed-form targets for fast unit tests (no BridgeStan, no compiler).
pub mod analytic {
    use super::Target;

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
