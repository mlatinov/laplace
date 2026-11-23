//! The `Kernel` trait: one MCMC transition, plus the hooks adaptation needs.

use rand::rngs::ChaCha8Rng;
use rand::RngExt;

use crate::state::{ChainState, Evaluator};

/// The per-chain random stream. Every random number a kernel uses comes from
/// here, in an order the kernel documents (Phase 2 equivalence tests replay
/// it draw for draw).
pub type ChainRng = ChaCha8Rng;

/// What one step reports. These are the kernel's *statistics*: every field
/// that does not apply to a kernel is `None` and is written to the CSV as
/// `NaN` (Q4: `0` would falsely claim "checked, none found").
///
/// The plan's "Stats type" is this struct rather than an associated type, so
/// that `Box<dyn Kernel>` works: Phase 3 composes kernels chosen at run time.
/// Kernel-specific numbers go in `extra`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepInfo {
    /// Acceptance probability of this step (`accept_stat__`).
    pub accept_prob: Option<f64>,
    /// Real gradient evaluations this step (`n_grad__`).
    pub n_grad: u64,
    pub divergent: Option<bool>,
    /// Hamiltonian at the returned state (`energy__`).
    pub energy: Option<f64>,
    pub treedepth: Option<u64>,
    pub n_leapfrog: Option<u64>,
    pub extra: Vec<(&'static str, f64)>,
}

/// The values adaptation may change. Named after the `@adapt` tags of the
/// Phase 2 kernel language (`step_size`, `metric`).
#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    pub step_size: Option<f64>,
    /// Diagonal inverse metric (per-coordinate variance scale).
    pub inv_metric: Option<Vec<f64>>,
}

/// Not `Send`: a kernel is built and used inside its own chain's thread
/// (nuts-rs chains are `!Send`).
pub trait Kernel {
    /// Name used on the command line and in the CSV provenance line.
    fn name(&self) -> &'static str;

    /// The kernel's own source text, hashed into the CSV provenance line
    /// ("no hidden behaviour": every kernel that ran is readable source).
    fn source(&self) -> std::borrow::Cow<'static, str>;

    /// The version recorded in the provenance line: the laplace-sample
    /// version for built-ins, `local` for a `.lapkernel` file.
    fn version(&self) -> String {
        crate::VERSION.to_string()
    }

    /// Called once, after the initial state is set and before the first step.
    fn init(
        &mut self,
        _state: &ChainState,
        _ev: &mut Evaluator,
        _rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// One transition. Must leave `state` consistent (see [`ChainState`]).
    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo>;

    fn tuning(&self) -> Tuning;
    fn set_tuning(&mut self, tuning: &Tuning) -> anyhow::Result<()>;

    /// True when the kernel runs its own warmup adaptation (nuts-rs) and the
    /// runner must not wrap it in [`crate::adapt::Adapter`].
    fn adapts_itself(&self) -> bool {
        false
    }
}

/// The Metropolis–Hastings accept step shared by every proposal kernel.
///
/// RNG: draws **exactly one** `f64` uniform on `[0, 1)` (`rng.random::<f64>()`),
/// always, even when `log_alpha >= 0`, so the stream position after a step
/// never depends on the outcome. Accept iff `ln(u) < log_alpha`. A NaN
/// `log_alpha` rejects.
///
/// Returns `(accepted, accept_prob)` with `accept_prob = min(1, exp(log_alpha))`
/// (0 for NaN).
pub fn mh_accept(log_alpha: f64, rng: &mut ChainRng) -> (bool, f64) {
    let u: f64 = rng.random();
    if log_alpha.is_nan() {
        return (false, 0.0);
    }
    (u.ln() < log_alpha, log_alpha.min(0.0).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn mh_accept_always_consumes_one_uniform() {
        for la in [f64::NAN, -1e300, -0.5, 0.0, 3.0, f64::INFINITY] {
            let mut a = ChainRng::seed_from_u64(1);
            let mut b = ChainRng::seed_from_u64(1);
            mh_accept(la, &mut a);
            let _: f64 = b.random();
            let x: f64 = a.random();
            let y: f64 = b.random();
            assert_eq!(x, y, "log_alpha = {la}");
        }
    }

    #[test]
    fn mh_accept_probability() {
        let mut r = ChainRng::seed_from_u64(1);
        assert_eq!(mh_accept(2.0, &mut r), (true, 1.0));
        assert_eq!(mh_accept(f64::NAN, &mut r), (false, 0.0));
        assert_eq!(mh_accept(f64::NEG_INFINITY, &mut r), (false, 0.0));
        let (_, p) = mh_accept((0.25f64).ln(), &mut r);
        assert!((p - 0.25).abs() < 1e-15);
    }
}
