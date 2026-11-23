//! Static Hamiltonian Monte Carlo: leapfrog with a diagonal metric and no
//! U-turn criterion, then a Metropolis correction.
//!
//! Path length: `steps` leapfrog steps, or with `jitter` (the default) a
//! number drawn uniformly from `1..=2*steps-1` each iteration (mean `steps`).
//! A fixed length can land near a whole number of periods of the dynamics
//! and barely move: on the 10-d standard normal, 10 fixed steps at the
//! adapted step size gave R-hat 1.36 and ESS 10 (Phase 1 report). Drawing
//! the length independently of the state keeps the kernel exact (a mixture
//! of valid HMC kernels).
//!
//! With inverse metric `m` (so the mass matrix is `M = diag(1/m)`):
//! ```text
//! p ~ N(0, M):  p_i = z_i / sqrt(m_i)
//! H(q, p) = -lp(q) + 0.5 * sum_i m_i p_i^2
//! leapfrog x steps:  p += eps/2 grad;  q += eps m p;  p += eps/2 grad
//! log_alpha = H(start) - H(end)
//! ```
//! A trajectory is *divergent* when `H(end) - H(start) > 1000` or the density
//! stops being finite (Stan's threshold); it is then rejected.
//!
//! RNG call order per step:
//! 1. with `jitter` only: one draw `rng.random_range(1..=2*steps-1)` (u64)
//!    for the number of leapfrog steps;
//! 2. `dim` normal draws `z[0..dim]` for the momentum, in index order;
//! 3. none during the leapfrog;
//! 4. one uniform in [`mh_accept`].
//!
//! Reports `n_leapfrog`, `divergent` and `energy` (the Hamiltonian at the
//! returned state; for a rejection, the starting one). `treedepth` does not
//! apply (NaN in the CSV).

use rand::RngExt;

use super::{check_metric, check_step_size, grads_since, metric_or_unit, normals};
use crate::kernel::{mh_accept, ChainRng, Kernel, StepInfo, Tuning};
use crate::state::{ChainState, Evaluator};

/// Stan's divergence threshold on the energy error.
pub const MAX_ENERGY_ERROR: f64 = 1000.0;

pub struct Hmc {
    pub step_size: f64,
    /// Leapfrog steps (the mean, with `jitter`).
    pub steps: u64,
    pub jitter: bool,
    pub inv_metric: Option<Vec<f64>>,
}

impl Hmc {
    pub fn new(step_size: f64, steps: u64) -> Self {
        Self {
            step_size,
            steps,
            jitter: true,
            inv_metric: None,
        }
    }
}

fn kinetic(p: &[f64], m: &[f64]) -> f64 {
    0.5 * p.iter().zip(m).map(|(p, m)| m * p * p).sum::<f64>()
}

impl Kernel for Hmc {
    fn name(&self) -> &'static str {
        "hmc"
    }
    fn source(&self) -> &'static str {
        include_str!("hmc.rs")
    }

    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let before = ev.n_grad();
        let d = state.q.len();
        let eps = self.step_size;
        let m = metric_or_unit(&self.inv_metric, d);
        let steps = if self.jitter && self.steps > 1 {
            rng.random_range(1..=2 * self.steps - 1)
        } else {
            self.steps
        };
        let z = normals(rng, d);
        let p0: Vec<f64> = (0..d).map(|i| z[i] / m[i].sqrt()).collect();
        let h0 = -state.lp + kinetic(&p0, &m);

        let mut q = state.q.clone();
        let mut p = p0;
        let mut lp = state.lp;
        let mut g = state.grad.clone();
        let mut broke = false;
        for _ in 0..steps {
            for i in 0..d {
                p[i] += 0.5 * eps * g[i];
                q[i] += eps * m[i] * p[i];
            }
            let (lp_new, g_new) = ev.eval(&q)?;
            lp = lp_new;
            g = g_new;
            if !lp.is_finite() {
                broke = true;
                break;
            }
            for i in 0..d {
                p[i] += 0.5 * eps * g[i];
            }
        }
        let h1 = if broke {
            f64::INFINITY
        } else {
            -lp + kinetic(&p, &m)
        };
        let divergent = broke || h1 - h0 > MAX_ENERGY_ERROR;
        let log_alpha = if divergent {
            f64::NEG_INFINITY
        } else {
            h0 - h1
        };
        let (accepted, prob) = mh_accept(log_alpha, rng);
        let energy = if accepted {
            *state = ChainState { q, lp, grad: g };
            h1
        } else {
            h0
        };
        Ok(StepInfo {
            accept_prob: Some(prob),
            n_grad: grads_since(ev, before),
            divergent: Some(divergent),
            energy: Some(energy),
            n_leapfrog: Some(steps),
            ..Default::default()
        })
    }

    fn tuning(&self) -> Tuning {
        Tuning {
            step_size: Some(self.step_size),
            inv_metric: self.inv_metric.clone(),
        }
    }

    fn set_tuning(&mut self, t: &Tuning) -> anyhow::Result<()> {
        if let Some(eps) = t.step_size {
            check_step_size(eps)?;
            self.step_size = eps;
        }
        if let Some(m) = &t.inv_metric {
            check_metric(m, m.len())?;
            self.inv_metric = Some(m.clone());
        }
        Ok(())
    }
}
