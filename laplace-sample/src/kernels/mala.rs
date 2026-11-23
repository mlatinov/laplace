//! Metropolis-adjusted Langevin algorithm with a diagonal metric.
//!
//! With inverse metric `m` and step size `eps`:
//! ```text
//! mu_fwd = q  + 0.5 * eps^2 * m * grad(q)
//! q'     = mu_fwd + eps * sqrt(m) * z,          z ~ N(0, I)
//! mu_bwd = q' + 0.5 * eps^2 * m * grad(q')
//! log_alpha = lp(q') - lp(q)
//!           + log N(q  | mu_bwd, eps^2 m) - log N(q' | mu_fwd, eps^2 m)
//! ```
//! The Gaussian normalising constants cancel (same covariance both ways), so
//! only the quadratic forms are computed. With `m = 1` this is exactly the
//! `mala.lapkernel` example in `docs/sampler/laplace_kernels.md` (Phase 2.1).
//!
//! RNG call order per step:
//! 1. `dim` normal draws `z[0..dim]`, in index order;
//! 2. one uniform in [`mh_accept`].

use super::{check_metric, check_step_size, grads_since, metric_or_unit, normals};
use crate::kernel::{mh_accept, ChainRng, Kernel, StepInfo, Tuning};
use crate::state::{ChainState, Evaluator};

pub struct Mala {
    pub step_size: f64,
    pub inv_metric: Option<Vec<f64>>,
}

impl Mala {
    pub fn new(step_size: f64) -> Self {
        Self {
            step_size,
            inv_metric: None,
        }
    }
}

/// `sum_i (x_i - mu_i)^2 / (2 eps^2 m_i)`, the negative Gaussian log kernel.
fn quad(x: &[f64], mu: &[f64], eps: f64, m: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() {
        let r = x[i] - mu[i];
        s += r * r / (2.0 * eps * eps * m[i]);
    }
    s
}

impl Kernel for Mala {
    fn name(&self) -> &'static str {
        "mala"
    }
    fn source(&self) -> &'static str {
        include_str!("mala.rs")
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
        let mu_fwd: Vec<f64> = (0..d)
            .map(|i| state.q[i] + 0.5 * eps * eps * m[i] * state.grad[i])
            .collect();
        let z = normals(rng, d);
        let q_new: Vec<f64> = (0..d)
            .map(|i| mu_fwd[i] + eps * m[i].sqrt() * z[i])
            .collect();
        let (lp_new, g_new) = ev.eval(&q_new)?;
        let log_alpha = if lp_new == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            let mu_bwd: Vec<f64> = (0..d)
                .map(|i| q_new[i] + 0.5 * eps * eps * m[i] * g_new[i])
                .collect();
            lp_new - state.lp - quad(&state.q, &mu_bwd, eps, &m) + quad(&q_new, &mu_fwd, eps, &m)
        };
        let (accepted, p) = mh_accept(log_alpha, rng);
        if accepted {
            *state = ChainState {
                q: q_new,
                lp: lp_new,
                grad: g_new,
            };
        }
        Ok(StepInfo {
            accept_prob: Some(p),
            n_grad: grads_since(ev, before),
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
