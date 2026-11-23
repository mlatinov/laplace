//! Random-walk Metropolis with a diagonal metric.
//!
//! Proposal: `q' = q + eps * sqrt(m) * z`, `z ~ N(0, I)`. Symmetric, so
//! `log_alpha = lp(q') - lp(q)`.
//!
//! RNG call order per step:
//! 1. `dim` normal draws `z[0..dim]`, in index order;
//! 2. one uniform in [`mh_accept`].
//!
//! RWM does not use the gradient, but it evaluates the target through the
//! same evaluator as every other kernel (density *and* gradient), so the
//! chain state stays consistent for composition in Phase 3. With
//! `propto = true` BridgeStan needs autodiff for the density anyway, so the
//! extra cost is small (Phase 0.6: 169 vs 205 us on the gp model).

use super::{check_metric, check_step_size, grads_since, metric_or_unit, normals};
use crate::kernel::{mh_accept, ChainRng, Kernel, StepInfo, Tuning};
use crate::state::{ChainState, Evaluator};

pub struct Rwm {
    pub step_size: f64,
    pub inv_metric: Option<Vec<f64>>,
}

impl Rwm {
    pub fn new(step_size: f64) -> Self {
        Self {
            step_size,
            inv_metric: None,
        }
    }
}

impl Kernel for Rwm {
    fn name(&self) -> &'static str {
        "rwm"
    }
    fn source(&self) -> &'static str {
        include_str!("rwm.rs")
    }

    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let before = ev.n_grad();
        let d = state.q.len();
        let m = metric_or_unit(&self.inv_metric, d);
        let z = normals(rng, d);
        let q_new: Vec<f64> = (0..d)
            .map(|i| state.q[i] + self.step_size * m[i].sqrt() * z[i])
            .collect();
        let (lp_new, g_new) = ev.eval(&q_new)?;
        let log_alpha = lp_new - state.lp;
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
