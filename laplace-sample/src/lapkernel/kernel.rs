//! A `.lapkernel` file as a [`Kernel`] (Phase 2.4).

use std::borrow::Cow;
use std::sync::Arc;

use nalgebra::DVector;

use super::check::{AdaptKind, Checked};
use super::interp::{Interp, Outcome};
use super::natives::Ctx;
use super::types::Value;
use crate::kernel::{mh_accept, ChainRng, Cost, Kernel, StepInfo, Tuning};
use crate::state::{ChainState, Evaluator};

/// Parsed and checked in laplace-sampler-syntax; re-exported at its old path.
pub use super::file::LapKernelFile;

/// A checked kernel with its tuning and state values: what one chain runs.
pub struct InterpretedKernel {
    file: Arc<LapKernelFile>,
    /// Leaked once per kernel instance: `Kernel::name` returns `&'static str`.
    name: &'static str,
    interp: Interp,
    dim: usize,
    tuning: Vec<Value>,
    state: Vec<Value>,
}

impl InterpretedKernel {
    /// Check `file` for a block of `dim` coordinates and set the tuning
    /// values to their defaults, then apply `sets` (name, value) on top.
    pub fn new(
        file: Arc<LapKernelFile>,
        dim: usize,
        sets: &[(String, f64)],
    ) -> anyhow::Result<Self> {
        let checked = Arc::new(file.check(Some(dim))?);
        let interp = Interp::new(checked);
        let tuning = interp
            .tuning_defaults(dim)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let name: &'static str = Box::leak(file.name().to_string().into_boxed_str());
        let mut k = Self {
            file,
            name,
            interp,
            dim,
            tuning,
            state: vec![],
        };
        for (name, value) in sets {
            k.set_named(name, *value)?;
        }
        Ok(k)
    }

    fn checked(&self) -> &Checked {
        &self.interp.checked
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.checked().tuning.iter().position(|t| t.name == name)
    }

    /// `--set name=value` for a scalar tuning value (bounds checked).
    pub fn set_named(&mut self, name: &str, value: f64) -> anyhow::Result<()> {
        let i = self.index_of(name).ok_or_else(|| {
            anyhow::anyhow!(
                "`{}` has no tuning value `{name}`; its tuning values are: {}",
                self.file.path,
                self.file.settable().join(", ")
            )
        })?;
        let v = match self.checked().tuning[i].ty {
            super::types::Ty::Int if value.fract() == 0.0 => Value::Int(value as i64),
            super::types::Ty::Int => anyhow::bail!("`{name}` is an int, got {value}"),
            super::types::Ty::Real => Value::Real(value),
            ref t => anyhow::bail!(
                "`{name}` is a {t}; only int and real tuning values can be set with --set"
            ),
        };
        self.tuning[i] = self
            .interp
            .check_tuning(i, &v, self.dim)
            .map_err(|m| anyhow::anyhow!("{}: {m}", self.file.path))?;
        Ok(())
    }

    /// The current value of a scalar tuning value.
    pub fn get_named(&self, name: &str) -> Option<f64> {
        self.index_of(name).and_then(|i| self.tuning[i].as_real())
    }

    pub fn target_accept(&self) -> Option<f64> {
        self.checked().target_accept
    }

    pub fn adapts(&self, kind: AdaptKind) -> bool {
        match kind {
            AdaptKind::StepSize => self.checked().step_size.is_some(),
            AdaptKind::Metric => self.checked().metric.is_some(),
        }
    }

    fn metric_value(&self) -> Option<DVector<f64>> {
        self.checked()
            .metric
            .and_then(|i| self.tuning[i].as_vector().cloned())
    }
}

impl Kernel for InterpretedKernel {
    fn name(&self) -> &'static str {
        self.name
    }

    fn source(&self) -> Cow<'static, str> {
        Cow::Owned(self.file.source.clone())
    }

    fn version(&self) -> String {
        "local".to_string()
    }

    fn adaptable(&self) -> crate::kernel::Adaptable {
        let (step_size, metric) = (
            self.adapts(AdaptKind::StepSize),
            self.adapts(AdaptKind::Metric),
        );
        crate::kernel::Adaptable {
            step_size,
            metric,
            missing: format!(
                "{} has nothing to adapt: tag its step size @adapt(step_size, target_accept = ...) \
                 and/or its inverse metric @adapt(metric)",
                self.file.path
            ),
        }
    }

    fn default_target_accept(&self) -> Option<f64> {
        self.target_accept()
    }

    fn init(
        &mut self,
        state: &ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        let metric = self.metric_value();
        let mut state = state.clone();
        if self.interp.checked.uses_grad {
            state.grad(ev)?;
        }
        let mut ctx = Ctx {
            dim: self.dim,
            state: Some(&state),
            ev: Some(ev),
            rng: Some(rng),
            metric: metric.as_ref(),
        };
        self.state = self
            .interp
            .init_state(&mut ctx, &self.tuning)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let before = Cost::now(ev);
        let metric = self.metric_value();
        if self.interp.checked.uses_grad {
            state.grad(ev)?;
        }
        let outcome = {
            let mut ctx = Ctx {
                dim: self.dim,
                state: Some(&*state),
                ev: Some(&mut *ev),
                rng: Some(&mut *rng),
                metric: metric.as_ref(),
            };
            self.interp
                .step(&mut ctx, &self.tuning, &mut self.state)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let accept_prob = match outcome {
            Outcome::Propose {
                q_new,
                log_correction,
            } => {
                let q_new: Vec<f64> = q_new.iter().copied().collect();
                // Usually a free cache hit: the kernel evaluated q_new
                // itself. Density only: a gradient is computed (or found in
                // the cache) only when a step that reads `grad` asks.
                let lp_new = ev.eval_lp(&q_new)?;
                let log_alpha = if lp_new == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    lp_new - state.lp + log_correction
                };
                let (accepted, p) = mh_accept(log_alpha, rng);
                if accepted {
                    *state = ChainState {
                        q: q_new,
                        lp: lp_new,
                        grad: None,
                    };
                }
                Some(p)
            }
            Outcome::Move { q_new } => {
                let q_new: Vec<f64> = q_new.iter().copied().collect();
                let lp_new = ev.eval_lp(&q_new)?;
                anyhow::ensure!(
                    lp_new.is_finite(),
                    "{}: move() to a point where the log density is {lp_new}",
                    self.file.path
                );
                *state = ChainState {
                    q: q_new,
                    lp: lp_new,
                    grad: None,
                };
                None
            }
        };
        Ok(StepInfo {
            accept_prob,
            ..before.since(ev)
        })
    }

    fn tuning(&self) -> Tuning {
        Tuning {
            step_size: self
                .checked()
                .step_size
                .and_then(|i| self.tuning[i].as_real()),
            inv_metric: self.metric_value().map(|m| m.iter().copied().collect()),
        }
    }

    fn set_tuning(&mut self, t: &Tuning) -> anyhow::Result<()> {
        if let Some(eps) = t.step_size {
            let i = self.checked().step_size.ok_or_else(|| {
                anyhow::anyhow!("{} has no @adapt(step_size) value", self.file.path)
            })?;
            self.tuning[i] = self
                .interp
                .check_tuning(i, &Value::Real(eps), self.dim)
                .map_err(|m| anyhow::anyhow!("{}: adapted step size: {m}", self.file.path))?;
        }
        if let Some(m) = &t.inv_metric {
            let i = self
                .checked()
                .metric
                .ok_or_else(|| anyhow::anyhow!("{} has no @adapt(metric) value", self.file.path))?;
            let v = Value::Vector(DVector::from_column_slice(m));
            self.tuning[i] = self
                .interp
                .check_tuning(i, &v, self.dim)
                .map_err(|m| anyhow::anyhow!("{}: adapted metric: {m}", self.file.path))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{chain_rng, run_chain, Init, Sampler};
    use crate::target::analytic::DiagGaussian;
    use crate::target::Target;
    use std::path::Path;

    fn shipped(name: &str) -> String {
        format!("{}/kernels/{name}.lapkernel", env!("CARGO_MANIFEST_DIR"))
    }

    fn target() -> Arc<dyn Target> {
        Arc::new(DiagGaussian {
            mu: vec![1.0, -2.0, 0.5],
            sd: vec![1.0, 3.0, 0.2],
        })
    }

    /// Run a sampler; returns (tuning before each saved step, step infos).
    fn run(s: &Sampler, warmup: u64, draws: u64) -> Vec<(Tuning, StepInfo)> {
        let t = target();
        let mut k = s.build(t.clone(), warmup, draws).unwrap();
        let mut out = vec![];
        run_chain(
            t.as_ref(),
            k.as_mut(),
            &Init::Radius(2.0),
            warmup,
            draws,
            &mut chain_rng(7, 1),
            |_| Ok(()),
            |_, info, tuning| {
                out.push((tuning.clone(), info.clone()));
                Ok(())
            },
        )
        .unwrap();
        out
    }

    fn sets(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn shipped_kernels_load_and_check() {
        for name in ["rwm", "mala"] {
            let f = LapKernelFile::load(Path::new(&shipped(name))).unwrap();
            assert_eq!(f.name(), name);
            let c = f.check(Some(5)).unwrap();
            assert!(c.step_size.is_some() && c.metric.is_some());
            assert_eq!(f.settable(), vec!["eps".to_string()]);
        }
    }

    #[test]
    fn set_fixes_tuning_without_adaptation() {
        let s = Sampler::from_arg(&shipped("mala"), &sets(&["eps=0.05", "adapt=false"])).unwrap();
        let out = run(&s, 30, 20);
        assert!(out.iter().all(|(t, _)| t.step_size == Some(0.05)));
        assert!(out.iter().all(|(t, _)| t.inv_metric == Some(vec![1.0; 3])));
        assert!(out
            .iter()
            .all(|(_, i)| i.accept_prob.is_some() && i.n_grad == 1));
    }

    #[test]
    fn default_adaptation_follows_the_tags_and_freezes() {
        let s = Sampler::from_arg(&shipped("mala"), &[]).unwrap();
        let out = run(&s, 300, 100);
        let first = &out[0].0;
        assert_ne!(first.step_size, Some(1.0), "step size adapted in warmup");
        assert_ne!(
            first.inv_metric,
            Some(vec![1.0; 3]),
            "metric adapted in warmup"
        );
        assert!(
            out.iter().all(|(t, _)| t == first),
            "tuning frozen after warmup"
        );
    }

    #[test]
    fn set_errors_name_the_problem() {
        let e = Sampler::from_arg(&shipped("mala"), &sets(&["step=0.1"])).unwrap_err();
        assert!(
            e.to_string()
                .contains("has no setting `step`; its settings are: eps, adapt, target_accept"),
            "{e}"
        );
        let e = Sampler::from_arg(&shipped("mala"), &sets(&["m=1"])).unwrap_err();
        assert!(e.to_string().contains("has no setting `m`"), "{e}");
        let s = Sampler::from_arg(&shipped("mala"), &sets(&["eps=-1"])).unwrap();
        let e = s.build(target(), 10, 10).err().unwrap();
        assert!(e.to_string().contains("`eps` must be >= 0, got -1"), "{e}");
    }

    #[test]
    fn a_direct_kernel_ending_in_move() {
        let f = LapKernelFile::from_source(
            "walk.lapkernel",
            "kernel walk { tuning { real<lower=0> h = 0.1; } step { move(q + h * std_normal_rng(dim)); } }".into(),
        )
        .unwrap();
        let s = Sampler::File(crate::runner::FileSampler {
            file: Arc::new(f),
            sets: vec![],
            adapt: true,
            target_accept: None,
        });
        let out = run(&s, 0, 10);
        assert!(out
            .iter()
            .all(|(t, i)| i.accept_prob.is_none() && t.step_size.is_none()));
    }

    #[test]
    fn a_file_that_does_not_check_is_rejected_up_front() {
        let e =
            LapKernelFile::from_source("bad.lapkernel", "kernel b { step { real x = 1; } }".into())
                .unwrap_err();
        assert!(
            e.to_string()
                .starts_with("bad.lapkernel:1:17: step { } never calls"),
            "{e}"
        );
    }
}
