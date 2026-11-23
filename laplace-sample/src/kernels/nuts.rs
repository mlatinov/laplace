//! `nuts()`: upstream nuts-rs 0.19.1, unpatched, on the whole model (D8; Q2
//! deferred to Phase 3).
//!
//! nuts-rs owns its trajectory, its own random stream and its own warmup
//! adaptation (nutpie-style: a gradient-informed diagonal metric and
//! dual-averaged step size). It is seeded once from the chain stream when
//! the kernel is initialised; after that it draws nothing from the chain
//! stream. Its draws match CmdStan *in distribution*, not draw for draw
//! (Phase 0.5).
//!
//! Differences from the nuts-rs defaults, each to match Stan:
//! - step-size jitter off (Stan's `stepsize_jitter = 0`), so the step size is
//!   constant after warmup;
//! - `maxdepth = 10`.
//!
//! The position is set exactly once, at initialisation. `set_position` must
//! not be called again: in nuts-rs it re-runs the step-size search and
//! re-initialises the metric, which after warmup makes the kernel depend on
//! the current state and biases it (Phase 0.5).

use std::collections::HashMap;
use std::sync::Arc;

use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
};

use crate::kernel::{ChainRng, Kernel, StepInfo, Tuning};
use crate::state::{ChainState, Evaluator};
use crate::target::Target;

#[derive(Debug)]
pub struct NutsLogpError(String);

impl std::fmt::Display for NutsLogpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NutsLogpError {}
impl LogpError for NutsLogpError {
    fn is_recoverable(&self) -> bool {
        // Only genuine evaluation failures reach here (a zero-density point
        // is -inf, not an error); nuts-rs treats recoverable ones as a
        // divergence. Unrecoverable would abort the chain.
        true
    }
}

pub struct TargetLogp {
    target: Arc<dyn Target>,
}

impl HasDims for TargetLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [(
            "unconstrained_parameter".to_string(),
            self.target.dim() as u64,
        )]
        .into_iter()
        .collect()
    }
}

impl CpuLogpFunc for TargetLogp {
    type LogpError = NutsLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.target.dim()
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, NutsLogpError> {
        self.target
            .log_density_gradient(position, grad)
            .map_err(|e| NutsLogpError(e.to_string()))
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(array.to_vec())
    }
}

type NutsChainOf = <DiagNutsSettings as Settings>::Chain<CpuMath<TargetLogp>>;

pub struct Nuts {
    target: Arc<dyn Target>,
    settings: DiagNutsSettings,
    chain: Option<NutsChainOf>,
    step_size: Option<f64>,
    inv_metric: Option<Vec<f64>>,
}

impl Nuts {
    /// `num_tune` is the warmup length; nuts-rs adapts during exactly that
    /// many draws and freezes afterwards.
    pub fn new(target: Arc<dyn Target>, num_tune: u64, num_draws: u64) -> Self {
        let mut settings = DiagNutsSettings {
            num_tune,
            num_draws,
            maxdepth: 10,
            store_gradient: true,
            ..Default::default()
        };
        settings.adapt_options.step_size_settings.jitter = None;
        // Report the adapted metric (CmdStan writes it; cmdstanr's
        // `$inv_metric()` expects it).
        settings.adapt_options.mass_matrix_options.store_mass_matrix = true;
        Self {
            target,
            settings,
            chain: None,
            step_size: None,
            inv_metric: None,
        }
    }

    pub fn set_max_depth(&mut self, depth: u64) {
        self.settings.maxdepth = depth;
    }

    pub fn set_target_accept(&mut self, delta: f64) {
        self.settings.adapt_options.step_size_settings.target_accept = delta;
    }

    pub fn set_initial_step_size(&mut self, eps: f64) {
        self.settings.adapt_options.step_size_settings.initial_step = eps;
    }

    pub fn max_depth(&self) -> u64 {
        self.settings.maxdepth
    }

    pub fn target_accept(&self) -> f64 {
        self.settings.adapt_options.step_size_settings.target_accept
    }
}

impl Kernel for Nuts {
    fn name(&self) -> &'static str {
        "nuts"
    }
    fn source(&self) -> &'static str {
        include_str!("nuts.rs")
    }

    fn adapts_itself(&self) -> bool {
        true
    }

    fn init(
        &mut self,
        state: &ChainState,
        _ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        let logp = TargetLogp {
            target: self.target.clone(),
        };
        // nuts-rs seeds its own ChaCha from this stream (one draw sequence,
        // deterministic for a given chain seed).
        let mut chain = self.settings.new_chain(0, CpuMath::new(logp), rng)?;
        chain.set_position(&state.q)?;
        self.chain = Some(chain);
        Ok(())
    }

    fn step(
        &mut self,
        state: &mut ChainState,
        _ev: &mut Evaluator,
        _rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let chain = self
            .chain
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("nuts: step before init"))?;
        let (position, _expanded, stats, progress) = chain.expanded_draw()?;
        let kinetic_energy = stats.point.kinetic_energy;
        let grad = stats
            .point
            .gradient
            .clone()
            .ok_or_else(|| anyhow::anyhow!("nuts-rs did not store the gradient"))?;
        *state = ChainState {
            q: position.into_vec(),
            lp: stats.point.logp,
            grad,
        };
        self.step_size = Some(progress.step_size);
        // Emitted only on draws where nuts-rs changed its metric. The field
        // holds per-coordinate standard deviations; the diagonal inverse
        // metric (CmdStan's convention) is their square.
        if let Some(sd) = &stats.hamiltonian.transformation.mass_matrix_inv {
            self.inv_metric = Some(sd.iter().map(|s| s * s).collect());
        }
        Ok(StepInfo {
            accept_prob: Some(stats.adapt.step_size.mean_tree_accept),
            n_grad: progress.num_gradients,
            divergent: Some(progress.diverging),
            // Stan's energy__ is -lp + kinetic energy. nuts-rs's own
            // `energy` lives in its transformed space and is offset by the
            // log-determinant of the metric (harmless for E-BFMI, wrong as
            // a CmdStan column).
            energy: Some(kinetic_energy - stats.point.logp),
            treedepth: Some(stats.depth),
            n_leapfrog: Some(progress.num_steps),
            extra: vec![],
        })
    }

    fn tuning(&self) -> Tuning {
        Tuning {
            step_size: self.step_size,
            inv_metric: self.inv_metric.clone(),
        }
    }

    fn set_tuning(&mut self, t: &Tuning) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.chain.is_none(),
            "nuts() adapts its own step size and metric; tuning can only be set before it starts"
        );
        anyhow::ensure!(
            t.inv_metric.is_none(),
            "nuts() does not accept an initial metric"
        );
        if let Some(eps) = t.step_size {
            super::check_step_size(eps)?;
            self.set_initial_step_size(eps);
        }
        Ok(())
    }
}
