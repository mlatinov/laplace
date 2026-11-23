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
    /// Block mode (Phase 3.4): the block target's shared "rest" and its
    /// gradient counter.
    block: Option<BlockHandle>,
    /// Draws taken so far (to check the warmup length at end_warmup).
    draws_taken: u64,
}

struct BlockHandle {
    rest: Arc<std::sync::Mutex<Vec<f64>>>,
    n_grad: Arc<std::sync::atomic::AtomicU64>,
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
            block: None,
            draws_taken: 0,
        }
    }

    /// `nuts()` on one block of `full` (Phase 3.4). `num_tune` must be the
    /// number of times this kernel will run during warmup (warmup x uses
    /// per sweep): nuts-rs adapts for exactly that many draws.
    ///
    /// Each step first copies the other blocks' current values into the
    /// block target and calls nuts-rs's `refresh_position` (the vendored
    /// patch, Q2): the cached log density and gradient are recomputed under
    /// the moved blocks, and step size and metric are left alone. Upstream's
    /// `set_position` would re-initialise them, which after warmup makes the
    /// kernel depend on the state and biases it (Phase 0.5).
    pub fn for_block(
        full: Arc<dyn Target>,
        indices: Vec<usize>,
        num_tune: u64,
        num_draws: u64,
    ) -> Self {
        let dim = full.dim();
        let bt = crate::target::BlockTarget::new(full, indices, vec![0.0; dim]);
        let handle = BlockHandle {
            rest: bt.rest.clone(),
            n_grad: bt.n_grad.clone(),
        };
        let mut k = Self::new(Arc::new(bt), num_tune, num_draws);
        k.block = Some(handle);
        k
    }

    fn sync_rest(&self, ev: &Evaluator) -> anyhow::Result<()> {
        if let Some(b) = &self.block {
            let view = ev.view().ok_or_else(|| {
                anyhow::anyhow!("nuts() on a block must run inside on(block, ...)")
            })?;
            *b.rest.lock().expect("rest lock") = view.base.clone();
        }
        Ok(())
    }

    fn block_grads(&self) -> Option<u64> {
        self.block
            .as_ref()
            .map(|b| b.n_grad.load(std::sync::atomic::Ordering::Relaxed))
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
    fn source(&self) -> std::borrow::Cow<'static, str> {
        include_str!("nuts.rs").into()
    }

    fn adapts_itself(&self) -> bool {
        true
    }

    fn init(
        &mut self,
        state: &ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        self.sync_rest(ev)?;
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
        ev: &mut Evaluator,
        _rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let grads_before = self.block_grads();
        if self.block.is_some() {
            self.sync_rest(ev)?;
        }
        let in_block = self.block.is_some();
        let chain = self
            .chain
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("nuts: step before init"))?;
        if in_block {
            // The other blocks may have moved: recompute logp and gradient
            // at the current block position, keeping step size and metric.
            chain.refresh_position(&state.q)?;
        }
        self.draws_taken += 1;
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
        // Whole model: nuts-rs's own count. Block: every evaluation through
        // the block target, the refresh included.
        let n_grad = match (grads_before, self.block_grads()) {
            (Some(a), Some(b)) => b - a,
            _ => progress.num_gradients,
        };
        Ok(StepInfo {
            accept_prob: Some(stats.adapt.step_size.mean_tree_accept),
            n_grad,
            divergent: Some(progress.diverging),
            // Stan's energy__ is -lp + kinetic energy. nuts-rs's own
            // `energy` lives in its transformed space and is offset by the
            // log-determinant of the metric (harmless for E-BFMI, wrong as
            // a CmdStan column).
            energy: Some(kinetic_energy - stats.point.logp),
            treedepth: Some(stats.depth),
            n_leapfrog: Some(progress.num_steps),
            ..Default::default()
        })
    }

    fn end_warmup(&mut self) -> anyhow::Result<()> {
        let tune = self.settings.num_tune;
        anyhow::ensure!(
            self.draws_taken == tune,
            "nuts() was set up to adapt for {tune} warmup steps but ran {}; it would keep \
             adapting after warmup (nuts() cannot sit under mix(), whose step counts are random)",
            self.draws_taken
        );
        Ok(())
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

#[cfg(test)]
mod block_tests {
    use super::*;
    use crate::adapt::Adapt;
    use crate::blocks::{parameters, Block};
    use crate::combinators::{Mix, On, Seq};
    use crate::diagnostics::{moment, verdict, Verdict};
    use crate::kernels::mala::Mala;
    use crate::runner::{chain_rng, initial_state, Init};
    use crate::target::analytic::Gaussian;
    use nalgebra::DMatrix;

    /// Two 3-coordinate blocks, correlated through the precision matrix.
    fn coupled() -> (Arc<dyn Target>, Vec<f64>, Block, Block) {
        let mut p = vec![vec![0.0; 6]; 6];
        for (i, row) in p.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = if i == j {
                    2.0
                } else {
                    0.3 + 0.05 * (i + j) as f64
                };
            }
        }
        let cov = DMatrix::from_fn(6, 6, |i, j| p[i][j])
            .try_inverse()
            .unwrap();
        let names: Vec<String> = ["a.1", "a.2", "a.3", "b.1", "b.2", "b.3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ps = parameters(&names).unwrap();
        (
            Arc::new(Gaussian { precision: p }),
            (0..6).map(|i| cov[(i, i)]).collect(),
            Block::new(&ps, &["a".into()], None).unwrap(),
            Block::new(&ps, &["b".into()], None).unwrap(),
        )
    }

    fn hybrid(t: &Arc<dyn Target>, a: &Block, b: &Block, warmup: u64, draws: u64) -> Seq {
        Seq {
            items: vec![
                Box::new(On {
                    block: a.clone(),
                    inner: Box::new(Nuts::for_block(t.clone(), a.indices.clone(), warmup, draws)),
                }),
                Box::new(On {
                    block: b.clone(),
                    inner: Box::new(Adapt::new(Box::new(Mala::new(1.0)), None, warmup).unwrap()),
                }),
            ],
        }
    }

    /// 3.4: block(a): nuts() alongside block(b): adapt(mala()) passes the
    /// moment check, with healthy chains, on a target whose blocks are
    /// correlated; and no component re-tunes after warmup.
    #[test]
    fn nuts_as_a_block_beside_another_block() {
        let (t, var, a, b) = coupled();
        let (warmup, draws) = (1000, 2000);
        let mut xs: Vec<Vec<Vec<f64>>> = vec![vec![]; 6];
        for chain in 1..=4 {
            let mut k = hybrid(&t, &a, &b, warmup, draws);
            let mut ev = Evaluator::new(t.as_ref());
            let mut rng = chain_rng(21, chain);
            let mut st = initial_state(&Init::Radius(2.0), &mut ev, &mut rng).unwrap();
            k.init(&st, &mut ev, &mut rng).unwrap();
            for _ in 0..warmup {
                k.step(&mut st, &mut ev, &mut rng).unwrap();
            }
            k.end_warmup().unwrap();
            let frozen = k.tunings();
            assert!(
                frozen.iter().all(|(_, t)| t.step_size.is_some()),
                "{frozen:?}"
            );
            let mut cols: Vec<Vec<f64>> = vec![vec![]; 6];
            for _ in 0..draws {
                k.step(&mut st, &mut ev, &mut rng).unwrap();
                assert_eq!(k.tunings(), frozen, "re-tuned after warmup");
                for (d, x) in st.q.iter().enumerate() {
                    cols[d].push(*x);
                }
            }
            for d in 0..6 {
                xs[d].push(std::mem::take(&mut cols[d]));
            }
        }
        let mut rows = vec![];
        for d in 0..6 {
            let x2: Vec<Vec<f64>> = xs[d]
                .iter()
                .map(|c| c.iter().map(|v| v * v).collect())
                .collect();
            rows.push(moment(&format!("E[x{d}]"), &xs[d], 0.0));
            rows.push(moment(&format!("E[x{d}^2]"), &x2, var[d]));
        }
        assert_eq!(verdict(&rows), Verdict::Pass, "{rows:#?}");
    }

    #[test]
    fn nuts_under_mix_is_refused_at_end_of_warmup() {
        let (t, _, a, _) = coupled();
        let nuts = On {
            block: a.clone(),
            inner: Box::new(Nuts::for_block(t.clone(), a.indices.clone(), 100, 10)),
        };
        let other = On {
            block: a.clone(),
            inner: Box::new(Mala::new(0.3)),
        };
        let mut k = Mix::new(&[1.0, 1.0], vec![Box::new(nuts), Box::new(other)]).unwrap();
        let mut ev = Evaluator::new(t.as_ref());
        let mut rng = chain_rng(1, 1);
        let mut st = initial_state(&Init::Radius(1.0), &mut ev, &mut rng).unwrap();
        k.init(&st, &mut ev, &mut rng).unwrap();
        for _ in 0..100 {
            k.step(&mut st, &mut ev, &mut rng).unwrap();
        }
        let e = k.end_warmup().unwrap_err();
        assert!(e.to_string().contains("cannot sit under mix()"), "{e}");
    }

    #[test]
    fn block_nuts_counts_its_refresh_gradient() {
        let (t, _, a, b) = coupled();
        let mut k = hybrid(&t, &a, &b, 10, 10);
        let mut ev = Evaluator::new(t.as_ref());
        let mut rng = chain_rng(2, 1);
        let mut st = initial_state(&Init::Radius(1.0), &mut ev, &mut rng).unwrap();
        k.init(&st, &mut ev, &mut rng).unwrap();
        let info = k.step(&mut st, &mut ev, &mut rng).unwrap();
        let nuts = &info.blocks[0];
        assert_eq!(nuts.label, "a");
        // at least the refresh plus one leapfrog step
        assert!(nuts.info.n_grad >= 2, "{:?}", nuts.info);
        assert_eq!(nuts.info.n_leapfrog.map(|n| n >= 1), Some(true));
    }
}
