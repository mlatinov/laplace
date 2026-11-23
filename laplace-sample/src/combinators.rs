//! Composing kernels (Phase 3.2): `on(block, k)`, `seq(...)`,
//! `mix(w: k, ...)`, `repeat(n, k)`. Each is a [`Kernel`], so they nest.
//!
//! Every combinator leaves the chain state consistent: after a block update
//! the full position, log density and *full* gradient describe the same
//! point (the gradient comes from the shared evaluator, keyed by the full
//! position, so it is free when the block kernel just evaluated it).
//!
//! RNG use: `on`, `seq` and `repeat` draw nothing themselves. `mix` draws
//! exactly one uniform (`rng.random::<f64>()`) per step, before the chosen
//! kernel runs.

use rand::RngExt;

use crate::blocks::Block;
use crate::kernel::{BlockInfo, ChainRng, Kernel, StepInfo, Tuning};
use crate::state::{ChainState, Evaluator, View};

/// Statistics of several steps folded into one: gradients and leapfrog
/// steps add up, acceptance is the mean, a divergence anywhere counts, tree
/// depth is the largest, energy is the last.
pub fn fold(infos: &[StepInfo]) -> StepInfo {
    let mean = |xs: Vec<f64>| (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64);
    let sum_u = |xs: Vec<u64>| (!xs.is_empty()).then(|| xs.iter().sum::<u64>());
    StepInfo {
        accept_prob: mean(infos.iter().filter_map(|i| i.accept_prob).collect()),
        n_grad: infos.iter().map(|i| i.n_grad).sum(),
        divergent: infos
            .iter()
            .filter_map(|i| i.divergent)
            .reduce(|a, b| a || b),
        energy: infos.iter().rev().find_map(|i| i.energy),
        treedepth: infos.iter().filter_map(|i| i.treedepth).max(),
        n_leapfrog: sum_u(infos.iter().filter_map(|i| i.n_leapfrog).collect()),
        extra: vec![],
        blocks: infos.iter().flat_map(|i| i.blocks.clone()).collect(),
    }
}

fn no_tuning() -> Tuning {
    Tuning {
        step_size: None,
        inv_metric: None,
    }
}

fn composite_set_tuning(name: &str) -> anyhow::Result<()> {
    anyhow::bail!("{name}(...) has no tuning of its own; tune the kernels inside it")
}

/// `on(block, k)`: run `k` on one block, the other coordinates fixed.
pub struct On {
    pub block: Block,
    pub inner: Box<dyn Kernel>,
}

impl On {
    fn block_state(&self, s: &ChainState) -> ChainState {
        ChainState {
            q: self.block.take(&s.q),
            lp: s.lp,
            grad: self.block.take(&s.grad),
        }
    }

    fn view(&self, s: &ChainState) -> View {
        View {
            indices: self.block.indices.clone(),
            base: s.q.clone(),
        }
    }
}

impl Kernel for On {
    fn name(&self) -> &'static str {
        "on"
    }
    fn source(&self) -> std::borrow::Cow<'static, str> {
        format!("on({}, {})", self.block.label, self.inner.source()).into()
    }

    fn init(
        &mut self,
        state: &ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            ev.view().is_none(),
            "on(...) cannot be nested inside another on(...)"
        );
        let bs = self.block_state(state);
        ev.set_view(Some(self.view(state)));
        let r = self.inner.init(&bs, ev, rng);
        ev.set_view(None);
        r
    }

    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        anyhow::ensure!(
            ev.view().is_none(),
            "on(...) cannot be nested inside another on(...)"
        );
        let mut bs = self.block_state(state);
        let old = bs.q.clone();
        let step_size = self.inner.tuning().step_size;
        ev.set_view(Some(self.view(state)));
        let r = self.inner.step(&mut bs, ev, rng);
        ev.set_view(None);
        let mut info = r?;
        let moved =
            bs.q.iter()
                .zip(&old)
                .any(|(a, b)| a.to_bits() != b.to_bits());
        if moved {
            self.block.put(&mut state.q, &bs.q);
            // Full gradient at the new point: a cache hit when the block
            // kernel evaluated it through the view; otherwise (nuts-rs runs
            // its own integrator) one real evaluation, counted.
            let before = ev.n_grad();
            let (lp, g) = ev.eval_full(&state.q)?;
            info.n_grad += ev.n_grad() - before;
            state.lp = lp;
            state.grad = g;
        }
        let mut flat = info.clone();
        flat.blocks.clear();
        Ok(StepInfo {
            blocks: vec![BlockInfo {
                label: self.block.label.clone(),
                info: flat,
                step_size,
            }],
            ..info
        })
    }

    fn tuning(&self) -> Tuning {
        self.inner.tuning()
    }
    fn set_tuning(&mut self, t: &Tuning) -> anyhow::Result<()> {
        self.inner.set_tuning(t)
    }
    fn adapts_itself(&self) -> bool {
        self.inner.adapts_itself()
    }
    fn end_warmup(&mut self) -> anyhow::Result<()> {
        self.inner.end_warmup()
    }
    fn tunings(&self) -> Vec<(String, Tuning)> {
        self.inner
            .tunings()
            .into_iter()
            .map(|(p, t)| (format!("on({})/{p}", self.block.label), t))
            .collect()
    }
}

/// `seq(k1, k2, ...)`: one step of each, in order.
pub struct Seq {
    pub items: Vec<Box<dyn Kernel>>,
}

impl Kernel for Seq {
    fn name(&self) -> &'static str {
        "seq"
    }
    fn source(&self) -> std::borrow::Cow<'static, str> {
        let parts: Vec<String> = self.items.iter().map(|k| k.source().into_owned()).collect();
        format!("seq({})", parts.join(", ")).into()
    }
    fn init(
        &mut self,
        state: &ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        self.items
            .iter_mut()
            .try_for_each(|k| k.init(state, ev, rng))
    }
    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let infos = self
            .items
            .iter_mut()
            .map(|k| k.step(state, ev, rng))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(fold(&infos))
    }
    fn tuning(&self) -> Tuning {
        no_tuning()
    }
    fn set_tuning(&mut self, _: &Tuning) -> anyhow::Result<()> {
        composite_set_tuning("seq")
    }
    fn adapts_itself(&self) -> bool {
        self.items.iter().any(|k| k.adapts_itself())
    }
    fn end_warmup(&mut self) -> anyhow::Result<()> {
        self.items.iter_mut().try_for_each(|k| k.end_warmup())
    }
    fn tunings(&self) -> Vec<(String, Tuning)> {
        self.items
            .iter()
            .enumerate()
            .flat_map(|(i, k)| {
                k.tunings()
                    .into_iter()
                    .map(move |(p, t)| (format!("seq[{i}]/{p}"), t))
            })
            .collect()
    }
}

/// `mix(w1: k1, w2: k2, ...)`: each step runs one kernel, chosen with
/// probability proportional to its weight, independently of the state.
pub struct Mix {
    /// Cumulative normalised weights; the last is exactly 1.
    cumulative: Vec<f64>,
    pub items: Vec<Box<dyn Kernel>>,
}

impl Mix {
    pub fn new(weights: &[f64], items: Vec<Box<dyn Kernel>>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            weights.len() == items.len() && !items.is_empty(),
            "mix(...) needs one weight per kernel"
        );
        anyhow::ensure!(
            weights.iter().all(|w| w.is_finite() && *w > 0.0),
            "mix(...) weights must be positive"
        );
        let total: f64 = weights.iter().sum();
        let mut acc = 0.0;
        let mut cumulative: Vec<f64> = weights
            .iter()
            .map(|w| {
                acc += w / total;
                acc
            })
            .collect();
        *cumulative.last_mut().unwrap() = 1.0;
        Ok(Self { cumulative, items })
    }

    /// Index chosen by a uniform `u` in [0, 1).
    pub fn choose(&self, u: f64) -> usize {
        self.cumulative
            .iter()
            .position(|c| u < *c)
            .unwrap_or(self.items.len() - 1)
    }
}

impl Kernel for Mix {
    fn name(&self) -> &'static str {
        "mix"
    }
    fn source(&self) -> std::borrow::Cow<'static, str> {
        let parts: Vec<String> = self
            .cumulative
            .iter()
            .zip(&self.items)
            .map(|(c, k)| format!("{c}: {}", k.source()))
            .collect();
        format!("mix({})", parts.join(", ")).into()
    }
    fn init(
        &mut self,
        state: &ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        self.items
            .iter_mut()
            .try_for_each(|k| k.init(state, ev, rng))
    }
    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let u: f64 = rng.random();
        let i = self.choose(u);
        let mut info = self.items[i].step(state, ev, rng)?;
        info.extra.push(("mix_choice", i as f64));
        Ok(info)
    }
    fn tuning(&self) -> Tuning {
        no_tuning()
    }
    fn set_tuning(&mut self, _: &Tuning) -> anyhow::Result<()> {
        composite_set_tuning("mix")
    }
    fn adapts_itself(&self) -> bool {
        self.items.iter().any(|k| k.adapts_itself())
    }
    fn end_warmup(&mut self) -> anyhow::Result<()> {
        self.items.iter_mut().try_for_each(|k| k.end_warmup())
    }
    fn tunings(&self) -> Vec<(String, Tuning)> {
        self.items
            .iter()
            .enumerate()
            .flat_map(|(i, k)| {
                k.tunings()
                    .into_iter()
                    .map(move |(p, t)| (format!("mix[{i}]/{p}"), t))
            })
            .collect()
    }
}

/// `repeat(n, k)`: `n` steps of `k`.
pub struct Repeat {
    pub n: u64,
    pub inner: Box<dyn Kernel>,
}

impl Kernel for Repeat {
    fn name(&self) -> &'static str {
        "repeat"
    }
    fn source(&self) -> std::borrow::Cow<'static, str> {
        format!("repeat({}, {})", self.n, self.inner.source()).into()
    }
    fn init(
        &mut self,
        state: &ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<()> {
        self.inner.init(state, ev, rng)
    }
    fn step(
        &mut self,
        state: &mut ChainState,
        ev: &mut Evaluator,
        rng: &mut ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let infos = (0..self.n)
            .map(|_| self.inner.step(state, ev, rng))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(fold(&infos))
    }
    fn tuning(&self) -> Tuning {
        self.inner.tuning()
    }
    fn set_tuning(&mut self, t: &Tuning) -> anyhow::Result<()> {
        self.inner.set_tuning(t)
    }
    fn adapts_itself(&self) -> bool {
        self.inner.adapts_itself()
    }
    fn end_warmup(&mut self) -> anyhow::Result<()> {
        self.inner.end_warmup()
    }
    fn tunings(&self) -> Vec<(String, Tuning)> {
        self.inner
            .tunings()
            .into_iter()
            .map(|(p, t)| (format!("repeat({})/{p}", self.n), t))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::{parameters, Block};
    use crate::diagnostics::{moment, verdict, Verdict};
    use crate::kernels::{mala::Mala, rwm::Rwm};
    use crate::runner::{chain_rng, run_chain, Init};
    use crate::target::analytic::Gaussian;
    use nalgebra::DMatrix;

    /// A kernel that only counts its calls (moves nowhere).
    struct Marker(u64);
    impl Kernel for Marker {
        fn name(&self) -> &'static str {
            "marker"
        }
        fn source(&self) -> std::borrow::Cow<'static, str> {
            "marker".into()
        }
        fn step(
            &mut self,
            _: &mut ChainState,
            _: &mut Evaluator,
            _: &mut ChainRng,
        ) -> anyhow::Result<StepInfo> {
            self.0 += 1;
            Ok(StepInfo::default())
        }
        fn tuning(&self) -> Tuning {
            no_tuning()
        }
        fn set_tuning(&mut self, _: &Tuning) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn mix_frequencies_match_the_weights() {
        let t = Gaussian {
            precision: vec![vec![1.0]],
        };
        let mut ev = Evaluator::new(&t);
        let mut st = ChainState::at(vec![0.0], &mut ev).unwrap();
        let mut m = Mix::new(
            &[1.0, 2.0, 7.0],
            vec![
                Box::new(Marker(0)),
                Box::new(Marker(0)),
                Box::new(Marker(0)),
            ],
        )
        .unwrap();
        let mut rng = chain_rng(5, 1);
        let n = 20_000u64;
        let mut counts = [0u64; 3];
        for _ in 0..n {
            let info = m.step(&mut st, &mut ev, &mut rng).unwrap();
            counts[info.extra[0].1 as usize] += 1;
        }
        for (k, p) in counts.iter().zip([0.1, 0.2, 0.7]) {
            let (mean, sd) = (n as f64 * p, (n as f64 * p * (1.0 - p)).sqrt());
            assert!(
                (*k as f64 - mean).abs() < 4.0 * sd,
                "count {k}, expected {mean} +- {sd}"
            );
        }
    }

    #[test]
    fn mix_draws_exactly_one_uniform_before_its_kernel() {
        let m = Mix::new(&[1.0, 1.0], vec![Box::new(Marker(0)), Box::new(Marker(0))]).unwrap();
        assert_eq!(m.choose(0.0), 0);
        assert_eq!(m.choose(0.4999), 0);
        assert_eq!(m.choose(0.5), 1);
        assert_eq!(m.choose(0.999_999), 1);
        assert!(Mix::new(&[1.0, -1.0], vec![Box::new(Marker(0)), Box::new(Marker(0))]).is_err());
    }

    fn coupled() -> (Gaussian, Vec<f64>) {
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
        (
            (Gaussian { precision: p }),
            (0..6).map(|i| cov[(i, i)]).collect(),
        )
    }

    fn two_blocks() -> (Block, Block) {
        let names: Vec<String> = ["a.1", "a.2", "a.3", "b.1", "b.2", "b.3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let p = parameters(&names).unwrap();
        (
            Block::new(&p, &["a".to_string()], None).unwrap(),
            Block::new(&p, &["b".to_string()], None).unwrap(),
        )
    }

    /// 3.2: a seq of two kernels, each leaving pi invariant on its block,
    /// passes the moment check, on a target whose blocks are correlated.
    #[test]
    fn seq_of_block_kernels_passes_the_moment_check() {
        let (t, var) = coupled();
        let (a, b) = two_blocks();
        let mut chains_x: Vec<Vec<Vec<f64>>> = vec![vec![]; 6];
        for chain in 1..=4 {
            let mut k = Seq {
                items: vec![
                    Box::new(On {
                        block: a.clone(),
                        inner: Box::new(Mala::new(0.6)),
                    }),
                    Box::new(On {
                        block: b.clone(),
                        inner: Box::new(Rwm::new(0.7)),
                    }),
                ],
            };
            let mut xs: Vec<Vec<f64>> = vec![vec![]; 6];
            let mut i = 0u64;
            run_chain(
                &t,
                &mut k,
                None,
                &Init::Radius(2.0),
                500,
                20_000,
                &mut chain_rng(11, chain),
                |_| Ok(()),
                |st, _, _| {
                    if i.is_multiple_of(10) {
                        for (d, x) in st.q.iter().enumerate() {
                            xs[d].push(*x);
                        }
                    }
                    i += 1;
                    Ok(())
                },
            )
            .unwrap();
            for d in 0..6 {
                chains_x[d].push(std::mem::take(&mut xs[d]));
            }
        }
        let mut rows = vec![];
        for d in 0..6 {
            let x2: Vec<Vec<f64>> = chains_x[d]
                .iter()
                .map(|c| c.iter().map(|v| v * v).collect())
                .collect();
            rows.push(moment(&format!("E[x{d}]"), &chains_x[d], 0.0));
            rows.push(moment(&format!("E[x{d}^2]"), &x2, var[d]));
        }
        assert_eq!(verdict(&rows), Verdict::Pass, "{rows:#?}");
    }

    #[test]
    fn on_keeps_the_full_state_consistent() {
        let (t, _) = coupled();
        let (a, _) = two_blocks();
        let mut k = On {
            block: a,
            inner: Box::new(Mala::new(0.5)),
        };
        let mut ev = Evaluator::new(&t);
        let mut st = ChainState::at(vec![0.3; 6], &mut ev).unwrap();
        let mut rng = chain_rng(1, 1);
        for _ in 0..50 {
            let info = k.step(&mut st, &mut ev, &mut rng).unwrap();
            assert_eq!(info.blocks.len(), 1);
            assert_eq!(info.blocks[0].label, "a");
            let mut g = vec![0.0; 6];
            let lp = crate::target::Target::log_density_gradient(&t, &st.q, &mut g).unwrap();
            assert_eq!(
                (st.lp, &st.grad),
                (lp, &g),
                "lp and full gradient match the full position"
            );
            assert_eq!(&st.q[3..], &[0.3; 3], "block b never moves");
        }
    }

    #[test]
    fn fold_combines_statistics() {
        let a = StepInfo {
            accept_prob: Some(0.2),
            n_grad: 3,
            divergent: Some(false),
            treedepth: Some(2),
            n_leapfrog: Some(3),
            energy: Some(1.0),
            ..Default::default()
        };
        let b = StepInfo {
            accept_prob: Some(0.6),
            n_grad: 5,
            divergent: Some(true),
            treedepth: Some(4),
            n_leapfrog: Some(7),
            energy: Some(2.0),
            ..Default::default()
        };
        let f = fold(&[a, b]);
        assert_eq!(f.accept_prob, Some(0.4));
        assert_eq!(
            (f.n_grad, f.divergent, f.treedepth, f.n_leapfrog, f.energy),
            (8, Some(true), Some(4), Some(10), Some(2.0))
        );
        assert_eq!(fold(&[StepInfo::default()]).accept_prob, None);
    }
}
