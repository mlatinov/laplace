//! Warmup adaptation as reusable parts (Phase 3 turns [`Adapter`] into the
//! `adapt()` combinator).
//!
//! - [`DualAveraging`]: Stan's step-size adaptation toward a target
//!   acceptance rate (Hoffman & Gelman 2014, with Stan's defaults).
//! - [`Welford`]: running per-coordinate variance.
//! - [`Schedule`]: Stan's windowed warmup (init buffer, doubling slow
//!   windows, term buffer).
//! - [`Adapter`]: drives any kernel through its `tuning()` / `set_tuning()`
//!   hooks during warmup, and **freezes** after warmup: nothing here can
//!   change a kernel's tuning once `iteration >= warmup`.

use crate::kernel::{Kernel, StepInfo, Tuning};

/// Stan's dual averaging (`stepsize_adaptation` in Stan, same constants).
#[derive(Clone, Debug)]
pub struct DualAveraging {
    pub target: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub t0: f64,
    mu: f64,
    counter: f64,
    s_bar: f64,
    x_bar: f64,
}

impl DualAveraging {
    pub fn new(target: f64, initial_step: f64) -> Self {
        let mut da = Self {
            target,
            gamma: 0.05,
            kappa: 0.75,
            t0: 10.0,
            mu: 0.0,
            counter: 0.0,
            s_bar: 0.0,
            x_bar: 0.0,
        };
        da.restart(initial_step);
        da
    }

    /// Restart around `step` (Stan: after every metric update).
    pub fn restart(&mut self, step: f64) {
        self.mu = (10.0 * step).ln();
        self.counter = 0.0;
        self.s_bar = 0.0;
        self.x_bar = 0.0;
    }

    /// Feed one acceptance statistic; returns the next step size to use.
    pub fn update(&mut self, accept: f64) -> f64 {
        let accept = if accept.is_nan() {
            0.0
        } else {
            accept.clamp(0.0, 1.0)
        };
        self.counter += 1.0;
        let eta = 1.0 / (self.counter + self.t0);
        self.s_bar = (1.0 - eta) * self.s_bar + eta * (self.target - accept);
        let x = self.mu - self.s_bar * self.counter.sqrt() / self.gamma;
        let x_eta = self.counter.powf(-self.kappa);
        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * x;
        x.exp()
    }

    /// The averaged step size, used once adaptation ends.
    pub fn final_step(&self) -> f64 {
        self.x_bar.exp()
    }
}

/// Running mean and variance per coordinate.
#[derive(Clone, Debug)]
pub struct Welford {
    n: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl Welford {
    pub fn new(dim: usize) -> Self {
        Self {
            n: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
        }
    }
    pub fn add(&mut self, x: &[f64]) {
        self.n += 1;
        let n = self.n as f64;
        for ((xi, mean), m2) in x.iter().zip(&mut self.mean).zip(&mut self.m2) {
            let d = xi - *mean;
            *mean += d / n;
            *m2 += d * (xi - *mean);
        }
    }
    pub fn count(&self) -> u64 {
        self.n
    }
    /// Stan's regularised variance:
    /// `n/(n+5) * var + 1e-3 * 5/(n+5)`.
    pub fn regularized_variance(&self) -> Vec<f64> {
        let n = self.n as f64;
        self.m2
            .iter()
            .map(|m2| {
                let var = m2 / (n - 1.0);
                (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            })
            .collect()
    }
    pub fn reset(&mut self) {
        let d = self.mean.len();
        *self = Self::new(d);
    }
}

/// Stan's warmup windows. Iterations `[0, init_buffer)` adapt the step size
/// only; slow windows `[init_buffer, warmup - term_buffer)` also collect
/// draws for the metric, doubling in length, the last one stretched to the
/// end; `[warmup - term_buffer, warmup)` adapt the step size only.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub warmup: u64,
    pub init_buffer: u64,
    pub term_buffer: u64,
    /// Exclusive end of every slow window, increasing.
    pub window_ends: Vec<u64>,
}

impl Schedule {
    /// Stan defaults: `init_buffer = 75`, `term_buffer = 50`, `window = 25`,
    /// shrunk to 15% / 10% / 75% of warmup when warmup is too short for them
    /// (Stan's `windowed_adaptation::restart` rule). Warmup under 20: no
    /// metric adaptation at all.
    pub fn stan(warmup: u64) -> Self {
        Self::with_buffers(warmup, 75, 50, 25)
    }

    /// Stan's window rule with explicit `init_buffer`, `term_buffer` and
    /// first `window` sizes.
    pub fn with_buffers(warmup: u64, init: u64, term: u64, window: u64) -> Self {
        let (mut init, mut term, mut window) = (init, term, window);
        if warmup < 20 {
            return Self {
                warmup,
                init_buffer: warmup,
                term_buffer: 0,
                window_ends: vec![],
            };
        }
        if init + window + term > warmup {
            init = (0.15 * warmup as f64) as u64;
            term = (0.1 * warmup as f64) as u64;
            window = warmup - (init + term);
        }
        let end = warmup - term;
        let mut ends = vec![];
        let mut start = init;
        let mut size = window;
        while start < end {
            let mut stop = start + size;
            // Stan: if the window after this one would not fit, stretch
            // this one to the end of the slow phase.
            if stop + 2 * size > end {
                stop = end;
            }
            ends.push(stop);
            start = stop;
            size *= 2;
        }
        Self {
            warmup,
            init_buffer: init,
            term_buffer: term,
            window_ends: ends,
        }
    }

    pub fn in_slow_window(&self, iteration: u64) -> bool {
        iteration >= self.init_buffer && iteration < self.warmup - self.term_buffer
    }

    pub fn ends_window(&self, iteration: u64) -> bool {
        self.window_ends.contains(&(iteration + 1))
    }
}

/// Which tuning values to adapt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptConfig {
    pub target_accept: f64,
    pub step_size: bool,
    pub metric: bool,
}

/// Drives a kernel's tuning during warmup.
pub struct Adapter {
    pub config: AdaptConfig,
    pub schedule: Schedule,
    pub da: DualAveraging,
    var: Welford,
    /// The final averaged step size has been installed.
    finished: bool,
}

impl Adapter {
    pub fn new(config: AdaptConfig, warmup: u64, dim: usize, initial_step: f64) -> Self {
        Self {
            config,
            schedule: Schedule::stan(warmup),
            da: DualAveraging::new(config.target_accept, initial_step),
            var: Welford::new(dim),
            finished: false,
        }
    }

    /// End of warmup: install the final averaged step size if the schedule
    /// has not already (the kernel ran fewer warmup steps than expected,
    /// e.g. inside mix). Nothing changes tuning after this.
    pub fn finish(&mut self, kernel: &mut dyn Kernel) -> anyhow::Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if self.config.step_size {
            kernel.set_tuning(&Tuning {
                step_size: Some(self.da.final_step()),
                inv_metric: None,
            })?;
        }
        Ok(())
    }

    /// Call after every step. `iteration` counts from 0 over warmup and
    /// draws together. A no-op from `iteration >= warmup - 1` on, apart from
    /// installing the final averaged step size at the end of warmup.
    pub fn after_step(
        &mut self,
        iteration: u64,
        q: &[f64],
        info: &StepInfo,
        kernel: &mut dyn Kernel,
    ) -> anyhow::Result<()> {
        let warmup = self.schedule.warmup;
        if iteration >= warmup || self.finished {
            return Ok(()); // frozen
        }
        let mut t = Tuning {
            step_size: None,
            inv_metric: None,
        };
        if self.config.step_size {
            t.step_size = Some(self.da.update(info.accept_prob.unwrap_or(0.0)));
        }
        if self.config.metric && self.schedule.in_slow_window(iteration) {
            self.var.add(q);
            if self.schedule.ends_window(iteration) {
                t.inv_metric = Some(self.var.regularized_variance());
                self.var.reset();
                if self.config.step_size {
                    // Stan restarts dual averaging around the current step.
                    let current = t.step_size.unwrap_or(self.da.final_step());
                    self.da.restart(current);
                }
            }
        }
        if iteration + 1 == warmup {
            self.finished = true;
            if self.config.step_size {
                t.step_size = Some(self.da.final_step());
            }
        }
        kernel.set_tuning(&t)
    }
}

/// `adapt(k)` (Phase 3.3): warmup adaptation of one kernel, as a kernel.
///
/// Stan's windowed adaptation (dual averaging toward `target_accept`, the
/// diagonal metric from the slow windows) driven through `k`'s tuning
/// hooks, for whatever `k` declares adaptable ([`Kernel::adaptable`]: the
/// `@adapt` tags of a `.lapkernel`, the tuning of a built-in). The schedule
/// is sized to the warmup steps `k` is expected to take (warmup x uses per
/// sweep); at `end_warmup` everything freezes wherever the schedule is.
pub struct Adapt {
    pub inner: Box<dyn Kernel>,
    config: AdaptConfig,
    expected: u64,
    adapter: Option<Adapter>,
    count: u64,
    warm: bool,
}

impl Adapt {
    /// `target_accept`: `adapt(k, target_accept = ...)`, else `k`'s own
    /// default (a `.lapkernel`'s tag, a built-in's documented default).
    /// `expected_warmup_steps`: warmup iterations x the times `k` runs per
    /// iteration.
    pub fn new(
        inner: Box<dyn Kernel>,
        target_accept: Option<f64>,
        expected_warmup_steps: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !inner.adapts_itself(),
            "adapt({0}): {0} adapts its own step size and metric; use it without adapt()",
            inner.name()
        );
        let a = inner.adaptable();
        anyhow::ensure!(
            a.step_size || a.metric,
            "adapt({}): {}",
            inner.name(),
            a.missing
        );
        let target = match (target_accept, a.step_size) {
            (Some(t), _) => t,
            (None, true) => inner.default_target_accept().ok_or_else(|| {
                anyhow::anyhow!(
                    "adapt({}): no target_accept given and the kernel has no default",
                    inner.name()
                )
            })?,
            (None, false) => 0.5, // metric only: no step size to aim
        };
        anyhow::ensure!(
            target > 0.0 && target < 1.0,
            "target_accept must be in (0, 1), got {target}"
        );
        Ok(Self {
            config: AdaptConfig {
                target_accept: target,
                step_size: a.step_size,
                metric: a.metric,
            },
            inner,
            expected: expected_warmup_steps,
            adapter: None,
            count: 0,
            warm: true,
        })
    }

    pub fn target_accept(&self) -> f64 {
        self.config.target_accept
    }
}

impl Kernel for Adapt {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn source(&self) -> std::borrow::Cow<'static, str> {
        self.inner.source()
    }
    fn version(&self) -> String {
        self.inner.version()
    }
    fn init(
        &mut self,
        state: &crate::state::ChainState,
        ev: &mut crate::state::Evaluator,
        rng: &mut crate::kernel::ChainRng,
    ) -> anyhow::Result<()> {
        self.inner.init(state, ev, rng)?;
        let initial = self.inner.tuning().step_size.unwrap_or(1.0);
        self.adapter = Some(Adapter::new(
            self.config,
            self.expected,
            state.q.len(),
            initial,
        ));
        Ok(())
    }
    fn step(
        &mut self,
        state: &mut crate::state::ChainState,
        ev: &mut crate::state::Evaluator,
        rng: &mut crate::kernel::ChainRng,
    ) -> anyhow::Result<StepInfo> {
        let info = self.inner.step(state, ev, rng)?;
        if self.warm {
            let a = self.adapter.as_mut().expect("init before step");
            a.after_step(self.count, &state.q, &info, self.inner.as_mut())?;
            self.count += 1;
        }
        Ok(info)
    }
    fn tuning(&self) -> Tuning {
        self.inner.tuning()
    }
    fn set_tuning(&mut self, t: &Tuning) -> anyhow::Result<()> {
        self.inner.set_tuning(t)
    }
    fn end_warmup(&mut self) -> anyhow::Result<()> {
        self.warm = false;
        if let Some(a) = self.adapter.as_mut() {
            a.finish(self.inner.as_mut())?;
        }
        self.inner.end_warmup()
    }
    fn tunings(&self) -> Vec<(String, Tuning)> {
        self.inner
            .tunings()
            .into_iter()
            .map(|(p, t)| (format!("adapt/{p}"), t))
            .collect()
    }
    fn adaptable(&self) -> crate::kernel::Adaptable {
        crate::kernel::Adaptable::none("it is already inside adapt()")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stan_schedule_for_1000() {
        let s = Schedule::stan(1000);
        assert_eq!(s.init_buffer, 75);
        assert_eq!(s.term_buffer, 50);
        // Stan's windows for 1000 warmup: 25, 50, 100, 200, 500 (stretched).
        assert_eq!(s.window_ends, vec![100, 150, 250, 450, 950]);
    }

    #[test]
    fn short_warmup_shrinks_buffers() {
        let s = Schedule::stan(100);
        assert_eq!((s.init_buffer, s.term_buffer), (15, 10));
        assert_eq!(*s.window_ends.last().unwrap(), 90);
        let tiny = Schedule::stan(10);
        assert!(tiny.window_ends.is_empty());
    }

    #[test]
    fn dual_averaging_moves_toward_target() {
        // Acceptance always above target -> step grows; below -> shrinks.
        let mut up = DualAveraging::new(0.8, 1.0);
        let mut down = DualAveraging::new(0.8, 1.0);
        for _ in 0..200 {
            up.update(1.0);
            down.update(0.0);
        }
        assert!(up.final_step() > 1.0);
        assert!(down.final_step() < 1.0);
    }

    #[test]
    fn welford_matches_two_pass() {
        let xs = [[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [7.0, 70.0]];
        let mut w = Welford::new(2);
        for x in &xs {
            w.add(x);
        }
        let n = 4.0;
        let mean0 = 14.0 / n;
        let var0: f64 = xs.iter().map(|x| (x[0] - mean0).powi(2)).sum::<f64>() / (n - 1.0);
        let reg = (n / (n + 5.0)) * var0 + 1e-3 * (5.0 / (n + 5.0));
        assert!((w.regularized_variance()[0] - reg).abs() < 1e-12);
    }
}

#[cfg(test)]
mod adapt_combinator_tests {
    use super::*;
    use crate::combinators::Mix;
    use crate::kernels::{mala::Mala, nuts::Nuts};
    use crate::lapkernel::kernel::{InterpretedKernel, LapKernelFile};
    use crate::runner::{chain_rng, run_chain, Init};
    use crate::target::analytic::DiagGaussian;
    use crate::target::Target;
    use std::sync::Arc;

    fn shipped(name: &str) -> Arc<LapKernelFile> {
        let p = format!("{}/kernels/{name}.lapkernel", env!("CARGO_MANIFEST_DIR"));
        Arc::new(LapKernelFile::load(std::path::Path::new(&p)).unwrap())
    }

    fn mean_accept(k: &mut dyn Kernel, t: &dyn Target, seed: u64, chain: u64) -> f64 {
        let mut acc = vec![];
        run_chain(
            t,
            k,
            &Init::Radius(2.0),
            1000,
            1000,
            &mut chain_rng(seed, chain),
            |_| Ok(()),
            |_, i, _| {
                acc.push(i.accept_prob.unwrap());
                Ok(())
            },
        )
        .unwrap();
        acc.iter().sum::<f64>() / acc.len() as f64
    }

    /// 3.3: adapt(./kernels/mala.lapkernel) reaches its target acceptance
    /// (0.574, the tag's) on the standard normal: the mean over 5 seeds x 4
    /// chains is within +-0.15, the band the Phase 1 review set for Stan's
    /// dual averaging.
    #[test]
    fn adapt_of_the_mala_lapkernel_reaches_its_target() {
        let t = DiagGaussian::std_normal(10);
        let mut means = vec![];
        for seed in [11, 22, 33, 44, 55] {
            for chain in 1..=4 {
                let inner = InterpretedKernel::new(shipped("mala"), 10, &[]).unwrap();
                let mut k = Adapt::new(Box::new(inner), None, 1000).unwrap();
                assert_eq!(k.target_accept(), 0.574, "the tag's default");
                means.push(mean_accept(&mut k, &t, seed, chain));
            }
        }
        let m = means.iter().sum::<f64>() / means.len() as f64;
        assert!((m - 0.574).abs() < 0.15, "mean acceptance {m}");
    }

    #[test]
    fn the_sampler_can_override_the_target() {
        let inner = InterpretedKernel::new(shipped("mala"), 3, &[]).unwrap();
        let k = Adapt::new(Box::new(inner), Some(0.65), 1000).unwrap();
        assert_eq!(k.target_accept(), 0.65);
    }

    #[test]
    fn nothing_to_adapt_names_the_missing_tag() {
        let f = Arc::new(
            LapKernelFile::from_source(
                "fixed.lapkernel",
                "kernel fixed { tuning { real<lower=0> h = 0.1; } step { propose(q + h * std_normal_rng(dim), 0); } }"
                    .into(),
            )
            .unwrap(),
        );
        let inner = InterpretedKernel::new(f, 3, &[]).unwrap();
        let e = Adapt::new(Box::new(inner), None, 100).err().unwrap();
        assert!(
            e.to_string()
                .contains("tag its step size @adapt(step_size, target_accept = ...)"),
            "{e}"
        );
        let t: Arc<dyn Target> = Arc::new(DiagGaussian::std_normal(2));
        let e = Adapt::new(Box::new(Nuts::new(t, 10, 10)), None, 10)
            .err()
            .unwrap();
        assert!(
            e.to_string()
                .contains("adapts its own step size and metric"),
            "{e}"
        );
        let mix = Mix::new(&[1.0], vec![Box::new(Mala::new(0.1))]).unwrap();
        let e = Adapt::new(Box::new(mix), None, 10).err().unwrap();
        assert!(
            e.to_string().contains("mix exposes no step size or metric"),
            "{e}"
        );
    }

    /// Inside mix the kernel runs fewer warmup steps than its schedule
    /// expects; tuning still freezes exactly at the end of warmup.
    #[test]
    fn adapt_inside_mix_freezes_at_end_of_warmup() {
        let t = DiagGaussian::std_normal(4);
        let a = Adapt::new(Box::new(Mala::new(1.0)), None, 1000).unwrap();
        let mut k = Mix::new(&[1.0, 1.0], vec![Box::new(a), Box::new(Mala::new(0.3))]).unwrap();
        let mut ev = crate::state::Evaluator::new(&t);
        let mut rng = chain_rng(3, 1);
        let mut st = crate::runner::initial_state(&Init::Radius(2.0), &mut ev, &mut rng).unwrap();
        k.init(&st, &mut ev, &mut rng).unwrap();
        for _ in 0..1000 {
            k.step(&mut st, &mut ev, &mut rng).unwrap();
        }
        k.end_warmup().unwrap();
        let frozen = k.tunings();
        assert_ne!(frozen[0].1.step_size, Some(1.0), "adapted during warmup");
        for _ in 0..300 {
            k.step(&mut st, &mut ev, &mut rng).unwrap();
            assert_eq!(k.tunings(), frozen, "tuning changed after warmup");
        }
    }
}
