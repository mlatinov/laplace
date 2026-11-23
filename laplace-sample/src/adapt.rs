//! Warmup adaptation as reusable parts (Phase 3 turns [`Adapter`] into the
//! `adapt()` combinator).
//!
//! - [`DualAveraging`]: Stan's step-size adaptation toward a target
//!   acceptance rate (Hoffman & Gelman 2014, with Stan's defaults).
//! - [`Welford`]: running per-coordinate variance.
//! - [`Schedule`]: Stan's windowed warmup (init buffer, doubling slow
//!   windows, term buffer).
//! - [`RobbinsMonro`]: an optional final step-size phase for kernels whose
//!   only adaptation signal is one Metropolis acceptance probability.
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

/// Robbins–Monro on `log(step)` with gains `c / (k + k0)`; the result is
/// the Polyak average of the iterates over the second half of the phase.
///
/// Why it exists: Stan's dual averaging keeps the *average acceptance over
/// its iterates* at the target, but its iterates never stop fluctuating, and
/// with a single-proposal kernel (one noisy MH probability per step) the
/// acceptance at the *averaged* step lands well above the target (MALA on a
/// 10-d normal: 0.69 against 0.574, see the Phase 1 report). Decaying gains
/// make the iterates converge to the step whose acceptance is the target.
#[derive(Clone, Debug)]
pub struct RobbinsMonro {
    pub target: f64,
    pub gain: f64,
    pub k0: f64,
    x: f64,
    k: f64,
    len: u64,
    sum: f64,
    count: u64,
}

impl RobbinsMonro {
    pub fn new(target: f64) -> Self {
        Self {
            target,
            gain: 1.5,
            k0: 5.0,
            x: 0.0,
            k: 0.0,
            len: 0,
            sum: 0.0,
            count: 0,
        }
    }

    /// Start a phase of `len` updates from `step`.
    pub fn start(&mut self, step: f64, len: u64) {
        self.x = step.ln();
        self.k = 0.0;
        self.len = len;
        self.sum = 0.0;
        self.count = 0;
    }

    /// Feed one acceptance probability; returns the next step size.
    pub fn update(&mut self, accept: f64) -> f64 {
        let accept = if accept.is_nan() {
            0.0
        } else {
            accept.clamp(0.0, 1.0)
        };
        self.k += 1.0;
        self.x += (accept - self.target) * self.gain / (self.k + self.k0);
        if self.k > (self.len / 2) as f64 {
            self.sum += self.x;
            self.count += 1;
        }
        self.x.exp()
    }

    pub fn final_step(&self) -> f64 {
        if self.count == 0 {
            self.x.exp()
        } else {
            (self.sum / self.count as f64).exp()
        }
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

/// How the step size is settled in the term buffer (the last, step-size-only
/// phase of warmup).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepFinish {
    /// Stan: dual averaging all the way; final step = `exp(x_bar)`. Term
    /// buffer 50.
    DualAveraging,
    /// Dual averaging until the last metric update, then [`RobbinsMonro`]
    /// over a 100-iteration term buffer. For single-proposal MH kernels.
    RobbinsMonro,
}

impl StepFinish {
    pub fn term_buffer(self) -> u64 {
        match self {
            StepFinish::DualAveraging => 50,
            StepFinish::RobbinsMonro => 100,
        }
    }
}

/// Which tuning values to adapt, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptConfig {
    pub target_accept: f64,
    pub step_size: bool,
    pub metric: bool,
    pub finish: StepFinish,
}

/// Drives a kernel's tuning during warmup.
pub struct Adapter {
    pub config: AdaptConfig,
    pub schedule: Schedule,
    pub da: DualAveraging,
    rm: RobbinsMonro,
    var: Welford,
}

impl Adapter {
    pub fn new(config: AdaptConfig, warmup: u64, dim: usize, initial_step: f64) -> Self {
        Self {
            config,
            schedule: Schedule::with_buffers(warmup, 75, config.finish.term_buffer(), 25),
            da: DualAveraging::new(config.target_accept, initial_step),
            rm: RobbinsMonro::new(config.target_accept),
            var: Welford::new(dim),
        }
    }

    fn rm_finish(&self) -> bool {
        self.config.finish == StepFinish::RobbinsMonro && self.schedule.term_buffer > 0
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
        if iteration >= warmup {
            return Ok(()); // frozen
        }
        let mut t = Tuning {
            step_size: None,
            inv_metric: None,
        };
        let term_start = warmup - self.schedule.term_buffer;
        let accept = info.accept_prob.unwrap_or(0.0);
        if self.config.step_size {
            if self.rm_finish() && iteration >= term_start {
                if iteration == term_start {
                    let current = kernel.tuning().step_size.unwrap_or(1.0);
                    self.rm.start(current, self.schedule.term_buffer);
                }
                t.step_size = Some(self.rm.update(accept));
            } else {
                t.step_size = Some(self.da.update(accept));
            }
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
        if iteration + 1 == warmup && self.config.step_size {
            t.step_size = Some(if self.rm_finish() {
                self.rm.final_step()
            } else {
                self.da.final_step()
            });
        }
        kernel.set_tuning(&t)
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
    fn robbins_monro_converges_to_the_target_of_a_smooth_curve() {
        // accept(step) = exp(-step): the root of accept = 0.5 is ln 2.
        let mut rm = RobbinsMonro::new(0.5);
        rm.start(1.0, 4000);
        let mut step = 1.0f64;
        for _ in 0..4000 {
            step = rm.update((-step).exp());
        }
        assert!(
            (rm.final_step() - 2f64.ln()).abs() < 0.02,
            "{}",
            rm.final_step()
        );
    }

    #[test]
    fn robbins_monro_finish_uses_a_100_iteration_term_buffer() {
        let s = Schedule::with_buffers(1000, 75, StepFinish::RobbinsMonro.term_buffer(), 25);
        assert_eq!(s.term_buffer, 100);
        assert_eq!(*s.window_ends.last().unwrap(), 900);
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
