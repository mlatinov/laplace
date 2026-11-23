//! Simulation-based calibration (Phase 5.2): `laplace sampler test <kernel>
//! --sbc N`.
//!
//! Talts et al. (2018): draw parameters from the prior, data from the
//! model given them, run the sampler on that data, and record the rank of
//! each true value among the posterior draws. A sampler that targets the
//! right posterior gives uniform ranks, whatever the data.
//!
//! - **Model:** a built-in hierarchical normal (non-centred, J = 8):
//!   `mu ~ N(0, 1)`, `tau ~ N+(0, 1)`, `eta_j ~ N(0, 1)`,
//!   `y_j ~ N(mu + tau * eta_j, 1)`. The prior and the data are drawn exactly
//!   in Rust; each replicate's data reach BridgeStan as a JSON string.
//! - **Ranks:** one chain per replicate, `L = 99` kept draws (thinned to keep
//!   them close to independent), rank = number of kept draws below the true
//!   value, so 0..=L.
//! - **Uniformity:** simultaneous ECDF bands (Säilynoja, Bürkner & Vehtari
//!   2022, "Graphical test for discrete uniformity and its applications in
//!   goodness-of-fit evaluation and multiple sample comparison"). With
//!   discrete ranks the count of ranks below `k` is exactly
//!   Binomial(N, k / (L + 1)) under uniformity. The statistic is the
//!   smallest two-sided binomial tail probability over k = 1..L; the band
//!   level is the `alpha`-quantile of that statistic under simulated
//!   uniform ranks (the same N and L), Bonferroni-corrected over the
//!   quantities. A quantity fails when its ECDF leaves the band.

use std::ffi::CString;
use std::sync::Arc;

use bridgestan::{Model, StanLibrary};
use rand::rngs::ChaCha8Rng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

use crate::runner::{chain_rng, run_chain, Init, Sampler};
use crate::target::{BridgeStanTarget, Target};

pub const SBC_MODEL: &str = "// laplace sampler test --sbc: a hierarchical normal, non-centred.
data {
  int<lower=1> J;
  vector[J] y;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] eta;
}
model {
  mu ~ normal(0, 1);
  tau ~ normal(0, 1);
  eta ~ std_normal();
  y ~ normal(mu + tau * eta, 1);
}
";

pub const J: usize = 8;
/// Kept draws per replicate: ranks are 0..=L.
pub const L: usize = 99;

/// The quantities ranked, in BridgeStan's constrained order.
pub fn quantities() -> Vec<String> {
    let mut q = vec!["mu".to_string(), "tau".to_string()];
    q.extend((1..=J).map(|j| format!("eta.{j}")));
    q
}

#[derive(Clone, Copy, Debug)]
pub struct SbcSettings {
    pub replicates: usize,
    pub warmup: u64,
    /// Iterations per kept draw.
    pub thin: u64,
    /// Family-wise level of the uniformity test.
    pub alpha: f64,
    pub seed: u64,
}

impl Default for SbcSettings {
    fn default() -> Self {
        Self {
            replicates: 200,
            warmup: 500,
            thin: 4,
            alpha: 0.01,
            seed: 20261123,
        }
    }
}

/// One replicate's truth and data.
struct Draw {
    truth: Vec<f64>,
    y: Vec<f64>,
}

fn prior_draw(seed: u64, replicate: usize) -> Draw {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(1_000_000 + replicate as u64);
    let z = |rng: &mut ChaCha8Rng| -> f64 { rng.sample(StandardNormal) };
    let mu = z(&mut rng);
    let tau = z(&mut rng).abs();
    let eta: Vec<f64> = (0..J).map(|_| z(&mut rng)).collect();
    let y: Vec<f64> = eta.iter().map(|e| mu + tau * e + z(&mut rng)).collect();
    let mut truth = vec![mu, tau];
    truth.extend(&eta);
    Draw { truth, y }
}

fn data_json(y: &[f64]) -> String {
    serde_json::json!({ "J": J, "y": y }).to_string()
}

/// Ranks of every quantity in one replicate.
fn replicate(
    lib: &Arc<StanLibrary>,
    sampler: &Sampler,
    s: &SbcSettings,
    r: usize,
) -> anyhow::Result<Vec<usize>> {
    let d = prior_draw(s.seed, r);
    let model = Model::new(
        lib.clone(),
        Some(CString::new(data_json(&d.y))?),
        s.seed as u32,
    )?;
    let params = crate::sample::model_params(&model)?;
    let target = Arc::new(BridgeStanTarget { model });
    let dyn_t: Arc<dyn Target> = target.clone();
    let draws = L as u64 * s.thin;
    let mut built = sampler.build_chain(dyn_t, &params, s.warmup, draws)?;
    let mut out = vec![0.0; 2 + J];
    let mut below = vec![0usize; 2 + J];
    let mut i = 0u64;
    run_chain(
        target.as_ref(),
        built.kernel.as_mut(),
        &Init::Radius(2.0),
        s.warmup,
        draws,
        &mut chain_rng(s.seed, 1 + r as u64),
        |_| Ok(()),
        |st, _, _| {
            if (i + 1).is_multiple_of(s.thin) {
                target
                    .model
                    .param_constrain::<&StanLibrary>(&st.q, true, false, &mut out, None)?;
                for (k, v) in out.iter().enumerate() {
                    below[k] += usize::from(*v < d.truth[k]);
                }
            }
            i += 1;
            Ok(())
        },
    )?;
    Ok(below)
}

// ---- the uniformity test --------------------------------------------------------------

/// `cdf[k][c] = P(Binomial(n, k / (L + 1)) <= c)` for k = 1..=L.
fn binomial_cdfs(n: usize) -> Vec<Vec<f64>> {
    (1..=L)
        .map(|k| {
            let p = k as f64 / (L + 1) as f64;
            // log pmf by recursion from c = 0.
            let mut logpmf = vec![0.0; n + 1];
            logpmf[0] = n as f64 * (1.0 - p).ln();
            for c in 1..=n {
                logpmf[c] =
                    logpmf[c - 1] + ((n - c + 1) as f64 / c as f64).ln() + (p / (1.0 - p)).ln();
            }
            let mut acc = 0.0;
            logpmf
                .iter()
                .map(|lp| {
                    acc += lp.exp();
                    acc.min(1.0)
                })
                .collect()
        })
        .collect()
}

/// The smallest two-sided pointwise tail probability of the rank ECDF over
/// k = 1..=L, and the k where it is reached.
fn statistic(ranks: &[usize], cdfs: &[Vec<f64>]) -> (f64, usize) {
    let mut hist = vec![0usize; L + 1];
    for &r in ranks {
        hist[r.min(L)] += 1;
    }
    let mut below = 0usize;
    let mut best = (f64::INFINITY, 0);
    for k in 1..=L {
        below += hist[k - 1];
        let cdf = &cdfs[k - 1];
        let lower = cdf[below];
        let upper = 1.0 - if below == 0 { 0.0 } else { cdf[below - 1] };
        let p = (2.0 * lower.min(upper)).min(1.0);
        if p < best.0 {
            best = (p, k);
        }
    }
    best
}

/// The level of the simultaneous band: the `alpha`-quantile of the
/// statistic under uniform ranks, by simulation (fixed seed).
fn band_level(n: usize, alpha: f64, cdfs: &[Vec<f64>], seed: u64) -> f64 {
    const SIMS: usize = 4000;
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5bc);
    let mut stats: Vec<f64> = (0..SIMS)
        .map(|_| {
            let ranks: Vec<usize> = (0..n).map(|_| rng.random_range(0..=L)).collect();
            statistic(&ranks, cdfs).0
        })
        .collect();
    stats.sort_by(|a, b| a.total_cmp(b));
    stats[((alpha * SIMS as f64).floor() as usize).min(SIMS - 1)]
}

/// One quantity's result.
#[derive(Clone, Debug)]
pub struct Uniformity {
    pub name: String,
    pub ranks: Vec<usize>,
    /// Smallest pointwise tail probability, and where (rank k).
    pub statistic: f64,
    pub at_rank: usize,
    /// The band: below it, the ECDF left the band.
    pub level: f64,
    pub pass: bool,
}

pub struct SbcResult {
    pub settings: SbcSettings,
    pub quantities: Vec<Uniformity>,
}

impl SbcResult {
    pub fn pass(&self) -> bool {
        self.quantities.iter().all(|q| q.pass)
    }
}

/// Test each quantity's ranks for uniformity, Bonferroni over quantities.
pub fn uniformity(
    names: &[String],
    ranks: &[Vec<usize>],
    alpha: f64,
    seed: u64,
) -> Vec<Uniformity> {
    let n = ranks.first().map_or(0, Vec::len);
    let cdfs = binomial_cdfs(n);
    let level = band_level(n, alpha / names.len() as f64, &cdfs, seed);
    names
        .iter()
        .zip(ranks)
        .map(|(name, r)| {
            let (stat, k) = statistic(r, &cdfs);
            Uniformity {
                name: name.clone(),
                ranks: r.clone(),
                statistic: stat,
                at_rank: k,
                level,
                pass: stat >= level,
            }
        })
        .collect()
}

/// A replicate's index and its ranks.
type Ranked = (usize, Vec<usize>);

/// Run SBC: compile the built-in model (cached), then `replicates`
/// replicates on all cores.
pub fn run_sbc(
    sampler: &Sampler,
    s: SbcSettings,
    progress: impl Fn(usize) + Sync,
) -> anyhow::Result<SbcResult> {
    let dir = crate::model::cache_root()?.join("suite");
    std::fs::create_dir_all(&dir)?;
    let stan = dir.join("sbc_hierarchical_normal.stan");
    if std::fs::read_to_string(&stan).ok().as_deref() != Some(SBC_MODEL) {
        std::fs::write(&stan, SBC_MODEL)?;
    }
    let prepared = crate::model::prepare(&stan)?;
    let lib = Arc::new(bridgestan::open_library(&prepared.library)?);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<anyhow::Result<Vec<Ranked>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                sc.spawn(|| -> anyhow::Result<Vec<Ranked>> {
                    let mut mine = vec![];
                    loop {
                        let r = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if r >= s.replicates {
                            return Ok(mine);
                        }
                        mine.push((r, replicate(&lib, sampler, &s, r)?));
                        progress(r);
                    }
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("sbc worker"))
            .collect()
    });
    let mut all: Vec<(usize, Vec<usize>)> = vec![];
    for r in results {
        all.extend(r?);
    }
    all.sort_by_key(|(r, _)| *r);
    let names = quantities();
    let by_quantity: Vec<Vec<usize>> = (0..names.len())
        .map(|k| all.iter().map(|(_, ranks)| ranks[k]).collect())
        .collect();
    Ok(SbcResult {
        settings: s,
        quantities: uniformity(&names, &by_quantity, s.alpha, s.seed),
    })
}

/// A text ECDF-difference plot: for each tenth of the rank range, the
/// ECDF minus the uniform CDF, as a bar.
fn ecdf_diff(ranks: &[usize]) -> String {
    let n = ranks.len() as f64;
    (1..10)
        .map(|d| {
            let k = d * (L + 1) / 10;
            let below = ranks.iter().filter(|&&r| r < k).count() as f64 / n;
            let diff = below - k as f64 / (L + 1) as f64;
            let bars = ((diff.abs() * 100.0).round() as usize).min(9);
            match diff {
                _ if bars == 0 => " 0".into(),
                x if x > 0.0 => format!("+{bars}"),
                _ => format!("-{bars}"),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn render(name: &str, r: &SbcResult) -> String {
    let s = r.settings;
    let mut out = format!(
        "laplace sampler test --sbc: {name}\n\n{} replicates of a hierarchical normal (J = {J}), {} warmup, \
         {L} kept draws thinned 1:{}; family-wise alpha {} (Bonferroni over {} quantities)\n\n",
        s.replicates,
        s.warmup,
        s.thin,
        s.alpha,
        r.quantities.len()
    );
    out += &format!(
        "{:<8} {:<6} {:>12} {:>10} {:>8}   ECDF - uniform at deciles (x0.01)\n",
        "quantity", "result", "min tail p", "band", "at rank"
    );
    for q in &r.quantities {
        out += &format!(
            "{:<8} {:<6} {:>12.2e} {:>10.2e} {:>8}   {}\n",
            q.name,
            if q.pass { "pass" } else { "FAIL" },
            q.statistic,
            q.level,
            q.at_rank,
            ecdf_diff(&q.ranks)
        );
    }
    out += &format!(
        "\nresult: {}\n",
        if r.pass() {
            "ranks are uniform: no evidence the sampler targets the wrong posterior"
        } else {
            "ranks are NOT uniform: the sampler does not target the posterior. Reading the deciles: \
             positive throughout = true values rank low, the sampler's draws sit too high (negative: too \
             low); positive then negative = too many extreme ranks, its draws are too narrow (negative then \
             positive: too wide)"
        }
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_ranks_pass_and_skewed_ranks_fail() {
        let names = vec!["a".to_string(), "b".to_string()];
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let uniform: Vec<usize> = (0..300).map(|_| rng.random_range(0..=L)).collect();
        // Ranks pushed toward the low end, as when draws sit too high.
        let skewed: Vec<usize> = (0..300)
            .map(|_| {
                let u: f64 = rng.random();
                ((u * u) * (L + 1) as f64).floor() as usize
            })
            .collect();
        let r = uniformity(&names, &[uniform, skewed], 0.01, 1);
        assert!(r[0].pass, "{:?}", (r[0].statistic, r[0].level));
        assert!(!r[1].pass, "{:?}", (r[1].statistic, r[1].level));
    }

    #[test]
    fn the_band_holds_its_level_on_uniform_ranks() {
        // Under uniformity the statistic falls below the band in about
        // alpha of the cases.
        let cdfs = binomial_cdfs(100);
        let level = band_level(100, 0.1, &cdfs, 7);
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let trials = 1000;
        let rejected = (0..trials)
            .filter(|_| {
                let ranks: Vec<usize> = (0..100).map(|_| rng.random_range(0..=L)).collect();
                statistic(&ranks, &cdfs).0 < level
            })
            .count() as f64
            / trials as f64;
        assert!((0.06..0.14).contains(&rejected), "{rejected}");
    }

    #[test]
    fn binomial_cdf_matches_a_direct_sum() {
        let c = binomial_cdfs(10);
        // k = 50: p = 0.5; P(X <= 5) = 638/1024.
        assert!((c[49][5] - 638.0 / 1024.0).abs() < 1e-12);
        assert!((c[49][10] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn prior_draws_are_deterministic_and_distinct() {
        let a = prior_draw(1, 0);
        let b = prior_draw(1, 0);
        let c = prior_draw(1, 1);
        assert_eq!(a.truth, b.truth);
        assert_ne!(a.truth, c.truth);
        assert!(a.truth[1] >= 0.0, "tau is half-normal");
        assert_eq!(data_json(&[1.0, 2.0]), r#"{"J":8,"y":[1.0,2.0]}"#);
    }
}
