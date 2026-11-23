//! Convergence diagnostics and the moment-check harness used by the
//! statistical tests and, in Phase 5, by `laplace sampler test` to judge
//! user kernels.
//!
//! `chains[c][i]` is draw `i` of chain `c` for one scalar quantity. Chains
//! are split in half first, as in Stan. ESS is Geyer's initial monotone
//! sequence on the multi-chain autocorrelation (BDA3 / Stan).
//!
//! - [`ess`] is the basic split ESS (`posterior::ess_basic`); it sets the
//!   MCSE of a mean.
//! - [`ess_bulk`] and [`rhat`] are rank-normalised, as in
//!   `posterior::ess_bulk` and `posterior::rhat` (Vehtari et al. 2021); they
//!   decide chain health.
//!
//! A moment check only counts when the chains are healthy
//! (`docs/sampler/laplace_kernels.md`, section 4): see [`Verdict`].

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn var(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)
}

/// Split each chain in two halves; an odd middle draw is dropped, as in
/// posterior's internal `.split_chains` (used by `ess_*` and `rhat`).
fn split(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = vec![];
    for c in chains {
        let h = c.len() / 2;
        out.push(c[..h].to_vec());
        out.push(c[c.len() - h..].to_vec());
    }
    out
}

/// Split R-hat (Gelman et al. BDA3, no rank normalisation).
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    rhat_of_split(&split(chains))
}

fn rhat_of_split(s: &[Vec<f64>]) -> f64 {
    let n = s[0].len() as f64;
    let means: Vec<f64> = s.iter().map(|c| mean(c)).collect();
    let w = s.iter().map(|c| var(c)).sum::<f64>() / s.len() as f64;
    let b = n * var(&means);
    let var_plus = (n - 1.0) / n * w + b / n;
    (var_plus / w).sqrt()
}

/// Biased autocovariance up to `max_lag` (direct sum, fine for the test
/// sizes used here).
fn autocov(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let m = mean(x);
    (0..=max_lag.min(n - 1))
        .map(|t| (0..n - t).map(|i| (x[i] - m) * (x[i + t] - m)).sum::<f64>() / n as f64)
        .collect()
}

/// Effective sample size of the pooled draws (basic split ESS).
pub fn ess(chains: &[Vec<f64>]) -> f64 {
    ess_of_split(&split(chains))
}

fn ess_of_split(s: &[Vec<f64>]) -> f64 {
    let m = s.len() as f64;
    let n = s[0].len();
    let nf = n as f64;
    let acov: Vec<Vec<f64>> = s.iter().map(|c| autocov(c, n - 1)).collect();
    let chain_var: Vec<f64> = acov.iter().map(|a| a[0] * nf / (nf - 1.0)).collect();
    let mean_var = chain_var.iter().sum::<f64>() / m;
    let means: Vec<f64> = s.iter().map(|c| mean(c)).collect();
    let mut var_plus = mean_var * (nf - 1.0) / nf;
    if s.len() > 1 {
        var_plus += var(&means);
    }
    if var_plus <= 0.0 || !var_plus.is_finite() {
        return f64::NAN;
    }
    let rho = |t: usize| -> f64 {
        let mean_acov = acov.iter().map(|a| a[t]).sum::<f64>() / m;
        1.0 - (mean_var - mean_acov) / var_plus
    };
    // Geyer's initial positive sequence, then forced monotone by averaging
    // pairs; a line-for-line port of `posterior:::.ess` (posterior 1.7).
    let mut rho_t = vec![0.0; n];
    let mut t = 0;
    let mut even = 1.0;
    rho_t[0] = even;
    let mut odd = rho(1);
    rho_t[1] = odd;
    while t + 5 < n && !(even + odd).is_nan() && even + odd > 0.0 {
        t += 2;
        even = rho(t);
        odd = rho(t + 1);
        if even + odd >= 0.0 {
            rho_t[t] = even;
            rho_t[t + 1] = odd;
        }
    }
    let max_t = t;
    if even > 0.0 {
        rho_t[max_t] = even;
    }
    let mut t = 0;
    while t + 4 <= max_t {
        t += 2;
        if rho_t[t] + rho_t[t + 1] > rho_t[t - 2] + rho_t[t - 1] {
            rho_t[t] = (rho_t[t - 2] + rho_t[t - 1]) / 2.0;
            rho_t[t + 1] = rho_t[t];
        }
    }
    let tau = -1.0 + 2.0 * rho_t[..max_t].iter().sum::<f64>() + rho_t[max_t];
    // Stan's lower bound on tau keeps ESS <= m n log10(m n).
    let mn = m * nf;
    let tau = tau.max(1.0 / mn.log10());
    mn / tau
}

/// Inverse standard normal CDF (Acklam's rational approximation, relative
/// error below 1.2e-9; ample for rank normalisation).
pub fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383_577_518_672_69e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let lo = 0.02425;
    if p < lo {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - lo {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -inv_normal_cdf(1.0 - p)
    }
}

/// Rank-normalise the pooled draws: average ranks for ties, then
/// `z = Phi^-1((r - 3/8) / (S + 1/4))` (Blom), reshaped back into chains.
pub fn rank_normalize(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let all: Vec<f64> = chains.iter().flatten().copied().collect();
    let s = all.len();
    let mut idx: Vec<usize> = (0..s).collect();
    idx.sort_by(|&a, &b| all[a].total_cmp(&all[b]));
    let mut rank = vec![0.0; s];
    let mut i = 0;
    while i < s {
        let mut j = i;
        while j + 1 < s && all[idx[j + 1]] == all[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            rank[idx[k]] = avg;
        }
        i = j + 1;
    }
    let z: Vec<f64> = rank
        .iter()
        .map(|r| inv_normal_cdf((r - 0.375) / (s as f64 + 0.25)))
        .collect();
    let mut out = vec![];
    let mut pos = 0;
    for c in chains {
        out.push(z[pos..pos + c.len()].to_vec());
        pos += c.len();
    }
    out
}

fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Bulk ESS: ESS of the split, then rank-normalised draws
/// (`posterior::ess_bulk`).
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    ess_of_split(&rank_normalize(&split(chains)))
}

/// Rank-normalised split R-hat: the larger of the bulk R-hat and the R-hat
/// of the folded draws `|x - median|` (`posterior::rhat`).
pub fn rhat(chains: &[Vec<f64>]) -> f64 {
    let all: Vec<f64> = chains.iter().flatten().copied().collect();
    let med = median(&all);
    let folded: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.iter().map(|x| (x - med).abs()).collect())
        .collect();
    let bulk = rhat_of_split(&rank_normalize(&split(chains)));
    let tail = rhat_of_split(&rank_normalize(&split(&folded)));
    bulk.max(tail)
}

/// Monte Carlo standard error of the mean.
pub fn mcse_mean(chains: &[Vec<f64>]) -> f64 {
    let all: Vec<f64> = chains.iter().flatten().copied().collect();
    (var(&all) / ess(chains)).sqrt()
}

/// One row of a moment check.
#[derive(Clone, Debug)]
pub struct Moment {
    pub name: String,
    pub estimate: f64,
    pub truth: f64,
    /// From the basic ESS ([`mcse_mean`]).
    pub mcse: f64,
    pub z: f64,
    /// Rank-normalised R-hat ([`rhat`]).
    pub rhat: f64,
    /// Bulk ESS ([`ess_bulk`]).
    pub ess_bulk: f64,
}

/// `z = (estimate - truth) / MCSE` for the mean of `f(x)` over the draws.
pub fn moment(name: &str, chains: &[Vec<f64>], truth: f64) -> Moment {
    let all: Vec<f64> = chains.iter().flatten().copied().collect();
    let estimate = mean(&all);
    let mcse = mcse_mean(chains);
    Moment {
        name: name.to_string(),
        estimate,
        truth,
        mcse,
        z: (estimate - truth) / mcse,
        rhat: rhat(chains),
        ess_bulk: ess_bulk(chains),
    }
}

/// Like [`moment`], against a reference that has its own Monte Carlo error
/// (a long CmdStan run shipped with the suite): `z = (estimate - reference)
/// / sqrt(mcse^2 + reference_mcse^2)`.
pub fn moment_vs_reference(
    name: &str,
    chains: &[Vec<f64>],
    reference: f64,
    reference_mcse: f64,
) -> Moment {
    let mut m = moment(name, chains, reference);
    m.z = (m.estimate - reference) / (m.mcse * m.mcse + reference_mcse * reference_mcse).sqrt();
    m
}

/// Chain-health thresholds (`docs/sampler/laplace_kernels.md`, section 4).
pub const MAX_RHAT: f64 = 1.01;
pub const MIN_ESS_BULK: f64 = 400.0;
/// A moment check passes when every |z| is below this.
pub const MAX_ABS_Z: f64 = 4.0;

/// The outcome of a moment check. A z-score only means something when the
/// chains are healthy, so health is judged first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every quantity healthy and every |z| < 4.
    Pass,
    /// Healthy chains, and some |z| >= 4: the kernel targets the wrong
    /// distribution.
    Fail,
    /// Some quantity has R-hat >= 1.01 or bulk ESS <= 400 (or a non-finite
    /// diagnostic): the run says nothing either way. Run longer.
    DidNotConverge,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "FAIL",
            Verdict::DidNotConverge => "did-not-converge",
        }
    }
}

/// Is this row's chain health acceptable?
pub fn healthy(m: &Moment) -> bool {
    m.rhat.is_finite() && m.ess_bulk.is_finite() && m.rhat < MAX_RHAT && m.ess_bulk > MIN_ESS_BULK
}

/// Judge a moment check: did-not-converge if any row is unhealthy, else
/// fail if any |z| >= 4, else pass.
pub fn verdict(rows: &[Moment]) -> Verdict {
    if rows.iter().any(|m| !healthy(m)) {
        Verdict::DidNotConverge
    } else if rows.iter().any(|m| m.z.is_nan() || m.z.abs() >= MAX_ABS_Z) {
        Verdict::Fail
    } else {
        Verdict::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;

    fn iid(seed: u64, chains: usize, n: usize) -> Vec<Vec<f64>> {
        let mut r = rand::rngs::ChaCha8Rng::seed_from_u64(seed);
        (0..chains)
            .map(|_| (0..n).map(|_| r.sample(StandardNormal)).collect())
            .collect()
    }

    /// Deterministic draws R can reproduce exactly; expected values are
    /// from posterior 1.7.0 (`ess_bulk`, `rhat`, `ess_basic`).
    fn fixture(n: usize, f: impl Fn(f64, f64) -> f64) -> Vec<Vec<f64>> {
        (0..4)
            .map(|c| (1..=n).map(|i| f(i as f64, c as f64)).collect())
            .collect()
    }

    #[test]
    fn matches_posterior_on_fixed_draws() {
        let x = fixture(300, |i, c| {
            (0.37 * i + c).sin() + 0.3 * (1.7 * i * (c + 1.0)).cos() + 0.2 * c
        });
        let y = fixture(301, |i, c| (0.11 * i + c).sin() + 0.4 * (2.3 * i).cos());
        let close = |a: f64, b: f64, tol: f64| (a / b - 1.0).abs() < tol;
        assert!(
            close(ess_bulk(&x), 241.8153627016, 1e-6),
            "{}",
            ess_bulk(&x)
        );
        assert!(close(rhat(&x), 1.0585435113, 1e-6), "{}", rhat(&x));
        assert!(close(ess(&x), 241.0775203804, 1e-6), "{}", ess(&x));
        assert!(
            close(ess_bulk(&y), 87.78882381442, 1e-6),
            "{}",
            ess_bulk(&y)
        );
        assert!(close(rhat(&y), 1.00230326658, 1e-6), "{}", rhat(&y));
        assert!(close(ess(&y), 79.97965192844, 1e-6), "{}", ess(&y));
    }

    #[test]
    fn verdict_checks_health_before_z() {
        let row = |z: f64, rhat: f64, ess_bulk: f64| Moment {
            name: "m".into(),
            estimate: 0.0,
            truth: 0.0,
            mcse: 1.0,
            z,
            rhat,
            ess_bulk,
        };
        assert_eq!(verdict(&[row(1.0, 1.001, 900.0)]), Verdict::Pass);
        assert_eq!(verdict(&[row(5.0, 1.001, 900.0)]), Verdict::Fail);
        // A huge z on unhealthy chains is not a failure: it is no evidence.
        assert_eq!(verdict(&[row(9.0, 1.2, 900.0)]), Verdict::DidNotConverge);
        assert_eq!(verdict(&[row(0.1, 1.001, 400.0)]), Verdict::DidNotConverge);
        assert_eq!(
            verdict(&[row(0.1, f64::NAN, 900.0)]),
            Verdict::DidNotConverge
        );
        assert_eq!(verdict(&[row(f64::NAN, 1.001, 900.0)]), Verdict::Fail);
    }

    #[test]
    fn inverse_normal_cdf_known_values() {
        assert!(inv_normal_cdf(0.5).abs() < 1e-12);
        assert!((inv_normal_cdf(0.975) - 1.959963984540054).abs() < 1e-8);
        assert!((inv_normal_cdf(0.001) + 3.090232306167813).abs() < 1e-8);
    }

    #[test]
    fn iid_draws_have_ess_near_n_and_rhat_near_1() {
        let c = iid(1, 4, 1000);
        let e = ess(&c);
        assert!(e > 3000.0 && e < 5000.0, "ess {e}");
        assert!((split_rhat(&c) - 1.0).abs() < 0.01);
    }

    #[test]
    fn ar1_ess_matches_theory() {
        // AR(1) with phi: ESS/N = (1 - phi) / (1 + phi).
        let phi: f64 = 0.9;
        let mut r = rand::rngs::ChaCha8Rng::seed_from_u64(2);
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                let mut x = 0.0;
                (0..5000)
                    .map(|_| {
                        let e: f64 = r.sample(StandardNormal);
                        x = phi * x + (1.0 - phi * phi).sqrt() * e;
                        x
                    })
                    .collect()
            })
            .collect();
        let rel = ess(&chains) / 20000.0;
        let theory = (1.0 - phi) / (1.0 + phi);
        assert!((rel / theory - 1.0).abs() < 0.15, "{rel} vs {theory}");
    }

    #[test]
    fn rhat_flags_chains_in_different_places() {
        let mut c = iid(3, 4, 1000);
        for x in c[0].iter_mut() {
            *x += 3.0;
        }
        assert!(split_rhat(&c) > 1.1);
    }
}
