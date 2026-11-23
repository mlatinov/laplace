//! Convergence diagnostics for the statistical tests (and, in Phase 5,
//! `laplace sampler test`).
//!
//! `chains[c][i]` is draw `i` of chain `c` for one scalar quantity. Chains
//! are split in half first, as in Stan. ESS follows BDA3 / Stan: Geyer's
//! initial monotone sequence on the multi-chain autocorrelation, *without*
//! rank normalisation (that is `posterior::ess_basic`, which tracks
//! `ess_bulk` closely for well-behaved targets like the reference ones).

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn var(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)
}

fn split(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = vec![];
    for c in chains {
        let h = c.len() / 2;
        // An odd draw in the middle is dropped, as in Stan.
        out.push(c[..h].to_vec());
        out.push(c[c.len() - h..].to_vec());
    }
    out
}

/// Split R-hat (Gelman et al. BDA3, no rank normalisation).
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    let s = split(chains);
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

/// Effective sample size of the pooled draws.
pub fn ess(chains: &[Vec<f64>]) -> f64 {
    let s = split(chains);
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
    // Geyer: sum consecutive pairs while positive, forced monotone.
    let mut tau = -1.0;
    let mut prev_pair = f64::INFINITY;
    let mut t = 0;
    while t + 1 < n {
        let mut pair = rho(t) + rho(t + 1);
        if pair <= 0.0 {
            break;
        }
        pair = pair.min(prev_pair);
        tau += 2.0 * pair;
        prev_pair = pair;
        t += 2;
    }
    // Stan's lower bound on tau keeps ESS <= m n log10(m n).
    let mn = m * nf;
    let tau = tau.max(1.0 / mn.log10());
    mn / tau
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
    pub mcse: f64,
    pub z: f64,
    pub rhat: f64,
    pub ess: f64,
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
        rhat: split_rhat(chains),
        ess: ess(chains),
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
