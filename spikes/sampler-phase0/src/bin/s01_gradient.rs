//! 0.1 — BridgeStan from Rust: gradient vs central finite differences.
use rand::{RngExt, SeedableRng};
use rand::rngs::ChaCha8Rng;
use sampler_phase0_spike::*;

fn main() -> anyhow::Result<()> {
    let tol = 1e-6;
    let mut all_ok = true;
    for name in ["std_normal", "simplex", "mapping"] {
        let lib = library(name)?;
        let m = model(&lib, name, 1)?;
        let n = m.param_unc_num();
        let mut rng = ChaCha8Rng::seed_from_u64(20261123);
        let mut worst = 0f64;
        for _ in 0..50 {
            let q: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();
            let mut g = vec![0.0; n];
            let lp = m.log_density_gradient(&q, true, true, &mut g)?;
            // propto=true needs autodiff types; log_density with propto=true
            // drops the same constants, so the FD target is consistent.
            let lp0 = m.log_density(&q, true, true)?;
            assert!((lp - lp0).abs() <= 1e-12 * lp.abs().max(1.0), "lp mismatch");
            for i in 0..n {
                let h = 1e-5 * q[i].abs().max(1.0);
                let mut qp = q.clone();
                let mut qm = q.clone();
                qp[i] += h;
                qm[i] -= h;
                let fd = (m.log_density(&qp, true, true)? - m.log_density(&qm, true, true)?) / (2.0 * h);
                let rel = (g[i] - fd).abs() / g[i].abs().max(1.0);
                worst = worst.max(rel);
            }
        }
        let ok = worst < tol;
        all_ok &= ok;
        println!("{name:>10}: dim_unc={n:>2}  max rel err over 50 points = {worst:.2e}  {}",
                 if ok { "PASS" } else { "FAIL" });
    }
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
