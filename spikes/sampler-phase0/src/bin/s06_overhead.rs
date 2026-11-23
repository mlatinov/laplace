//! 0.6 — time per gradient through BridgeStan on the gp library model, at
//! CmdStan's own posterior draws (out/cmdstan/gp-*.csv), unconstrained.
use sampler_phase0_spike::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let lib = library("gp")?;
    let m = model(&lib, "gp", 1)?;
    let n = m.param_unc_num();
    // read alpha, rho, sigma from CmdStan CSVs
    let mut pts = vec![];
    for c in 1..=4 {
        let txt = std::fs::read_to_string(format!("out/cmdstan/gp-{c}.csv"))?;
        let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
        let head: Vec<&str> = lines.next().unwrap().split(',').collect();
        let idx: Vec<usize> = ["alpha", "rho", "sigma"]
            .iter()
            .map(|v| head.iter().position(|h| h == v).unwrap())
            .collect();
        for l in lines {
            let f: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            let theta: Vec<f64> = idx.iter().map(|&i| f[i]).collect();
            let mut q = vec![0.0; n];
            m.param_unconstrain(&theta, &mut q)?;
            pts.push(q);
        }
    }
    let mut g = vec![0.0; n];
    // warm up caches
    for q in &pts[..100] {
        m.log_density_gradient(q, true, true, &mut g)?;
    }
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        for q in &pts {
            std::hint::black_box(m.log_density_gradient(q, true, true, &mut g)?);
        }
    }
    let per_grad = t.elapsed().as_secs_f64() / (reps * pts.len()) as f64;
    let t = Instant::now();
    for _ in 0..reps {
        for q in &pts {
            std::hint::black_box(m.log_density(q, true, true)?);
        }
    }
    let per_lp = t.elapsed().as_secs_f64() / (reps * pts.len()) as f64;
    println!("gp (N=40): BridgeStan log_density_gradient = {:.2} us/eval, log_density = {:.2} us/eval  ({} evals each)",
             per_grad * 1e6, per_lp * 1e6, reps * pts.len());
    Ok(())
}
