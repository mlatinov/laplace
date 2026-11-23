//! 0.3 — 4 threads, one model instance each, gradients in a loop.
use rand::{RngExt, SeedableRng};
use rand::rngs::ChaCha8Rng;
use sampler_phase0_spike::*;
use std::hash::{DefaultHasher, Hash, Hasher};

const POINTS: usize = 200;
const SWEEPS: usize = 500; // 100k gradient evaluations per thread

fn run(lib: &bridgestan::StanLibrary, name: &str, pts: &[Vec<f64>]) -> anyhow::Result<u64> {
    let m = model(lib, name, 1)?;
    let mut h = DefaultHasher::new();
    let mut g = vec![0.0; m.param_unc_num()];
    for sweep in 0..SWEEPS {
        for q in pts {
            let lp = m.log_density_gradient(q, true, true, &mut g)?;
            if sweep == 0 || sweep == SWEEPS - 1 {
                lp.to_bits().hash(&mut h);
                for x in &g {
                    x.to_bits().hash(&mut h);
                }
            }
        }
    }
    Ok(h.finish())
}

fn main() -> anyhow::Result<()> {
    for name in ["simplex", "mapping"] {
        let lib = library(name)?;
        let n = model(&lib, name, 1)?.param_unc_num();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let pts: Vec<Vec<f64>> = (0..POINTS)
            .map(|_| (0..n).map(|_| rng.random_range(-2.0..2.0)).collect())
            .collect();
        let baseline = run(&lib, name, &pts)?;
        let t = std::time::Instant::now();
        let hashes: Vec<u64> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..4).map(|_| s.spawn(|| run(&lib, name, &pts))).collect();
            hs.into_iter().map(|h| h.join().expect("thread panicked").unwrap()).collect()
        });
        let ok = hashes.iter().all(|&h| h == baseline);
        println!("{name:>8}: 4 threads x {} grads in {:.2}s; bitwise-identical to single-thread baseline: {}",
                 POINTS * SWEEPS, t.elapsed().as_secs_f64(), if ok { "PASS" } else { "FAIL" });
        if !ok {
            println!("  baseline {baseline:x}, threads {hashes:x?}");
            std::process::exit(1);
        }
    }
    Ok(())
}
