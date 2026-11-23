//! `cargo run --release --bin compile -- std_normal simplex ...`
use sampler_phase0_spike::*;

fn main() -> anyhow::Result<()> {
    for name in std::env::args().skip(1) {
        let t = std::time::Instant::now();
        let so = compile(&model_path(&name))?;
        println!("{name}: {} ({:.1}s)", so.display(), t.elapsed().as_secs_f64());
    }
    Ok(())
}
