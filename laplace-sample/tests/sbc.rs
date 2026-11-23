//! Phase 5.2 check: simulation-based calibration passes the built-in NUTS
//! and fails the deliberately broken MALA (tests/fixtures/broken_mala.lapkernel,
//! the shipped MALA with its Metropolis-Hastings correction dropped).
//! Needs BridgeStan and a C++ toolchain, so `--ignored`:
//!
//!   cargo test -p laplace-sample --release --test sbc -- --ignored --nocapture

use laplace_sample::runner::Sampler;
use laplace_sample::sbc::{render, run_sbc, SbcSettings};

fn sbc(spec: &str) -> bool {
    let sampler = Sampler::from_arg(spec, &[]).unwrap();
    let r = run_sbc(&sampler, SbcSettings::default(), |_| {}).unwrap();
    println!("{}", render(spec, &r));
    r.pass()
}

#[test]
#[ignore]
fn sbc_passes_the_built_in_nuts() {
    assert!(sbc("nuts"));
}

#[test]
#[ignore]
fn sbc_passes_the_shipped_mala_kernel() {
    assert!(sbc(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/kernels/mala.lapkernel"
    )));
}

#[test]
#[ignore]
fn sbc_fails_mala_without_its_correction() {
    assert!(!sbc(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/broken_mala.lapkernel"
    )));
}
