//! `laplace-sample`: the optional custom-sampler runtime for laplace.
//!
//! A compiled `.stan` model is loaded through BridgeStan (log density and
//! gradient on the unconstrained scale) and sampled with one of the kernels
//! in [`kernels`]. Output is CmdStan-shaped CSV that `cmdstanr::as_cmdstan_fit()`
//! reads (see `docs/sampler/phase-0-report.md` for the header contract).
//!
//! The `laplace` compiler never links this crate; `laplace sample ...` finds
//! the `laplace-sample` binary on `PATH` and forwards its arguments (D1).

pub mod bridgestan_src;
pub mod kernel;
pub mod model;
pub mod state;
pub mod target;

/// Version recorded in every CSV this binary writes.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Lower-case hex SHA-256 (provenance hashes, download verification).
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
