# Phase 0 spike: BridgeStan, CSV and nuts-rs

Throwaway code that de-risks the custom-sampler plan (`laplace_kernels.md`,
Phase 0). It is a standalone crate (its own `[workspace]`), never a member of
the compiler build, and nothing in `src/` of the compiler imports it.
Findings are in `docs/sampler/phase-0-report.md`; raw outputs in `results/`.

## Prerequisites

- BridgeStan 2.9.0 source at `~/.laplace/bridgestan/bridgestan-2.9.0`
  (or `$LAPLACE_BRIDGESTAN`):
  ```sh
  mkdir -p ~/.laplace/bridgestan && cd ~/.laplace/bridgestan
  curl -L -o bs.tar.gz https://github.com/roualdes/bridgestan/releases/download/v2.9.0/bridgestan-2.9.0.tar.gz
  echo "b6034f83fb2ffb71ce7be9525b7dded1a225e9a4eaf4b75d4c7e97e3ade05a4c  bs.tar.gz" | sha256sum -c
  tar xzf bs.tar.gz && rm bs.tar.gz
  ```
- R with `cmdstanr` + CmdStan (2.39.0 was used) and `posterior`.

## Running (from this directory)

```sh
cargo build --release
./target/release/compile std_normal simplex mapping blocks gp   # BridgeStan .so files
Rscript R/cmdstan_reference.R   # CmdStan references + models/gp.json
./target/release/s01_gradient   # 0.1 gradient vs finite differences
./target/release/s02_mapping    # 0.2 unconstrained-name mapping
./target/release/s03_threads    # 0.3 4 threads, bitwise identical
./target/release/s05_nuts       # 0.5 nuts-rs full model + Gibbs variants
Rscript R/check_nuts.R          # 0.5 moment checks
Rscript R/csv_header.R          # 0.4 minimal header + probes
Rscript R/csv_probe2.R          # 0.4 follow-up probes
./target/release/s06_overhead   # 0.6 BridgeStan time per gradient (gp)
```

## The vendored nuts-rs

`vendor/nuts-rs-0.19.1` is nuts-rs 0.19.1 from crates.io plus one method,
`Chain::refresh_position` (search for `LAPLACE SPIKE PATCH` in
`src/chain.rs`). It is wired in with `[patch.crates-io]` in `Cargo.toml`.
That method is the whole of the D8 question; see the report.
