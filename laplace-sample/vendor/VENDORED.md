# Vendored: nuts-rs 0.19.1 + `refresh_position`

`nuts-rs-0.19.1/` is the nuts-rs 0.19.1 crate as published on crates.io
(MIT, see `nuts-rs-0.19.1/LICENSE`; upstream https://github.com/pymc-devs/nuts-rs),
with one addition: `Chain::refresh_position` in `src/chain.rs` (search for
`LAPLACE SPIKE PATCH`). The exact diff is `nuts-rs-refresh_position.patch`.

Why: `nuts()` as one block of a sweep (Phase 3.4) must move its chain after
another block changes the log density, *without* re-initialising step size
and mass matrix. Upstream's `set_position` does re-initialise them, which
after warmup makes the kernel depend on the current state and biases it
(Phase 0.5: `docs/sampler/phase-0-report.md`).

Decision record: Q2 in `docs/sampler/open-questions.md`. Vendored in-repo as
a path dependency for Phase 3; whether to upstream the method or keep a
fork is decided before merge. Do not edit anything here except to re-apply
the patch on a new nuts-rs version.

How it is wired: `laplace-sample` depends on `nuts-rs = "=0.19.1"` and the
workspace root's `[patch.crates-io]` points it here. A patch source is not a
workspace member, so `cargo fmt` and `cargo clippy` never touch the
upstream code. (A plain path dependency inside the workspace would be a
member, and `exclude` does not change that; checked with Cargo 1.95.)
