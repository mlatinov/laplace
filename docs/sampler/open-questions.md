# Custom samplers: open questions

Questions that `laplace_kernels.md` does not settle. Each has the options seen
and a recommendation. Answered questions move to the bottom with the answer.

## Open

### Q1 — Where the `laplace-sample` crate lives (Phase 1.1)
The plan names `laplace_tools` as the main project. That repo holds the LSP and
VS Code extension, is not a Cargo workspace, and is not where the `laplace`
binary is built.
- (a) Make the `laplace` repo root a workspace with members `.` and `laplace-sample`.
- (b) Put a standalone crate in the `laplace` repo with its own `[workspace]`.
- (c) Create a separate repository.

**Recommendation: (a).** One lockfile and `cargo test --workspace`. The
`laplace` binary and its output are unchanged.

### Q2 — How to carry the nuts-rs `refresh_position` patch (D8)
nuts-rs's `set_position` re-initialises step size and mass matrix, which
biases block sweeps (Phase 0 report, 0.5). A 5-line `Chain::refresh_position`
fixes it.
- (a) Open an upstream PR to `pymc-devs/nuts-rs`, and depend on a pinned git fork until it merges.
- (b) Keep a vendored copy permanently.
- (c) Port NUTS.

**Recommendation: (a).** The PR is outward-facing, so it needs Metodi's go-ahead.

### Q3 — CSV provenance key format (Phase 1.7, 4.2)
cmdstanr keeps only the last of repeated `# key = value` lines.
- (a) Use one key per block: `# laplace_kernel_<label> = …`.
- (b) Put all kernels on one line.
- (c) Keep repeated lines and accept that only the last one shows in `$metadata()`.

**Recommendation: (a).**

### Q4 — `energy__` for kernels with no Hamiltonian (Phase 1.7)
`as_cmdstan_fit()` needs the column to exist.
- (a) Write `NaN`, so `ebfmi` is `NA`.
- (b) Write `-lp__`, so `ebfmi` is a number that means nothing.

**Recommendation: (a).**

## Answered
(none yet)
