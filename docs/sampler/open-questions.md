# Custom samplers: open questions

Questions that `laplace_kernels.md` does not settle. Each has the options seen
and a recommendation. Answered questions move to the bottom with the answer.

## Open

### Q2 — How to carry the nuts-rs `refresh_position` patch (D8) — **deferred to Phase 3**
nuts-rs's `set_position` re-initialises step size and mass matrix, which
biases block sweeps (Phase 0 report, 0.5). A 5-line `Chain::refresh_position`
fixes it.
- (a) Open an upstream PR to `pymc-devs/nuts-rs`, and depend on a pinned git fork until it merges.
- (b) Keep a vendored copy permanently.
- (c) Port NUTS.

**Status (after STOP 0):** deferred. Phase 1 runs NUTS on the whole model
only, so it uses **upstream nuts-rs, unpatched**. Do not contact upstream. The
vendored patch stays on the spike branch (`sampler/phase-0`,
`spikes/sampler-phase0/vendor/`) for Phase 3.4, where this question is
decided.

### Q6 — CI for `laplace-sample` (Phase 1)
CI's clippy and tests run on the root package only; `cargo fmt --check`
already covers the workspace. `laplace-sample`'s unit tests need no
BridgeStan.
- (a) Add `cargo clippy -p laplace-sample --all-targets -- -D warnings` and
  `cargo test -p laplace-sample` to `ci.yml`.
- (b) Leave CI as it is.

**Recommendation: (a).**

### Q7 — Two deviations from the plan's adaptation and HMC (Phase 1.4, 1.5)
- **Robbins–Monro finish.** Stan's dual averaging left MALA's post-warmup
  acceptance at about 0.69 against 0.574, failing the 1.5 check. rwm, mala
  and hmc keep Stan's windows and dual averaging, then settle the step size
  with Robbins–Monro over a 100-iteration term buffer.
- **HMC path-length jitter.** Fixed L gave R-hat 1.36 on the standard
  normal. L is now drawn uniformly from 1..=2·steps−1 by default
  (`--set jitter=false` turns it off).

Options:
- (a) Accept both.
- (b) Revert to pure Stan dual averaging and relax the 1.5 check.
- (c) Other.

**Recommendation: (a).** Details in the Phase 1 report.

### Q8 — The Q5 message wording for `hmc` (cmdlaplacer)
Static HMC reports real divergences; only tree depth is NaN. The specified
text says divergences don't apply.
- (a) Use per-sampler wording ("Tree depth doesn't apply to hmc; …").
- (b) Keep the exact text for every non-NUTS sampler.

**Recommendation: (a).**

### Q9 — Which C++ compiler BridgeStan builds with
BridgeStan uses the system default (g++ here). The CmdStan install uses
clang++ via `make/local`. On the gp model, the same BridgeStan build is 1.4×
faster per gradient with clang++.
- (a) Document it and honour `CXX` (already part of the cache key).
- (b) Default to the CmdStan install's `make/local`.
- (c) Nothing.

**Recommendation: (a).**

### Q10 — Distributing `laplace-sample`
It is not in the release assets or `self-update`; the install hint is
`cargo install --path laplace-sample`.
- (a) Add it to the release workflow in Phase 5.
- (b) Keep it source-only.

**Recommendation: (a), in Phase 5.**

## Answered

### Q1 — Where the `laplace-sample` crate lives (Phase 1.1)
**Answer: (a).** The `laplace` repo becomes a Cargo workspace with members `.`
and `laplace-sample`. `laplace_kernels.md` "Locations" is corrected:
compiler and sampler work go in `laplace`, and `laplace_tools` holds the LSP and VS Code
extension and is not touched by this plan.

### Q3 — CSV provenance key format (Phase 1.7, 4.2)
**Answer: (a).** Write one line per block, plus one line for the sampler:
```
# laplace_kernel_<label> = <name> <version> <hash>
# laplace_sampler = <name> (sha256: …)
```

### Q4 — NUTS columns for kernels that are not NUTS (Phase 1.7)
**Answer: (a), extended.** Write `NaN` in *every* NUTS column that does not
apply (`energy__`, `treedepth__`, `n_leapfrog__`, `divergent__`), because 0
would falsely say "checked, none found". If NaN breaks `$summary()`,
`$draws()` or `$diagnostic_summary()`, use 0 only for the columns that break, and list them in the report.

**Verified after review** (`spikes/sampler-phase0/R/csv_probe3.R`,
`results/csv_probe3.log`): NaN in all four columns, together and one at a
time. `as_cmdstan_fit()`, `$summary()`, `$draws()`, `$diagnostic_summary()`
and `$sampler_diagnostics()` all run, so **no column needs 0**. What the
user sees:
- `num_divergent` = NaN (honest);
- `ebfmi` = NA, with the warning "E-BFMI not computed because 'energy__' contains NAs" (honest);
- `num_max_treedepth` = 0 even when `treedepth__` is all NaN (see Q5).

### Q5 — `num_max_treedepth` reads 0 for non-NUTS runs
cmdstanr drops NAs when it counts tree-depth hits, so an all-NaN
`treedepth__` still reports `num_max_treedepth = 0`. That falsely says "checked, none found",
and the CSV cannot change it.

**Answer:** handle it in cmdlaplacer (step 1.9). When the sampler is not NUTS,
`laplace_sample()` shows one `message()` (not `warning()`) after the fit:
"Tree depth and divergence diagnostics don't apply to <sampler>; see n_grad__
and accept_stat__ instead." A testthat test covers it.
