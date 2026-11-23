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

## Answered

### Q6 — CI for `laplace-sample`
**Answer:** CI runs clippy and the fast `laplace-sample` tests on every push;
the `--ignored` statistical tests run on a weekly schedule and on manual
dispatch, with `~/.laplace/bridgestan/` cached.
**Done:** `.github/workflows/sampler.yml` (every push and PR, Ubuntu and
macOS) and `.github/workflows/sampler-statistical.yml` (Mondays 04:17 UTC
and `workflow_dispatch`; cache key `bridgestan-<version>-<os>`, guarded by a
unit test against the pinned version). `ci.yml` is unchanged.

### Q7 — Robbins–Monro finish and HMC path-length jitter
**Answer:** HMC jitter accepted. Robbins–Monro was to be kept only if its ESS
per gradient was at least dual averaging's for mala and hmc on both
reference targets.
**Outcome: reverted.** Over 5 seeds, RM won on three pairs and lost on hmc /
standard normal (0.0274 vs 0.0311 ESS per gradient, about 2.6 standard errors).
All of rwm, mala and hmc use Stan's dual averaging again, and the 1.5 check
is widened to +-0.15 (MALA now settles at 0.718 on the test seed). Numbers
are in the Phase 1 report, "Follow-up".

### Q8 — The non-NUTS note in cmdlaplacer
**Answer:** build the message from the data: name only the NUTS columns that
are all-NaN, and always mention tree depth when its column is NaN.
**Done:** `laplace_sample()` reads the sampler diagnostics after the fit,
runs cmdstanr's divergence / tree-depth / E-BFMI checks only for columns
with values, and names the all-NaN ones (mala: all four; hmc: tree depth
only).

### Q9 — Which C++ compiler builds the model
**Answer:** `CXX` or `LAPLACE_CXX` from the environment, then the `CXX` from
the detected CmdStan's `make/local`, then the system default; print the
compiler used; document it.
**Done:** `laplace-sample/src/compiler.rs`. `LAPLACE_CXX` is checked before
`CXX`, since it is the laplace-specific setting. CmdStan is `CMDSTAN`, else the
newest `~/.cmdstan/cmdstan-<version>`. The choice is printed as
`Compiler: ...` on every run and is part of the cache key. Documented in
`laplace-sample/README.md`.

### Q10 — Distributing `laplace-sample`
**Answer:** deferred to Phase 5 (release assets).

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
