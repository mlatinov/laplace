# Custom samplers: open questions

Questions that `laplace_kernels.md` does not settle. Each has the options seen
and a recommendation. Answered questions move to the bottom with the answer.

## Open

### Q11 — Post-v1 built-ins for the kernel context (Phase 2.3)
The built-in kernel context is one registry
(`laplace-sample/src/lapkernel/builtins.rs`). Each entry is a name, a type
rule, an implementation and its documented RNG use, so each of these is one
addition. **Not in v1**, recorded as candidates:
- `hessian(vector x)`: the dense Hessian at x (BridgeStan
  `log_density_hessian`). Opens Riemannian/Newton-type kernels; cost is
  O(dim) gradients or autodiff Hessian, so it needs a documented cost model.
- `hessian_vector_product(vector x, vector v)`: BridgeStan
  `log_density_hessian_vector_product`. Cheaper than the full Hessian.
- `is_warmup`: int, 1 during warmup. Lets a kernel behave differently while
  adapting. Note the Phase 0 lesson: anything that changes behaviour after
  warmup based on the current state breaks invariance, so a kernel using it
  must not re-tune after warmup.
- `iteration`: int, the iteration number. Same caution as `is_warmup`.

**Recommendation:** add them only when a kernel needs one, each with
registry unit tests (value and RNG use) like the v1 entries.

## Answered

### Q2 — How to carry the nuts-rs `refresh_position` patch
**Answer (Phase 3):** vendor nuts-rs with the patch inside the repo, keeping
its MIT license file; no GitHub fork, no contact with upstream. Whether to
upstream it or keep a fork is decided before merge.
**Done:** `laplace-sample/vendor/nuts-rs-0.19.1` (+ `VENDORED.md` and the
patch), wired as `nuts-rs = "=0.19.1"` patched to the vendored copy by the
workspace root's `[patch.crates-io]`, so it is not a workspace member and
fmt/clippy never touch upstream code.

### Q12 — The `mala` Metropolis-ratio regrouping
**Answer:** not reviewed; go with the recommendation unless it affects
Phase 3. It does not (it only changes rounding inside `mala`), so (a): the
regrouping stays. Explained in the Phase 3 report.

### Q13 — `metric` in the context
**Answer:** (a). Context `metric` is the kernel's own `@adapt(metric)` value,
else all ones. In Phase 3 it is per block, sized to the block dimension.

### Q14 — `target_accept` for a `.lapkernel`
**Answer:** (b), required. `@adapt(step_size, target_accept = 0.574)` carries
the author's default; `adapt(k, target_accept = ...)` in a `.lapsampler`
overrides it; a missing one is a check error.
**Done:** the `.lapkernel` checker rejects `@adapt(step_size)` without
`target_accept` (so `laplace sampler check` reports it); the 0.8 fallback is
gone.

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
**Answer:** deferred to Phase 5 (release assets). **Phase 5 go:** prepare
the release workflow for laplace-sample binaries; do not run or publish it.

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

### Q15 — Hybrids against plain NUTS
**Answer (Phase 4 go):** report only, no warning threshold. Report the
ratio to plain NUTS, and ESS per second alongside ESS per gradient.
Density-only and gradient evaluations are counted apart (`n_logp__`).

### Q16 — Unsuffixed CSV columns for a single whole-model block
**Answer (Phase 4 go):** general: any single block covering the whole model
writes the unsuffixed columns.

### Q17 — Compiler version for kernel packages
**Answer (Phase 5 go):** kernel packages ship in 0.3.0, after Phase 5 and
the editor-support update. The branch stays at 0.3.0. Custom samplers are
labelled "experimental" in all 0.3.0 docs. Nothing is tagged or released.

### Q18 — `laplace new kernel` package naming
**Answer:** yes, `--package <name>`; the default is the kernel's name.

### Q19 — Interpreter overhead on cheap densities
**Answer:** agreed, no interpreter optimization before v1. The docs state
the measured overhead and when to write a Rust `Kernel` instead.

### Q21 — Barker's adaptation lands below its target
**Answer (Phase 5 review):** run SBC on langevin::barker; if it passes, the
suite failures are slow mixing: tune its test file and document the
adaptation. **Done:** SBC passes (200 replicates, same settings as MALA's;
`phase-5/sbc-barker.txt`). With `tests/barker.test.toml` at 10000 draws
(100000 on the correlated Gaussian) barker passes the must-pass tier and
its release gate (`phase-5/release-gate-tuned.txt`); the kernel and the
test file document that acceptance settles at 0.25-0.37, below 0.4.

### Q20 — `laplace-sample`'s version and distribution
**Answer (Phase 5 review):** laplace-sample takes the compiler's version
(0.3.0) and moves in lockstep; `laplace` warns on a mismatch when it hands
a command over. **Done:** both packages inherit `[workspace.package]
version` from the root Cargo.toml. Before forwarding (and in the release
gate) `laplace` runs `laplace-sample --version` and, if it is not its own
version, warns with both versions and the install command (tested: match,
mismatch, no answer). Built-in kernels' provenance now reads
`<kernel> 0.3.0`. Installer and `self-update` support stays a later
decision.
