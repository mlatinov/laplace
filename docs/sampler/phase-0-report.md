# Phase 0 report

Spike for the custom-sampler plan (`laplace_kernels.md`). All code is in
`spikes/sampler-phase0/` (a standalone crate, never part of the compiler
build); raw outputs are in `spikes/sampler-phase0/results/`. Branch
`sampler/phase-0`. Nothing outside `spikes/` and `docs/sampler/` changed, and
all 805 existing compiler tests still pass.

**Short version.** All three unknowns are resolved, and none of them blocks the plan.

- **BridgeStan works.** Gradients are exact to about 5e-10, threads are safe, and outputs are bitwise reproducible.
- **D4: keep `as_cmdstan_fit()`.** It needs a 7-line header, and every file must
  carry the seven NUTS columns, *including non-NUTS runs* (padded).
- **D8: partial yes.** nuts-rs works, but only with a one-method patch. Its
  official `set_position` is not just slow for block sweeps, it **biases the
  posterior**.
- **Speed cost.** A BridgeStan gradient costs about 1.24× a CmdStan gradient on the GP model.

## Done
- Step 0.1 — BridgeStan 2.9.0 from Rust, gradient vs central finite differences — `56645d4`
- Step 0.2 — unconstrained-name → parameter mapping, TP/GQ with RNG, round trip — `1fd79f3`
- Step 0.3 — 4 threads × 100k gradients, one model each, bitwise identical — `37d931d`
- Step 0.4 — minimal CmdStan CSV header, laplace additions probed, example file — `53f2eaf`
- Step 0.5 — nuts-rs full-model NUTS vs CmdStan; block-sweep experiment with 3 variants — `545b718`
- Step 0.6 — BridgeStan vs CmdStan time per gradient on the `gps` library model — `841bf38`

## Checks
| Check | Result | Numbers |
| --- | --- | --- |
| 0.1 gradient vs central FD, rel. err < 1e-6 | PASS | max 1.2e-10 (std_normal, 10-d), 4.4e-10 (5-simplex), 5.4e-10 (17-d mixed constraints); 50 random points each |
| 0.2 K-simplex has K−1 unconstrained coords | PASS | 5-simplex → 4; 4-simplex → 3; `cholesky_factor_corr[3]` → 3; round trip error ≤ 6.7e-16 |
| 0.3 4 threads, no crash, identical outputs | PASS | 4 × 100 000 gradients each, hash of every lp and gradient bit-identical to a single-thread baseline (simplex, mapping) |
| 0.4 minimal header found; `$summary()`, `$draws()`, `$diagnostic_summary()` work | PASS | 7 comment lines; working example `docs/sampler/phase-0-csv-example.csv` |
| 0.4 extra `# laplace_…` lines / extra `__` columns harmless | PASS | both fine; see caveat on repeated keys below |
| 0.5 full-model NUTS, std_normal vs exact moments | PASS | max \|z\| 2.05 (mean and E[x²], 10 coords), R-hat ≤ 1.003, ESS ≥ 4904 |
| 0.5 full-model NUTS, std_normal vs CmdStan 4×10 000 | PASS | max \|z\| 1.87 |
| 0.5 full-model NUTS, gp vs CmdStan (same data) | PASS | max \|z\| 0.92; ESS/grad nuts-rs 0.141 vs CmdStan 0.121 |
| 0.5 external position change between draws | **Partial** | works only with a patch, see D8 |
| 0.5 same seed → identical output | PASS | md5 of all nuts-rs and Gibbs CSVs identical across two runs |
| 0.6 overhead (report only) | — | BridgeStan 205 µs/grad vs CmdStan ≈ 166 µs/leapfrog → **≈1.24×** |

### 0.1 / D9: BridgeStan version and fetching
- **Pinned: BridgeStan v2.9.0** (latest, released 2026-07-06), and the `bridgestan`
  crate at `=2.9.0`. The two must move together, because the crate's FFI matches
  that release's C header.
- **Source:** `https://github.com/roualdes/bridgestan/releases/download/v2.9.0/bridgestan-2.9.0.tar.gz`,
  sha256 `b6034f83fb2ffb71ce7be9525b7dded1a225e9a4eaf4b75d4c7e97e3ade05a4c`,
  35 MB download, 248 MB unpacked, at `~/.laplace/bridgestan/bridgestan-2.9.0/`.
- **Recommendation for Phase 1.2:** write our own fetch (`curl` + `tar` +
  sha256 check, same approach as `self-update`) and **do not** enable the
  crate's `download-bridgestan-src` feature. That feature hard-codes
  `~/.bridgestan/` (D9 says `~/.laplace/bridgestan/`) and pulls in `ureq`/TLS,
  which the project avoided for `self-update`.
- **Second download:** the first `make` also fetches the **stanc3 v2.39.0
  binary** from GitHub into `bin/stanc` (9.3 MB). A first build therefore needs
  network twice. For offline use, the Makefile honours `STANC=`, so it can point at a
  CmdStan install's stanc.
- `bridgestan::compile_model` itself is fine to use: it just runs `make` with
  `STAN_THREADS=true`. It writes `<name>_model.so` *next to the `.stan` file*,
  so the Phase 1.2 cache should copy the `.stan` into its cache directory and
  compile there.
- Compile times: about 29 s for the first model (one-time builds included), then
  15–21 s per model.
- `Model::new` refuses a library built without `STAN_THREADS=true`, so that
  flag is enforced, not optional.

### 0.2: how unconstrained names map to parameters (input for Phase 3.1)
`param_unc_names()` gives `name` or `name.k`, **in declaration order, contiguous
per parameter**. `k` counts unconstrained coordinates. It is *not* an index into
the parameter's shape:

| param | declared | unc names | unc len | constrained len |
| --- | --- | --- | --- | --- |
| mu | `real` | `mu` | 1 | 1 |
| sigma | `real<lower=0>` | `sigma` | 1 | 1 |
| beta | `vector[3]` | `beta.1..3` | 3 | 3 |
| theta | `simplex[4]` | `theta.1..3` | 3 | 4 |
| L | `cholesky_factor_corr[3]` | `L.1..3` | 3 | 9 |
| M | `matrix[2,2]` | `M.1.1, M.2.1, M.1.2, M.2.2` (column-major) | 4 | 4 |
| r | `array[2] real<lower=-1,upper=1>` | `r.1, r.2` | 2 | 2 |

**Partition rule:** the block for a parameter is the contiguous run of
unconstrained names whose text before the first `.` equals the parameter name.
`.` cannot occur in a Stan identifier, so the split is unambiguous. The spike
asserts that these runs tile `0..param_unc_num()` exactly and agree in order
with `param_names(false, false)`. Consequences:
- Whole-parameter blocks (D7) are straightforward.
- Slices would be meaningless for constrained types, because `L.2` is not `L[2]`.
  That supports keeping D7.
- A zero-size parameter (`vector[0]`) has no unconstrained coordinates. The
  coverage check (D2) should treat it as trivially covered.
- Tuple-typed parameters were not tested.

`param_constrain(include_tp, include_gq)` with an `Rng` gives TP and GQ.
`param_num(true, true)` = 30 here (24 params + 4 TP + 2 GQ). The same Rng seed
gives the same GQ.

### 0.3: threads
Each thread builds its own `Model` from one shared `StanLibrary`, which is
`Send + Sync` given `STAN_THREADS`. There were no crashes, and outputs are bitwise identical
to a single-thread baseline. 4 × 100k gradients take 0.07 s (simplex) and 0.22 s (mapping).

### 0.4 / D4: the CSV header
The required header was found by greedy deletion. Starting from a real CmdStan
2.39 CSV, each comment line was dropped and stayed dropped only if
`as_cmdstan_fit()` + `$summary()` + `$draws()` + `$diagnostic_summary()` still
worked with no new warnings. The result is 7 lines. Indentation and
`(Default)` are optional, since cmdstanr strips both:

```
# model = <name>_model
# method = sample
# num_samples = 1000
# save_warmup = false
# thin = 1
# metric = diag_e
# id = <chain id>
lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,<extras…>,<params…>
```

Results of the probes (`results/csv_header.log`, `results/csv_probe2.log`):

| Probe | Result |
| --- | --- |
| nuts-rs draws written with that header + the 7 NUTS columns | works |
| `# laplace_sampler = …` / `# laplace_kernel… = …` comment lines | works; they show up in `fit$metadata()` |
| extra columns `n_grad__`, `accept_stat_beta__` | works; they become sampler diagnostics and stay out of `$summary()` |
| `# algorithm = laplace` | works |
| **non-NUTS column set** (`lp__, accept_stat__, stepsize__, n_grad__`) | **fails inside `as_cmdstan_fit()`** ("subscript out of bounds") |
| non-NUTS run padded: `treedepth__=0`, `n_leapfrog__=n_grad`, `divergent__=0`, `energy__=NaN` | works; `ebfmi` is `NA`, which is honest |
| repeated `# laplace_kernel = …` lines | parse, but **only the last one survives** in `$metadata()` |
| no comment header at all | fails |

**Recommendation for D4: go with `cmdstanr::as_cmdstan_fit()`.** It needs no
`laplace_fit` class. Phase 1.7 changes as a result:
1. **Always** write the seven NUTS columns. For kernels that are not NUTS, pad
   them as above. The plan currently says "the NUTS columns when NUTS runs",
   which would break every MALA/RWM/HMC run.
2. Write one metadata key per kernel, `# laplace_kernel_<label> = …`, instead of
   repeating `# laplace_kernel =`. Otherwise cmdstanr keeps only the last.
   (Open question Q3.)
3. Keep `metric = diag_e` even for kernels without a metric. cmdstanr reads the
   inverse metric based on it.

**Risk:** this relies on cmdstanr's CSV parser (tested on cmdstanr
0.9.0.9000), which is not a documented contract. Mitigation: the Phase 1.9
testthat test runs `as_cmdstan_fit()` on real `laplace-sample` output, so a
cmdstanr change shows up as a test failure, and the D4 fallback stays available.
`accept_stat__` and `energy__` in the example file are spike filler, because nuts-rs
does not expose them through `Progress` (see D8 notes).

### 0.5 / D8: nuts-rs
Setup: nuts-rs 0.19.1 (`DiagNutsSettings` defaults, 4 chains, 1000 + 1000),
with the BridgeStan model wrapped as a `CpuLogpFunc`. BridgeStan errors are
mapped to recoverable `LogpError`s, so a Stan rejection becomes a divergence.

**Full-model NUTS matches CmdStan.** On std_normal, the largest |z| is 2.05 against exact
moments and 1.87 against a 4×10 000 CmdStan run. On the GP model, the largest |z| is 0.92 against CmdStan
on the same data, and ESS per gradient is slightly *better* than CmdStan (0.141 vs 0.121).

**Can the position be changed from outside between draws? Partially.**
The public API has `Chain::set_position(&[f64])`. Reading the source
(`chain.rs`, `adapt_strategy.rs`, `stepsize/adapt.rs`,
`transform/adapt/diagonal.rs`) shows that it does more than move the chain.
Every call:
- re-runs the **initial step-size search** (it resets to `initial_step` and
  doubles or halves it at the *current* point), and
- re-initialises the **diagonal mass matrix from the current gradient**.

The types needed to work around this (`GlobalStrategy`, `DiagMassMatrix`,
`TransformedHamiltonian`) are not exported. A block sweep needs something narrower: "the
other block moved, so recompute the cached logp and gradient here, and keep the
tuning". I tested three ways of handling that on a 2-block Gaussian (`models/blocks.stan`:
`(a[i], b[i])` correlated 0.9, with sd(b) = 1, 5 and 0.2). Each block has its own
nuts-rs chain, swept a-then-b, with 4 chains × (1000 + 2000) sweeps:

| Variant | Block-a step size in sampling, mean (sd) | Largest \|z\| | Max R-hat | Min ESS | Gradients (sampling) | ESS/grad |
| --- | --- | --- | --- | --- | --- | --- |
| `set_position` every sweep (official API) | 0.87 (**0.65**) | 3.86 | **1.046** | **78** | 202 259 | 0.00039 |
| no refresh (stale cached logp/gradient) | 0.28 (0.48) | 3.43* | **2.78** | **5** | 8 206 946 | ≈0 |
| `refresh_position` (one-method patch) | 0.75 (0.04) | 1.81 | 1.005 | 1203 | 61 654 | **0.0195** |

\*It passes \|z\| < 4 only because its MCSE is huge. The chains have not mixed.

- **The official API is *biased*, not just slow.** All six second-moment z-scores are
  positive (2.2 to 3.9, about 20–40 % too much variance). Re-tuning at the current
  state makes the kernel depend on the state, which breaks invariance. It also
  fails chain health (R-hat 1.046, ESS 78).
- **No refresh** is wrong, as expected: the leapfrog starts from a gradient of the
  wrong density.
- **The patch** passes everything, at 50× the ESS per gradient of the official API.
  It adds one method to the `Chain` trait. The default implementation falls back to
  `set_position`, and the `NutsChain` version is 5 lines:
  `self.state = self.hamiltonian.init_state(math, position)?`, the second half of
  `set_position` without the `strategy.init(...)` call.
  Diff: `spikes/sampler-phase0/results/nuts-rs-refresh_position.patch`.
- **Does adaptation still behave after external changes?** Yes, with the patch.
  Step-size and metric adaptation run through warmup and then freeze, with
  post-warmup step size sd of 0.04 (jitter only). See "Risks" for one caveat on
  *what* it adapts to.

**Recommendation for D8: use nuts-rs + `refresh_position`. Do not port NUTS.**
Send the method upstream to `pymc-devs/nuts-rs` (it is small, general, and
useful to anyone doing Gibbs-within-NUTS). Until it merges, depend on a git fork
pinned to a commit, or keep the vendored copy. Porting NUTS only makes sense if
upstream refuses *and* maintaining the fork becomes a burden. Whether and how
to contact upstream is Metodi's call (Q2).

Other nuts-rs notes for Phase 1:
- `Progress` exposes step size, gradients, steps, divergence and tuning.
  It does **not** expose accept stat or energy. These exist inside the
  `Storable` stats from `expanded_draw()`, or could go into the same patch.
- nuts-rs adaptation is nutpie-style (gradient-informed diagonal metric,
  different windows), not Stan's. `nuts()` therefore matches CmdStan in
  distribution, not draw for draw.
- nuts-rs uses `rand` 0.10. Use its ChaCha (`rand::rngs::ChaCha8Rng`, feature
  `chacha`) instead of mixing in a second `rand_chacha`.

### 0.6: overhead (report only)
Model: `examples/gps-model` (the `gps::rbf_cov` library model), N = 40,
evaluated at CmdStan's own posterior draws (`results/overhead.txt`):

| Measure | µs per gradient |
| --- | --- |
| BridgeStan `log_density_gradient` | 205 |
| BridgeStan `log_density` (propto = true) | 169 |
| CmdStan 2.39 sampling time / leapfrogs, sampler overhead included | 163–171 |
| CmdStan built with `STAN_THREADS=true` | 164–167 |
| nuts-rs full run through BridgeStan, everything included | ≈ 203 |

- BridgeStan is **about 1.24× slower per gradient**. The cause is not
  `STAN_THREADS` (ruled out above) or `-O` level (both `-O3`).
- An untested hypothesis: thread-local access to Stan's autodiff stack is slower from a
  `dlopen`ed `-fPIC` shared library.
- nuts-rs itself adds no measurable overhead.
- `log_density` with `propto = true` costs almost as much as the gradient,
  because BridgeStan needs autodiff to drop constants.
- CmdStan's own "Gradient evaluation took 0.5 ms" is one cold evaluation, so it
  is not comparable.

## Changes to existing code
- none. New: `spikes/sampler-phase0/` (spike crate, R scripts, vendored
  nuts-rs, results) and `docs/sampler/` (this report, open questions, CSV
  example). The compiler's `Cargo.toml` is untouched; the spike has its own
  `[workspace]`.

## Decisions made during the phase
- **Where the work lives.** `laplace_kernels.md` names
  `/home/metodi/Data_Science_Projects/laplace_tools` as the main project. On
  disk that repo holds `laplace-lsp` and `vscode-laplace`, not the compiler, and
  it is not a Cargo workspace. The spike therefore lives in the compiler repo
  (`laplace`), which is where `laplace sample` dispatch will be added in 1.1.
  Where the `laplace-sample` crate goes is open question Q1.
- **Test model for 0.6/0.5:** the GP model from `examples/gps-model/build/model.stan`
  with simulated data (seed 1, N = 40, written to `models/gp.json`). The
  survival model was not run, because 0.6 asks for one of the two.
- **MCSE and z** come from `posterior::mcse_mean` (bulk-ESS based). Second
  moments are checked as E[x²] with `mcse_mean(x²)`.
- **Vendoring vs forking nuts-rs for the experiment:** vendored copy plus
  `[patch.crates-io]`. This made the experiment reproducible with no external fork.

## Open questions for Metodi
Also recorded in `docs/sampler/open-questions.md`.
- **Q1 — Where does the `laplace-sample` crate live?** (a) Make the `laplace`
  repo root a Cargo workspace with members `.` and `laplace-sample`. (b) Make it a
  standalone crate in the same repo with its own `[workspace]`, like the spike. (c) Put it
  in a separate repo. **Recommendation: (a).** One lockfile and one
  `cargo test --workspace`. The `laplace` binary's build and output are unchanged,
  and the release workflow builds `-p laplace` as before.
- **Q2 — How do we carry the nuts-rs patch?** (a) Open an upstream PR and depend on a
  pinned git fork until it merges. (b) Keep a vendored copy permanently. (c) Port NUTS.
  **Recommendation: (a).** Opening the PR is outward-facing, so it is yours to do or approve.
- **Q3 — Provenance key format.** (a) `# laplace_kernel_<label> = …`, one key
  per block. (b) Put all kernels on one `# laplace_kernel = …` line. (c) Keep
  repeated lines and accept that `$metadata()` shows only the last.
  **Recommendation: (a).** It survives cmdstanr and reads well.
- **Q4 — `energy__` for non-Hamiltonian kernels.** (a) `NaN`, so `ebfmi` is
  `NA`. (b) `-lp__`, so `ebfmi` is a meaningless number. **Recommendation: (a).**

## Risks or plan changes for the next phase
- **Plan change, Phase 1.7:** always emit the 7 NUTS columns, with padding for
  non-NUTS kernels. Otherwise `as_cmdstan_fit()` fails.
- **Plan change, Phase 1.4/3.4:** `nuts()` depends on the nuts-rs patch (Q2).
  Phase 3.4 should keep the 0.5 three-way experiment as a regression test,
  because the two wrong variants are exactly the bugs Phase 3.1 warns about.
- **Rule for Phase 3 combinators,** learned from the `set_position` result:
  nothing may re-tune from the current state after warmup. Any per-sweep
  "re-initialise at current point" makes the kernel state-dependent and biased.
- **Adaptation inside block sweeps (Phase 3.3).** With the patch, block a's
  adapted step size was 0.75, against 1.13 for NUTS on the fixed conditional
  a | b. During warmup, draw-based variance estimates see the *marginal*
  spread of a (sd 1) rather than the *conditional* spread (sd 0.44), because b
  moves between draws. This does not affect correctness, since tuning freezes after warmup,
  but it costs efficiency. Stan-style windowed metric estimation in `adapt()` will
  have the same issue. Worth a design note in 3.3, not a blocker.
- **Phase 1.3:** use one `propto` setting everywhere in a chain. `lp` from a
  gradient call and from a density call must be on the same constant, or
  MH ratios break silently. `propto = false` density-only calls are not
  cheaper enough to justify mixing.
- **Phase 1.2:** the first build needs network twice (BridgeStan tarball, then
  stanc). Plan an offline path through `LAPLACE_BRIDGESTAN` + `STANC=`.
- **cmdstanr coupling (D4)** is covered by the 1.9 testthat test.
- **Speed:** expect BridgeStan runs to be about 1.2× CmdStan per gradient. ESS-per-gradient
  comparisons are unaffected. Wall-clock comparisons should say so.

## Review outcome (after STOP 0)

Decisions on the open questions are recorded in `open-questions.md`.
- Q1 answered (a): workspace. Q2 deferred to Phase 3: Phase 1 uses upstream nuts-rs unpatched, full model only. Q3 answered (a). Q4 answered (a), extended to NaN in every NUTS column that does not apply.
- Q4 verified (`R/csv_probe3.R`). NaN in `energy__`, `treedepth__`, `n_leapfrog__` and `divergent__` breaks none of `$summary()`, `$draws()`, `$diagnostic_summary()` or `$sampler_diagnostics()`, so **no column falls back to 0**. Caveat: cmdstanr reports `num_max_treedepth = 0` even for an all-NaN `treedepth__`.

Accepted plan changes, carried into Phase 1:
- **1.7** always writes the seven NUTS columns. Non-NUTS kernels write NaN in the ones that do not apply. Provenance uses `# laplace_kernel_<label> = <name> <version> <hash>` per block, plus `# laplace_sampler = <name> (sha256: …)`.
- **1.5**: never re-tune after warmup. Add a test asserting that tuning values (step size, metric) are identical across every post-warmup iteration.
- **1.8** also reports: the Stan Math version bundled in BridgeStan v2.9.0 against CmdStan's, the compiler flags of each, and how the gradient timing was measured.
- Overhead (≈1.24× per gradient) is accepted, with no optimisation work. The block step-size gap is noted, with no action now.
