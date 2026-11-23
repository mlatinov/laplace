# Phase 1 report

Phase 1 of `docs/sampler/laplace_kernels.md`: runtime and Level 1.
`laplace sample model.laplace --sampler nuts|rwm|mala|hmc` works end to end,
including from R. Branch `sampler/phase-1` (from `sampler/phase-0` at
`493fd0d`) in `laplace`; branch `laplace-sample` in `cmdlaplacer`. Nothing is
pushed.

> **Superseded in part by "Follow-up" at the end.** After review:
> - the Robbins–Monro finish was reverted to Stan's dual averaging;
> - moment checks now count only on healthy chains;
> - the compiler is chosen by an explicit order, and the speed claim below
>   is corrected: which compiler is faster depends on the model.

**Short version.**
- **Every plan check passes.** Two only pass because of changes I made along
  the way, and they need your sign-off (Q7):
  - Stan's dual averaging left MALA's acceptance at 0.69 against a 0.574
    target. A Robbins–Monro finish fixes it.
  - Static HMC with a fixed path length barely moved on the standard normal.
    A jittered path length fixes it.
- **`nuts` and `mala` match long CmdStan runs** on the gp, survival and ts
  library models.
  - On survival, `mala` mixes badly at 1000 draws. Its |z| pass means little
    there.
- **The Phase 0 per-gradient gap is the compiler, not BridgeStan.** CmdStan
  here builds with clang++; BridgeStan was using the system g++. With
  clang++, BridgeStan is *faster* than CmdStan on the GP model.

## Done
- Housekeeping — plan moved to `docs/sampler/laplace_kernels.md` (`8693c2f`); Q5 (num_max_treedepth) recorded as answered (`69fa2b4`)
- Step 1.1 — workspace, `laplace` external-subcommand dispatch, `laplace-sample` crate + CLI — `b842342`
- Step 1.2 — `.laplace`/`.stan` loading, BridgeStan fetch (D9), build cache — `5cd2a2a`
- Step 1.3 — `Target`, `ChainState`, cached `Evaluator`, `Kernel` trait — `c648781`
- Step 1.4 — `rwm`, `mala`, `hmc`, `nuts` kernels, RNG order documented — `183bf01`
- Step 1.5 — dual averaging, windowed diagonal metric, Robbins–Monro finish — `1a1dfe4`
- Step 1.6 — chain runner, ChaCha stream per chain, inits, sampler table — `8f7469c`
- Step 1.7 — CSV writer and the working `sample` command — `7f70c9b`
- Checks for 1.4/1.5 — statistical tests on the reference targets — `710666e`
- Step 1.8 — validation against CmdStan on three library models — `4389cc7`
- Step 1.9 — cmdlaplacer `laplace_sample()` + testthat suite — cmdlaplacer `1ff42c7`

Every commit was checked out alone and built, tested, `cargo fmt --check`ed
and `clippy -D warnings`ed. Tests grow from 808 to 845 (plus 11 ignored
statistical tests). The 805 compiler tests that existed before pass at every
commit. `laplace build` output is unchanged: the only compiler change is the
dispatch in `src/main.rs`.

## Checks
| Check | Result | Numbers |
| --- | --- | --- |
| 1.1 existing compiler tests | PASS | 805/805 at every commit, plus 3 new dispatch tests |
| 1.1 `laplace sample --help` reaches `laplace-sample` | PASS | real binary, and a fake one that echoes its args and exit status (`tests/sample_dispatch.rs`) |
| 1.1 missing binary gives install hint | PASS | exit 1, `cargo install --path laplace-sample …` |
| 1.2 second run hits the cache | PASS | gp model: 20.6 s cold, 0.42 s cached; unit test with a fake compiler |
| 1.2 one-character change rebuilds | PASS | real run (`normal(0, 1)` → `normal(0, 2)`) and unit test; also changes to an `#include`d file and to the toolchain |
| 1.3 cache never returns a stale gradient | PASS | 8 unit tests (1-ulp miss, ±0, aliasing, LRU, target change + `invalidate`) |
| 1.4 moment check, 10-d std normal | PASS ×4 | max \|z\|: rwm 1.89, mala 2.14, hmc 2.30, nuts 2.28 |
| 1.4 moment check, 10-d rotated Gaussian (κ = 100) | PASS ×4 | max \|z\|: rwm 2.46, mala 2.14, hmc 1.37, nuts 1.50 (see chain health below) |
| 1.5 MALA acceptance after warmup within ±0.05 of target | PASS | 0.5750 vs 0.574 (failed at 0.718 with Stan's dual averaging; see Decisions) |
| Review: tuning identical across every post-warmup iteration | PASS | all four samplers, step size and metric (`runner::tests::tuning_is_frozen_after_warmup_for_every_sampler`) |
| 1.6 same seed → identical CSVs; chains differ | PASS | byte-identical over 4 chains × 2 runs (`cmp`), and a unit test for every sampler |
| 1.7 `as_cmdstan_fit()` reads the files, `$summary()` works | PASS | all four samplers; `$draws()`, `$diagnostic_summary()`, `$metadata()`, `$inv_metric()` too (`docs/sampler/phase-1/check_csv.txt`) |
| 1.8 means and SDs vs long CmdStan, \|z\| < 4 | PASS ×6 | table below |
| 1.9 testthat, `fit$summary()` runs | PASS | 24/24, 0 warnings, 0 skipped; `R CMD check`: Status OK |

### 1.4 / 1.5 chain health and cost (`docs/sampler/phase-1/statistical.txt`)
4 chains × (1000 + 1000), seed 20261123. ESS is split, non-rank-normalised
(`posterior::ess_basic`-style). ESS/grad = min ESS / sampling-phase
`n_grad__`.

| Target | Sampler | max \|z\| | max R-hat | min ESS | ESS / grad |
| --- | --- | --- | --- | --- | --- |
| std normal | rwm | 1.89 | 1.046 | 88 | 0.022 |
| std normal | mala | 2.14 | 1.008 | 781 | 0.195 |
| std normal | hmc | 2.30 | 1.010 | 999 | 0.025 |
| std normal | nuts | 2.28 | 1.004 | 1830 | 0.092 |
| rotated κ=100 | rwm | 2.46 | **1.624** | **7** | 0.002 |
| rotated κ=100 | mala | 2.14 | **1.253** | **12** | 0.003 |
| rotated κ=100 | hmc | 1.37 | 1.004 | 1054 | 0.027 |
| rotated κ=100 | nuts | 1.50 | 1.004 | 999 | 0.020 |
| rotated κ=100, 4 × 50 000 draws | rwm | 2.87 | 1.008 | 448 | 0.002 |
| rotated κ=100, 4 × 50 000 draws | mala | 1.37 | 1.004 | 1075 | 0.005 |

RWM and MALA pass the 1000-draw moment check on the rotated target only
because their MCSE is large. A diagonal metric cannot undo a rotation, and
neither kernel has momentum. I kept the plan's check as written and added
long-run tests (`*_corr_gauss_long`) that also require R-hat < 1.01 and
ESS > 400. Both pass, which shows the kernels are correct, just slow there.

### 1.8 match against CmdStan (`laplace-sample/validation/results/*.md`)
Reference: CmdStan 2.39.0, 4 × 25 000 draws. Runs: 4 × (1000 + 1000), seed
20261123. Compared on every parameter, transformed parameter and `lp__`:
z = Δ / √(MCSE₁² + MCSE₂²) for mean and SD.
- **gp:** `examples/gps-model`, `gps::rbf_cov`, N = 40.
- **survival:** Makeham survival from `laplace-testbed`, 500 subjects with
  censoring.
- **ts:** AR(p) from `laplace-testbed`, T = 300, with an RNG in generated
  quantities.

| Model | Sampler | max \|z\| (mean, sd) | max R-hat | min ESS | ESS / grad | CmdStan ESS / grad |
| --- | --- | --- | --- | --- | --- | --- |
| gp | nuts | 2.59 | 1.003 | 1915 | **0.129** | 0.082 |
| gp | mala | 2.44 | 1.004 | 837 | **0.209** | |
| survival | nuts | 1.10 | **1.015** | 437 | 0.0098 | 0.0099 |
| survival | mala | 1.60 | **1.604** | **7** | 0.0017 | |
| ts | nuts | 1.49 | 1.002 | 2048 | **0.137** | 0.095 |
| ts | mala | 1.83 | 1.007 | 545 | 0.136 | |

- **Survival, `nuts`.** R-hat 1.015 is just over Section 4's 1.01 bar at
  4 × 1000 draws. Its ESS per gradient equals CmdStan's (0.0098 vs 0.0099),
  so this is the posterior's correlated `a`/`gamma`/`eta0` geometry, not the
  sampler.
- **Survival, `mala`.** It does not mix at this length. Its |z| < 4 means
  little with an MCSE that wide.
- **Elsewhere, MALA is surprisingly competitive** per gradient. On gp it gets
  0.209 ESS per gradient, against NUTS's 0.129.

### 1.8 timing, versions and flags (asked for in the Phase 0 review)
**Versions.** Both stacks use the same Stan 2.39.0, Stan Math 5.3.0 and
stanc3 v2.39.0. The difference is the compiler:

| | BridgeStan 2.9.0 (laplace-sample) | CmdStan 2.39.0 (cmdstanr) |
| --- | --- | --- |
| Stan / Stan Math | 2.39.0 / 5.3.0 | 2.39.0 / 5.3.0 |
| stanc3 | v2.39.0 (downloaded by BridgeStan's Makefile) | v2.39.0 |
| Compiler | **g++ 16.2.1** (system default; no `make/local`) | **clang++ 22.1.8** (`make/local`: `CXX=clang++`) |
| Optimisation | `-O3` | `-O3` |
| Other flags | `-std=c++17 -pthread -D_REENTRANT -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -DBOOST_DISABLE_ASSERTS -DBRIDGESTAN_EXPORT -DSTAN_OVERRIDE_EIGEN_ASSERT`, `STAN_THREADS=true` | `-std=c++17 -D_REENTRANT -DBOOST_DISABLE_ASSERTS`, no threads |

**How the timing was measured.**
- **CmdStan:** for each chain, the "Elapsed Time (Sampling)" line of the CSV
  divided by that chain's summed `n_leapfrog__` (one gradient per leapfrog).
- **laplace-sample:** for each chain, wall time from the end of warmup to the
  last row written, measured inside `laplace-sample` and printed to stderr
  (never in the CSV), divided by its sampling-phase `n_grad__`.
- **Both:** 4 chains in parallel on a 16-core machine. The time includes the
  sampler's own work, `param_constrain` (TP + GQ) and CSV writing, so it is
  an upper bound on pure gradient cost.

Results:

| Model | CmdStan µs/grad | laplace-sample nuts µs/grad | laplace-sample mala µs/grad |
| --- | --- | --- | --- |
| gp | 164.1 | 207.0 | 207.5 |
| survival | 98.2 | **73.5** | 70.0 |
| ts | 23.8 | 24.1 | 60.0* |

\*ts generates 299 RNG draws per draw in GQ. With one gradient per draw,
MALA's per-gradient time is dominated by `param_constrain` and output, not
the gradient.

**Controlled experiment** (the gp model, same seed, 4 chains run one at a
time, 3 repeats): the same BridgeStan build at **204–208 µs per gradient with
g++** and **147–148 µs with clang++**. So the Phase 0 "1.24× slower" was the
compiler, not BridgeStan or `-fPIC`. With the same compiler, BridgeStan is
not slower than CmdStan here. Q9 asks what to do with this.

## Changes to existing code
- `Cargo.toml`: a `[workspace]` with members `.` and `laplace-sample`, plus
  `exclude = ["spikes"]` (Q1). The root package is still the default member,
  so `cargo build`, `cargo install --path .` and the release workflow's
  `cargo build --release --locked` still build only `laplace`. The release
  workflow's version check (`sed` on the first `version =` line) still reads
  the compiler's version.
- `Cargo.lock`: gains `laplace-sample`'s dependencies. None are linked into
  `laplace`.
- `src/main.rs` (+51/−2): one `#[command(external_subcommand)]` variant.
  `sample`, `sampler` and `new` are found as `laplace-sample` on `PATH` and
  run with the arguments unchanged, passing the exit status through. A
  missing binary gives an install hint. Any other unknown subcommand still
  gets clap's own "unrecognized subcommand" error, exit 2, as before. One
  `after_help` line tells `laplace --help` readers the sampling commands
  exist, since external subcommands don't appear in clap's list. This is the
  smallest change that makes D1 discoverable.
- `tests/sample_dispatch.rs`: new, 3 tests.
- CI (`.github/workflows/ci.yml`): **unchanged**. `cargo fmt --check` already
  covers the whole workspace. Clippy and tests still run on the root package
  only (Q6).
- cmdlaplacer:
  - new `R/laplace_sample.R` and `man/laplace_sample.Rd`;
  - one `export()` in `NAMESPACE`;
  - `DESCRIPTION` gains `testthat` / `withr` in Suggests and
    `Config/testthat/edition: 3`;
  - new `tests/`.

  Nothing else changed. Your untracked `laplace.toml`/`laplace.lock` there
  are untouched.

## Decisions made during the phase
- **MALA step-size adaptation: Stan's procedure plus a Robbins–Monro finish
  (deviation; needs sign-off, Q7).**
  - *The problem:* the 1.5 check failed at 0.718. Stan's dual averaging keeps
    the *average* acceptance over its iterates at the target. Its iterates
    never stop fluctuating, though, and with a single noisy MH probability
    per step, acceptance at the *averaged* step lands well above target.
    This is the same effect that puts Stan's post-warmup `accept_stat__`
    above `adapt_delta`.
  - *What I measured,* over 12 seeds: Stan as-is gave mean acceptance 0.693.
    A larger γ or a longer term buffer didn't fix it.
  - *The fix:* keep Stan's windows and dual averaging up to the last metric
    update. Then, over a 100-iteration term buffer (Stan uses 50), run
    Robbins–Monro on log ε with gains 1.5/(k + 5), and take the Polyak
    average of the second half. This gave mean acceptance 0.566 over 12
    seeds, every chain within 0.063.
  - *Scope:* rwm, mala and hmc only. `nuts` uses nuts-rs's own adaptation.
  - *Code:* `adapt::{StepFinish, RobbinsMonro}`.
- **HMC jitters its path length by default (deviation from "static"; Q7).**
  - *The problem:* with 10 fixed steps at the adapted ε ≈ 1.2, a trajectory
    on the standard normal covers about two full periods and ends near its
    start: R-hat 1.36, ESS 10.
  - *The fix:* draw L uniformly from {1, …, 2·steps − 1} each iteration, so
    the mean is still `steps`. The draw doesn't depend on the state, so the
    kernel stays exact. Results: R-hat 1.010, ESS 999.
  - `--set jitter=false` gives fixed L. The extra RNG draw is documented as
    the first draw of the step.
- **RWM adapts too.** The plan names MALA and HMC for automatic adaptation.
  Unadapted RWM with ε = 1 is close to useless, so RWM adapts toward 0.234.
  `--set adapt=false` turns adaptation off for any of the three.
- **Default targets:** rwm 0.234, mala 0.574, hmc 0.8, nuts 0.8. The last two
  are Stan's `adapt_delta`.
- **`nuts()` details:**
  - step-size jitter off, which is Stan's default and keeps tuning frozen
    after warmup;
  - `maxdepth` 10;
  - `energy__` recomputed as −lp + kinetic. nuts-rs's own value lives in its
    transformed space and is offset by the metric's log-determinant;
  - the adapted metric is reported (nuts-rs's `mass_matrix_inv` holds
    standard deviations, squared to CmdStan's inverse-metric convention).
    Without it, cmdstanr's `$inv_metric()` fails.
- **`Kernel` trait:**
  - The plan's "Stats type" is the `StepInfo` struct, with `Option` fields
    plus `extra`, not an associated type. That keeps `Box<dyn Kernel>`
    possible for Phase 3 composition. `None` fields become `NaN` in the CSV.
  - Kernels are not `Send`: nuts-rs chains aren't, and each kernel lives in
    its own chain thread.
- **RWM evaluates the gradient too** (through the same evaluator). This keeps
  `ChainState` consistent for Phase 3 composition. With `propto = true` the
  density alone already needs autodiff (Phase 0.6: 169 vs 205 µs).
- **Provenance:**
  - Level 1 is one block over the whole model, labelled `rest` (it *is*
    `rest : <kernel>`), so the kernel line is `# laplace_kernel_rest = mala
    0.1.0 sha256:<hash of the kernel's Rust source file>`.
  - `# laplace_sampler = <name> (sha256: …)` hashes the sampler text:
    sampler, settings, and each block's label, kernel and kernel hash.
  - The settings it covers are written out as `# laplace_settings =
    adapt:true step_size:1 …`. That uses `key:value`, not `key=value`,
    because cmdstanr drops a header line whose value contains a second `=`.
    That is the same parser constraint as Q3, found by the R test.
- **Other CSV header lines:**
  - Beyond the 7 required keys: `num_warmup`, `seed`, `laplace_sample_version`.
  - CmdStan's adaptation block after the column names: step size and
    diagonal inverse metric, which cmdstanr reads.
  - No timings anywhere, so the CSV stays byte-identical.
- **RNG streams:**
  - kernel stream: ChaCha8 with seed `seed`, stream = chain id (1-based, as
    CmdStan numbers chains);
  - generated-quantities RNG: a separate seed (SplitMix64 of seed and
    chain), so GQ never shifts kernel draws;
  - transformed-data RNG: the run's seed, the same for every chain (as
    CmdStan does);
  - default seed: random, recorded in the header.
- **Inits:** `--init R` (a radius; 0 means zeros), `--init file.json` for
  every chain, or one `--init` per chain. Random inits are retried up to 100
  times, as Stan does.
- **CLI defaults:**
  - output in `.` as `<stem>-<chain>.csv`, CSV paths on stdout, progress on
    stderr;
  - `.laplace` input runs `laplace build` in the file's directory, so
    `build/<stem>.stan` stays on disk as the artifact (`LAPLACE_BIN`
    overrides the binary);
  - cache root override: `LAPLACE_SAMPLE_CACHE`.
- **Cache key includes the toolchain** (`CXX`, `CXXFLAGS`, BridgeStan's
  `make/local`). The compiler experiment showed it changes the result, and
  without it switching `CXX` would silently reuse the other compiler's build.
- **BridgeStan fetch:** our own `curl` + `tar` with the pinned sha256 (Phase 0
  recommendation), not the crate's download feature. A unit test asserts the
  crate version equals the pinned source version.
- **cmdlaplacer:**
  - calls `laplace-sample` directly; for a `.laplace` model, that in turn
    needs `laplace`;
  - adds `iter_warmup`/`iter_sampling` to the plan's signature, because they
    are CLI flags with no other way to set them;
  - for non-NUTS samplers, passes `check_diagnostics = FALSE` to
    `as_cmdstan_fit()`. Otherwise cmdstanr *warns* "E-BFMI not computed"
    next to our `message()`. The note replaces that NUTS-only check.
- **Reference targets:**
  - `corr_gauss_10` is randomly rotated (correlations up to ±0.82, generated
    by a seeded, committed R script), so a diagonal metric cannot undo it;
  - truth is read from the same JSON the model uses.

## Open questions for Metodi
Also in `docs/sampler/open-questions.md`.
- **Q6 — Should CI test `laplace-sample`?** Its unit tests need no
  BridgeStan and take seconds.
  - *Options:* (a) add `cargo clippy -p laplace-sample --all-targets -- -D
    warnings` and `cargo test -p laplace-sample` to `ci.yml`, or (b) leave CI
    as is.
  - *Recommendation:* **(a).**
- **Q7 — Two deviations: the Robbins–Monro finish and HMC path-length
  jitter.**
  - *Options:* (a) accept both, (b) revert to pure Stan dual averaging and
    relax the 1.5 check to what Stan achieves (≈ 0.69 ± 0.08 for MALA),
    (c) other.
  - *Recommendation:* **(a).** Both are documented in the code and switchable
    (`--set jitter=false`).
- **Q8 — The Q5 message is wrong for `hmc`.** The text you specified says
  divergences don't apply. Static HMC does report real divergences (4 of 4000
  on gp), and only tree depth is NaN.
  - *Options:* (a) per-sampler wording, e.g. for hmc "Tree depth doesn't
    apply to hmc; …"; (b) keep your exact text for every non-NUTS sampler.
  - *Recommendation:* **(a).** It's a one-line change in
    `sample_diagnostics_note()`.
- **Q9 — Which compiler BridgeStan builds with.** BridgeStan uses the system
  default (g++ here); your CmdStan uses clang++, which is 1.4× faster per
  gradient on gp.
  - *Options:* (a) document it, honour `CXX`, which is already in the cache
    key; (b) default to the CmdStan install's `make/local` when cmdstanr's
    CmdStan is found; (c) nothing.
  - *Recommendation:* **(a) now.** (b) is convenient but couples
    laplace-sample to CmdStan.
- **Q10 — How `laplace-sample` is installed.** It is not in the release
  workflow or `self-update`; the install hint is `cargo install --path
  laplace-sample`.
  - *Options:* (a) add it to release assets in Phase 5, or (b) keep it
    source-only.
  - *Recommendation:* **(a), in Phase 5** (out of Phase 1 scope).

## Risks or plan changes for the next phase
- **Phase 2 equivalence depends on the documented RNG orders.**
  - `mh_accept` always draws exactly one uniform.
  - hmc's optional path-length draw comes first.
  - Normal draws use `rand_distr::StandardNormal`, so the interpreter must
    use the same functions from the same `rand`/`rand_distr` versions.
  - The interpreter will also need `metric` (diagonal) and the Robbins–Monro
    finish if `@adapt(step_size)` is to behave like the Rust kernels.
- **Phase 3 adaptation.** `Adapter` already drives any kernel through
  `tuning()` / `set_tuning()`. The marginal-vs-conditional metric caveat from
  Phase 0 still stands.
- **`#include` with BridgeStan's `compile_model`.** It always adds
  `--include-paths=<staged dir>`, and we add the source directory after it.
  This works for `laplace build`'s default single-file output, but a real
  `--split-functions` build has not been compiled through it yet. The cache
  key already covers included files.
- **First build needs network twice** (the BridgeStan tarball, then stanc),
  as noted in Phase 0. `LAPLACE_BRIDGESTAN` plus `STANC=` is the offline
  path; it is untested.
- **cmdstanr coupling (D4).** The cmdlaplacer end-to-end tests and
  `laplace-sample/tests/r/check_csv.R` now pin the behaviour we rely on:
  NaN columns, metadata keys, `$inv_metric()`.
- **Survival-type posteriors.** Plain NUTS at 4 × 1000 sits at R-hat ≈ 1.015
  in both CmdStan-equivalent cost and in our run. Phase 5's must-pass tier
  should use targets where 1000 draws are enough, as the plan's list does.

## Follow-up (Phase 1 review)

The review's six items, done as follow-up commits on `sampler/phase-1`.
**Outcome: Robbins–Monro was reverted (item 2).** Per the review, that means
stopping here: `sampler/phase-2` was *not* created and Phase 2 has not
started. Under the new harness, every must-pass check passes.

### 1. Moment checks only count on healthy chains
- **Diagnostics.**
  - `diagnostics` now has rank-normalised bulk ESS and R-hat, as posterior
    computes them (the larger of the bulk and folded R-hat).
  - The basic ESS now matches `posterior:::.ess` exactly; it still sets the
    MCSE.
  - A unit test pins all three to posterior 1.7.0 within 1e-6, on fixed
    draws R reproduces exactly, for both even- and odd-length chains.
    Matching took three fixes: ρ₀ fixed at 1, monotone pairs by averaging,
    and splitting before rank-normalising, with an odd middle draw dropped
    as posterior's internal `.split_chains` does.
- **Verdict.** `diagnostics::verdict` returns one of three outcomes:
  - **did-not-converge** if any checked quantity has R-hat ≥ 1.01 or bulk
    ESS ≤ 400 (or a non-finite value);
  - otherwise **fail** if any |z| ≥ 4;
  - otherwise **pass**.

  It lives in the library, so Phase 5's `laplace sampler test` judges user
  kernels with the same code.
- **Draw counts** go per kernel × target in
  `laplace-sample/tests/statistical.toml`. A test checks that every pair is
  listed exactly once. Every reference check must *pass*, and a
  did-not-converge result fails the test with a message saying to add draws,
  never to loosen thresholds.

Results (`docs/sampler/phase-1/statistical.txt`, final code). All eight are
must-pass, and all pass:

| Target | Sampler | Draws / chain (thin) | max \|z\| | max R-hat | min bulk ESS | ESS / grad | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| std normal | rwm | 10 000 (10) | 1.52 | 1.003 | 865 | 0.022 | pass |
| std normal | mala | 1 000 | 1.71 | 1.007 | 634 | 0.159 | pass |
| std normal | hmc | 1 000 | 2.22 | 1.006 | 1375 | 0.035 | pass |
| std normal | nuts | 1 000 | 2.28 | 1.003 | 2158 | 0.109 | pass |
| rotated κ=100 | rwm | 100 000 (100), warmup 5000 | 2.14 | 1.005 | 696 | 0.0017 | pass |
| rotated κ=100 | mala | 50 000 (50), warmup 5000 | 1.60 | 1.003 | 997 | 0.0050 | pass |
| rotated κ=100 | hmc | 1 000 | 1.26 | 1.003 | 1298 | 0.033 | pass |
| rotated κ=100 | nuts | 1 000 | 1.50 | 1.004 | 995 | 0.020 | pass |

At the plan's 1000 draws, three pairs would have been did-not-converge:
rwm on std normal (bulk ESS 88), rwm on the rotated target (R-hat 1.56) and
mala on the rotated target (R-hat 1.25). These are the pairs that previously
"passed" on a wide MCSE.

**1.8, relabelled** (`laplace-sample/validation/results/*.md`, rerun on the
final code). `match_cmdstan.R` applies the same verdict, using posterior's
rank-normalised R-hat and bulk ESS on every compared quantity:

| Model | Sampler | Draws / chain | max \|z\| | max R-hat | min bulk ESS | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| gp | nuts | 1000 | 2.59 | 1.003 | 1915 | pass |
| gp | mala | 1000 | 1.60 | 1.006 | 836 | pass |
| survival | nuts | **4000** | 1.14 | 1.003 | 1855 | pass |
| survival | mala | 1000 | 1.00 | **1.772** | **6** | **did-not-converge** |
| ts | nuts | 1000 | 1.49 | 1.002 | 2048 | pass |
| ts | mala | 1000 | 1.92 | 1.006 | 496 | pass |

Survival `nuts` was also did-not-converge at 1000 draws (R-hat 1.015), so it
gets 4000 in the script's draw table. Survival `mala` stays at 1000 and is
reported as did-not-converge, as the review asked.

### 2. Q7: HMC jitter kept; Robbins–Monro reverted
Mean ESS per gradient over 5 seeds (11, 22, 33, 44, 55), with each pair's draw
counts from `statistical.toml`. ESS per gradient = min bulk ESS over the
checked quantities ÷ sampling gradients. All 40 runs passed the harness.

| Target | Sampler | Robbins–Monro (sd) | Dual averaging (sd) | Accept RM / DA | RM ≥ DA |
| --- | --- | --- | --- | --- | --- |
| std normal | mala | 0.2131 (0.0152) | 0.1972 (0.0233) | 0.579 / 0.689 | yes |
| std normal | hmc | 0.0274 (0.0022) | **0.0311** (0.0022) | 0.814 / 0.902 | **no** |
| rotated κ=100 | mala | 0.00456 (0.00033) | 0.00417 (0.00012) | 0.577 / 0.672 | yes |
| rotated κ=100 | hmc | 0.0340 (0.0018) | 0.0304 (0.0037) | 0.813 / 0.934 | yes |

RM loses on hmc / standard normal, by about 13% (≈ 2.6 standard errors).
There, hitting the 0.8 target exactly costs ESS: a smaller-than-"optimal"
step pays off on that target. By the review's rule I **reverted to Stan's dual
averaging** for rwm, mala and hmc:
- `RobbinsMonro`, `StepFinish` and the temporary `step_finish` setting are
  removed;
- the term buffer is back to Stan's 50;
- the 1.5 check is widened to ±0.15.

MALA's post-warmup acceptance on the test seed is **0.718**: inside the band,
but only 0.006 below its upper edge (0.724). The 5-seed mean was 0.689. The
check is deterministic (fixed seed), but any change to the RNG stream or the
adaptation could tip it over (see Risks).

### 3. Q9: compiler order
`laplace-sample/src/compiler.rs`, documented in `laplace-sample/README.md`.
First match wins:
1. `LAPLACE_CXX`, then `CXX`. `LAPLACE_CXX` comes first as the
   laplace-specific setting.
2. The `CXX` in the detected CmdStan's `make/local`: `CMDSTAN`, else the
   newest `~/.cmdstan/cmdstan-<version>`.
3. Otherwise make's default.

The choice is passed to `make` as `CXX=…`, printed on every run (`Compiler:
clang++ (from CmdStan's …/cmdstan-2.39.0/make/local)` here), and replaces the
raw `CXX` in the cache key. Unit tests cover the precedence, `make/local`
syntax and CmdStan detection.

**A finding that changes the rationale.** With clang++ now picked up from
CmdStan, the 1.8 rerun showed survival getting *slower*. A controlled re-check
(same BridgeStan build, same seed, chains one at a time, two repeats each):

| Model | g++ µs/grad | clang++ µs/grad | Faster |
| --- | --- | --- | --- |
| gp | 202.5–202.8 | 146.3–149.6 | clang++ 1.37× |
| survival | 66.7–72.7 | 120.1–122.3 | **g++ 1.73×** |
| ts | 21.0–21.1 | 25.8–25.9 | **g++ 1.23×** |

So Phase 1's "clang++ is 1.4× faster" held for gp only. Which compiler is
faster depends on the model. Step 2 is still useful as a *like-for-like* rule:
laplace-sample and CmdStan then build with the same compiler. The docs now say
exactly that, and point to `LAPLACE_CXX` for anyone tuning speed. The order
itself is as the review specified.

One build detail: BridgeStan's prebuilt support objects (`bridgestan.o`, TBB)
keep whichever compiler first built them; only the model is rebuilt with the
chosen `CXX`. Every clang++ result above came from such a mixed build. It
links and runs fine, since both compilers use the same libstdc++.

1.8 cost table (rerun; laplace-sample now uses clang++, like CmdStan):

| Model | CmdStan ESS/grad (µs/grad) | nuts ESS/grad (µs/grad) | mala ESS/grad (µs/grad) |
| --- | --- | --- | --- |
| gp | 0.082 (162.8) | 0.129 (148.6) | 0.209 (152.5) |
| survival | 0.0099 (95.5) | 0.0104 (121.6) | n/a, did not converge |
| ts | 0.095 (23.7) | 0.137 (26.8) | 0.124 (67.5) |

### 4. Q8: the note is built from the data
`laplace_sample()` now works from the data:
- It reads the sampler diagnostics after the fit and finds the NUTS columns
  that are NaN in every draw.
- It runs cmdstanr's divergence / tree-depth / E-BFMI checks only for the
  columns with values. It calls `as_cmdstan_fit(check_diagnostics = FALSE)`,
  then `diagnostic_summary()` with just those checks.
- It names the all-NaN columns in one `message()`, with tree depth always
  named when its column is NaN.

| Sampler | Message |
| --- | --- |
| mala, rwm | "Tree depth, leapfrog steps, divergences and E-BFMI don't apply to mala: `treedepth__`, `n_leapfrog__`, `divergent__` and `energy__` are NaN in every draw. See n_grad__ and accept_stat__ instead." |
| hmc | "Tree depth doesn't apply to hmc: `treedepth__` is NaN in every draw. See n_grad__ and accept_stat__ instead." (Divergences and E-BFMI are still checked.) |
| nuts | none |

Tests:
- unit tests on synthetic draws (all four NaN, only tree depth, none, partly
  NaN, and singular/plural wording);
- end-to-end tests for mala (no warnings), hmc and nuts.

Totals: 34/34 tests pass, and `R CMD check` reports Status OK.
`posterior` is added to Suggests, since the tests use it directly.

### 5. Q6: CI
- **`.github/workflows/sampler.yml`:** every push and PR, Ubuntu and macOS,
  `cargo clippy -p laplace-sample --all-targets --locked -D warnings` and
  `cargo test -p laplace-sample --locked`. The statistical tests are
  `#[ignore]` and skipped there.
- **`.github/workflows/sampler-statistical.yml`:** Mondays 04:17 UTC plus
  `workflow_dispatch`. It runs the `--ignored` statistical tests in release
  mode, with `~/.laplace/bridgestan/` cached under the key
  `bridgestan-2.9.0-<os>`. A unit test fails if that key stops matching
  `BRIDGESTAN_VERSION`.
- `ci.yml` is unchanged; its `cargo fmt --check` already covers the
  workspace.
- Neither workflow has run on GitHub yet, because nothing is pushed. Their
  YAML parses, and their commands are the ones I ran locally.

### 6. Q10
Deferred to Phase 5 (release assets).

### Risks after the follow-up
- **The 1.5 check has a 0.006 margin** on its fixed seed (MALA at 0.718
  against a 0.724 bound). It is deterministic, but a Phase 2 change to RNG
  consumption or adaptation could tip it. If it does, look at the 5-seed
  mean (0.689) before calling it a regression.
- **The compiler is a model-dependent speed lever.** Up to 1.7× either way.
  Comparisons with CmdStan are fair under the default order, but users
  chasing speed need to know about `LAPLACE_CXX`, which is now documented.
- **Phase 2 equivalence:** the RNG orders documented in Phase 1 are
  unchanged. Dual averaging uses no randomness, so the revert does not touch
  them.
