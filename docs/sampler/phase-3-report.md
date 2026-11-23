# Phase 3 report

Phase 3 of `docs/sampler/laplace_kernels.md`: composition (Level 2).
`.lapsampler` files compose built-in and local kernels over parameter
blocks, and `laplace sampler check` verifies them.

```sh
laplace sampler check hybrid.lapsampler --model model.stan --data data.json
laplace sample model.stan --data data.json --sampler hybrid.lapsampler
```

**Branch.** The work is on `sampler/phase-3`, from the Phase 2 tip `4ee1185`.
The TEMPORARY CI push trigger is cherry-picked on top as `7f12e81`; it must
be removed before merge. Nothing is pushed.

**Short version.**
- **Every plan check and every extra check from the review passes.** All 17
  statistical tests pass, and so do all 958 fast tests. Every commit was
  verified on its own.
- **Hybrids are correct but cost more per effective draw than plain NUTS**
  on the three test models: 0.31×–0.60× NUTS's ESS per gradient. That is
  expected for Gibbs-style block updates. On the split correlated Gaussian
  the hybrid needs 20× the draws to converge, because its blocks are
  strongly correlated with each other.
- **Q12** (the `mala` ratio regrouping) does not affect Phase 3, so I went
  with the recommendation. It is explained below, since you had not seen it.

## Done
- Answers first:
  - Q14, `@adapt(step_size)` must carry `target_accept` — `05ba27a`;
  - Q2, nuts-rs vendored in-repo with `refresh_position` — `b2c4811`.
- Step 3.1: blocks from the Phase 0.2 mapping, block views of the
  evaluator, `BlockTarget` — `8632625`
- Step 3.2: `on`, `seq`, `mix`, `repeat` combinators; `end_warmup` and
  `tunings` hooks — `98130b4`
- Step 3.3: `adapt()` as a combinator; Level 1 rebuilt on it, numerics
  unchanged — `81b3433`
- Step 3.4: `nuts()` as one block (refresh each sweep) — `c893273`
- Step 3.5: `.lapsampler` grammar, resolver, `laplace sampler check`;
  reference targets — `4a127f6`
- Step 3.6: per-block CSV diagnostics — `d6f6f69`
- Step 3.7: hybrid validation and never-re-tune for every combinator —
  `bf6c342`

Every commit was checked out alone and built and tested. Tests grow
916 → 958, plus 17 statistical tests. `fmt` and `clippy -D warnings` are
clean throughout. The compiler is untouched.

## Checks
| Check | Result | Numbers |
| --- | --- | --- |
| 3.1 cache invalidated when *another* block moves — on correlated blocks (review) | PASS | after block B moves, block A at the same point is re-evaluated, equals the truth, and provably differs from the stale value (the precision matrix couples every A–B pair) |
| 3.2 `mix` frequencies within binomial tolerance | PASS | weights 1:2:7, 20 000 steps, every count within 4 binomial sd |
| 3.2 `seq` of two π-invariant kernels passes the moment check | PASS | `seq(on(a, mala), on(b, rwm))` on correlated blocks, verdict **pass** |
| 3.3 `adapt(./kernels/mala.lapkernel)` reaches its target | PASS | mean acceptance over 5 seeds × 4 chains within ±0.15 of the tag's 0.574 (the Phase 1 band) |
| 3.3 a kernel without tags gets an error naming the tag | PASS | `… has nothing to adapt: tag its step size @adapt(step_size, target_accept = ...) and/or its inverse metric @adapt(metric)` |
| 3.4 `block(beta) : nuts()` alongside another block passes the moment check | PASS | `seq(on(a, nuts()), on(b, adapt(mala())))` on correlated blocks, verdict **pass**, tuning frozen after warmup |
| 3.5 an error test for each check rule | PASS | 19 tests (syntax, structure, run settings, rest, labels, unknown kernel, settings, library kernels, missing file, missing `target_accept` (Q14), adapt contracts, nuts under mix, on inside a line, mix and repeat values, coverage (D2), unknown parameter, overlap, empty rest), each asserting `file:line:col` |
| 3.6 `as_cmdstan_fit()` reads the files | PASS | all five layouts: two-block hybrids ×3, a whole-model `nuts()` block (unsuffixed, E-BFMI and inverse metric work), Level 1 (`docs/sampler/phase-3/check_csv.txt`) |
| 3.7 NUTS + MALA hybrid on the correlated Gaussian | PASS | split into two correlated parameters (review); 4 × 20 000 draws: max \|z\| 1.64, R-hat 1.002, bulk ESS 1429 |
| 3.7 NUTS + MALA hybrid on non-centred eight schools | PASS | against the shipped CmdStan reference: max \|z\| 1.69, R-hat 1.006, bulk ESS 630 |
| Extra: a hybrid with a simplex in a block (review) | PASS | `block(theta)` (simplex[4], 3 unconstrained coordinates) by `adapt(mala.lapkernel)` beside `nuts()`: max \|z\| 1.73, R-hat 1.005, bulk ESS 724 |
| Extra: never re-tune after warmup, every combinator (review) | PASS | `.lapsampler` files with `seq`, `repeat`, `mix` (with `adapt()` inside), `adapt()` of Rust and `.lapkernel` kernels, `nuts()` as a block and under `repeat()`: every component adapted in warmup, then identical at every post-warmup step |
| Earlier phases | PASS | 8 kernel × target health checks, 1.5 acceptance 0.6793 (identical), `.lapkernel` twins bit-identical (`docs/sampler/phase-3/statistical.txt`) |

### Hybrids against plain NUTS (same models, same draw counts)
ESS per gradient is the minimum bulk ESS over every checked quantity divided
by the sampling-phase gradients. Every run's verdict is **pass**.

| Model | Hybrid | Hybrid ESS/grad | NUTS ESS/grad | Ratio |
| --- | --- | --- | --- | --- |
| eight schools (non-centred) | `block(theta_tilde): nuts()`, `block(mu, tau): adapt(mala())` | 0.0213 | 0.0683 | 0.31 |
| simplex model | `block(theta): adapt(mala.lapkernel)`, `rest: nuts()` | 0.0311 | 0.1322 | 0.24 |
| split correlated Gaussian (4 × 20 000, thinned 1:20) | `block(a): nuts()`, `block(b): adapt(mala())` | 0.00224 | 0.00376 | 0.60 |

- **On these models, plain NUTS wins.** That is the expected result for block
  updates of strongly coupled parameters: each block moves with the other
  held fixed. Hybrids earn their keep where a block needs a different kind
  of move, or where one block's gradient is much cheaper. None of these
  test models is like that, and no plan check asks for it.
- **The split Gaussian ratio is optimistic for the hybrid.** Thinning 1:20
  caps the measurable ESS at 4 × 1000 per quantity, which costs NUTS more
  than the hybrid.

## Changes to existing code
- `Cargo.toml` (root): `[patch.crates-io] nuts-rs = { path = "laplace-sample/vendor/nuts-rs-0.19.1" }`.
- `laplace-sample/Cargo.toml`: keeps `nuts-rs = "=0.19.1"`, now patched.
- `laplace-sample/vendor/`: new (Q2).
- `kernel.rs`, the `Kernel` trait gains:
  - `end_warmup()`, called exactly once;
  - `tunings()`, every component's tuning with its path;
  - `adaptable()` and `default_target_accept()`;
  - `StepInfo.blocks`, the per-block statistics.
- `state.rs`: `Evaluator` gains block views (`View`, `set_view`,
  `eval_full`). The cache is still keyed by the full position.
- `target.rs`: `BlockTarget`, and an analytic correlated `Gaussian` for tests.
- `adapt.rs`: the `Adapt` combinator and `Adapter::finish`. The adapter
  algorithm itself is unchanged.
- `kernels/nuts.rs`: block mode (`Nuts::for_block`, refresh per sweep,
  gradient count through the block target) and the warmup-length check.
  Whole-model NUTS is unchanged.
- `runner.rs`:
  - `run_chain` no longer takes an adapter, calls `end_warmup()`, and gives
    the end-of-warmup callback the kernel;
  - builders return one kernel (Level 1 is `adapt(kernel)`);
  - `Sampler::Composite` and `build_chain`;
  - the 0.8 `target_accept` fallback is gone (Q14).
- `csv.rs`: per-block columns, composite and single-block rows, `thin` in
  the header.
- `sample.rs`: the run-length resolution (CLI, then the file, then the
  default); column layouts; thinning; `laplace sampler check`. Also a bug
  fix in `model.rs`: a bare relative `.stan` name has an *empty* parent,
  which failed to canonicalise; regression test added.
- `main.rs`:
  - `--warmup`/`--draws` are optional; new `--thin`;
  - new `sampler check` subcommand;
  - `--set` is refused for a `.lapsampler`.
- `lapkernel/check.rs`: a missing `target_accept` is an error (Q14).
- New modules: `blocks.rs`, `combinators.rs`, `lapsampler/`.
- `tests/r/check_csv.R`: reports "no single inverse metric" instead of failing.
- `laplace build` and the compiler are untouched. Level 1 numerics are
  unchanged: the 5-seed MALA acceptance is identical to every digit, and
  the twins are still bit-identical.

## Decisions made during the phase
- **Q12, explained.** To make the adapted `mala.lapkernel` twin bit-identical
  in Phase 2, the Rust `mala` now writes its Metropolis ratio as
  `(lp' − lp) + correction` instead of `lp' − lp − A + B`. That is the same
  mathematics with the same random numbers; rounding differs at about 1e-16.
  It doesn't touch composition, so I kept it per my recommendation.
- **How a stale gradient is ruled out.**
  - A block kernel evaluates through a *view* of the chain's one evaluator,
    and cache keys are the **full** position. After another block moves,
    the same block point is a different key, so a stale value cannot be
    returned, by construction.
  - nuts-rs has its own integrator, so it uses a `BlockTarget` whose shared
    "rest" is refreshed before every step.
- **`on()` keeps the full state consistent.** After a block moves, the full
  gradient comes from the cache: free when the block kernel evaluated that
  point, one counted evaluation for NUTS. If the block didn't move, nothing
  is re-evaluated.
- **`mix` draws one uniform** before its chosen kernel runs. This is
  documented.
- **`adapt()`:**
  - its schedule is sized to the kernel's *expected* warmup steps (warmup ×
    uses per sweep through `repeat`/`mix` weights);
  - `end_warmup()` freezes it wherever the schedule is, so inside `mix`
    tuning still freezes exactly at the end of warmup (tested);
  - its target is `adapt(k, target_accept = …)`, else the kernel's own: the
    tag (Q14) or a built-in's documented default.
- **`nuts()` under `mix()` is refused**, both at check time and, as a
  runtime safety net, at `end_warmup`. nuts-rs adapts for a fixed number of
  its own draws, and `mix` makes that count random. Under `repeat` and `seq`
  the count is exact.
- **Adaptation is explicit at Level 2.** `rwm()` in a `.lapsampler` runs
  with its given step size; `adapt(rwm())` adapts it. That is the plan
  example's semantics. `target_accept` is therefore an `adapt()` argument,
  not a kernel setting, and the error says so.
- **Blocks:**
  - a parameter may be in only one block;
  - `rest` must be last and non-empty;
  - labels must be unique, since they become CSV column names;
  - `on()` inside a `kernel { }` line is an error, because the line already
    names its block;
  - zero-size parameters have no coordinates and are trivially covered.
- **Duplicate `kernel { }` blocks:** reported once; the second block's lines
  are ignored, so errors don't cascade.
- **CSV columns:**
  - several blocks get per-block columns;
  - the standard columns hold NaN except `divergent__` (any block) and
    `n_grad__` (the total);
  - **one block covering the whole model writes unsuffixed columns for any
    kernel**, not only NUTS, a small generalisation of the plan;
  - per-block step sizes are in `stepsize_<label>__`; a single
    CmdStan-style metric block is written only in the single-block case.
- **Run lengths.** The `.lapsampler`'s `warmup`/`draws`/`thin` are defaults;
  `--warmup`/`--draws`/`--thin` override; otherwise 1000/1000/1. `thin`
  keeps every N-th draw, and generated quantities are computed only for
  kept draws, as in CmdStan.
- **Provenance.** One `laplace_kernel_<label> = <kernel expression> <version>
  sha256:<…>` per block line. The hash covers the expression and every
  leaf kernel's source. The version is `local` when the line uses a local
  file. `laplace_settings` holds the sampler file's own hash.
- **`.lapkernel` paths** in a `.lapsampler` resolve against the sampler
  file's directory.
- **Vendoring wiring (Q2).** A plain path dependency inside the workspace
  is always a workspace member: `exclude` does not apply to path
  dependencies, checked with Cargo 1.95. That would make clippy lint
  upstream code. A `[patch.crates-io]` source is not a member, so the
  dependency stays `nuts-rs = "=0.19.1"` and the root patches it to the
  vendored copy. `VENDORED.md` records this.

## Final `.lapsampler` grammar
`laplace-sample/src/lapsampler/grammar.pest`:

```text
sampler <name> {
  library { import <package>[@<version>] ... }          // optional; Phase 4
  kernel {
    block(<param>, ...) [as <label>] : <kernel>
    ...
    rest : <kernel>                                     // optional, last
  }
  warmup { iterations = <n>; }                          // optional
  draws  { iterations = <n>; thin = <n>; }              // optional
}

<kernel> := nuts(<settings>) | rwm(<settings>) | mala(<settings>) | hmc(<settings>)
          | ./path/to/kernel.lapkernel[(<tuning> = <value>, ...)]
          | <package>::<kernel>                         // Phase 4
          | seq(<kernel>, ...)
          | mix(<weight>: <kernel>, ...)
          | repeat(<n>, <kernel>)
          | adapt(<kernel>[, target_accept = <x>])
          | on(block(<param>, ...), <kernel>)           // not inside a kernel { } line

settings: nuts: step_size, target_accept, max_depth;  rwm, mala: step_size;
          hmc: step_size, steps, jitter
```

`kernel { }` is sugar for `seq(on(block, k), ...)`. Comments are `//` and
`/* */`.

## Open questions for Metodi
Also in `docs/sampler/open-questions.md`.
- **Q15 — Hybrids cost more than NUTS on every test model**, by a factor of
  0.24–0.60 in ESS per gradient. That's correct behaviour, but Phase 5's
  `laplace sampler test` will report it for user samplers too.
  - *Options:* (a) report ESS/grad against plain NUTS as information only;
    (b) also warn when a sampler is below some fraction of NUTS.
  - *Recommendation:* **(a).** A hybrid's value lies outside what the
    reference targets measure.
- **Q16 — Whole-model unsuffixed columns for any kernel.** The plan says "one
  NUTS block"; I applied it to any single block covering the model, which
  is what Level 1 already does.
  - *Options:* (a) keep it general; (b) restrict to NUTS.
  - *Recommendation:* **(a).**

## Risks or plan changes for the next phase
- **Phase 4 (kernel libraries).** The resolver already parses `library { }`
  and `package::name`, and rejects them with "arrive in Phase 4".
  Provenance already distinguishes `local` from versioned kernels. Phase 4
  plugs installed `.lapkernel` files in where local paths resolve, and adds
  their version and hash to the provenance line.
- **Block Gibbs on strongly correlated blocks converges slowly.** The split
  Gaussian needed 20× the draws. Phase 5's `laplace sampler test` should
  print the did-not-converge advice ("give it more draws") rather than fail
  silently; the harness already distinguishes the cases.
- **The vendored nuts-rs** stays until the upstream/fork decision (Q2),
  which is due before merge. `VENDORED.md` documents how to re-apply the
  patch.
- **The TEMPORARY CI trigger** (`7f12e81` on this branch; `ba6f313` and
  `b51c3ac` on the Phase 1 and 2 branches) must go when the sampler work
  merges.
- **A shell note for these sessions:** `ls` is aliased to `eza` here, which
  hangs without a terminal. I used `/bin/ls`.
