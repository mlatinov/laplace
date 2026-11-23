# Laplace custom samplers: implementation plan for Claude Code

This file is the handoff for building custom samplers in Laplace. Work through it **one phase at a time**. Every phase ends at a **STOP** point. At each STOP, write the phase report, then wait for Metodi to review it and say "go" before starting the next phase. Never start the next phase on your own, even when the current one went smoothly.

The design reference is the doc "Laplace Custom Samplers — Design Spec (draft)". This file repeats every part of it that you need.

---

## 1. Context

Laplace is a Rust source-to-source preprocessor. It compiles `.laplace` files into plain, inspectable `.stan` files. Stan stays the backend, and the generated `.stan` is always a real file on disk.

This feature adds an **optional** second way to run a model:

```
.laplace → laplace build → model.stan ─┬─ stanc3 + CmdStan → Stan NUTS        (default, unchanged)
                                       └─ BridgeStan .so → laplace-sample      (new, optional)
                                                             ↑ .lapsampler / .lapkernel files
```

The feature has three levels for users:

1. **Level 1: pick a sampler.** Run `laplace sample model.laplace --sampler mala`. In R, `laplace_sample(mod, data, sampler = "mala")` returns a normal cmdstanr fit.
2. **Level 2: compose kernels.** A `.lapsampler` file says which kernel updates which parameter block, for example `block(beta) : nuts()`.
3. **Level 3: write kernels.** A `.lapkernel` file holds one step function in a Stan-like syntax. It runs in an interpreter in Rust, and the density and gradient come from BridgeStan.

### Locations

- Compiler and sampler: `/home/metodi/Data_Science_Projects/laplace`, on GitHub as `mlatinov/laplace`. All work in this plan goes here, and the repo becomes a Cargo workspace with members `.` and `laplace-sample`.
- Editor support: `/home/metodi/Data_Science_Projects/laplace_tools`, on GitHub as `mlatinov/laplace_tools`. It holds the LSP and the VS Code extension and is not touched by this plan.
- R wrapper: `mlatinov/cmdlaplacer`, a separate repo.
- The CLI install pattern is `cargo install --path . --root ~/.local --force`.

### Non-negotiables

- **Do not change the `laplace build` code path.** Parser, resolve, codegen and validate output for `.laplace` → `.stan` must stay byte-identical. Every existing compiler test must still pass after every phase.
- **No hidden behaviour.** Every kernel a user runs must be readable source, and every run must record in its CSV header which kernels produced it.
- **Stan stays the default.** Nothing in this feature changes what `cmdstanr` / `laplace_model()` do today.

---

## 2. Accepted decisions

All nine decisions in the spec are accepted. Implement them as written.

| ID | Decision |
| --- | --- |
| D1 | `laplace-sample` is a **separate binary** with its own crate. `laplace sample`, `laplace sampler ...` and `laplace new kernel` dispatch to it cargo-style: look up `laplace-sample` on `PATH` and forward the arguments. If the binary is missing, print an install hint. The compiler never links BridgeStan. |
| D2 | A `.lapsampler` that leaves a model parameter uncovered is an **error**. A final `rest : <kernel>` line is the explicit catch-all. |
| D3 | `.lapkernel` is **interpreted** (a tree-walking interpreter). A Rust `Kernel` trait is the fast path for built-ins. |
| D4 | `laplace_sample()` returns `cmdstanr::as_cmdstan_fit()` if the Phase 0 spike finds a workable CSV header. Otherwise it returns a `laplace_fit` S3 class wrapping `posterior::draws_array`. |
| D5 | Kernel packages are **kernels-only** in v1 (`kind = "kernels"`). No mixed function and kernel packages. |
| D6 | `laplace release` on a kernel library **requires** the must-pass test tier, with a `--skip-sampler-tests` flag to opt out. |
| D7 | Blocks hold **whole parameters** in v1. No slices like `beta[1:3]`. |
| D8 | `nuts()` wraps **nuts-rs**, provided the spike shows its chain accepts a position that changes between calls. Otherwise port NUTS. |
| D9 | The BridgeStan source is **auto-downloaded at a pinned version** to `~/.laplace/bridgestan/`, overridable with the env var `LAPLACE_BRIDGESTAN`. |

**Rename since the spec doc:** the kernel's tuning block is `tuning {}`, not `params {}`. The old name sat too close to Stan's `parameters` block.

---

## 3. Ground rules for Claude Code

1. **One phase per session.** Within a phase, do the steps in order. Make one commit per step with a clear message (`sampler: phase 1.4 — built-in MALA kernel`).
2. **Every step has checks.** A step is done only when its checks pass. Statistical checks use **fixed seeds** and the tolerances in Section 4, so tests are deterministic.
3. **Don't guess on open questions.** If a step needs a decision this file doesn't cover, write it to `docs/sampler/open-questions.md` with the options you see and your recommendation. Then carry on with the parts that don't depend on it, or stop.
4. **Look before building.** Before touching existing code (CLI dispatch, manifest, lockfile, `laplace doc`, `laplace release`), read how it works now and say in the report what you changed and why. Prefer the smallest change.
5. **Keep spike code separate.** Phase 0 code goes in `spikes/` and is never imported by the real crate.
6. **No feature creep.** Variational methods, SVGD, SGLD, minibatching, discrete parameters, GPU and parameter slices are all out of scope.
7. **At every STOP,** write `docs/sampler/phase-N-report.md` from the template in Section 6, then stop and wait.

---

## 4. Testing conventions (used in every phase)

- **Reference targets** are small `.stan` models with known answers, under `laplace-sample/tests/targets/`. Phase 1 starts with two targets, and Phase 5 completes the set.
- **Moment check:** z = (estimate − truth) / MCSE. A check passes if |z| < 4 for every checked quantity.
- **Chain health:** R-hat < 1.01 and bulk ESS > 400 on must-pass targets. Use 4 chains, 1000 warmup and 1000 draws unless a test says otherwise.
- **Equivalence check:** when two implementations of the same kernel get the same seed and the same RNG call order, their draws must match to 1e-10. This is the strongest test available, so use it wherever it applies.
- **Speed:** mark long statistical tests `#[ignore]` and run them with `cargo test -- --ignored`. Fast unit tests run by default.
- **Cost:** report ESS per gradient evaluation (`n_grad__`) for every statistical run, so samplers can be compared on cost.

---

## Phase 0: Spike (de-risk BridgeStan, CSV and nuts-rs)

**Goal:** prove the three unknowns before writing real code. Everything goes in `spikes/`.

### 0.1 BridgeStan from Rust
- Add the `bridgestan` crate. Pin the latest BridgeStan release and record the version.
- Use the crate's `compile_model` with `STAN_THREADS=true` on a 10-dimensional standard normal and a simplex model.
- Load the model and call `log_density_gradient` with `propto = true` and `jacobian = true`.
- **Check:** the gradient agrees with central finite differences to 1e-6 relative error.

### 0.2 Parameter mapping
- Call `param_unc_num`, `param_unc_names`, `param_names` and `param_constrain` (with transformed parameters, generated quantities and an RNG).
- **Check:** a K-simplex shows K − 1 unconstrained coordinates. Write down exactly how unconstrained names map to parameters, which is the input for block partitioning in Phase 3.

### 0.3 Threads
- Run 4 threads, each with its own model instance, evaluating gradients in a loop.
- **Check:** no crashes, and identical inputs give identical outputs across threads.

### 0.4 CSV compatibility (decides D4)
- Generate a real CmdStan CSV for the standard normal model as a reference.
- Write a CSV by hand from spike draws. Find the **minimal header** for which `cmdstanr::as_cmdstan_fit()` returns a working object where `$summary()`, `$draws()` and `$diagnostic_summary()` all work.
- Check whether extra comment lines (`# laplace_sampler = ...`) and extra columns (`accept_stat_beta__`, `n_grad__`) break anything.
- **Check:** document the exact header lines required, with a working example file.

### 0.5 nuts-rs (decides D8)
- Wrap a BridgeStan model as nuts-rs's log-density trait (`CpuLogpFunc` or the current equivalent) and run full-model NUTS.
- Find out whether a nuts-rs chain can have its position **changed from outside between draws**, which block sweeps need, and whether step-size and metric adaptation still behave after that.
- **Check:** full-model NUTS means match CmdStan's within the |z| < 4 rule. Give a yes, no or partial answer on external position changes, with the relevant API.

### 0.6 Overhead (report only)
- Measure the time per gradient through BridgeStan against CmdStan's reported gradient time for one of Metodi's library models (GP or survival).

### STOP 0
Write `docs/sampler/phase-0-report.md` covering:
- the pinned BridgeStan version and how its source is fetched (D9);
- the required CSV header and the recommendation for D4;
- the nuts-rs findings and the recommendation for D8;
- the overhead numbers;
- anything that changes the plan.

**Wait for approval.**

---

## Phase 1: Runtime and Level 1

**Goal:** `laplace sample model.laplace --sampler nuts|rwm|mala|hmc` works end to end, including from R.

### 1.1 Crate and CLI dispatch
- Create the `laplace-sample` crate. Make the `laplace` repo root a Cargo workspace with members `.` and `laplace-sample` (Q1).
- Use `clap` for the flags `--data`, `--sampler`, `--chains`, `--parallel-chains`, `--warmup`, `--draws`, `--seed`, `--init`, `--set key=val` and `--output-dir`.
- In the `laplace` CLI, add **only** external-subcommand dispatch (D1).
- **Check:** all existing compiler tests pass. `laplace sample --help` reaches `laplace-sample`, and a missing binary gives the install hint.

### 1.2 Model loading and caching
- Accept a `.laplace` file (run `laplace build` as a subprocess) or a `.stan` file.
- Fetch the BridgeStan source per D9. Compile with `STAN_THREADS=true` and cache under `~/.laplace/cache/`, keyed by a hash of the `.stan` text, the BridgeStan version and the flags.
- **Check:** a second run hits the cache with no recompile, and a one-character change in the `.stan` file forces a rebuild.

### 1.3 Core abstractions
Design these with block restriction in mind, but implement only the full model now:
- `Target`: log density and gradient on the unconstrained scale.
- `ChainState`: `q`, `lp`, `grad`, plus a **gradient cache keyed by position**, so re-evaluating the same point is free.
- `Kernel` trait: a `step` function plus a `Stats` type, and `tuning()` / `set_tuning()` hooks for later adaptation.
- `StepInfo`: `accept_prob: Option<f64>`, `n_grad`, `divergent`, `energy` and extra statistics.
- **Check:** unit tests for the cache, including a test that a stale gradient is never returned.

### 1.4 Built-in Rust kernels
- `rwm`, `mala` and static `hmc` (leapfrog with a diagonal metric), written so that each one's **RNG call order is documented**, because Phase 2 equivalence tests depend on it.
- `nuts`, wrapping nuts-rs over the full model.
- **Check:** each kernel passes the moment check on the two Phase 1 targets (a 10-dimensional standard normal and a 10-dimensional correlated Gaussian with condition number 100).

### 1.5 Step-size and metric adaptation
- Implement Stan's dual averaging toward a target acceptance rate, plus windowed diagonal-metric estimation, as **reusable structs**. In Phase 3 they become the `adapt()` combinator.
- In Level 1, `--sampler mala` and `--sampler hmc` adapt automatically during warmup. `nuts` uses its own adaptation. Adaptation is frozen after warmup.
- **Check:** after warmup, MALA's acceptance on the standard normal is within ±0.05 of its target.

### 1.6 Chain runner
- Use one thread per chain and a ChaCha RNG stream per chain, derived from `seed` and the chain id.
- Inits are Uniform(−2, 2) on the unconstrained scale. User inits come from JSON through `param_unconstrain`.
- After each draw, produce the output row with `param_constrain` (including transformed parameters and generated quantities).
- **Check:** the same seed gives identical CSVs across runs, and different chains give different draws.

### 1.7 CSV writer
- Use the header from the spike, plus `# laplace_sampler = ...` and `# laplace_kernel = ...` provenance lines.
- Columns: `lp__`, `accept_stat__`, `stepsize__`, `n_grad__`, the NUTS columns when NUTS runs, then the model quantities.
- **Check:** `cmdstanr::as_cmdstan_fit()` (or the D4 fallback) reads the files, and `$summary()` works.

### 1.8 Match against CmdStan
- Run `nuts` and `mala` on two or three of Metodi's library test models (GP, survival, a time-series model).
- **Check:** posterior means and standard deviations match a long CmdStan run within the |z| < 4 rule. Report ESS per gradient for each sampler.

### 1.9 cmdlaplacer: `laplace_sample()`
- `laplace_sample(model, data, sampler, chains, seed, sampler_args, output_dir)`. `model` is a `CmdStanModel` or a path. Write the data with `cmdstanr::write_stan_json()`, call the CLI with `system2()`, and wrap the result per D4.
- Change nothing else in cmdlaplacer.
- **Check:** a testthat test on a small model, and `fit$summary()` runs.

### STOP 1
Write the report with the 1.8 match table, ESS per gradient for each sampler, anything surprising in the runtime design, and the open questions. **Wait for approval.**

---

## Phase 2: Kernel language (Level 3, local files only)

**Goal:** a `.lapkernel` file sitting next to the model can be used as a sampler. No packaging yet.

### 2.1 Grammar
Write a `pest` grammar for:

```
//@laplace
//@brief Metropolis-adjusted Langevin algorithm
//@param eps Step size
kernel mala {
  tuning {
    real<lower=0> eps = 0.1 @adapt(step_size);
  }
  state { }          // optional: variables kept across iterations
  init  { }          // optional: sets state
  functions { }      // optional: helpers, Stan syntax
  step {
    vector[dim] mu_fwd = q + 0.5 * eps^2 * grad;
    vector[dim] q_new  = mu_fwd + eps * std_normal_rng(dim);
    (real lp_new, vector[dim] g_new) = log_density_grad(q_new);
    vector[dim] mu_bwd = q_new + 0.5 * eps^2 * g_new;
    propose(q_new, normal_lpdf(q | mu_bwd, eps) - normal_lpdf(q_new | mu_fwd, eps));
  }
}
```

- Types are `int`, `real`, `vector`, `matrix` and tuples, with Stan-style bounds on tuning declarations. Statements are declarations, assignment, tuple destructuring, `if`, `for` and `while`.
- **Check:** parse tests for valid files, plus error tests that check messages point at the right line and column.

### 2.2 Static checks
- Type and size checking, with `dim` bound to the block size.
- Every path through `step {}` ends in **exactly one** `propose()` or `move()`.
- An `@adapt(...)` tag only appears on tuning declarations, and the tag name is one of `step_size` or `metric`.
- **Check:** each rule has a failing-file test with a clear message.

### 2.3 Interpreter
- Tree-walking evaluator. Pick one linear-algebra crate (nalgebra is the suggestion) and use it everywhere.
- Built-in context: `q`, `lp`, `grad`, `dim`, `log_density(x)`, `log_density_grad(x)` (through the Phase 1 cache), `metric` and `leapfrog(q, p, eps, steps, metric)`.
- Built-in functions are a **small, listed subset**: `std_normal_rng`, `normal_rng`, `uniform_rng`, `normal_lpdf`, `std_normal_lpdf`, `dot_product`, `sum`, `exp`, `log`, `sqrt`, `square` and elementwise arithmetic. Document the list in code, and add more only when a kernel needs them.
- RNG draws come from the chain's stream.
- **Check:** unit tests for every built-in function, including the order in which RNG values are consumed.

### 2.4 Plug into Level 1
- `InterpretedKernel` implements the `Kernel` trait, so `laplace sample model.laplace --sampler ./kernels/mala.lapkernel` works.
- `--set eps=0.05` overrides a tuning value.

### 2.5 Equivalence and overhead
- Write `rwm.lapkernel` and `mala.lapkernel`.
- **Check:** with the same seed, the `.lapkernel` versions produce the same draws as the Rust versions to 1e-10. If they don't, the RNG call order differs; fix that before anything else.
- **Report only:** interpreter time per step against BridgeStan gradient time on one library model.

### STOP 2
Write the report with the equivalence results, the overhead numbers, the final built-in function list, and any grammar decisions made along the way. **Wait for approval.**

---

## Phase 3: Composition (Level 2)

**Goal:** `.lapsampler` files compose built-in and local kernels over parameter blocks.

### 3.1 Blocks
- Partition the unconstrained vector by parameter using the Phase 0.2 mapping.
- Add a `BlockTarget` wrapper: the log density as a function of one block with the rest held fixed.
- **Check:** the cache is invalidated when **another** block moves, which is the subtle bug to guard against. Test it directly.

### 3.2 Combinators
- `on(block, k)`, `seq(...)`, `mix(w: k, ...)` and `repeat(n, k)`. The `kernel {}` block is sugar for `seq(on(...), ...)`.
- **Check:** `mix` picks kernels with frequencies within the binomial tolerance of their weights. A `seq` of two kernels that each leave π invariant passes the moment check.

### 3.3 `adapt()`
- Generalize the Phase 1.5 structs into a combinator. A kernel qualifies through `@adapt` tags (interpreted kernels) or the trait's `tuning()` hooks (Rust kernels).
- A kernel without the tags gets an error naming the missing tag.
- **Check:** `adapt(./kernels/mala.lapkernel)` reaches its target acceptance on the standard normal.

### 3.4 `nuts()` as one block
- Implement per the D8 outcome.
- **Check:** `block(beta) : nuts()` alongside another block passes the moment check.

### 3.5 `.lapsampler` parser and resolver
```
sampler hybrid {
  kernel {
    block(beta)              : nuts()
    block(sigma, tau) as sc  : adapt(mala(), target_accept = 0.57)
    rest                     : rwm()
  }
  warmup { iterations = 1000; }
  draws  { iterations = 1000; thin = 1; }
}
```
- Kernel references resolve to built-ins or **local paths** for now. A `library {}` block parses, but namespaced kernels give "available in Phase 4".
- Values in the file are defaults that CLI flags override.
- Add `laplace sampler check file.lapsampler` to verify coverage (D2), that every kernel resolves, and the adapt contracts.
- **Check:** an error test for each check rule.

### 3.6 Block diagnostics
- Add per-block columns `accept_stat_<label>__`, `stepsize_<label>__`, `divergent_<label>__`, `energy_<label>__`, `treedepth_<label>__` and `n_leapfrog_<label>__`. A block's label is its `as` name, or else its parameter names joined by `_`.
- When the whole model is one NUTS block, drop the suffixes so `diagnostic_summary()` works.
- **Check:** `as_cmdstan_fit()` or the fallback still reads the files.

### 3.7 Validation
- Run a NUTS + MALA hybrid on the correlated Gaussian and on non-centred eight schools.
- **Check:** the moment check passes against the reference.

### STOP 3
Write the report with the hybrid results, ESS per gradient against plain NUTS, and the final `.lapsampler` grammar. **Wait for approval.**

---

## Phase 4: Kernel libraries

**Goal:** kernels can be packaged, installed, locked, documented and scaffolded with the existing package manager.

### 4.1 Manifest and install
- Support `kind = "kernels"` in `laplace.toml`, with `.lapkernel` files under `kernels/`.
- Make `laplace add`, `install` and `update` handle it, along with `--path` dependencies. **Read the existing resolver first** and make the smallest change.
- **Check:** all existing package-manager tests pass, a kernel package installs from a path and from git, and its lockfile entry has a checksum.

### 4.2 Namespaces and provenance
- `library {}` in `.lapsampler` brings installed kernel packages into scope, and `langevin::mala` resolves. Versions come only from `laplace.toml` and `laplace.lock`.
- The CSV header records every kernel's name, version and hash.

### 4.3 `laplace doc` for kernels
- Show the doc comments, tuning values with their `@adapt` tags, and whether the step ends in `propose` or `move`.

### 4.4 `laplace new kernel <name>`
- Outside a library, scaffold `laplace.toml`, `kernels/<name>.lapkernel`, `tests/<name>.test.toml` and `README.md`. Inside one, add only the kernel and its test file. `--kind direct` gives a skeleton ending in `move()`.
- **Check:** a freshly scaffolded kernel passes `laplace sampler check` unchanged.

### 4.5 `laplace sampler list`
- List built-in and installed kernels with their tuning values.

### 4.6 First real library
- Build `laplace-langevin` as a `--path` dependency, containing `mala` and one more kernel, such as preconditioned MALA. Use it from a `.lapsampler` in a real model.
- **Check:** the end-to-end run works and the provenance lines are correct.

### STOP 4
Write the report with the changes to existing package-manager code (listed explicitly), the `laplace-langevin` run, and any friction found while acting as a library author. **Wait for approval.**

---

## Phase 5: Hardening

### 5.1 `laplace sampler test <kernel | file.lapsampler>`

| Target | Dim | Checked against | Tier |
| --- | --- | --- | --- |
| Standard normal | 10 | Exact moments | Must pass |
| Correlated Gaussian, condition number 100 | 10 | Exact moments | Must pass |
| Eight schools, non-centred | 10 | Long CmdStan reference shipped with the suite | Must pass |
| Neal's funnel | 10 | Exact variance of the log scale | Report only |
| Banana (Rosenbrock-type) | 2 | Exact moments | Report only |
| Bimodal Gaussian mixture | 2 | Mass in each mode | Report only |

- Read per-kernel settings from `tests/*.test.toml`. Print a pass/fail table that includes ESS per gradient.
- A failed must-pass check returns a non-zero exit code.

### 5.2 SBC
- `--sbc N` runs N replicates on a built-in hierarchical normal model and checks rank uniformity with simultaneous ECDF bands.
- **Check:** the built-in NUTS passes. A deliberately broken kernel (MALA with the correction term dropped) **fails**. Keep that broken kernel as a test fixture.

### 5.3 Release gate (D6)
- `laplace release` on a `kind = "kernels"` package runs the must-pass tier, with `--skip-sampler-tests` to opt out.

### 5.4 Docs
- Write a README section for `laplace-sample` and draft wiki pages for the three levels, using the doc-comment conventions.

### STOP 5
Write the final report. **Wait for approval** before any release.

---

## 5. Out of scope for v1

Variational methods and SVGD, SGLD and minibatching, SMC and tempering (which need Laplace to generate a split prior and likelihood; this is a later feature), discrete parameters, parameter slices in blocks, GPU execution, and mixed function and kernel packages.

---

## 6. Phase report template

```markdown
# Phase N report

## Done
- Step N.x — one line each, with commit hash

## Checks
| Check | Result | Numbers |
| --- | --- | --- |

## Changes to existing code
- File — what changed and why (or "none")

## Decisions made during the phase
- What was decided, the alternatives, why

## Open questions for Metodi
- Question — options — recommendation

## Risks or plan changes for the next phase
```
