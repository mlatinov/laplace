# Phase 2 report

Phase 2 of `docs/sampler/laplace_kernels.md`: the kernel language (Level 3,
local files only). A `.lapkernel` file next to the model can be used as a
sampler:

```sh
laplace sample model.laplace --sampler ./kernels/mala.lapkernel --set eps=0.05
```

The work is on branch `sampler/phase-2`, created from `eb86c30` as asked. It
does not contain `ba6f313` (the temporary CI push trigger, which is only on
`sampler/phase-1`), so pushes to this branch would not run the BridgeStan
workflow. Nothing is pushed.

**Short version.**
- **Grammar, checks, interpreter, Level 1.** A pest grammar parses the
  plan's syntax, and the static checker enforces every 2.2 rule with
  `file:line:col` errors. A tree-walking interpreter runs the kernel, and one
  registry holds the built-in context and functions. At Level 1 a
  `.lapkernel` gets the same automatic warmup adaptation as the built-ins.
- **The `.lapkernel` twins of `rwm` and `mala` match the Rust kernels bit for
  bit.** That holds for fixed tuning and for default adaptation, on
  closed-form targets and on real BridgeStan builds. The maximum difference is
  0; the bar was 1e-10.
- **One change to a Rust kernel.** `mala`'s Metropolis ratio is regrouped, at
  the rounding level only; its RNG call order is untouched. Without it the
  adapted twins drifted apart (details below; Q12 asks you to confirm).
- **The interpreter is cheap.** It costs about 5–6 µs per MALA step on the GP
  model, 3–4% of one BridgeStan gradient.

## Done
- Before 2.1: the 1.5 MALA acceptance check uses the mean over 5 seeds —
  `453d8bd`. The mean is 0.689 at that commit and 0.679 after 2.5's rounding
  change. Per-seed values run 0.63–0.75, so a single seed could sit near the
  ±0.15 edge.
- Step 2.1: grammar and parser — `658d03f`
- Step 2.2: static checks, plus the built-in registry (the checker needs its
  type rules) — `260ed6c`
- Step 2.3: interpreter; per-built-in tests, including RNG consumption; post-v1
  registry candidates recorded (Q11) — `2b9cf08`
- Step 2.4: `InterpretedKernel` at Level 1, plus the `rwm` / `mala` `.lapkernel`
  files — `1c0771e`
- Step 2.5: equivalence tests (4 fast + 4 BridgeStan), the `mala` ratio
  regrouping, and the overhead measurement — `40a3c86`

I checked out every commit alone and built and tested it:

| Commit | Tests passed | Ignored |
| --- | --- | --- |
| `453d8bd` | 853 | 9 |
| `658d03f` | 864 | 9 |
| `260ed6c` | 887 | 9 |
| `2b9cf08` | 906 | 9 |
| `1c0771e` | 912 | 9 |
| `40a3c86` | 916 | 14 |

Every commit is `fmt` and `clippy -D warnings` clean. The 805 compiler tests
pass throughout, and `laplace build` is untouched.

## Checks
| Check | Result | Numbers |
| --- | --- | --- |
| 2.1 parse tests for valid files | PASS | the plan's MALA example verbatim; a file using every construct; precedence and associativity |
| 2.1 error tests point at line and column | PASS | missing `;`, unknown section, unclosed block, keyword as a name, bad operator, missing `kernel`; each asserts `file:line:col` |
| 2.2 a failing-file test per rule | PASS | 23 tests (types, sizes with `dim`, terminal paths ×6, `@adapt` ×4 rule groups, scoping, structure, functions), each asserting message and position |
| 2.3 every built-in, including RNG order | PASS | 19 tests: values against closed forms; RNG draws replayed by hand from the same seed, then the stream position compared |
| 2.4 `--sampler ./kernels/mala.lapkernel` works | PASS | unit tests plus a real CLI run on the gp model: α 0.838, ρ 1.26, σ 0.270 (CmdStan reference 0.837, 1.25, 0.269); cmdstanr reads it |
| 2.4 `--set eps=0.05` | PASS | fixes the step size; bounds-checked (`eps=-1` is rejected); vector tuning values are not settable |
| 2.5 rwm, fixed tuning, ≤ 1e-10 | PASS | max difference **0** (closed-form ×2 and BridgeStan ×2 targets) |
| 2.5 rwm, default adaptation | PASS | **0** |
| 2.5 mala, fixed tuning | PASS | **0** |
| 2.5 mala, default adaptation | PASS | **0** (failed at ~10 before the ratio regrouping; see below) |
| 2.5 overhead (report only) | — | 5.0–5.9 µs per step = 3.3–4.0% of a gp gradient |
| Phase 1 harness after the `mala` change | PASS | all 8 reference checks pass (`docs/sampler/phase-2/statistical.txt`) |

**What the equivalence compares** (`docs/sampler/phase-2/equivalence.txt`):
- every saved iteration of 4 chains × (1000 warmup + 1000 draws), same seed;
- position, `lp`, acceptance probability, gradient count, and the step size
  and metric in force.
- (a) is Rust `--set step_size=0.3 adapt=false` against `.lapkernel`
  `--set eps=0.3 adapt=false`;
- (b) is both sides with Level 1's default adaptation.
- The fast versions run on every `cargo test`. The BridgeStan versions
  (`twins_*`) run with the other `--ignored` tests, including CI's weekly
  run.

### Why `mala` needed one arithmetic change, and what it was
The first run of (b) failed: max difference 9.9. Fixed tuning passed at 6e-15.
I traced where the chains parted. There was no single flipped decision. The
chains drifted, with step sizes differing by 1e-11 at iteration 76 and the gap
growing. The cause:
- Rust computed the ratio as `lp' − lp − A + B`, with A and B the two
  quadratic forms.
- `propose(q', correction)` computes `(lp' − lp) + correction`.
- The two agree to about 1e-16, and so did the `normal_lpdf` difference whose
  constants cancel only to rounding.
- During warmup, dual averaging turns acceptance differences into log-step
  differences with a gain of about √t/γ ≈ 20√t. Early large steps amplify
  position differences too. So 1e-16 passes 1e-10 within about 100 iterations.

The fix:
- **Rust `mala`** now groups the same arithmetic as `(lp' − lp) + (B − A)`,
  the `propose()` form. Same mathematics and same RNG calls; only the rounding
  changes, at the 1e-16 level. The statistical checks all still pass.
- **`mala.lapkernel`** computes the correction with the same element-wise
  quadratic forms: `sum(square(q' − μ_fwd) ./ two_var) − sum(square(q − μ_bwd) ./ two_var)`.

Your rule was not to change the Rust kernels' RNG call order, and I haven't.
This is a change to their *arithmetic grouping*, so I'm flagging it (Q12). The
alternative is to leave Rust alone and accept that the adapted MALA twin
cannot meet 1e-10.

## Changes to existing code
- **`laplace-sample/src/kernels/mala.rs`:** the Metropolis ratio grouping, as
  above. No RNG change.
- **`laplace-sample/src/kernel.rs`:**
  - `Kernel::source()` returns `Cow<'static, str>`, because a file kernel's
    text isn't `'static`;
  - new `Kernel::version()`: the crate version for built-ins, `local` for a
    file.
- **`laplace-sample/src/csv.rs`:** a header kernel line now carries its version
  (`KernelLine`). The `laplace_sampler` hash covers the version too, so
  built-in runs' sampler hashes differ from Phase 1's.
- **`laplace-sample/src/runner.rs`:** a new `Sampler` enum, either a built-in
  `SamplerSpec` or a `.lapkernel` file with its `--set` values, plus its
  Level 1 adaptation.
- **`laplace-sample/src/sample.rs`, `main.rs`:** use `Sampler`. `--sampler`
  accepts a `.lapkernel` path, and a bad file is rejected before any model is
  compiled.
- **Tests:** `laplace-sample/tests/statistical.rs` uses the 5-seed 1.5 check
  and adds the `twins_*` and overhead tests; there is a new
  `laplace-sample/tests/equivalence.rs`.
- **Dependencies:** `pest`/`pest_derive` 2.9 and `nalgebra` 0.35 (Cargo.toml,
  Cargo.lock).
- **The compiler (`laplace`):** untouched.

## Decisions made during the phase
**Grammar and checks (2.1, 2.2)**
- **Doc comments** (`//@laplace`, `//@brief`, `//@param`) are read by a line
  scan of the file header. The grammar treats every comment alike, so `//@`
  inside a kernel is just a comment.
- **Tuple literals** `(a, b)` are in the grammar. The plan lists tuples, and
  they are needed to return or build one.
- **`@adapt(step_size, target_accept = 0.574)`:** an optional named argument
  sets the target acceptance. Level 1 uses `--set target_accept`, else the
  tag, else 0.8 (Stan's `adapt_delta`); Q14.
- **`@adapt(metric)`** must be on a `vector[dim]`. It may omit its default
  (all ones). The context's `metric` is that value, or all ones if the kernel
  has none (Q13).
- **Tuning values are read-only** inside the kernel; only adaptation and
  `--set` change them. Their defaults must be chain-free: literals, `dim`,
  earlier tuning values, and pure functions such as `rep_vector`.
- **Terminals.** Exactly one `propose()`/`move()` on every path. Nothing may
  follow one, none may sit inside a loop (the count would depend on data), and
  they are only allowed in `step`.
- **Scope.** No shadowing, of variables, functions or built-ins. A function
  sees only its parameters, other functions and the built-ins: not tuning or
  state.
- **Stan conventions:**
  - `_lpdf` calls with more than one argument must use `|`, and `|` anywhere
    else is an error;
  - `int / int` truncates;
  - `vector * vector` is an error that suggests `.*` or `dot_product`.
- **Sizes.** With `dim` known (Level 1: the model's unconstrained dimension),
  sizes built from literals, `dim` and integer arithmetic are checked
  statically; other sizes are checked at run time. Without `dim` (for Phase
  3's `laplace sampler check`), types are still checked.
- **Error reporting.** The checker reports every error in a file, not only
  the first.

**Interpreter (2.3)**
- **Values** are nalgebra `DVector`/`DMatrix`.
- **Arithmetic order mirrors the Rust kernels.** Elementwise in index order,
  scalars applied per element, and `x ^ 2` (a literal 2) is `x * x`. `&&`/`||`
  short-circuit, which is tested with an RNG call on the skipped side.
- **State variables** without an initializer start at zero.

**`propose()` and `move()` (2.4)**
- **`propose(q', c)`** does: an evaluation through the cache (free if the
  kernel evaluated q′ itself), then log α = (lp(q′) − lp) + c (or −∞ when
  lp(q′) = −∞), then the same `mh_accept` as the Rust kernels, which always
  draws exactly one uniform.
- **`move(q')`** jumps unconditionally: `accept_stat__` is NaN, and moving to
  a zero-density point is an error.

**The built-in list, final** (all in the one registry,
`laplace-sample/src/lapkernel/builtins.rs`):
- context: `q`, `lp`, `grad`, `dim`, `metric`, `log_density(x)`,
  `log_density_grad(x)`, `leapfrog(q, p, eps, steps, metric)`;
- terminals: `propose(q_new, log_correction)`, `move(q_new)`;
- random numbers: `std_normal_rng()`, `std_normal_rng(n)`,
  `normal_rng(mu, sigma)`, `uniform_rng(a, b)`;
- densities: `normal_lpdf(y | mu, sigma)`, `std_normal_lpdf(y)`;
- math: `dot_product`, `sum`, `exp`, `log`, `sqrt`, `square`, and
  **`rep_vector`**. `rep_vector` was added because the twins need it for their
  metric default; the plan said to add built-ins only when a kernel needs one.
- Elementwise arithmetic (`+ − * / .* ./ ^`, unary `−`) is part of the
  interpreter.

**The registry.** Each entry is a name, a doc string, its RNG use, whether it
needs a running chain, a type rule and an implementation. Adding a built-in is
one entry. Hessian, Hessian-vector product, `is_warmup` and `iteration` are
recorded as post-v1 candidates (Q11), and a test asserts they are absent.

**Level 1 (2.4).** `--sampler` values ending in `.lapkernel` are files, and
anything else is a built-in name. The adapter is built exactly as for the
built-ins (Stan's windows and dual averaging), with the initial step taken
from the kernel's own default. The CSV records `laplace_sampler = <path>` and
`laplace_kernel_rest = <kernel name> local sha256:<file hash>`.

**Twins.**
- `kernels/rwm.lapkernel` and `kernels/mala.lapkernel` ship in the crate.
  They are test fixtures and readable references.
- `mala.lapkernel` writes its correction as quadratic forms, not as the plan
  example's `normal_lpdf` difference. The file's comment explains why.

## Open questions for Metodi
Also in `docs/sampler/open-questions.md`.
- **Q12 — The `mala` ratio regrouping.**
  - *Options:* (a) keep it: same mathematics and RNG order, and it is what
    makes adaptive equivalence possible; (b) revert it and accept that the
    adapted MALA twin matches only in distribution.
  - *Recommendation:* **(a).**
- **Q13 — `metric` in the context vs `@adapt(metric)`.** Today `metric`
  equals the kernel's `@adapt(metric)` tuning value if it has one, else all
  ones.
  - *Options:* (a) keep that; (b) drop the context entry and make kernels
    always declare their metric; (c) make `metric` the only metric and allow
    `@adapt(metric)` without a variable.
  - *Recommendation:* **(a).** It is the least surprising for both kernels
    that adapt and kernels that don't.
- **Q14 — Default `target_accept` for a `.lapkernel` without one.** It is
  currently 0.8 (Stan's).
  - *Options:* (a) keep 0.8; (b) require the tag argument whenever there is
    `@adapt(step_size)`.
  - *Recommendation:* **(b).** The right target depends on the kernel (0.234
    RWM, 0.574 MALA, ~0.8 HMC), and a silent 0.8 is wrong for most
    single-proposal kernels.

## Risks or plan changes for the next phase
- **Equivalence is now exact, which makes it brittle.** Any change to the
  Rust kernels' arithmetic grouping, the interpreter's evaluation order, or
  `rand`/`rand_distr` versions breaks bit-for-bit equality. The tests then say
  so immediately; this is intended.
- **Interpreter overhead is per step.** About 5–6 µs per MALA step is
  negligible next to a 150 µs gradient, but on a model whose gradient costs
  ~1 µs the interpreter would dominate. Phase 3 composition multiplies steps
  per iteration. If it matters, the fix is to stop cloning nalgebra vectors on
  every variable read and to cache function lookups. Not needed now.
- **Phase 3.1 `BlockTarget`.** `dim` is already the *block* size in the
  checker and interpreter. `q`, `grad` and `log_density` will need to be the
  block's view, which the registry's chain context is shaped to supply.
- **`laplace sampler check` (3.5)** can reuse `LapKernelFile::check(None)`.
  The checker already runs without a dimension.
- **CI:** this branch lacks the temporary push trigger, so the `twins_*`
  BridgeStan tests run on GitHub only after a merge, or if the trigger is
  cherry-picked.
