# laplace-sample

> **Experimental in laplace 0.3.0.** Custom samplers are new: the file formats,
> commands and CSV columns may still change. Stan's NUTS through CmdStan stays
> the default way to run a laplace model, and nothing here changes it.

`laplace-sample` is the optional custom-sampler runtime for laplace. It runs
a compiled model through [BridgeStan](https://github.com/roualdes/bridgestan)
and writes one CmdStan-style CSV per chain, which `cmdstanr::as_cmdstan_fit()`
reads (or `cmdlaplacer::laplace_sample()` from R). It is a separate binary:
`laplace sample ...`, `laplace sampler ...` and `laplace new kernel` find it
on `PATH` and forward their arguments.

```sh
cargo install --path laplace-sample --root ~/.local --force
laplace sample model.laplace --data data.json --sampler mala --seed 1
```

## Three levels

| Level | You write | Example | Guide |
| --- | --- | --- | --- |
| 1. Pick a sampler | nothing | `--sampler mala`, `--sampler ./kernels/k.lapkernel` | [Level 1](../docs/sampler/wiki/Level-1-Pick-a-sampler.md) |
| 2. Compose kernels | a `.lapsampler` | `block(beta) : nuts()`, `rest : adapt(mala())` | [Level 2](../docs/sampler/wiki/Level-2-Compose-kernels.md) |
| 3. Write kernels | a `.lapkernel` | one `step { ... }` ending in `propose()` or `move()` | [Level 3](../docs/sampler/wiki/Level-3-Write-kernels.md) |

Kernels can be packaged and shared like function libraries (`kind =
"kernels"`, see the main README's "Kernel packages"), tested on a reference
suite (`laplace sampler test`), and checked by simulation-based calibration
(`laplace sampler test --sbc N`).

## When to use it, and when not

**Use plain NUTS unless you have a reason not to.** Composing kernels is a
tool for geometry NUTS handles poorly: a block that needs a different kind
of move, multimodality, a block whose gradient is much cheaper than the
rest. It is **not** faster in general. On every well-behaved model we
measured, the hybrids lost to plain NUTS, both per gradient and per second:

| Model | Hybrid | ESS/grad vs NUTS | ESS/s vs NUTS |
| --- | --- | --- | --- |
| Eight schools (non-centred) | `block(theta_tilde): nuts()`, `block(mu, tau): adapt(mala())` | 0.31 | 0.29 |
| Correlated Gaussian, split in two blocks | `block(a): nuts()`, `block(b): adapt(mala())` | 0.60 | 0.54 |
| Simplex model | `block(theta): adapt(mala.lapkernel)`, `rest: nuts()` | 0.24 | 0.14 |
| Survival (Weibull-type, real data) | `block(eta0, beta): adapt(langevin::mala)`, `rest: adapt(langevin::barker)` | 0.24 | 0.22 |

The reason is geometry. A block update moves one block with the others held
fixed, so strongly correlated parameters in different blocks mix slowly.
The survival posterior has correlations up to 0.91 between blocks.
(`docs/sampler/phase-3-report.md`, `phase-4-report.md`.)

Where something else did win, it was on geometry NUTS finds hard. These are
the reference suite's report-only toy targets (`docs/sampler/phase-5/`), not
evidence about real models:

| Target | Kernel | Result | ESS/s vs NUTS |
| --- | --- | --- | --- |
| Bimodal Gaussian mixture | `rwm` | passes (mass in each mode) | 9.9 |
| Banana (Rosenbrock-type) | `hmc` | passes | 13.7 |

**Check before you trust a sampler.** Run `laplace sampler test` on it
(below). The CSV header records exactly which kernels produced a run, and
`n_grad__` / `n_logp__` record what it cost.

## Output

A run writes `<model>-<chain>.csv` in the format CmdStan writes:

- **Header comments:** the seed, the run settings, the sampler and its
  hash, and one provenance line per kernel line:

  ```text
  # laplace_kernel_reg = adapt(langevin::mala) langevin@0.1.0 sha256:664d1a...
  ```

  That is the kernel expression, where it came from (a package version
  from `laplace.lock`, `local`, or laplace-sample's version for
  built-ins), and the hash of its source.
- **Standard columns:** `lp__`, `accept_stat__`, `stepsize__`,
  `treedepth__`, `n_leapfrog__`, `divergent__` and `energy__`. A column
  that does not apply to a kernel is `NaN`, never 0.
- **Cost columns:**
  - `n_grad__`: gradient evaluations;
  - `n_logp__`: density-only evaluations (`rwm`, and `propose()`/`move()`
    in a kernel that never reads `grad`).

  With `--thin`, a saved row counts every iteration since the previous
  saved row, so the column sums to the run's cost.
- **Several blocks:** per-block columns `accept_stat_<label>__`,
  `stepsize_<label>__`, `divergent_<label>__`, `energy_<label>__`,
  `treedepth_<label>__` and `n_leapfrog_<label>__`.

A density-only evaluation is cheaper than a gradient but not free. It costs
0.75–0.98 of a gradient's time, because BridgeStan has to run Stan's
autodiff forward pass to drop constant terms.

## Testing a kernel

```sh
laplace sampler test mala                       # a built-in
laplace sampler test kernels/k.lapkernel        # a local kernel
laplace sampler test langevin::mala             # an installed kernel package's
laplace sampler test hybrid.lapsampler          # a composed sampler
laplace sampler test kernels/k.lapkernel --sbc 200
```

**The reference suite.** Three targets must pass:

- a standard normal (10-d);
- a correlated Gaussian with condition number 100;
- non-centred eight schools, against a long CmdStan reference.

Three more are reported but don't gate anything: Neal's funnel, a banana
and a bimodal mixture.

- **The rule:** every checked moment within 4 Monte Carlo standard errors,
  and only when the chains are healthy (R-hat < 1.01, bulk ESS > 400).
  Unhealthy chains are "did not converge", which asks for more draws,
  never a looser threshold.
- **The table:** ESS per gradient, ESS per second, gradient and
  density-only evaluations, and the ratio of both efficiencies to plain
  NUTS on the same target. The ratios are information, never a threshold.
- **Draw counts:** each kernel's `tests/<kernel>.test.toml` (`laplace new
  kernel` writes one).
- **Exit code:** non-zero when a must-pass target does not pass.

**Simulation-based calibration.** `--sbc N` draws N datasets from a
built-in hierarchical normal model and checks that the ranks of the true
values among the sampler's draws are uniform, with simultaneous ECDF
bands. NUTS passes. A MALA with its Metropolis-Hastings correction dropped
fails on every quantity (`tests/fixtures/broken_mala.lapkernel`).

**The release gate.** `laplace release` on a kernel package runs every
kernel through the must-pass tier first. `--skip-sampler-tests` opts out,
and says so.

## Speed: interpreted kernels and the Rust `Kernel` trait

A `.lapkernel` runs in a tree-walking interpreter. Measured overhead:

- **GP library model:** 5.1 µs per MALA step, 3.5% of one gradient
  (147.8 µs). The interpreter is invisible.
- **Densities costing about 1 µs** (the suite's Gaussians, eight
  schools): the overhead dominates. The interpreted `mala.lapkernel`
  makes bit-identical draws to the built-in Rust `mala` at about 1/6 of
  its ESS per second.
- **Per-coordinate loops:** the Barker kernel runs at about 47 µs per
  step on a 10-d standard normal.

When to write a kernel in Rust instead, by implementing the `Kernel` trait
in `laplace-sample/src/kernel.rs` as the built-ins in
`src/kernels/` do:

- the model's gradient costs a few µs or less **and** the kernel is
  hot; or
- the step loops over coordinates in the kernel language.

Otherwise the `.lapkernel` is the better choice: readable, recorded in the
CSV header by its hash, and shareable as a package. The interpreter will
not be optimised before v1 (Q19).

## Editor support

`laplace_sample::editor` checks `.lapkernel` and `.lapsampler` text and
returns positioned diagnostics, with no BridgeStan or compiled model; see
`docs/sampler/lsp-integration.md`.

## How a model gets built

- A `.laplace` file is compiled with `laplace build` first (in its own
  directory, so `build/<name>.stan` stays on disk); a `.stan` file is used
  as is.
- The BridgeStan source (pinned 2.9.0) is downloaded once to
  `~/.laplace/bridgestan/` and verified against a pinned SHA-256. Its
  Makefile then downloads the matching `stanc`. Point `LAPLACE_BRIDGESTAN` at
  an existing BridgeStan 2.9.0 tree to skip the download.
- The compiled model library is cached in `~/.laplace/cache/bridgestan/`
  (`LAPLACE_SAMPLE_CACHE` to move it), keyed by the `.stan` text, every file
  it `#include`s, the BridgeStan version and the toolchain below. A rerun of
  an unchanged model does not recompile.

### Which C++ compiler

The first match wins, and the choice is printed on every run
(`Compiler: ...`):

1. `LAPLACE_CXX`, then `CXX`, from the environment.
2. The `CXX` set in the `make/local` of the CmdStan that cmdstanr uses:
   `CMDSTAN` if set, else the newest `~/.cmdstan/cmdstan-<version>`. Only
   `CXX` is taken from that file.
3. Otherwise `make`'s default: BridgeStan's own `make/local` if it sets
   `CXX`, else the system compiler.

Step 2 is there so laplace-sample and CmdStan build models with the same
compiler, so their timings compare like with like. It is not a promise of
speed. The compiler changes gradient time a lot, and which one is faster
depends on the model. Measured with the same BridgeStan build
(`docs/sampler/phase-1-report.md`, "Follow-up"):

| Model | g++ | clang++ |
| --- | --- | --- |
| gp (`gps::rbf_cov`) | 203 µs | 148 µs |
| survival (Makeham) | 70 µs | 121 µs |
| ts (AR) | 21 µs | 26 µs |

For the fastest build of a particular model, try both with `LAPLACE_CXX`.

## Environment variables

| Variable | Effect |
| --- | --- |
| `LAPLACE_CXX` | C++ compiler for model builds (takes precedence over `CXX`) |
| `CXX` | C++ compiler, if `LAPLACE_CXX` is not set |
| `CMDSTAN` | CmdStan installation whose `make/local` `CXX` is used |
| `LAPLACE_BRIDGESTAN` | Use this BridgeStan source tree instead of downloading |
| `LAPLACE_SAMPLE_CACHE` | Model build cache directory |
| `LAPLACE_BIN` | The `laplace` binary used for `.laplace` inputs |
| `STANC` | Passed through to BridgeStan's Makefile: use this `stanc` |
