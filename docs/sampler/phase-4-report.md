# Phase 4 report

Phase 4 of `docs/sampler/laplace_kernels.md`: kernel libraries. Branch
`sampler/phase-4`, from the Phase 3 tip; the TEMPORARY CI trigger is on top
(`a38323d`). Nothing is pushed.

**Not done, because the usage limit was reached:** the per-commit worktree
verification (each Phase 4 commit built and tested on its own). The final
tree passes 979 tests with clippy clean, and every commit was tested when
it was made.

## Done
- Cost accounting, done first as its own commit (`f7d1945`):
  - density-only evaluations are counted apart in a new `n_logp__` column;
  - gradients are computed only when a kernel needs them;
  - ESS per second is in every efficiency report.
- Fix found during 4.6 (`f10aa40`): thinned rows now count the iterations
  dropped before them.
- 4.1, kernel packages (`kind = "kernels"`): `44cce5b`.
- 4.2, library kernels in a `.lapsampler`: `e4b06e5`.
- 4.3, `laplace doc` for kernels: `839f43f`.
- 4.4, `laplace new kernel`: `13a104c`.
- 4.5, `laplace sampler list`: `b49bce3`.
- 4.6, `laplace-langevin`: `68c4fcd`. The library is in
  `~/Data_Science_Projects/laplace-langevin`; it has no git repository and
  no remote.

## Cost accounting
- **Finding:** every evaluation was a gradient evaluation. `rwm` paid for,
  and reported in `n_grad__`, gradients it never used.
- **Now:**
  - `rwm`, `propose()` and `move()` evaluate the density only;
  - a kernel that needs a gradient computes it when needed, and it is
    counted;
  - `n_logp__` follows `n_grad__` in the CSV.
- **Draws are bit-identical** for rwm, mala, hmc, nuts, rwm.lapkernel and a
  hybrid (`phase-4/cost-draws-unchanged.txt`). Only the counts change: rwm
  goes from 1000 gradient evaluations to 1000 density-only ones.
- **A density-only evaluation is not free.** It costs 0.75–0.98 of a
  gradient's time, because BridgeStan with `propto = true` still runs Stan's
  autodiff forward pass.
- **Thinning:** each saved row's `n_grad__`/`n_logp__` now counts every
  iteration since the previous saved row. Before this, a file thinned 1:20
  overstated ESS per gradient 20-fold.

### Phase 3 hybrids against plain NUTS, re-run (`phase-4/statistical-cost.txt`)
| Model | ESS/grad: hybrid / NUTS | ESS/s: hybrid / NUTS |
| --- | --- | --- |
| eight schools | 0.0213 / 0.0683 = 0.31 | 14124 / 48787 = 0.29 |
| split correlated Gaussian | 0.00224 / 0.00376 = 0.60 | 1362 / 2522 = 0.54 |
| simplex model | 0.0311 / 0.1322 = 0.24 | 12931 / 94027 = 0.14 |

These runs take only 25–1500 ms, so the ESS per second figures are noisy.

## Changes to existing package-manager code (complete list)
- `Cargo.toml`: compiler version 0.2.0 → 0.3.0 (see Q17).
- `src/manifest.rs`: `PackageKind` (defaults to functions; never written
  for one), `KERNEL_PACKAGES_SINCE`, and the rule that a kernel package must
  carry a `laplace` requirement excluding older compilers. The rule is
  checked in `read_package_manifest`.
- `src/resolve/lockfile.rs`: a `kind` field on lock entries, written only
  for kernel packages.
- `src/resolve/mod.rs`:
  - the lock records `kind`;
  - `install_into` checks a kernel package's shape;
  - a function library may not depend on a kernel package;
  - `laplace_home`/`default_cache_root` moved here from `main.rs`.
- `src/package.rs`: `kernel_names` and `check_kernel_package` (new
  functions).
- `src/main.rs`:
  - a model that imports a kernel package is an error;
  - `laplace doc` forwards to `laplace-sample` for a kernel package;
  - cache helpers now delegate to `resolve`.
- `README.md`: a "Kernel packages" section.

## Package-manager safety
- **A manifest without `kind`** is a function library, exactly as before.
- **Released libraries:** gaussian_process, survival, ts, lts, splines,
  cluster and transformations, each at its latest tag, install into fresh
  projects. `laplace.lock` and `laplace.toml` are byte-identical before and
  after Phase 4 (`phase-4/released-libraries.txt`, script beside it).
- **Build output:** 14 testbed models build byte-identical against those
  releases.
- **Old compilers:** I ran the real 0.2.0 binary on a project whose lock
  holds a kernel package (`phase-4/old-compiler.txt`).
  - `install`, `add`, `update`, `doc`, and `build` of a model that imports
    the kernel package all exit 1 with "requires laplace >=0.3 ...
    self-update".
  - The lock is never rewritten.
  - A model that doesn't import it builds normally.
  - Documented in the README.
- **The old-compiler message comes from the manifest, not the lock.**
  0.2.0 ignores unknown lock fields. What stops it is that every code path
  touching the package reads its `laplace.toml` first, and a kernel
  package's `laplace = ">=0.3"` fails there.

## 4.6: `laplace-langevin` on a real model (`phase-4/langevin-run.txt`)
- **The library:**
  - `langevin::mala`, the verified MALA;
  - `langevin::barker`, the Barker proposal. It passes the moment checks on
    the standard normal and the correlated Gaussian.
- **The project** combines the survival library (git) and langevin
  (`--path`). Check, doc, list and sample all work end to end.
- **Provenance:** each line names `langevin@0.1.0` and its hash matches
  sha256(expression + newline + installed kernel source).
- **Results** on the survival model, against a CmdStan reference
  (4 × 25000 draws): all three samplers are unbiased.

  | Sampler | Converges? | ESS/s vs NUTS |
  | --- | --- | --- |
  | `block(eta0, beta): adapt(mala)`, `rest: adapt(barker)` | yes, at 100k draws per chain | 0.22× |
  | `block(eta0, beta): nuts()`, `rest: adapt(barker)` | yes, at 100k draws per chain | 0.07× |
  | `rest: adapt(barker)` | no (R-hat 1.017 at 100k draws) | 0.06× |

  The posterior is strongly correlated (|ρ| up to 0.91).
- **No library kernel or hybrid beat NUTS on ESS per second.** On the
  standard normal, `langevin::mala` gets twice NUTS's ESS per gradient but
  0.31× its ESS per second: at about 1 µs per gradient, the interpreter's
  overhead dominates.
- **Barker's adaptation:** its post-warmup acceptance (0.25–0.37) is below
  its 0.4 target. It is correct, but dual averaging overshoots on a kernel
  whose acceptance barely changes with step size.

## Decisions made during the phase
- A `.lapsampler`'s imports resolve in the current directory's
  `laplace.lock`, the same rule every `laplace` command follows.
- `laplace-sample` depends on the `laplace` library crate (lock, manifest,
  installed-package code). The compiler still never links the sampler.
- A kernel package holds only `kernels/*.lapkernel`: no function sources,
  no `exports`, no `[dependencies]`. A version in an `import` line is an
  error, because versions come only from `laplace.toml` and `laplace.lock`.
- Provenance version for a kernel line: laplace-sample's version for
  built-ins only; otherwise `pkg@ver` for each library kernel, joined with
  `+`, with `local` appended when a local file is used.

## Friction found as a library author
1. `laplace new kernel X` outside a package names the *package* X. A package
   whose name differs from its first kernel (langevin containing mala)
   needed the folder and manifest written by hand (Q18).
2. A plain `cargo build` in the workspace doesn't rebuild `laplace-sample`,
   so the dispatch found a stale binary. Use `cargo build --workspace`.
3. Every interpreted per-coordinate loop costs real time: Barker runs at
   about 48 µs per step on a 10-dimensional target.

## Open questions
Q17–Q19 in `open-questions.md`.
