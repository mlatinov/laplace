# Phase 5 report (final report)

Phase 5 of `docs/sampler/laplace_kernels.md`: hardening. This is also the
final report on the custom-sampler work. Branch `sampler/phase-5`, from the
Phase 4 tip; the TEMPORARY CI trigger is `7031b53`. Nothing is pushed,
merged, tagged or released. The merge itself is planned in
`docs/sampler/merge-plan.md`.

**Short version.**
- **Every Phase 5 check passes:**
  - the reference suite runs every built-in;
  - SBC passes NUTS and fails the broken MALA on all ten quantities;
  - the release gate blocks a kernel that does not pass.
- **Your additions are all done:**
  - the editor-support API, with its doc;
  - ESS per second and the NUTS ratios in the suite;
  - the docs, labelled experimental;
  - the release workflow, prepared and not run;
  - the merge plan, rehearsed.
- **Custom samplers are a tool, not a speed-up.** On every well-behaved
  model plain NUTS was the most efficient sampler. The only wins over NUTS
  are on toy geometry it handles poorly, and the docs say exactly that.

**Verification:** every Phase 5 commit was checked out alone, built and
tested, with fmt and clippy (`-D warnings`) clean. Tests grow from 980 to
996, plus 21 slow ones.

## Done (Phase 5)
| Step | Commit |
| --- | --- |
| Q18 `laplace new kernel --package`; Q10, Q17–Q19 recorded | `8e3d085` |
| Editor-support library API + `lsp-integration.md` | `953b630` |
| 5.1 `laplace sampler test`, the reference suite | `d137ca5` |
| 5.2 SBC, `--sbc N`, broken-MALA fixture | `3f74d0c` |
| 5.3 release gate for kernel packages | `6def695` |
| 5.4 docs: READMEs, three wiki drafts | `7aabe3e` |
| Release workflow for laplace-sample, prepared (Q10) | `0274794` |
| SBC in the statistical workflow | `a8c3bde` |
| Merge plan | `1f5c273` |

## Checks
| Check | Result | Numbers |
| --- | --- | --- |
| 5.1 pass/fail table with ESS per gradient | PASS | Every row also shows ESS/s, gradient and density-only evaluations apart, and both ratios to plain NUTS (`phase-5/suite-*.txt`) |
| 5.1 a failed must-pass returns non-zero | PASS | Exit 1 on did-not-converge or FAIL (tested; seen for mala/rwm before their eight-schools counts were raised, and for langevin::barker) |
| 5.1 settings from `tests/*.test.toml` | PASS | Found by convention or `--config`. The built-ins' ship as `laplace-sample/tests/*.test.toml` (embedded). The scaffolded file parses. Unknown keys and targets are errors. |
| 5.2 the built-in NUTS passes SBC | PASS | 200 replicates; every quantity's minimum tail probability ≥ 5.7e-3 against a band of 3e-5 |
| 5.2 MALA with the correction dropped fails | PASS (it fails) | All 10 quantities, minimum tail p 0 to 3e-10. Kept as `tests/fixtures/broken_mala.lapkernel`. |
| 5.2 the correct shipped MALA passes | PASS | So the failure is the missing correction, not SBC |
| 5.3 `laplace release` on a kernel package runs the must-pass tier | PASS | Workflow tests: every kernel is tested; a failure blocks the release (nothing committed or tagged); `--skip-sampler-tests` skips and warns; a missing binary explains itself; a function library runs nothing. Real run: a git copy of laplace-langevin is refused because `barker` does not converge (`phase-5/release-gate.txt`). |
| Editor API without BridgeStan | PASS | Tested, including the doc example. `bridgestan` loads its library only at run time, which these functions never do. |
| Docs | done | README sections and three wiki drafts, labelled experimental |

### The suite on every built-in (must-pass tier), with the ratios to NUTS
| Kernel | std_normal_10 | corr_gauss_10 | eight_schools_nc | ESS/s vs NUTS (std, corr, 8s) |
| --- | --- | --- | --- | --- |
| nuts | pass | pass | pass | 1 |
| mala | pass | pass | pass (after 5000 draws) | 1.37, 0.24, 1.05 |
| hmc | pass | pass | pass | 0.52, 1.80, 0.60 |
| rwm | pass | pass | pass (after 200000 draws) | 0.26, 0.09, 0.08 |
| mala.lapkernel / rwm.lapkernel | same rows as the Rust kernels, bit for bit | | | about 1/6 of the Rust kernels' ESS/s |
| hybrid_eight_schools.lapsampler | not applicable | not applicable | pass | 0.29 |

Report-only tier:
- **NUTS** does not converge on the funnel even at 10000 draws. E[v] is
  biased: the known centred-funnel failure.
- **rwm on the mixture** passes at 9.9× NUTS's ESS/s.
- **hmc on the banana** passes at 13.7×.

These are the only cases where something beat NUTS, and they are toy
geometry; the docs present them that way.

## Additions from the go message
- **Editor-support API.**
  - `laplace_sample::editor::{check_lapkernel, check_lapsampler}` take
    source text and return `Diagnostic { file, line, column, severity,
    message }`, sorted.
  - Every diagnostic is an `Error` today; `Warning` exists so adding
    warnings later is not a breaking change.
  - Block coverage is the one check left out, because it needs the model.
  - Documented in `docs/sampler/lsp-integration.md`; no LSP features built.
- **5.1 cost reporting** (Q15): ESS per gradient, ESS per second, the
  ratio of each to plain NUTS, and `n_logp__` counted apart.
- **5.4 docs.**
  - Hybrids are presented as a tool for geometry NUTS handles poorly,
    using the Phase 3/4 tables: 0.14–0.54× NUTS's ESS/s on four models.
  - The measured interpreter overhead is stated: 3.5% of a gradient on
    the GP model, about 1/6 of the Rust kernel's ESS/s on 1 µs densities,
    and 47 µs per step for Barker's coordinate loops. So is when to write
    a Rust `Kernel` (Q19).
- **Release assets (Q10).** `.github/workflows/release-sampler.yml` runs
  only by hand and publishes only when asked. It packages the binary with
  LICENSE, its README and the vendored nuts-rs MIT licence. The packaging
  steps were simulated locally; the workflow was not run.
- **Merge prep.** `docs/sampler/merge-plan.md`:
  - only `sampler/phase-5` merges;
  - five TEMPORARY commits, one per pushed or local phase branch; only
    `7031b53` is in `phase-5`, dropped by a rehearsed, conflict-free
    rebase;
  - three new workflows and none changed on `main`;
  - what comes before tagging 0.3.0.

## Changes to existing code in Phase 5
- **`src/release.rs`:** `ReleasePlan.kernels`; `validate_package` checks a
  kernel package's shape.
- **`src/main.rs`:** `release --skip-sampler-tests` and the gate, which
  runs `laplace-sample` as a separate binary (D1).
- **`laplace-sample`:**
  - `authoring.rs`: `--package`;
  - `main.rs`: `sampler test`, `--sbc`;
  - `Cargo.toml`: `serde`/`toml` become runtime dependencies;
  - new modules `editor`, `suite`, `sbc`;
  - test data: `tests/*.test.toml`, three new targets, the fixture.
- **CI:** `sampler-statistical.yml` also runs the SBC checks; new
  `release-sampler.yml`.
- **Docs:** `README.md`, `laplace-sample/README.md`, `docs/sampler/wiki/`.
- **Not touched:** `laplace build` and codegen. Build output is unchanged.

## Decisions made during the phase
- **Suite verdicts:** did-not-converge on a must-pass target fails the
  run, and the table prints the `[targets.<name>]` lines to add. A
  `.lapsampler` written for one model is "not applicable" to the other
  targets, and fails only if it runs on none.
- **NUTS's report-only counts** were raised to 10000 draws. Those rows are
  every ratio's baseline, and 1000 draws did not converge on any of them.
- **SBC design:**
  - one chain per replicate, 99 kept draws thinned 1:4;
  - family-wise α = 0.01, Bonferroni over 10 quantities;
  - the band is calibrated by simulating the discrete null with the same N
    and L, so it is exact for discrete ranks.
- **The release gate runs before the steps are shown**, so `--dry-run`
  answers whether the release would pass.
- **The prepared release workflow** runs by hand only and covers no
  Windows build (BridgeStan's build path is untested there).

## Open questions for Metodi
Both are in `open-questions.md`.

- **Q20, laplace-sample's version and distribution:** it reports 0.1.0
  beside compiler 0.3.0, and the installer knows only `laplace`.
  **Recommend:** version it with the compiler; installer support later.
- **Q21, Barker's adaptation:** it settles below its acceptance target.
  **Recommend:** leave it for v1 and document it in the kernel.

## Risks
- **Interpreted kernels lose on cheap models**, by design until v1 (Q19);
  the docs say when to write Rust.
- **ESS per second is noisy** on targets that take milliseconds. The suite
  sums sampling time over chains, which run in parallel threads.
- **The slow suite has run only on this machine** since the push-trigger
  runs. Run `sampler-statistical` once by hand after the merge.
- **The laplace-langevin library** is a local folder with no repository.
  Its `barker` needs a tuned `tests/barker.test.toml` before it could pass
  its own release gate (the gate correctly refuses it now).
