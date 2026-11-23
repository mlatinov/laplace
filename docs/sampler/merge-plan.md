# Merge plan: the custom-sampler work

**Nothing here has been run.** Nothing is merged, tagged or released. This
is the plan for when Metodi says go.

## What merges

**Only `sampler/phase-5` merges.** Each phase branch was created from the
previous phase's tip *without* its temporary trigger, so `sampler/phase-5`
already contains every commit of phases 0–4. It is one linear chain on
`main`'s current tip (`ec3f51b`, the 0.2.0 release), and `main` is an
ancestor of it, so the merge can be a fast-forward.

| Branch | Ahead of `main` | Status after the merge |
| --- | --- | --- |
| `sampler/phase-0` | 8 | superseded: delete |
| `sampler/phase-1` (pushed) | 26 | superseded: delete locally and on `origin` |
| `sampler/phase-2` (pushed) | 33 | superseded: delete locally and on `origin` |
| `sampler/phase-3` (pushed) | 43 | superseded: delete locally and on `origin` |
| `sampler/phase-4` | 53 | superseded: delete |
| `sampler/phase-5` | 61 at the time of writing (plus the report commits) | **merge this one** |

The other local branches (`ecosystem-test-fixes`,
`patch-1-visibility-hofs-templates-macros`, `patch-1.1-availability-support`)
have no commits outside `main` and are unaffected.

## The TEMPORARY trigger commits

Each phase branch carries exactly one commit that adds a temporary
`push: branches: ["sampler/**"]` trigger to
`.github/workflows/sampler-statistical.yml`. It existed so the statistical
workflow could be tested on GitHub's runners before the file was on the
default branch.

| Branch | TEMPORARY commit | Dropped by |
| --- | --- | --- |
| `sampler/phase-1` | `ba6f313` | deleting the branch (not in `phase-5`) |
| `sampler/phase-2` | `b51c3ac` | deleting the branch (not in `phase-5`) |
| `sampler/phase-3` | `7f12e81` | deleting the branch (not in `phase-5`) |
| `sampler/phase-4` | `a38323d` | deleting the branch (not in `phase-5`) |
| `sampler/phase-5` | `7031b53` | **the rebase below**: it sits under the Phase 5 commits |

Check that it is the only one left: `git log --oneline --grep=TEMPORARY
main..sampler/phase-5` prints exactly `7031b53`.

## Steps

```sh
git switch -c sampler/merge sampler/phase-5
git rebase --onto 7031b53~1 7031b53          # drop the TEMPORARY trigger
git log --oneline --grep=TEMPORARY main..HEAD   # must print nothing
git diff sampler/phase-5 HEAD --stat         # must be: sampler-statistical.yml, 6 deletions
cargo fmt --check && cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
git switch main && git merge --ff-only sampler/merge
```

**Rehearsed** (in a scratch worktree, on `sampler/phase-5` at the time of
writing): the rebase is conflict-free; no TEMPORARY commit is left; the
only difference from the branch is the six trigger lines; a fast-forward
is possible. Rerun the rehearsal on the final tip before doing it for real.

Then, after pushing `main`, delete the superseded branches (`git branch
-D sampler/phase-{0..5} sampler/merge`; `git push origin --delete
sampler/phase-1 sampler/phase-2 sampler/phase-3`).

## CI changes on `main`

The merge adds three workflow files and changes no existing one (`ci.yml`
and `release.yml` are untouched):

1. **`.github/workflows/sampler.yml`** runs clippy (`-D warnings`) and the
   fast `laplace-sample` tests on every push and PR, on Ubuntu and macOS. It
   needs no BridgeStan.
   - `ci.yml` keeps checking the compiler (`cargo fmt --check` for the
     whole workspace, clippy and tests for the root package).
   - `laplace-sample` depends on the compiler's library, so a compiler
     change that breaks the sampler shows up here.
   - **Decide:** make `sampler / check` a required status check on `main`,
     like `ci`. Recommended.
2. **`.github/workflows/sampler-statistical.yml`** runs the slow suite every
   Monday 04:17 UTC and on demand: the statistical tests and the SBC checks,
   with real BridgeStan builds, BridgeStan cached per runner OS. After the
   rebase it has **no push trigger**. `schedule` and `workflow_dispatch`
   start working as soon as the file is on the default branch.
   - **Check after the merge:** run it once by hand (Actions >
     sampler-statistical > Run workflow). The pushed phase branches ran it
     through the temporary trigger, so the BridgeStan download and cache
     have worked on GitHub's runners before.
3. **`.github/workflows/release-sampler.yml`** (Q10) is prepared, not in use.
   It runs only when started by hand and uploads only with `publish`
   ticked, so merging it publishes nothing.
   - **Decide for 0.3.0:** run it by hand after the compiler release
     (`tag = v0.3.0`, `publish = true`), or fold its `build` job into
     `release.yml` so every `v*` tag ships both binaries. Recommended: by
     hand for 0.3.0, while custom samplers are experimental; fold it in
     once they are not.
   - `install.sh`, `laplace self-update` and the AUR `laplace-bin` package
     know only the `laplace` binary. Shipping `laplace-sample` through them
     is a later decision (open question Q20).

## Before tagging 0.3.0 (not part of the merge)

- **The editor-support update in `laplace_tools`** comes first (Q17). It
  uses `laplace_sample::editor` (`docs/sampler/lsp-integration.md`).
- **`CHANGELOG.md` needs a `## [0.3.0]` section.** `release.yml` refuses a
  tag without one. It should say:
  - custom samplers are experimental;
  - kernel packages need `laplace >= 0.3`;
  - the lock gains a `kind` field for kernel packages only.
- **The `laplace-sample` version** is still 0.1.0 while the compiler is
  0.3.0 (Q20). It appears in `laplace-sample --version` and in built-in
  kernels' CSV provenance.

## The R wrapper (`cmdlaplacer`, separate repo)

Branch `laplace-sample` holds `laplace_sample()` (`1ff42c7`, `c5144f2`,
`c3dbd86`). It shells out to `laplace sample`, so it works with any compiler
that dispatches to `laplace-sample`. Merge it after the laplace merge. Its
docs should carry the same "experimental" label.

## Leave alone

`docs/laplace-for-stan-users.tex` is untracked in the working tree and is
not part of this work.
