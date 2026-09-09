# laplace

Source-to-source preprocessor: `.laplace` -> plain `.stan`. Adds a package manager,
namespaced imports (`pkg::func()`), and doc lookup on top of Stan, which has none
of these natively.

## Non-negotiable design constraints
- Do NOT attempt to parse full Stan grammar/semantics. Only the `library { }` block
  is deeply parsed, plus a shallow top-level-block scan (`src/parser/blocks.rs`) used
  to locate/strip/reject blocks. `functions { }`, `data { }`, `parameters { }`,
  `model { }` etc. are treated as opaque text and passed through unmodified except
  for `pkg::func(` call-site rewriting.
- Build output must be deterministic: identical input + lockfile -> byte-identical
  `.stan` output, every time. No timestamps, no non-deterministic map iteration
  order in generated code.
- The compiled `.stan` file is a first-class artifact meant to be read and committed
  to git. Never produce output a human wouldn't want to open and debug directly.
- Namespacing is implemented as name-mangling (`pkg::func` -> `pkg__func`), not real
  Stan namespaces (Stan has none). Every exported function from every transitively
  resolved package must end up with a globally unique mangled name.
- Two-file manifest model: `laplace.toml` (ranges, hand-edited) + `laplace.lock`
  (exact pins + checksums, machine-written). `laplace install` reads ONLY the lock.

## Source dialects
- **`.laplace`** — a project/model file. Full Stan block structure, plus an optional
  `library { }` import block.
- **`.laplacelib`** — a *library* source file (`src/parser/laplacelib.rs`). Relaxed:
  bare function definitions, an optional `functions { }` wrapper (unwrapped at load
  time), and an optional `library { }` block. The model-shaped blocks (`data`,
  `transformed data`, `parameters`, `transformed parameters`, `model`,
  `generated quantities`) are a hard parse error — a library provides functions to a
  model, it is not a model.
- **`.stan`** inside a package — plain Stan, verbatim passthrough, cannot import.
  Import parsing and `pkg::func` resolution are the *same code* for both dialects
  (`library_block::parse_import_statements`, `codegen`); only the entry-point
  validation of which top-level blocks are allowed differs. Do not fork them.

## Locked decisions on transitive dependencies

These are expensive to change once libraries are published. Do not revisit casually.

1. **The lockfile is a DAG, not a flat list.** `laplace.lock` has a top-level
   `root = [...]` (the project's direct dependencies) and a `dependencies = [...]`
   on every `[[package]]`. `laplace install` reconstructs the full graph from that
   alone, re-resolving nothing. Locks written before this format read fine: a
   missing `root` means "every locked package is direct".
2. **Diamonds unify to exactly one version (Cargo-style).** Every requirement on a
   package name — from the project and from every other package — is collected, and
   one version satisfying all of them is chosen (the highest available). No overlap
   means a hard error naming every requirer and its range. Two versions of the same
   package never coexist in a build. Cycles are a hard error with the path printed.
   The algorithm lives in `src/resolve/graph.rs` and is pure: everything about the
   outside world arrives through the `PackageProvider` trait, so it is unit-tested
   against synthetic graphs with no filesystem or git.
3. **Imports are private; there are no re-exports.** A package may call `dep::func()`
   only for packages *it* declares in its own `[dependencies]`. The project may call
   `pkg::func()` only for packages in its own `library { }` block. A transitive
   dependency is compiled into the build but is not in scope for anyone who did not
   ask for it directly.
   - Mangling stays plain `pkg__func` for transitive dependencies too — no hash, no
     version fragment. That is safe precisely *because* of decision 2: one version
     per package name per build makes the package name a unique prefix. If single-
     version unification is ever relaxed, mangling must be revisited at the same time.
   - Non-exported functions keep their source names (they share Stan's one flat
     function namespace). Two packages defining the same private helper is a hard
     error (`CodegenError::PrivateFunctionCollision`), not a silent clobber and not
     an automatic rename — auto-mangling privates would change the output of every
     existing build.
4. **Transitive dependencies are flattened, never chained.** In `--split-functions`
   mode each *directly imported* package gets one `<pkg>.stanfunctions` file
   containing itself plus its private transitive dependencies. laplace never emits
   an `#include` of one `.stanfunctions` file from another. Each package is emitted
   exactly once across the whole build (a diamond is not duplicated): it lands in the
   file of the first direct import that needs it, in dependency order, and the
   `#include` lines are emitted in that same order, so every function is defined
   before it is used.

## Current milestone
See laplace-project-plan.md task list. State which numbered task you're on at the
start of each session.
