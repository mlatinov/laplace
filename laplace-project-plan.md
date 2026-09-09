# laplace — project plan & session prompts

A source-to-source preprocessor for Stan: package manager + namespaces + doc lookup,
compiling `.laplace` files down to plain, inspectable `.stan` files.

---

## 1. Locked design decisions (recap)

- **Not a Stan compiler.** laplace never parses full Stan semantics. It treats
  `functions { ... }` bodies as opaque text except for a shallow scan to find
  function *signatures* (name + preceding doc comment) for renaming/doc-extraction.
- **Source file extension:** `.laplace` (not `.laplace.stan`).
- **Dedicated `library { }` block** — separate from `functions { }`. Only the
  `library` block is parsed by laplace (imports). `functions { }` is byte-for-byte
  passthrough, never touched by the renamer.
- **Build output is a real, visible `.stan` file** — committed to git, not gitignored.
  `laplace build` must be deterministic: same source + same lockfile → byte-identical
  output. This is the trust/debuggability guarantee — no runtime dependency on laplace.
- **Namespacing = renaming, not real namespaces.** `gps::rbf_cov` → `gps__rbf_cov`.
  Collisions are avoided because the package name is baked into the mangled name.
  This holds for transitive dependencies too — no hash or version fragment — because
  resolution guarantees one version per package name per build (see section 6).
- **Packages** are folders: `laplace.toml` (manifest) + `.stan` and/or `.laplacelib`
  file(s) with `// @laplace` doc comments above exported functions
  (Doxygen/roxygen2-style convention, not a new file format — plain comments,
  ignored by stanc). A package manifest may carry its own `[dependencies]`, which is
  what lets a library depend on another library.
- **Two source dialects for libraries.** `.stan` files in a package are plain Stan,
  passed through verbatim, and cannot import. `.laplacelib` files are the library
  dialect: bare function definitions plus an optional `library { }` import block and
  an optional `functions { }` wrapper, with the model-shaped blocks rejected.
- **Output shape is opt-in.** `laplace build` inlines every imported function into one
  self-contained `.stan` file by default; `--split-functions` instead writes one
  `<pkg>.stanfunctions` file per directly-imported package and `#include`s them.
- **Two-file dependency model, renv/Cargo-style:**
  - `laplace.toml` — hand-edited, loose version ranges (`gps = "^1.0"`)
  - `laplace.lock` — machine-generated, exact pins + checksums, committed to git
  - `laplace install` reads only the lock (reproducible restore on a new machine)
  - `laplace add <pkg>` / `laplace update <pkg>` re-resolve and rewrite the lock
- **Docs are structured, not just comments.** `@laplace` blocks get extracted into a
  `docs.json` sidecar per installed package version. `laplace doc gps::rbf_cov` reads
  that sidecar — no re-parsing of Stan at lookup time.
- **Stack:** Rust, hand-written shallow scanners for the parser (comment/string-aware
  brace matching — no PEG grammar was needed, since laplace never parses Stan
  semantics), TOML for manifests, local filesystem cache at
  `~/.laplace/packages/<name>/<version>/`.

---

## 2. Repo layout

```
laplace/
  Cargo.toml
  src/
    main.rs            # CLI entry (clap): build, install, add, update, doc, init
    lib.rs             # library surface -- main.rs is a thin shell over it
    parser/
      mod.rs
      brace_match.rs    # comment/string-aware brace matching (CodeMask)
      blocks.rs         # shallow top-level Stan block scanner (data{}, model{}, ...)
      library_block.rs  # finds `library { }`, extracts import statements
      laplacelib.rs     # the .laplacelib dialect: validate, strip, unwrap
      signatures.rs     # scans a .stan file for function sigs + preceding doc comments
    resolve/
      mod.rs            # add/update/install, registry + git fetching, checksums
      graph.rs          # PURE version resolution: diamonds, cycles, topo order
      git.rs            # git fetch into a scratch dir
      lockfile.rs       # read/write laplace.lock (the full DAG)
    codegen/
      mod.rs            # renaming pass + splicing / .stanfunctions emission
      rename.rs
    docs/
      mod.rs            # doc comment -> docs.json extraction + doc lookup/render
    package.rs          # package dir -> InstalledPackage (sources + imports)
    manifest.rs         # laplace.toml (project) + package laplace.toml parsing
    validate.rs         # optional stanc type-check pass
  tests/
    cli.rs              # end-to-end tests against the real binary
  examples/
    gps-model/
      model.laplace
      laplace.toml
      laplace.lock
  claude-instructions/  # gitignored: session prompts/specs, kept locally only
```

---

## 3. `CLAUDE.md`

`CLAUDE.md` at the repo root carries the non-negotiable design constraints, the
source-dialect rules, and the four locked decisions on transitive dependencies
(section 6 below). Read it first in any new session; keep it in sync whenever a
decision is locked in here.

---

## 4. Task breakdown — one prompt per Claude Code session

Do these roughly in order. Tasks 1 and 2 (parsing) and task 3 (dependency
resolution) don't depend on each other and can be built/tested independently.

### Task 1 — Library block + import parsing

```
Implement src/parser/library_block.rs. Given a .laplace file's text, find the
`library { }` block and extract each `import` statement inside it, supporting
both bare imports (`import gps`) and pinned imports (`import gps@1.0.0`).
Return a Vec<ImportStatement { name: String, version: Option<String> }>.
Also return the byte range of the library block itself, since codegen will need
to delete it from the final output.
Do not touch anything outside the library block. Write unit tests against inline
string fixtures covering: no library block present, empty library block, multiple
imports, pinned + unpinned mixed, and malformed input (missing closing brace).
```

### Task 2 — Function signature + doc comment extraction

```
Implement src/parser/signatures.rs. Given a .stan file's text (this runs both on
installed packages AND on the user's own functions{} block), find every top-level
function declaration and, if immediately preceded by a comment block starting with
`// @laplace`, parse out @brief, @param (repeatable), @return, @example tags.
Return a Vec<FunctionSig { name, params: Vec<(name, type)>, return_type, doc: Option<Doc> }>.
This must NOT require the doc comment to be present — undocumented functions still
need their name+signature extracted for renaming purposes, just with doc: None.
Write tests using the gps::rbf_cov example from laplace-project-plan.md as a fixture.
```

### Task 3 — Manifest + lockfile + dependency resolution

```
Implement src/manifest.rs and src/resolve/. Parse project-level laplace.toml
(dependencies with semver ranges) and package-level laplace.toml (name, version,
exports). Implement lockfile read/write (src/resolve/lockfile.rs) matching the
schema in laplace-project-plan.md section 1, including a checksum field.
Implement `laplace add <pkg>[@version]`: resolve the latest version matching the
range (or add a new range if unspecified), fetch it (assume a local filesystem
"registry" for now -- a directory of package folders -- real git/http fetching is
a later task), write/update both laplace.toml and laplace.lock.
Implement `laplace install`: read laplace.lock only, copy each pinned package into
~/.laplace/packages/<name>/<version>/, verify checksum, do NOT consult laplace.toml
ranges at all. Write tests covering: fresh install from a lock, checksum mismatch
should error, a lock referencing a package not in the local registry should error
with a clear message.
```

### Task 4 — Renaming + codegen (the core "compiler" step)

```
Implement src/codegen/. Given: (a) the user's parsed .laplace file (library block
imports + everything else as opaque spans), (b) resolved installed packages with
their extracted FunctionSig lists from Task 2, produce the final .stan text:

1. For each imported package, rename every exported function `foo` to `pkgname__foo`
   -- both in its own definition AND in any internal call-sites where the package's
   own code calls its own other exported functions.
2. In the user's original source (outside the library block), rewrite every
   `pkgname::foo(` call-site to `pkgname__foo(`.
3. Detect and error clearly on: an import that doesn't exist in the lock, a
   `pkg::func` call where func isn't in that package's exports, two imported
   packages exporting the same mangled name (shouldn't happen given the pkg prefix,
   but assert it as a sanity check).
4. Splice: functions block = [renamed imported functions, in lockfile order] +
   [user's original functions{} content, untouched]. Delete the library{} block
   entirely from output. Everything else (data/parameters/model/etc blocks) passes
   through byte-for-byte except for the pkg::func( rewriting from step 2.
5. Output must be deterministic -- iterate packages/functions in a stable, sorted
   order, not hashmap iteration order.

Test against the gps rbf_cov example end-to-end: input .laplace -> expected .stan,
byte-for-byte, run the test twice to confirm determinism.
```

### Task 5 — CLI wiring (`laplace build`)

```
Implement src/main.rs using clap. Wire up:
  laplace build <file.laplace> [-o build/model.stan]
  laplace install
  laplace add <pkg>[@version]
  laplace update <pkg>
`build` should: parse the .laplace file, resolve deps against laplace.lock (must
already be installed -- if not, print instructions to run `laplace install` first,
don't auto-install), run codegen, write output, and print a one-line summary
(line count, dependency count) matching the format shown in
laplace-project-plan.md section on running it.
Add an --check flag that runs codegen but only diffs against existing build output
without writing, exits non-zero if they differ (useful for CI to catch stale
committed .stan files).
```

### Task 6 — Doc extraction + `laplace doc` lookup

```
Implement src/docs/. At install time (hook into `laplace install` from Task 3),
run the Task 2 signature extractor over each installed package's .stan file(s) and
write a docs.json sidecar into the package's installed directory containing all
FunctionSig + Doc data as structured JSON.
Implement `laplace doc <pkg>::<func>`: load that package's docs.json (need to
resolve which installed version via the project's lockfile), find the function,
pretty-print it to terminal matching the format shown in laplace-project-plan.md
(brief, params, return, example). Handle: package not installed, function not
found in package, function exists but has no @laplace doc comment (print
signature only with a note that no docs are available).
```

### Task 7 (later, optional) — stanc validation pass

```
After `laplace build` writes output, optionally shell out to `stanc` (assume it's
on PATH) to type-check the generated file. On error, print stanc's raw error
output with a prepended note showing which imported package(s) contributed code
near the failing line (best-effort using the splice boundaries from Task 4 --
exact source-mapping back to laplace source is out of scope for v1).
Gate this behind a --validate flag on `laplace build`, off by default so build
doesn't require stanc installed.
```

### Task 8 — `#include`-based `.stanfunctions` output (done)

```
Add `laplace build --split-functions`. Off by default, so a plain build still
produces one self-contained .stan file. When on, each directly-imported package's
renamed functions go into `<pkg>.stanfunctions` next to the compiled .stan file,
and the functions{} block gets one `#include "<pkg>.stanfunctions"` line per
package instead. The user's own functions{} content stays inline either way.
A zero-import project is unaffected. Multi-file packages still concatenate into
one .stanfunctions file. `--check` diffs the .stanfunctions files too.
```

### Task 9 — `.laplacelib`: libraries that import libraries (done)

```
Add the `.laplacelib` source dialect and full transitive dependency support:
a package manifest may declare its own [dependencies]; laplace.lock grows into a
DAG; resolution unifies diamonds and rejects cycles; transitive functions are
flattened into their parent package's output. See section 6 for the four locked
decisions this settled.
```

---

## 6. Locked decisions: transitive dependencies

Settled while implementing tasks 8 and 9. These are the expensive-to-change ones —
they are baked into published packages' mangled symbol names and into the lockfile
format, so revisit them only deliberately. They are mirrored in `CLAUDE.md`.

### 6.1 The lockfile is a DAG, not a flat list

`laplace.lock` gained two fields:

```toml
root = ["regression"]          # the project's *direct* dependencies

[[package]]
name = "regression"
version = "1.0.0"
checksum = "sha256:..."
source = "registry"
dependencies = ["stats"]       # the edges of the graph

[[package]]
name = "stats"
version = "1.4.0"
checksum = "sha256:..."
source = "registry"
dependencies = []
```

`laplace install` reconstructs the entire graph from this alone and re-resolves no
version range whatsoever — that is still the whole point of the lock. Locks written
before this format read fine: a missing `dependencies` means "leaf", and a missing
or empty `root` means "every locked package is a direct dependency", which is
exactly what those locks meant when they were written.

### 6.2 Diamonds unify to one version; cycles are a hard error

Cargo-style. Every requirement on a package name — from the project's `laplace.toml`
and from every package's own `[dependencies]` — is collected, and one version
satisfying *all* of them is chosen (the highest available). If no version satisfies
all of them, that is a hard error naming every requirer and its range:

```
error: cannot pick one version of `stats`:
  this project requires `^1`
  `regression@1.0.0` requires `^2`
  available versions: 1.0.0, 2.0.0
```

Two versions of the same package never coexist in a build — they would mangle to
identical `pkg__func` names and silently clobber each other. `A -> B -> A` is
rejected with the cycle path printed rather than recursed into.

`add` and `update` re-resolve the whole graph but pass the already-locked versions
as *preferences*, so adding one dependency does not silently bump every other one;
only the package being added or updated is free to move.

The algorithm lives in `src/resolve/graph.rs` and is pure: everything it knows about
the outside world arrives through the `PackageProvider` trait, so it is unit-tested
against synthetic graphs with no filesystem, git, or parsing involved.

### 6.3 Imports are private; there are no re-exports (v1)

A package may call `dep::func()` only for packages it declares itself. The project
may call `pkg::func()` only for packages in its own `library { }` block. A package
that is in the build only as someone else's transitive dependency is not in scope
for anyone who did not ask for it directly, and importing it in a `library { }`
block tells you to `laplace add` it properly.

Consequences for mangling:

- Exported functions stay plain `pkg__func`, transitive dependencies included — no
  content hash, no version fragment. That is safe **because of 6.2**: one version
  per package name per build makes the bare package name a unique prefix, and it
  keeps `--split-functions` output human-readable. **If single-version unification
  is ever relaxed, the mangling scheme must be revisited in the same change.**
- Non-exported functions keep their source names, because they share Stan's one flat
  function namespace with everything else. Two packages defining the same private
  helper is a hard error naming both, not a silent clobber. It is deliberately *not*
  an automatic rename: auto-mangling private names would change the compiled output
  of every existing project, and the byte-identical-output guarantee is worth more
  than the convenience. A future version could mangle privates behind an opt-in.

### 6.4 Transitive dependencies are flattened, never chained

In `--split-functions` mode, each *directly imported* package gets one
`<pkg>.stanfunctions` file containing itself plus its private transitive
dependencies. laplace never emits an `#include` of one `.stanfunctions` file from
another — "what has to ship next to my `.stan` file" stays answerable by reading the
`library { }` block.

Each package is still emitted exactly once across the whole build, so a diamond is
not duplicated: a shared package lands in the file of the *first* direct import that
needs it in dependency order, and the `#include` lines go out in that same order, so
every function is defined before the file that calls it. Chained cross-file includes
are a possible future mode; they are deliberately not in v1.

---

## 7. Possible next work

Tasks 1–9 are done. Candidates for what comes after, roughly in order of how much
they'd be missed:

- **Re-exports.** 6.3 punted them: a library cannot expose a dependency's functions
  to its own consumers. `pub use`-style re-exports are the obvious next step, and
  are the one thing likeliest to force a revisit of the mangling scheme.
- **Chained `.stanfunctions` includes** as an opt-in alternative to 6.4's flattening,
  for projects where the same library is used by several models and duplicating it
  per model is wasteful.
- **`laplace init` for `.laplacelib` packages** — it currently scaffolds a manifest
  from a directory of plain `.stan` files and never writes a `[dependencies]` table.
- **A real registry** (git/http-backed index) instead of the local filesystem one.
- **Opt-in private-name mangling**, per the note in 6.3, once there is a way to make
  it non-breaking for already-committed `.stan` output.
