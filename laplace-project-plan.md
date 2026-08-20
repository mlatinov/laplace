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
- **Packages** are folders: `laplace.toml` (manifest) + `.stan` file(s) with
  `// @laplace` doc comments above exported functions (Doxygen/roxygen2-style
  convention, not a new file format — plain comments, ignored by stanc).
- **Two-file dependency model, renv/Cargo-style:**
  - `laplace.toml` — hand-edited, loose version ranges (`gps = "^1.0"`)
  - `laplace.lock` — machine-generated, exact pins + checksums, committed to git
  - `laplace install` reads only the lock (reproducible restore on a new machine)
  - `laplace add <pkg>` / `laplace update <pkg>` re-resolve and rewrite the lock
- **Docs are structured, not just comments.** `@laplace` blocks get extracted into a
  `docs.json` sidecar per installed package version. `laplace doc gps::rbf_cov` reads
  that sidecar — no re-parsing of Stan at lookup time.
- **Stack:** Rust + `pest` (PEG grammar) for the shallow parser, TOML for manifests,
  local filesystem cache at `~/.laplace/packages/<name>/<version>/`.

---

## 2. Repo layout

```
laplace/
  Cargo.toml
  src/
    main.rs            # CLI entry (clap): build, install, add, update, doc
    parser/
      mod.rs
      library_block.rs # finds `library { }`, extracts import statements
      signatures.rs     # scans a .stan file for function sigs + preceding doc comments
    resolve/
      mod.rs            # dependency graph resolution, semver matching
      lockfile.rs        # read/write laplace.lock
    codegen/
      mod.rs             # renaming pass + splicing into functions{} block
      rename.rs
    docs/
      mod.rs             # doc comment -> docs.json extraction + doc lookup/render
    manifest.rs          # laplace.toml (project) + package laplace.toml parsing
  tests/
    fixtures/
      simple_import/     # a minimal .laplace + expected build/model.stan
      no_import/         # passthrough-only case, sanity check
      collision/         # two packages exporting the same function name
  examples/
    gps-model/
      model.laplace
      laplace.toml
      laplace.lock
```

---

## 3. `CLAUDE.md` — seed this at the repo root

Paste this in as-is once you `git init`. It gives every future Claude Code session
the context this whole conversation built up, without you re-explaining it.

```markdown
# laplace

Source-to-source preprocessor: `.laplace` -> plain `.stan`. Adds a package manager,
namespaced imports (`pkg::func()`), and doc lookup on top of Stan, which has none
of these natively.

## Non-negotiable design constraints
- Do NOT attempt to parse full Stan grammar/semantics. Only the `library { }` block
  is deeply parsed. `functions { }`, `data { }`, `parameters { }`, `model { }` etc.
  are treated as opaque text and passed through unmodified except for `pkg::func(`
  call-site rewriting.
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

## Current milestone
See laplace-project-plan.md task list. State which numbered task you're on at the
start of each session.
```

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

---

## 5. Suggested first move

Start a Claude Code session in an empty repo with just `CLAUDE.md` (section 3
above) and this file committed. First prompt to paste in:

```
Read CLAUDE.md and laplace-project-plan.md. Set up the Cargo workspace per the
repo layout in the plan (empty modules with TODOs are fine for now), then
implement Task 1 in full with tests.
```

That gets you a real, tested, committable first slice — import parsing — without
touching anything else. Task 2 can be done in a separate session in parallel
since it doesn't depend on Task 1's output.
