# Prompt: write the laplace README

Paste this directly into Claude Code at the repo root.

---

```
Read CLAUDE.md, laplace-project-plan.md, and the actual current source in
src/ (especially main.rs's Command enum, resolve/mod.rs, and manifest.rs) so
the README reflects what is genuinely implemented right now, not the
aspirational plan. Before writing anything, check whether git-based package
sources (a `git = "..."` dependency form) are implemented yet or still local-
filesystem-registry only, and write the "installing libraries" section to
match whichever is actually true today -- do not describe git support as
available if it isn't wired up yet.

Write README.md to replace the placeholder at the repo root. Use clear
prose and real, runnable command blocks -- every command shown must be one
that actually works against the current CLI (verify against main.rs's
Command enum and its flags, don't invent flags that don't exist). Structure:

## 1. What laplace is (short, 2-3 sentences)
One paragraph explaining the core idea: a source-to-source preprocessor that
compiles `.laplace` files to plain `.stan`, adding a package manager and
namespaced imports (`pkg::func()`) on top of a language that has neither.
Explicitly state the philosophy: laplace never hides what actually gets sent
to Stan -- `laplace build` always produces a real, readable, committable
`.stan` file, and laplace itself is never a runtime dependency of the model.

## 2. Installing laplace
Cover building from source with `cargo install --path . --root <dir>`,
adding that dir to PATH, and verifying with `laplace --help`. Note the Rust
version/edition requirement from Cargo.toml. Mention that `--validate` on
`laplace build` is optional and requires `stanc`/cmdstan on PATH or pointed
to via the LAPLACE_STANC env var -- laplace itself does not require Stan to
be installed for anything except that one flag.

## 3. Quick start: your first .laplace file
A complete, minimal, copy-pasteable walkthrough: write a two-line library
block importing a package, add data/parameters/model blocks, run
`laplace add`, `laplace install`, `laplace build`, and show the expected
build/model.stan output. Use a genuinely trivial example (a single-argument
function), not the full GP kernel example, so a first-time reader isn't
overwhelmed. Explain each generated-output difference (library block gone,
functions renamed pkg__func, everything else byte-identical) inline as
comments next to the shown output, so the reader isn't just told "it works"
but can see exactly what changed and why.

## 4. Creating a library (this is the most important section -- write it
as a complete, standalone tutorial someone could follow having read nothing
else in this README)
Cover, in this order:
  a) A library is just a directory with .stan file(s) plus one added file,
     laplace.toml -- no files need to move or be duplicated from an existing
     Stan-functions repo.
  b) The @laplace doc-comment convention: show a real example with @brief,
     @param (repeated), @return, @example, and explain that these are plain
     Stan comments -- stanc ignores them entirely, so the file remains valid,
     usable Stan on its own even without laplace.
  c) Explicitly explain that a function does NOT need an @laplace comment to
     exist in the file -- undocumented functions are still scanned (so they
     can be renamed if called internally by an exported function) but won't
     appear in `laplace doc` output.
  d) The laplace.toml manifest: name, version, exports -- and stress that
     `exports` is the actual privacy boundary. A function present in the
     .stan file but absent from `exports` cannot be called as pkg::func from
     outside, regardless of whether it has a doc comment.
  e) If `laplace init` exists in the current source, document it here as the
     recommended way to scaffold the manifest (scans existing @laplace
     comments and pre-fills exports, prints which functions were
     included/excluded so the user can adjust by hand). If it does not exist
     yet, skip this and just show the manifest being hand-written, and don't
     reference a command that doesn't exist.
  f) How to make the library available to install: whatever the CURRENT
     implementation actually supports -- check resolve/mod.rs and write
     accurate instructions for either (a) local filesystem registry only
     (placing the directory under the registry root, which is
     ~/.laplace/registry by default or LAPLACE_REGISTRY if set), or
     (b) git-based sources if that's implemented, showing the
     `git = "url", tag = "..."` manifest form and/or the
     `laplace add <pkg> --git <url> --tag <version>` CLI form -- whichever
     is real, and note if both are supported.

## 5. Using a library in a model
Show library { import pkg } and library { import pkg@1.2.0 } (pinned),
pkg::func(...) call sites, and explain the laplace.toml / laplace.lock split:
laplace.toml is hand-edited with loose version ranges, laplace.lock is
machine-written with exact pins and checksums and should be committed to
git. `laplace install` reads ONLY the lock, never laplace.toml's ranges
directly -- this is what makes a fresh clone reproducible. Show
`laplace add`, `laplace update`, and `laplace install` and explain when
each is used (add when introducing a new dependency, update when bumping an
existing one, install when restoring an already-locked project on a new
machine, e.g. after git clone).

## 6. Building
Document `laplace build <file.laplace> [-o path]`, the default output path
behavior (build/<stem>.stan), `--check` (for CI, diffs without writing,
non-zero exit on drift), and `--validate` (shells out to stanc, off by
default, configurable via LAPLACE_STANC). State plainly that build output
is deterministic and meant to be committed to git, and that a human should
be able to open build/*.stan and fully understand it without laplace
installed.

## 7. Looking up documentation
Document `laplace doc <pkg>::<func>`, show real example output, and note it
reads a docs.json sidecar generated at install time -- no network access or
re-parsing needed at lookup time.

## 8. Project layout / for contributors
Briefly point at CLAUDE.md and laplace-project-plan.md for anyone who wants
to work on laplace itself, rather than duplicating that content here -- this
README is for USERS of laplace, not contributors to it.

Keep the tone plain and example-driven throughout -- prefer showing a real
command and its real output over describing behavior abstractly. Do not
invent commands, flags, or file formats that don't exist in the current
source; if something from laplace-project-plan.md hasn't actually been
built yet, either omit it or clearly mark it as planned/not yet available
rather than documenting it as current behavior.
```
