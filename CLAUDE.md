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
