# Editor support: the `laplace_sample::editor` API

Custom samplers are **experimental** in laplace 0.3.0.

An editor integration (the laplace language server in `laplace_tools`, for
example) can report errors in `.lapkernel` and `.lapsampler` files as they
are typed, by calling one Rust library API. This document describes that
API. It does not describe any LSP features: none are built here.

## What it needs

- The `laplace-sampler-syntax` crate (crate name `laplace_sampler_syntax`)
  as a library dependency, from this repository, for example as a git
  dependency pinned to a commit or tag. It holds the parsers and every
  model-free check; laplace-sample builds on it.
- **No BridgeStan, no libclang, no C++ toolchain, no compiled model.**
  - Its dependencies are `anyhow`, `laplace` (the compiler's library),
    `nalgebra` and `pest`. Building it needs a C compiler: pest assembles
    one small file through `cc`. Linking any Rust program on Linux or macOS
    needs the same toolchain, so in practice nothing extra is installed.
  - CI builds it in a container without libclang (`sampler.yml`,
    `syntax-without-libclang`).
  - Do **not** depend on `laplace-sample` for this. It needs libclang at
    build time (bridgestan's build script runs bindgen), and it does not
    compile outside this workspace, because its vendored nuts-rs comes from
    a `[patch]` that applies only here.
- `laplace_sample::editor` is the same module, re-exported, for code that
  already links laplace-sample.

## The API

```rust
use laplace_sampler_syntax::editor::{check_lapkernel, check_lapsampler, Diagnostic, Project, Severity};

// A .lapkernel, as the editor holds it (unsaved text is fine).
let ds: Vec<Diagnostic> = check_lapkernel("kernels/mala.lapkernel", &text, None);

// A .lapsampler. `dir` is the file's directory (local kernel paths resolve
// against it); `project` says where laplace.lock is and where packages are
// installed, for `library { import ... }`.
let project = Project { dir: workspace_root, cache_root: home.join(".laplace/packages") };
let ds = check_lapsampler("samplers/hybrid.lapsampler", &text, &dir, &project);
```

`Project::current()` gives the command-line behaviour: the current directory
and `~/.laplace/packages`.

### `Diagnostic`

| Field | Meaning |
| --- | --- |
| `file` | The file the problem is in. Usually the path you passed. A `.lapsampler` diagnostic about a library kernel names it `<package>/kernels/<name>.lapkernel`, never a machine-specific cache path. |
| `line`, `column` | 1-based; columns count characters (Unicode scalar values: not bytes, not UTF-16 code units, so an editor speaking LSP converts). Where the problem **starts**: a point, not a range. Highlight the token or line there. `0, 0` means the whole file (it could not be read). |
| `severity` | `Error` or `Warning`. Every diagnostic today is an `Error`: the file would be refused by `laplace sample`. `Warning` is in the API so adding warnings later is not a breaking change. |
| `message` | One line, apart from one case below. |

**The multi-line case:** a `.lapsampler` that refers to a kernel file that
does not check gets one diagnostic at the reference. Its message is
`<kernel> does not check:`, followed by that kernel's own diagnostics, one
per indented line, each with its own `file:line:column`. An editor can show
the first line at the reference and offer the rest as related information,
or call `check_lapkernel` on that file directly.

Diagnostics come sorted by file and position.

### What is checked

- **`check_lapkernel`:**
  - syntax, where a syntax error stops parsing, so there is exactly one;
  - types;
  - names, including built-ins from the kernel context;
  - `propose()`/`move()` ending the step;
  - the tuning/state/init rules;
  - `@adapt` tags, including the required `target_accept`.

  `dim` is the size of the block the kernel will update. Pass `None` in an
  editor; it only sharpens size checks.
- **`check_lapsampler`:**
  - syntax and structure: duplicate sections, `rest` last, unique labels;
  - run settings;
  - built-in kernels and their settings;
  - local `.lapkernel` files, read from disk relative to `dir` and checked;
  - library kernels: `import` resolved in `project`'s `laplace.lock`,
    exactly as `laplace sample` resolves them; a path dependency is
    re-synced into the cache;
  - the `adapt()` contracts;
  - `nuts()` not under `mix()`.

### What is not checked

**Block coverage** is not checked: every model parameter in exactly one
block (D2). It needs the model's parameter list, which means compiling the
model with BridgeStan. `laplace sampler check file.lapsampler --model
model.laplace` does it on the command line.

## Stability

These two functions, `Diagnostic`, `Severity` and `Project` are the
supported surface. Everything else in `laplace_sample` is internal and may
change. Message wording may change; positions and severities are tested
(`laplace-sampler-syntax/src/editor.rs`).
