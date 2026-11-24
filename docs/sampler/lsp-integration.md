# Editor support: the `laplace_sampler_syntax::editor` API

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

## Names: hover, completion, definitions

Three functions describe the names in sampler files. Each one reads the
table the checker uses, so it can't drift from what is checked.

```rust
use laplace_sampler_syntax::editor::{builtins, combinators, kernel_index};

for b in builtins() { /* name, kind, signatures, doc, rng, needs_chain */ }
for c in combinators() { /* name, signature, doc, settings, in_kernel_lines, example */ }
let (kernels, problems) = kernel_index(&project, sampler_dir);
```

- **`builtins()`:** every `.lapkernel` built-in, from the registry, in the
  checker's order.
  - `kind` is `Value` (`q`, `lp`, `grad`, `dim`, `metric`), `Function`, or
    `Terminal` (`propose`, `move`, which end a step).
  - `signatures` has one entry per overload, with types:
    `real log_density(vector[dim] x)`. A test checks each signature against
    the type rule.
- **`combinators()`:** `seq`, `mix`, `repeat`, `adapt` and `on`.
  - `settings` lists the named arguments after the kernel (`adapt` takes
    `target_accept`).
  - `in_kernel_lines` is false for `on`: a kernel line already means
    `on(block, kernel)`, so don't offer it in completion.
- **`kernel_index(project, from)`:** every kernel a `.lapsampler` in
  directory `from` can name.
  - The kinds:
    - the built-ins, with their `.lapsampler` settings and defaults;
    - the kernels of each kernel package in `project`'s `laplace.lock`;
    - the `.lapkernel` files under `project.dir`, skipping hidden
      directories and `target`.
  - `name` is what the sampler writes: `mala`, `langevin::mala`, or
    `./kernels/mine.lapkernel`, relative to `from`.
  - `file` is the `.lapkernel` to open. For a path dependency, it is the
    file in the package's own directory, not the cache copy that each
    resolve overwrites. A built-in has no `.lapkernel`;
    `BuiltIn::source()` gives its Rust implementation.
  - `settings` are a built-in's settings or a kernel's tuning values, each
    with its declaration, default, `@adapt` tag and `//@param` text.
    `ends_in` lists the terminals the step can end in.
  - `doc` is the text `laplace sampler doc` prints.
  - A kernel that does not check has `error` set. A package that does not
    resolve isn't listed; its message is in `problems`.
  - Resolving packages syncs path dependencies into the cache, as
    `check_lapsampler` does.

## Stability

The supported surface:

- `check_lapkernel` and `check_lapsampler`, with `Diagnostic`, `Severity`
  and `Project`;
- `builtins`, `combinators` and `kernel_index`, with their types.

Everything else is internal and may change. Message wording may change; positions and severities are tested
(`laplace-sampler-syntax/src/editor.rs`).
