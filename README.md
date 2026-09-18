# laplace

laplace is a source-to-source preprocessor for [Stan](https://mc-stan.org/): it
compiles `.laplace` files down to plain `.stan` files, adding a package manager
and namespaced imports (`pkg::func()`) on top of a language that has neither
natively.

laplace never hides what actually gets sent to Stan. `laplace build` always
produces a real, readable, committable `.stan` file, and laplace itself is
never a runtime dependency of your model — once `build/model.stan` exists, you
can hand it to `stanc`/cmdstan without laplace installed at all.

Libraries can depend on other libraries: a package declares its own
`[dependencies]`, laplace resolves the whole graph (unifying diamonds,
rejecting cycles), and pins it in `laplace.lock`. See [Libraries that depend on
libraries](#libraries-that-depend-on-libraries).

## Installing laplace

laplace is built from source with Cargo (edition 2021 — any reasonably recent
stable Rust toolchain works):

```sh
cargo install --path . --root ~/.local
```

Add the install directory's `bin/` to your `PATH` if it isn't already there
(e.g. `export PATH="$HOME/.local/bin:$PATH"`), then verify:

```sh
laplace --help
```

laplace does not require Stan or `stanc` to be installed for anything except
the optional `--validate` flag on `laplace build` (see [Building](#building)),
which shells out to `stanc` on `PATH` or wherever `LAPLACE_STANC` points.

## Quick start: your first `.laplace` file

This walks through the smallest possible example: one package, one function,
one model that calls it.

First, a package. A package is just a directory with a `.stan` file and a
`laplace.toml`. Here's a package called `mathutils` with a single exported
function:

```stan
// mathutils.stan
// @laplace
// @brief Doubles a real value.
// @param x The value to double.
// @return 2 * x.
// @example mathutils::double(3.0)
real double(real x) {
  return 2 * x;
}
```

```toml
# laplace.toml
name = "mathutils"
version = "1.0.0"
exports = ["double"]
```

(How to make a package like this installable is covered in full in [Creating
a library](#creating-a-library) — for this walkthrough, assume `mathutils` is
already sitting in your local registry at `~/.laplace/registry/mathutils/1.0.0/`.)

Now, in a project directory, write a model that imports it:

```stan
// model.laplace
library {
  import mathutils
}

data {
  real x;
}

transformed data {
  real y = mathutils::double(x);
}
```

Resolve and install the dependency, then build:

```sh
laplace add mathutils
laplace install
laplace build model.laplace
```

```
$ laplace add mathutils
added mathutils@1.0.0
$ laplace install
installed 1 package
$ laplace build model.laplace
wrote build/model.stan (21 lines, 1 dependency)
```

And `build/model.stan`:

```stan
functions {
// @laplace
// @brief Doubles a real value.
// @param x The value to double.
// @return 2 * x.
// @example mathutils::double(3.0)
real mathutils__double(real x) {                  // <- pkg::func mangled to pkg__func
  return 2 * x;
}


}


data {                                             // <- data/parameters/model blocks:
  real x;                                          //    byte-identical to the source
}

transformed data {
  real y = mathutils__double(x);                   // <- call site rewritten to match
}
```

What changed, and what didn't:
- The `library { ... }` block is gone entirely — it's a laplace-only construct
  and has no meaning to `stanc`.
- `mathutils`'s `double` function was renamed `mathutils__double` (namespacing
  is name-mangling, not a real Stan feature) and spliced into a `functions {}`
  block, doc comment and all — `// @laplace` comments are plain Stan comments,
  so they survive into the compiled output.
- The call site `mathutils::double(x)` became `mathutils__double(x)`.
- The `data { }` and `transformed data { }` blocks are otherwise
  byte-for-byte what you wrote — laplace never parses or touches Stan
  statements outside of `pkg::func(` rewriting.

## Creating a library

This section is self-contained — you can follow it without having read
anything else in this README.

### A library is just a directory

No files need to move or be duplicated from an existing Stan-functions repo.
Take a directory of `.stan` files that already work as ordinary Stan, and add
one file: `laplace.toml`. That's it — it's now a laplace package.

### The `@laplace` doc-comment convention

Functions you want documented (and, as covered below, exportable) get a
comment block starting with `// @laplace` directly above them:

```stan
// @laplace
// @brief Squared exponential (RBF) covariance matrix.
// @param x Vector of input locations.
// @param alpha Marginal standard deviation of the GP.
// @param rho Length-scale of the GP.
// @return An N x N positive semi-definite covariance matrix.
// @example gps::rbf_cov(x, 1.0, 0.5)
matrix rbf_cov(vector x, real alpha, real rho) {
  return gp_exp_quad_cov(x, alpha, rho);
}
```

`@brief`, `@param` (repeatable, one per parameter), `@return`, and `@example`
are all recognized. These are plain `//` Stan comments — `stanc` ignores them
completely, so the file stays valid, usable Stan on its own even if you never
run it through laplace.

### Undocumented functions are still functions

A function does **not** need an `// @laplace` comment to exist in the file.
Undocumented functions are still scanned — so they get renamed correctly if
an exported function calls them internally — they just won't show up in
`laplace doc` output, and `laplace init` (below) won't guess them as exports.

### The manifest: `name`, `version`, `exports`

```toml
name = "gps"
version = "1.0.0"
exports = ["rbf_cov"]
```

`exports` is the actual privacy boundary, not the doc comment. A function
that's present in the `.stan` file but absent from `exports` cannot be called
as `pkg::func` from outside the package — regardless of whether it has an
`@laplace` comment. Documentation and visibility are two separate concerns.

One caveat about non-exported functions: laplace leaves their names alone
(only exports get the `pkg__` prefix), and Stan has a single flat function
namespace. If two packages in the same build both define a private helper
called `softplus`, the build fails with an error naming both packages rather
than silently emitting two definitions. Give private helpers distinctive
names, or export them.

### Scaffolding the manifest with `laplace init`

Rather than hand-writing `laplace.toml`, run `laplace init` inside a
directory of `.stan` files to generate a starter manifest. It guesses `name`
from the directory name, sets `version = "0.1.0"`, and pre-fills `exports`
with every `@laplace`-documented function — printing which functions were
included and which were left out so you can adjust either list by hand:

```
$ laplace init
wrote laplace.toml for `mypkg`
included 1 exported function: add_one
note: 1 undocumented function left out of exports (add manually if this guess is wrong): helper
```

It refuses to run (rather than overwrite) if `laplace.toml` already exists in
that directory.

### Making the library available to install

laplace resolves dependencies from two kinds of source, and you can mix both
in the same project:

**Local filesystem registry** — the default. Place the package directory
under the registry root, named `<registry>/<pkg-name>/<version>/`:

```
~/.laplace/registry/gps/1.0.0/
  laplace.toml
  gps.stan
```

The registry root defaults to `~/.laplace/registry`, overridable with the
`LAPLACE_REGISTRY` environment variable. Once it's there, `laplace add gps`
resolves against it directly.

**Git repositories** — pin a dependency straight to a git repo instead of a
registry entry, either by hand-editing `laplace.toml`:

```toml
[dependencies]
gps = { git = "https://github.com/user/gps-stan", tag = "0.1.0" }
# or, pinned to a commit instead of a tag:
gps2 = { git = "https://github.com/user/gps2-stan", rev = "abc123" }
```

or via the CLI:

```sh
laplace add gps --git https://github.com/user/gps-stan --tag 0.1.0
```

Exactly one of `tag`/`rev` must be set per git dependency. A git-sourced
package is otherwise treated identically to a registry one from that point
on — same checksum in the lock, same install cache layout.

By default the package is expected at the **root** of the repository: its
`laplace.toml` sits next to the repo's `.git`. If the repo keeps the package
below the top level — beside a README, or one of several packages in a
monorepo — point `subdir` at the directory holding its `laplace.toml`:

```toml
[dependencies]
gps = { git = "https://github.com/user/gps-stan", tag = "0.1.0", subdir = "laplace" }
```

```sh
laplace add gps --git https://github.com/user/gps-stan --tag 0.1.0 --subdir laplace
```

The subdirectory is part of the source's identity: it is recorded in the lock
as `git+<url>@<ref>#<subdir>`, only that directory is checksummed and
installed, and two dependencies naming the same repo and ref but different
subdirectories are different packages (and conflict, like any other two git
sources for one package name). It must be a plain relative path inside the
repository — an absolute path or one containing `..` is rejected, in
`laplace.toml` and in `laplace.lock` alike.

## Libraries that depend on libraries

A library can build on another library. Say `regression` wants to use `stats`'s
`mean_`. Two things change relative to a leaf package.

**1. The manifest gets a `[dependencies]` table**, with exactly the same syntax
as a project's `laplace.toml`:

```toml
# regression/laplace.toml
name = "regression"
version = "1.0.0"
exports = ["centre"]

[dependencies]
stats = "^1.0"
```

**2. The source file becomes `.laplacelib`** instead of `.stan`, so it can carry
a `library { }` block and `pkg::func()` calls:

```stan
// regression/regression.laplacelib
library {
  import stats
}

// @laplace
// @brief Centre a vector on its mean.
// @param x The vector to centre.
// @return `x` minus its mean.
vector centre(vector x) {
  return x - stats::mean_(x);
}
```

`.laplacelib` is a relaxed dialect: bare function definitions, an optional
`functions { }` wrapper around them, and an optional `library { }` block. The
model-shaped blocks (`data`, `transformed data`, `parameters`, `transformed
parameters`, `model`, `generated quantities`) are rejected with a clear error —
a library provides functions to a model, it isn't a model. A package can mix
`.laplacelib` and plain `.stan` files freely; the `.stan` ones are ordinary
Stan, passed through verbatim, and can't import anything.

### What the consumer sees

Nothing new. A project that wants `regression` just adds it, and `stats` comes
along automatically:

```
$ laplace add regression
added regression@1.0.0
$ laplace install
installed 2 packages
$ laplace build model.laplace
wrote build/model.stan (33 lines, 1 dependency + 1 transitive)
```

`laplace.toml` still records only what you asked for; `laplace.lock` records the
whole graph, including the edges between packages:

```toml
root = ["regression"]

[[package]]
name = "regression"
version = "1.0.0"
checksum = "sha256:..."
source = "registry"
dependencies = ["stats"]

[[package]]
name = "stats"
version = "1.0.0"
checksum = "sha256:..."
source = "registry"
dependencies = []
```

That's what lets `laplace install` restore the full graph on a new machine
without re-resolving a single version range.

### Imports are private

`regression` importing `stats` does **not** give *your* model access to
`stats`. If you want to call `stats::mean_` yourself, `laplace add stats` and
import it in your own `library { }` block — at which point laplace makes sure
both of you agree on one version of it. Libraries can't borrow each other's
dependencies either: a package may only call packages listed in its own
`[dependencies]`.

There is no re-export mechanism yet.

### Version conflicts and cycles

A build contains **exactly one version of any package**. When several
dependents want the same package, laplace picks one version satisfying all of
their ranges (the newest that qualifies). If no version qualifies, the build
stops and names everyone involved:

```
error: cannot pick one version of `stats`:
  this project requires `^1`
  `regression@1.0.0` requires `^2`
  available versions: 1.0.0, 2.0.0
```

The fix is to widen one of the ranges, not to run two copies — two copies would
mangle to the same `stats__mean_` name and clobber each other.

Circular dependencies are rejected outright, with the cycle printed:

```
error: dependency cycle: alpha -> beta -> alpha
```

`laplace add` and `laplace update` only move the package you name. Everything
else stays at its locked version unless a constraint forces it to move, so
adding one dependency never silently bumps the rest of your graph.

## Using a library in a model

Import a package in the `library { }` block, bare (latest resolved version)
or pinned:

```stan
library {
  import gps
  import stats@1.2.0
}
```

...then call its exported functions as `pkg::func(...)` anywhere in the rest
of the file.

Dependencies are tracked across two files:

- **`laplace.toml`** — hand-edited, loose version ranges (`gps = "^1.0"`) or
  git sources. This is where you state what you're willing to accept.
- **`laplace.lock`** — machine-written, exact resolved versions and
  checksums for the *whole* graph, transitive dependencies included, plus the
  edges between them. This is what actually gets installed. Commit it to git.

`laplace install` reads **only** the lock, never `laplace.toml`'s ranges —
that's what makes a fresh clone reproducible: two machines with the same
`laplace.lock` always install identical package versions.

The three commands, and when to reach for each:

- **`laplace add <pkg>[@version]`** — introducing a new dependency. Resolves
  a version matching the range (or pins a new one), pulls in whatever that
  package itself depends on, fetches it all, and updates both `laplace.toml`
  and `laplace.lock`.
- **`laplace update <pkg>`** — bumping an existing dependency to the latest
  version matching its current range in `laplace.toml`. Only `<pkg>` moves;
  the rest of the graph keeps its pins unless a constraint forces otherwise.
- **`laplace install`** — restoring an already-locked project on a new
  machine (e.g. right after `git clone`). Reads `laplace.lock` only.

## Building

```sh
laplace build <file.laplace> [-o build/model.stan]
```

By default, output goes to `build/<stem>.stan` — e.g. `model.laplace` builds
to `build/model.stan`. Override with `-o`/`--output`.

`--split-functions` changes the output *shape*. By default every imported
function is inlined into the compiled `.stan` file, so you ship one
self-contained file. With `--split-functions`, each directly imported package
instead gets its own `<pkg>.stanfunctions` file next to the output, and the
`functions { }` block gets an `#include` line per package:

```
$ laplace build model.laplace --split-functions
wrote build/model.stan (15 lines, 1 dependency + 1 transitive)
wrote build/regression.stanfunctions (stats, regression)
note: keep the .stanfunctions files next to build/model.stan and compile with the include path
set to that directory -- `stanc --include-paths=build`, or `cmdstan_model(..., include_paths = "build")` in cmdstanr
```

```stan
// build/model.stan
functions {
#include "regression.stanfunctions"

}


data {
  int<lower=1> N;
  vector[N] y;
}

model {
  vector[N] c = regression__centre(y);
  c ~ std_normal();
}
```

```stan
// build/regression.stanfunctions
// Generated by laplace from package `regression` 1.0.0. Do not edit.
// Bundled dependencies of `regression`: stats 1.0.0.

// @laplace
// @brief Arithmetic mean of a vector.
real stats__mean_(vector x) {
  return sum(x) / num_elements(x);
}

// @laplace
// @brief Centre a vector on its mean.
vector regression__centre(vector x) {
  return x - stats__mean_(x);
}
```

Notes on split mode:

- Functions you wrote yourself in the `.laplace` file's `functions { }` block
  stay inline. Only *imported* package functions move out.
- A transitive dependency has no file of its own — it's flattened into the file
  of the package that pulled it in (`stats` above). laplace never generates an
  `#include` of one `.stanfunctions` file from another, so "what has to ship
  next to my `.stan` file" is answerable from your `library { }` block. If two
  of your direct imports share a dependency, it's still emitted exactly once,
  and the `#include` lines are ordered so every function is defined before it's
  used.
- The `.stanfunctions` files must travel with the `.stan` file, and `stanc`
  must be told where they are: it does not resolve `#include` relative to the
  including file, so pass `--include-paths=<dir>` (or `include_paths =` in
  cmdstanr). `laplace build --validate` does this for you.
- A project with no imports is unaffected — `--split-functions` is a no-op.
- It's off by default because one self-contained `.stan` file is the more
  portable artifact. Turn it on when the inlined output has grown too big to
  read comfortably.

`--check` is for CI: it runs codegen and diffs the result against the
existing output file instead of writing, exiting non-zero if they differ.
Use it to catch a committed `.stan` file that's gone stale relative to its
`.laplace` source. With `--split-functions` it checks the `.stanfunctions`
files too.

`--validate` shells out to `stanc` after writing, to type-check the
generated file. It passes the output directory as an include path and sends
`stanc`'s generated C++ to a scratch directory, so nothing but the `.stan`
(and any `.stanfunctions`) lands in `build/`. It's off by default, so a normal `laplace build` never
requires Stan to be installed. Point it at a specific binary with the
`LAPLACE_STANC` environment variable if `stanc` isn't on `PATH`.

Build output is deterministic: the same `.laplace` source plus the same
`laplace.lock` always produces byte-identical `.stan` output. It's meant to
be committed to git — a human should be able to open `build/model.stan` and
fully understand it without laplace installed at all.

## Looking up documentation

```sh
laplace doc <pkg>::<func>
```

```
$ laplace doc mathutils::double
mathutils::double(x: real) -> real

Doubles a real value.

Parameters:
  x  The value to double.

Returns:
  2 * x.

Example:
  mathutils::double(3.0)
```

This reads a `docs.json` sidecar that's generated once, at install time —
there's no network access or Stan re-parsing at lookup time. A function with
no `@laplace` comment still prints its signature, with a note that no
documentation is available for it.

## Design decisions

A few choices are worth stating outright, because they're baked into published
packages' symbol names and into the lockfile format, and are expensive to
change later.

**laplace is not a Stan compiler.** It never parses Stan's grammar or
semantics. Only the `library { }` block is parsed in depth, plus a shallow
scan for top-level block keywords and function signatures. Everything else —
every statement in every block — is opaque text, passed through byte for byte
except for `pkg::func(` call-site rewriting. That's the reason the compiled
output is trustworthy: laplace can't quietly change the meaning of code it
never understood.

**Build output is deterministic.** The same source plus the same
`laplace.lock` always produces byte-identical `.stan` output. No timestamps,
no hash-map iteration order leaking into generated code. It's meant to be
committed and diffed.

**The lockfile is a graph, not a list.** `laplace.lock` records `root` (your
direct dependencies) and a `dependencies` list on every package, so
`laplace install` reconstructs the entire transitive graph without re-resolving
a single version range. Locks written before this format still read correctly:
a missing `dependencies` means "leaf", and a missing `root` means "every
locked package is direct".

**One version of any package per build.** Every requirement on a package name
— from your `laplace.toml` and from every package's own `[dependencies]` — is
collected, and one version satisfying all of them is chosen. If none exists,
the build stops and names every requirer. Two copies of a package would mangle
to the same `pkg__func` names and silently clobber each other, so this isn't
negotiable. Cycles are rejected with the path printed.

**Mangling is a plain `pkg__func` prefix**, for transitive dependencies too —
no content hash, no version fragment. That's safe *because* of the
one-version-per-build rule, which makes a package name a unique prefix, and it
keeps `--split-functions` output readable. If single-version unification is
ever relaxed, the mangling scheme has to be revisited in the same change.

**Non-exported functions keep their names.** Only exports get the `pkg__`
prefix. Since Stan has one flat function namespace, two packages defining the
same private helper is a hard error naming both, rather than a silent double
definition. Auto-mangling private names would be tidier but would change the
compiled output of every existing project, and byte-identical output is worth
more.

**Imports are private, and there are no re-exports.** A package may call only
the packages in its own `[dependencies]`; your model may call only the
packages in its own `library { }` block. Depending on something transitively
doesn't put it in scope.

**Transitive dependencies are flattened, never chained.** In
`--split-functions` mode, laplace never emits an `#include` of one
`.stanfunctions` file from another, so "what has to ship next to my `.stan`
file" is answerable from the `library { }` block alone. Each package is still
emitted exactly once across the build, ordered so every function is defined
before it's used.
