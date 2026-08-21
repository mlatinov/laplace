# laplace

laplace is a source-to-source preprocessor for [Stan](https://mc-stan.org/): it
compiles `.laplace` files down to plain `.stan` files, adding a package manager
and namespaced imports (`pkg::func()`) on top of a language that has neither
natively.

laplace never hides what actually gets sent to Stan. `laplace build` always
produces a real, readable, committable `.stan` file, and laplace itself is
never a runtime dependency of your model — once `build/model.stan` exists, you
can hand it to `stanc`/cmdstan without laplace installed at all.

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
  checksums. This is what actually gets installed. Commit it to git.

`laplace install` reads **only** the lock, never `laplace.toml`'s ranges —
that's what makes a fresh clone reproducible: two machines with the same
`laplace.lock` always install identical package versions.

The three commands, and when to reach for each:

- **`laplace add <pkg>[@version]`** — introducing a new dependency. Resolves
  a version matching the range (or pins a new one), fetches it, and updates
  both `laplace.toml` and `laplace.lock`.
- **`laplace update <pkg>`** — bumping an existing dependency to the latest
  version matching its current range in `laplace.toml`.
- **`laplace install`** — restoring an already-locked project on a new
  machine (e.g. right after `git clone`). Reads `laplace.lock` only.

## Building

```sh
laplace build <file.laplace> [-o build/model.stan]
```

By default, output goes to `build/<stem>.stan` — e.g. `model.laplace` builds
to `build/model.stan`. Override with `-o`/`--output`.

`--check` is for CI: it runs codegen and diffs the result against the
existing output file instead of writing, exiting non-zero if they differ.
Use it to catch a committed `.stan` file that's gone stale relative to its
`.laplace` source.

`--validate` shells out to `stanc` after writing, to type-check the
generated file. It's off by default, so a normal `laplace build` never
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

## Project layout / for contributors

If you want to work on laplace itself rather than use it, start with
[`CLAUDE.md`](CLAUDE.md) for the non-negotiable design constraints and
[`laplace-project-plan.md`](laplace-project-plan.md) for the task breakdown
and current milestone — this README covers usage only.
