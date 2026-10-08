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
// mathutils v1.0.0 (pub) -- mathutils/mathutils.stan:7   <- where this came from
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

For a package written as plain `.stan` files, `exports` is the privacy
boundary, not the doc comment. A function that's present in the `.stan` file
but absent from `exports` cannot be called as `pkg::func` from outside the
package — regardless of whether it has an `@laplace` comment. Documentation
and visibility are two separate concerns.

A package written as `.laplacelib` files uses the `pub` keyword instead, and
leaves `exports` out of the manifest entirely — see
[Libraries that depend on libraries](#libraries-that-depend-on-libraries) below.
Listing a `.laplacelib` item in `exports` is an error, so there is never a
question of which one wins.

Either way, private functions are still *emitted* into the compiled output —
public ones call them — and every function, public or private, gets the
`pkg__` prefix. Private is an access rule, not hiding, and two packages can
safely define a private helper of the same name.

### Scaffolding the manifest with `laplace init`

Rather than hand-writing `laplace.toml`, run `laplace init` inside a
directory of `.stan` files to generate a starter manifest. It guesses `name`
from the directory name, sets `version = "0.1.0"`, and pre-fills `exports`
with every `@laplace`-documented function — printing which functions were
included and which were left out so you can adjust either list by hand.
(`exports` only ever names functions from plain `.stan` files; for a
`.laplacelib` package `init` reports which items are `pub` and which are
still private, since only the source can say.)

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
# no `exports`: a .laplacelib file says what's public with `pub`

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
pub vector centre(vector x) {
  return x - stats::mean_(x) / scale(x);
}

real scale(vector x) {          // private: no `pub`
  return sd(x);
}
```

**`pub` is how a `.laplacelib` file says what's public.** Items are private to
their package unless marked `pub`, and the manifest's `exports` is left out —
in fact naming a `.laplacelib` item in `exports` is an error telling you to
write `pub` instead. Calling a private item from outside its package is a
laplace error with a file, line and column:

```
error: `stats::sum_values` is private to package `stats`
  --> model.laplace:9:12
  help: only items marked `pub` can be used outside their package
```

Visibility belongs to a *name*, not to one definition: if a name has several
overloads, either all of them are `pub` or none are. `pub` never appears in
generated output, and `laplace doc` won't show a private item.

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

## Templates: boilerplate that spans blocks

Some Stan patterns aren't a function — they're a handful of declarations and
statements spread across `parameters`, `transformed parameters` and `model`.
Non-centred parameterization is the classic one. A template packages that up,
and `@use` drops it into a model in one line.

In a `.laplacelib`:

```stan
// @laplace
// @brief Non-centred parameterization for a vector of coefficients.
pub @template ncp($name: ident, $N: expr) {
  parameters {
    vector[$N] ${name}_raw;
    real<lower=0> ${name}_sigma;
  }
  transformed parameters {
    vector[$N] $name = ${name}_sigma * ${name}_raw;
  }
  model {
    ${name}_raw ~ std_normal();
    ${name}_sigma ~ exponential(1);
  }
}
```

In your model:

```stan
library {
  import stats
}

@use stats::ncp(theta, K);

data {
  int<lower=1> K;
  vector[K] y;
}
model {
  y ~ normal(theta[1], 1);
}
```

and the compiled `.stan`:

```stan
data {
  int<lower=1> K;
  vector[K] y;
}

parameters {
  // begin @use stats::ncp(theta, K) -- model.laplace:5
  vector[K] theta_raw;
  real<lower=0> theta_sigma;
  // end @use stats::ncp
}

transformed parameters {
  // begin @use stats::ncp(theta, K) -- model.laplace:5
  vector[K] theta = theta_sigma * theta_raw;
  // end @use stats::ncp
}

model {
  // begin @use stats::ncp(theta, K) -- model.laplace:5
  theta_raw ~ std_normal();
  theta_sigma ~ exponential(1);
  // end @use stats::ncp

  y ~ normal(theta[1], 1);
}
```

Blocks you don't have are created, in Stan's own block order. Pieces go
*before* your own content in each block, in `@use` order — a template makes
things (`theta`) that your code then uses, and Stan wants declarations first.

### The two placeholder kinds

`$name: ident` takes a **name**. It's what lets you use one template twice
without the two expansions colliding:

```stan
@use stats::ncp(theta, K);
@use stats::ncp(beta, P);
```

`${name}` is the explicit form, and it's required when you're building a longer
name out of it (`${name}_raw`). Only `ident` placeholders can do that.

`$mu: expr` takes a whole **expression** — a prior, a linear predictor — and
carries it through untouched. laplace parenthesizes it when the surrounding
operators could change its meaning and leaves it alone when they can't, so
`observation(y, mu + theta, sigma)` gives you `y ~ lognormal((mu + theta),
sigma)` while a distribution passed as an `expr` still works after a `~`.

### Hygiene: why a template can't collide with your model

**Every variable a template declares has to be named from an `ident`
placeholder.** A fixed `real tmp;` is refused when the library is written, not
when you use it — because the second `@use` would redeclare it.

That one rule makes the rest fall out. A template can't quietly reach for one
of your variables either: once fixed names are gone, the only things a body may
name are its own placeholders and its own `for` loop variables, so anything
else is a reference to your model and laplace rejects it. If a template needs
one of your variables, it has to ask for it as a placeholder.

And if an expansion would declare a name that already exists — yours, or
another expansion's — the build stops and names both sources:

```
error: `theta_raw` is declared twice: once by model.laplace:14, and again by
       `@use stats::ncp(theta, K)` at model.laplace:5:1
  help: give one of them a different name -- that is what the `ident` placeholder is for
```

Function calls inside a template body resolve in the **library's** scope, not
yours, so a template can use its own package's private helpers and you never
have to know they exist.

## Statement macros: one line, repeated

A template spans blocks. A macro expands **in place, inside one block** — and
can repeat itself over a list, which is the thing templates can't do.

In a `.laplacelib`:

```stan
// @laplace
// @brief Give several parameters the same prior.
pub @macro priors(each $p: ident, $dist: expr) : stmt in model {
  $p ~ $dist;
}
```

The header says everything: `each $p` is the list parameter, `: stmt` is what
the body expands to, and `in model` is where it may be used.

In your model:

```stan
model {
  @expand stats::priors([alpha, beta, gamma], normal(0, 1));
  y ~ normal(alpha + beta * x, gamma);
}
```

and the compiled `.stan`:

```stan
model {
  // begin @expand stats::priors -- model.laplace:15
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  gamma ~ normal(0, 1);
  // end @expand stats::priors
  y ~ normal(alpha + beta * x, gamma);
}
```

The expansion replaces the `@expand` line and keeps its indentation. Macros
declare variables too, and the `ident` placeholder keeps the names apart:

```stan
pub @macro z_scores(each $p: ident, $scale: expr) : stmt in transformed parameters {
  real ${p}_z = $p / $scale;
}
```

over `[alpha, beta]` gives you `alpha_z` and `beta_z`.

### What the header buys you

`in <blocks>` is checked in both directions, which is the point of writing it
down. Expand a macro somewhere it doesn't belong and laplace says so:

```
error: macro `stats::priors` cannot be expanded in `generated quantities`
  --> model.laplace:19:3
  help: it declares `in model`
```

And a macro that claims a block its own body couldn't legally go in is caught
when the *library* is written, not when you use it — Stan won't take `y ~ ...`
in `generated quantities`, so a `~` body claiming that block is wrong before
anyone touches it. Same for calling an `_rng` function outside `transformed
data` and `generated quantities`.

Macros get the same hygiene rules as templates: declared names must come from
an `ident` placeholder, no reaching for your variables, and collisions are
refused — including a list that repeats an element, and including names a
template expansion in the same model would introduce.

An empty list is an error rather than a silent no-op: an `@expand` line that
produces nothing looks like it does something.

## Passing functions to functions

Stan can't take a function as an argument. laplace can, by generating one
specialized copy of your function per distinct function you pass in — the
same idea as a C++ template or a Rust generic. Nothing generic reaches the
`.stan` file.

Declare the shape in the parameter list:

```stan
functions {
  real add_one(real x) {
    return x + 1;
  }

  real apply_twice(real x, func(real) -> real f) {
    real a = f(x);
    return f(a);
  }
}

transformed data {
  real r = apply_twice(5, add_one);
}
```

compiles to:

```stan
functions {
  real add_one(real x) {
    return x + 1;
  }

// monomorphized: apply_twice with f = add_one -- model.laplace:5
real apply_twice__add_one(real x) {
  real a = add_one(x);
  return add_one(a);
}
}

transformed data {
  real r = apply_twice__add_one(5);
}
```

The argument is a bare function **name** — one of yours, or a `pub` library
function written `pkg::func`. There are no lambdas. Passing the same
function twice produces one copy; a higher-order function you never call is
not emitted at all, since it has no Stan form.

Copies are emitted at the end of the `functions { }` block, after everything
they might call, with a forward declaration at the top for any copy another
function calls.

### Sized return types and `@wait`

A library author writing a higher-order function often needs a local
variable for `f`'s result — and Stan needs local declarations to be
*sized*, which the author can't know in advance. `@wait(f)` stands for
"whatever type ends up bound to `f`":

```stan
pub matrix expand_rows(vector x, func(real) -> vector f) {
  matrix[num_elements(x), @wait(f).size] out;
  for (i in 1:num_elements(x)) {
    @wait(f) row = f(x[i]);
    out[i] = row';
  }
  return out;
}
```

The size has to come from somewhere, and a Stan signature carries none. So
laplace lets you annotate a return type, and strips the annotation from the
output:

```stan
vector[2] to_pair(real x) {        // laplace source
  return [x, x * 2]';
}
```

```stan
vector to_pair(real x) {           // Stan output
  return [x, x * 2]';
}
```

Bind that into `expand_rows` and `@wait(f).size` becomes `2`, `@wait(f)`
becomes `vector[2]`. Use `.rows` and `.cols` for a matrix return. If the
bound function has no annotation, laplace says so at the call site rather
than letting stanc complain about generated code.

A size may also be an expression over the function's own `int` parameters
(`vector[K] basis(real t, int K)`). Those are only known per call, so
`@wait(f)` is then allowed just where the declaration is initialized by a
direct call — `@wait(f) r = f(t, K);` — and the arguments get substituted
into the size.

### What isn't supported yet

Each of these is a laplace error with a `help:` line, not a surprise from
stanc:

- a `func` inside a `func`, or passing a higher-order function as an argument
- using a functional parameter for anything but calling it (so it can't be
  stored, returned, or forwarded to another function)
- one higher-order function calling another
- a recursive higher-order function
- binding a Stan built-in — wrap it first: `real exp_(real x) { return exp(x); }`
- an array as a functional parameter's return type

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

// stats v1.0.0 (pub) -- stats/stats.laplacelib:5
// @laplace
// @brief Arithmetic mean of a vector.
real stats__mean_(vector x) {
  return sum(x) / num_elements(x);
}

// regression v1.0.0 (pub) -- regression/regression.laplacelib:9
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

**Every function is mangled, private ones included.** Stan has one flat
function namespace, so prefixing everything with its package name is what
makes two packages' same-named private helpers unable to collide. Visibility
(`pub`, or `exports` for `.stan` files) decides only who may *name* an item
as `pkg::item`; it has nothing to do with how the item is named in the output.

**Generated code says where it came from.** Every item spliced in from a
library carries a one-line provenance comment — package, version, visibility,
and the file and line in the library's own source. It's a package-relative
path with no timestamp, so the same input and lockfile still produce
byte-identical output on another machine.

**Templates are the one place laplace parses Stan.** A template body is
laplace's own construct — delimited, small, and written against laplace — so it
is fully parsed, and a mistake in one is a loud error on the library author's
code. Everywhere else, block contents stay opaque text. Even inside a template,
laplace knows only Stan's *declaration* shape and whether an identifier is
followed by `(`; it has no expression parser and no type checker.

**`__` is reserved.** Since `pkg::func` becomes `pkg__func`, a hand-written
identifier containing `__` is rejected in `.laplace` and `.laplacelib`
sources, and in package names. Generated and hand-written names are then
provably disjoint rather than merely unlikely to clash. (Plain `.stan` package
files are exempt — they're ordinary Stan and predate the rule.)

**Functions as arguments are compiled away, not emulated.** One specialized
copy per distinct function bound, named `<hof>__<bound>`, with the generic
original never emitted. Copies go at the end of the `functions { }` block so
each one follows everything it calls — Stan wants a function declared before
it's used, and a library function bound to one of yours would otherwise be
emitted before the function it calls.

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
