# Changelog

All notable changes to the laplace compiler are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

This file tracks *compiler* releases (the `laplace` binary). A Stan library's
own versions are its business, tagged with `laplace release`.

## [0.2.0] - Unreleased

### Read this first

- **Packages can now require a compiler version.** A package or project
  `laplace.toml` may say `laplace = ">=0.2"`. A compiler outside that range
  refuses to build, install or document it, naming the required range, its
  own version and `laplace self-update`. Library authors who use `pub`,
  `.laplacelib` files, templates, macros or functions-as-arguments should add
  `laplace = ">=0.2"`. Compilers older than 0.2.0 do not know the key and
  ignore it, so they still fail on the newer syntax itself. The key protects
  your users from 0.2 onwards.
- **`.laplacelib` + `pub` is the visibility model for library sources**
  (introduced in 0.1.0, restated here because 0.2.0 is the first version that
  ships it as a release). In a `.laplacelib` file an item is private unless
  marked `pub`, and listing it in `exports` is an error. Plain `.stan`
  package files still use `exports`, unchanged. `laplace init` no longer
  writes an `exports` key for a package whose sources are all `.laplacelib`.

### Added

- `laplace --version` prints the version and short commit
  (`laplace 0.2.0 (e8b84c9)`). `laplace version --verbose` adds the build
  date, target triple, and the cache and registry directories in use.
- The `laplace = "<semver range>"` manifest key (see above).
- `laplace init --update` syncs an existing `laplace.toml` with the sources.
  It adds newly documented `.stan` functions to `exports`, fills in a missing
  `name` or `version`, and keeps comments, order and unknown keys. It warns
  about `exports` entries whose function is gone, and removes them only with
  `--prune`. Running it twice changes nothing.
- Path dependencies: `laplace add <pkg> --path <dir> [--subdir <sub>]`, or
  `pkg = { path = "../my-lib" }`. A path package is re-read on every install,
  build and doc, so library edits show up without a version bump or tag. It
  is cached apart from versioned packages, and the lock records
  `source = "path+<dir relative to the project>"`.
- `laplace install --locked`: the CI form. It refuses a lock containing path
  dependencies. Plain `install` warns about them.
- `laplace release <version|patch|minor|major> [--dry-run]` tags a package
  release. It refuses if the working tree is dirty, the branch is behind its
  remote, or the tag already exists. It also checks that the package is
  releasable: it loads, every `// @laplace` block documents something, and a
  `.laplacelib` package has a `pub` item. It matches the repository's `v`/no
  `v` tag style, then commits, tags and pushes.
- `laplace self-update [--check] [--version X] [--yes]` updates the compiler
  from GitHub Releases, verifying the sha256 and replacing the binary
  atomically. A package-manager install is left alone (it prints the right
  command), and a `cargo install` gets the matching `cargo install` command.
  `--check` exits 10 when an update exists.
- Release automation: prebuilt binaries for Linux (x86_64, aarch64), macOS
  (x86_64, arm64) and Windows (x86_64) with sha256 checksums, a shell
  installer, and AUR `laplace-bin` / `laplace-git` PKGBUILDs.

### Changed

- A cached package whose source changed without a version bump is now
  detected by checksum and replaced, and its `docs.json` regenerated:
  `refreshed <name>@<version> (source changed)`. Previously the stale copy
  could be used silently.
- `laplace install` against a source that no longer matches the lock's
  checksum explains that the package changed without a version bump (or its
  tag moved), installs neither copy, and points at `laplace update <name>`.
- `laplace add --git <url> --tag X` with a tag the repository lacks lists the
  tags it has and suggests the newest, instead of git's raw error.
- A tag naming a different version than its `laplace.toml` (tag `0.1.2`,
  manifest `0.1.1`) is a warning.
- "private to package" errors from `laplace doc` and `laplace build` name the
  installed version (and, for `doc`, the locked source), since an old tag or a
  stale cache is the usual cause.
- `laplace init` on a `.laplacelib`-only package prints the pub/private
  summary, split into documented and undocumented public items, instead of a
  misleading "no doc comments" warning. On an existing `laplace.toml` it now
  suggests `laplace init --update`.

## [0.1.0]

The first version: everything up to and including patch 1. Never tagged or
published as a release; 0.2.0 is the first.

### Added

- `.laplace` model files compiled to plain, readable, deterministic `.stan`.
- A package manager: `laplace.toml` (ranges) + `laplace.lock` (exact pins and
  checksums), `add`, `update`, `install` (reads only the lock), a local
  filesystem registry, and git sources (`--git` with `--tag`/`--rev`,
  `--subdir`).
- Namespaced imports (`library { import pkg }`, `pkg::func()`), implemented
  as name mangling (`pkg__func`).
- `laplace doc pkg::func`, from `// @laplace` doc comments, with HTML output.
- `--split-functions`: one `#include`d `.stanfunctions` file per package.
- `.laplacelib` library sources: libraries that import libraries, with a
  dependency DAG in the lock, Cargo-style single-version unification, and
  private imports.
- Patch 1: `pub` visibility, provenance comments, functions as arguments
  (monomorphization, `@wait`), block-spanning templates (`@template`/`@use`)
  and statement macros (`@macro`/`@expand`).
- `laplace build --validate` (type-check with `stanc`) and `laplace init`.

[0.2.0]: https://github.com/mlatinov/laplace/releases/tag/v0.2.0
[0.1.0]: https://github.com/mlatinov/laplace/tree/d603268
