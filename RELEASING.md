# Releasing the laplace compiler

This is about shipping the **`laplace` binary** (a *compiler release*). It has
nothing to do with `laplace release`, which tags a *package release* of a Stan
library.

A compiler release is driven by a `vX.Y.Z` tag: pushing one runs
`.github/workflows/release.yml`, which builds every platform, publishes a
GitHub Release, and attaches the assets `laplace self-update`, the shell
installer and the AUR `laplace-bin` package all read. The tag must equal the
version in `Cargo.toml`, and `CHANGELOG.md` must have a section for it. The
workflow checks both before it builds anything.

## 1. Prepare

1. Start from an up-to-date `main` with a clean tree:

   ```sh
   git switch main && git pull --ff-only
   git status            # nothing to commit
   ```

2. Bump the version in `Cargo.toml` (`version = "X.Y.Z"`), then run
   `cargo build` so `Cargo.lock` picks it up (the release builds with
   `--locked`, which fails on a stale lockfile).

3. Update `CHANGELOG.md`: rename `## [X.Y.Z] - Unreleased` to
   `## [X.Y.Z] - YYYY-MM-DD` (or add the section if there is none), make sure
   it says everything a user upgrading needs to know, and update the link at
   the bottom. The section body becomes the GitHub Release notes, and
   `laplace self-update` shows it to users as "what changed".

4. Run the same gate CI runs:

   ```sh
   cargo fmt --check
   cargo clippy --all-targets --locked -- -D warnings
   cargo test --locked
   ```

## 2. Commit, tag, push

```sh
git commit -am "laplace X.Y.Z"
git tag -a vX.Y.Z -m "laplace X.Y.Z"
git push origin main
git push origin vX.Y.Z
```

The tag push is what starts the release workflow.

## 3. Verify the release

When the workflow is green, open the release on GitHub and check:

- [ ] Five archives are attached, each with a `.sha256`:
      `laplace-x86_64-unknown-linux-gnu`, `laplace-aarch64-unknown-linux-gnu`,
      `laplace-x86_64-apple-darwin`, `laplace-aarch64-apple-darwin`,
      `laplace-x86_64-pc-windows-msvc` (all `.tar.gz`).
- [ ] `SHA256SUMS` and `install.sh` are attached.
- [ ] The notes are the CHANGELOG section.

Then try it from the outside:

```sh
# Fresh install through the installer
curl -fsSL https://github.com/mlatinov/laplace/releases/latest/download/install.sh \
  | LAPLACE_INSTALL_DIR=/tmp/laplace-check sh
/tmp/laplace-check/laplace --version          # laplace X.Y.Z (<commit>)

# An older standalone binary finds and installs the update
laplace self-update --check                    # exits 10, shows the notes
laplace self-update
```

If something is wrong, delete the GitHub Release *and* the tag, fix it, and
release a new patch version. Do not move a published tag: `self-update`, the
AUR package and anyone's `--tag` pin all trust that it never changes.

## 4. AUR

Both packages live in `packaging/aur/`. The AUR names `laplace-bin` and
`laplace-git` were free when checked (2026-10-09).

First-time setup per package: `git clone ssh://aur@aur.archlinux.org/<pkgname>.git`
(this creates the AUR repo on first push), and set the `# Maintainer:` line.

### `laplace-bin` (every release)

```sh
cd packaging/aur/laplace-bin
# set pkgver=X.Y.Z, pkgrel=1
updpkgsums                         # replaces the 'SKIP' placeholders with the
                                   # real sha256s from the release (check they
                                   # match the published .sha256 files)
makepkg -si                        # build and install locally as a check
makepkg --printsrcinfo > .SRCINFO
# copy PKGBUILD and .SRCINFO into the AUR clone, then:
git commit -am "X.Y.Z" && git push
```

Never publish with `sha256sums_*=('SKIP')`. That placeholder only exists
because the checksums cannot be known until the release is built.

### `laplace-git` (once, then only when the build changes)

It builds `main`, so a release needs no change. Before the first upload:

```sh
cd packaging/aur/laplace-git
makepkg -si
makepkg --printsrcinfo > .SRCINFO
```

## Naming

The binary is always called `laplace`, whatever the package is called.
On crates.io `laplace` is taken (an unrelated placeholder crate), so
publishing there would need another *package* name, e.g. `laplace-stan`,
with `[[bin]] name = "laplace"` and `[lib] name = "laplace"` in `Cargo.toml`
so the binary and the `laplace::` library path stay the same. Nothing is
published to crates.io today; installs go through GitHub Releases, the AUR,
or `cargo install --locked --git https://github.com/mlatinov/laplace`.
