#!/bin/sh
# Install the laplace compiler from its GitHub Releases.
#
#   curl -fsSL https://github.com/mlatinov/laplace/releases/latest/download/install.sh | sh
#
# Environment:
#   LAPLACE_VERSION      release to install, e.g. 0.2.0 (default: latest)
#   LAPLACE_INSTALL_DIR  where to put the binary (default: ~/.local/bin)
#   LAPLACE_DOWNLOAD_BASE  override the download location (testing/mirrors);
#                        must serve <base>/laplace-<target>.tar.gz[.sha256]
#
# Downloads laplace-<target>.tar.gz and its .sha256, verifies the checksum,
# and installs the binary. Nothing outside LAPLACE_INSTALL_DIR is touched,
# and a failed download or checksum leaves any existing laplace in place.
set -eu

REPO="https://github.com/mlatinov/laplace"

say() { printf 'laplace-install: %s\n' "$1"; }
die() { printf 'laplace-install: error: %s\n' "$1" >&2; exit 1; }

detect_target() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$arch" in
        x86_64 | amd64) arch=x86_64 ;;
        aarch64 | arm64) arch=aarch64 ;;
        *) die "no prebuilt laplace for CPU '$arch' -- build from source: cargo install --locked --git $REPO" ;;
    esac
    case "$os" in
        Linux) echo "$arch-unknown-linux-gnu" ;;
        Darwin) echo "$arch-apple-darwin" ;;
        MINGW* | MSYS* | CYGWIN*) echo "x86_64-pc-windows-msvc" ;;
        *) die "no prebuilt laplace for '$os' -- build from source: cargo install --locked --git $REPO" ;;
    esac
}

fetch() {
    # fetch <url> <output file>
    if command -v curl >/dev/null 2>&1; then
        curl --fail --silent --show-error --location --output "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet --output-document="$2" "$1"
    else
        die "need curl or wget to download laplace"
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        die "need sha256sum or shasum to verify the download"
    fi
}

main() {
    target=$(detect_target)
    archive="laplace-$target.tar.gz"
    if [ -n "${LAPLACE_DOWNLOAD_BASE:-}" ]; then
        base="$LAPLACE_DOWNLOAD_BASE"
    elif [ -n "${LAPLACE_VERSION:-}" ]; then
        base="$REPO/releases/download/v${LAPLACE_VERSION#v}"
    else
        base="$REPO/releases/latest/download"
    fi
    install_dir="${LAPLACE_INSTALL_DIR:-$HOME/.local/bin}"

    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT INT TERM

    say "downloading $archive"
    fetch "$base/$archive" "$tmp/$archive" || die "download failed: $base/$archive"
    fetch "$base/$archive.sha256" "$tmp/$archive.sha256" || die "download failed: $base/$archive.sha256"

    expected=$(cut -d ' ' -f 1 < "$tmp/$archive.sha256")
    actual=$(sha256_of "$tmp/$archive")
    [ "$expected" = "$actual" ] || die "checksum mismatch for $archive (expected $expected, got $actual)"

    tar -xzf "$tmp/$archive" -C "$tmp"
    binary="laplace"
    [ -f "$tmp/laplace-$target/laplace.exe" ] && binary="laplace.exe"
    [ -f "$tmp/laplace-$target/$binary" ] || die "the archive holds no $binary"

    mkdir -p "$install_dir"
    # Stage beside the destination, then rename: atomic, and an existing
    # laplace is never left half-written.
    cp "$tmp/laplace-$target/$binary" "$install_dir/.$binary.new"
    chmod 755 "$install_dir/.$binary.new"
    mv -f "$install_dir/.$binary.new" "$install_dir/$binary"

    say "installed $("$install_dir/$binary" --version) to $install_dir/$binary"
    case ":$PATH:" in
        *":$install_dir:"*) ;;
        *) say "note: $install_dir is not on your PATH -- add it, e.g. export PATH=\"$install_dir:\$PATH\"" ;;
    esac
    say "update later with: laplace self-update"
}

main "$@"
