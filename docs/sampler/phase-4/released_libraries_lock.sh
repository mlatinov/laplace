#!/usr/bin/env bash
# Phase 4 regression check: every released function library installs into a
# fresh project, and the laplace.lock it writes is byte-identical between
# the compiler before Phase 4 and after.
#
#   released_libraries_lock.sh <laplace binary> <output dir>
#
# Each library gets its own fresh project and its own empty ~/.laplace (HOME
# is pointed at a scratch directory), so nothing cached is reused. After
# `add`, `install` must also succeed from the lock alone.
set -euo pipefail
bin=$(realpath "$1"); out=$(realpath -m "$2")
rm -rf "$out"; mkdir -p "$out"
libs=(
  "gaussian_process laplace-gaussian_processes- 0.1.2"
  "survival laplace-survival 0.1.2"
  "ts laplace-ts 0.1.2"
  "lts laplace-lts 0.2.2"
  "splines laplace-splines 0.1.1"
  "cluster laplace-cluster 0.1.1"
  "transformations laplace-transform 0.1.2"
)
for entry in "${libs[@]}"; do
  read -r name repo tag <<<"$entry"
  work=$(mktemp -d); mkdir -p "$work/home" "$work/project"
  cp ~/.gitconfig "$work/home/" 2>/dev/null || true
  ( cd "$work/project"
    HOME="$work/home" "$bin" add "$name" --git "https://github.com/mlatinov/$repo" \
      --tag "$tag" --subdir laplace >/dev/null
    rm -rf "$work/home/.laplace/packages"
    HOME="$work/home" "$bin" install >/dev/null
    cp laplace.lock "$out/$name.lock"
    cp laplace.toml "$out/$name.toml" )
  echo "$name $tag: $(sha256sum "$out/$name.lock" | cut -c1-16)"
  rm -rf "$work"
done
