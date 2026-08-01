#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

snapshot=docs/public-api.snapshot
temporary=$(mktemp)
trap 'rm -f "$temporary"' EXIT

if ! cargo public-api --version >/dev/null 2>&1; then
  echo "cargo-public-api is required: cargo install cargo-public-api --locked" >&2
  exit 1
fi

: > "$temporary"
for package in \
  diffpack-core \
  diffpack-default-loader \
  diffpack-web \
  diffpack-vite-compat \
  diffpack-next \
  diffpack-tanstack
do
  echo "# $package" >> "$temporary"
  cargo public-api -p "$package" --simplified --omit blanket-impls --omit auto-trait-impls \
    >> "$temporary"
done

if [[ ${1:-} == --update ]]; then
  cp "$temporary" "$snapshot"
  echo "updated $snapshot"
  exit 0
fi

if ! diff -u "$snapshot" "$temporary"; then
  echo "public API changed; review it and run scripts/check-public-api.sh --update" >&2
  exit 1
fi

echo "public API snapshot matches"
