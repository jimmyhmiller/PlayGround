#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

mode=${1:-slice}
crate=${2:-}
extracted_crates=(
  diffpack-core
  diffpack-default-loader
  diffpack-web
  diffpack-vite-compat
  diffpack-next
  diffpack-tanstack
)

usage() {
  echo "usage: $0 slice <crate> | phase | final" >&2
  exit 2
}

check_formatting() {
  local package
  for package in "${extracted_crates[@]}"; do
    cargo fmt -p "$package" -- --check
  done
}

check_public_api() {
  ./scripts/check-public-api.sh
}

check_core_neutrality() {
  local forbidden='diffpack_(next|tanstack|vite_compat|web|default_loader)::|JsxExtensions::NextJs|Target::ReactServer|LoaderKind|SourceLanguage::Css|tsr-split|TSS_SERVER_FN_BASE|node:module|node:url|node:path'
  if rg -n "$forbidden" crates/diffpack-core/src crates/diffpack-core/Cargo.toml; then
    echo "framework, loader, or host policy leaked into diffpack-core" >&2
    exit 1
  fi
}

case "$mode" in
  slice)
    [[ -n "$crate" ]] || usage
    case " ${extracted_crates[*]} " in
      *" $crate "*) ;;
      *) echo "unknown extracted crate: $crate" >&2; usage ;;
    esac
    ./scripts/check-crate-boundaries.sh
    check_core_neutrality
    cargo fmt -p "$crate" -- --check
    cargo check --workspace
    cargo test -p "$crate"
    git diff --check
    ;;
  phase)
    ./scripts/check-crate-boundaries.sh
    check_core_neutrality
    check_formatting
    check_public_api
    cargo check --workspace
    for crate in "${extracted_crates[@]}"; do
      cargo test -p "$crate"
    done
    git diff --check
    ;;
  final)
    "$0" phase
    cargo test --workspace --lib
    cargo test --workspace --tests
    check_public_api
    check_core_neutrality
    ;;
  *) usage ;;
esac
