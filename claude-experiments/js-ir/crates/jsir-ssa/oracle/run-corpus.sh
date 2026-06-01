#!/usr/bin/env bash
# Run the jsir-ssa React-Compiler parity corpus against the IN-REPO oracle.
# This replaces the old /tmp/react-rust dependency: both the oracle CLI and the
# React fixtures now live under this directory and are version-controlled here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"   # crates/jsir-ssa/oracle -> repo root

export REACT_CC="$HERE/react-cc.js"
export REACT_FIXTURES="$HERE/fixtures"

cd "$REPO"
exec cargo run --release -q -p jsir-ssa --example corpus -- "$@"
