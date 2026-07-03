#!/usr/bin/env bash
# Rebuild + verify the self-host compiler(s), then UPDATE the committed seed(s).
#
# Refreshes BOTH seeds:
#   * selfhost/seed/coil-seed         — full compiler (LLVM + arm64), from main.coil
#   * selfhost/seed/coil-seed-nollvm  — LLVM-free compiler (arm64 only), from main_a64.coil
#
# Run this whenever you change selfhost/src in a way that touches the language the
# COMPILER ITSELF is written in (new syntax/semantics the current seed wouldn't parse),
# so the seeds can always compile the next revision. Keeping the seeds in step with
# source is the one discipline that keeps the Rust-free bootstrap working forever.
#
# Each seed is only updated if its rebootstrap fully verifies, so you can never commit
# a broken seed. Pass a seed name to refresh just one: `refresh-seed.sh nollvm` / `full`.
#
# Usage: selfhost/refresh-seed.sh [full|nollvm|both]     (default: both)
set -uo pipefail
cd "$(dirname "$0")/.."                 # repo root
WHICH="${1:-both}"
mkdir -p selfhost/seed
updated=()

if [ "$WHICH" = both ] || [ "$WHICH" = full ]; then
  echo "=== [full] verifying before touching the seed ==="
  ./selfhost/rebootstrap.sh /tmp/coil-newseed || { echo "[full] VERIFY FAILED — seed NOT updated"; exit 1; }
  cp /tmp/coil-newseed selfhost/seed/coil-seed
  chmod +x selfhost/seed/coil-seed
  {
    echo "commit: $(git rev-parse HEAD)"
    echo "built:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source: selfhost/src/main.coil (LLVM + arm64)"
    echo "proof:  arm64 fixpoint (stage2.o==stage3.o) + gate-full + arm64 gate-run"
  } > selfhost/seed/SEED_VERSION
  updated+=("selfhost/seed/coil-seed" "selfhost/seed/SEED_VERSION")
fi

if [ "$WHICH" = both ] || [ "$WHICH" = nollvm ]; then
  echo "=== [nollvm] verifying before touching the seed ==="
  ./selfhost/rebootstrap-nollvm.sh /tmp/coil-newseed-nollvm || { echo "[nollvm] VERIFY FAILED — seed NOT updated"; exit 1; }
  cp /tmp/coil-newseed-nollvm selfhost/seed/coil-seed-nollvm
  chmod +x selfhost/seed/coil-seed-nollvm
  {
    echo "commit: $(git rev-parse HEAD)"
    echo "built:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source: selfhost/src/main_a64.coil (LLVM-free)"
    echo "proof:  no-libLLVM + arm64 fixpoint (stage2.o==stage3.o) + arm64 gate-run"
  } > selfhost/seed/SEED_VERSION_NOLLVM
  updated+=("selfhost/seed/coil-seed-nollvm" "selfhost/seed/SEED_VERSION_NOLLVM")
fi

echo
echo "seed(s) updated:"
for f in "${updated[@]}"; do
  case "$f" in *coil-seed*) echo "  $f  ($(du -h "$f" | cut -f1))";; *) echo "  $f";; esac
done
echo "review and commit:"
echo "  git add ${updated[*]} && git commit -m 'refresh self-host seed(s)'"
