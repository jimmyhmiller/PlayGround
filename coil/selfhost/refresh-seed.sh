#!/usr/bin/env bash
# Rebuild + verify the self-host compiler, then UPDATE the committed seed.
#
# Run this whenever you change selfhost/src in a way that touches the language the
# COMPILER ITSELF is written in (new syntax/semantics the current seed wouldn't parse),
# so the seed can always compile the next revision. Keeping the seed in step with source
# is the one discipline that keeps the Rust-free bootstrap working forever.
#
# It refuses to update the seed unless rebootstrap.sh fully verifies (fixpoint + gates),
# so you can never commit a broken seed.
#
# Usage: selfhost/refresh-seed.sh
set -uo pipefail
cd "$(dirname "$0")/.."                 # repo root

echo "=== verifying a fresh compiler before touching the seed ==="
./selfhost/rebootstrap.sh /tmp/coil-newseed || { echo "VERIFY FAILED — seed NOT updated"; exit 1; }

mkdir -p selfhost/seed
cp /tmp/coil-newseed selfhost/seed/coil-seed
chmod +x selfhost/seed/coil-seed
{
  echo "commit: $(git rev-parse HEAD)"
  echo "built:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source: selfhost/src/main.coil"
  echo "proof:  arm64 fixpoint (stage2.o==stage3.o) + gate-full + arm64 gate-run"
} > selfhost/seed/SEED_VERSION

echo
echo "seed updated -> selfhost/seed/coil-seed  ($(du -h selfhost/seed/coil-seed | cut -f1))"
echo "review and commit:"
echo "  git add selfhost/seed/coil-seed selfhost/seed/SEED_VERSION && git commit -m 'refresh self-host seed'"
