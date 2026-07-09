#!/usr/bin/env bash
# Gate: the coil i8080 port must reproduce the emulator101 C reference byte-for-byte
# on the standard CP/M diagnostic ROMs. Builds both, diffs their BDOS output.
#   apps/invaders/validate-cpu.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../..
ROMS=$ROOT/roms
COIL=${COIL:-$ROOT/target/release/coil}

echo "==> build C golden harness"
cc -O2 ref/cputest.c ref/8080emu_plain.c -o /tmp/inv_cputest_c

echo "==> build coil port"
"$COIL" build cputest.coil -o /tmp/inv_cputest_coil >/dev/null

fail=0
for rom in 8080PRE CPUTEST 8080EXM; do
    /tmp/inv_cputest_c    "$ROMS/$rom.COM" > "/tmp/inv_${rom}_c.txt"
    /tmp/inv_cputest_coil "$ROMS/$rom.COM" > "/tmp/inv_${rom}_coil.txt"
    if diff -q "/tmp/inv_${rom}_c.txt" "/tmp/inv_${rom}_coil.txt" >/dev/null; then
        echo "  $rom: coil == C  ✓"
    else
        echo "  $rom: MISMATCH ✗"; diff "/tmp/inv_${rom}_c.txt" "/tmp/inv_${rom}_coil.txt" | head; fail=1
    fi
done
[ $fail -eq 0 ] && echo "ALL CPU ORACLES MATCH" || { echo "CPU VALIDATION FAILED"; exit 1; }
