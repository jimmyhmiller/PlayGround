#!/usr/bin/env bash
# AddressSanitizer gate for the STW heap-read path (alloc-site profiler dump +
# heap dumps). A local regression gate (this project uses local gates, no CI).
#
# Why: the dump reads per-thread NON-ATOMIC alloc-site counters. If it ever read
# a live (unjoined) worker's counter Vec while that worker's record_alloc did a
# Vec::resize (realloc → frees the old buffer), that is a heap-use-after-free —
# reproduced under ASan during review BEFORE the STW fix, reachable from a plain
# spawn-don't-join program. The fix reads under a pause_world() STW pause so the
# worker is parked (no concurrent realloc) during the read.
#
# These concurrent tests grow a worker's counter Vec while the main thread dumps;
# WITHOUT the fix that surfaces as an ASan UAF, WITH it ASan stays clean. ASan is
# dispositive here (TSan SIGSEGVs on macOS-aarch64; it can't model the
# park-protocol happens-before anyway).
#
#   ./scripts/asan_dump_race.sh
set -uo pipefail
cd "$(dirname "$0")/.."

HOST="$(rustc -vV | sed -n 's/host: //p')"
echo "ASan dump-race gate on ${HOST} (nightly + -Zsanitizer=address)"

ASAN_OPTIONS=detect_leaks=0 RUSTFLAGS="-Zsanitizer=address" \
  cargo +nightly test -p gcrust-rt --target "${HOST}" \
  alloc_site_profile_dump_concurrent -- --nocapture
status=$?

if [ "${status}" -eq 0 ]; then
  echo "PASS: STW dump-read path is AddressSanitizer-clean"
else
  echo "FAIL: AddressSanitizer flagged the dump read path (status ${status})"
fi
exit "${status}"
