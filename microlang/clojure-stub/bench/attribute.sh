#!/usr/bin/env bash
# Sampling attribution for one bench file: run it on microclj under macOS
# `sample`, bucket the frames by symbol, and print a cost breakdown. The cheap
# counters (# STATS lines, MICROLANG_STATS=1) usually name the bottleneck
# category already; this answers "which shims exactly" when they don't.
#
#   clojure-stub/bench/attribute.sh clojure-stub/bench/corpus/json-pipeline/bench.clj \
#       [vendor-dir ...]
#
# Buckets (frames counted at the TOP of each sampled stack):
#   jit-code     anonymous region — JIT-compiled guest code itself
#   dispatch     shim_dispatch / resolve_or_default / def_method — protocol glue
#   call-glue    shim_call / tail_call / fast_invoke / make_closure
#   collections  pv/hamt/tam/thm/tv/arr/cons/amap shims — persistent-structure ops
#   alloc        Heap::alloc / alloc_raw
#   gc           collect / evacuate / scan
#   interp       CekMachine / TreeWalk eval
#   runtime      the rest of Runtime:: (decode/equal/prim dispatch…)
#   UNCLASSIFIED anything else — if this exceeds ~10%, distrust the table and
#                extend the buckets (costs hiding in a miscategorized frame are
#                exactly how attribution lies).
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$PWD"
MICROCLJ="$ROOT/target/release/microclj"
BENCH="$ROOT/clojure-stub/bench"

FILE="${1:?usage: attribute.sh <bench.clj> [vendor-dir ...]}"; shift || true
MP="$BENCH"
for v in "$@"; do MP="$MP:$ROOT/$v"; done

OUT="$(mktemp -d)"
echo "==> running $FILE under sample (30s cap)…" >&2
MICROLANG_PATH="$MP" "$MICROCLJ" --jit "$FILE" > "$OUT/run.tsv" &
PID=$!
sleep 0.7   # skip prelude load; sample the workload steady state
sample "$PID" 30 -file "$OUT/sample.txt" >/dev/null 2>&1 || true
wait "$PID" || true

awk '
  # leaf-weighted: count each sample line by its indent depth leaf weight is
  # hard from `sample` output, so bucket EVERY frame occurrence weighted by
  # its sample count prefix — a coarse but honest inclusive-cost view.
  match($0, /^[ +!:|]*[0-9]+ /) {
    n = $0; sub(/^[ +!:|]*/, "", n); split(n, parts, " "); cnt = parts[1];
    line = $0
    # The program spine (main → eval loop) has inclusive count == everything;
    # counting it would just dilute the table.
    if (line ~ /::main|::drive|run_src|eval1|Session.*eval|lang_start|__rust_begin|^ *[0-9]+ start|Thread_|DispatchQueue/) next
    if      (line ~ /shim_dispatch|resolve_or_default|observed_dispatch|def_method/) b["dispatch"] += cnt
    else if (line ~ /shim_call|shim_tail_call|shim_finish_tail|finish_tail_fast|run_trampoline|fast_invoke|make_closure|invoke/) b["call-glue"] += cnt
    else if (line ~ /shim_pv_|shim_hamt_|shim_tam_|shim_thm_|shim_tv_|shim_cons|shim_arr|shim_first|shim_rest|amap|pv_conj|pv_nth|hamt_/) b["collections"] += cnt
    else if (line ~ /alloc/) b["alloc"] += cnt
    else if (line ~ /collect|evacuate|scan_object|minor|major/) b["gc"] += cnt
    else if (line ~ /CekMachine|TreeWalk|eval_ir|::cek::/) b["interp"] += cnt
    else if (line ~ /Runtime|runtime::|decode|view_gc|equal|microlang|clojure_stub/) b["runtime"] += cnt
    else if (line ~ /\?\?\?/) b["jit-code"] += cnt
    else b["UNCLASSIFIED"] += cnt
    total += cnt
  }
  END {
    if (total == 0) { print "no samples captured"; exit 1 }
    for (k in b) printf "%-14s %8d  %5.1f%%\n", k, b[k], 100*b[k]/total | "sort -k2 -rn"
  }
' "$OUT/sample.txt"
echo "raw sample in $OUT/sample.txt" >&2
