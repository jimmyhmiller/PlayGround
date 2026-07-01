#!/usr/bin/env bash
# DWARF/lldb gate for the arm64 backend — mirrors tests/debuginfo.rs:
#  1. dwarfdump: subprogram DIEs exist at real source lines
#  2. lldb name breakpoint lands POST-prologue at a source line, params correct
#  3. struct rendering through a typed pointer
#  4. slice frame variable shows {data, len}
#  5. file:line breakpoint resolves and stops
# usage: gate-lldb.sh <coil-self-bin>
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN="${1:?usage: gate-lldb.sh <coil-self-bin>}"
SRC=/tmp/coil-lldb-gate.coil
EXE=/tmp/coil-lldb-gate
cat > "$SRC" <<'EOF'
(module app)
(import "lib/alloc.coil" :use *)
(import "lib/io.coil" :use *)
(import "lib/slice.coil" :use *)
(import "lib/control.coil" :use *)
(defstruct Point [(x i64) (y i64)])
(defn dist2 [(p (ptr Point)) (scale i64)] (-> i64)
  (let [dx (load (field p x))
        dy (load (field p y))
        (mut acc) 0]
    (store! acc (iadd (imul dx dx) (imul dy dy)))
    (imul (load acc) scale)))
(defn sum-slice [(s (slice u8))] (-> i64)
  (let [n (slice-len s) (mut i) 0 (mut tot) 0]
    (loop (if (icmp-ge (load i) n) (break)
      (do (store! tot (iadd (load tot) (cast i64 (slice-get s (load i)))))
          (store! i (iadd (load i) 1)))))
    (load tot)))
(defn main [] (-> i64)
  (let [p (alloc-stack Point)]
    (store! (field p x) 3)
    (store! (field p y) 4)
    (let [d (dist2 p 2)
          sl "AB"]
      (isub (iadd d (sum-slice sl)) 181))))
EOF
fail=0
check() { # name, haystack, needle
  if echo "$2" | grep -F "$3" >/dev/null; then echo "  ok  $1"; else echo "  FAIL $1 (missing: $3)"; fail=1; fi
}
"$BIN" build "$SRC" -o "$EXE" --backend arm64 >/dev/null 2>&1 || { echo "FAIL: build"; exit 1; }
"$EXE"; [ $? -eq 0 ] || { echo "FAIL: program exit (want 0)"; exit 1; }

DD=$(dwarfdump --debug-info "$EXE.o" 2>/dev/null)
check "subprogram DIEs" "$DD" "DW_TAG_subprogram"
check "dist2 subprogram" "$DD" 'DW_AT_name	("app.dist2")'
check "frame base x29" "$DD" "DW_OP_reg29"
check "fbreg locations" "$DD" "DW_OP_fbreg"

OUT=$(lldb -b -o 'br set -n app.dist2' -o run -o 'frame variable' -o 'p *p' "$EXE" 2>&1)
check "post-prologue stop at source line" "$OUT" "at coil-lldb-gate.coil:"
check "param p typed" "$OUT" "(app.Point *) p = 0x"
check "param scale value" "$OUT" "(long) scale = 2"
check "mut local reads through deref" "$OUT" "(long) acc = 0"
check "struct render" "$OUT" "(x = 3, y = 4)"

OUT2=$(lldb -b -o 'br set -n app.sum-slice' -o run -o 'frame variable s' "$EXE" 2>&1)
check "slice render" "$OUT2" '(data = "AB", len = 2)'

OUT3=$(lldb -b -o 'br set -f coil-lldb-gate.coil -l 11' -o run -o 'frame variable dx dy' "$EXE" 2>&1)
check "file:line breakpoint stops" "$OUT3" "stop reason = breakpoint"
check "locals assigned before line 11" "$OUT3" "(long) dx = 3"

if [ "$fail" -eq 0 ]; then echo "GATE PASS — lldb debugging works end-to-end"; else echo "lldb gate FAILED"; fi
exit $fail
