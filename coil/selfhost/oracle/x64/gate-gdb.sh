#!/usr/bin/env bash
# Debug-info gate for the x86-64 backend: the DWARF the backend emits must
# actually drive a debugger. Every check here failed at some point during the
# port, so none of them is ceremony:
#   * a breakpoint set BY FUNCTION NAME must resolve and land on the right
#     source line (needs subprograms + the line table + prologue_end);
#   * `info args` / `info locals` must print real VALUES — this is the one that
#     stayed broken longest, because DW_AT_frame_base was still naming arm64's
#     x29 and the slot offsets were computed against the wrong base;
#   * a struct must render through a pointer (`print *p`), which needs the DI
#     type graph, not just locations;
#   * stepping must advance the reported line.
#
# Usage: selfhost/oracle/x64/gate-gdb.sh <coil-binary>
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN=${1:?usage: gate-gdb.sh <coil-binary>}
[ -x "$BIN" ] || { echo "GATE FAIL: binary not executable: $BIN"; exit 2; }
command -v gdb >/dev/null 2>&1 || { echo "GATE SKIP: no gdb on PATH"; exit 0; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/dbg.coil"; EXE="$WORK/dbg"
cat > "$SRC" <<'EOF'
(module dbg)
(import "io.coil" :use *)
(defstruct Pt [(x i64) (y i64)])
(defn addpt [(p (ptr Pt)) (d i64)] (-> i64)
  (let [sum (iadd (load (field p x)) (load (field p y)))
        scaled (imul sum d)]
    scaled))
(defn main [] (-> i64)
  (let [pt (alloc-stack Pt)]
    (store! (field pt x) 3)
    (store! (field pt y) 4)
    (let [r (addpt pt 10)]
      (print-int (stdout) r) (print-str (stdout) "\n")))
  0)
EOF

"$BIN" build "$SRC" -o "$EXE" --backend x64 >/dev/null 2>&1 \
  || { echo "GATE FAIL: could not build the debug program"; exit 2; }

out=$(gdb -batch -ex "break dbg.addpt" -ex run \
          -ex "info args" -ex next -ex next -ex "info locals" \
          -ex "print *p" -ex bt "$EXE" 2>&1)

fail=0
check() {  # check <description> <regex>
  if printf '%s' "$out" | grep -qE "$2"; then echo "  ok   — $1"
  else echo "  FAIL — $1"; fail=$((fail+1)); fi
}
check "breakpoint resolves by function name"  'Breakpoint 1, dbg\.addpt'
check "breakpoint lands on the defn's body line" 'dbg\.coil:5'
check "parameter p has a real pointer value"  '^p = 0x[0-9a-f]+'
check "parameter d reads 10"                  '^d = 10$'
check "stepping advances the source line"     'scaled \(imul sum d\)'
check "local sum reads 7"                     '^sum = 7$'
check "local scaled reads 70"                 '^scaled = 70$'
check "struct renders through a pointer"      '\{x = 3, y = 4\}'
check "backtrace names the caller"            '#1 .* main \(\)'
# no variable may come back as an unreadable address — the symptom of a wrong
# DW_AT_frame_base, which still lets every other check above pass.
if printf '%s' "$out" | grep -q 'error reading variable'; then
  echo "  FAIL — some variable was unreadable (frame base is wrong)"; fail=$((fail+1))
else
  echo "  ok   — every variable was readable"
fi

echo
[ "$fail" = 0 ] && { echo "x64 gate-gdb: PASS"; exit 0; } || { echo "x64 gate-gdb: $fail check(s) FAILED"; exit 1; }
