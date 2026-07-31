#!/usr/bin/env bash
# `(target-os)` must answer for the platform being COMPILED FOR, not the one
# running the compiler — and a `const` that selects on it must fold to the
# TARGET's literal.
#
# This is the property that cannot be checked by running a program: a
# cross-compiled binary does not execute on the build host, so the only place
# the answer is visible is the emitted IR. The gate therefore compares
# `emit-ir` output across --target values rather than program output.
#
# The bug it exists for: src/stdlib/fs.coil's open() flags are plain integers that
# differ per OS (512 is O_CREAT on darwin and O_TRUNC on Linux). They used to be
# hardcoded to the darwin values, so every create-or-truncate open failed on
# Linux. Selecting on a RUNTIME probe would fix that but cost a branch per use
# and still be wrong for a freestanding target; selecting on `(target-os)` keeps
# them true compile-time constants AND correct under cross-compilation. If
# `(target-os)` ever silently falls back to the host, this gate catches it.
#
# Usage: scripts/compiler/oracle/gate-target-os.sh <coil-binary>
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN=${1:?usage: gate-target-os.sh <coil-binary>}
[ -x "$BIN" ] || { echo "GATE FAIL: binary not executable: $BIN"; exit 2; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
fail=0

# ---- a const selecting on (target-os) folds to the TARGET's literal ----
cat > "$WORK/flags.coil" <<'EOF'
(module flagtest)
(import "io.coil" :use *)
(defn pick [(l Code) (d Code)] (-> Code)
  (if (code-eq (target-os) `linux) l d))
(defn gen [] (-> Code)
  `(do (const O_CREAT ~(pick `64 `512))
       (const O_TRUNC ~(pick `512 `1024))))
(meta (gen))
(defn main [] (-> i64)
  (print-int (stdout) O_CREAT) (print-int (stdout) O_TRUNC) 0)
EOF

folded() {  # folded <triple-or-empty> -> the two folded literals, space separated
  if [ -z "$1" ]; then
    "$BIN" emit-ir "$WORK/flags.coil" 2>/dev/null
  else
    "$BIN" emit-ir "$WORK/flags.coil" --target "$1" 2>/dev/null
  fi | grep -o 'print-int(ptr %call[0-9]*, i64 [0-9]*' | grep -o '[0-9]*$' | tr '\n' ' '
}

check() {  # check <label> <got> <want>
  if [ "$(printf '%s' "$2" | xargs)" = "$3" ]; then echo "  ok   — $1"
  else echo "  FAIL — $1: got '$(printf '%s' "$2" | xargs)', want '$3'"; fail=$((fail+1)); fi
}

check "no --target folds to the host's flags"      "$(folded '')"                            "64 512"
check "--target linux folds to the Linux flags"    "$(folded x86_64-pc-linux-gnu)"           "64 512"
check "--target darwin folds to the darwin flags"  "$(folded aarch64-apple-darwin24.0.0)"    "512 1024"
check "--target macos is darwin too"               "$(folded x86_64-apple-macosx14.0.0)"     "512 1024"

# ---- the folded program must carry NO runtime OS probe --------------------
# The whole point of folding at comptime is that the branch disappears. If a
# /proc/self/maps probe survives into the IR, the constant is not a constant.
if "$BIN" emit-ir "$WORK/flags.coil" 2>/dev/null | grep -q "proc/self/maps"; then
  echo "  FAIL — a runtime OS probe survived into the emitted IR"; fail=$((fail+1))
else
  echo "  ok   — no runtime OS probe in the emitted IR"
fi

# ---- and the same must hold for the real user of this: src/stdlib/fs.coil --------
# Linux O_CREAT|O_TRUNC = 64|512; darwin = 512|1024. Reading the operands of the
# BitOr in write-file pins the flags the library will actually pass to open(2).
# Only write-file's own O_CREAT|O_TRUNC — fs.coil has other BitOr calls, and a
# blanket grep picks those up too.
fsflags() {
  if [ -z "$1" ]; then "$BIN" emit-ir src/stdlib/fs.coil 2>/dev/null
  else "$BIN" emit-ir src/stdlib/fs.coil --target "$1" 2>/dev/null; fi \
    | awk '/define .*@fs.write-file/,/^}/' \
    | grep -o 'BitOr\$i64\$|"(i64 [0-9]*, i64 [0-9]*' | head -1 \
    | sed 's/.*(i64 \([0-9]*\), i64 \([0-9]*\)/\1 \2/'
}
check "src/stdlib/fs.coil flags follow the host"    "$(fsflags '')"                         "64 512"
check "src/stdlib/fs.coil flags follow --target"    "$(fsflags aarch64-apple-darwin24.0.0)" "512 1024"

echo
[ "$fail" = 0 ] && { echo "gate-target-os: PASS"; exit 0; } || { echo "gate-target-os: $fail check(s) FAILED"; exit 1; }
