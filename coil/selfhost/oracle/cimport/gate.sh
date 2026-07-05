#!/usr/bin/env bash
# Oracle gate for the self-hosted cimport (selfhost/src/cimport.coil).
#
# The Rust reference (`coil cimport`, src/cimport.rs) is the SPEC. This gate
# proves the coil-native importer, compiled by a self-hosted coil, produces
# BYTE-IDENTICAL bindings to the reference across the whole test corpus — the
# same discipline as the other self-host gates (identical, not "looks right").
#
# Usage: selfhost/oracle/cimport/gate.sh [self-host-coil] [rust-coil]
#   defaults: self-host = selfhost/seed/coil-seed, rust = ./target/debug/coil
# Requires clang (skips cleanly if absent, like tests/cimport.rs).
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 2

SELF="${1:-selfhost/seed/coil-seed}"
RUST="${2:-./target/debug/coil}"
LF=(--link-flag -L/opt/homebrew/opt/llvm/lib --link-flag -lLLVM)
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

command -v clang >/dev/null 2>&1 || { echo "SKIP: clang not found"; exit 0; }
[ -x "$SELF" ] || { echo "no self-host coil: $SELF"; exit 2; }
[ -x "$RUST" ] || { echo "no rust coil (cargo build): $RUST"; exit 2; }

echo "=== building selfhost/src/cimport.coil with $SELF ==="
"$SELF" build selfhost/src/cimport.coil -o "$TMP/cimport" "${LF[@]}" || { echo "BUILD FAILED"; exit 1; }

# Corpus mirrors tests/cimport.rs (functions, scalars, structs, unions, typedefs,
# enums, object-like #defines, and the red-team refusals).
emit() { printf '%b' "$2" > "$TMP/$1.h"; }
emit types   'struct Pt { int x; long y; };\nint add(int a, long b);\n'
emit libc    'double sqrt(double x);\nlong labs(long n);\n'
emit redteam 'struct Flags { int a : 3; int b : 5; int c; };\nenum Color { RED, GREEN };\nint paint(enum Color c);\nlong double precise(long double x);\n_Bool is_ready(int x);\n'
emit typedef 'typedef unsigned long size_t;\ntypedef size_t mysize;\ntypedef struct Point Point;\nstruct Point { int x; int y; };\nmysize span(Point *p, size_t n);\n'
emit consts  'enum E { A, B = 10, C };\n#define WIDTH 4\n#define NAME "coil"\n#define DOUBLE(x) ((x)+(x))\n'
emit union   'union U { int i; float f; double d; };\nstruct Bad { union U u; int tag; };\nint ok(int x);\n'
emit refuse  'struct S { int a; int b; };\nunion V { struct S s; long l; };\nint ok(int x);\n'
emit bitfield 'struct Rgb { unsigned b:5; unsigned g:6; unsigned r:5; };\nstruct Mixed { int a:3; int b:5; int c; };\n'
emit floats   '#define HALF 3.5\n#define BIG 3.14159265358979323846\n#define WHOLE 4.0f\n'
# real system headers (parser-at-scale), incl. math.h (float #defines: M_PI etc.).
for sys in ctype.h string.h stdlib.h math.h; do printf '#include <%s>\n' "$sys" > "$TMP/sys_${sys%.h}.h"; done

pass=0; fail=0
check() {
  local name="$1" hdr="$2"
  "$RUST" cimport "$hdr" -o "$TMP/$name.rust" 2>/dev/null
  "$TMP/cimport" "$hdr" > "$TMP/$name.coil" 2>/dev/null
  if diff -q "$TMP/$name.rust" "$TMP/$name.coil" >/dev/null; then
    echo "  PASS  $name"; pass=$((pass+1))
  else
    echo "  FAIL  $name"; diff "$TMP/$name.rust" "$TMP/$name.coil" | head -20; fail=$((fail+1))
  fi
}
echo "=== byte-identical vs rust reference ==="
for h in types libc redteam typedef consts union refuse bitfield floats; do check "$h" "$TMP/$h.h"; done
for sys in ctype string stdlib math; do check "sys_$sys" "$TMP/sys_$sys.h"; done

echo ""
echo "cimport gate: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "SELF-HOSTED cimport: byte-exact with the Rust reference." || exit 1
