#!/usr/bin/env bash
# Proves the two mechanisms the compile-and-run metaprogram design rests on.
# Run from the REPO ROOT (imports resolve relative to CWD):  metaprog-poc/compile-and-run/run.sh [coil-binary]
set -uo pipefail
cd "$(dirname "$0")/../.."
COIL=${1:-./coil}
D=metaprog-poc/compile-and-run
OUT=$(mktemp -d)
DL="--link-flag -Wl,-undefined,dynamic_lookup"
EX="--link-flag -Wl,-export_dynamic"

echo "=== 1. callback: a coil dylib calls BACK into the coil host process ==="
$COIL build $D/plugin.coil --shared -o "$OUT/plugin.dylib" $DL || exit 1
$COIL build $D/host.coil   -o "$OUT/host"   $EX || exit 1
( cd "$OUT" && ./host ) || exit 1

echo "=== 2. a METAPROGRAM as a normal program: no Code type, no ECodeOp, no interpreter."
echo "       It imports the compiler's real reader and walks a real Sexp. ==="
$COIL build $D/meta.coil  --shared -o "$OUT/meta.dylib" $DL || exit 1
$COIL build $D/mhost.coil -o "$OUT/mhost" $EX || exit 1
( cd "$OUT" && ./mhost ) || exit 1

echo "=== 3. the same, through the REAL code-* library (selfhost/src/codelib.coil):"
echo "       a compiled metaprogram walking and BUILDING Sexp with the shipped vocabulary ==="
$COIL build $D/meta2.coil  --shared -o "$OUT/meta2.dylib" $DL || exit 1
$COIL build $D/mhost2.coil -o "$OUT/mhost2" $EX || exit 1
( cd "$OUT" && ./mhost2 ) || exit 1

echo "=== 4. THE TOWER: macros in macro bodies (type-directed definition-time expansion) ==="
echo "       when-as-statement + cond-in-Code-position expand; a Code->Code helper"
echo "       called with Code args stays a FUNCTION call — under BOTH engines"
$COIL run $D/tower_test.coil; rc=$?
[ $rc -eq 45 ] || { echo "tower test FAILED (compiled engine, exit $rc, want 45)"; exit 1; }
COIL_META=interp $COIL run $D/tower_test.coil; rc=$?
[ $rc -eq 45 ] || { echo "tower test FAILED (interpreter engine, exit $rc, want 45)"; exit 1; }
$COIL run $D/tower_fmt_test.coil 2>/dev/null; rc=$?
[ $rc -eq 70 ] || { echo "tower fmt test FAILED (exit $rc, want 70)"; exit 1; }
$COIL run $D/tower_msg_test.coil >/dev/null 2>&1; rc=$?
[ $rc -eq 0 ] || { echo "tower msg test FAILED (exit $rc, want 0)"; exit 1; }
echo "tower: OK (45/45 both engines; fmt-in-a-macro logged; objc-style msg macro with"
echo "       TYPE-SYNTAX arguments works in a metaprogram body — the parse-level trigger)"

echo "=== 5. ARBITRARY CODE in a metaprogram, via the real compiled engine ==="
echo "       generics + ArrayList + string-keyed HashMap + StrBuf + malloc + libc"
echo "       strlen, all AT EXPANSION TIME (every one impossible in the interpreter)"
COIL_META=compiled $COIL run $D/arbitrary_test.coil || { echo "compiled-engine arbitrary test FAILED"; exit 1; }
echo "arbitrary-code metaprogram: OK (exit 0)"

echo "=== 6. A GUI AT COMPILE TIME: the Mandelbrot viewer metaprogram ==="
echo "       a Cocoa window opens ON THE MAIN THREAD during expansion, renders the"
echo "       set live, and the accepted view's coordinates become the program's"
echo "       constants (COIL_MANDEL_AUTO=1 scripts the session; drop it to drive"
echo "       the viewer yourself: WASD pan, Z/X zoom, I/O iters, Q/RETURN accept)"
COIL_META_MAIN=1 COIL_MANDEL_AUTO=1 $COIL run $D/mandel_test.coil \
  --link-flag -framework --link-flag AppKit --link-flag -lobjc > "$OUT/mandel.txt" 2>/dev/null; rc=$?
[ $rc -eq 0 ] || { echo "mandelbrot GUI metaprogram FAILED (exit $rc)"; exit 1; }
grep -q "COMPILE-TIME GUI" "$OUT/mandel.txt" || { echo "mandelbrot output missing"; exit 1; }
head -6 "$OUT/mandel.txt"
echo "compile-time GUI: OK"

rm -rf "$OUT"
echo "=== all mechanisms verified ==="
