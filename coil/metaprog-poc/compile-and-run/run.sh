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

rm -rf "$OUT"
echo "=== both mechanisms verified ==="
