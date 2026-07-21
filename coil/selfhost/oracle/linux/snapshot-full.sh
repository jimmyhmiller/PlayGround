#!/usr/bin/env bash
# Bless the Linux IR reference snapshot from a VERIFIED compiler (fixpoint +
# gate-run + gate-cli green first — the snapshot's authority is that compiler).
# Usage: selfhost/oracle/linux/snapshot-full.sh <coil-binary>
set -euo pipefail
cd "$(dirname "$0")/../../.."
BIN=${1:?usage: snapshot-full.sh <coil-binary>}
REF=selfhost/oracle/linux/full-reference
LIST=selfhost/oracle/full/corpus.txt
ARM64_ONLY="examples/shim.coil examples/everything.coil"   # arm64-register shim conventions
mkdir -p "$REF"
n=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case " $ARM64_ONLY " in *" $f "*) continue;; esac
  "$BIN" emit-ir "$f" > "$REF/$(echo "$f" | tr '/' '_').dump"
  n=$((n+1))
done < "$LIST"
echo "blessed $n Linux IR references -> $REF"
