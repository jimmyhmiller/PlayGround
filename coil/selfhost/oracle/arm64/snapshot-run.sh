#!/usr/bin/env bash
# Behavioral reference snapshot for the arm64 backend gate.
#
# Builds every corpus program with the SELF-HOST compiler's LLVM backend and
# records its runtime behavior (stdout + exit code) into reference/.  The
# arm64 gate (gate-run.sh) then builds the SAME programs with the self-host
# arm64 backend and diffs runtime behavior byte-for-byte.  Runtime equality —
# not IR equality — is the contract between two different backends.
#
# usage: snapshot-run.sh <coil-self-bin>
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root
BIN="${1:?usage: snapshot-run.sh <coil-self-bin>}"
HERE=selfhost/oracle/arm64
REF="$HERE/reference"
mkdir -p "$REF"
LF=()
fail=0
n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  # corpus line: <file> [args...]
  set -- $line
  RUSTREF=""
  if [ "$1" = "R" ]; then RUSTREF=1; shift; fi
  f="$1"; shift
  id=$(echo "$f" | tr '/.' '__')
  exe="/tmp/coil-arm64-ref-$id"
  RBIN="$BIN"; [ -n "$RUSTREF" ] && RBIN=./target/debug/coil
  if ! "$RBIN" build "$f" -o "$exe" >/dev/null 2>"$REF/$id.buildlog"; then
    echo "BUILD FAIL: $f (see $REF/$id.buildlog)"; fail=1; continue
  fi
  cp "$exe" /tmp/coil-arm64-fixed-$id && out=$(/tmp/coil-arm64-fixed-$id "$@" </dev/null 2>"$REF/$id.stderr"; echo "EXIT:$?")
  code="${out##*EXIT:}"
  printf '%s' "${out%EXIT:*}" > "$REF/$id.stdout"
  echo "$code" > "$REF/$id.exit"
  rm -f "$REF/$id.buildlog"
  n=$((n+1))
done < "$HERE/corpus.txt"
echo "snapshotted $n programs into $REF"
exit $fail
