#!/usr/bin/env bash
# Every entry in the SHARED full/corpus.txt must have a blessed reference dump on
# BOTH platforms — full/reference (macOS) and linux/full-reference — except the
# entries arm64-only.txt deliberately excludes from Linux.
#
# Why this gate exists. corpus.txt is shared between the two snapshot scripts but
# the reference dirs are not, and the two rebootstrap scripts run different gate
# sets. That combination silently produced a repo where `gate-full` was green on
# Linux and dead on macOS: selfhost/oracle/features/fs_lib.coil was added to the
# corpus with a linux/full-reference dump and no macOS one, and nothing that ran on
# the Linux box could see it. This gate is cheap, needs no compiler, and runs
# identically on both platforms, so that asymmetry cannot hide again.
#
# It also catches the reverse (a macOS-only blessing) and orphaned dumps left
# behind when an entry leaves the corpus.
#
# Usage: selfhost/oracle/gate-corpus-coverage.sh
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root

LIST=selfhost/oracle/full/corpus.txt
MAC=selfhost/oracle/full/reference
LNX=selfhost/oracle/linux/full-reference
EXCL=selfhost/oracle/linux/arm64-only.txt

[ -f "$LIST" ] || { echo "gate-corpus-coverage: no corpus at $LIST"; exit 1; }

# The Linux-exempt set, from the single shared list (see arm64-only.txt).
exempt=" $(grep -v '^[[:space:]]*#' "$EXCL" 2>/dev/null | tr '\n' ' ') "

mangle() { echo "$1" | tr '/' '_'; }

fail=0
seen_mac=""
seen_lnx=""

while IFS= read -r f; do
  [ -z "$f" ] && continue
  d="$(mangle "$f").dump"
  # Trailing space matters: the membership test below is *" $b "*, so every entry
  # needs a space on BOTH sides or the last one added never matches.
  seen_mac="$seen_mac $d "

  if [ ! -f "$MAC/$d" ]; then
    echo "MISSING macOS reference : $f"
    echo "                           expected $MAC/$d"
    echo "                           bless it on a Mac: ./selfhost/oracle/snapshot-full.sh"
    fail=$((fail+1))
  fi

  case "$exempt" in *" $f "*) continue;; esac
  seen_lnx="$seen_lnx $d "

  if [ ! -f "$LNX/$d" ]; then
    echo "MISSING Linux reference : $f"
    echo "                           expected $LNX/$d"
    echo "                           bless it on Linux: ./selfhost/oracle/linux/snapshot-full.sh <coil>"
    fail=$((fail+1))
  fi
done < "$LIST"

# Orphans: a dump with no corpus entry is stale and will never be re-blessed.
for p in "$MAC"/*.dump; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  case "$seen_mac" in *" $b "*) ;; *)
    echo "ORPHAN macOS reference  : $b (no entry in $LIST)"; fail=$((fail+1));;
  esac
done
for p in "$LNX"/*.dump; do
  [ -e "$p" ] || continue
  b=$(basename "$p")
  case "$seen_lnx" in *" $b "*) ;; *)
    echo "ORPHAN Linux reference  : $b"
    echo "                           nothing blesses or reads it — either it left $LIST,"
    echo "                           or it is listed in $EXCL (Linux-exempt entries are"
    echo "                           checked for a per-arch error, never diffed against a dump)"
    fail=$((fail+1));;
  esac
done

n=$(grep -cve '^[[:space:]]*$' "$LIST")
if [ "$fail" -ne 0 ]; then
  echo "gate-corpus-coverage: $fail problem(s) across $n corpus entries"
  exit 1
fi
echo "gate-corpus-coverage: PASS ($n corpus entries blessed on both platforms)"
