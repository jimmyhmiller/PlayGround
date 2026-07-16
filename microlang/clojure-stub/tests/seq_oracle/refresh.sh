#!/usr/bin/env bash
# Regenerate the two spec files from the REAL implementations:
#
#   expected.txt       what `clojure`      prints for probe.clj  (JVM Clojure)
#   expected_cljs.txt  what ClojureScript  prints for probe.clj  (via node)
#
# Run this only when probe.clj changes. Neither file is ever hand-edited: they
# are what the implementations actually answered.
#
# WHY BOTH. The rule (Jimmy) for what microclj must reproduce: where Clojure and
# ClojureScript AGREE, that is the language and we match it. Where they DISAGREE
# it is an artifact of the host and we deliberately do not chase it. Keeping
# both answers on disk makes that rule mechanically checkable instead of a
# judgement call — and it has already overturned three "obvious" assumptions
# (see KNOWN_DIVERGENCES in ../seq_oracle.rs).
#
# `clojure` must run from a directory that does not shadow core.clj.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
CLJS_DEPS='{:deps {org.clojure/clojurescript {:mvn/version "1.11.132"}}}'

# Write via a temp file and move only on success. A probe form that throws makes
# the runtime die PART WAY THROUGH, and redirecting straight into the spec would
# silently enshrine a truncation that the test then happily "passes" against.
emit() { # emit <outfile> <description> <command...>
  local out="$1" what="$2"; shift 2
  local tmp; tmp="$(mktemp)"
  if ! "$@" > "$tmp" 2>/dev/null; then
    echo "$what exited non-zero -- $out NOT updated. Re-run without redirect to see why." >&2
    rm -f "$tmp"; exit 1
  fi
  # Keep only well-formed "label<TAB>value" lines (cljs.main interleaves
  # compiler warnings on stdout), and require that we got some.
  grep -E '^[^	]+	' "$tmp" > "$tmp.f" || true
  if [ ! -s "$tmp.f" ]; then
    echo "$what emitted no label<TAB>value lines -- $out NOT updated." >&2
    rm -f "$tmp" "$tmp.f"; exit 1
  fi
  mv "$tmp.f" "$out"; rm -f "$tmp"
  echo "$out refreshed: $(wc -l < "$out" | tr -d ' ') lines from $what." >&2
}

emit "$HERE/expected.txt" "JVM Clojure" \
  env -C /tmp clojure -M "$HERE/probe.clj"

# The SAME probe.clj, unmodified, on ClojureScript. `-i` loads+evaluates it.
emit "$HERE/expected_cljs.txt" "ClojureScript" \
  env -C /tmp clojure -Sdeps "$CLJS_DEPS" -M -m cljs.main -re node -O none -i "$HERE/probe.clj"

# Report where the two hosts disagree: those lines are host artifacts, and
# microclj is NOT expected to match either one in particular.
echo >&2
echo "Lines where Clojure and ClojureScript DISAGREE (host artifacts, do not chase):" >&2
join -t'	' "$HERE/expected.txt" "$HERE/expected_cljs.txt" 2>/dev/null \
  | awk -F'\t' '$2 != $3 { printf "  %-26s clj=%-34s cljs=%s\n", $1, $2, $3 }' >&2 || true
