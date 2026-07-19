#!/usr/bin/env bash
# The library gauntlet: run every corpus workload (REAL libraries, real-world
# program shapes) on microclj AND real JVM Clojure, join per-corpus with
# compare.clj (checksum-refusal + NOISY rules), and print one combined table.
#
#   clojure-stub/bench/gauntlet.sh              # full budgets (publishable)
#   clojure-stub/bench/gauntlet.sh --quick      # small budgets, for iterating
#   clojure-stub/bench/gauntlet.sh --only NAME  # one corpus entry (substring)
#   clojure-stub/bench/gauntlet.sh --update     # record baseline.tsv per corpus
#   clojure-stub/bench/gauntlet.sh --out DIR    # keep raw TSVs
#
# Baselines are REFERENCE DATA to diff against, not a gate — the policy is
# "record, diagnose, fix", never "block".
#
# Requires `clojure` on PATH and a release+jit microclj:
#   cargo build --release --features jit -p clojure-stub --bin microclj
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
ROOT="$PWD"
MICROCLJ="$ROOT/target/release/microclj"
BENCH="$ROOT/clojure-stub/bench"

QUICK=0
ONLY=""
UPDATE=0
OUT="$(mktemp -d)"
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --update) UPDATE=1; shift ;;
    --out) OUT="$2"; KEEP=1; mkdir -p "$OUT"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x "$MICROCLJ" ]; then
  echo "no microclj at $MICROCLJ — build it first:" >&2
  echo "  cargo build --release --features jit -p clojure-stub --bin microclj" >&2
  exit 2
fi

# corpus entry -> "vendor-path jvm-alias" (mirrors each corpus manifest.edn;
# the manifest stays the human-readable source of truth).
corpus_vendor() {
  case "$1" in
    match-engine)  echo "clojure-stub/vendor/core.match match" ;;
    json-pipeline) echo "clojure-stub/vendor/data.json json" ;;
    property-run)  echo "clojure-stub/vendor/test.check test.check" ;;
    *) echo "" ;;
  esac
}

# Same code path, smaller budgets: quick mode runs from a COPY of the bench
# tree with a patched harness (see bench.sh for the knob-verification rule).
RUN_BENCH="$BENCH"
if [ "$QUICK" = 1 ]; then
  RUN_BENCH="$OUT/bench_quick"
  mkdir -p "$RUN_BENCH"
  cp -R "$BENCH/corpus" "$RUN_BENCH/corpus"
  cp "$BENCH/deps.edn" "$BENCH/compare.clj" "$RUN_BENCH/"
  sed -e 's/^(def warmup-ms 3000)/(def warmup-ms 300)/' \
      -e 's/^(def samples 25)/(def samples 5)/' "$BENCH/harness.clj" > "$RUN_BENCH/harness.clj"
  grep -q '^(def warmup-ms 300)' "$RUN_BENCH/harness.clj" && grep -q '^(def samples 5)' "$RUN_BENCH/harness.clj" || {
    echo "--quick could not patch harness.clj's knobs (were they renamed?). Refusing to run." >&2
    exit 3
  }
  echo "QUICK run — reduced warmup/samples. Indicative only, do not publish." >&2
fi

COMBINED_MICRO="$OUT/all_micro.tsv"
COMBINED_JVM="$OUT/all_jvm.tsv"
: > "$COMBINED_MICRO"
: > "$COMBINED_JVM"

FAILED=0
for dir in "$RUN_BENCH"/corpus/*/; do
  name="$(basename "$dir")"
  [ -n "$ONLY" ] && case "$name" in *"$ONLY"*) ;; *) continue ;; esac
  read -r vendor alias <<< "$(corpus_vendor "$name")"
  if [ -z "$vendor" ]; then
    echo "== $name: no vendor mapping in gauntlet.sh (add it + manifest.edn) — skipping" >&2
    continue
  fi

  echo "==> $name (microclj --jit)" >&2
  MICROLANG_PATH="$RUN_BENCH:$ROOT/$vendor" "$MICROCLJ" --jit "$dir/bench.clj" > "$OUT/${name}_micro.tsv" || {
    echo "== $name: microclj run FAILED" >&2; FAILED=1; continue; }

  echo "==> $name (JVM Clojure)" >&2
  (cd "$RUN_BENCH" && clojure -M:"$alias" "corpus/$name/bench.clj") > "$OUT/${name}_jvm.tsv" || {
    echo "== $name: JVM run FAILED" >&2; FAILED=1; continue; }

  # per-corpus checksum gate (refuses on mismatch)
  echo >&2
  echo "── $name ──"
  (cd /tmp && clojure -M "$BENCH/compare.clj" "$OUT/${name}_micro.tsv" "$OUT/${name}_jvm.tsv") || FAILED=1

  grep '^RESULT' "$OUT/${name}_micro.tsv" >> "$COMBINED_MICRO" || true
  grep '^RESULT' "$OUT/${name}_jvm.tsv" >> "$COMBINED_JVM" || true

  if [ "$UPDATE" = 1 ] && [ "$QUICK" = 0 ]; then
    { echo "# recorded $(git -C "$ROOT" rev-parse --short HEAD) $(date +%Y-%m-%d)"
      paste <(grep '^RESULT' "$OUT/${name}_micro.tsv") <(grep '^RESULT' "$OUT/${name}_jvm.tsv") \
      | awk -F'\t' '{printf "%s\t%s\t%.4f\t%.4f\t%.2f\n", $2, $3, $4, $13, $4/$13}'
    } > "$BENCH/corpus/$name/baseline.tsv"
    echo "recorded $BENCH/corpus/$name/baseline.tsv" >&2
  fi
done

echo
echo "══ corpus combined ══"
(cd /tmp && clojure -M "$BENCH/compare.clj" "$COMBINED_MICRO" "$COMBINED_JVM") || FAILED=1

if [ "$KEEP" = 1 ]; then
  echo "raw TSVs in $OUT" >&2
else
  rm -rf "$OUT"
fi
exit $FAILED
