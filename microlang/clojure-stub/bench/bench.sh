#!/usr/bin/env bash
# Run the perf suite on microclj and on real JVM Clojure, then table the two.
#
#   clojure-stub/bench/bench.sh              # full run (~3s warmup/workload)
#   clojure-stub/bench/bench.sh --quick      # fast, for iterating; NOT publishable
#   clojure-stub/bench/bench.sh --out DIR    # keep the raw TSVs
#
# Requires `clojure` on PATH and a release+jit microclj:
#   cargo build --release --features jit -p clojure-stub --bin microclj
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
ROOT="$PWD"
MICROCLJ="$ROOT/target/release/microclj"
BENCH="$ROOT/clojure-stub/bench"

QUICK=0
OUT="$(mktemp -d)"
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --out) OUT="$2"; KEEP=1; mkdir -p "$OUT"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x "$MICROCLJ" ]; then
  echo "no microclj at $MICROCLJ — build it first:" >&2
  echo "  cargo build --release --features jit -p clojure-stub --bin microclj" >&2
  exit 2
fi

# The suite requires the shared harness namespace (harness.clj). Both runtimes
# resolve it from the same directory: microclj via MICROLANG_PATH, the JVM via
# the deps.edn there (:paths ["."]).
RUN_DIR="$BENCH"
if [ "$QUICK" = 1 ]; then
  # Same code path, smaller budgets — a quick run must never be a DIFFERENT
  # harness, only a less patient one. The knobs live in harness.clj, so quick
  # mode runs from a COPY of the bench dir with a patched harness.
  RUN_DIR="$OUT/bench_quick"
  mkdir -p "$RUN_DIR"
  cp "$BENCH/suite.clj" "$BENCH/deps.edn" "$RUN_DIR/"
  sed -e 's/^(def warmup-ms 3000)/(def warmup-ms 300)/' \
      -e 's/^(def samples 25)/(def samples 5)/' "$BENCH/harness.clj" > "$RUN_DIR/harness.clj"
  # If harness.clj's knobs are ever renamed the sed above would silently no-op
  # and "--quick" would quietly run the FULL budget while claiming otherwise.
  # A benchmark harness that lies about its own settings is worse than none.
  grep -q '^(def warmup-ms 300)' "$RUN_DIR/harness.clj" && grep -q '^(def samples 5)' "$RUN_DIR/harness.clj" || {
    echo "--quick could not patch harness.clj's knobs (were they renamed?). Refusing to run." >&2
    exit 3
  }
  echo "QUICK run — reduced warmup/samples. Indicative only, do not publish." >&2
fi

echo "==> microclj --jit" >&2
MICROLANG_PATH="$RUN_DIR" "$MICROCLJ" --jit "$RUN_DIR/suite.clj" > "$OUT/microclj.tsv"

# The JVM runs from the bench dir itself: its deps.edn pins the classpath to
# exactly that dir, so nothing in the repo can shadow clojure.core.
echo "==> JVM Clojure" >&2
(cd "$RUN_DIR" && clojure -M suite.clj) > "$OUT/jvm.tsv"

echo >&2
(cd /tmp && clojure -M "$BENCH/compare.clj" "$OUT/microclj.tsv" "$OUT/jvm.tsv")

if [ "$KEEP" = 1 ]; then
  echo >&2
  echo "raw TSVs in $OUT" >&2
else
  rm -rf "$OUT"
fi
