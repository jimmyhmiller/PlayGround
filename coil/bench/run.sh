#!/usr/bin/env bash
# Benchmark Coil's compiled output against C (clang) and Zig on matched programs.
#
#   Coil : coil build      (LLVM -O3 pipeline, the default since the optimizer landed)
#   C    : cc -O3          the fair baseline -- same LLVM backend, same opt tier
#   Zig  : zig -OReleaseFast (Zig's optimizer + backend)
#
# Each program prints one integer/float; the harness checks every language agrees
# before timing (a fast wrong answer is meaningless). Timing is via hyperfine.
#
#   bench/run.sh            # run every benchmark
#   bench/run.sh fib loop   # run a subset
#
# Zig is OPTIONAL: if `zig` is absent — or can't link on this host — its column is
# skipped with a note, and Coil-vs-C still runs. (On macOS 26 the bundled `zig`
# linker can't resolve the newer libSystem, so we compile Zig to an object and link
# it with the system `cc`, which does work.)
set -euo pipefail
cd "$(dirname "$0")/.."
export LLVM_SYS_211_PREFIX="${LLVM_SYS_211_PREFIX:-$(echo /opt/homebrew/Cellar/llvm@21/* 2>/dev/null | tr ' ' '\n' | tail -1)}"
CC="${CC:-cc}"
COIL=target/debug/coil

command -v hyperfine >/dev/null || { echo "need hyperfine (brew install hyperfine)"; exit 1; }
echo "building the coil compiler..."
cargo build -q

# Does Zig work end-to-end on this host (compile an object + link it with cc)?
HAVE_ZIG=0
if command -v zig >/dev/null; then
  ZT="$(mktemp -d)"
  printf 'export fn main() c_int { return 0; }\n' > "$ZT/probe.zig"
  if zig build-obj -OReleaseFast "$ZT/probe.zig" -femit-bin="$ZT/probe.o" >/dev/null 2>&1 \
     && "$CC" "$ZT/probe.o" -o "$ZT/probe" >/dev/null 2>&1; then
    HAVE_ZIG=1
  fi
  rm -rf "$ZT"
fi
[ "$HAVE_ZIG" = 1 ] && echo "zig: enabled ($(zig version))" || echo "zig: skipped (absent or can't link on this host)"

# Build a Zig benchmark: compile to an object (zig's optimizer), link with cc (the
# system linker, which handles the host libSystem). Echoes the binary path on success.
build_zig() { # <src.zig> <out.o> <out.bin>
  zig build-obj -OReleaseFast "$1" -femit-bin="$2" >/dev/null 2>&1 && "$CC" "$2" -o "$3" >/dev/null 2>&1
}

BENCHES=("$@"); [ ${#BENCHES[@]} -eq 0 ] && BENCHES=(fib tak loop float memory structcall slicesum genreduce)
T="$(mktemp -d)"
RESULTS="bench/RESULTS.md"
{
  echo "# Coil vs C vs Zig benchmarks"
  echo
  echo "Host: \`$(uname -msr)\`, $($CC --version | head -1)$([ "$HAVE_ZIG" = 1 ] && echo ", zig $(zig version)")."
  echo "Coil = \`coil build\` (LLVM -O3); C = \`cc -O3\`; Zig = \`-OReleaseFast\`. All verified to print identical results."
  echo
} > "$RESULTS"

for b in "${BENCHES[@]}"; do
  echo "=== $b ==="
  "$COIL" build "bench/$b.coil" -o "$T/$b.coil" >/dev/null
  "$CC" -O3 "bench/$b.c" -o "$T/$b.c3"

  oc=$("$T/$b.coil"); o3=$("$T/$b.c3")
  if [ "$oc" != "$o3" ]; then
    echo "MISMATCH in $b: coil=$oc  c-O3=$o3"; exit 1
  fi

  # Optional Zig build + agreement check.
  zig_args=()
  if [ "$HAVE_ZIG" = 1 ] && [ -f "bench/$b.zig" ] && build_zig "bench/$b.zig" "$T/$b.zo" "$T/$b.zig"; then
    oz=$("$T/$b.zig")
    if [ "$oz" != "$oc" ]; then
      echo "MISMATCH in $b: coil=$oc  zig=$oz"; exit 1
    fi
    zig_args=(-n "Zig" "$T/$b.zig")
    echo "result = $oc (coil == c-O3 == zig)"
  else
    echo "result = $oc (coil == c-O3)"
  fi

  hyperfine -N --warmup 2 --runs 8 \
    -n "C -O3"    "$T/$b.c3" \
    -n "Coil -O3" "$T/$b.coil" \
    "${zig_args[@]}" \
    --export-markdown "$T/$b.md"

  { echo "## \`$b\`  (result \`$oc\`)"; echo; cat "$T/$b.md"; echo; } >> "$RESULTS"
done

echo
echo "wrote $RESULTS"
