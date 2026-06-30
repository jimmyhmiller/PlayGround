#!/usr/bin/env bash
# Compiler-throughput comparison: how fast does Coil's compiler build a program of a
# given size, vs clang and zig building the EQUIVALENT program? (compile_scale.sh
# measures Coil alone; this puts it next to the other toolchains.)
#
# Measured in BOTH build modes, because developers care about each:
#   Release  — coil `build` (LLVM -O3) | `cc -O3` | `zig build-exe -OReleaseFast`
#   Debug    — coil `build -g` (light pipeline + DWARF; coil's only non-O3 mode today)
#              | `cc -g -O0` | `zig build-exe` (Zig's default Debug mode)
#
# To be apples-to-apples the generated program uses only features all three share: a
# reachable chain of N plain integer functions (no generics/macros), emitted
# identically in Coil, C, and Zig. We verify all three print the same result at the
# smallest size, then time each compiler producing a full executable.
#
#   bench/compile_compare.sh                # default sizes
#   bench/compile_compare.sh 250 1000 4000  # custom function counts
#
# The Coil compiler is built `--release` (a fair match for the release clang/zig
# binaries). Zig needs a linker that works on this host: macOS 26's SDK ships core
# libSystem stubs as arm64e-only, and zig's self-hosted Mach-O linker only gained the
# arm64->arm64e fallback in 0.16. So we require zig >= 0.16 (e.g. `brew install zig`);
# a 0.15 default `zig` is left alone (ghostty etc.) and zig is just skipped here.
set -euo pipefail
cd "$(dirname "$0")/.."
export LLVM_SYS_211_PREFIX="${LLVM_SYS_211_PREFIX:-$(echo /opt/homebrew/Cellar/llvm@21/* 2>/dev/null | tr ' ' '\n' | tail -1)}"
COIL=target/release/coil
CC="${CC:-cc}"
echo "building the coil compiler (--release)..."
cargo build --release -q

# Find a zig >= 0.16 (one-step `build-exe` links natively on macOS 26). Prefer an
# explicit $ZIG, else the Homebrew binary, else the PATH default — but only accept it
# if its major.minor is >= 0.16.
zig_ok() { local z="$1"; [ -x "$z" ] || command -v "$z" >/dev/null 2>&1 || return 1
  local v; v=$("$z" version 2>/dev/null) || return 1
  python3 -c "import sys;mm=tuple(int(x) for x in '$v'.split('.')[:2]);sys.exit(0 if mm>=(0,16) else 1)"; }
ZIG=""
for cand in "${ZIG:-}" /opt/homebrew/bin/zig zig; do
  [ -n "$cand" ] || continue
  if zig_ok "$cand"; then ZIG="$cand"; break; fi
done
[ -n "$ZIG" ] && echo "zig: $("$ZIG" version) ($ZIG)" || echo "zig: skipped (need >= 0.16; default is $(zig version 2>/dev/null || echo absent))"

SIZES=("$@"); [ ${#SIZES[@]} -eq 0 ] && SIZES=(250 1000 4000)
T="$(mktemp -d)"

# g_i(x) = x*i + g_{i-1}(x);  g0(x) = x;  main = g_{N-1}(1) = sum(0..N-1).
gen_coil() { local n="$1"; { echo '(extern printf :cc c [(ptr i8) ...] (-> i32))'
  echo "(defn g0 [(x :i64)] (-> :i64) x)"
  for ((i=1;i<n;i++)); do echo "(defn g$i [(x :i64)] (-> :i64) (iadd (imul x $i) (g$((i-1)) x)))"; done
  echo "(defn main [] (-> :i64) (do (printf c\"%ld\\n\" (g$((n-1)) 1)) 0))"; } > "$2"; }
gen_c() { local n="$1"; { echo '#include <stdio.h>'; echo 'static long g0(long x){return x;}'
  for ((i=1;i<n;i++)); do echo "static long g$i(long x){return x*$i + g$((i-1))(x);}"; done
  echo "int main(void){printf(\"%ld\\n\", g$((n-1))(1));return 0;}"; } > "$2"; }
gen_zig() { local n="$1"; { echo 'const c = @cImport(@cInclude("stdio.h"));'; echo 'fn g0(x: i64) i64 { return x; }'
  for ((i=1;i<n;i++)); do echo "fn g$i(x: i64) i64 { return x *% $i +% g$((i-1))(x); }"; done
  echo "pub fn main() void { _ = c.printf(\"%ld\\n\", g$((n-1))(1)); }"; } > "$2"; }

timeit() { local a b d; a=$(_one "$@"); b=$(_one "$@"); d=$(_one "$@"); python3 -c "print(f'{min($a,$b,$d):.3f}')"; }
_one() { local s e; s=$(python3 -c 'import time;print(time.time())'); "$@" >/dev/null 2>&1 || { echo "FAILED: $*" >&2; exit 1; }
  e=$(python3 -c 'import time;print(time.time())'); python3 -c "print($e-$s)"; }

# Generate every size once (shared by both build modes).
for n in "${SIZES[@]}"; do
  gen_coil "$n" "$T/c_$n.coil"; gen_c "$n" "$T/c_$n.c"; [ -n "$ZIG" ] && gen_zig "$n" "$T/c_$n.zig"
done

RESULTS="bench/COMPILE_COMPARE.md"
{
  echo "# Compiler throughput — Coil vs clang vs zig"
  echo
  echo "Host: \`$(uname -msr)\`, $($CC --version | head -1)$([ -n "$ZIG" ] && echo ", zig $("$ZIG" version)")."
  echo "Coil compiler built \`--release\`. Equivalent programs (a reachable chain of N plain"
  echo "\`g_i\` integer functions, no generics/macros) built to a full executable, best of 3"
  echo "runs, lower is faster. Coil has no distinct Debug/Release build *modes* yet, so its"
  echo "\"debug\" column is \`build -g\` (the light, non-O3 pipeline + DWARF)."
} > "$RESULTS"

# Correctness gate (release, smallest size): all three print the same number.
n0="${SIZES[0]}"
"$COIL" build "$T/c_$n0.coil" -o "$T/cb" >/dev/null 2>&1; oc=$("$T/cb")
"$CC" -O3 "$T/c_$n0.c" -o "$T/cbc"; occ=$("$T/cbc")
[ "$oc" = "$occ" ] || { echo "MISMATCH coil=$oc clang=$occ"; exit 1; }
if [ -n "$ZIG" ]; then "$ZIG" build-exe "$T/c_$n0.zig" -OReleaseFast -lc -femit-bin="$T/cbz" >/dev/null 2>&1; oz=$("$T/cbz")
  [ "$oz" = "$oc" ] || { echo "MISMATCH coil=$oc zig=$oz"; exit 1; }; fi

# Time one (mode label, coil flags, cc flags, zig flags); appends a table to RESULTS.
do_mode() {
  local label="$1" cflags="$2" ccflags="$3" zflags="$4"
  { echo; echo "## $label builds"; echo
    echo "| functions | LOC | coil (s) | clang (s) | zig (s) | coil ms/fn | clang ms/fn | zig ms/fn |"
    echo "|---:|---:|---:|---:|---:|---:|---:|---:|"; } >> "$RESULTS"
  echo "-- $label --"; printf "%8s %9s %9s %9s\n" "fns" "coil(s)" "clang(s)" "zig(s)"
  local n cf cc_ zf loc ct clt zt cms clms zms
  for n in "${SIZES[@]}"; do
    cf="$T/c_$n.coil"; cc_="$T/c_$n.c"; zf="$T/c_$n.zig"; loc=$(wc -l < "$cf" | tr -d ' ')
    # shellcheck disable=SC2086
    ct=$(timeit "$COIL" build "$cf" -o "$T/cb" $cflags)
    # shellcheck disable=SC2086
    clt=$(timeit "$CC" $ccflags "$cc_" -o "$T/cbc")
    zt="—"; zms="—"
    if [ -n "$ZIG" ]; then
      # shellcheck disable=SC2086
      zt=$(timeit "$ZIG" build-exe "$zf" $zflags -lc -femit-bin="$T/cbz")
      zms=$(python3 -c "print(f'{1000*$zt/$n:.3f}')")
    fi
    cms=$(python3 -c "print(f'{1000*$ct/$n:.3f}')"); clms=$(python3 -c "print(f'{1000*$clt/$n:.3f}')")
    printf "%8s %9s %9s %9s\n" "$n" "$ct" "$clt" "$zt"
    echo "| $n | $loc | $ct | $clt | $zt | $cms | $clms | $zms |" >> "$RESULTS"
  done
}

do_mode "Release" ""   "-O3"     "-OReleaseFast"
do_mode "Debug"   "-g" "-g -O0"  ""

echo
echo "wrote $RESULTS"
