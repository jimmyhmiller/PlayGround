#!/usr/bin/env bash
# The ONE place that knows how to link the LLVM backend. Emits the `--link-flag …`
# arguments for `coil build`, in one of two modes. rebootstrap.sh consumes this;
# nothing else should hand-roll an LLVM link line.
#
#   llvm-link-flags.sh dynamic   # link Homebrew's libLLVM.dylib  (small, needs LLVM installed)
#   llvm-link-flags.sh static    # link LLVM's component archives (fat, self-contained)
#
# There is a third build that needs none of this: selfhost/rebootstrap-nollvm.sh
# builds selfhost/src/main_a64.coil, which has no LLVM backend at all and links
# only libSystem. See the table in rebootstrap.sh.
#
# DYNAMIC is the historical default and what the committed seed expects. The
# resulting compiler carries a hard dependency on
# /opt/homebrew/opt/llvm/lib/libLLVM.dylib and will not run on a machine without
# it.
#
# STATIC links the LLVM component archives instead, the way rustc and zig ship
# LLVM (rustc statically links it into a ~200MB librustc_driver; there is no
# system libLLVM anywhere in a rustup toolchain). The result is ~92MB versus
# ~3.5MB, and depends only on macOS system libraries in /usr/lib — no Homebrew.
# Two details make that possible:
#   * z3. Homebrew's `llvm-config --system-libs` reports libz3.dylib regardless of
#     which components you ask for, because Homebrew builds LLVM with
#     LLVM_ENABLE_Z3_SOLVER. z3 is used by clang's static analyzer, not by codegen,
#     so the components below do not reference it and it is deliberately omitted.
#     Passing it would reintroduce a Homebrew dependency for nothing.
#   * zstd. Homebrew's libzstd has no /usr/lib copy, so `-lzstd` resolves to a
#     Homebrew dylib. Link the static archive by full path instead when it exists.
set -uo pipefail

MODE="${1:-dynamic}"
LLVM_CONFIG="${LLVM_CONFIG:-/opt/homebrew/opt/llvm/bin/llvm-config}"
command -v "$LLVM_CONFIG" >/dev/null 2>&1 || LLVM_CONFIG=llvm-config
command -v "$LLVM_CONFIG" >/dev/null 2>&1 || {
  echo "llvm-link-flags: no llvm-config (set LLVM_CONFIG=/path/to/llvm-config)" >&2; exit 1; }

# the components the Coil backend actually calls into (see selfhost/src/ffi.coil):
# IR construction + the three targets it can emit for + the O3 pass pipeline.
COMPONENTS="core target analysis passes aarch64 x86 webassembly"

emit() { for f in "$@"; do printf -- '--link-flag %s ' "$f"; done; }

case "$MODE" in
  dynamic)
    emit "-L$("$LLVM_CONFIG" --libdir)" -lLLVM
    ;;
  static)
    emit "-L$("$LLVM_CONFIG" --libdir)"
    # shellcheck disable=SC2046
    emit $("$LLVM_CONFIG" --link-static --libs $COMPONENTS)
    emit -lm -lz
    ZSTD_A="$(brew --prefix zstd 2>/dev/null)/lib/libzstd.a"
    if [ -f "$ZSTD_A" ]; then emit "$ZSTD_A"; else emit -lzstd; fi
    emit -lxml2 -lc++
    ;;
  *)
    echo "llvm-link-flags: unknown mode '$MODE' (want: static | dynamic)" >&2; exit 1
    ;;
esac
echo
