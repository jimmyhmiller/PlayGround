#!/usr/bin/env bash
# Build + run the freestanding bare-metal aarch64 dogfood. This is the BUILD RECIPE —
# the freestanding-ness (target, linker script, entry) lives HERE and in the program,
# composed on top of the compiler's generic mechanism (emit-obj). The compiler bakes
# in NO "freestanding mode": it just emits an object; this recipe links + runs it.
#
#   ./freestanding/run.sh             # build + run hello.coil under qemu
#   ./freestanding/run.sh uart        # the typed-register PL011 driver (lib/mmio.coil)
#   ./freestanding/run.sh hello build # build only (emit .elf)
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/Cellar/llvm@18/18.1.8/bin:$PATH"
export LLVM_SYS_180_PREFIX="/opt/homebrew/Cellar/llvm@18/18.1.8"

PROG="${1:-hello}"          # hello | uart
SRC=freestanding/$PROG.coil
OBJ=/tmp/coil-bare.o
BOOT=/tmp/coil-boot.o
ELF=/tmp/coil-bare.elf

# 1. compiler: emit a bare-metal aarch64 object (the generic mechanism — no link).
cargo run -q emit-obj "$SRC" --target aarch64-unknown-none -o "$OBJ"

# 1b. assemble the crt0 boot stub (sets SP, zeroes .bss, calls the Coil entry).
clang -target aarch64-unknown-none -c freestanding/start.s -o "$BOOT"

# 2. recipe: link freestanding with ld.lld — no crt0/libc from a toolchain (start.s
#    IS the runtime), our linker script, entry = _start (from the script's ENTRY).
#    `--gc-sections` GCs unreferenced functions (the compiler emits per-function
#    sections), so importing the stdlib does NOT drag its unused libc calls into the
#    link — `-ffunction-sections -Wl,--gc-sections`, exactly as Zig/Rust/C do.
ld.lld --gc-sections -T freestanding/virt.ld "$BOOT" "$OBJ" -o "$ELF"

# verify: a freestanding image has NO undefined symbols (no libc dependency).
if nm -u "$ELF" 2>/dev/null | grep -q .; then
  echo "FAIL: freestanding image has undefined symbols (libc leak):" >&2
  nm -u "$ELF" >&2
  exit 1
fi

[ "${2:-run}" = build ] && { echo "built $ELF"; exit 0; }

# 3. run under full-system qemu: no OS, the ELF IS the kernel. PL011 UART -> stdout.
#    The program PSCI-poweroffs, so qemu exits on its own (timeout is a safety net).
exec timeout 10 qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -kernel "$ELF"
