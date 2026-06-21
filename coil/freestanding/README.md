# Freestanding bare-metal dogfood (§7) — Coil runs as low as Zig/C

`hello.coil` is a Coil program that runs **bare metal**: no libc, no OS, no crt0. It
boots under `qemu-system-aarch64 -M virt`, prints over the PL011 UART, and PSCI-
poweroffs. This is the strongest "as low as Zig/C" statement there is — Zig and C run
bare-metal; now Coil does too.

```
./freestanding/run.sh          # build + run under qemu  ->  hi / from coil (bare metal)
./freestanding/run.sh build    # build the .elf only
cargo test --test freestanding # the same, gated on ld.lld + qemu being installed
```

## How it's built (no baked-in "freestanding mode")

The compiler bakes in NOTHING freestanding-specific. It exposes the generic mechanism
— `coil emit-obj <src> --target aarch64-unknown-none` emits an object — and the
freestanding-ness lives entirely in this **recipe** (`run.sh`) + the **program**:

1. `coil emit-obj … --target aarch64-unknown-none` → a bare-metal aarch64 object.
2. `ld.lld -T virt.ld -e bare.start obj -o elf` → link with no crt0/libc, our linker
   script, entry = the Coil `start` function.
3. `qemu-system-aarch64 -M virt -nographic -kernel elf` → the ELF *is* the kernel.

Same principle as "expressiveness through macros, not core hacks": the core stays
minimal (emit an object); freestanding composes on top.

## What it validates

- **Coil's codegen is freestanding-clean.** The linked image has ZERO undefined
  symbols (`run.sh`/the test assert this) — no implicit malloc/abort/panic/runtime.
  The "no-runtime moat" is real, by evidence.
- **The explicit alloc/IO design works — and is FORCED.** With no ambient runtime
  there is no libc malloc and no libc IO, so the program MUST supply its own. It does,
  using the capability-as-a-VALUE pattern (the same shape as `lib/alloc`/`lib/io`):
  - a **static-arena allocator** (`Arena` over an `alloc-static` buffer in .bss — a
    bump pointer, no malloc);
  - a **UART `Writer`** (a `{write-fn, ctx}` value with an MMIO backend).
  The dogfood allocates scratch from the arena and writes it *through* the Writer —
  the two explicit capabilities composing, with no runtime at all. This is exactly
  where "allocation and IO as explicit values" shines: it's mandatory, and it works.
- **The machine layer is reachable from Coil today.** UART MMIO is a volatile store to
  a fixed address (`inttoptr` + `store volatile` via the `llvm-ir` escape hatch); PSCI
  poweroff is an `hvc` via inline asm. No new core features needed.

## Friction surfaced (the dogfood's findings)

1. **No dead-code elimination → the stdlib dragged libc (the headline) — FIXED at the
   link level.** Coil emits ALL non-generic defns, used or not, so `(import
   "lib/alloc.coil" …)` pulled in `malloc`/`free`/`realloc`/`abort` as undefined
   symbols even when unused — a freestanding program would link libc just for an
   `import`. Fixed exactly as Zig/Rust/C do, a BUILD/LINK change (not a codegen
   rewrite): the compiler emits per-function sections (`-ffunction-sections`) for ELF
   targets, and the recipe links with `ld.lld --gc-sections` → unused defns are GC'd.
   Now `import lib/alloc` + `--gc-sections` links with ZERO undefined symbols (see the
   `importing_stdlib_does_not_drag_libc_with_gc_sections` test, which also documents
   the regression: WITHOUT `--gc-sections` the link fails on `malloc`). (The
   per-function sections are ELF-only — Mach-O/COFF use a different section grammar and
   dead-strip by symbol; emitting an ELF section name on Mach-O is a hard LLVM error,
   so the host build is untouched.)
2. **`arena-allocator` malloced its buffer** (`lib/alloc.coil`) — the convenience
   constructor wasn't freestanding. FIXED: added `arena-over-buffer (buf, cap)`, a
   constructor over a caller-provided buffer (no malloc — the buffer can be an
   `alloc-static` array in .bss); `arena-allocator` now wraps it with `malloc`.
   — KNOWN GAP (next investigation): a program that IMPORTS lib/alloc and runs its
   arena (`arena-over-buffer` + `create`) LINKS clean but does not yet RUN under qemu,
   while the self-contained `hello.coil` (which builds its `Arena` on the stack) runs
   fine. Isolated: importing lib/alloc and using nothing runs; the fault is in the
   stdlib-arena runtime path bare-metal (root cause TBD — likely runtime-init/global
   state). The DCE claim above is link-level and proven; the stdlib-arena *runtime*
   bare-metal is a separate follow-up, not overclaimed.
3. **`llvm-ir` can't have a `void` result** — `validate_type` rejects `void` outside a
   return, so an effect-only IR snippet must return a dummy `i64` (the UART/poweroff
   helpers return `i64 0`). Minor; a `void` llvm-ir result would be tidier.

## Verdict
Coil runs bare-metal, no OS — the moat demonstrated at the extreme. The explicit
alloc/IO design holds where it's mandatory. The one real friction (no DCE → stdlib
drags libc) has a standard build/link fix (function-sections + `--gc-sections`).
