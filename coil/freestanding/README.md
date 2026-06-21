# Freestanding bare-metal dogfood (§7) — Coil runs as low as Zig/C

`hello.coil` is a Coil program that runs **bare metal**: no libc, no OS, no crt0. It
boots under `qemu-system-aarch64 -M virt`, prints over the PL011 UART, and PSCI-
poweroffs. This is the strongest "as low as Zig/C" statement there is — Zig and C run
bare-metal; now Coil does too.

```
./freestanding/run.sh          # build + run hello.coil  ->  hi / from coil (bare metal)
./freestanding/run.sh uart     # the typed-register PL011 driver -> PL011 via typed registers
./freestanding/run.sh arena    # the stdlib arena, RUN bare-metal -> stdlib arena on bare metal: 42
./freestanding/run.sh hello build   # build the .elf only
cargo test --test freestanding # the same, gated on the bare-metal toolchain being installed
```

## The capstone: device registers as a typed bitfield — a PURE MACRO

`uart.coil` drives the PL011 UART the RIGHT way (poll the flag register's TXFF bit,
then write the data register), with the device registers expressed as typed bitfields
via the `defmmio-reg` macro (`lib/mmio.coil`):

```
(defmmio-reg UARTFR 150994968 [(txff 5 1)])   ; 0x09000018, TXFF = bit 5
;; generates: UARTFR-read (volatile load), UARTFR-write (volatile store),
;;            UARTFR-txff  ((read >> 5) & 1)
```

This is "expressiveness through macros" at its hardest domain — a hardware
device-register interface as a 41-line LIBRARY, with **zero core feature**: `grep -ri
mmio|defmmio|device.reg src/` in the compiler is empty. The macro generates ordinary
defns over the existing `llvm-ir` (volatile load/store) + `iand`/`ishr`/`ishl`
primitives; the mask `(1<<width)-1` is emitted as const-foldable code. The moat (grow
the language by macro, never hack a feature into the core) demonstrated at the extreme.

## How it's built (no baked-in "freestanding mode")

The compiler bakes in NOTHING freestanding-specific. It exposes the generic mechanism
— `coil emit-obj <src> --target aarch64-unknown-none` emits an object — and the
freestanding-ness lives entirely in this **recipe** (`run.sh`) + the **program**:

1. `coil emit-obj … --target aarch64-unknown-none` → a bare-metal aarch64 object
   (`+strict-align`, see below).
2. assemble `start.s` (the crt0 boot stub: set SP, enable FP/SIMD, zero .bss, call the
   entry) → `boot.o`.
3. `ld.lld --gc-sections -T virt.ld boot.o obj -o elf` → link with no crt0/libc from a
   toolchain (`start.s` IS the runtime), our linker script, entry = `_start`.
4. `qemu-system-aarch64 -M virt -nographic -kernel elf` → the ELF *is* the kernel.

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
   — CLOSED: a program that IMPORTS lib/alloc and RUNS its arena (`arena-over-buffer`
   + `create`) now genuinely executes bare-metal (`arena.coil` → "stdlib arena on bare
   metal: 42"). Getting there meant fixing the bare-metal startup, triaged with `qemu
   -d int` in three layers — all in the recipe/build, NO core change:
   (1) **No stack pointer.** The entry prologue spilled to `SP≈0` → Data Abort (FAR
   `0xff..f0`). Fix: the `start.s` crt0 sets `SP` = top of a reserved stack.
   (2) **FP/SIMD trapped.** The optimizer emits SIMD for struct/array init; FP/SIMD is
   trapped at EL1 by reset → Undefined Instruction (ESR EC `0x7`). Fix: crt0 sets
   `CPACR_EL1.FPEN`.
   (3) **MMU off → Device memory.** With no page tables, RAM is Device memory, which
   faults on unaligned/SIMD access (the SIMD struct-init at a non-16-aligned `.bss`
   global → Alignment fault, DFSC `0x21`). Fix: emit the bare-metal object with
   `+strict-align` (derived from the `-none` triple in `compile_to_object_for`), so the
   backend emits no UNALIGNED wide accesses (an aligned vector copy may survive,
   harmless) — far cheaper than bringing up the MMU, and the standard bare-metal choice.
   The crt0 also zeroes `.bss` (the stdlib's `alloc-static` globals start zeroed).
   `hello.coil` worked before any of this only by luck (it doesn't spill / use SIMD);
   the crt0 + `+strict-align` make EVERY bare-metal program (incl. the stdlib) robust.
3. **`llvm-ir` can't have a `void` result** — `validate_type` rejects `void` outside a
   return, so an effect-only IR snippet must return a dummy `i64` (the UART/poweroff
   helpers return `i64 0`). Minor; a `void` llvm-ir result would be tidier.

## Verdict
Coil runs bare-metal, no OS — LINK *and* RUN, including the stdlib's arena. The moat
demonstrated at the extreme; the explicit alloc/IO design holds where it's mandatory.
All the freestanding machinery (crt0 boot stub, linker script, `--gc-sections`, the
`+strict-align` bare-metal codegen) lives in the recipe + a 3-line target-feature
derivation — NO core feature; freestanding composes on top, the way the goal demands.
