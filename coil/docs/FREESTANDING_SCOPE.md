# Freestanding / no-runtime — SCOPE (for review before building)

The §7 axis: Coil running with NO runtime — no libc, no crt0, (option A) no OS at
all. The Leader's framing is right: the FIRST friction is the SETUP (how to
build+run a no-runtime program at all), infrastructure before code. This scopes
that. Findings are evidence-backed; decisions are flagged for a steer.

## Finding 1 — Coil's codegen is already freestanding-CLEAN

A minimal program and a struct-heavy program each emit **ZERO undefined symbols**
(`nm -u` empty) — no implicit `malloc`, `abort`, panic-to-stderr, or runtime
helper. LLVM **inlines** small fixed-size struct `memcpy`s (no libc `memcpy` dep).
The ONLY runtime couplings are:
- **crt0 → `main`**: the C runtime's start code calls `main`. (Coil specially emits
  `main` as the `main` symbol; everything else is module-mangled.)
- **explicitly-externed libc** (`memcpy`/`malloc`/`write`): the program asks for
  these by name; freestanding code simply doesn't — it provides its own.

So the "no runtime, AOT" moat holds at the codegen level. Freestanding is a
BUILD/LINK + tiny-lib problem, not a codegen-rewrite.

Caveat: LLVM may emit real `memcpy`/`memset` CALLS for large/variable-size copies
(loop idioms). Freestanding needs minimal `memcpy`/`memset` defs (a few lines of
Coil/`llvm-ir`) — surfaced, easily provided.

## Finding 2 — the host (macOS) cannot RUN a freestanding binary

Evidence: `cc -nostdlib …` → `ld: dynamic executables or dylibs must link with
libSystem.dylib`, even with `-e _main`. macOS mandates libSystem and has no stable
syscall ABI. **True freestanding can't run on the host as a macOS binary.**

## Finding 3 — available tooling (what's installed)

- **`ld.lld`** (LLVM cross-linker) — present. Can link a Linux/bare-metal ELF from a
  cross-targeted object (with `-nostdlib -static -e <entry>` + a linker script).
- **`qemu-system-aarch64` / `qemu-system-arm`** — present (FULL-SYSTEM emulation:
  boots bare-metal/kernel images). NO qemu-user (`qemu-x86_64`/`qemu-aarch64`), so a
  Linux-syscall ELF can't run on the host without the Linux server.
- No Linux cross-`cc`/sysroot.

## The two viable paths

**(A) Bare-metal aarch64 under `qemu-system-aarch64 -M virt`** — SELF-CONTAINED, the
purest no-OS test (the FUTURE_WORK §7 MCU/UART demo). No libc, no OS, no crt0:
- IO = write bytes to the PL011 UART at MMIO `0x0900_0000` (a volatile store —
  expressible via `(llvm-ir … "store volatile …")` today; or a `:bits`/`:explicit`
  device-register layout + an `int↔ptr` fixed address, both already supported).
- "exit" = halt loop (or PSCI poweroff via an `hvc`/`smc` — a `defcc` syscall-like
  convention, the convention-as-type moat).
- entry = a Coil function as the ELF entry (`ld.lld -e <coil-entry>`); qemu `-kernel`
  loads the ELF and jumps to it. A linker script places `.text` at virt's load addr.
- Run: `qemu-system-aarch64 -M virt -cpu cortex-a72 -nographic -kernel <elf>` → the
  UART output appears on stdout. Testable + self-contained.

**(B) Linux x86_64/aarch64 freestanding ELF** — raw Linux syscalls (`write`/`exit`
via `svc`/`syscall`), `ld.lld -static -nostdlib -e _start`. More "normal," but needs
the Linux host (`computer.jimmyhmiller.com`) to RUN — **requires permission** (per
the standing rule). Build+link happen on the host; only the run is remote.

## Setup blockers (infrastructure, both paths)

1. **Link path**: `build_executable` is hardcoded `cc obj -o out`. Freestanding needs
   `ld.lld` + `-nostdlib -static -e <entry>` (+ a linker script for A). → add a
   freestanding build mode / link-flag + linker-script hook. (The one real
   build-system change.)
2. **Entry symbol**: no codegen change needed — link with `-e <coil-mangled-entry>`
   (e.g. `-e app.start`); the linker sets the entry. (`main` stays the libc entry for
   normal builds.)
3. **A tiny `lib/sys` (or `lib/bare`)**: UART/MMIO write (A) or `write`/`exit`
   syscalls (B), via `llvm-ir` volatile/`svc`. ~20 lines, library.
4. **`memcpy`/`memset`** freestanding defs (for the large-copy idiom case).
5. **Allocator**: a static-buffer arena (no malloc) — `lib/alloc.coil`'s arena over a
   `(static (array u8 N))`. This is where the alloc-as-value design is FORCED.

## The deeper payoff (the Leader's point)
With NO ambient runtime, the explicit-capability design is MANDATORY: you MUST supply
your own allocator + IO — no libc fallback. So a freestanding program validates
"allocation and IO as explicit values" exactly where it shines. The scope already
confirms the codegen adds no hidden runtime; the dogfood proves the stdlib's
allocator/IO interfaces work with a self-provided backend (a static arena + an MMIO/
syscall Writer).

## Decisions for the steer
1. **Path A (bare-metal qemu, self-contained, purest) or B (Linux host, permission)?**
   Lean: **A** — no external deps/permission, and it's the strongest "as low as
   Zig/C" statement (no OS at all), exercising MMIO/layout/convention (the moat).
2. **Build mechanism**: a dedicated `freestanding` build mode (bundles ld.lld + the
   linker script + flags), or a generic `--link-with`/`--link-flag` passthrough?
3. **IO/exit mechanism**: `(llvm-ir …)` volatile/`svc` (works today, minimal), or a
   `defcc` syscall convention + `:bits` device-register layout (more on-brand, more
   setup)? Lean: start with `llvm-ir` to ship the dogfood, then upgrade to the
   convention/layout form to showcase the moat.

## Proposed minimal first dogfood (path A)
A bare-metal aarch64 program: a static-arena allocator + a UART Writer (volatile
store to `0x0900_0000`), reusing `lib/io.coil`'s `Writer` capability with a custom
MMIO backend, that prints "hi from coil\n" and halts — run under `qemu-system-aarch64
-M virt -nographic`. Proves: no runtime, no OS, and the explicit alloc/IO design
working with self-provided backends. Then iterate on whatever friction it surfaces.
