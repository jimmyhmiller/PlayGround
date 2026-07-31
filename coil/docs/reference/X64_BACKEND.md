# The x86-64 backend (selfhost, zero-deps)

A third backend for the **self-hosted** Coil compiler (`src/compiler/`), written
in Coil, that lowers the monomorphized `Program` directly to x86-64 machine code
in an **ELF** object — no LLVM anywhere in the path. It is the Linux sibling of
the arm64 backend (`docs/reference/ARM64_BACKEND.md`), and everything that document says
about *why* (debug backend, DWARF always on, compile speed over code quality)
applies here unchanged.

## Files
- `src/compiler/x64.coil` — x86-64 instruction encoder. Pure: composes REX /
  ModRM / SIB / displacement bytes into a plain byte buffer; label+fixup
  management for branches.
- `src/compiler/elf.coil` — ELF64 `ET_REL` writer: sections, `.rela.*`
  relocations (`R_X86_64_64`, `PC32`, `PLT32`), symbol table with the
  local-before-global ordering ELF requires, string tables.
- `src/compiler/dwarf.coil` — **shared** with the arm64 backend; the frame-base
  register and slot bias are now parameters (`fbreg`/`fbbias`) rather than
  hardcoded to x29.
- `src/compiler/codegen_x64.coil` — the lowering: mono `Program` → code + data +
  relocs + DWARF. Mirrors `codegen_a64.coil` function for function, so the two
  can be diffed against each other.
- `src/compiler/main_x64.coil` — the LLVM-free entry point (the `main_a64.coil`
  equivalent): omits `codegen.coil`/`normalize.coil` entirely.
- `src/compiler/x64_selftest.coil` — the encoder's verification harness.

## What differs from the arm64 backend
Everything not listed here is deliberately identical.

- **Registers.** `rbx` is the slot base (callee-saved, so it survives calls with
  no save/restore around each one), `rbp` the frame pointer, `rax`/`r10`/`r11`
  the scratch set. The scratch set is small and explicit because several x86
  instructions pin registers: `idiv` uses `rdx:rax`, variable shifts read `cl`,
  `cmpxchg` compares against `rax`.
- **The ABI is SysV AMD64, not AAPCS64.** There is no HFA rule; instead each
  aggregate ≤16 bytes is classified per *eightbyte* (INTEGER dominates SSE
  within a group), >16 bytes goes in memory, and a >16-byte return uses a
  hidden pointer in `rdi` (also returned in `rax`). Variadic calls must set
  `al` to the number of SSE registers used.
- **Object format is ELF.** Symbols carry no leading underscore (that is a
  Mach-O convention). Symbol order is a hard requirement, not a convention:
  every local must precede every global, with `.symtab`'s `sh_info` marking the
  boundary — a linker rejects the object otherwise. Debug sections carry REAL
  relocations against `.text`, unlike the Mach-O path where section-relative
  addresses plus ld64's debug map do the work.
- **Stack alignment.** SysV wants `rsp ≡ 0 (mod 16)` at every `call`. Entry is
  `8 (mod 16)`, and the prologue does an ODD number of pushes (`rbp`, `rbx`), so
  the frame subtraction is `8 (mod 16)`. Getting this wrong does not fault at
  the call site — it faults deep inside libc, in a `movaps`.
- **`frem` is a libm call.** SSE has no remainder instruction, so `frem` lowers
  to `fmod`/`fmodf` and the driver appends `-lm` for this backend.
- **Fixed-width patchable instructions.** The frame size is only known after the
  body is emitted, so the prologue emits `sub rsp, imm32` and
  `lea rbx, [rsp+disp32]` at their maximum width and patches the immediates in
  place. A shorter encoding chosen up front would have to move code to widen.
- **`:shim` conventions** name x86-64 registers (`rdi`, `r10`, …). An arm64 `xN`
  name is a hard error with the same per-arch diagnostic the LLVM backend gives
  for the reverse case, rather than silently picking the same-numbered register.

## Gates
`tests/compiler/oracle/x64/`:
- `gate-encode.sh` — every instruction the encoder can emit is diffed against
  **llvm-mc**, an independent assembler. 108 cases covering the encoding edge
  cases: `r8`–`r15` in each operand position, `rsp`/`r12` as a memory base
  (forces SIB), `rbp`/`r13` as a base (cannot use mod=00), the disp8/disp32
  boundaries, byte registers 4–7 (which need a bare REX to not mean `ah`/`ch`/
  `dh`/`bh`), and the immediate-size ladder. A handful of encoders deliberately
  emit a wider-than-canonical form (the patchable ones); those are marked
  `#wide` and verified by disassembling them back instead.
  *Teeth: swapping the `sub` opcode for `add` is caught immediately.*
- `gate-elf.sh` — hand-assembles `int main(){puts("elf ok");return 42;}`,
  writes it through `elf.coil`, then **links it with `cc` and runs it**. Also
  checks readelf's view and that the link produces no warnings (a missing
  `.note.GNU-stack` silently gives the whole program an executable stack).
  *Teeth: a wrong `sh_info` makes the real linker reject the object.*
- `python3 scripts/oracle.py runtime gate x64` — builds the 56-program corpus with `--backend x64`, runs each,
  and diffs stdout+exit byte-for-byte against the LLVM backend's behavior.
  Runtime equality, not IR equality, is the contract between backends.
  *Teeth: compiling signed `<` as unsigned fails 8 programs.*
- `gate-cabi.sh` — the SysV eightbyte rules, differentially against **gcc**.
  This gate exists because the corpus structurally cannot reach those rules:
  for Coil-to-Coil calls this backend passes aggregates BY POINTER (same as
  arm64), so struct classification only happens at the C boundary. Here a
  gcc-compiled translation unit is the callee, and every field carries a
  distinct prime weight — the first draft summed them evenly, which meant a
  backend that swapped two arguments still produced the right total.
  *Teeth: this is where the bug that broke the first self-host attempt lived.*
- `python3 scripts/dev.py build x64` — the finale: stage2/stage3 through the x64
  backend, **stage2.o == stage3.o byte-identical**, and the x64-built compiler
  itself passes the corpus.
- `python3 scripts/dev.py build nollvm-linux` — builds `main_x64.coil` and proves
  with `ldd` that the result links no libLLVM (only libc + libm).

## Results (2026-07-28, Strix Halo x86-64 host)
Same compiler binary, LLVM backend vs `--backend x64`, end-to-end `build`
including the `cc` link:

| input                             | LLVM     | x64      | speedup   |
|-----------------------------------|----------|----------|-----------|
| `src/compiler/main.coil` (~15k lines w/ libs) | 21.25 s | **1.61 s** | **13.2×** |
| `src/examples/json.coil`              | 129 ms   | 58 ms    | 2.2×      |

The ≥10× target the arm64 backend set is beaten on the input that matters.
Small programs are bound by the shared `cc`-link and process floor.

## Status
- [x] x86-64 encoder, 108/108 cases byte-identical to llvm-mc.
- [x] ELF64 writer; generated objects link with `cc` and run.
- [x] Full lowering: 56/56 behavioral corpus, including the adversarial ABI
      stress, narrow/odd-width integers, NaN-aware float comparisons, atomics,
      6-arg variadics + fnptr tables, deep recursion, 8-variant sums, bitfields.
- [x] DWARF always on: gdb resolves breakpoints by name, prints parameters and
      locals with correct values, and renders structs through pointers.
- [x] Self-hosting: `bootstrap-x64.sh` — fixpoint byte-identical, and the
      x64-built compiler passes the corpus.
- [x] LLVM-free build: `main_x64.coil` links only libc+libm, self-hosts to a
      byte-identical fixpoint, and passes the corpus.
- [ ] (future) direct executable emission, to drop the `cc` link floor.

## Bugs the gates caught (a sampler, since each argues for a gate)
- `mov %rax,(%rax)` — `emit-copy-to-mem!` used its scratch register as both the
  destination pointer and the value temp. This one bug accounted for 20 of the
  23 initial corpus failures; the copy helpers now defend against the aliasing.
- `setb` for ordered `<` — an unordered SSE compare sets CF=ZF=PF=1, so `b`/`be`
  answer TRUE for NaN. `OLT`/`OLE` now swap the operands and use `a`/`ae`, the
  only conditions that are false when unordered.
- A 16-byte aggregate spilled to the stack was passed as ONE eightbyte by the
  caller and read as TWO by the callee. Invisible to the corpus; it only
  surfaced when the x64-built compiler tried to compile something, because
  `run-pipeline` takes several slices past the sixth argument.
- Arm64 register numbers (`9`, `10`, `11`) leaking through the port as x86
  `r9`/`r10`/`r11`.
- A `>16B` struct passed to a C function went by POINTER rather than as a stack
  copy. AAPCS64 really does pass those indirectly, so the arm64 code this was
  ported from was correct — the SysV rule is different, and the port inherited
  the wrong one. Fixing it needed an explicit `isc` flag on the signature:
  dyn-dispatch signatures also declare aggregate params by value while passing
  references, so the two cases cannot be told apart from the types alone.

## An LLVM-backend bug this work surfaced — and fixed
Writing `gate-cabi.sh` exposed a real bug in the OTHER backend. `codegen.coil`
classified a by-value aggregate purely from its type, so a `{double,double}`
always became two separate `double` parameters. SysV does not work that way:
an aggregate is passed in registers only if enough are still free at its
position, and otherwise the WHOLE thing goes in memory — it never half-spills.
With seven doubles ahead of it the callee read `qx=5, qy=0` instead of
`qx=3, qy=5`. Verified three ways: gcc alone said 418, the coil LLVM path said
357, the x64 backend said 418.

`c-signature` now tracks integer and SSE register consumption across the
parameter list (counting the sret pointer, scalars, and each aggregate's
eightbytes) and demotes an `ACDirect` aggregate to `byval` when either file
would overflow — which is exactly what clang emits. The accounting is x86-only:
AArch64's `ACDirect` coercion is a single slot whose HFA/size rules the
classifier already handles, and cross-compiled arm64 IR is unchanged
(`[2 x double]`, as before).

The fix changed no existing program's output: `gate-full` is still byte-exact
across all 60 IR snapshots, because the demotion only triggers in the
register-exhausted case that nothing in the corpus reached. `gate-cabi.sh` now
covers both sides of that boundary — `c_p2_fits`/`c_f2_fits` land exactly in
the last free registers and must stay in them, while `c_p2_over`/`c_mix_over`
must demote. Reverting the fix trips checks 4, 14 and 16 (SSE, INTEGER and
mixed-class exhaustion) while the `_fits` cases stay green.
