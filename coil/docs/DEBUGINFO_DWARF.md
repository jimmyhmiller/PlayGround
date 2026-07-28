# DWARF debug info

`coil build -g <file>` / `coil run -g <file>` emit DWARF so `lldb`/`gdb` can map the
program to its source. **`-g` implies the native arm64 backend** (`--backend arm64`),
which is the debug backend: it always emits DWARF, it does not optimize, and it is the
only path that emits debug info — the LLVM backend emits none.

    coil build examples/fib.coil -o /tmp/fib -g
    lldb /tmp/fib
    (lldb) breakpoint set --file fib.coil --line 7
    Breakpoint 1: where = fib`fib + 68 at fib.coil:7:26
    (lldb) run
    * frame #0: fib`fib(n=10) at fib.coil:7:26
      frame #1: fib`main at fib.coil:11:8

## What is emitted

`selfhost/src/dwarf.coil` (419 lines, no dependencies) builds the
`__debug_abbrev` / `__debug_info` / `__debug_str` / `__debug_line` section bytes
directly, from events `codegen_a64.coil` collects during emission. DWARF v4, language
C:

- **A compile unit** per source file, and **one `DW_TAG_subprogram` per function** at
  its `defn` line.
- **Per-expression line rows** — every `Expr` carries a span, so each statement maps to
  its own `file:line:col`. `next` walks the body line by line, and a line breakpoint
  lands where you asked.
- **Parameters and locals** as `DW_OP_fbreg` over `x29`, so `frame variable` and
  `p x` work.
- **DI types** for scalars, pointers, structs, slices and arrays. Sums and function
  pointers are **omitted rather than guessed** — no entry is better than a wrong type.

## Relocations (the part that makes `dsymutil` work)

On Mach-O the DWARF stays in the `.o` and the linker records a debug map pointing at
it, so a `-g` build runs **`dsymutil`** to gather `<exe>.dSYM` next to the executable
and keeps the `.o`.

For that to work every DWARF address must carry a relocation — the CU's and each
subprogram's `low_pc`, and the line program's `set_address`. Each is an 8-byte
**unsigned section relocation** against `__text` with the offset as the addend, exactly
clang's Mach-O scheme (`dw-reloc!` populating the relocation list, `codegen_a64.coil`
emitting them, and a section-relocation path in `macho.coil`'s writer gated on a
negative-symbol sentinel so every existing extern relocation is byte-identical).
Without them `dsymutil` rejects the object ("No valid relocations found. Skipping."),
producing an empty `.dSYM`, zero line rows, and "No source available" in lldb.

## Gaps

- A **function-name** breakpoint needs the module-qualified name (`dbg.add`); line
  breakpoints, the primary workflow, are unaffected.
- **Sums and function pointers** have no DI type.
- Functions from an imported file have no line info (spans across `import` are
  `DUMMY` until the reader stamps real multi-source spans) — never wrong info, just
  none.

Gated in `selfhost/oracle/gate-cli.sh`: with the `.o` deleted so lldb must use the
`.dSYM` alone, a line breakpoint still resolves — which fails on a seed that emits no
relocations. Beyond the gate, `frame variable` at such a breakpoint reports typed
values (`(app.Pt *) p = 0x16fdfde70`, `(long) s = 42`).
