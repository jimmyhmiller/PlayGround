# The Coil Bytecode Interpreter

`selfhost/src/interp.coil` is a bytecode interpreter for the self-hosted Coil
compiler. It consumes the **same monomorphized `Program`** the native backends
consume — the output of `parse -> resolve -> check -> mono` — compiles each
mono'd function to a compact stack-bytecode ISA, and executes it on a small VM
with a **real memory model**. It is reached only through the additive
`coil interp <file>` driver subcommand; the compile pipeline and the native
backends are byte-unchanged.

    coil interp examples/fib.coil ; echo $?     # => 55
    coil interp examples/extern.coil            # => prints 12345 (via libc putchar)

`main`'s `i64` return **is** the process exit code. Program stdout goes to real
stdout (the VM calls real libc).

This document is the contract later agents extend against: the ISA, the value and
memory model, the frame layout, and FFI dispatch. Keep it in sync with the code.

## Design in one sentence

Compile each `Func` to stack bytecode where **every value is one 8-byte cell**
(`i64`, or a float's bit pattern, or a real host pointer), execute it against
**genuine process memory** (real `malloc` heap, a malloc-backed per-frame buffer
for `alloc-stack`, real loads/stores), and dispatch `extern` calls to **real
native libc functions** — so pointer arithmetic, casts, and FFI behave exactly as
in compiled code, because they *are* the same operations on the same memory.

## Value model

A VM value is a single `i64` cell, mirroring codegen_a64's `AVal`:

| Coil value        | cell contents                                             |
|-------------------|-----------------------------------------------------------|
| `i8..i64`, `bool` | the integer, sign/zero-extended to the type's width       |
| `f32` / `f64`     | the IEEE-754 **bit pattern** (round-tripped via memory)   |
| `(ptr T)`, `(ref T)`, `(fnptr …)` | a **real host address**                   |
| aggregate by value (struct/sum/slice) | the **address** of its bytes (a frame slot) |

Floats are bit-cast to/from `i64` through a one-word memory round-trip
(`bits->f64`/`f64->bits`, etc.), the idiom the language guide prescribes for
bitcasts. Integer results are canonicalized to their exact width after each
arithmetic op (the "container invariant": a value in a cell is always
sign/zero-extended from bit N-1), matching `emit-canon-int!`.

## Memory model

Pointers are **real addresses in this process**. There is no sandboxed linear
memory — the interpreter runs in-process, so a pointer handed to `printf` or read
by pointer arithmetic is a genuine address.

- **Heap** (`(alloc … :heap)`, storage 2): real `malloc(size)`; `(free p)` is
  real `free`.
- **Static** (`alloc-static`, storage 1): a `malloc`'d, zeroed region (leaked for
  the process lifetime — statics live forever).
- **Stack** (`alloc-stack`, storage 0): bump-allocated from a **per-frame buffer**
  (see below). Returns a real address into that buffer, valid for the frame's
  lifetime.
- **Loads/stores** (`ELoad`/`EStore`, and mut-cell access): real
  `load`/`store` at the address, at the value's byte width (1/2/4/8), with
  sign/zero extension chosen by the pointee type (`mem-load`/`mem-store`).

Byte sizes and alignments (`isize-of`/`ialign-of`) agree with the native
backends' `ty-size`/`ty-align` for scalars, pointers (8), slices (16 = `{ptr,
len}`), arrays, and vecs — so layouts seen across the FFI boundary and by pointer
arithmetic are identical to compiled code.

## Frame layout

Each function activation (`vm-exec`) mallocs three regions and frees them on
return:

- **locals**: an `i64[nlocals]` array. Parameters occupy locals `0..nparams`;
  every `let`/temporary binding gets the next slot. A local holds a value cell
  directly. A `(mut x)` binding instead holds a **pointer** to a frame-buffer cell
  (see `MUTCELL`), so `(load x)`/`(store! x v)` lower to real `ELoad`/`EStore` —
  exactly as the type-checker already rewrote them.
- **frame buffer**: a 1 MiB malloc'd region with a bump pointer, backing
  `alloc-stack`, `(mut …)` cells, spilled refs, and string-literal slices.
  Overflow is a hard error, never silent.
- **operand stack**: a 4096-cell (`32 KiB`) `i64[]`. Bytecode is stack-machine
  code: operands are pushed and popped here.

Calls recurse through the **host call stack** — `OP_CALL` invokes `vm-exec`
re-entrantly — so Coil recursion (e.g. `fib`) maps onto native recursion. The
whole interpreter runs on the driver's 512 MiB pipeline pthread, so deep
recursion has room.

## The bytecode ISA

One instruction is `Instr {op, a, b, c}` (four `i64`s). A function is an
`ArrayList Instr` plus `nlocals`. Jump targets are resolved from label ids to
instruction indices after the body is emitted (`resolve-labels`).

Compilation is single-pass and type-directed: `compile-expr` emits an
expression's code (leaving exactly one value on the operand stack unless the path
diverged) and **returns its static `Type`** — the direct analogue of
codegen_a64's `emit-expr` returning an `AVal`. Types flow upward so arithmetic
opcodes know int-vs-float, width, and signedness.

| op | name         | operands                     | effect |
|----|--------------|------------------------------|--------|
| 1  | `IMM`        | a=value                      | push a |
| 2  | `GETLOCAL`   | a=slot                       | push locals[a] |
| 3  | `SETLOCAL`   | a=slot                       | locals[a] = pop |
| 4  | `POP`        |                              | discard top |
| 5  | `ALLOCA`     | a=size b=align               | push frame-alloc(size,align) (uninit) |
| 6  | `LOAD`       | a=size b=signed              | addr=pop; push mem-load(addr,size,signed) |
| 7  | `STORE`      | a=size                       | v=pop; addr=pop; mem-store; push 0 (unit) |
| 8  | `BIN`        | a=op b=mode c=signbits       | r=pop; l=pop; push binop(l,r) |
| 9  | `CMP`        | a=op b=mode c=signed         | r=pop; l=pop; push (0/1) |
| 10 | `NOT`        | a=isbool                     | v=pop; push logical/bitwise not |
| 11 | `JMP`        | a=target                     | pc = a |
| 12 | `JZ`         | a=target                     | v=pop; if v==0 pc=a |
| 13 | `CALL`       | a=funcidx b=nargs            | pop nargs; push vm-exec(a, args) |
| 14 | `CALLEXT`    | a=externidx b=nargs          | pop nargs; push extern-call(a, args) |
| 15 | `RET`        |                              | retval = pop; halt |
| 16 | `ICANON`     | a=bits b=signed              | v=pop; push canon-int(v,bits,signed) |
| 17 | `I2F`        | a=fbytes b=srcsigned         | int -> float bits |
| 18 | `F2I`        | a=destbits b=destsigned c=srcfbytes | float -> int (toward zero) |
| 19 | `F2F`        | a=destbytes b=srcbytes       | f32 <-> f64 |
| 21 | `MUTCELL`    | a=slot b=size                | v=pop; cell=frame-alloc; store v; locals[a]=cell |
| 22 | `FREE`       |                              | addr=pop; real free(addr); push 0 |
| 23 | `ALLOC_HEAP` | a=size                       | push (i64) malloc(size) |
| 24 | `ALLOC_STATIC`| a=size                      | addr=malloc(size); zero; push addr |
| 30 | `STRSLICE`   | a=ptr b=len                  | frame slot {ptr,len}; push its addr |
| 31 | `ALLOCZ`     | a=size b=align               | frame-alloc + zero; push addr |

### `BIN` / `CMP` operand encoding

- `mode`: 0 = integer, 1 = `f32`, 2 = `f64`.
- integer `BIN` `signbits` (`c`) packs `signed | (bits << 1)` so the result can be
  canonicalized to width `bits` afterward.
- `BIN` op codes match codegen_a64's `emit-bin`: `0 add 1 sub 2 mul 3 div 4 rem
  5 udiv 6 urem 7 and 8 or 9 xor 10 shl 11 shr`. Signed vs unsigned div/rem/shr
  chosen from the type's signedness.
- `CMP` op codes: `0 lt 1 le 2 gt 3 ge 4 eq 5 ne`. Unsigned integer comparisons
  use a sign-bit-flip trick; float comparisons use the ordered `fcmp-*` ops.

### Control flow and divergence

`compile-expr` tracks a `term` (terminated) flag on the builder, mirroring
codegen_a64's `terminated`. `break`/`continue` set it; `EIf` merges the flags of
its branches; sequence compilation stops emitting after a terminated statement
(dead code). Each non-diverging path pushes exactly one value, so the operand
stack stays balanced at control-flow joins. Loops allocate a hidden result local:
`break v` stores `v` there and jumps to the break label; the loop expression's
value is that local. `continue` jumps to the loop's top label — which is *before*
the body, so a `for`-macro's top-of-loop increment still runs (matching C `for`).

## FFI dispatch

`extern` calls (`CALLEXT`) dispatch through a **builtin table** (`extern-call`):
the extern's C symbol (its name's last dot component, `last-component2`, matching
the backends' `g-last-component`) selects a real native libc call with the
correct signature. Currently wired: `putchar`, `putc`, `write`, `puts`, `exit`,
`malloc`, `free`. Arguments are passed as `i64` cells and cast to the callee's C
types at the call site. An extern not in the table is a **hard error** (never a
silent no-op) — add it to the table to support it.

This is self-contained (no `dlsym`): the libc symbols are declared as
module-qualified `interp.*` externs (so the Coil scope name never collides while
the C symbol is the real one, e.g. `interp.write` -> `write`), with signatures
matching the compiler's existing declarations of those symbols.

## What is implemented (the slice) and what hard-errors

Implemented end-to-end (verified by `selfhost/oracle/interp/gate-interp.sh`):
integer + float arithmetic and comparisons, `bool`, casts (int width, int<->float,
float<->float, pointer/fnptr identity), `let`/`(mut …)` locals, `if`, `do`,
`loop`/`while`/`for`/`break`/`continue`, direct calls and recursion, `alloc-stack`
/ heap / static allocation, pointer `load`/`store`, `sizeof`/`alignof`, string and
cstring literals, spilled refs, and libc FFI.

**Not yet implemented** (each raises a clear `idie` hard error — never a silent
stub): structs (`EField`, aggregate load/store, struct layout), sums
(`EConstruct`, `EMatch`), `EIndex`, bit-structs, trait objects (`EMakeDyn` /
`EDynDispatch`), function pointers of Coil functions (`EFnPtrOf` / `ECallPtr`),
inline LLVM IR (`ELlvmIr`), float remainder, variadic FFI, and unsigned-narrow
logical right shift. These are the natural next extensions: add struct/sum layout
(port codegen_a64's `g-natural-layout!` / sum `{i32 tag, [words] payload}`
model), aggregate copy load/store, and the corresponding `compile-expr` arms and
opcodes, keeping this document in sync.

## Gate

`selfhost/oracle/interp/gate-interp.sh <coil-bin>` runs `coil interp` over
`selfhost/oracle/arm64/corpus.txt` and diffs stdout + exit code against the LLVM
reference snapshots in `selfhost/oracle/arm64/reference` — the same snapshots the
arm64 backend gate uses. Runtime equality with the compiled program is the
contract. As the interpreter grows toward the whole language, this number climbs
toward the arm64 gate's.
