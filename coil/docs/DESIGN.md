# Coil — Calling Convention & Allocation as Types

> **Working name:** *Coil* (provisional — evokes low-level winding/control and "assembling"). Bikeshed later.
>
> **Status:** Design sketch v0. No implementation yet. This document specifies the
> two type-system moves that define the language and the honest strategy for
> lowering them to **raw LLVM**.

---

## 1. Thesis

A low-level language where the two facts a normal compiler keeps to itself —
**where values live across a call** (calling convention) and **where data lives
in memory** (allocation) — are made **explicit, first-class, and checked by the
type system**. You get assembly-level control over both, and you recover
ergonomics through **Lisp-style macros** that build higher-level constructs on
top of the primitives.

The single design bet that justifies doing *both* at once:

> **A closure is not a primitive.** It is `(code-pointer-with-a-calling-convention,
> environment-with-an-allocation)`. Boxed-env, inline-env, GC'd-env, and
> stack-only-non-escaping closures are just different points in the
> *(calling-convention × allocation)* space. The same is true of coroutines,
> async frames, vtables, and trampolines. If the type system can talk about
> conventions and allocations, these stop being built-in language features and
> become **library code written in macros.**

That is the whole point of the language: a minimal, fully-explicit machine-level
core, plus a macro system powerful enough that "high-level" features are
*derived*, not blessed.

---

## 2. Design center & non-goals

**Center**

- Total control over calling convention, register placement, stack discipline,
  and clobber/preserve sets — expressed in source, enforced by the checker.
- Total control over allocation — region/allocator is part of a pointer's type.
- A homoiconic Lisp surface where almost everything lowers to explicit IR, and
  macros build the rest. (Same philosophy as the `mlir-lisp` experiment, but with
  a real type system layered on, and **LLVM** as the backend.)

**Non-goals (v0)**

- *Not* memory-safe by construction. We choose **descriptive-but-checked**
  placement (see §5.3), not full linear/borrow soundness. Macros can build safer
  layers on top.
- *Not* a GC language. (That is the `codex-experiments/lang` direction; this is
  deliberately the opposite end.)
- *Not* source-portable across architectures for code that names physical
  registers. Register-level types are arch-specific by nature; portability is a
  macro/abstraction concern (§4.2, §8), not a core guarantee.
- *Not* trying to beat LLVM's optimizer. When a convention coincides with one
  LLVM knows, we ride its optimizer; when it doesn't, we accept a shim (§7).

---

## 3. Substrate: raw LLVM + a Lisp front end

```
  .coil source (s-expr)
        │  reader
        ▼
  AST  ──►  macro expander (compile-time eval)  ──►  expanded AST
        │
        ▼
  type checker  (bidirectional; ABI + region aware)   ──►  typed core
        │
        ▼
  lowering      (per-convention strategy; §7)
        │
        ▼
  LLVM IR  (via inkwell)  ──►  native object file  ──►  cc/ld  ──►  executable
```

- **Backend:** raw LLVM through `inkwell`. Host/implementation language: Rust.
- **AOT, no JIT.** Coil emits a native object via LLVM's `TargetMachine` and
  links with the system `cc`. This is the right model for a language about
  total ABI/memory control: no runtime LLVM dependency, real linker, C/asm
  interop at link time, and `:shim` trampolines become ordinary relocations the
  assembler/linker resolves. There is no `eval`/JIT — the only way to run a
  program is to compile it.
- **Comptime is a tree-walking interpreter, not a JIT.** Macro expansion runs
  the compile-time Lisp (see Macros, below) with no dependency on LLVM, and
  `coil build` expands every macro to a fixpoint *before* any code is emitted.
  So the compile-time and run-time phases are cleanly separated and the JIT
  never enters the picture.
- **Reader:** s-expressions. Homoiconic so macros are ordinary tree transforms.
- **Compile-time evaluation:** macros run in a compile-time interpreter that can
  inspect types, target data layout, and convention descriptions (Terra/Zig-style
  staging). Conventions and layouts are *first-class compile-time values*.
- The core is tiny: literals, `let`, control flow, `op`/intrinsics, function
  definition, and the two type-system features below. Everything else
  (`struct`, `match`, closures, vtables, calling-convention adapters) is macros.

---

## 4. Type-system move #1 — Calling convention as type

### 4.1 What a function type *is*

A function type is not `(i64) -> i64`. It is a full ABI:

```
FnType :=
  { params  : [ (Type, Location) ]      ; each parameter's type AND where it lives
    ret      : (Type, Location)          ; return type AND where it comes back
    clobber  : RegSet                     ; registers the callee may destroy
    preserve : RegSet                     ; registers the callee guarantees to keep
    stack    : StackDiscipline }          ; alignment, red-zone, who cleans up, ...
```

Two functions with the *same logical signature* but *different ABI* have
**different types**. A function pointer carries its full `FnType`. This is the
load-bearing decision: it makes mismatched calls a type error and makes adapters
(§4.5) first-class.

### 4.2 The location language

`Location` says where a value sits at the call boundary:

```lisp
@rdi              ; a named physical GPR
@xmm0             ; a named vector/FP register
@(stack +16)      ; a stack slot at a frame offset
@(pair rdx rax)   ; a value split across two registers (e.g. 128-bit return)
@(reg-class gpr)  ; "any GPR" — the convention/allocator picks; abstract over arch
@implicit         ; passed by the convention's own rule (e.g. sret pointer)
```

Physical-register locations are inherently arch-specific. `@(reg-class …)` is the
portable escape hatch: a convention can place "the first integer arg" without the
source naming `rdi` vs `x0`. Targets are compile-time values, so a macro can
select the right physical mapping per arch (§8).

### 4.3 Conventions are data: `defcc`

You define a convention once, as a compile-time value, then apply it by name:

```lisp
(defcc sysv-min                 ; a hand-rolled minimal System V-ish convention
  :params  [@rdi @rsi @rdx @rcx @r8 @r9]   ; integer args, then spill to stack
  :fp      [@xmm0 .. @xmm7]
  :ret     @rax
  :clobber #{rax rcx rdx rsi rdi r8 r9 r10 r11 xmm0..xmm15}
  :preserve #{rbx rbp r12 r13 r14 r15 rsp}
  :stack   {:align 16 :red-zone 128 :cleanup :caller})

(defcc fast2                    ; a custom two-register fastcall of our own design
  :params  [@rax @rdx]
  :ret     @rax
  :clobber #{rax rdx rcx}
  :preserve #{rbx rbp rsp r12 r13 r14 r15}
  :stack   {:align 16 :cleanup :caller})
```

A function declares which convention it uses; parameter locations may be taken
from the convention or overridden per-parameter:

```lisp
(defn fib :cc sysv-min
  [(n :i64)]            ; location defaulted from sysv-min → @rdi
  (-> :i64)             ; → @rax
  ...)

(defn raw-syscall :cc (raw :params [@rax @rdi @rsi @rdx @r10 @r8 @r9] :ret @rax
                           :clobber #{rcx r11} :preserve :all-else)
  [(num :i64) (a :i64) (b :i64) (c :i64)]
  (-> :i64)
  ...)
```

### 4.4 Typing rule for a call

A call site `(call f a b …)` checks:

1. **Arity & element types** of `params` against the arguments (bidirectional;
   §9).
2. **Convention identity / structural match** between the callee's `FnType` and
   the convention expected at the call site (function pointers carry it).
3. The caller's **live registers** at the call must be disjoint from the callee's
   `clobber` set, *or* spilled. The checker can surface "you have a live value in
   `r10`, which `f` clobbers."

A mismatch is a **type error**, not undefined behavior. The repair is an adapter.

### 4.5 Adapters / trampolines (the first macro payoff)

An **adapter** is a function whose job is to convert between two conventions. Its
type is literally `FnType_A -> FnType_B`. Because conventions are data, a macro
can *synthesize* the adapter body — the register/stack shuffle — from the two
descriptions:

```lisp
;; want to call a sysv-min function through a fast2 pointer:
(def f-fast (adapt my-sysv-fn :from sysv-min :to fast2))
;; `adapt` is a macro: it reads both conventions, computes the moves
;; (e.g. mov rdi, rax ; mov rsi, rdx ; call ; mov rax, rax), and emits a
;; naked thunk with the fast2 signature.
```

This is the mechanism behind FFI correctness, hand-written-asm interop, JIT
trampolines, and `extern` declarations — all one macro over convention data.

---

## 5. Type-system move #2 — Allocation as type

### 5.1 Pointers carry a region/allocator

A pointer type is parameterized by **where its target lives** and **how long**:

```
Ptr<T, R>       ; T = pointee type, R = region/allocator + lifetime
```

```lisp
(Ptr i64 'frame)          ; on the current stack frame; may not escape it
(Ptr Node 'arena)         ; in an arena/region; freed in bulk when arena dies
(Ptr Buf  malloc)         ; individually heap-allocated; must be freed via `free`
(Ptr T    'static)        ; module-static storage; lives forever
(Ptr T    (in r12))       ; lives in a register (no address-of) — degenerate region
```

### 5.2 Regions, allocators, escape

- A **region** is a named scope with a lifetime (`'frame`, `'arena`, user-defined).
  Pointers into a region cannot outlive it — assigning a `(Ptr T 'frame)` into a
  longer-lived location is a type error (escape check).
- An **allocator** is a compile-time value (à la Zig, but lifted to the type
  level) that owns an alloc/free pair. `Ptr<T, malloc>` and `Ptr<T, my_pool>` are
  *distinct types*; you cannot `free` a pool pointer with libc `free`.
- Allocators can be passed to functions, so polymorphism over allocation is just
  ordinary parameterization:

```lisp
(defn make-node [A:Allocator (val :i64)] (-> (Ptr Node A))
  (let [p (alloc A Node)]
    (set! (-> p val) val)
    p))
```

### 5.3 Soundness stance: *descriptive-but-checked* (a deliberate choice)

Full static memory safety here means rebuilding a borrow/region checker — the
exact complexity the GC-lang experiment fled from. For a "complete control"
language we choose a middle path:

- Region/allocator is in the type and **drives codegen** (placement, which free).
- **Escape analysis is checked** for the clear cases (region pointer stored
  somewhere longer-lived → error). 
- We do **not** promise linear-types soundness (no whole-program use-after-free
  proof). Double-free / leak are *possible* but typed surfaces make them
  diagnosable, and **macros can build sound layers** (linear `Owned<T>`, RAII
  `defer`, ref-counting) on top of the typed primitive.

This keeps "truly complete control" while adding guardrails, and leaves the
safety/ergonomics ceiling open to library design rather than baking one policy in.

### 5.4 Drop / free

`free`/`drop` is convention-and-region aware: `(free A p)` requires `p :
(Ptr T A)`. Region pointers (`'arena`, `'frame`) reject individual `free` — they
are released by region teardown. A `defer`/scope macro can emit the teardown.

---

## 6. The payoff: closures (and friends) are *derived*

With both features, a closure is constructed, not built-in:

```lisp
;; A closure value = (code ptr with convention C, env ptr with allocation R)
(deftype (Closure C R Env)
  (struct
    (code (FnPtr C))          ; convention C decides how the env is threaded in
    (env  (Ptr Env R))))      ; allocation R decides where the captured state lives
```

Different closure *representations* are different instantiations:

| Representation                 | `C` (convention)                 | `R` (allocation) |
|--------------------------------|----------------------------------|------------------|
| Classic boxed closure          | env via `nest`/static-chain ptr  | `malloc`/heap    |
| Stack closure (non-escaping)   | env via hidden first arg         | `'frame`         |
| Arena closure                  | env via hidden first arg         | `'arena`         |
| Register closure (tiny env)    | env captured in a pinned reg     | `(in rXX)`       |

The `nest` parameter attribute in LLVM (§7.5) exists precisely for the
static-chain case — so the "classic" representation lowers cleanly. The others
fall out by choosing `C` and `R`. Coroutine frames, vtables (a `Ptr` to a
`'static` table of `FnPtr`s), and async state machines are the same story.

---

## 7. Lowering to LLVM (the honest part)

**The constraint you must design around:** LLVM's calling conventions are a
**closed enum** baked into the backend (`ccc`, `fastcc`, `coldcc`, `tailcc`,
`swiftcc`, `swifttailcc`, `ghccc`, `preserve_most/all/none`, the x86 ones, …).
You **cannot** write "arg in `r9`, return in `r12`, clobber only `r10`" in pure
IR. So the type system describes conventions abstractly, and **each convention
carries a lowering strategy** that maps it onto something LLVM can emit.

### 7.1 What LLVM *does* give us

- Per-arg attributes: `inreg`, `byval`, `sret`, `signext`/`zeroext`, `align`,
  `noalias`, `preallocated`, and crucially **`nest`** (static-chain pointer —
  built for closures, §6).
- `musttail` — guaranteed tail calls (required to lower some conventions and
  trampolines faithfully).
- **`anyregcc` + patchpoints/statepoints** — JIT machinery that lets a *call site*
  pin values into registers. This is the closest LLVM comes to "I decide register
  placement," and it is exactly what JITs use for GC and register pinning.

### 7.2 Per-convention lowering strategies

Every `defcc` resolves, at lowering time, to one of:

1. **`:native <llvm-cc>`** — the convention coincides with one LLVM knows
   (e.g. our `sysv-min` ≈ `ccc` on x86-64 SysV). Emit a normal LLVM function with
   that CC and the right per-arg attributes. **Full optimization, zero overhead.**
   This is the common, fast path; design the standard conventions to land here.

2. **`:shim`** — the convention is exotic (custom register set / clobber mask).
   Emit a **`naked`** function plus **module-level inline asm** (or
   `llvm.inline_asm`) that performs the exact register/stack discipline, wrapping
   an inner `ccc` function that holds the real body. Costs a shim; gives bit-exact
   ABI. This is also how `adapt` (§4.5) lowers.

3. **`:patchpoint`** — use `anyregcc` + a patchpoint so the call site pins
   arguments to the requested registers. Good for JIT-time conventions and for
   conventions defined relative to "wherever these values already are."

4. **`:patched-llvm`** — (escape hatch, discouraged) add a real CC to an LLVM
   fork, Swift/GHC-style. Maximum fidelity and optimization, but ties you to a
   fork. Kept in the design as a known option, not a v0 dependency.

The type system is the source of truth; the strategy is per-convention metadata.
When you write a convention that *happens* to match LLVM, you pay nothing; when
you go off-road, you pay a shim — and you knew that from the convention's
declared strategy.

### 7.3 Shim shape (sketch)

```
; a `:shim` convention `fast2` wrapping body __fib_impl (ccc)
define void @fib_fast2() naked {
  ; inline asm: marshal @rax,@rdx (fast2) -> rdi,rsi (ccc), call, marshal back
  call void asm sideeffect
    "mov %rdi, %rax\0A mov %rsi, %rdx\0A call __fib_impl\0A ret",
    "~{...clobbers...}"()
  unreachable
}
```

The marshalling string is *generated from the convention data* — never
hand-written per function.

### 7.4 Allocation lowering

- `'frame` → `alloca` (escape-checked so it never leaks past the frame).
- `'static` → an LLVM global.
- `malloc`/custom allocator → calls to the allocator's declared alloc/free
  symbols; the allocator value carries the symbol names.
- `'arena` → bump-pointer over a region object; teardown frees the block.
- `(in rXX)` → keep the value in a vreg / pinned physical reg; address-of is a
  type error.

---

## 8. The macro layer

Macros are where ergonomics live. Because conventions, layouts, allocators, and
the target are all **compile-time values**, macros can compute over them:

- `defcc` / `adapt` — conventions as data; synthesize trampolines (§4.5).
- `extern` — declare a foreign function with a convention; calls are checked.
- `struct` / `union` / `enum` — layout computed from target `DataLayout`;
  `repr(C)` vs packed vs niche are macro choices, not built-ins.
- `defclosure`, `vtable`, `defer`/RAII, `coroutine` — derived per §6.
- `for-arch` — select physical register mappings per target so a convention
  written with `@(reg-class …)` resolves correctly on x86-64 vs aarch64.
- Higher-level languages: you can grow a "normal" surface (the way `mlir-lisp`
  grows `+`, `if`, `defun` as macros) without the core knowing about it.

Hygiene: gensym'd bindings by default; explicit capture when a macro *wants* to
talk about a physical register or convention by name.

---

## 9. Type checking approach

**Bidirectional** (synthesize/check), matching the `codex-experiments/lang`
direction and avoiding global HM inference:

- Function definitions require full parameter + return types *and* a convention.
- Locations default from the convention; per-parameter overrides are checked
  against the convention's register classes (you can't put two integer args in
  the same physical reg).
- Region/lifetime is checked structurally; escape is a check-mode failure.
- Allocator identity is nominal (`malloc` ≠ `my_pool`).
- Calls check convention match + clobber/live-set disjointness (§4.4).

---

## 10. Worked example (end to end, aspirational)

```lisp
(defcc sysv-min :params [@rdi @rsi @rdx @rcx @r8 @r9] :ret @rax
  :clobber #{rax rcx rdx rsi rdi r8 r9 r10 r11}
  :preserve #{rbx rbp r12 r13 r14 r15 rsp}
  :stack {:align 16} :lower :native ccc)

(defn add :cc sysv-min [(a :i64) (b :i64)] (-> :i64)
  (iadd a b))                       ; a@rdi, b@rsi, result@rax

(defcc fast2 :params [@rax @rdx] :ret @rax
  :clobber #{rax rdx rcx} :preserve :all-else
  :stack {:align 16} :lower :shim)  ; off-road → naked+asm thunk

(def add-fast (adapt add :from sysv-min :to fast2))   ; synthesized trampoline

(defn use [] (-> :i64 :cc sysv-min)
  (call add-fast 2 3))              ; checked: fast2 args pinned to rax,rdx
```

Lowering: `add` → `ccc` function; `add-fast` → `naked` thunk marshalling
`rax,rdx → rdi,rsi`; `use` → `ccc` function calling the thunk.

---

## 11. Open questions

1. **Convention equality:** nominal (by `defcc` name) or structural (by the
   register/stack tuple)? Structural enables more adapter inference; nominal is
   simpler and gives better errors. *Leaning structural with named aliases.*
2. **Clobber/live-set checking cost:** full register liveness at every call is
   real analysis. v0 could check only *declared* live registers and defer full
   liveness to a later tier.
3. **How abstract can `@(reg-class …)` get** before it stops being "complete
   control"? Where's the line between portable convention and arch-specific one?
4. **Soundness ceiling for allocation:** how much of a linear/`Owned<T>` layer do
   we ship as blessed library vs leave to users?
5. **Vectors/aggregates by value** across custom conventions: defer to LLVM's
   rules on the `:native` path, but `:shim` conventions need explicit
   split/spill rules in the convention data.
6. **Debug info / unwinding** for `:shim` (naked) functions — CFI is manual.
7. **Reader & macro evaluator**: reuse the `mlir-lisp` reader, or fresh?

---

## 12. Roadmap (suggested)

- **M0 — Reader + core. ✅ done.** s-expr reader, AST, `defn`/`let`/arith/
  `if`/calls lowering to LLVM via `inkwell`, JIT runner. (`fib.coil` → 55.)
- **M1 — Convention types. ✅ done.** `defcc`; each function's convention sets
  the real LLVM calling convention on the function and every call site
  (`:native c|fast|cold`). Checker rejects arity/unbound/ill-formed conventions.
- **M2 — Shim lowering. ✅ done.** `:lower shim` for register layouts LLVM's CC
  enum can't express: a `ccc` `__impl` body + a `naked` trampoline marshalling
  the convention's registers ↔ SysV, and register-constrained inline-asm call
  sites. Verified through JIT, including recursion through the exotic ABI
  (`shim.coil` → 42; `fact 5` → 120). *Core thesis demonstrated.*
  - **Remaining for M2:** the `adapt` macro (general convention-to-convention
    trampolines, §4.5) and a safe parallel-move schedule in the trampoline (the
    current marshaller assumes non-colliding source/dest registers). Both wait
    on the macro layer / a small register-move solver.
- **M3 — Allocation types. ✅ done.** A pointer's region is part of its type:
  `(ptr frame)` → `alloca`, `(ptr static)` → a global, `(ptr heap)` →
  `malloc`/`free`. `alloc`/`load`/`store!`/`free`, with a real bidirectional
  type checker (the language is no longer i64-only). Region soundness checked:
  `frame` pointers can't cross a function boundary; `free` only accepts `heap`.
  (`allocation.coil` → 42.)
  - **Remaining for M3:** richer pointee types (pointers to pointers/structs),
    user-defined allocators as values (§5.1), arena regions with bulk teardown,
    and `defer`/RAII (a macro, once the macro layer exists).
- **Macros — ✅ done (engine).** `defmacro`/`def` evaluated by a compile-time
  Lisp interpreter (quasiquote/unquote/splicing, `gensym`, list & symbol
  builtins, lambdas, recursion, helper functions). Macros compute and can emit
  whole top-level definitions (a top-level `(do ...)` is spliced). They compose
  with conventions and regions. Pipeline: `read → expand → parse → check →
  codegen`; inspect with `--expand`. (`macros.coil` → 41.) *The "Lisp-like
  macros" half of the pitch.* Remaining: hygiene is `gensym`-based (not
  automatic), and there is no module/`import` system yet.
- **Function pointers — ✅ done.** `(fnptr CC [types] ret)` type, `(fnptr-of
  name)` for a function's address, indirect `(call-ptr fp args...)` honoring the
  convention (native conventions only — shim fnptrs are future). `cast` also
  reinterprets between pointer types (opaque-pointer no-op), enabling type-erased
  `void*` environments. Array-of-fnptr gives a vtable/dispatch table.
- **M4 — Closures (manual memory) — ✅ done.** A closure is **not** a primitive:
  it's a struct `{ code: fnptr-with-a-convention, env: ptr-with-a-region }`. The
  env region is the lifetime/ownership story — `heap` closures escape and are
  freed by hand; `frame` closures are non-escaping (the escape rule enforces it).
  `closure.coil` shows two heterogeneous heap closures behind one `Closure` type
  and one generic `apply`, allocated and freed manually → 42. Remaining: a
  `defclosure` macro to remove the boilerplate; the `nest`-register convention so
  the env is threaded out-of-band; downward `frame` closures (needs the borrow
  relaxation below).
- **`extern` + C interop — ✅ done.** `(extern name :cc c [types] (-> ret))`
  declares a foreign function's convention + signature; calls are type-checked
  and the symbol is resolved at link time. Pointer regions are erased at the
  boundary (the foreign side doesn't track them). Programs do real I/O
  (`putchar`/`write`/`puts`); `extern.coil` prints `12345`. Remaining: variadics
  (`printf`), and `:shim`-convention externs (calling hand-written asm through a
  custom register ABI — the call-site marshalling already exists from M2).
- **C types — ✅ done.** Integer widths `i8/i16/i32/i64`; typed pointers
  `(ptr REGION TYPE)` with a foreign `c` region (so `(ptr c (ptr c i8))` is
  `char**`); pointer indexing `(index p i)` (GEP); width `(cast :iN e)`
  (sext/trunc). `main` may take `(argc :i32) (argv (ptr c (ptr c i8)))`.
  Codegen threads each value's Coil type so loads/indexing use the right width
  and pointee. `args.coil` reads and echoes its command line.
- **Structs & arrays — ✅ done.** `(defstruct Name [(field :type) ...])` and
  `(array T N)`; `(alloc REGION TYPE)` allocates any type; `(field p name)` and
  array `(index p i)` produce element pointers (struct/array GEP); structs nest
  by value (built in definition order) or self-reference by pointer.
  `structs.coil` → 42. Remaining: unsigned types, struct/array literals, and
  passing aggregates by value across `:shim`/`extern` boundaries (ABI work).
- **M5 — Macro stdlib.** `struct`/`enum`/`vtable`/`adapt`/`defer`, a small
  "normal" surface grown entirely in macros on top of the typed core.

---

*Next concrete step after this doc: M0/M1 skeleton (Rust + inkwell), or drill
deeper into any one section (the adapter algorithm and the shim generator are the
most interesting to specify precisely).*
