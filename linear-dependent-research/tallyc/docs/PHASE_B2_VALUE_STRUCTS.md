# Phase B2 — flat multi-field structs BY VALUE (+ layout control)

**STATUS: CORE BUILT (steps 1–2 + small-struct FFI).** Flat records now compile
BY VALUE: `struct Point { x, y }` is two SSA registers — zero malloc, no tag
(IR-tested); it crosses `Fix` function boundaries as consecutive i64 params and
an `{i64 × w}` return (the two-register C convention); records nest (a record
field of record type flattens inline); implemented in `dep_codegen.rs`
(`Val::{Int,Agg}`, `record_shape`/`type_shape`, `expect_width`). The initial
pass auto-boxed records at generic boundaries; that is GONE — see the
zero-non-explicit-boxing paragraph below.
`%foreign` flattens record arguments, which on AArch64/SysV IS the by-value ABI
for a ≤2-register integer C struct — tested against a real C function taking
`struct { long long x, y; }` by value. The prelude's read-back pairs (`ARead`,
`Cell`, `CL`/`VL`, `Taken`) became flat records too: no malloc per read even at
-O0. An `Own` field inside a record keeps its linear accounting unchanged.

**ZERO NON-EXPLICIT BOXING (the follow-up pass, landed):** the boxed
representation of records is GONE — a record has exactly one representation,
flat — and with it every compiler-inserted malloc. What replaced each implicit
box:

- `Arr a n` is **real AoS**: element width from the erased type argument
  (per-instantiation, computed by `type_shape`'s parameter substitution),
  stride-w fills/loads/stores. `Own`/`valloc` payloads store INLINE at their
  width (`alloc` IS the explicit box — `Own Point` is one 16-byte cell).
- Generic FLAT records (`ARead a n`, `Taken a l`) carry their layout in the
  value: construction concatenates actual widths, the match splits flexibly
  (one flexible field absorbs the surplus; ambiguity is an error).
- Total-fold (`NatElim`) accumulators widen — one phi per component.
- Everything that genuinely cannot know a layout is a **guided hard error**,
  never a hidden box: a record into a generic field of a BOXED datatype
  (store `Own T`, or declare a concrete container — which inlines it flat); a
  record into an abstract-typed `%partial` parameter; a width-sensitive
  primitive (`unbox`/`aget`/`anew`/`aset`/`vread`/`vwrite`/`vtake`) over an
  ABSTRACT type inside a generic `%partial` fn (one compiled body cannot
  serve two layouts — specialize; non-`%partial` generics inline and are
  fine); a width-CHANGING `vwrite` (the cell was allocated at the old width);
  records as total-fold RESULTS of boxed-family/accumulator folds (make the
  fn `%partial`).

**The honest remainder:** `repr`/field-width/bit-field layout control (§4
below), and the one implicit allocation left in the LANGUAGE: the
multi-constructor / recursive datatype cell (the constructor-allocates model
predating Phase A). Making that explicit — enums as flat tagged-union VALUES,
recursion only through explicit `Own` — is FUTURE_WORK §5.1–5.2's Phase A and
is a surface-language redesign, not a codegen pass.

The original plan follows; the key deviation from it is that no separate "typed
lowering IR" was needed — the erased terms already carry enough type information
locally (`Case`/`Constr` carry the datatype name, `Fix` carries its Π telescope),
which is what made the two-representation scheme implementable in one pass.

## Where the line is today

- **Zero-cost already:** `%builtin Nat` (a machine i64), TRANSPARENT newtypes
  (single non-erased field — the value IS the field, `transparent_field`),
  nullary constructors (shared constants), `Arr a n` (one flat buffer), and
  everything multiplicty-0 (erased).
- **Boxed:** every other constructor application is a malloc'd cell
  `[tag, field…]` of i64 slots. A multi-field `struct Point { x, y }` costs a
  malloc per construction and a pointer chase per read.
- **What SoA buys meanwhile:** the struct-heavy NUMERIC loop is expressible flat
  today as a structure of arrays — one contiguous `Arr` per field
  (`examples/arr_soa.tal`), bare loads, C-parity. What SoA does not give you is
  a by-value aggregate crossing a function boundary in registers, AoS layout,
  or C-struct interop for `%foreign`.

## Why this is a REWRITE, not a feature

The whole backend (`dep_codegen.rs`) is value-monomorphic: **every runtime value
is one `IntValue` (i64)** — an integer, or a pointer disguised as one. `compile`
returns `IntValue`; the env is `Vec<Option<IntValue>>`; every generated helper
function (`Fix` bodies, eliminator helpers, acc-folds, convoy folds) has an
all-i64 signature. A by-value struct breaks the invariant everywhere at once: a
`Point` argument is TWO registers (or an LLVM `{i64, i64}` aggregate), so every
seam — env slots, helper signatures, call sites, phi nodes, constructor stores,
match loads — must know the VALUE'S TYPE.

And the backend is deliberately type-poor: it compiles kernel-checked, erased
terms and only ever inspects constructor names. The types exist upstream (the
elaborator/kernel had them all); they are simply not threaded into codegen.
Faking it without types (e.g. stack-allocating structs and passing the pointer)
is UNSOUND the moment a struct value outlives its frame — which codegen cannot
see without knowing what escapes. We do not ship that.

## The plan (in order)

1. **Typed lowering IR.** Between erasure and codegen, annotate each binder and
   each application with its RUNTIME LAYOUT: `Scalar` (i64 — everything today) or
   `Agg(n)` (n i64 fields, single-constructor, non-recursive, no linear fields —
   the "flat record" class; `transparent_field` generalizes to `flat_record`).
   The kernel is untouched; this is codegen-input metadata, computed by the
   elaborator which already knows every type.
2. **Value representation.** `enum Val { Int(IntValue), Agg(StructValue) }` in
   codegen; env slots hold `Val`. Helper-function signatures derive from binder
   layouts (an `Agg(2)` param is an LLVM `{i64, i64}`). Constructor application
   of a flat record = `insertvalue` chain (NO malloc); `match` = `extractvalue`s
   (no tag — single constructor). Passing/returning follows LLVM's aggregate
   rules (small aggregates in registers).
3. **Embedding.** A flat record stored INTO a boxed cell or an `Arr` widens the
   slot count (the cell/array stride becomes `sum(field layouts)`); `aget`/`aset`
   of a record element load/store n slots. This is what makes `Arr Point n` a
   real AoS — and it must stay bounds-erased.
4. **Layout control.** Only after 1–3: `#[repr(...)]`-style pragmas fixing field
   order/width (i32/i16/i8 fields stop being "everything is i64"), explicit
   offsets, and THEN `%foreign` functions taking/returning C structs by value
   (the AAPCS64/SysV rules — coil's `abi_coerce` is prior art in the sibling
   language). Bit fields last, as sugar over masked loads (an mmio-style macro
   layer may be enough — see coil's `:bits`).
5. **Proof obligations.** The IR zero-overhead suite extends per step: a flat
   record constructs with zero mallocs; `Arr Point` is one malloc of `n*16`; a
   record crossing a call is registers (no stack traffic at -O2); erasure holds.

## Order of value

(1)+(2) alone give locals/params/returns by value — the hot-loop win. (3) gives
AoS. (4) gives C interop. Estimate: (1)+(2) is a focused multi-session effort
touching most of `dep_codegen.rs`; (3) and (4) are each additional sessions.

## Non-goals

- No escape analysis / stack-allocation heuristics for BOXED types (LLVM already
  scalarizes non-escaping cells at -O2 — measured: the `ARead` pairs vanish).
- No change to the kernel, the QTT rules, or erasure semantics.
