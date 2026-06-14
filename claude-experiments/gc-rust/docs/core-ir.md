# Core IR — the monomorphic, typed contract between the middle-end and codegen

The core IR is what the type checker + monomorphizer produce and what codegen
consumes. It is the single frozen interface that lets Phase 2 and Phase 3 be
built in parallel.

## Design principles

1. **Fully monomorphic.** No type variables, no generics, no trait bounds. Every
   function is a concrete `(arg reprs) -> ret repr`. `id<T>` instantiated at
   `i64` and at `String` are *two different* core functions with mangled names.
2. **Explicit representation.** Every type is a `Repr` that says exactly how it
   is laid out: a scalar (with width + signedness), an inline value aggregate, a
   GC reference (pointer to a heap object of a known `Layout`), or unit. There is
   no "uniform boxed" anything.
3. **Explicit GC shape.** Every reference type carries a `Layout` describing its
   heap object: which fields are GC pointers (traced) vs raw scalars (untraced).
   This is what gets lowered to a `gc::TypeInfo` + `type_id`. Codegen never has
   to *infer* what's a pointer — the IR already says.
4. **Desugared control flow.** `?`, `for`, method calls, compound assignment,
   operator traits → explicit calls / matches / loops. `if`/`match` remain (they
   lower naturally to LLVM branches/switches), but everything else is lowered.
5. **Closures lifted.** Each closure becomes a top-level function plus an
   explicit environment record (a reference type) whose `Layout` classifies each
   capture as pointer or scalar. Codegen just builds the env object and calls.

## Repr — how a value is represented

```rust
enum Repr {
    Unit,                                   // zero-size, no storage
    Scalar(ScalarRepr),                     // i8..u64, f32/f64, bool, char
    Value(ValueId),                         // inline value aggregate (struct/enum)
    Ref(LayoutId),                          // GC pointer to a heap object
}

enum ScalarRepr { I8, I16, I32, I64, U8, U16, U32, U64, F32, F64, Bool, Char }
```

- `Scalar` lowers to the obvious LLVM int/float; signedness drives `sdiv/udiv`,
  `ashr/lshr`, signed/unsigned compares, and sext/zext on widening casts.
- `Value(id)` lowers to an LLVM aggregate (`{...}` or a tagged union for value
  enums). Stored inline in fields/arrays/locals; never independently heap-boxed.
  Passed by value (LLVM byval / split into registers) at call boundaries.
- `Ref(id)` lowers to `ptr`. Points at a heap object whose layout is `id`. These
  are the GC roots codegen must spill before allocations/calls.

## Value aggregate layout (inline, unboxed)

```rust
struct ValueLayout {
    fields: Vec<Repr>,        // struct fields, in order
    // for value enums:
    variants: Option<Vec<ValueVariant>>, // tag + per-variant payload reprs
    size: u32, align: u32,
}
```

A value enum is `{ tag: iN, payload: union(max variant) }`, niche-packed later.
A value type may transitively contain `Ref`s (e.g. `value struct Span { s: String }`)
— those nested refs are still traced when the value lives inside a heap object,
because the *containing* reference object's `Layout` flattens them into its own
pointer-field list.

## Heap object layout (reference types) → gc::TypeInfo

```rust
struct Layout {
    // GC-traced pointer slots come FIRST (matches gc::TypeInfo value fields),
    // then raw scalar bytes, then optional varlen tail.
    ptr_fields: u16,          // number of leading 8-byte GC pointer slots
    raw_bytes: u16,           // untraced scalar bytes after the pointer slots
    varlen: VarLen,           // None | Values(ptr elems) | Bytes
    // source-level field order → (section, offset) so field access can index.
    field_map: Vec<FieldLoc>,
    name: String,             // for FrameOrigin/debug + mangling
}
enum VarLen { None, Values, Bytes }
enum FieldLoc { Ptr(u16), Raw{offset:u16, repr:ScalarRepr} }
```

This maps 1:1 onto `gc::TypeInfo::for_header(Full::SIZE).with_fields(ptr_fields)
.with_raw_bytes(raw_bytes)[.with_varlen_*]`. The monomorphizer assigns each
distinct reachable `Layout` a stable `type_id` (its index in the runtime type
table). **The pointer-fields-first discipline is mandatory** — it's how the
inherited scanner (`gc::scan_object`) knows which slots to trace.

Reference structs/enums, closure environments, `Vec<T>` backing arrays,
`String`, and boxed nodes are all `Layout`s. A reference enum is a heap object
whose first word is the tag (raw) followed by the active variant's fields; all
variants share one `type_id` per enum (the scanner uses a variant-aware shape) —
OR each variant gets its own `type_id` (simpler; matches ai-lang). v0 choice:
**one `type_id` per (enum, variant)** — uniform with ai-lang, lets the scanner
stay variant-free.

## Functions, items, program

```rust
struct CoreProgram {
    funcs: Vec<CoreFn>,          // all monomorphic, name-mangled
    layouts: Vec<Layout>,        // every reachable heap shape; index = type_id
    values: Vec<ValueLayout>,    // every reachable inline aggregate
    entry: FuncId,               // monomorphized `main`
}

struct CoreFn {
    name: String,                // mangled: base + instantiation hash
    params: Vec<Repr>,
    ret: Repr,
    body: CoreBlock,
    // GC frame info is computed by codegen from the Refs that are live across
    // allocation/call points; the IR just provides typed locals.
}
```

## Expressions (typed, desugared)

```rust
enum CoreExpr {
    // literals
    ConstInt(u64, ScalarRepr), ConstFloat(f64, ScalarRepr),
    ConstBool(bool), ConstChar(char), ConstStr(String /* -> Ref(String) */),
    Unit,
    Local(LocalId),
    // arithmetic / compare — repr carries width+signedness
    Bin(BinOp, Repr, Box<CoreExpr>, Box<CoreExpr>),
    Un(UnOp, Repr, Box<CoreExpr>),
    Cast { value: Box<CoreExpr>, from: Repr, to: Repr },   // numeric/sext/zext/trunc/fp
    // direct call to a known monomorphic function
    Call(FuncId, Vec<CoreExpr>),
    // indirect call through a closure value: (env, code_ptr)
    CallClosure { callee: Box<CoreExpr>, args: Vec<CoreExpr> },
    // build a closure: env layout + captures + code function
    MakeClosure { code: FuncId, env: LayoutId, captures: Vec<CoreExpr> },
    // reference-type construction → heap alloc + field stores
    New { layout: LayoutId, fields: Vec<CoreExpr> },
    // value-type construction → inline aggregate
    MakeValue { value: ValueId, fields: Vec<CoreExpr> },
    // field access: by (section, offset) resolved from Layout/ValueLayout
    Field { base: Box<CoreExpr>, loc: FieldLoc },
    // enum construction + match
    MakeVariant { layout: LayoutId, tag: u32, fields: Vec<CoreExpr> },
    Match { scrutinee: Box<CoreExpr>, arms: Vec<CoreArm> },  // exhaustive, tag switch
    If(Box<CoreExpr>, Box<CoreBlock>, Box<CoreBlock>),
    Block(Box<CoreBlock>),
    // loops (for/while/loop all desugar to Loop + Break)
    Loop(Box<CoreBlock>), Break(Option<Box<CoreExpr>>), Continue,
    Return(Option<Box<CoreExpr>>),
    Assign { local: LocalId, value: Box<CoreExpr> },
}

struct CoreBlock { stmts: Vec<CoreStmt>, tail: Option<CoreExpr> }
enum CoreStmt { Let(LocalId, CoreExpr), Expr(CoreExpr) }
struct CoreArm { tag: u32, binds: Vec<LocalId>, body: CoreExpr }
```

Notes:
- **`?`** desugars to a `Match` on the `Result`/`Option` that returns `Err/None`
  early (a `Return`) and binds `Ok/Some` payload otherwise.
- **method calls** + **operator traits** resolve to a concrete `Call(FuncId, …)`
  during monomorphization (the impl is selected from the receiver type).
- **`for x in it`** desugars to `let mut i = it.into_iter(); loop { match i.next()
  { Some(x) => …, None => break } }` once iterators exist; until then `for` over
  ranges desugars to an integer-counter loop.

## Mangling

`name :: <inst-types>` → `name$<blake3(inst reprs)[..8]>`. Deterministic, stable
across builds, so the bitcode cache (later) keys cleanly.

## What codegen does with this (Phase 3)

- One LLVM function per `CoreFn`. Params/ret lowered by `Repr`.
- `New`/`MakeVariant`/`MakeClosure`/`ConstStr` → `ai_gc_alloc_*(thread, type_id)`
  then field stores; spill live `Ref` locals to the frame first.
- Frame: count `Ref`-typed locals/temps live across an alloc/call → that's
  `num_roots`; emit prologue/finalize/epilogue per `docs/gc.md`.
- `Match` → load tag (raw word for ref enums, extract for value enums) → switch.
- Loop back-edges → `emit_safepoint_poll`.
```
