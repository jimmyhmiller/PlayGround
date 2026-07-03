# Phase Scalars — C's primitive type ladder (sized ints + floats), safely

*The gap between tallyc and C is no longer the memory model (Phases A–D closed
that) — it is the **scalar vocabulary**. C has `u8/u16/u32/u64`, `i8/…/i64`,
`f32/f64`, and a dense-layout story for all of them. tallyc has exactly one
runtime number: `Nat`, lowered to `i64`. This document designs the primitive
scalar family and its layout, keeping the full dependent-linear safety story.*

## 0. Diagnosis — why everything is `i64` today

`Nat` does **double duty**, and that is the whole problem:

1. It is the **index domain** — the *total, solver-decided* fragment: `Vec a n`,
   `Fin n`, `Lt i n`, `Vec (m+n) ≡ Vec (n+m)`. This must stay packed-`i64` with
   unary *semantics* and must **reduce inside types** (`src/solver.rs`).
2. It is the **runtime machine integer** — every native op is injected as
   `postulate mul : Nat -> Nat -> Nat` (`rust_surface.rs` ~:4526) and lowered
   against a single hardcoded `self.i64t` (`dep_codegen.rs`).

The representation bakes "one scalar type" in three places:

- `Val` is only `Int(IntValue) | Agg(Vec<IntValue>)` — no float, no width,
  no signedness (`dep_codegen.rs:516`).
- `type_width()` returns **slot counts** (8-byte units); `Arr` stride is literally
  `w * 8` (`dep_codegen.rs:1928`), so `Arr U8 n` cannot be a byte buffer.
- native ops are the `Nat -> Nat -> Nat` prelude block, all unsigned/monus.

## 1. Design principle — split the two roles along the total/partial line

Keep `Nat` as the **index domain only** (untouched: total, packed, solver-decided,
`{0}`-fragment). Add a **primitive scalar family** as the *runtime* numbers —
`ω`-fragment, **kernel-opaque** values, exactly like the native ops already are:

```
I8  I16  I32  I64      -- signed machine integers, two's-complement WRAPPING
U8  U16  U32  U64      -- unsigned machine integers, WRAPPING
Bool                   -- i1
F32  F64               -- IEEE-754 floats            (Phase 2)
```

Each is an **opaque type constant** (a postulate `Type`, like `Str`/`Own` —
kernel never reduces it) carrying a codegen descriptor `Scalar { bits, signed,
is_float }`. Machine-int semantics deliberately **differ from `Nat`**: `sub`
wraps (two's complement) rather than monus; `div`/`cmp`/`shr` are signedness-
directed. So they are genuinely different types with their own op set — not an
overload of the Nat ops.

**Safety is unchanged and free.** Bounds stay `Fin`/`Lt` in the erased `Nat`
index. Scalars are ordinary `ω` data. Linearity and erasure are
representation-independent (already true for `Own` inside records). No new
trusted code: the ops are opaque postulates the kernel re-checks structurally,
codegen dispatches on the operand representation.

## 2. Codegen representation change

```rust
struct Scalar { bits: u8, signed: bool, is_float: bool }   // NEW

enum Val<'c> {
    Int(IntValue<'c>),                 // Nat: i64, unsigned/monus — UNCHANGED
    Scalar(BasicValueEnum<'c>, Scalar),// NEW: a machine scalar, self-describing
    Agg(Vec<Field<'c>>),               // records; Field carries its own Scalar
}
```

Width comes from the LLVM value; **signedness/float-ness** must be tracked
because LLVM ints are sign-agnostic — hence the `Scalar` tag rides on the value.

### 2.1 Layout — from slot-counts to bytes

`type_width` (slots) → `layout(t) -> Layout { size, align, fields }` where a
scalar field is `{ offset, Scalar }`. `Arr a n`:

- size = `n * sizeof(a)` (a real `malloc(n)` for `Arr U8 n`);
- `aget i` = `gep (iN) base i` → `load iN` → widen to the working register
  (zext/sext by signedness) — or return a `Scalar` directly;
- `aset i v` = truncate `v` to `iN` → `store iN`.

Records-in-arrays keep AoS but at the **C offset table**, not 8-byte slots, so
`Arr Point n` with `Point{ x:I32, y:I32 }` is stride 8, not 16.

## 3. Surface & ops

- **Typed literals**: `0u8`, `255u8`, `-1i32`, `3.14f64` (Phase 2). A bare
  literal stays `Nat` (index domain) — no silent retyping.
- **Ops as kernel-opaque postulates**, one machine-scalar set with C semantics,
  polymorphic over an erased scalar-type implicit inferred from the operands;
  codegen recovers the `Scalar` from the operand `Val` (not from the erased
  type). `mul` on a non-scalar type is a guided compile error.
- **Casts** — the honest, explicit conversions (no implicit promotion):
  `cast : {0 a b : Scalar} -> a -> b` lowering to `trunc`/`zext`/`sext`
  (and `fptosi`/`sitofp`/`fptrunc`/`fpext` in Phase 2), chosen by the two
  descriptors. `a`,`b` inferred from context/annotation.

## 4. Slices (each lands with a C-twin benchmark, project cadence)

- **S1 — sized-element arrays. DONE (v2.4).** `Arr U8`…`Arr I64` are dense
  buffers at true byte width (`Arr U8 n` = `malloc(n)`, stride 1): `anew`/`aget`/
  `aset` truncate on store and widen on load (`zext` for `U*`, `sext` for `I*`),
  bounds still the erased `Lt` proof, buffer still linear. Scalar values are i64
  in registers with explicit `u8`/`nat_u8`-style conversions (C promotion made
  explicit). Proven byte-for-byte C-identical: `examples/bytes_bench.tal` vs
  `bench/bytes.c` compile to the same `<16 x i64>` SIMD reduce over `<16 x i8>`
  loads. IR tests: `bytes_arr_is_dense_u8_storage`, `..._signed_i16_sign_extends`.
  *Delivered: dense strings/buffers/parsing/hashing storage.*
- **S2 — first-class scalar values + C arithmetic. DONE (v2.5).** No `Val`
  variant was needed: values ride the i64 register (ints masked to width) and the
  width/signedness is recovered at each op from the erased spine type. `sadd`…
  `sshr`/`sneg`/`slt`/`sle`/`seq` with C wrapping + signedness; the universal
  `cast`; typed literals (`200u8`, `3.14`, `2.0f32`) via a bit-reinterpret
  postulate; type-annotated `let` to drive inference. `examples/scalars.tal`.
- **S3 — mixed-width record fields + FFI ABI.** C struct layout (`{i32,i32,f64}`
  at real offsets), `%foreign` at the true small-struct ABI.
- **S4 — floats (f32/f64). DONE (v2.5, minus the FFI FP-register ABI).** Floats
  ride the i64 register as their bit pattern, decoded only at the op — so no
  `Val` variant. `fadd`/`fsub`/`fmul`/`fdiv`/`fneg`, ordered `flt`/`fle`/`feq`,
  `f64_of_nat`/`nat_of_f64`, float literals, and `cast` for int↔float.
  `Arr F64/F32 n` are dense for free (S1). Measured C-parity: `float_bench.tal`
  vs `bench/float.c` → same ordered vectorized `fadd` reduction, 0.08s.
  *Remaining:* the FP-register by-value ABI so `%foreign` can call C math
  (`double sin(double)`) — currently floats cross FFI as i64 (GP register),
  wrong for the C ABI. Tracked in S3.

## 5. Trusted-base note

The `Scalar` ops and casts are `unsafe`-audited primitives (like `dlt`,
`%foreign`) — the C side of arithmetic. Everything above them is safe: erasure
still deletes every `0`, linearity still counts every `1`, bounds are still
`Fin`/`Lt`. The kernel re-checks the elaborated term; a wrong scalar op can only
be rejected, never make an unsound program pass.
