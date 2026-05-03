# 10 — Small `DynFunc` Helpers

## Implementation status: ✅ Implemented (DynFunc methods only)

- **Code**: `DynFunc::bool_not(v)`, `DynFunc::bit_eq(a, b)`, `DynFunc::nanbox_to_int(v)` in [`crates/dynlang/src/lib.rs`](../../crates/dynlang/src/lib.rs).
- **Tests**: 3 tests in dynlang `tests` module — `bool_not_inverts_falsey_and_truthy`, `bit_eq_returns_nanbox_bool`, `nanbox_to_int_truncates`. All run end-to-end through the interpreter.
- **Migration**: beagle deleted `Lowerer::bool_not`, `Lowerer::bit_eq`, and the free fn `nanbox_to_int`. Eight call sites switched from `self.X(f, ...)` to `f.X(...)`.
- **Not implemented**: `Lowered::run_main` + `MainArgs` enum (proposed in this doc). Beagle's 4-line arity match in `real_main` is fine in place; not worth abstracting.

## Problem

Several short patterns in beagle's lowering exist only because the
toolkit's primitives stop one level below the natural verb:

- **`bool_not(v)`** ([`lower.rs:1208-1213`](../../crates/beagle/src/lower.rs#L1208))
  — `is_falsey(v)` + `select(true, false)`. Three lines every site.
- **`bit_eq(a, b)`** ([`lower.rs:1201-1206`](../../crates/beagle/src/lower.rs#L1201))
  — `icmp(Eq) + select(true, false)`. Same shape; needed because `icmp`
  returns a backend-flavor predicate value, not a NanBox bool.
- **`nanbox_to_int(v)`** ([`lower.rs:2023-2026`](../../crates/beagle/src/lower.rs#L2023))
  — `unbox_number(v) + float_to_int`. Used at every array index.
- **`Lowered::invoke_main`** — the host hand-rolls the arity → args-vec
  match in `real_main` ([`main.rs:313-317`](../../crates/beagle/src/main.rs#L313)).
  `Lowered` knows the arity; let it own the call.

These are individually tiny but show up at every site that uses them.

## Proposed API

Add to `DynFunc`:

```rust
impl DynFunc {
    /// Logical not on a NanBox value. `v` is treated by the configured
    /// truthiness policy (see is_falsey); result is a NanBox bool.
    pub fn bool_not(&mut self, v: Value) -> Value;

    /// NanBox `==` with bit equality semantics. Correct for nil checks
    /// against pointers, integer-valued floats stored as canonical bits,
    /// pointer identity. Not IEEE-eq (NaN == NaN here).
    pub fn bit_eq(&mut self, a: Value, b: Value) -> Value;

    /// Decode a NanBox-encoded float and truncate to i64. Used by index
    /// arithmetic.
    pub fn nanbox_to_int(&mut self, v: Value) -> Value;
}
```

And on `Lowered`:

```rust
impl Lowered {
    /// Invoke `main` with the given args. Validates arity against the
    /// declared `main_arity` and pads/raises as appropriate.
    pub fn run_main(&self, jit: &Jit, args: MainArgs) -> JitOutcome { ... }
}

pub enum MainArgs {
    /// Pass nothing (must match `fn main()`).
    None,
    /// Pass a single host-built NanBox (typical "args vector" pattern).
    /// If `main` takes 0 args, ignored.
    Single(u64),
    /// Pass exactly N args. Errors if N != main_arity.
    Exact(Vec<u64>),
}
```

## Implementation plan

1. **`DynFunc::bool_not` / `bit_eq` / `nanbox_to_int`.** Three small
   methods. Inline the existing bodies; document the semantics
   beagle's call-site comments already capture (especially `bit_eq`'s
   NaN-equals-NaN caveat — it's the right answer for dynamic-language
   `==` on NanBox values, but surprising on the surface).

2. **`Lowered::run_main` + `MainArgs`.** Light wrapper over
   `gc.run_jit_with_threshold(&jit, main, &args, threshold)` that
   handles the arity-mismatch panic-vs-pad decision. Threshold stays
   a separate param (or a method like `with_gc_threshold`).

3. **Beagle migration.** Replace inline `bool_not`/`bit_eq`/
   `nanbox_to_int` definitions in lower.rs with toolkit calls. Replace
   the arity match in `real_main` with `lowered.run_main(&jit,
   MainArgs::Single(args_val))`. Estimated delta: −20 LOC.

## Open questions / risks

- **Truthiness policy.** `bool_not` depends on the embedder's notion
  of "falsey" — beagle treats `nil` and `false` as falsey, everything
  else (incl. `0.0`!) as truthy. That policy lives in `is_falsey`
  already. `bool_not` just composes it; embedders that want different
  truthiness override `is_falsey`, not `bool_not`.
- **`bit_eq` vs `dyn_eq`.** Beagle's `==` is bit-equality because the
  binary_trees / raycast subset doesn't compare strings or boxed
  numbers that aren't bit-canonical. A real `eq` for general dynamic
  values needs string-pool lookup, NaN handling, etc. Keep `bit_eq`
  as the primitive; document that it's *not* the language-level `==`
  for full programs. Embedders that need full `==` build it on top.
- **`nanbox_to_int` truncation policy.** `f64 → i64` via `as` panics
  in debug for out-of-range values on some targets but saturates on
  others. Document the policy match dynvalue; if `unbox_number` already
  picks one, follow it.
