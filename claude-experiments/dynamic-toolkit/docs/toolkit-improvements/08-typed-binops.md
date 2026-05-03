# 08 — Type-Specialized Binop Selection

## Implementation status: ✅ Implemented

- **Code**: `dynlang::TypeHint` (`#[non_exhaustive]` enum: `Unknown` / `Number` / `Bool`) and `DynFunc::add/sub/mul/div/lt/gt/le/ge` taking `(Value, TypeHint, Value, TypeHint)` in [`crates/dynlang/src/lib.rs`](../../crates/dynlang/src/lib.rs). `le` and `ge` synthesize `!gt` / `!lt` for the non-numeric path (since dyn_le/dyn_ge primitives don't exist).
- **Tests**: 5 tests in dynlang `tests` module — fast-path correctness for add (with panic-stub bound to verify slow path doesn't fire), sub/mul/div, lt/gt/le/ge, the conservative dyn path with inline float fallback, plus a sanity check on `TypeHint::default()`.
- **Migration**: beagle's eight `if both_num { num_X } else { dyn_X }` arms in `Ast::{Add,Sub,Mul,Div}` and `Ast::Condition` collapsed to single `f.X(l, lh, r, rh)` calls. Added `From<&Ty> for dynlang::TypeHint` bridge in `crates/beagle/src/types.rs`.

## Problem

`DynFunc` exposes two parallel binop families:

- `num_*` — assume both operands are NanBox-encoded numbers; emit
  `bitcast → fop → bitcast`. Skips the tag-check branch.
- `dyn_*` — full fast/slow dispatch with tag tests and slow-path call.

Picking between them is the embedder's job, and beagle does it inline at
every site:

```rust
// lower.rs:605-685, repeated 8 times for +, -, *, /, <, >, <=, >=
Ast::Add { left, right, .. } => {
    let lt = self.types.type_of(left);
    let rt = self.types.type_of(right);
    let l = self.lower_expr(f, left);
    let r = self.lower_expr(f, right);
    if lt.is_number() && rt.is_number() {
        f.num_add(l, r)
    } else {
        f.dyn_add(l, r)
    }
}
```

Eight identical scaffolds, plus the comparison ops have additional
branching for `<=`/`>=` because dynamic comparison only exposes `<` and
`>` ([`lower.rs:669-684`](../../crates/beagle/src/lower.rs#L669)).

The shape leaks two implementation details:

1. **The `num_*` / `dyn_*` distinction.** Conceptually there's one `add`
   that picks based on operand type. Today the API forces the caller to
   choose; tomorrow's frontend may need a third variant (e.g. `int_*`
   for unboxed ints) and every embedder will need a third arm.
2. **The "what counts as a number" question.** Beagle's `Ty::is_number`
   is the policy. Other embedders will have their own type lattices but
   the same selection logic.

## Proposed API

A `TypeHint` carried alongside `Value`, plus binop methods that accept it:

```rust
// Embedder-visible:
#[derive(Clone, Copy)]
pub enum TypeHint {
    /// No information; emit the conservative `dyn_*` form.
    Unknown,
    /// Operand is statically known to be a NanBox-encoded number.
    Number,
    /// Operand is statically known to be a NanBox-encoded boolean.
    Bool,
    // ...future: ObjectKind(ObjTypeId), String, etc.
}

impl DynFunc {
    pub fn add(&mut self, l: Value, lh: TypeHint, r: Value, rh: TypeHint) -> Value {
        match (lh, rh) {
            (TypeHint::Number, TypeHint::Number) => self.num_add(l, r),
            _ => self.dyn_add(l, r),
        }
    }
    // sub/mul/div/lt/gt/le/ge — same shape.
}
```

Beagle's lowering collapses to:

```rust
Ast::Add { left, right, .. } => {
    let lh = TypeHint::from(self.types.type_of(left));
    let rh = TypeHint::from(self.types.type_of(right));
    let l = self.lower_expr(f, left);
    let r = self.lower_expr(f, right);
    f.add(l, lh, r, rh)
}
```

A small `From<Ty>` bridge per embedder is the only customization point.

### Stretching to `Value`-carries-hint

The fancier shape carries the hint inside `Value`:

```rust
struct TypedValue { v: Value, hint: TypeHint }
impl DynFunc { fn add(&mut self, l: TypedValue, r: TypedValue) -> TypedValue }
```

This is cleaner at the call site but requires every `DynFunc` method to
return `TypedValue`, breaks today's `Value` API, and forces embedders to
think about hints even when they don't have any. **Don't do this** —
explicit hints at the binop boundary is the right tradeoff.

### `<=` and `>=`

Today `f.dyn_lt` / `f.dyn_gt` exist but `f.dyn_le` / `f.dyn_ge` don't,
so beagle synthesizes `le = !gt(l,r)`. Either add the missing primitives
or document the synthesis pattern in `DynFunc` docs.

## Implementation plan

1. **`dynir::TypeHint` enum.** Lives next to `Type` since the IR layer
   is what knows whether `num_*` is sound. Re-exported through
   `dynlang`.

2. **Method overloads on `DynFunc`.** Add `add(l, lh, r, rh) -> Value`
   etc. for the eight operations. Internally they pattern-match on the
   hint pair and dispatch to the existing `num_*` / `dyn_*` primitives.
   Don't remove the primitives — embedders that want fine control still
   need them.

3. **Fill in `dyn_le` / `dyn_ge`.** Currently synthesized at every call
   site as `!dyn_gt` / `!dyn_lt`. Not strictly required for this
   proposal but eliminates the same-shape boilerplate at the site.

4. **Beagle migration.** Replace the eight `if lt.is_number() && ...`
   blocks with single calls. Add a `From<Ty> for TypeHint` impl.
   Estimated delta: −40 LOC.

## Open questions / risks

- **Soundness of `num_*` under aliasing.** `num_add` assumes both bits
  are valid NanBox floats. If an embedder's type analysis is buggy and
  it passes hints `(Number, Number)` for non-number bits, the result
  is silent garbage. Today's API has the same hazard — the embedder is
  always responsible for correctness — but the hint-based form makes
  the responsibility one step further from the misbehavior. Worth
  documenting in `DynFunc::add` doc comments.
- **Mixed integer/float specialization.** A future
  `TypeHint::IntInRange { lo, hi }` opens unboxed integer codegen
  paths. The enum is non-exhaustive from day one to leave room.
- **Trait-shaped alternative.** `f.add::<NumNum>(l, r)` via a marker
  trait avoids runtime hint dispatch — the matching happens at
  monomorphization. Fine if the embedder always knows hints
  statically; useless otherwise. The enum form generalizes to
  embedder-time decisions.
