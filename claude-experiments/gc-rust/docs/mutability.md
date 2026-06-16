# Mutability model (frozen v1)

**Core principle: everything is immutable by default. Mutability is an explicit,
declared capability — never the default — and this applies to ALL values, not
just `let` bindings.** gc-rust has no ownership/borrow checker, so mutability is
controlled *statically through the access path*: the compiler tracks whether the
path you are mutating through is rooted in a `mut`-declared binding or receiver.

## What "immutable" freezes — deep / transitive

An immutable binding freezes the **entire object graph reachable through it**:

```
let p = Point { x: 1, y: 2 };
p = other;     // ERROR: cannot reassign immutable binding `p`
p.x = 5;       // ERROR: cannot assign through immutable binding `p`
p.set_x(5);    // ERROR: `set_x` needs a `mut self`; `p` is immutable

let mut p = Point { x: 1, y: 2 };
p = other;     // ok
p.x = 5;       // ok
p.set_x(5);    // ok
```

Immutability is a property of the **path**, not the heap object. Because there is
no ownership, two bindings may alias the same heap data; whether you may mutate
depends on whether *the binding you reached it through* is `mut`. This is the
honest, implementable form of deep immutability without a borrow checker.

## The rules

1. **Bindings.** `let x = ...` is immutable; `let mut x = ...` is mutable.
   Reassigning (`x = ...`) an immutable binding is an error.

2. **Fields and elements (transitive).** `place.field = ...`,
   `array_set(place, ..)`, `place[i] = ...` require the **root** of `place` to be
   a `mut` binding (or `mut self`). The mutability flows down the whole access
   path: `a.b.c = v` requires `a` to be `mut`.

3. **Parameters.** `fn f(x: T)` receives `x` immutable; `fn f(mut x: T)` receives
   it mutable. A function can only mutate a parameter (or through it) if the
   parameter is declared `mut`. NOTE: `mut` on a parameter governs what the
   *callee* may do to its own binding — it does NOT write back to the caller
   (no `&mut`/aliasing). Mutations to heap objects reachable through the param
   ARE visible to the caller (shared heap); mutations that rebind the param are
   not.

4. **Receivers.** Methods declare `fn m(self, ...)` (immutable receiver) or
   `fn m(mut self, ...)` (mutable receiver). Calling a `mut self` method requires
   the receiver expression to be rooted in a `mut` binding.

5. **Value structs/enums** remain immutable as a type property (rebuild to
   "change" them) — a `mut` binding of a value struct allows whole-binding
   reassignment but not in-place field assignment, consistent with their flat
   value semantics. (Pre-existing rule, kept.)

6. **Closures.** A closure that mutates a captured variable requires that
   variable to be `mut` in the enclosing scope (the capture carries the
   mutability of the captured binding).

## What this is NOT

- It is **not** alias control. Two `mut` bindings to the same heap object can
  both mutate it; we do not track exclusivity (no borrow checker).
- It is **not** runtime-enforced. All checks are static, at lower/typecheck time.
- `mut` does **not** create caller-visible out-params. Rebinding a `mut` param
  is local; only shared-heap mutations are observable across the call.

## Implementation

Enforced in `src/lower.rs` during typecheck/lowering:

- A per-scope environment records each local's `is_mut`.
- Assignment lowering computes the **root binding** of the assignment target's
  access path and rejects the assignment if that root is not `mut` (covering
  rules 1, 2, 5).
- Method-call lowering checks the receiver's root mutability against the callee's
  `self` mutability (rule 4).
- Parameter binding seeds the scope env with each param's `is_mut` (rule 3).
- Closure capture records the captured binding's mutability (rule 6).

Errors are reported as structured diagnostics (Phase 5) with the message
"cannot assign through immutable binding `x` (declare it `let mut x`)".
