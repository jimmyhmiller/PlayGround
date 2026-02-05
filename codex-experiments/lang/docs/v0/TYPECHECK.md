# Typechecking v0 (Draft)

Bidirectional typing: some expressions synthesize a type, others are checked against an expected type.

## 1. Core rules

- Function parameters and return types must be explicitly annotated.
- `let` bindings infer from initializer if no annotation is given.
- Literals require context if ambiguous (e.g., integer literal without annotation).
- No global type inference or HM-style generalization.

## 2. Expressions

### Synthesis

Expressions that synthesize a type:

- Literals (if unambiguous)
- Variables
- Function calls (from callee signature)
- Field access
- Struct literals (from struct type)

### Checking

Expressions checked against an expected type:

- `if` and `match` branches (all branches must check to same type)
- `let x: T = expr` checks `expr` against `T`
- Function bodies are checked against declared return type

## 3. Control flow

- `if` requires `Bool` condition; both branches must typecheck to same type.
- `while` requires `Bool` condition; the type of `while` is `Unit`.
- `match` requires all patterns to be exhaustive (exhaustiveness can be deferred to a later pass; for v0 it is ok to allow non-exhaustive and warn).

## 4. Structs and enums

- Struct literal must include all fields.
- Enum variant construction uses `Enum::Variant(...)` syntax.
- Field access type is determined by the struct definition.

## 5. Functions and calls

- Calls require arity match.
- Overload resolution does not exist in v0.
- Varargs are only allowed on `extern fn` declarations; calls may supply any number of extra arguments after fixed params.

## 6. Traits (v0)

- Trait constraints are syntactic only for now.
- Trait method calls are desugared to `Trait::method(value, ...)` and resolved via impl lookup.

## 7. Raw pointers

- `RawPointer<T>` is opaque.
- No pointer arithmetic in v0.
- Equality comparison allowed.

## 8. Type equality

- Nominal type equality for `struct` and `enum`.
- Type aliases are not supported in v0.

