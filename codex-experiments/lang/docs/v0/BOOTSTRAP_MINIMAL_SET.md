# Bootstrap Minimal Feature Set

This document captures the smallest set of language + compiler features that materially reduce the bootstrap compiler "mess" while staying within v0 scope (no traits, no borrow checker, no higher-kinded types).

The goal is to make the self-hosted compiler codebase *simple* and *compact* without requiring advanced features.

## Minimal Set (Must-Have)

1. **Parametric generics (struct/enum/fn only, monomorphized)**
   - Enables a reusable `List<T>` (or `Vec<T>`) and small algebraic helpers like `Option<T>`.
   - Collapses dozens of bespoke list types: `ParamList`, `FieldList`, `TokenList`, etc.

2. **Modules + `use` with file-based resolution**
   - Avoids global namespace collisions and repetitive prefixes.
   - Allows a clean stdlib and compiler architecture.

3. **Tuple values (literal + return value)**
   - Eliminates `ParseFoo` structs whose only purpose is returning `(value, rest)`.
   - Keeps parsing code readable and reduces boilerplate.

4. **Default numeric literal typing (I64/F64)**
   - Removes the need to introduce `let zero: I64 = 0;` everywhere.
   - Still allows explicit annotation for narrower types.

5. **Char / byte literal (`'a'`, `0x61`)**
   - Simplifies lexer code by removing hard-coded ASCII constants.
   - Improves clarity for keyword checks and tokenization.

## Implementation Order (Incremental)

### Phase 0: Quick Wins (1–2 files, immediate payoff)
- Default numeric literal typing to `I64`/`F64` when no context exists.
- (Optional) Byte/char literal in lexer/parser/typechecker.

### Phase 1: Modules
- Implement `module` and `use` resolution in the Rust compiler.
- File → module mapping: `foo/bar.lang` maps to `foo::bar`.

### Phase 2: Generics (Monomorphized)
- Add type parameters to `struct`, `enum`, `fn`.
- Monomorphize at codegen time for concrete instantiations.
- Add a minimal `List<T>` in stdlib.

### Phase 3: Tuple Values
- Tuple literal `(a, b)` and return values.
- Pattern support is optional; can add later.

## Out of Scope (for this bootstrap step)
- Traits / impl blocks / typeclasses
- Higher-kinded types
- Global type inference
- GC-safe managed string interop beyond current `String` ABI

## Success Criteria
- Compiler sources no longer define per-type lists (`ParamList`, `FieldList`, etc.).
- Parser functions return tuples or generic helpers.
- stdlib can be organized into modules without naming collisions.
- Self-hosted compiler size drops significantly without loss of clarity.
