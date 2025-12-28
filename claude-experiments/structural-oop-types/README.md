# Structural OOP Type Checker

A type checker for a small object-oriented language demonstrating **structural typing** with **row polymorphism** and **equi-recursive types** for self-reference. This enables Cook-style "object as closure" programming with full type inference.

## What This Accomplishes

This project implements the type system described in William Cook's ["On Understanding Data Abstraction, Revisited"](https://www.cs.utexas.edu/~wcook/Drafts/2009/essay.pdf), where objects are self-referential records and operations like `insert` return new objects of the same recursive type.

### Key Features

1. **Row Polymorphism** - Functions accepting objects only require the fields they use. A function `s => s.isEmpty` works with *any* object that has an `isEmpty` field.

2. **Equi-Recursive Types** - Self-referential types are handled implicitly via cycle detection during unification. No explicit `μ` binders needed in source code.

3. **Structural Subtyping** - Objects are structurally typed. An object with fields `{a, b, c}` can be passed where `{a, b}` is expected.

4. **Full Type Inference** - No type annotations required. The system infers recursive types, row variables, and polymorphic function types automatically.

## The Cook Set Example

The motivating example from Cook's paper - immutable integer sets with `insert` and `union`:

```javascript
let rec Insert = s => n => {
  isEmpty: false,
  contains: i => i == n || s.contains i,
  insert: i => Insert this i,
  union: s2 => Union this s2
}
and Union = s1 => s2 => {
  isEmpty: s1.isEmpty && s2.isEmpty,
  contains: i => s1.contains i || s2.contains i,
  insert: i => Insert this i,
  union: s => Union this s
}
in
let Empty = {
  isEmpty: true,
  contains: i => false,
  insert: i => Insert this i,
  union: s => Union this s
}
in Empty
```

**Inferred type:**
```
μα. { isEmpty: bool,
      contains: int → bool,
      insert: int → α,
      union: { contains: int → bool, isEmpty: bool | ρ } → α }
```

Key observations:
- `insert` returns `α` - the same recursive set type
- `union` accepts any object with `contains` and `isEmpty` (structural polymorphism)
- `this` refers to the object being constructed, enabling the recursive type

## Syntax

```
// Literals
true, false, 42, -10

// Arrow functions
x => x
(x) => x + 1
x => y => x

// Objects (with : for fields)
{ x: 42, y: true }
{ isEmpty: true, get: this }

// Field access
obj.field

// Operators
a == b      // integer equality
a && b      // boolean and
a || b      // boolean or

// Conditionals
cond ? then : else

// Let bindings
let x = 1 in x + 1

// Recursive bindings
let rec f = x => f x in f

// Mutual recursion
let rec f = x => g x
and g = x => f x
in f 1
```

## Usage

```bash
# REPL
cargo run

# Type-check expression
cargo run -- 'x => x'

# Run examples
cargo run -- --examples
```

## How It Works

### Type Representation

Types are represented as a union-find graph where nodes can be:
- **Type variables** - Unification variables
- **Constants** - `bool`, `int`
- **Arrows** - Function types `τ₁ → τ₂`
- **Records** - `{ field₁: τ₁, field₂: τ₂ | ρ }` where `ρ` is a row variable

### Row Polymorphism

Rows are lists of field/type pairs with a tail that's either:
- **Empty** - Closed row (objects have exactly these fields)
- **Row variable** - Open row (may have additional fields)

When type-checking `obj.field`, we unify `obj` with `{ field: α | ρ }`, requiring only that one field exists.

### Equi-Recursive Types

When type-checking `{ f: this }`:
1. Create fresh type variable `α` for `this`
2. Infer object type as `{ f: α }`
3. Unify `α` with `{ f: α }`
4. This creates a cycle in the type graph representing `μα. { f: α }`

The cycle is detected during display, not during unification. This is the "equi-recursive" approach (types equal to their unfoldings) vs "iso-recursive" (explicit fold/unfold).

### Key Files

- `lexer.rs` / `parser.rs` - JavaScript-like syntax
- `expr.rs` - AST definition
- `node.rs` / `store.rs` - Type graph with union-find
- `unify.rs` - Unification with row polymorphism
- `infer.rs` - Type inference
- `display.rs` - Pretty printing with μ-binder detection

## Next Steps

### Language Features

1. **Type Annotations** - Optional annotations for documentation and error messages
   ```javascript
   let f: int -> int = x => x in f 1
   ```

2. **Polymorphic Let** - Currently let-bound values are monomorphic. Add let-generalization:
   ```javascript
   let id = x => x in (id 1, id true)  // Currently fails
   ```

3. **Pattern Matching** - Destructuring objects
   ```javascript
   let { x, y } = point in x + y
   ```

4. **Algebraic Data Types** - Sum types alongside objects
   ```javascript
   type Option a = None | Some a
   ```

5. **Modules** - Separate compilation, interfaces

### Type System Extensions

1. **Subtyping** - Full structural subtyping (currently only via row polymorphism)
   - Width subtyping: `{a, b, c} <: {a, b}`
   - Depth subtyping: `{x: {a, b}} <: {x: {a}}`

2. **Bounded Polymorphism** - Constrain type variables
   ```javascript
   let f: forall a <: {x: int}. a -> int = obj => obj.x
   ```

3. **Effect Types** - Track side effects in types

4. **Refinement Types** - Dependent types for verification

### Implementation Improvements

1. **Error Messages** - Currently minimal. Add source locations, expected vs actual types, suggestions.

2. **Incremental Type Checking** - For IDE integration

3. **Optimization** - The union-find could use path compression more aggressively

4. **Evaluation** - Add an interpreter to actually run programs

5. **Compilation** - Generate JavaScript, WASM, or native code

### Research Directions

1. **F-bounded Polymorphism** - For more expressive self-types (see Abadi & Cardelli)

2. **Virtual Classes** - Family polymorphism for mutually recursive class hierarchies

3. **Gradual Typing** - Mix static and dynamic typing

4. **Session Types** - For typed communication protocols

## References

- Cook, ["On Understanding Data Abstraction, Revisited"](https://www.cs.utexas.edu/~wcook/Drafts/2009/essay.pdf) (2009) - The inspiration
- Wand, "Complete Type Inference for Simple Objects" (LICS 1987) - Row polymorphism foundation
- Rémy, "Type Inference for Records" (1993) - Row types with presence/absence
- Amadio & Cardelli, "Subtyping Recursive Types" (1993) - Equi-recursive theory
- Pierce, *Types and Programming Languages* - General reference

## License

MIT
