# gc-rust language reference

gc-rust is Rust-flavored, OCaml-weight, GC'd. It looks like Rust minus the
machinery that exists *only* to serve borrow-checking. There is **no** `&`,
`&mut`, lifetimes, `Box`, `Rc`, `move`, or `unsafe`. The GC owns memory.

This document is the grammar/surface reference. It is implemented; for the
runnable feature tour see `tour.md`, and for the cross-cutting rules see
`mutability.md` (immutable-by-default), `modules.md` (`mod`/`use`/`pub`), and
`overflow.md` (wrapping arithmetic).

## 1. Lexical

- Line comments `// …`, block comments `/* … */` (nestable).
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`.
- Integer literals with optional type suffix: `42`, `0xFF`, `0b1010`, `0o17`,
  `1_000`, `7u8`, `255u8`, `100i64`, `3u32`. Unsuffixed integers default to
  `i64` (inference may narrow).
- Float literals: `3.14`, `1.0e-9`, `2.5f32`. Unsuffixed floats default `f64`.
- Char literals: `'a'`, `'\n'`, `'\u{1F600}'`. Type `char` (a Unicode scalar).
- String literals: `"…"` with escapes; `String` type. Byte strings `b"…"`
  (`Bytes`) — later.
- Bool: `true`, `false`.
- Keywords: `fn let mut struct enum impl trait for in if else match while loop
  return break continue true false as where pub mod use type const static
  self Self value`.
  - `value` is gc-rust-specific: marks a struct/enum as a *value type* (stored
    inline, never independently heap-allocated). See §6.
- Operators: `+ - * / % == != < <= > >= && || ! & | ^ << >> = += -= *= /= %=
  -> => :: . .. ..= ? @`.
- Punctuation: `( ) { } [ ] , ; :`.

## 2. Items

```
program   := item*
item      := fn_def | struct_def | enum_def | trait_def | impl_block
           | type_alias | const_def | mod_decl | use_decl
visibility := "pub"?
```

### Functions
```
fn_def := vis "fn" IDENT generics? "(" params? ")" ret? where? block
params := param ("," param)* ","?
param  := "mut"? IDENT ":" type
ret    := "->" type
```
```rust
fn add(a: i64, b: i64) -> i64 { a + b }
fn id<T>(x: T) -> T { x }
fn clamp<T: Ord>(x: T, lo: T, hi: T) -> T where T: Copy { ... }
```

### Structs (reference type by default; `value` for inline)
```
struct_def := vis "value"? "struct" IDENT generics? struct_body
struct_body := "{" field ("," field)* ","? "}"   // named
             | "(" type ("," type)* ","? ")" ";"  // tuple
             | ";"                                  // unit
field := vis IDENT ":" type
```
```rust
struct Point { x: f64, y: f64 }          // reference type (GC heap)
value struct Vec3 { x: f64, y: f64, z: f64 }  // inline value type
value struct Pair<A, B>(A, B);           // tuple value type
```

### Enums
```
enum_def := vis "value"? "enum" IDENT generics? "{" variant ("," variant)* ","? "}"
variant  := IDENT ( "(" type ("," type)* ")" | "{" field,* "}" )?
```
```rust
enum Option<T> { None, Some(T) }
enum Result<T, E> { Ok(T), Err(E) }
value enum Color { Red, Green, Blue }    // niche-packable, inline
enum Tree<T> { Leaf, Node(Tree<T>, T, Tree<T>) }   // self-recursive ⇒ ref
```

### Traits + impls (monomorphic / static dispatch)
```
trait_def  := vis "trait" IDENT generics? (":" bound ("+" bound)*)? "{" trait_item* "}"
trait_item := fn_sig ";" | fn_def | "type" IDENT ";" | const_sig
impl_block := "impl" generics? (trait_for)? type where? "{" impl_item* "}"
trait_for  := type "for"
```
```rust
trait Show { fn show(self) -> String; }
trait Eq { fn eq(self, other: Self) -> bool; }

impl Show for i64 { fn show(self) -> String { int_to_string(self) } }
impl<T: Show> Show for Option<T> {
    fn show(self) -> String {
        match self { Option::None => "None", Option::Some(x) => x.show() }
    }
}
```

### Modules / use / type alias / const
```rust
mod geometry { pub fn area(...) -> f64 { ... } }
use geometry::area;
type Ints = Vec<i64>;
const TAU: f64 = 6.2831853;
```

## 3. Types
```
type := path generics?            // i64, Point, Vec<i64>, Option<T>, geometry::Shape
      | "(" type,* ")"            // tuple type, () = unit
      | "[" type ";" expr "]"     // fixed array [i64; 8]
      | "fn" "(" type,* ")" ret?  // function type
      | "Self" | IDENT            // type variable / Self
```
Primitive types: `i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 bool char` plus
`String`, `()` (unit). `isize/usize` later.

## 4. Expressions (Rust precedence)
```
expr := assign
assign := or (("="|"+="|...) assign)?
or := and ("||" and)*
and := cmp ("&&" cmp)*
cmp := bitor (("=="|"!="|"<"|"<="|">"|">=") bitor)*   // non-chaining
bitor := bitxor ("|" bitxor)*
bitxor := bitand ("^" bitand)*
bitand := shift ("&" shift)*
shift := add (("<<"|">>") add)*
add := mul (("+"|"-") mul)*
mul := cast (("*"|"/"|"%") cast)*
cast := unary ("as" type)*
unary := ("-"|"!") unary | postfix
postfix := primary (call | index | field | try)*
call := "(" args? ")"
index := "[" expr "]"
field := "." (IDENT | INT)           // .x  or  .0 (tuple)
try := "?"
primary := literal | path | "(" expr,* ")" | block | if | match | while
         | loop | "[" array "]" | struct_lit | closure | "return" expr?
         | "break" expr? | "continue"
```
- `if`/`match`/`loop`/`block` are expressions.
- `match`: `match e { pat => expr, pat if guard => expr, _ => expr }`.
- Closures: `|x: i64| x + 1`, `|x| { ... }` (param types inferred where
  possible). Closures capture by GC reference — no `move`.
- Method call `x.f(a)` resolves via traits/inherent impls (monomorphized).
- `?` on `Result`/`Option` early-returns the `Err`/`None`.
- Struct literal: `Point { x: 1.0, y: 2.0 }`; functional update later.

## 5. Statements
```
stmt := "let" "mut"? pattern (":" type)? ("=" expr)? ";"
      | expr ";"
      | item
block := "{" stmt* expr? "}"
```
Patterns: `_`, literals, identifiers (binding), `Enum::Variant(p, …)`,
struct patterns `Point { x, y }`, tuple patterns `(a, b)`, `mut` bindings.

## 6. Representation model (the fast part)

- **Primitives** are unboxed, native width, signedness-correct.
- **`value struct` / `value enum`** are stored *inline* (in their container, on
  the stack, in registers) — no header, no indirection, like a Rust struct.
  They live on the GC heap only transiently as part of a containing reference
  object. Arrays of value types are *flat*.
- **Plain `struct`/`enum`** are *reference types*: GC-heap-allocated, header +
  `TypeInfo`-described layout, passed by pointer, shared freely (cycles OK —
  the copying collector handles them). This is the default because it's the
  ergonomic, cycle-friendly choice; reach for `value` when you want flatness.
- A **self-recursive** type (e.g. `enum Tree`) must be a reference type (an
  inline value type of unbounded size is impossible); the checker enforces this.
- **Generics monomorphize**: every instantiation is its own concrete type with
  its own layout + `TypeInfo`. `Vec<Vec3>` stores `Vec3` flat; no boxing.
- **Traits**: static dispatch by default, impls monomorphized at call sites.
  `dyn Trait` (vtables) is a later, opt-in addition.
- Explicit boxing when you want indirection/sharing of a value type: `Gc<T>`
  (a one-field reference wrapper) — later; v0 gets it implicitly by using a
  plain `struct`.

## 7. Deliberately absent (vs Rust)

borrow checker, lifetimes, `&`/`&mut`, `Box/Rc/Arc/Cell/RefCell`, `move`,
`unsafe` (v0), async, macros (v0), `impl Trait` in arg position (v0 uses
explicit generics). These exist in Rust largely to manage ownership; the GC
removes the need.
