# A tour of gc-rust

gc-rust is a Rust-flavored, garbage-collected, monomorphized systems language.
It compiles through LLVM to native code (JIT today, AOT via `gcr build`). It
keeps the parts of Rust that make code clear and fast, and drops the parts that
exist only to manage memory by hand — there are no lifetimes, no borrows, no
`&`/`&mut`, no `Box`. A precise copying garbage collector owns memory. Generics
monomorphize, so abstraction is free: a `value struct` passed by value compiles
to the same registers a hand-written one would.

This tour shows the language by example. Every snippet here runs.

## Hello, numbers

```rust
fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() -> i64 {
    fib(32)          // main returns an i64; the driver prints it
}
```

```
$ gcr run hello.gcr
2178309
```

`main` returns `i64`. `print_int(x)` / `print_float(x)` produce output as a side
effect.

## Scalars and arithmetic

The numeric types are `i8 i16 i32 i64`, `u8 u16 u32 u64`, `f32 f64`, plus `bool`
and `char`. Signedness is real: unsigned division, shifts, and comparisons use
the unsigned instructions; widening casts sign- or zero-extend correctly.

```rust
fn main() -> i64 {
    let a = 200u32;
    let b = 3u32;
    let q = a / b;            // unsigned division -> 66
    let f = 2.5 * 4.0;        // f64
    q as i64 + (f as i64)     // 66 + 10 = 76
}
```

Literals default to `i64` / `f64`; a suffix (`7u8`, `2.5f32`) or the expected
type pins them. `sqrt`, `abs`, `floor`, `ceil` are built-in float intrinsics.

## Control flow

`if`/`else`, `while`, `loop`/`break`/`continue`, `for i in lo..hi`, and `match`
are all expressions.

```rust
fn main() -> i64 {
    let mut sum = 0;
    for i in 0..100 { sum = sum + i; }   // 4950
    let mut k = 0;
    while k < 10 { k = k + 1; }
    sum + k
}
```

## Structs

A plain `struct` is a **reference type**: it lives on the GC heap and is passed
by pointer (shared freely; cycles are fine — the copying collector handles
them). Fields are mutable through a binding.

```rust
struct Counter { n: i64 }

fn main() -> i64 {
    let mut c = Counter { n: 0 };
    c.n = 10;
    c.n = c.n + 32;
    c.n                       // 42
}
```

A `value struct` is an **inline value type**: stored flat (in registers, in its
container), never independently heap-allocated, passed by value. Reach for it
when you want zero-indirection data — the workhorse of fast numeric code.

```rust
value struct Vec3 { x: f64, y: f64, z: f64 }

fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z   // a, b passed in registers, no heap
}
```

## Enums and pattern matching

```rust
enum Shape {
    Circle(i64),
    Rect(i64, i64),
}

fn area(s: Shape) -> i64 {
    match s {
        Shape::Circle(r) => r * r * 3,
        Shape::Rect(w, h) => w * h,
    }
}
```

Reference enums are heap objects; `value enum`s are inline tagged unions (no
heap), including payloads. **`match` is checked for exhaustiveness** — a match
that misses a variant and has no `_` wildcard is a compile error:

```
error: non-exhaustive match on `Shape`: missing variant(s) Rect
       (add the arms or a `_` wildcard)
```

Recursive enums build trees and lists naturally:

```rust
enum Tree { Leaf, Node(Tree, Tree) }

fn count(t: Tree) -> i64 {
    match t {
        Tree::Leaf => 1,
        Tree::Node(l, r) => 1 + count(l) + count(r),
    }
}
```

## Generics — monomorphized, no boxing

Generic functions and types are specialized per concrete instantiation. There is
no boxing and no type erasure: `Pair<i64, f64>` stores an `i64` and an `f64`
flat, and `id::<i64>` and `id::<String>` are two separate native functions.

```rust
struct Pair<A, B> { a: A, b: B }

fn first<A, B>(p: Pair<A, B>) -> A { p.a }

fn main() -> i64 {
    let p = Pair { a: 10, b: 2.5 };
    first(p)                  // 10
}
```

## Traits — static dispatch

```rust
trait Area { fn area(self) -> i64; }

struct Square { side: i64 }
impl Area for Square {
    fn area(self) -> i64 { self.side * self.side }
}

fn describe<T: Area>(x: T) -> i64 { x.area() }
```

Trait methods dispatch statically and monomorphize. Bounds are **checked**:
calling `describe` with a type that doesn't implement `Area` is a compile error:

```
error: the trait bound `Foo: Area` is not satisfied (required by `describe`)
```

## Option, Result, and `?`

```rust
enum Option<T> { None, Some(T) }
enum Result<T, E> { Ok(T), Err(E) }

fn checked_div(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 { Result::Err("divide by zero") } else { Result::Ok(a / b) }
}

fn compute(x: i64) -> Result<i64, String> {
    let q = checked_div(x, 2)?;      // `?` returns the Err early
    let r = checked_div(q + 6, 2)?;
    Result::Ok(r + 1)
}
```

## Closures

Closures capture by GC reference (no `move`, no lifetimes). A closure is a heap
object holding its captures plus a code pointer; calling it is an indirect call.

```rust
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

fn main() -> i64 {
    let base = 10;
    let add_base = |n: i64| n + base;     // captures `base`
    apply(add_base, 32)                   // 42
}
```

## Arrays and a growable Vec

`Array<T>` is a fixed-length, GC-managed buffer. `array_new(n)`, `a[i]`,
`a[i] = v`, and `array_len(a)` are the primitives; reference-element arrays are
traced by the GC, scalar arrays are flat.

```rust
fn main() -> i64 {
    let a: Array<i64> = array_new(100);
    for i in 0..100 { a[i] = i * i; }
    let mut sum = 0;
    for j in 0..100 { sum = sum + a[j]; }
    sum
}
```

A growable vector is just a struct over an `Array<T>` that reallocates and copies
when it fills — written in gc-rust itself (see `examples/vec.gcr`).

## What's deliberately absent (vs Rust)

The borrow checker, lifetimes, `&`/`&mut`, `Box`/`Rc`/`Arc`/`Cell`/`RefCell`,
`move`, `unsafe`. These exist in Rust to manage ownership by hand; the GC removes
the need. You keep the syntax, the types, the monomorphization, and the speed.

## Running and building

```
gcr run  app.gcr          # JIT compile and run
gcr build app.gcr -o app  # compile to a standalone native executable
gcr check app.gcr         # type-check only
```

See `docs/language.md` for the full grammar, `docs/gc.md` for how the collector
and codegen cooperate, and `docs/core-ir.md` for the monomorphic IR.
