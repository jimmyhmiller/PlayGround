# Bidirectional GC Language: Syntax Sketch

This is a sketch for a language with:

- Rust-style syntax
- GC memory model
- Bidirectional typing (not Hindley-Milner)
- Required function signature annotations
- `enum`, `trait`, `struct`, `fn`
- Multithreading + JIT-first runtime

## Design center

This language keeps Rust's readability and algebraic data modeling, but deliberately avoids Rust's ownership/lifetime machinery.

Things we **won't** do that Rust has to do:

- No borrow checker.
- No explicit lifetimes in types.
- No ownership moves/partial moves/drop checking.
- No `unsafe` for ordinary shared heap programming.
- No `Box`/`Rc`/`Arc` split for basic memory safety.
- No manual memory-layout work just to express recursive data.
- No separate "async color" driven by ownership workarounds.

Tradeoff: we accept GC/runtime complexity instead of ownership-system complexity.

## Core syntax sketch

```lang
module collections::list;

pub enum List<T> {
  Nil,
  Cons(head: T, tail: List<T>),
}

pub struct Vec2 {
  x: F64,
  y: F64,
}

pub trait Eq<T> {
  fn eq(a: T, b: T) -> Bool;
}

impl Eq<Vec2> {
  fn eq(a: Vec2, b: Vec2) -> Bool {
    a.x == b.x && a.y == b.y
  }
}

pub fn length<T>(xs: List<T>) -> I64 {
  match xs {
    List::Nil => 0,
    List::Cons(_, tail) => 1 + length(tail),
  }
}
```

## Bidirectional typing model (surface)

The surface language follows bidirectional checking rules:

- Function declarations require full parameter + return types.
- `let` without annotation is inferred from initializer.
- Ambiguous literals require type context or annotation.
- Generic instantiation can be explicit at call sites when needed.

```lang
fn add_one(x: I64) -> I64 {
  x + 1
}

fn demo() -> I64 {
  let a = add_one(41);      // inferred I64
  let b: I64 = 1;           // explicit check mode
  a + b
}
```

No global HM-style generalization pass. Type checking proceeds by synthesize/check flow.

## Structs and enums

```lang
struct User {
  id: I64,
  name: String,
  roles: List<String>,
}

enum Result<T, E> {
  Ok(T),
  Err(E),
}

fn describe(u: User) -> String {
  if length(u.roles) == 0 {
    "guest"
  } else {
    "member"
  }
}
```

Recursive and shared data are normal heap objects under GC. No lifetime plumbing required.

## Traits and impls

```lang
trait Show<T> {
  fn show(value: T) -> String;
}

impl Show<I64> {
  fn show(value: I64) -> String {
    int_to_string(value)
  }
}

fn print_show<T: Show>(value: T) -> Unit {
  println(Show::show(value));
}
```

Trait dispatch model (current direction):

- Trait calls are semantically dynamic (dictionary/vtable passing).
- There is no separate `Dyn<T>` surface type.
- Static dispatch/monomorphization is treated as an optimization the compiler may apply when it can prove a concrete implementation.

## Concurrency sketch

```lang
struct Counter {
  value: Atomic<I64>,
}

fn worker(counter: Counter, n: I64) -> Unit {
  let mut i = 0;
  while i < n {
    counter.value.fetch_add(1);
    i = i + 1;
  }
}

fn main() -> I64 {
  let counter = Counter { value: Atomic::new(0) };
  let t1 = spawn fn() { worker(counter, 1_000_000) };
  let t2 = spawn fn() { worker(counter, 1_000_000) };

  t1.join();
  t2.join();

  counter.value.load()
}
```

No borrow checker means thread safety is enforced by type-level shared-state APIs (`Atomic`, channels, actor/mailbox types), not aliasing rules.

## Dynamic loading sketch (typed)

```lang
// Host side
trait Plugin {
  fn name() -> String;
  fn run(input: Bytes) -> Bytes;
}

fn main() -> Unit {
  let p: Plugin = load_plugin("./plugins/compress.lmod", Plugin);
  println(p.name());
}
```

```lang
// Plugin side
module plugin::compress;

impl Plugin {
  fn name() -> String { "compress" }
  fn run(input: Bytes) -> Bytes { compress_bytes(input) }
}
```

Loader checks ABI/type signature hashes before linking JITed code.

## Dispatch and optimization policy

```lang
trait Hash<T> {
  fn hash(value: T) -> I64;
}

fn hash_all<T: Hash>(xs: List<T>) -> I64 {
  match xs {
    List::Nil => 0,
    List::Cons(x, rest) => Hash::hash(x) + hash_all(rest),
  }
}
```

`Hash::hash` is a dynamic trait call by language semantics. In optimized JIT tiers, the compiler can devirtualize or specialize `hash_all` when concrete `T` is known and profitable.

## Trait coherence and orphan rule

We follow Rust-style coherence to keep trait resolution deterministic, including with dynamic loading.

- One canonical impl per `(Trait, Type)` in the program.
- Orphan rule: an impl is only legal if either the trait is local or the type is local to the defining module/package.
- Dynamically loaded modules are checked against already-loaded impls; conflicting impls are rejected at load time.

```lang
// In package A
pub trait Serialize<T> {
  fn serialize(value: T) -> Bytes;
}

// In package B
pub struct User { id: I64 }

// Legal in B: type is local to B
impl Serialize<User> {
  fn serialize(value: User) -> Bytes { ... }
}

// Illegal in C: trait and type are both foreign
// impl Serialize<User> { ... }
```

## Built-in display and reflection

We want structs/enums to be printable and inspectable out of the box, so runtime type metadata is part of the object model.

```lang
struct User {
  id: I64,
  name: String,
}

fn main() -> Unit {
  let u = User { id: 42, name: "jimmy" };
  println(u);                 // default display/pretty-print
  inspect(u);                 // structured debug view
  let fields = reflect(u);    // runtime field metadata + values
}
```

Layout/metadata requirements:

- Every heap object header includes a pointer to canonical `TypeInfo`.
- `TypeInfo` contains kind (struct/enum), type name, size/alignment, and field/variant tables.
- Field table entries include field name, offset, field type id, and visibility flags.
- Enum metadata includes variant tags and per-variant payload layouts.
- Metadata lives in read-only sections and is emitted by JIT/AOT codegen in a stable format.
- Dynamic module loader interns/merges `TypeInfo` identities so reflection stays coherent across loaded modules.

This lets us implement default `Display`, `Debug`, and reflection APIs without user-authored derive boilerplate.

## What this buys us over Rust-style ownership complexity

- Easier recursive data modeling.
- Fewer type-level concepts for everyday code.
- Easier refactors across function boundaries (no lifetime propagation).
- Simpler plugin/dynamic loading ergonomics for application developers.
- No explicit `dyn`/existential syntax burden in user code for trait values.
- Faster onboarding for people coming from OCaml/TypeScript/Kotlin-like memory models.

## What we still must solve (runtime-side)

- Multithreaded GC (safepoints, barriers, pause behavior).
- Stable ABI for traits/enums/struct layout across dynamically loaded modules.
- Deoptimization/patching strategy for JITed code updates.
- Predictable performance despite allocation-heavy idioms.
- Strong enough devirtualization/specialization so dynamic trait semantics still perform well in hot paths.

This sketch is intentionally opinionated: keep Rust syntax and type clarity, remove ownership/lifetime burden, and pay the complexity in runtime + compiler backend.
