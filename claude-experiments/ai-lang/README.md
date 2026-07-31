# ai-lang

A statically-typed language whose programs live in a **content-addressed
codebase** rather than in text files. A definition is identified by the hash
of its canonical AST; names are a mutable layer on top. That one decision is
what makes renames free, refactors verifiable, tests cacheable, and code
shippable to another machine by hash.

```sh
ai-lang add examples/threads_demo.ail
ai-lang run main        # 14850
```

---

## Part 1 — The codebase system

The examples below use a small program: an `Item` struct, and three functions
that price one.

```rust
struct Item { name: String, cents: Int, qty: Int }

def line_total(it: Item) -> Int = it.cents * it.qty
def apply_discount(total: Int, pct: Int) -> Int = total - (total * pct / 100)
def checkout(it: Item, pct: Int) -> Int = apply_discount(line_total(it), pct)
```

### Code is content; names are a lookup table

A definition's identity is the hash of its canonical AST. Source text is a
*projection* of that AST, not the source of truth. Ingesting a file stores
definitions, not lines:

```sh
$ ai-lang add shop.ail
added 419 defs to .ai-lang (419 new typeschemes cached)
```

Project any definition back to source at any time:

```sh
$ ai-lang view checkout
def checkout(it: Item, pct: Int) -> Int = apply_discount(line_total(it), pct)
```

Two properties fall straight out of this. A definition's **type never needs
recomputing** — its hash determines it, so a type is computed once and cached
forever. And reading a definition back **re-verifies its hash**, so bit rot or
a stray edit cannot pass silently.

### Rename is O(1) and cannot break a caller

Callers reference definitions by hash, never by name. A rename moves one alias
and touches nothing else, so every caller's stored code — and therefore every
caller's own identity — is untouched.

```sh
$ ai-lang ls | grep -w checkout
checkout                          81c1b5d8d6784b36

$ ai-lang rename line_total subtotal
renamed line_total -> subtotal  (hash aa5f4b62ff6ee6af unchanged; no callers broken)

$ ai-lang ls | grep -w checkout
checkout                          81c1b5d8d6784b36        # ← identical

$ ai-lang view checkout
def checkout(it: Item, pct: Int) -> Int = apply_discount(subtotal(it), pct)
```

Nothing recompiled, no caller was edited, and `view` renders the new name
because names are resolved when code is projected back to text. `alias` does
the same but keeps the old name; `move` renames to a dotted path.

### Exact dependency queries — no grep

"Who calls this" is a lookup, not a text search, and it cannot be fooled by a
comment or a shadowed name.

```sh
$ ai-lang usages subtotal
checkout                          81c1b5d8d6784b36
  1 usage(s) of subtotal

$ ai-lang deps checkout
Item                              1cf6202932b172fc
apply_discount                    7201a681af13f981
subtotal                          aa5f4b62ff6ee6af

$ ai-lang deps checkout --reverse --transitive
```

Add `--json` to any of these for tooling.

### Updates, the dependency cone, and todos

Changing a definition creates a *new* definition. Dependents still point at
the old one, and that is not a failure state — it is a worklist. `--dry-run`
reports the impact without committing anything:

```sh
$ ai-lang update apply_discount disc.ail --dry-run
updated apply_discount: 7201a681af13 -> 48522baedf18
todos (3):
  - test_checkout (545022417fe2): still references old version of `apply_discount`
  - main (7d125fc977dc): still references old version of `apply_discount`
  - checkout (81c1b5d8d678): still references old version of `apply_discount`
[dry-run] no names were moved
```

`--propagate` rewrites the whole cone in dependency order and typechecks each
rewritten definition:

```sh
$ ai-lang update apply_discount disc.ail --propagate
updated apply_discount: 7201a681af13 -> 48522baedf18
  propagated checkout:      81c1b5d8d678 -> fbd22f23556a
  propagated test_checkout: 545022417fe2 -> 4c74aea45d23
  propagated main:          7d125fc977dc -> 98d9086b8748
todos: none
```

Now revert that edit and propagate again:

```sh
  propagated test_checkout: 4c74aea45d23 -> 545022417fe2
  propagated main:          98d9086b8748 -> 7d125fc977dc
```

The hashes land **exactly back where they started**. Identical content is
identical identity, so undoing a change is not a new state to track — it is
the old state, reached again.

`ai-lang todos` lists anything still on the worklist; `ai-lang propagate
<name>` verifies a cone typechecks without changing it.

### Structural refactors

These transform the stored code and then propagate, rather than editing text:

```sh
$ ai-lang inline double            # beta-reduce every call site, capture-correctly
updated quad: d2a5d80cd8e5 -> c45e85619039
  propagated top: 899b1c097d8e -> fa883cc4db8c

$ ai-lang view quad
def quad(p0: Int) -> Int = ((p0 * 2) * 2)
```

`reorder-params <name> <perm>` permutes parameters and rewrites every call
site so behavior is preserved. `extract <name> <sel> <new_name>` lifts a
subexpression into its own definition. All of them refuse rather than
half-apply — `inline` errors clearly if the target is self-recursive or used
as a value.

### Tests that never run twice

Tag a zero-arg `-> Int` definition `ai:test`; returning 0 means pass.

```sh
$ ai-lang tag test_checkout ai:test
$ ai-lang meta test_checkout
{"tags":["ai:test"]}
```

Because the runner knows which tests are pure (effects are inferred — see
Part 2) and a pure test's result is a function of its hash, **a passing pure
test is never re-run**:

```sh
$ ai-lang test
  PASS  test_checkout
  1 passed, 0 failed, 0 cached

$ ai-lang test
  PASS  test_checkout  (cached)
  0 passed, 0 failed, 1 cached
```

Change anything the test transitively depends on and its hash changes, so the
cached result simply does not apply. There is no invalidation logic to get
wrong, and no stale green.

Tag a definition `ai:cli` and it becomes an entry point with generated
`--help`, flags, and defaults:

```sh
$ ai-lang cli                      # list entry points
$ ai-lang cli <name> --help
$ ai-lang cli <name> arg1 --flag
```

### Branches, history, diff, merge

Branching is O(1) and copies nothing, because definitions are shared by
construction:

```sh
$ ai-lang branch experiment
created branch experiment at a12a6ff87be02fb2 (defs shared, none copied)

$ ai-lang branches
  experiment
* main

$ ai-lang switch experiment
$ ai-lang rename checkout do_checkout

$ ai-lang diff main experiment
+ do_checkout                   fbd22f23556a1ba6
- checkout                      fbd22f23556a1ba6
```

That diff is worth reading twice: the *same hash* under two names. A rename is
a namespace delta, not a changed definition — which is exactly why it can
never produce a merge conflict in the code itself.

```sh
$ ai-lang history
  10 commit(s) on experiment
HEAD a2f7b7449dce03c0
     a4962be8007d4a36
     a12a6ff87be02fb2
     ...

$ ai-lang undo
undid last change; head is now a4962be8007d4a36 (defs are immutable, none destroyed)
```

Every name-changing operation auto-commits, so one `undo` reverses one such
operation. (A rename is a `set` plus a `remove`, so it takes two undos; a
propagating refactor moves several names and takes more.)

`merge <from> <into>` is a 3-way merge over names. A name changed differently
on both sides is a **reported conflict, never silently resolved**.

### Running code

```sh
$ ai-lang run main
900

$ ai-lang eval line_total '{"name":"w","cents":250,"qty":4}'
1000
```

`eval` takes JSON arguments matching the definition's parameter types and
prints the result as JSON, so any definition is directly callable without
writing a harness around it.

### Live deploy and instant rollback

A node keeps every deployed version resident, and a *binding* is a named
pointer at one hash. Deploying flips the pointer; rolling back flips it back.
Both are instant, because neither moves any code.

```sh
$ ai-lang serve --bind 127.0.0.1:7788 &

$ ai-lang deploy v1.ail 127.0.0.1:7788 --bind api=greet
binding api -> fc2ac61b (created)

$ ai-lang invoke 127.0.0.1:7788 api 41
42

$ ai-lang deploy v2.ail 127.0.0.1:7788 --bind api=greet
binding api -> 0ee15eb1 (was fc2ac61b)

$ ai-lang invoke 127.0.0.1:7788 api 41
4100

$ ai-lang rollback 127.0.0.1:7788 api
binding api -> fc2ac61b (was 0ee15eb1)

$ ai-lang invoke 127.0.0.1:7788 api 41
42
```

A binding's signature is pinned when it is created, so a later deploy cannot
change its type out from under callers. A live `state` whose type changed
requires an explicit `--migrate <state>=<def>` naming an `fn(OldT) -> NewT`,
typechecked **on the node against the live cell's type**. Dropping a live
state requires `--allow-state-drop`, and nodes can be token-protected with
`--token`.

### Structural editing as a service

```sh
ai-lang serve-edit      # one JSON request per line in, one response per line out
```

The same edit algebra, exposed for editors and agents rather than for a human
at a shell.

---

## Part 2 — Inferred effects and capabilities

Every definition gets an **effect signature**, inferred from the call graph.
You never annotate one, and code you did not write cannot lie about it.
Effects are `IO, Net, State, Atom, Mut, FFI`.

```sh
$ ai-lang effects square              $ ai-lang effects loud
square                                loud
  effects:   pure                       effects:   {IO}
  pure:      yes                        pure:      no
  mobile:    yes                        mobile:    yes
  cacheable: yes                        cacheable: no
```

Three guarantees are derived: **pure**, **mobile** (no FFI, no Atom — safe to
ship to another thread or node), **cacheable** (safe to memoize across `at()`).

It is effect-*polymorphic*: a higher-order function's effect is a function of
its arguments.

```sh
$ ai-lang effects twice               # twice(f, n) = f(f(n))
  effects:   pure
  + the effect of its argument(s): #0  (effect-polymorphic)
```

So `twice(square, n)` infers pure and cacheable, while `twice(loud, n)` infers
`{IO}`. It sees through returned closures too, so `make_runner(loud)()`
resolves to `{IO}` precisely rather than falling back to "could do anything".

**This is enforcement, not just a query.** A node carries an effect policy and
infers the effects of shipped code *before installing it*. Refused code is
never compiled and never runs.

```sh
# Default: compute + node state, but no process I/O, no network, no arbitrary C
AI_LANG_AT_EFFECTS=pure ai-lang serve
# → node effect policy [pure] forbids effect(s) [ffi] in shipped code
```

So you can accept and run someone else's code and *know* it is pure. The same
inference drives the `at()` result cache, and makes mobility a compile-time
check:

```rust
let counter = atom(0);
spawn(|| swap(counter, inc))
// typecheck: spawn thunk captures an `Atom`: threads are share-nothing, so a
// shared mutable cell can't be captured (this is what makes data races
// impossible). Capture a snapshot via `deref`, or model shared state as a
// node `state` reached through a message instead.
```

That last check is what makes threads share-nothing for mutable state by
construction, rather than by convention.

---

## Part 3 — Distribution: `state` and `at()`

A `state` binding is a **node singleton keyed by its content hash**: its
initializer runs exactly once per node, and installing a hash that is already
live is a no-op. `at(node, thunk)` ships a closure to a node and runs it
there, where `state` references resolve to *that node's* live cell.

```rust
enum Cmd { Bump(Int), Get }

state counter: Atom<Int> = atom(0)

def handle(c: Cmd) -> Int =
    match c {
        Cmd::Bump(d) => swap(counter, |n: Int| n + d),
        Cmd::Get     => deref(counter),
    }

def ask(node: Node, c: Cmd) -> Int =
    match at(node, || handle(c)) {
        Result::Ok(v)  => v,
        Result::Err(f) => why(f),
    }

def main() -> Int = {
    let n = tcp_node(127, 0, 0, 1, get_node_port(0));
    let _a = ask(n, Cmd::Bump(5));
    let _b = ask(n, Cmd::Bump(10));
    println(int_to_string(ask(n, Cmd::Get)))     // 15
}
```

```sh
ai-lang run main --nodes=1
```

The atom never travels; only the closure does, and only by hash. A node that
does not have the code asks for it, and the client ships the definitions. A
bare `Atom<T>` may not cross `at()` as a capture or a thunk return — that
would fork it — and the typechecker rejects it.

`--nodes=N` spawns N workers and exposes them via `node_count()` and
`get_node_port(i)`. `examples/gol.ail` runs a 7x7 Game of Life across 49
worker processes, one `at()` call per cell per generation.

---

## Part 4 — The language

### Structs and enums

Field and variant names are part of a type's identity, so renaming a field
produces a genuinely different type.

```rust
struct Point { x: Float, y: Float }
struct Item  { name: String, cents: Int, qty: Int }

enum Shape  { Circle(Float), Rect(Point) }
enum Cmd    { Bump(Int), Get }
```

Variants carry zero or one payload; multi-payload variants compose a struct.

### Functions, `let`, blocks

Every `def` declares its parameter and return types. A block is a sequence of
`let` bindings ending in an expression — that trailing expression is the value.

```rust
def line_total(it: Item) -> Int = it.cents * it.qty

def apply_discount(total: Int, pct: Int) -> Int = total - (total * pct / 100)

def checkout(it: Item, pct: Int) -> Int = apply_discount(line_total(it), pct)

def main() -> Int = {
    let it = Item { name: "widget", cents: 250, qty: 4 };
    println(int_to_string(checkout(it, 10)))     // 900
}
```

Operators dispatch on operand type, so `+ - * /` and the comparisons work on
both `Int` and `Float`:

```rust
def pi() -> Float = 3.14159265
def circle_area(r: Float) -> Float = pi() * r * r
```

### Pattern matching

```rust
def area(s: Shape) -> Result<Float, String> =
    match s {
        Shape::Circle(r) =>
            if r < 0.0 { Result::Err("negative radius") }
            else { Result::Ok(pi() * r * r) },
        Shape::Rect(p) => Result::Ok(p.x * p.y),
    }
```

### Generics

```rust
struct ListCell<T> { head: T, tail: List<T> }
enum   List<T>     { Cons(ListCell<T>), Nil }

// Fully generic — never touches a T as a value, only the spine.
def list_length<T>(xs: List<T>) -> Int = list_length_acc(xs, 0)

def list_reverse<T>(xs: List<T>) -> List<T> = list_reverse_acc(xs, List::Nil)

def first_or<T>(xs: List<T>, d: T) -> T =
    match xs {
        List::Cons(c) => c.head,
        List::Nil     => d,
    }
```

Type arguments are inferred bottom-up at construction sites: given
`Cons(ListCell { head: 1, tail: Nil })`, the field `head: 1` pins `T = Int`
and that flows up through `ListCell<T>` to `List<T>`.

### Errors are values — the language is total

There is no exception mechanism, no panic, and no unchecked tier. Fallible
operations return `Result<T, E>`, including array and bytes indexing.

```rust
enum Result<T, E> { Ok(T), Err(E) }

def safe_div(a: Int, b: Int) -> Result<Int, Int> =
    if b == 0 { Result::Err(404) } else { Result::Ok(a / b) }
```

`?` unwraps an `Ok` and early-returns an `Err` from the enclosing function,
which must itself return a `Result` with the same error type:

```rust
def compute(x: Int, d: Int) -> Result<Int, Int> = {
    let q = safe_div(x, d)?;
    let r = safe_div(q + 6, 2)?;
    Result::Ok(r + 1)
}
```

Even bounds checks flow through this protocol — an out-of-bounds index is an
ordinary `Err` value, not a trap:

```rust
def sum_arr(a: Array<Int>, i: Int, acc: Int) -> Result<Int, IndexError> =
    if i >= array_len(a) { Result::Ok(acc) }
    else { sum_arr(a, i + 1, acc + array_get(a, i)?) }
```

You pay nothing for this on the happy path: a checked access that succeeds
never materializes a `Result` at all.

### Closures and higher-order functions

```rust
def twice(f: fn(Int) -> Int, n: Int) -> Int = f(f(n))

def make_runner(f: fn(Int) -> Int) -> fn(Int) -> Int = |n: Int| f(n)

def demo() -> Int = twice(|n: Int| n * 3, 2)     // 18
```

### `defer` — deterministic cleanup

`defer expr;` runs `expr` when the enclosing block exits, on *every* path
including a `?` early-return, in LIFO order. It adds no binding, and it is the
alternative to finalizers for external resources.

```rust
def with_buffer(log: Ptr) -> Int = {
    let buf = malloc(16);
    defer free(buf);
    defer ptr_write_i64(log, 0, 1);
    let z = ptr_write_i64(buf, 0, 42);
    ptr_read_i64(buf, 0)
}
```

### Atoms — shared mutable identity

An `Atom<T>` is a mutable identity over an immutable value. `swap` applies a
*pure* function and lock-free compare-and-sets the result back.

```rust
def bump_twice() -> Int = {
    let c = atom(0);
    let _a = swap(c, |n: Int| n + 1);
    let _b = swap(c, |n: Int| n + 1);
    deref(c)                                     // 2
}
```

### Threads

`spawn` runs a zero-arg closure on a fresh OS thread; `join` blocks for its
result.

```rust
def tally(n: Int, acc: Int) -> Int = if n == 0 { acc } else { tally(n - 1, acc + n) }

def parallel_sum() -> Int = {
    let h1 = spawn(|| tally(1000, 0));
    let h2 = spawn(|| tally(1000, 0));
    join(h1) + join(h2)                          // 1001000
}
```

Threads are **share-nothing for mutable state by compile-time guarantee** —
see Part 2.

### C FFI

The outside world is reached by declaring C symbols directly. HTTP is libcurl,
crypto is libcrypto, OS is libc; JSON is a parser written in ai-lang itself.

```rust
extern "C" lib "c" {
    fn malloc(size: Int) -> Ptr
    fn free(p: Ptr) -> Int
    fn getenv(name: Ptr) -> Ptr
    fn clock_gettime(clk_id: Int, tp: Ptr) -> Int
}
```

Which is why `http_get` is a real request, not a shim:

```rust
def report(label: String, r: Result<String, String>) -> Int =
    match r {
        Result::Ok(body) => println(string_concat(label, body)),
        Result::Err(e)   => println(string_concat("failed: ", e)),
    }

def main() -> Int = report("example.com: ", http_get("https://example.com"))
```

---

## Performance

Identical algorithms in ai-lang, Rust (`-O3 + lto`), and Go. Each program
self-times its core workload, and the harness rejects a run if the languages
disagree on the checksum — so every benchmark is also a correctness test.

| benchmark | ai-lang | Rust | Go | vs Rust |
|---|---|---|---|---|
| fib | 9 ms | 6 ms | 6 ms | 1.5x |
| loop_mix | 433 ms | 412 ms | 418 ms | 1.05x |
| mandelbrot | 63 ms | 59 ms | 59 ms | 1.07x |
| nbody | 229 ms | 24 ms | 20 ms | 9.5x |
| binary_trees | 47 ms | 87 ms | 46 ms | **0.54x** |

`loop_mix` and `mandelbrot` are at parity; `binary_trees` beats Rust's
malloc/free and matches Go's GC. `nbody` is the outlier, and pays for the
checked-array protocol.

---

## Known issues

Two real bugs, both found while verifying the examples above:

1. **`inline`, `reorder-params`, and `extract` fail when any dependent calls an
   extern**, which includes `println` and most of the OS surface. They report
   `kernel invariant violated` and *refuse* the refactor rather than
   half-applying it, so nothing is corrupted. `update --propagate` is
   unaffected and can do the same work.

2. **`at()` can abort at runtime** if the program never destructures a
   `Failure` — for example if it matches `Result::Err(_e) => ...` with a
   wildcard. The first `at()` call then dies with:

   ```text
   ai_net_at: install_current_at_binding must be called before any at() in JIT
   thread caused non-unwinding panic. aborting.
   ```

   Workaround: match the `Failure` variants explicitly
   (`Failure::Crashed(_n)` and friends). This is a hard abort in a language
   whose stated invariant is that errors are values, so it should be a
   compile-time error instead.
