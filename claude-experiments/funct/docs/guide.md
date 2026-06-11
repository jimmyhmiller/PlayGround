# The funct Language Guide

funct is a small, functional, embeddable scripting language. It has `|>` pipes
and UFCS, real pattern matching, immutable data, atoms as the only escaping
mutable state, and a stack-based bytecode VM whose state can be paused,
snapshotted, and resumed.

This guide is the language tour and reference. For every built-in function see
the [Standard Library Reference](stdlib.md). For embedding funct in a Rust
program see the [README](../README.md).

- [Running funct](#running-funct)
- [Values and types](#values-and-types)
- [Literals](#literals)
- [Operators](#operators)
- [Bindings](#bindings)
- [Blocks are expressions](#blocks-are-expressions)
- [Functions](#functions)
- [Closures](#closures)
- [Pipes and UFCS](#pipes-and-ufcs)
- [Lists, tuples, records, strings](#lists-tuples-records-strings)
- [Ranges](#ranges)
- [Control flow](#control-flow)
- [Pattern matching](#pattern-matching)
- [Variants and types](#variants-and-types)
- [Error handling: Result, Option, and `?`](#error-handling-result-option-and-)
- [Atoms](#atoms)
- [Modules](#modules)
- [Testing](#testing)
- [The host interface (`extern`)](#the-host-interface-extern)
- [Comments and statement separators](#comments-and-statement-separators)
- [Gotchas and deliberate restrictions](#gotchas-and-deliberate-restrictions)

---

## Running funct

```bash
funct run program.ft      # compile and run a file (the `run` keyword is optional)
funct program.ft          # same thing
funct test file-or-dir    # run #[test] functions (see Testing)
```

When run from the CLI, the script's directory is the module root (so
`import "math/vec"` resolves to `math/vec.ft` next to it).

---

## Values and types

Every value is one of these. `typeof(v)` returns the name shown.

| `typeof` | Description | Literal example |
|---|---|---|
| `Unit` | the empty value (like `()`/void) | `()` |
| `Bool` | boolean | `true`, `false` |
| `Int` | 64-bit signed integer | `42`, `1_000` |
| `Float` | 64-bit float | `3.14`, `1.0` |
| `Str` | UTF-8 string | `"hi"` |
| `List` | ordered, immutable sequence | `[1, 2, 3]` |
| `Tuple` | fixed-size heterogeneous group | `(1, "a", true)` |
| `Record` | string-keyed map (fields) | `{ x: 1, y: 2 }` |
| `Variant(Tag)` | a tagged value | `Some(1)`, `Circle { r: 2.0 }` |
| `Fn` | a function or closure | `x => x + 1` |
| `Atom` | the only mutable cell | `atom(0)` |
| `Range` | an integer range | `1..10`, `1..=10` |

All data is **immutable**. "Updating" a list or record returns a new value; the
original is untouched. The single exception is the [atom](#atoms).

Equality is **structural** for everything except atoms, which compare by
identity:

```text
[1, 2] == [1, 2]          // true
{ x: 1 } == { x: 1 }      // true
Some(1) == Some(1)        // true
atom(1) == atom(1)        // false — different cells
let a = atom(1); a == a   // true  — same cell
```

---

## Literals

```text
42            1_000_000        // ints; `_` separators are ignored
3.14          1.0              // floats
true   false                  // bools
"hello"                       // string
()                            // Unit (rarely written; it's what empty blocks produce)
```

**Strings** support interpolation with `${expr}` and the usual escapes. A bare
`{` or `}` is an ordinary character — only `${` begins interpolation — so
brace-heavy text like JSON needs no escaping:

```text
let name = "ada"
"hi ${name}!"             // "hi ada!"
"1 + 2 = ${1 + 2}"        // "1 + 2 = 3"
"a brace { and } here"    // "a brace { and } here" — braces are literal
"price $20"               // "price $20" — a $ not followed by { is literal
"esc \${not interp}"      // "esc ${not interp}" — backslash escapes the $
"a\nb"                    // newline escape
```

Strings are indexed and measured by **character**, not byte:

```text
"hello"[1]                // "e"
len("héllo")             // 5
```

---

## Operators

**Arithmetic** — `+ - * / % **`. `**` is exponent and is right-associative;
unary minus binds tighter than `**`.

```text
2 + 3 * 4                 // 14
2 ** 3 ** 2               // 512   (right assoc: 2 ** (3 ** 2))
-2 ** 2                   // 4      ((-2) ** 2)
```

Numeric rules:

- `Int op Int` stays `Int`. Integer `/` truncates, `%` is remainder: `7 / 2 == 3`, `7 % 3 == 1`.
- Mixing `Int` and `Float` produces `Float`: `1 + 0.5 == 1.5`.
- `1.0 == 1` is `true` (numeric equality across Int/Float).
- Integer operations **fault on overflow** rather than wrapping.
- `/` or `%` by zero **faults**.

**Comparison** — `< <= > >=` on numbers and on strings (lexicographic). `==`
and `!=` work on any values (structural; see above).

**Boolean** — `and`, `or`, `not`. `and`/`or` short-circuit, and conditions must
be real `Bool`s (there is no truthiness):

```text
false and (1 / 0 == 0)    // false — right side never runs
if 1 { 2 }                // ERROR: condition must be Bool
```

`+` also concatenates strings (`"a" + "b"`) and lists (`[1] + [2] == [1, 2]`).

---

## Bindings

`let` binds an immutable name. Re-binding the same name (shadowing) is allowed:

```text
let x = 1
let x = x + 1     // x is now 2 (a new binding)
```

Mutable locals use `let mut`, and exist **only inside functions** — there is no
mutable top-level state (use an [atom](#atoms) for that). Assignment uses `=`
and the compound `+=`:

```text
fn f() {
    let mut x = 1
    x = x + 1
    x += 3
    x                 // 5
}
```

Assigning to a non-`mut` local is a compile error. `let mut x` at the top level
is an error that points you to atoms.

**Destructuring** works in `let` (and function parameters and `for`):

```text
let (a, b) = (1, 2)
let [x, y, z] = [1, 2, 3]
let { x, y } = { x: 1, y: 2 }
```

---

## Blocks are expressions

A `{ ... }` block evaluates to its last expression. A block with no trailing
expression is `Unit`.

```text
let y = {
    let a = 2
    a * 3
}                     // y == 6
```

> Note: `{ x: 1 }` is a **record literal**, and `{ x }` is record shorthand for
> `{ x: x }`. A brace in expression position is read as a record when it looks
> like one. To force a code block, give it statements.

---

## Functions

Two forms: `=` for an expression body, `{ }` for a block body.

```text
fn double(x) = x * 2
fn add(a, b) {
    a + b
}
```

Functions may be **forward-referenced** (declaration order doesn't matter) and
recursive:

```text
fn a() = b() + 1
fn b() = 41           // a() == 42

fn fib(n) = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
```

`return` exits early:

```text
fn f(x) {
    if x > 10 { return 100 }
    x
}
```

Parameters can be **patterns** (destructuring):

```text
fn first((a, _)) = a
fn norm({ x, y }) = x * x + y * y
```

**Tail calls are optimized**: a self-call in tail position becomes a loop, so
deep tail recursion runs in constant native stack (`go(1000000, 0)` is fine).
Non-tail recursion uses heap-allocated frames, so it also won't blow the host
stack (`sum_to(50000)` works), though it does allocate.

---

## Closures

`=>` makes a lambda. Closures capture surrounding bindings.

```text
let f = x => x + 1
let g = (x, y) => x * y
let h = () => 42
let k = x => {              // block body
    let y = x * 2
    y + 1
}
```

Captured immutable bindings are captured by value. A captured `let mut`
slot is **shared** — closures see each other's updates:

```text
fn make_counter() {
    let mut n = 0
    () => { n = n + 1; n }
}
let c = make_counter()
c(); c(); c()              // 3

fn make() {
    let mut n = 0
    let inc = () => { n = n + 1; n }
    let get = () => n
    (inc, get)             // both share the same n
}
```

---

## Pipes and UFCS

`x |> f` feeds `x` as the first argument of `f`. `x |> f(a)` becomes
`f(x, a)`. Both are just sugar, so any function works as a method too.

```text
5 |> double               // double(5)
5 |> add(3)               // add(5, 3) == 8
[1, 2, 3, 4]
  |> map(x => x * 2)
  |> filter(x => x > 2)
  |> sum                  // 18
```

Use `_` as a **placeholder** to pipe into a different argument position:

```text
3  |> sub(10, _)          // sub(10, 3)
10 |> sub(_, 3)           // sub(10, 3)
```

You can also pipe into a [variant constructor](#variants-and-types):
`3 |> Some` makes `Some(3)`.

**UFCS** method syntax is the same idea: `x.f(a)` means `f(x, a)`.

```text
5.double()                // double(5)
[1, 2, 3].len()           // 3
(2.25).sqrt()             // sqrt(2.25)
```

When `x` is a record that actually **has** a field named `f`, field access wins
over UFCS:

```text
let r = { double: x => x * 3 }
r.double(5)               // 15 — calls the stored closure, not a global
```

---

## Lists, tuples, records, strings

**Lists** are immutable sequences. Index with `[i]` (out of bounds faults);
concatenate with `+`; grow with `push` (returns a new list).

```text
[1, 2, 3][1]              // 2
[1, 2] + [3]              // [1, 2, 3]
push([1], 2)              // [1, 2]
```

**Tuples** are fixed-size and indexed positionally:

```text
(1, "a", true)[2]         // true
```

**Records** are string-keyed. Access with `.field`, build with `{ k: v }`, use
`{ k }` shorthand, and update with spread `{ ..base, k: v }` (which returns a
new record — the original is unchanged):

```text
let p = { x: 1, y: 2 }
p.x + p.y                 // 3

let x = 9
{ x }.x                   // 9   (shorthand for { x: x })

let p2 = { ..p, x: 10 }   // { x: 10, y: 2 }; p still { x: 1, y: 2 }
```

See the stdlib for `keys`, `values`, `entries`, `has`, `get`, `assoc`,
`dissoc`, `merge`, and the nested `get_in`/`assoc_in`/`update_in` helpers.

---

## Ranges

`a..b` is half-open (excludes `b`); `a..=b` is inclusive. Ranges are lazy —
materialize with `to_list`, iterate in a `for`, or map over them.

```text
to_list(1..4)             // [1, 2, 3]
to_list(1..=4)            // [1, 2, 3, 4]
1..=3 |> map(x => x * 10) // [10, 20, 30]
for x in 1..=10 { ... }
```

---

## Control flow

**`if` / `else`** is an expression; the condition must be a `Bool`. An `if`
with no `else` that doesn't run yields `Unit`.

```text
if 1 < 2 { "yes" } else { "no" }
fn grade(n) = if n > 89 { "A" } else if n > 79 { "B" } else { "C" }
```

**`while`** loops while its condition holds:

```text
let mut i = 0
while i < 5 { i = i + 1 }
```

There is no dedicated `loop` keyword; write `while true { ... break }`.

**`for ... in`** iterates lists, ranges, and strings, and can destructure:

```text
for x in [1, 2, 3] { ... }
for x in 1..=10 { ... }
for c in "abc" { ... }              // c is each character as a Str
for (a, b) in [(1, 2), (3, 4)] { ... }
```

**`break` / `continue`** work in both loops, including as match-arm bodies.
They affect the nearest enclosing loop only; a lambda defined inside a loop
**cannot** break it (that's a compile error). `break`/`continue` outside a loop
is a compile error.

```text
for x in 1..=100 {
    if x > 4 { break }
    if x % 2 == 0 { continue }
    ...
}
```

---

## Pattern matching

`match` tries arms top to bottom and evaluates the first that matches. If none
match it **faults** (`no pattern matched`).

```text
match 2 {
    1 => "one",
    2 => "two",
    _ => "many",          // wildcard
}
```

Patterns include:

```text
// variants (record-style and positional)
match s {
    Circle { radius } => 3.14 * radius * radius,
    Square { side } => side * side,
    Point => 0.0,
}

// Option / Result
match Some(41) { Some(x) => x + 1, None => 0 }
match Err("boom") { Ok(v) => v, Err(m) => m }

// guards
match n {
    x if x > 0 => 1,
    x if x < 0 => -1,
    _ => 0,
}

// lists, with a rest binding
match xs {
    [] => 0,
    [x] => x,
    [first, ..rest] => first + len(rest),
}

// or-patterns
match n { 1 | 2 | 3 => "low", _ => "high" }

// range patterns
match n { 0..10 => "digit", 10..=99 => "two", _ => "big" }

// as-binding (name the whole matched value)
match [1, 2] { [a, _] as whole => a + len(whole) }   // 3

// nested
match Some((1, [2, 3])) { Some((a, [b, c])) => a + b + c, _ => 0 }   // 6

// tuple / record (record subject needs parens — see Gotchas)
match ({ x: 1, y: 2 }) {
    { x: 0, y } => y,
    { x, .. } => x * 100,
}
```

A **subjectless** `match { ... }` is an anonymous one-argument function — handy
to name a classifier or drop into a pipe:

```text
let classify = match {
    0 => "zero",
    n if n > 0 => "pos",
    _ => "neg",
}
classify(5)               // "pos"
7 |> match { 0 => "zero", _ => "other" }   // "other"
```

---

## Variants and types

A **variant** is a tagged value. Variant tags are dynamic — any capitalized
name followed by `{ ... }` (named fields) or `( ... )` (positional fields)
constructs one. You don't have to declare anything first:

```text
Circle { radius: 2.0 }    // named-field variant, tag "Circle"
Pair(1, 2)                // positional variant, tag "Pair"
```

`Some`, `None`, `Ok`, and `Err` are just conventional variant tags used by the
stdlib for [Option and Result](#error-handling-result-option-and-).

You read a variant's contents by matching it. Arity matters — `Pair(1, 2)` does
not match `Pair(a)`.

A `type` declaration documents a sum type but creates **no runtime binding**
(tags are dynamic, annotations are discarded). It's useful as documentation and
to keep match arms readable:

```text
type Shape = Circle { radius: Float } | Square { side: Float } | Point
```

---

## Error handling: Result, Option, and `?`

There are no exceptions for ordinary errors. Functions return `Ok(v)`/`Err(e)`
or `Some(v)`/`None`, and `?` propagates the failure case:

```text
fn safe_div(a, b) = if b == 0 { Err("div by zero") } else { Ok(a / b) }

fn calc(a, b) {
    let x = safe_div(a, b)?    // unwraps Ok, or returns Err early
    Ok(x + 1)
}
calc(10, 2)                    // Ok(6)
calc(1, 0)                     // Err("div by zero")
```

`?` works on `Option` too — it unwraps `Some` or returns `None` early:

```text
fn first_doubled(xs) {
    let h = head(xs)?
    Some(h * 2)
}
```

`unwrap_or(opt_or_result, default)` collapses either type to a plain value.

A **fault** is different from an `Err`: faults are for programmer/contract
errors (out-of-bounds index, division by zero, type mismatch, `no pattern
matched`, a failed `assert`). They abort the current run with a clear message
rather than being caught — the equivalent of a panic.

---

## Atoms

An **atom** is the only mutable, escaping cell in the language. Everything else
is immutable, so all shared mutable state goes through atoms.

```text
let a = atom(41)
@a                        // 41   — @a is sugar for deref(a)
@a + 1                    // 42
a.value                   // 41   — field-style deref

swap!(a, x => x + 1)      // apply a function; returns the new value
reset!(a, 9)              // set directly; returns the new value
```

`@a`, `deref(a)`, and `a.value` are three spellings of the same read.

Atoms compare by **identity** (see [Values](#values-and-types)), so they have
stable identity through closures and snapshots.

**Watchers** fire after each successful mutation, receiving `(old, new)`:

```text
let log = atom([])
watch(a, "key", (old, new) => swap!(log, xs => push(xs, (old, new))))
unwatch(a, "key")
```

For nested state there are path-based atom helpers (`swap_in!`, `reset_in!`) and
the pure `get_in`/`assoc_in`/`update_in` — see the [stdlib](stdlib.md).

A common pattern: copy the atom out, build a new value functionally, write it
back.

```text
let s = @state
reset!(state, { ..s, score: s.score + 1 })
```

---

## Modules

Modules are files. `import "math/vec"` resolves to `math/vec.ft` under the
module root (the script's directory via the CLI, or `vm.set_module_root(..)`
when embedded). Only `export`ed items are importable, and **imports are
explicit only — there is no wildcard import**.

```text
// math/vec.ft
export fn lerp(a, b, t) = a + (b - a) * t
export let origin = (0.0, 0.0)
fn helper(x) = ...                       // module-private
```

```text
import { lerp, clamp } from "math/vec"   // named
import { lerp as mix } from "math/vec"   // renamed
import "math/vec" as vec                 // qualified: vec.lerp(...)
import "math/vec"                        // alias defaults to last segment: vec
```

A module sees the prelude, registered host natives, its own definitions, and
its imports — nothing else (no access to the main program's globals or other
modules' internals). Modules load once and are cached (so atom state is shared
across importers), import cycles are detected and reported with the chain, and
`vm.reload_module(path)` hot-swaps a module's functions for all importers.

---

## Testing

Annotate any zero-argument function with `#[test]` — beside the code it tests or
in a separate file — and run `funct test <file-or-dir>`:

```text
export fn area(s) = ...

#[test]
fn squares_have_side_squared_area() {
    assert_eq(area(Square { side: 3.0 }), 9.0)
}
```

Each test runs in a fresh engine (full isolation; top-level atoms reset), and
the test file's directory is the module root so a test can import the module
under test. Discovery is by annotation only — importing a module never runs its
tests, and a `#[test]` fn is an ordinary fn otherwise.

Assertions (`assert`, `assert_eq`, `assert_ne`, `fail`) fault with the failing
function, line, and both values. See the [stdlib](stdlib.md#assertions).

---

## The host interface (`extern`)

When funct is embedded, the host program registers native functions and
injects globals. A script declares what it expects from the host with `extern`,
so it stays compilable and testable on its own:

```text
extern let canvas_w                       // a host-provided global value
extern fn request_render()                // a host-provided native function
extern fn mask_paint(name, x, y, r, v)
```

If the host registered a matching native/global, the `extern` binds to it. If
not, the name still compiles, and faults **loudly only if actually called** — so
pure-logic tests in a widget file run fine under `funct test` without a host.

Externs are part of a module's export surface, so a shared interface file can be
pulled in by any import form:

```text
import "host"                             // bare: brings in the whole surface
import { mask_paint } from "host"         // named
import { mask_paint as paint } from "host"
import "host" as h                        // then h.mask_paint(...)
```

For the Rust side (registering functions, types, and globals; calling script
functions; snapshots), see the [README](../README.md).

---

## Comments and statement separators

```text
// line comments run to end of line
1 + 1 // trailing comments are fine
```

Statements are separated by newlines or `;`:

```text
fn f() { let a = 1; let b = 2; a + b }
```

A multi-line boolean condition may continue on the next line with a leading
`and`/`or`.

---

## Gotchas and deliberate restrictions

- **`match {` is always subjectless.** To match a record *literal*, parenthesize
  it: `match ({ x: 1 }) { ... }` (the same restriction Rust has for struct
  literals in conditions).
- **Lambdas inside match guards:** a bare `ident =>` is read as the arm's arrow.
  Parenthesize a lambda used inside a guard.
- **No truthiness.** `if`, `while`, and `and`/`or` require real `Bool`s.
- **No top-level mutation.** `let mut` and assignment live inside functions;
  top-level shared state is an [atom](#atoms).
- **Integer overflow faults** (no wrapping); `/` and `%` by zero fault.
- **No wildcard imports.** Use named or qualified imports; `import *` is a
  dedicated error.
- **`type` creates no binding.** It's documentation; variant tags are dynamic.
- **Gradual types are deferred.** Type annotations in `type` definitions parse
  and are discarded; match exhaustiveness is checked at runtime (a missed case
  is a `no pattern matched` fault, not a compile error).
- **`index_of` returns `Some(i)`/`None`**, never `-1`.
- **Atoms are equal only to themselves**; all other values use structural
  equality.
