# The Live & Typed surface language

A statically-typed language in which **every definition can be redefined while
the program runs**. Types are enforced across edits: a change that breaks code
marks it Broken at install (never silently), running computations freeze at the
exact point a stale assumption would be *used* (never crash), and live data
migrates lazily across type versions. This file is the language surface;
`RUNTIME_DESIGN.md` explains the machinery, `UNIFICATION.md` the engine.

## Types

| type | values |
|---|---|
| `i64` | 64-bit integers (`/` traps on a zero divisor — resumably) |
| `f64` | IEEE doubles — `1.5`, full IEEE comparison semantics |
| `bool` | `true` / `false` |
| `str` | immutable interned strings — `"hello\n"` (escapes: `\n \t \" \\`) |
| `()` | unit |
| `[T]` | a mutable, growable array — `[1, 2, 3]`, `a[i]`, `a[i] = e;`, `len(a)`, `push(a, e)`; `let xs: [T] = [];` for empty |
| `fn(T, …) -> R` | a first-class function value — a bare `name` is one |
| `Name` | a reference to a struct or enum value (GC-managed; no `&` — every nominal value is a heap reference) |
| `foreign type Name` | an opaque native handle (never migrates, never traced) |

## Items

```text
struct Account { balance: i64, fee: i64 = 0 }        // defaulted fields enable
                                                     // auto-migration on evolve
enum Shape {
    Circle { r: i64 },
    Rect { w: i64, h: i64 },
    Point,                                           // fieldless variant
}

fn area(s: Shape) -> i64 { ... }                     // recursion + mutual recursion fine

letonce window = open_window();                      // runs ONCE; survives every
                                                     // reload (native state lives here)
foreign type Window;
foreign fn draw(w: Window, n: i64) -> ();            // host binds the native impl
```

## Statements & expressions

```text
let x = expr;            x = expr;              return expr;
if cond { ... } else { ... }                    while cond { ... }
emit(expr);              yield;                 f(a, b)
a.field                  Account { balance: 100 }
Shape::Circle { r: 5 }   Shape::Point
match s {
    Circle { r } => { ... }                     // binds the variant's fields
    Rect { w, h } => { ... }
    Point => { ... }
}
let a = match s { Circle { r } => 3 * r * r, Point => 0 };   // match is an expression
let xs = [1, 2, 3];      xs[0] = 9;      push(xs, 4);      len(xs)
let f = double;          f(21)           // a function VALUE; calls late-bind
```

- Operators: `+ - * / < > <= >= == != !` and unary `-`. On strings, `+`
  concatenates and `==`/`!=` compare by content (interning makes it an id
  compare); on `f64` everything is IEEE (including `NaN != NaN`).
- `let x: T = e;` — annotations are optional except where nothing else pins
  the type (`let xs: [i64] = [];`).
- A block's trailing expression is an implicit return.
- `yield;` is an explicit safe point: hosts that step a program stop there to
  interleave, and it is the natural place to watch edits land. (Edits do NOT
  require yields — they land between any two instructions of a running
  program.)
- `emit(e)` appends to the engine's observable output — the effect stream
  tests and the REPL display.

## What editing live means, per feature

**Functions.** A redefinition takes effect at the *next call* (late binding).
Frames already running keep their pinned version to completion.

**Structs.** A new version installs with the next call/read picking it up.
Live objects migrate *lazily at their next field access*:
- additive + defaulted changes migrate automatically;
- a representation change (e.g. `i64` → `Money`) is a **gap**: code that
  reaches such an object freezes (`MissingMigration`) until you install a
  transformer, then resumes in place.

**Enums.** The showcase:
- **Adding a variant** re-verifies every function whose `match` covers the
  enum; a now-non-exhaustive match becomes Broken *at install* — a running
  program traps at its next call to it (it can never silently fall through an
  arm), and re-evaling the match with the new arm repairs and resumes it.
- **Removing/reshaping a variant** gaps *only that variant's objects*; sibling
  variants keep migrating via auto-derived identity mappings. Installing a
  per-variant migration (old variant → some new variant + field sources)
  un-gaps them; enum migrations merge per variant, so repairing one gap never
  discards the derived mappings.
- A `match` on an object built under an older version crosses the migration
  barrier first — pinned old code that meets a migrated-in variant it has no
  arm for freezes at the match (con-freeness), never misbehaves.

**Strings.** Interned and immortal: a string in a pinned native frame can
never dangle, so strings never participate in migration at all.

**Arrays.** Structural, not versioned: only nominal types migrate. An array is
fixed to its element type and *every write is checked against it* — pinned old
code can never publish an ill-typed element (it freezes at the write). Arrays
of structs hold references, so the structs inside migrate lazily as usual;
out-of-bounds is a resumable freeze, not a crash.

**Function values.** A function value is the function's NAME, not a code
pointer. Calling one resolves the current version at call time — so a stored
handler (in an array, a `letonce`, a struct field) picks up every live
redefinition with no re-registration. A breaking edit to the referenced
function traps the indirect call exactly like a direct one; repairing the
function repairs every stored reference at once.

**Floats & division.** `x / 0` freezes at the division (supply a value with
the repair API and the frame resumes); float division follows IEEE and never
traps.

**`letonce` globals.** Initialized exactly once, never re-run on reload — the
place native resources and long-lived state live so an edit changes *code*,
not the running world.

## Not yet (deliberately)

Closures (lambdas with captures — function *values* exist, capture-free),
maps/other collections, modules, generics, migration transformers in surface
syntax (they are supplied through the engine API / the REPL's `:migrate`).
Each of these lands the same way everything above did: on every tier at once,
with its live-evolution semantics defined before its syntax.
