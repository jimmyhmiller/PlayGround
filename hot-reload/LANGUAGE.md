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
| `i64` | 64-bit integers |
| `bool` | `true` / `false` |
| `str` | immutable interned strings — `"hello\n"` (escapes: `\n \t \" \\`) |
| `()` | unit |
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
```

- Operators: `+ - * < > <= >= == != !`. On strings, `+` concatenates and
  `==`/`!=` compare by content (interning makes it an id compare).
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

**`letonce` globals.** Initialized exactly once, never re-run on reload — the
place native resources and long-lived state live so an edit changes *code*,
not the running world.

## Not yet (deliberately)

Arrays/collections, floats, first-class functions/closures, modules, match as
an expression, migration transformers in surface syntax (they are supplied
through the engine API / the REPL's `:migrate`). Each of these lands the same
way strings and enums did: on every tier at once, with its live-evolution
semantics defined before its syntax.
