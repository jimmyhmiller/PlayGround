# How to write code in this repository

These are hard preferences, not suggestions. When a rule here conflicts with a
convention you learned elsewhere, this file wins. When you think a rule genuinely
should be broken, say so out loud and explain why — don't quietly break it.

## State

State is the root of most problems in programming. The point is not to eliminate it
but to make it visible, centralized, and easy to trace.

- Centralize state. Do not scatter it across the system.
- Prefer a single state value that everything reads from — a Clojure-style atom, an
  Elm-architecture model, one store. One place where state lives, and explicit,
  inspectable transitions between values of it.
- I should be able to point at exactly where state changes and how it got there.
- Everything around that core should be pure data transformations.

## Nothing is private

Everything is public. Always.

- Do not add `private`, `#field`, closures-as-encapsulation, or module-level hiding.
- Do not design an API surface that "protects invariants" by hiding internals.
- An application is not a library. It is meant to be modified, poked at, and relied
  on from the inside. Hiding internals removes leverage from the person reading the
  code, which is the opposite of what a system should do.
- If something genuinely must not be touched, an `_` prefix or an `internal` naming
  convention is the most I want — and even that is usually unnecessary.

## Tooling before features

Before building much of anything, build the means to see it run.

- Ways to exercise subparts of the system in isolation.
- Structured output I can pipe into a visualizer, not prose logs.
- Ways to inspect the running system.
- Ways to see the state, and its transitions, at any point.

If you are proposing a plan and there is no way to observe the thing you're about to
build, the observability is step one.

## Explicit over implicit

- If a function uses something, it takes it as a parameter. No reaching for ambient
  context, globals, module-level singletons, or implicit environment.
- Prefer named parameters / destructured object arguments where the language has
  them. Positional booleans and long positional argument lists are bad.
- I will take a performance hit for named arguments in ordinary application code.
  The exception is code where performance is the actual point — there, write the
  straightforward, obvious, fast thing.

## Abstractions

- Small, obvious, boring. No clever indirection, no frameworks-within-the-codebase,
  no "extensible" machinery built for a second caller that doesn't exist.
- This applies at both high level and low level.
- Duplication is cheaper than a bad abstraction. Do not deduplicate two things just
  because they currently look alike.

## Polymorphism and dispatch

- Polymorphism is usually the wrong tool. Avoid it unless there is a very strong
  reason.
- Prefer enums / algebraic data types / tagged unions, and dispatch with an explicit
  `switch` / `match` / `case` that lists every variant.
- I would rather see all the cases in one place than chase a dynamic dispatch through
  the codebase.

## Types

- I lean dynamic. Types are useful in places; they are not the goal.
- Do not contort the design to satisfy a type system. Do not build elaborate type-level
  machinery. Keep things open and modifiable.

## Comments and docstrings

- Do not write comments that restate what the code does.
- Do not write AI-style narration (`// Loop over the items`, `// Set the flag to true`).
  When in doubt, write no comment.
- Comments are for the non-obvious: a hack, a workaround, a subtle invariant, a reason
  something surprising is the way it is.
- No docstrings on application functions. What a function does changes; the docstring
  won't.

## Tests

- Tests are good. The right tests are end-to-end, round-trip, or snapshot tests that
  exercise real behavior.
- Do not write unit tests that reach deep into the middle of the system and require a
  pile of mocks and setup to reach the thing under test. If a test needs heavy mocking,
  that is a signal, not a thing to work around.

## I/O at the edges

- Data transformations are pure functions. I/O lives in its own function, at the edge,
  that wires the pure pieces together.
- Describe things as data wherever possible; interpret the data at the boundary.
- **No dependency injection. No inversion of control. No swappable I/O interfaces.**
  Inversion of control makes code extremely hard to read. If avoiding it means
  duplicating some logic, duplicate the logic. Only propose an abstraction here if you
  have a genuinely compelling reason, and say the reason out loud.

## File organization

- Not single-responsibility. Not one-class-per-file. Related things live together.
- Never create tiny files that hold one function or one export. That is the failure
  mode I care most about.
- Err on the side of fewer, larger, cohesive files. Split a file when it has actually
  become cumbersome, not preemptively.

## Naming

- Descriptive, but not long. If a name has to be enormous to be accurate, something
  upstream is wrong.
- **Never abbreviate.** No `str`, `btn`, `cfg`, `idx`, `req`, `res`, `tmp`, `msg`.
  Write the real word.

## Imports

- **Never use wildcard/star imports.** `import *`, `use foo::*`, `from x import *`,
  `:refer :all` — never, under any circumstances. This is a fundamental anti-pattern.
- Either import the specific names you use, or use a qualified alias
  (`[clojure.string :as string]` → `string/join`).

## JavaScript / TypeScript

- No classes. Ever.
- Immutable data by default.
- Plain functions and plain objects.
- Destructured named arguments.

## Rust

- Structs and enums, used heavily — they make the borrow checker tractable.
- Explicit `match` over trait-object polymorphism.
- Same rules otherwise: public by default, no wildcard imports, no abbreviations.
