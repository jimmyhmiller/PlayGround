# Namespacing & name resolution

Coil's module system is **Clojure-like**: every module names its own definitions,
nothing leaks between modules without an explicit import, and one namespace —
`coil.core` — is auto-referred everywhere so the builtins are always in scope.

This doc covers what is namespaced, how a name resolves, and the one subtle part:
why name resolution runs in **two phases** when staged macros (`(meta …)`) are
present, and why the undefined-reference check is gated on the final phase.

Implemented across commits `fd8c75072` (macros), `36a0f34ba` (coil.core + trait
names), `f65ab9213` (trait methods + the undefined-reference check).

## The forms

```lisp
(module app)                              ; this file's namespace
(import "lib/control.coil" :use *)        ; refer ALL of control's exported names
(import "lib/slice.coil"   :use [slice-for push])  ; refer just these
(import "lib/io.coil"      :as io)        ; qualified access only: (io/print …)
(export foo Bar)                          ; what THIS module exposes (default: all public)
```

- `import` alone (`(import "x.coil")`) makes a module's names reachable **only**
  qualified via an alias — it does **not** refer anything. This matches Clojure's
  `require` (vs `require … :refer`). It is a behavior change from the old global
  model, where importing a file dumped its macros into the global namespace.
- `:use *` / `:use [names]` = Clojure's `:refer :all` / `:refer [names]`.
- `:as alias` enables `alias/name`.

## What is namespaced

Everything a module defines is renamed `module.name` internally and resolved
through module scope:

| Kind | Example | Notes |
|---|---|---|
| Functions | `app.helper` | `main` is never renamed (the entry point) |
| Structs / sums | `app.Point`, `app.Some` | variant constructors too |
| **Macros** | `control.when` | a bare `(when …)` expands only if `when` is a macro visible here |
| **Trait names** | `coil.core.Eq`, `app.Show` | you can define your own `Eq` without colliding |
| **Trait methods** | `=`, `show` | name→all declaring traits; a call picks the one in scope |
| Conventions | `app.fast2` | |

Deliberately **not** namespaced: **externs** (C symbols are global by nature —
`malloc` is `malloc`), and **consts** (flat global, like C `#define`).

## How a bare name resolves (module M)

In order:

1. **M's own** definitions → `M.name`.
2. A bare **extern** → left as-is (C symbol).
3. A **`:use`d** module's *exported* name → `target.name`.
4. **`coil.core`** (auto-referred) → `coil.core.name`.
5. Otherwise: for a **call**, a hard "undefined function" error (see gating below);
   for other positions, left bare for a later pass.

`alias/name` skips straight to M's `:as` aliases (export-checked). A `.`-qualified
head (`control.for`) is hygiene-generated and trusted as already-resolved.

The same scoping is mirrored in three places, by design:

- `resolve.rs::resolve` — the canonical name resolver (functions, types, traits, …).
- `lib.rs::resolve_macro` — which `(head …)` calls are macro calls.
- `comptime.rs::resolve_in_module` — **referential hygiene**: a symbol written in a
  macro template resolves in the *macro's* namespace (own defs + the macro's own
  `:use`d imports), not the use site. This is why `slice-for`'s generated `(for …)`
  becomes `control.for` regardless of who calls `slice-for`, so a library's macros
  work without the user importing the library's dependencies.

## coil.core (the prelude)

`src/prelude.coil` is `(module coil.core)`, compiled into the compiler and
auto-loaded. It defines the operator traits (`Eq`/`Hash`/`Add`/`Sub`/`Mul`/`Div`/
`Rem`/`Ord`) and their `i64`/`f64` impls, so `=`, `+`, `<`, `hash`, `case` work in
any module with **no import** — exactly like `clojure.core`. The auto-refer is the
last step of name resolution (`resolve.rs`, `CORE` constant). Diagnostics strip the
`coil.core.` prefix (`check.rs::display_name`) so errors read `Eq`, not
`coil.core.Eq`.

## Trait-method resolution (the per-module part)

Method names are **not** globally unique. `check.rs` maps a method name to *every*
trait that declares it; a call resolves it like so (`resolve_method` /
`trait_visible`):

- **one** candidate (every operator) → use it directly.
- **several** (two modules each define a trait with a `show` method) → keep only the
  ones whose trait is visible in the caller's module (own / `:use`d / `coil.core`).
  Exactly one visible → use it; **zero or many** → a hard error telling you to
  `:use` the one you mean or rename.

`check::check_with(program, imports, exports)` threads the import tables for this;
plain `check::check` passes empty tables (fine for single-module and the comptime
sub-program checks, where every method is single-candidate anyway).

## The undefined-reference check and the staged-resolution gating

A still-bare callee after resolution that is **neither an extern nor a trait
method** is an undefined reference, reported at resolve time (`resolve.rs`: a
`bare_ok` allowlist = externs ∪ method names).

This check is **gated** by a `strict: bool` flag on `resolve_program`, and that
gating exists because of staged macros.

### Why two resolve passes

`(meta …)` runs Coil code at compile time and **splices the definitions it returns
into the program**:

```lisp
(defn gen [] (-> Code) `(defn answer [] (-> i64) 42))
(meta (gen))                       ; produces (defn answer …) at compile time
(defn main [] (-> i64) (answer))   ; refers to a name that doesn't exist yet
```

When the resolver first walks `main`, `answer` does not exist — it only appears once
`gen` *runs*. But to run `gen`, you must resolve + check it (it's a real function).
Chicken-and-egg → resolution is staged (`lib.rs::elaborate`):

1. **Intermediate resolve** (`strict = false`): produce a well-formed `program` so
   the generators can be found and checked. The program still references
   not-yet-generated names, so the undefined check is **off** here.
2. **Run the metas**: `gen` executes → `(defn answer …)`.
3. **Final resolve** (`strict = true`): re-resolve
   `tagged2 = original forms − meta forms + generated forms`. Now `answer` exists and
   resolves; `totally-undefined` is correctly rejected.

`strict` is also `false` for the macro-detection subset (an intentionally
incomplete program). It is `true` whenever the program is final:
`!has_metas` for the no-meta path, and the `program2` resolve for the meta path.

**Soundness:** `program2` is a superset of all real runtime forms, so every name
still undefined after generation is caught. Proven by
`undefined_call_still_caught_in_a_meta_program` (a meta program that calls both a
generated def and a genuinely-undefined fn — the undefined one still errors).

### This is not a Coil quirk

Definition-generating metaprogramming forces name resolution into ≥2 phases in every
language that has it; the universal rule is *the early phase must not treat
not-yet-generated names as errors*:

- **Rust**: pipeline is parse → macro expansion → name resolution; resolution runs
  after expansion precisely because macros generate items names refer to.
- **Racket**: formal phase levels + an expander that collects a body's definitions
  before resolving references (a fixpoint loop — the general form of Coil's two-pass).
- **Template Haskell**: `$(...)` splices emit declarations; a staging restriction +
  declaration groups bring them into scope before dependent code is checked.
- **C++**: two-phase name lookup in templates defers dependent names to instantiation.

Coil's two separate passes are the coarse-grained version. If meta-generated code
ever needed to contain *more* metas, the two passes would become a fixpoint loop
(as in Racket).

## Tests

- `tests/modules.rs` — `macros_are_namespaced_not_global`,
  `macro_references_resolve_in_defining_module`,
  `second_order_macro_hygiene_across_modules`,
  `trait_methods_are_namespaced_per_module`, `undefined_call_is_a_resolve_error`,
  export/`:as`/`:use` coverage.
- `tests/meta.rs` — generation + `undefined_call_still_caught_in_a_meta_program`
  (the gating soundness test).
- `tests/traits.rs`, `tests/operators.rs` — trait/operator dispatch (the
  single-candidate method path).
