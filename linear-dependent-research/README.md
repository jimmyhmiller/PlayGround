# Tally — research notes

> Working codename: **Tally** (from *tallying* uses — resource accounting is the
> heart of the system). This is a research repository, **not** an implementation.
> The goal is to build and understand the formalisms a language like this would
> rest on before writing a line of compiler.

## The target

A programming language that is:

- **Dependently typed** — types can mention values; we can state and prove
  precise specifications (lengths, bounds, protocols, invariants).
- **Linearly typed** — values carry *multiplicities*; resources (memory, file
  handles, capabilities) are tracked so that "use exactly once" / "use at most
  once" is enforced by the type system.
- **Incredibly low level** — we want to express what C expresses: raw pointers,
  manual allocation/free, in-place mutation, explicit layout, no mandatory GC.
- **100% memory safe** — no use-after-free, no double-free, no out-of-bounds, no
  data races, *despite* operating at that low level.
- **Shen-like** — the type system is *programmable*: typing rules are data, the
  checker is (in part) a logic/constraint engine, and users can extend the rules
  rather than being stuck with a fixed set.

Prior art that gets *part* of the way:

- **ATS** — dependent types + linear types + "views" (separation-logic-style
  assertions about memory). Closest existing relative. We borrow heavily.
- **Idris 2** — dependent types + Quantitative Type Theory (multiplicities). The
  cleanest existing account of how linear + dependent *coexist*.
- **Rust** — affine ownership + borrows + regions, but no dependent types and a
  fixed (non-programmable) checker.
- **L3 / Linear Locations** (Ahmed, Fluet, Morrisett) — the minimal core showing
  how to make raw pointers + strong update + free memory-safe.
- **Shen** — a sequent-calculus, logic-programming-based, *programmable* type
  system; optional/Turing-complete checking.

## The three load-bearing ideas

1. **Quantitative Type Theory (QTT)** gives us *one* type theory that is both
   dependent and linear, and — via multiplicity `0` — makes the dependent layer
   **runtime-erasable**. This is what reconciles "linear" with "used in a type"
   and what keeps the abstraction cost at zero. See `docs/01-qtt-core.md`.

2. **The pointer/capability split** (L3, generalized by ATS "views") gives us
   C-level memory operations that are safe: pointers are cheap and copyable, the
   *permission to dereference* is linear. Strong update and manual `free` fall
   out soundly. See `docs/02-memory-views.md`.

3. **A programmable, stratified judgment** (ATS statics + Shen-style rules-as-data)
   gives us the extensibility and the decidability story: a decidable static
   index/constraint domain underneath, an extensible Horn-clause rule layer on
   top. See `docs/03-programmable-checker.md`.

## Map of documents

| File | What it covers |
|------|----------------|
| `docs/00-design-goals.md`  | Goals, non-goals, positioning, the hard tensions and how we intend to resolve each. |
| `docs/01-qtt-core.md`      | Quantitative Type Theory: rigs, multiplicity-annotated contexts, the 0/1 judgment, the erasure theorem, a core calculus (λ-Tally). |
| `docs/02-memory-views.md`  | The memory model: locations, pointers vs. capabilities/views, separating conjunction, `alloc/read/write/free`, strong update, regions, fractional/borrow permissions. |
| `docs/03-programmable-checker.md` | Stratified statics + Shen-style rules-as-data; constraint domains; decidability and the termination/fuel story. |
| `docs/04-roadmap.md`       | What to formalize next, open problems, proof obligations, a possible mechanization path. |
| `notes/bibliography.md`    | Annotated references (papers, systems). Citation details to be verified before relying on them. |
| `notes/glossary.md`        | Terminology, kept consistent across docs. |

## Status

Early. `00`–`03` sketch the formal core; `04` is the working agenda. Everything
here is a draft meant to be argued with.
