# Library / Language split — charter & operating procedure

Governance for building Scheme (R7RS-small + R6RS numeric tower + hygienic macros
+ call/cc) on the reusable core, without the core bending to Scheme.

This is deliberately short. The boundary is **discovered, not designed** — you
cannot know it upfront (that is the lesson of the original toolkit, which bent to
its one homegrown driver). So there is no big upfront architecture plan here.
There is a criterion, a per-change checklist, a living map, and — most important —
structural enforcement. Docs describe the boundary; structure holds it.

## The criterion (one sentence)

**Library = mechanism behind an interface; Language = a policy that plugs into an
interface.** A thing belongs in the library *iff a genuinely different second
language would use the same mechanism in the same shape.* Syntax, the set of
special forms, and chosen semantics are the frontend; they plug in, they do not
reach in.

Default when unsure: **it starts in the frontend.** Graduating to the library
when a second consumer needs it is cheap; un-bending the core after a language
crept in is not.

## Invariants (non-negotiable)

1. **Axis-neutral IR.** The standard IR carries semantic ops (`Prim`, `Call`,
   `Dispatch`, `Lambda`) and never language- or strategy-specific nodes. Scheme
   lowers to it. No `Ir::CallCc`, no `Ir::SyntaxRules`.
2. **No language names in the core.** The `core` crate contains zero Scheme
   identifiers. A leak is a CI failure, not a style nit.
3. **Interface-or-frontend.** If the library must know a language-specific fact,
   it gains an *interface* the language plugs; the fact lives in the language.
4. **Two-consumer rule.** Nothing earns library status by "might be reused." It
   graduates only when a genuinely different second consumer needs the same
   shape. A live second frontend keeps this honest.
5. **Mechanism up, policy down.** Continuation capture/resume is library; `call/cc`
   is Scheme. The macro-expansion *driver* (re-entrant, GC-rooted) is library;
   `syntax-rules` hygiene is Scheme. Dispatch + inline caches are library; the
   *method model* (generic functions, protocols) is the language.

## The split map (living — update as you build)

| Concern | Library (mechanism / interface) | Scheme frontend (policy / plug) |
|---|---|---|
| Standard IR | axis-neutral `Ir` | lowering: special forms → `Ir` |
| Execution | `CodeSpace` tiers (interp, bytecode, JIT) | — (reused as-is) |
| Value model | `Repr`/`ModelEmit`; numeric-tower TYPES (fixnum/bignum/rational) + arithmetic | which tower levels are exposed; exactness/reader-number syntax; contagion policy |
| GC | moving collector, rooting, barriers | — |
| Dispatch | `Dispatch` (mega/IC), speculation | Scheme has little; a method model (if added) is the plug |
| Continuations | capture/resume, the frame/stack strategy | `call/cc`, `dynamic-wind`, multiple values (surface over the mechanism) |
| Eval driver | incremental read→expand→analyze→run; re-entrancy; rooting | the reader table; the set of special forms |
| Macros | "a macro is a transformer invoked during analysis" + hooks | `syntax-rules`/`syntax-case` hygiene algorithm |
| Runtime library | pairs, vectors, strings, hashtables, numeric ops | which are bound in `(scheme base)` etc. |
| Tail calls | proper-TCO calling convention/frame support | mandating TCO at the surface |

Rule of thumb reading the table: the noun on the left is library; the *specific
choice* of it is the frontend.

## Operating procedure (apply to every change)

Answer in order:

1. **Mechanism or policy?** Policy → frontend, done. Mechanism → go to 2.
2. **Would the live second consumer (Clojure/Lua stub) need this same shape?**
   No → frontend, or expose an interface it can plug differently. Yes → library,
   behind an interface.
3. **Does it touch the IR?** It must be axis- and language-neutral or it does not
   go in the IR.
4. **Run the gates** (below). If any is red, the split is wrong; fix the split,
   not the gate.

## Enforcement (the part that actually works)

- **Crate boundary.** `core` is the library crate; `scheme` depends on it and may
  use only its public API. Privacy is the cop: when Scheme wants a private
  internal, that is the design signal (frontend, or new interface).
- **Live second frontend.** A skeletal `clojure` (or `lua`) crate compiles
  against the same `core` interfaces. A change that breaks it, or that needs a
  Scheme-shaped interface, is a caught bend.
- **`core` name grep** (CI): fail if any Scheme identifier appears in `core`.
- **Grand-matrix test** stays green: the axes stay orthogonal and any-combination
  still holds.

## Anti-pattern we are preventing

The original toolkit had one homegrown driver, no second consumer, and no wall
between core and frontend. The "generic" core silently bent to it, and Clojure —
a genuinely different language — could not be built on it. No document prevented
that. A second real consumer plus a structural boundary would have.
