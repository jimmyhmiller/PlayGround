# 08 — Prior art: Vale's "Higher RAII", and where λ-Tally sits

Evan Ovadia's [*Higher RAII, and the Seven Arcane Uses of Linear Types*](https://verdagon.dev/blog/higher-raii-uses-linear-types)
(May 2024) is the closest thing in the wild to one half of what we are doing,
and reading it against our artifacts sharpens the design. This doc captures the
comparison: what Vale's Higher RAII is, where we independently arrived at the
same place, the one deep divergence (what linearity is *for*), the striking
convergence (his `LinearKey` is our cursor; his footnote is our mechanism), and
the features worth stealing.

## What Higher RAII is

- **Vale's working definition of "linear"** is not the textbook "no alias, no
  copy, use exactly once". It is: **a linear value must eventually be
  _explicitly_ destroyed; it cannot silently go out of scope** (doing so is a
  compile error). Vale linear structs may be aliased, copied, and read freely;
  it is the *owning reference* that is linear, and "use" means "destroy".
- **Higher RAII = linear types + whitelisted destroyers.** A value that must be
  destroyed by a *specific* function (or set of them), with the raw `destruct`
  callable only inside the defining module. So you can force "this `Transaction`
  must eventually be `commit`ted or `rollback`ed", and the compiler holds you to
  it across scopes.
- **vs. ordinary RAII** (C++ destructors, Rust `Drop`): those give you *one
  implicit, zero-argument, void-returning* destructor. Higher RAII gives
  **multiple named destructors, with parameters and return values, called
  explicitly**, and — unlike `defer` — the obligation can be handed to a caller
  and discharged *past* the end of the current function.
- **Why mainstream languages can't:** C++ allows one destructor; Rust's borrow
  checker dislikes references in structs, and (Aria Beingessner's point) the
  whole generic ecosystem assumes `T: Drop`, so `Vec::pop` etc. assume they may
  drop a `T`. Retrofitting "must-move" types breaks that assumption everywhere.
- **The eight (he says seven) uses:** cache-consistency tokens; linear futures
  (don't drop a result — deeper than `#[must_use]`, which only looks at the
  current scope); `FutureVow` (must resolve a promise); typestate
  `InProgressEvent → Event` (drive the state machine to its final state); linear
  messages/channels (must handle; must drain before dropping the receiver);
  `Transaction` (must decide); `OwningIndex` (prevent leaks in "collectionized"
  graph structures); and `LinearKey` (solve lookup-after-remove).
- **Framing:** correctness = *safety* ("bad things don't happen") + *liveness*
  ("good things do happen"); Higher RAII is a rare tool for **liveness**, which
  he says no language has nailed. He points at Austral and Mojo as likely
  mainstream vehicles.

## Where we independently arrived at the same place

Our strict linearity — the `List` in `poc/tally_dll.py` cannot be implicitly
dropped; forgetting is a compile error; you must drain it and `drop_empty` —
*is* a linear struct in Vale's sense. Our checker error

    leak: list `l` is live at end of scope; must drop_empty it

is Vale's `Undestructed linear` error. And our `free` / `pop_front` /
`remove(cursor) -> elem` / `drop_empty` are exactly his **named destructors with
parameters and return values**, the thing he stresses C++/Rust's single `Drop`
cannot express. We get this for free because our consumption is always an
explicit function call, never an implicit `Drop`.

So: **we built Higher RAII without setting out to.** The leak-freedom theorem in
`agda/CombinedSound.agda` (a closed well-typed program leaves no live cell) is
the machine-checked statement of the obligation Vale's Higher RAII enforces.

## The deep divergence: what linearity is *for*

This is the most important difference, and Ovadia's own footnote [1] pinpoints
it. In Vale, linearity carries **only the obligation** ("must destroy"). Memory
safety — no use-after-free on the freely-aliasable references — comes from a
**separate, runtime mechanism: generational references** (a generation check on
each dereference). Linearity and memory safety are *decoupled*.

We *fuse* them. The trick is the L3 split (`docs/02`, `docs/07`):

| | Vale | λ-Tally |
|---|---|---|
| "don't forget to clean up" (no leak) | linear, no implicit drop | same: strict linearity |
| use-after-free safety | generational references (**runtime** gen-check per deref) | the linear `Perm`, separated from the aliasable `Addr` (**static**) |
| can you alias a linear value? | yes — aliases are gen-refs, safe via the runtime check | the *address* yes; the *permission* no |
| a bare alias can be dereferenced… | …with a runtime generation check | …only by presenting the linear `Perm` (you cannot fabricate one) |
| cost | a cheap runtime check on deref | **zero runtime**; cost is author-side proof obligations |

The axis in one line: **Vale buys generality with a cheap runtime check; we buy
zero overhead with compile-time proofs.** Both are more expressive than Rust
(whose `Drop` is affine and single-arg); they sit at different points on the
cost/generality curve.

## The striking convergence: `LinearKey` is our cursor, and footnote [29] is our mechanism

His use #8, the `LinearHashMap`:

    insert(k, v)        -> LinearKey      -- a linear token
    get(&LinearKey)     -> &V             -- NON-optional: the key proves liveness
    remove(LinearKey)   -> V              -- consumes the token

is our **cursor**, with one difference:

| | Vale `LinearKey` | our `Cursor` |
|---|---|---|
| multiplicity | linear (one; must be consumed/moved) | copyable (`ω`) + an **erased** membership proof |
| "still exists" proof | the linear key itself | the ghost proof against the region |
| dangling caught by | you can't hold a stale key (it's linear) | removal invalidates the proof → stale cursor fails to type-check |

And then his footnotes [28]/[29] describe **our exact mechanism**:

> "the existence of a linear reference as proof that the pointee still exists …
> can be used to **elide generation checks** … 'linear reference counting' …
> spooky knowledge at a distance."

That is precisely what our region + permission does: use a linear capability to
*statically* prove liveness and **elide the runtime gen-check entirely**. He
flags this in a footnote as nebulous future work; `poc/tally_dll.py` plus the
kernel are a concrete (if scoped) instance of it. We are standing where his
footnote points.

Calibration, both ways: Vale is a real, shipping, general-purpose language with
generational references as an always-available fallback when the static proof is
infeasible. We have a proved kernel and a closed-program POC, with no runtime
fallback. Same destination, very different maturity.

His use #7 (`OwningIndex`, preventing leaks in graph structures) is our exact
situation — except we *avoid the detour he describes*. He notes Rust pushes you
to "collectionize" (`HashMap<u64, Node>` + ID references), which turns
use-after-free into lookup-after-remove *and* leaks orphaned nodes. We keep
**real intrusive pointers** (`next`/`prev` as `Addr`) and the linear
permission/region gives both no-leak and no-UAF — no ID-map indirection.

## Features worth stealing

1. **Whitelisted destroyers via module privacy.** Vale's safety hinge is that
   `destruct` only works in the defining module, so users cannot bypass the safe
   API. We have `free`/`drop_empty` as the only consumers but have not *enforced*
   "only the library may consume the raw `Perm`". That privacy boundary is what
   makes the user/author (Layer-1/Layer-2) split sound; we should formalize it.
2. **An `onException`/panic story.** His sharpest engineering point: linear types
   fight stack unwinding — a panic destroys everything via a zero-arg destructor,
   but linear values have none. His fix is a separate `onException` consumer per
   linear type. Our calculus has no exceptions, so we have dodged this; a real
   surface language needs an answer (abort-only, or per-type `onException`),
   designed in rather than retrofitted.

## Where the QTT framing already answers his open questions

- **"Is the *type* linear, or is our *usage* of it linear?"** (his open design
  question, fifth bullet of "Can other languages add Higher RAII"). QTT answers
  this by construction: the multiplicity lives on the **usage** (the context
  annotation), so the *same* type can be used at `0`, `1`, or `ω` depending on
  context. We *also* get inherently-linear types when a value contains a `Perm`.
  So we have both, cleanly separated, rather than one global choice.
- **"Generics assume droppability"** (his reason Rust can't retrofit, citing
  Beingessner). This does not bite us: QTT bakes multiplicities into generics
  from day one, so a container over a linear `T` simply cannot drop it — `pop`
  must *return* the element. We are better-positioned than a retrofit precisely
  because we did not start from an affine ecosystem.
- **Reconciling linearity with aliasing** (his third "add it to other languages"
  step: "designate one reference as the linear owning reference"). That
  designated owning reference is exactly our `Perm`; the aliases are our `Addr`s.

## Net

- Strong independent validation: Vale's Higher RAII = our strict linearity, and
  our checked no-leak theorem is the formal version of his liveness pitch.
- One clear divergence (linearity for *obligation only* + runtime gen-refs, vs.
  linearity for obligation *and* static memory safety at zero cost).
- One striking convergence: his `LinearKey`/"elide generation checks" footnote is
  our cursor/region mechanism — we built the thing his footnote gestures at.
- Two concrete features to adopt: enforced whitelisted destroyers (module
  privacy on `Perm` consumption) and a panic/`onException` discipline.

See also: `docs/02-memory-views.md` (the L3 split), `docs/07-implementation-guide.md`
§6 (the soundness obligation), `agda/CombinedSound.agda` (the no-leak proof),
`poc/tally_dll.py` (the cursor/region in running code).
