# Decisions

Load-bearing choices, with the reasoning that produced them. This file is law: code that
contradicts a decision here is a bug in the code or an amendment to this file, never a
silent divergence. Amendments append; they do not rewrite history.

---

## D1. Closed world, verified types, speculative monomorphisation with generic backups

**Decision.** The compiler assumes a **closed world**: it sees every function that can be
called. On that basis:

1. TypeScript annotations are used, but they are **proven, not trusted**. Closed world means
   whole-program inference can compute what actually flows into each parameter, so an
   annotation is a *contract we discharge*. Where inference proves the annotation, the check
   folds to nothing. Where it cannot, a runtime guard stays in the code.
2. For untyped code, types are **inferred** by interprocedural optimistic dataflow over the
   whole program.
3. Where a value stays polymorphic, we emit **speculative monomorphic specialisations**
   ahead of time, each entered through a guard, with a **generic version retained** as the
   fallback target.

**Why not blind trust.** TypeScript's type system is deliberately unsound (bivariant
parameters, `any`, unchecked casts, mutable covariant arrays). Trusting an annotation and
compiling unguarded code from it produces memory-unsafe output on legal TypeScript. Closed
world is what upgrades an annotation from a hint to a provable fact, so this decision is
about *earning* the right to compile the annotation away.

**Consequence.** There is no deoptimisation machinery. A failed guard is ordinary control
flow into the generic version, which is compiled into the same binary. No OSR, no frame
rewriting, no deopt metadata. See [D4](#d4-guards-are-control-flow-not-a-node-kind).

**Cost accepted.** `eval`, dynamic `import()` of unknown code, and monkey-patching from
outside the compiled unit are not supported under the closed-world assumption. A future
open-world mode would keep the generic versions and drop the guards' fold-away, at a
significant performance cost.

---

## D2. GC: abstract IR nodes, collector policy chosen at lowering

**Decision.** The IR carries `Safepoint` and `Barrier` nodes with abstract semantics, and a
type-level distinction between **managed references** (may move) and **raw pointers** (never
move). A concrete collector is selected during lowering.

**The invariant that forces day-one design.** A moving collector is only implementable if the
IR was built for it from the start, so we adopt its two rules unconditionally:

- **R1. No interior pointer is live across a safepoint.** Enforced by construction: address
  arithmetic on a managed reference is not a first-class IR value. `Load`/`Store` carry
  `(base_ref, offset)` and fold offsets internally, so there is no exposed interior pointer
  to leak across a safepoint.
- **R2. A safepoint redefines every live reference it receives.** Live refs are *inputs* to
  the `Safepoint` node and re-emerge as its *projections*; uses after the safepoint read the
  projection, not the original. This is the relocation dance (LLVM's `gc.statepoint` model).
  It cannot be retrofitted later without touching every pass, which is exactly why it is
  here on day one.

Both rules are machine-checked by the verifier on every phase in tests, not left to
discipline. See [DESIGN.md](DESIGN.md#5-the-gc-contract).

**Cost accepted.** Explicit relocation projections make graphs larger and every pass has to
respect R2. Non-moving collectors pay for a generality they do not need.

---

## D3. NaN-boxing, with `Box`/`Unbox` as real IR nodes

**Decision.** A dynamic value is one 64-bit word, NaN-boxed: doubles are stored directly,
everything else lives in the NaN payload space. The IR keeps **unboxed** `int`, `f64` and
`ref` as distinct types, and `Box`/`Unbox` are ordinary nodes that the optimiser cancels in
pairs.

**Why.** JavaScript and TypeScript numbers are doubles. Any representation that heap-boxes a
non-small double loses numeric code outright, which is where the V8 comparison is won or
lost.

**Consequence for GC.** The collector must identify references *inside* boxed words. The
stack map therefore records a slot's **kind** (raw ref, boxed word, or scalar), not merely
whether it is live.

---

## D4. Guards are control flow, not a node kind

**Decision.** A type guard is `If(TypeTest(v, T))` followed by a `Cast(T)` on the taken
edge. There is no `Guard` opcode.

**Why.** Every existing control-flow peephole, dominator computation, code motion pass and
the dataflow solver already understand `If`, `Region`, `Phi` and `Cast`. A distinct `Guard`
node would need bespoke handling in each. Simple's own `CastNode` is documented as "upcast
the input to a `t`, used after guard test to lift an input", so this reuses a proven
mechanism, and its `idealize` (fold when the input already `isa` the cast type) is exactly
how a discharged TypeScript annotation disappears.

**Consequence.** Slow paths are ordinary CFG paths carrying a `cold` hint. Cold blocks are
laid out out-of-line at encode time and bias register-allocation spill costs. Nothing else
in the pipeline needs to know they are slow.

---

## D5. Written in Coil; node and type handles are integer ids

**Decision.** The toolkit is written in Coil. Nodes and types are addressed by dense
integer ids into arena tables, not by pointers.

**Why ids.** Global value numbering, worklists, visit bitsets, deterministic printing,
golden tests and on-disk serialisation all want a small dense integer. Ids also keep the
compiler's own memory management trivial (an arena plus a free list) instead of entangling
it with Coil's allocator lifetimes.

**Why a sum type for `Ty` and not a tagged struct.** Forgetting a kind in `meet` is a silent
miscompile, and `defsum` + exhaustive `match` turns that class of mistake into a compile
error. Variable-arity children (tuple members, shape fields) live in a side array addressed
by `(offset, length)`, which keeps the sum itself small and non-recursive.

---

## D6. The first driver is a minimal dynamic core language, in s-expression syntax

**Decision.** The first front end is `dyn`, a small dynamically-typed language with optional
type annotations, written in s-expressions and read with Coil's bundled `sexp` module. A
real TypeScript front end comes later and lowers to the same core IR.

**Why.** The chosen milestone driver is a dynamic language from day one, so that boxing,
guards, shapes and GC pressure are real from the first commit rather than retrofitted. Using
s-expressions means that costs approximately zero parser work, so effort goes into the IR.
Optional annotations exercise the gradual-typing path immediately.

**Consequence.** `dyn` is not the product; it is the test harness for the IR. It must stay
small enough to never become the reason a design changes.

---

## D7. Textual IR is a first-class, round-trippable format

**Decision.** The IR has a textual form that both prints and parses, and round-tripping is a
tested identity.

**Why.** It decouples IR work from front-end work (a graph can be written by hand), it makes
golden tests readable diffs instead of opaque blobs, it gives every phase a `--dump-after`
for free, and it is the only practical way to file a reduced reproduction of an optimiser
bug.
