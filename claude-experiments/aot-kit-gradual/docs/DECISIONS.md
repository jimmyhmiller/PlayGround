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

---

# Laws learned by breaking them

D1 to D7 were decided up front. Everything below was learned the hard way, each after a bug that a
fully green test suite could not see. They are laws in the same sense: code that contradicts one is
a bug in the code.

If you are new here, **D8 and D9 are the two that will cost you a day.**

---

## D8. A rewrite may only act on a PROVEN type, and proven is narrower than it sounds

**Decision.** `compute` is an optimistic analysis and may act on any type it currently holds. Every
*irreversible* rewrite must first ask `n-ty-proven?`, which is **transitive** over the node's whole
input cone and is a **fixpoint** test, not a "not ANY" test.

**The insight that took three attempts to state.** The first version was "an analysis may act on a
provisional type; a transformation may not", which is right but not actionable. The second was
"require `~ctrl` exactly, never merely high", which is **wrong**, and it was written into a review
checklist as if settled. The correct statement:

> **`ANY` is the ABSENCE of information. Every other high type is a CLAIM someone computed.**
> `~ctrl` is just the high element of the control axis, so satisfying "exactly `~ctrl`" proves
> nothing on its own.

`region-compute` seeded its meet with `~ctrl`, so a merge nobody had analysed reported *provably
unreachable* out of nothing, and six rewrites believed it.

**Why transitive.** `phi-compute` deliberately skips high paths, which is load-bearing for loops, and
that hides provisionality below any local check. A node can hold a perfectly reasonable-looking type
computed from an input that is still ANY.

**Why a fixpoint and not just "not ANY".** A stale LOW type is exactly as provisional. With only the
ANY test, a flat diamond still compiled to `return 1` on 9 of 40 worklist seeds, through
`n-int-con?`, which asks "is this operand the literal 0" and believed a phi that momentarily
reported `int=0`.

**Consequence.** A proof-gated rewrite is *deferred*, not declined, so `iterate!` sweeps: nothing
else would re-queue it.

**What it was costing, all on a green gate.** A merge whose input is a merge compiled to one wrong
constant on 55 of 200 seeds. Two merges feeding a merge deleted the entire program on 31 of 200,
with `Stop` left with no inputs, `g-verify` clean, and the printer omitting dead nodes so the diagram
merely looked smaller. `cast-idealize` discharged a `Cast` on an unanalysed input, and since
`ty-isa ANY t` is true for every target, every `Cast` on a raw graph deleted itself: that is D4's
only guard mechanism erasing itself. Four `compute`s were made to RISE by a `ty-high?` that answered
a per-axis question with a single-axis test, which aborted the compiler on a 14-node program.

---

## D9. Construction has contracts, and they are checked where they can be

**Decision.** Building a graph is not free-form. Three windows must be respected, and the code
hard-errors on the parts of them that are cheap to detect.

**The in-progress window for merges.** A merge with a null final input reports CONTROL, and a `Phi`
on it reports its DECLARED type. A loop body must be built *and peepholed* inside that window, and
the phis closed before the control back edge. `n-set-def!` panics by name if control closes first.

Get it wrong and the loop is **deleted with no error reported**: the phi momentarily reads `int=0`
because its back-edge value has no type yet, `i + 1` folds to the literal `1`, the phi loses its
only other user, and the whole loop goes. Every step is locally justified.

**The multi window for `If`.** An `If` is in progress until *all* of its projections exist. Peephole
them one at a time and the first folds to `~ctrl`, the `If` loses its last use, the kill cascade
takes it, and the sibling then rewrites against a corpse whose inputs were cleared but whose type
was not: a stale buffer word. Flat, that word happened to be pinned `Start` and the answer was
accidentally right. One level of nesting down it was a dead node. Use `n-if-arms!`, which opens the
window with `n-multi-open!` and closes it with `n-multi-close!`; `n-in`/`n-out` are bounds-checked so
an out-of-range read cannot look plausible again.

**Region and Phi arity are ONE invariant.** Dropping a region path drops the matching value from
every phi on that region, in the same operation. Diverging for even one peephole gives a phi reading
the wrong arm, which is a miscompile that typechecks cleanly.

**Still open, and it will bite while wiring memory edges.** A dead node can be wired as an input and
nothing says so at the wiring site. See the note in [ROADMAP.md](ROADMAP.md#the-next-concrete-piece-of-work).

---

## D10. A shape allocates a field's alias at the edge that introduces it

**Decision.** Shapes form a transition tree rooted at the field-less shape. `shape-transition` is
memoised on `(shape, name)`, so the hidden-class tree a whole-program build produces is a function of
the program and not of visit order. **The alias class and byte offset for a field are allocated at
the transition that introduces that field, and inherited unchanged by every descendant.**

**Why that one sentence is what makes memory SSA work.** A store through `{x}` and a load through
`{x, y}` must name the same alias class and the same offset, because `{x, y}` was reached *from*
`{x}` and a transition never moves an existing field. Allocate per-shape-per-field instead and those
two touch different words, so load-after-store forwarding silently stops firing: no error, just
worse code, which is the hardest kind of bug to notice.

**Consequence, and it is the point rather than a wart.** `{x, y}` and `{y, x}` are genuinely
different layouts, because the edge that introduced `x` differs. That is the entire content of
"hidden class", and it is what `tests/shape-test.coil` uses as its witness.

---

## D11. A tool is only a tool if it can FAIL

**Decision.** Every checker reports a **named code**, and every identity or coverage claim has a
**counted floor** under it saying how much it actually compared.

**Why.** "The verifier returned non-zero" is satisfied by a verifier that always fails. "Corrupting
the phi/region lockstep reports `VERR-PHI-ARITY`" is a claim about a specific check noticing. The
same applies to a round trip: `print` then `parse` is the identity would pass for a printer that
emitted only op names, so the gate records how many node lines and characters it compared.

**How claims get validated.** Revert the fix and confirm the gate goes red; where a guard cannot be
made to fail, measure it instead. `ty-injective?` returning false is unreachable by construction, so
what is recorded is that always-false takes 9 of 12 tests red while always-true keeps all 12 green.

**Corollary, learned separately.** A fixture can be correct **by accident**, and its construction
order is part of it. The flat `if (0)` passed while the same construct one level down miscompiled,
and the first raw diamond reproduced nothing on 200 seeds until it was built the way the witness was
built. So a gate states which shape and which order it needs.

---

## D12. The oracle has to RUN the program

**Decision.** From M3 on, every optimisation is gated on the observable result being identical
before and after the pass, over the whole corpus, with arguments bound to values **inside** their
declared types.

**Why.** Golden strings and `g-verify` both stayed green through a merge deleted from a diamond, a
`Return` dropped from `Stop`, and a discharged type check. A graph missing an arm is still a
structurally valid graph. Only running it noticed.

**Why inside the declared types.** Feeding an argument something its declaration forbids does not
test the optimisation, it tests what happens when you lie to the compiler. Bindings are keyed by
type rather than node id, because optimisation deletes arguments and shifts every later id.

**Corollary.** The interpreter refuses to guess. Integer overflow reports `EV-OVERFLOW` rather than
wrapping, because JavaScript promotes to a double and doing that properly needs M5's value domain; a
wrapped answer would make the oracle quietly disagree with the language it exists to define. A
failing `Cast` is a **compiler** bug and has its own status.

---

## D13. A memory access reads a REFERENCE, and the two checks over it live in different places

**Decision.** A `Load` or a `Store`'s pointer operand must be **an object and nothing else**
(`ty-only-obj?`), and the verifier reports `VERR-PTR-SLOT` when it is not. It **abstains** while the
pointer's type is high, which is the same optimistic case `load-compute` special-cases. Which
**shape** the pointer has is a different question and is deliberately not asked there: it is the
interpreter's `EV-SHAPE` and, from M9, a guard.

**Why "and nothing else" rather than "could be an object".** `arith-compute` types `o + 8` as plain
`dyn`, and `dyn` includes `VK-OBJ`, so the weaker form accepts exactly the address arithmetic
[R1](#d2-the-ir-is-gc-abstract-safepoints-are-explicit) exists to keep out of the graph. Before this
check existed, `Load(mem, Const int=8)` and `Load(mem, Add(o, 8))` both built, verified clean,
round-tripped and ran. D2 claims both GC rules are machine-checked by the verifier on every phase;
that claim was simply not true for R1, and it held only because nothing in the IR was yet NAMED an
address. Nothing would have gone red the day M8's safepoints or any address-computing lowering
arrived.

**What M5 owes.** When `dyn` grows objects, a value may honestly be `dyn` at a memory access before
anything has refined it. This rule then requires a **`Cast` to `obj` first**, which is the same
shape D4 already gives every other narrowing: the guard is ordinary control flow and the `Cast` is
its value side. The alternative, accepting `dyn`, weakens the rule to nothing.

**The two checks that are NOT here, and why.** "The memory state carries the class this node names"
and "the pointer's shape carries this word" are both statically checkable, and both are left to the
interpreter (`EV-MEM`, `EV-SHAPE`) rather than made verifier rules. An alias bitset and a shape set
are both **unions over paths**, so a state that statically carries a class need not carry it on the
path taken, and a static rule strong enough to be worth having would make `EV-SHAPE` unreachable
from any verifier-clean graph. `tests/mem-test.coil`'s
`running_it_catches_the_memory_contracts_the_verifier_cannot` is the argument for that split. What
the compiler owes instead is that no **rewrite** launders such an access into a plausible value,
which is what `load-compute`'s two refusals and `load-idealize`'s `access-refused?` are for.

---

## D14. An object's identity across two runs is its reachable heap, not its allocation index

**Decision.** `rt-eq?` compares an object by its index in the run's allocation stream, and that stays
the implementation of `===` **within one run**. The **differential oracle** uses `ev-render-outcome`
instead: the status, the result, and the heap reachable from it, with objects numbered by
**discovery order** in a deterministic walk and shapes named by their **field names**.

**Why.** A table index is a run artefact. It fails in both directions, and both were measured.
Removing an allocation nothing reads shifts every later index, so the oracle reports
`DIFFERENTIAL FAILURE` on a transformation that changed nothing observable, and that transformation
is exactly 4c-3's dead-allocation removal. In the other direction, two objects of different shapes
that land at the same index compare EQUAL, so the oracle could not see a rewrite that returned the
wrong object. A shape ID is a build artefact for the same reason an object index is: `{x}` and `{y}`
are both shape 1 in their own runs.

**Why discovery order.** It makes sharing and aliasing observable (`{a: o, b: o}` renders differently
from `{a: o, b: o2}`) while an allocation nothing can reach is invisible. That is isomorphism of the
two reachable heaps, which is identity modulo renaming, which is what "the same outcome" means.

**Why rendered to text.** The comparison has to outlive `ev-reset!`: the second build cannot run
until the first run's heap is cleared, and carrying a heap across runs is the one thing that must
not happen.

**Corollary.** This is also what makes a **memory phi's arm order** observable at all. Reading the
returned object's field out of the state the `MemMerge` produced is what turns LAW 5's failure mode
into a red gate; before it, swapping `fx-shape-poly`'s class-x memory phi's arms left the whole gate
green.
