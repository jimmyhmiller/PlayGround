# Design

An ahead-of-time compiler toolkit built on a sea-of-nodes IR, aimed at dynamic languages and
at languages that layer types over a dynamic base. Written in [Coil](https://github.com/jimmyhmiller/PlayGround/tree/master/coil).

The performance target is "as good as V8, without a JIT". The decisions that make that
plausible are recorded in [DECISIONS.md](DECISIONS.md); this document is the architecture.

---

## 1. Shape of the pipeline

```
  source (dyn, later TypeScript)
      |
      v
  [Parse]        s-expressions -> AST -> sea-of-nodes IR, with peepholes running during build
      |
      v
  [Opto]         pessimistic peepholes to fixpoint, then OPTIMISTIC INTERPROCEDURAL dataflow
      |          (this is the type inference engine; it also builds the call graph)
      v
  [Specialise]   clone monomorphic versions per observed argument shape; insert guards;
      |          keep the generic version as the fallback
      v
  [Verify]       graph well-formedness + GC invariants R1/R2 + lattice sanity
      |
      v
  [LoopTree]     loop nesting, break never-exit loops, place loop-back safepoints
      |
      v
  [Select]       ideal nodes -> machine nodes (arm64 first)
      |
      v
  [GCM]          global code motion: build the CFG, place each node in its best block
      |
      v
  [LocalSched]   list-schedule within each block
      |
      v
  [RegAlloc]     graph-colouring allocation with live-range splitting
      |
      v
  [Encode]       instruction encoding, cold-block outlining, STACK MAP emission
      |
      v
  [Export]       Mach-O object (ELF later), linked with cc
```

Phases are re-runnable and each has a dump. `Verify` is not a single phase in practice: it is
a function called after every phase when assertions are on.

The pipeline order is Simple's, and deliberately so: instruction selection *before* global
code motion is what lets the scheduler see real machine costs. The phases that do not exist
in Simple at all are `Specialise`, `Verify` and stack-map emission, plus safepoint placement
inside `LoopTree`.

---

## 2. The IR core

### Nodes

A node is `(op, type, inputs, outputs)`. Inputs are use-def edges, ordered and possibly null,
with input 0 conventionally the controlling `CFG` node. Outputs are the def-use direction,
unordered, maintained automatically so the graph is walkable both ways.

Nodes are addressed by dense `NodeId` integers into an arena. See
[D5](DECISIONS.md#d5-written-in-coil-node-and-type-handles-are-integer-ids).

Three behaviours define an op, and they are the whole optimiser:

- **`compute()`**: the node's type, from its inputs' types. Must be **monotone**: given
  inputs that only fall in the lattice, the output only falls. This is what makes the
  optimistic dataflow pass terminate and be correct.
- **`idealize()`**: a local graph rewrite returning a better node, or nothing. Algebraic
  identities, control-flow collapse, load-after-store forwarding.
- **`hash()`/`eq()`**: structural identity, for global value numbering.

`peephole()` = `compute()`, then replace with a constant if the type is a constant, then
`idealize()`, then GVN. Running it during parsing means the graph is never much larger than
it needs to be.

### The worklist

Iterative peepholes run to a fixpoint on a worklist. A node goes back on the worklist when
any input changes, and additionally when a node it *depends on* changes, where "depends on" is
recorded explicitly by any peephole that inspected a non-input node. Missing dependency
tracking is the classic source of a stale graph that is wrong only under some visit orders,
so the worklist is seeded with a **deterministic pseudo-random order** in tests: the same
program must optimise identically regardless of the seed, and we test several seeds.

The dependency is recorded on the node whose TYPE was inspected, which for `phi-compute` is the
region's control INPUT and not the region: a region already reachable does not change type when a
second path goes live, so a dep recorded on it is never flushed. That one lost `.in(i)` relative to
Simple's `PhiNode.compute` published a phi type that excluded a value the program produces.

### Proven, versus the optimistic answer so far

An analysis may act on a high type; an irreversible rewrite may not. The distinction the IR needs is
sharper than high versus low, and getting it wrong produced seven separate miscompiles:

- **ANY is the absence of information.** It is what `n-new` seeds a type to and what a missing input
  reads as. Every OTHER high type is a CLAIM someone computed: XCtrl says a control edge is dead,
  `~dyn` and `int=~[5..8]` say no value fits. `ty-unanalysed?` is the test; `ty-high?` cannot make
  that distinction and must not be used for it.
- **XCtrl is the HIGH element of the control axis**, so `meet(~ctrl, ctrl)` is `ctrl` and "exactly
  `~ctrl`" is not proof of anything on its own. It is proof only once nothing its type depends on can
  still move.
- **A type is PROVEN when its whole input cone is at a fixpoint**, which is what `n-ty-proven?`
  answers, and every irreversible rewrite asks it. It has to be transitive because `phi-compute`
  skips region paths whose control is still high (which is load-bearing for loops: a loop phi must
  report its entry value while the back edge is unanalysed, or phi, exit test and back edge all wait
  on each other at ANY). A Phi's type is therefore provisional while its own type and its inputs'
  types are all low, which no local check can see. And "not ANY" is not enough either: a stale LOW
  type is exactly as provisional, so the test is the fixpoint condition itself.
- **A refused rewrite is DEFERRED, not declined**, so `iterate!` sweeps: drain the worklist, and if
  that drain changed anything, push every live node and drain again. Nothing else re-queues a node
  whose blocker's type never changes again.

`compute` correspondingly never manufactures a claim from an absence: `region-compute` and
`if-compute` report ANY while an input they would draw a conclusion from is unanalysed.

### Op definitions in one place

Each op needs a label, an arity contract, `compute`, `idealize`, hash/eq, a printer, and later
a register mask and an encoder. Hand-writing seven parallel `case` arms per op is how the
tables drift apart. Once the required set of per-op behaviours has stabilised (roughly after
the first dozen ops), a `defnode` metaprogram generates the enum, the tables and the dispatch
skeletons from a single declaration. Until then they are hand-written, because generating from
a shape we have not yet validated is worse than a little duplication.

---

## 3. The type lattice

This is where the design departs from Simple most, and it is the piece most worth getting
right first: `compute()` for every op is a function into this lattice, so an error here is an
error everywhere.

### The construction, and why it is uniform

Types form a symmetric bounded lattice with `meet` (greatest lower bound) and an involutive
`dual` (`~`). `join(a,b) = ~(~a meet ~b)`. `a isa b` iff `meet(a,b) == b`. Types are
**interned**, so structural equality is pointer/id equality and cyclic types do not require
exponential comparisons.

**Orientation, because it reads backwards.** `ANY` is the top and means "nothing is known
yet, no value is possible"; `ALL` is the bottom and means "any value at all". Analysis starts
at `ANY` and *falls* toward `ALL` as it learns. So a more specific type is **higher**, and
`a isa b` means "a is at least as specific as b". That makes `x isa ALL` true for every `x`,
and `ANY isa x` true for every `x`, which is the opposite of the reflex most people bring
from subtyping. It is Simple's convention, it is what makes the optimistic pass a monotone
fall, and it is pinned by an explicit test so it cannot quietly flip.

Every axis of every type is stored so that:

- **`meet` is the widening or union operation on the stored representation**, and
- **`dual` is the complementing operation on the stored representation**,

with "high" (optimistic, above the centreline) detected by the stored form being *inverted*
or *empty*. Simple establishes this twice, and both are reused verbatim:

- integer ranges store `(min, max)`; `meet` is `(min(min), max(max))`, `dual` swaps them, and
  `min > max` means high. An inverted range under a widening meet computes exactly the
  semantic intersection of the high elements.
- function-pointer sets store a `fidx` bitset; `meet` is bitwise OR, `dual` is bitwise NOT,
  and the empty set is the top of that axis.

Adopting one rule for every axis is what keeps the lattice laws provable rather than hoped
for.

### Termination

Bitset axes are finite and converge on their own. Interval axes do not: a loop induction
variable can widen forever. Simple's answer is a small **widening counter** carried in the
type; after a few widenings on the same axis the type falls straight to that axis' bottom.
We keep it, and the lattice test suite includes "every ascending chain terminates" as a
property.

### The value axis: what a dynamic value is

The bottom of the value sublattice is `Dyn`, the union of every value kind. It sits strictly
above `BOTTOM`, which remains reserved for "unrelated types were mixed" and for the non-value
types (control, memory, RPC, tuples).

A value type is a **kind bitset** plus per-kind refinements:

```
  kinds : bitset of { undefined, null, bool, int, flt, str, sym, bigint, obj, fun }
  int   : (lo, hi, widen)          when int   is present
  flt   : (lo, hi) or a constant   when flt   is present
  obj   : a set of shape ids       when obj   is present
  fun   : a fidx bitset            when fun   is present
```

`meet` unions the kinds and meets each refinement; `dual` complements the kinds and duals each
refinement. Unions are therefore first class, which matters far more than it sounds: the
difference between `number` and `number | undefined` is the difference between a load and a
load plus a branch, and TypeScript code is full of the latter.

`int` is a separate kind from `flt` even though JavaScript has one number type. That split is
the single highest-value refinement in the lattice: it is what turns `a + b` into an integer
add. `number` is simply `int | flt`.

### Where TypeScript types enter

An annotation becomes a `Cast` at the boundary where the value enters annotated code. The
optimistic interprocedural pass then computes what actually flows there. If the computed type
`isa` the annotation, `Cast.idealize` folds the cast away and the annotation cost nothing. If
not, the cast stays as a guard, and the code behind it is entered conditionally. This is
[D1](DECISIONS.md#d1-closed-world-verified-types-speculative-monomorphisation-with-generic-backups)
made mechanical: annotations are discharged, never assumed.

### Shapes

An object's layout is a **shape** (a hidden class): an ordered list of `(name, type, offset)`
fields plus a shape id. Shape ids are lattice elements in their own right, so "this is an
object with shape #7" is a type, and a field load through a known shape is a fixed-offset
load. Shape *sets* let a polymorphic site stay precise (two shapes, two inline paths) instead
of collapsing to a dictionary lookup. Shape transitions (adding a field) are edges in a shape
graph built at compile time from the whole program.

---

## 4. Memory, effects, and objects

Memory is in SSA form, split into **alias classes**. Each field of each shape gets an alias
number; a `Store` produces a new memory value for exactly its alias, and a `Load` consumes
the memory of exactly its alias. `MemMerge` and memory `Phi`s join them at control flow
merges.

The payoff is that load-after-store forwarding, store elimination and code motion all fall
out of ordinary dataflow on the memory edges, with no separate alias analysis pass. The cost
is that allocation must name every alias it touches: a `New` node takes the memory of every
alias it initialises and produces a projection per alias, which is what makes a freshly
allocated object's fields provably not aliased with anything else.

---

## 5. The GC contract

Per [D2](DECISIONS.md#d2-gc-abstract-ir-nodes-collector-policy-chosen-at-lowering) the IR is
built for a moving collector even before one exists.

### Reference vs raw

`ref` (managed, may move) and `ptr` (raw, never moves) are distinct types. Only `ref` values
are traced and relocated. Only `ref` values may be safepoint operands.

### Safepoints

A `Safepoint` node takes control, memory, and **every live reference**, and produces a
relocated projection for each of those references. Uses dominated by the safepoint read the
projections. Safepoints are placed at:

- every `New` (allocation can trigger a collection),
- every `Call` (the callee may allocate; a call is a safepoint at its return point),
- every loop back edge (so a counted loop with no allocation is still interruptible).

Because live refs are real inputs, they are naturally kept alive through code motion and get
real locations from the register allocator, which is precisely what the stack map needs. And
because uses read the projections, a collector that moved the object is correct by
construction.

### Barriers

`Barrier(mem, obj, field, val)` is abstract. Lowering expands it to nothing, to card marking,
or to a snapshot-at-the-beginning sequence, according to the selected collector. Keeping it
abstract in the IR means the optimiser can eliminate redundant barriers (same object, same
block, no intervening safepoint) without knowing which collector will be used.

### Stack maps

At `Encode`, each safepoint emits a stack map: for every live location, whether it holds a
raw reference, a NaN-boxed word that may contain a reference, or a scalar. Boxed words need
the distinction because of
[D3](DECISIONS.md#d3-nan-boxing-with-boxunbox-as-real-ir-nodes).

### The verifier

R1 and R2 are checked mechanically:

- **R1**: no value of raw-pointer type derived from a `ref` is live across a safepoint.
- **R2**: no use of a reference is dominated by a safepoint that took that reference as an
  operand, unless it reads the projection.

The verifier runs after every phase in tests. This is what "accounting for the GC from day
one" means operationally: not a plan to add GC support later, but an invariant that fails the
build the moment a pass breaks it.

---

## 6. Specialisation

After the optimistic interprocedural pass there is a call graph and, for each function, the
set of argument types that actually reach it.

For each `(callee, argument type tuple)` worth specialising, clone the callee with the
arguments pinned to that tuple and re-run peepholes; the clone monomorphises. The call site
becomes a `TypeTest` chain choosing between clones, falling through to the generic version.
Guard-and-fallback is plain control flow
([D4](DECISIONS.md#d4-guards-are-control-flow-not-a-node-kind)), so inlining, code motion and
register allocation need no changes.

Which specialisations exist is a cost model over call-site counts, loop depth and clone size,
with an optional offline profile as an input. Nothing about correctness depends on the model
being good; a bad model makes slow code, not wrong code.

---

## 7. Backend

Target arm64 (Apple silicon) first, since that is the development machine.

`Select` rewrites ideal nodes into machine nodes. A machine node declares an input register
mask per input, an output mask, a kill mask, whether it is two-address, whether it commutes,
whether it is cheaper to rematerialise than to spill, and how to encode itself. That interface
(Simple's `MachNode`) is the entire porting surface for a new CPU, and keeping it narrow is
what will make an x86-64 or RISC-V port a contained job rather than a rewrite.

Register allocation is graph colouring: build live ranges, build the interference graph,
coalesce copies, colour, and on failure split live ranges and repeat. Cold blocks from
guard fallbacks get high spill preference, which is the main way profile information reaches
the machine code.

**What we do not write from scratch.** Coil's self-hosted compiler already contains, in Coil:
`a64.coil` (an AArch64 instruction encoder with label and fixup handling), `macho.coil` (a
Mach-O object writer with relocations and symbol tables), and `dwarf.coil` (line tables and
subprogram info). Those solve the mechanical half of a backend and are adapted rather than
rewritten. What we write is the interesting half: selection, scheduling, allocation, stack
maps.

---

## 8. Development tooling

Tooling is a day-one deliverable, not a follow-up, because a sea-of-nodes graph is
unreadable without it.

- **Textual IR, printed and parsed**, round-trip tested as an identity
  ([D7](DECISIONS.md#d7-textual-ir-is-a-first-class-round-trippable-format)). Hand-written
  graphs are the reduced test cases for optimiser bugs.
- **`--dump-after=PHASE`** for every phase, in textual IR.
- **Graphviz output**, with control edges, data edges and memory edges visually distinct.
- **A verifier** runnable after any phase, checking graph well-formedness, type monotonicity,
  and the GC invariants.
- **Deterministic ids and stable printing**, so a golden test diff means something.
- **Per-phase timing and node counts**, printed on request, so a compile-time regression is
  visible rather than discovered later.
- **Lattice property tests**: commutativity, associativity and idempotence of `meet`,
  involutivity of `dual`, `join`/`meet` duality, reflexivity and transitivity of `isa`,
  monotonicity of every `compute`, and termination of ascending chains. These run over
  exhaustively generated small types plus randomly generated large ones.

**The graph text's identity, stated exactly**, because the claim above is the one a reader will
lean on. Printing a graph and parsing it back is an identity **up to a dense renumbering of the
live nodes**: `n<k>` in a line is a print INDEX, the node's position in the listing, not its arena
id, because the arena ids of dead nodes cannot be recreated. Everything else is exact, and every
field is mandatory: a Const's aux type equals its computed type and is printed anyway, a Region's
unused slot 0 is always `_` and is printed anyway. An optional field would give two texts for one
graph, which is the same defect the type format refuses with `TERR-DUP-KIND`, and the one wart the
type format still has (`int=[min..max]` versus `w0`) is what tolerating a second spelling costs.
Types are restored from the text rather than recomputed, so a text that lies about a type is a
verifier violation and not a silently corrected one. What the format does not carry is `outs`
order, the GVN table, `deps` and `hash`, all of which a parse rebuilds or does not need; a pin
outside the two roots is refused rather than dropped, since it means a construction window is
still open. A failed parse leaves the graph in an **unspecified partial state** that the caller
must not use — the strict entry point reports the failing check by name and stops, precisely so
that a partial graph is never handed on.

Later, a live graph viewer. Simple drives one over a websocket; the jim editor's widget bus
is the natural host here.

---

## 9. Deviations from Simple, summarised

| Area | Simple | Here |
|---|---|---|
| Source language | statically typed, safe | dynamic, with optional types layered on |
| Value lattice | int / float / pointer / struct | kind bitset with per-kind refinements; `Dyn` is a real element; unions first class |
| Numbers | separate int and float types | `int` and `flt` kinds unified as `number`, split back apart by inference |
| Objects | declared structs, fixed layout | shapes (hidden classes) with a compile-time transition graph, and shape sets |
| Type annotations | checked by the front end | discharged by whole-program inference, guarded where not provable |
| Polymorphism | monomorphic by construction | speculative specialisation with a retained generic version |
| GC | `calloc`, never freed | safepoints with relocating projections, abstract barriers, stack maps, verified invariants |
| Allocation | `New` per struct | `New` is a safepoint; bump allocation with a slow-path call after lowering |
| Exceptions | none | planned: calls get exception edges, landing pads are ordinary CFG (see ROADMAP) |
| Textual IR | printer only | printer and parser, round-trip tested |
| Backend | x86-64, arm64, RISC-V, ELF | arm64 and Mach-O first, reusing Coil's encoder and object writer |

The two things Simple has that we take almost unchanged are the ones that took it 24 chapters
to get right: the peephole-plus-worklist engine, and the optimistic interprocedural dataflow
pass in `Opto` (which already carries an explicit whole-world assumption, builds its own call
graph as function-pointer types stabilise, and is therefore the closed-world inference engine
this project needs).
