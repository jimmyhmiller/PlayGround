# Roadmap

Ordered milestones. Each has a **gate**: a check that must be green before the next milestone
starts. A gate is executable, not a judgement call. Nothing is "done" while its gate is red,
and no milestone is skipped ahead of an earlier red gate.

Two rules that apply throughout:

- **No stubs that return a plausible value.** Anything unimplemented raises a hard, named
  error. A placeholder that returns `0` or `-1` turns a missing feature into a miscompile
  hunt.
- **Every optimisation is differentially tested.** From M3 onward there is an IR interpreter,
  and the gate for every later optimisation is that the program's observable result is
  identical before and after the pass, on the whole test corpus. This is the only practical
  way to keep a graph optimiser honest.

---

## Where things are

`tools/gate.sh` is the single source of truth; every number below is what it prints today.

| | Milestone | Status | Gate |
|---|---|---|---|
| **M0** | The type lattice | done | `tests/ty-test.coil` |
| **M1** | Nodes, peepholes, GVN | done | `tests/node-test.coil` |
| **M2** | Control flow | done | `tests/control-test.coil` |
| | Dominators and the loop tree | done | `tests/loop-tree-test.coil` |
| **M3** | Verifier, interpreter, textual IR | done | `verify-`, `eval-`, `text-`, `gtext-test.coil` |
| **M4** | Memory SSA and shapes | **in progress**: shapes, the memory type, all four memory ops and load-after-store forwarding are in and run; store-after-store elimination and dead-store removal are not | `tests/shape-test.coil`, `tests/mem-test.coil` |
| M5 | Dynamic values end to end | not started | |
| M6 | Functions, calls, closures | not started | |
| M7 | Closed-world inference | not started | |
| M8 | The GC contract | not started | |
| M9 | Specialisation and guards | not started | |
| M10 | arm64 backend | not started | |
| M11 | A real collector | not started | |
| M12 | TypeScript front end | not started | |
| M13 | Performance | not started | |

Run `tools/gate.sh` for the live counts. Do not trust a test count written in prose, here or
anywhere else: four of them in this file were once historical rather than current, which is a number
nobody can check.

## If you are picking this up

Start with [../README.md](../README.md), which has the module map and the reading order. The two
things most likely to cost you a day are in [DECISIONS.md](DECISIONS.md): **D8** (a rewrite may only
act on a *proven* type, and what proven means is narrower than it sounds) and **D9** (the
construction contracts for merges and multi-nodes). Both were learned by getting them wrong, and
both fail in ways a green test suite cannot see.

[JOURNAL.md](JOURNAL.md) has the long-form record: why each design choice was made, what each
adversarial review found, and the measurements behind every claim. Read it before changing something
that looks gratuitous.

## The next concrete piece of work

M4 slice 4c-2 and 4c-3: the two rewrites of M4's gate that are still open. 4c-1
(load-after-store forwarding) is done and is `load-idealize` in `src/node.coil`.

- **Store-after-store elimination.** `Store(p, v2, Store(p, v1, m))` drops the inner store. Same
  decision procedure as 4c-1 and the same refusals: same alias class, same pointer NODE, no type
  read anywhere. It is a harder rewrite than the load side for one reason, and it is worth
  knowing before starting: the inner store may have OTHER readers, so dropping it is only legal
  when this store is its only memory consumer.
- **Forwarding past a provably different pointer.** Simple's `Load.idealize` also walks past a
  store to a *provably different* allocation (two distinct `New`s never alias), which is
  structural and safe. It needs a distinct-allocation predicate that does not exist yet. Its
  offset-overlap test reads a TYPE and is exactly the shape D8 forbids acting on provisionally,
  and its push-a-load-up-through-a-Phi rule builds a new Phi on the merge's region, which lands
  in LAW 5's region/phi arity lockstep. Neither is in 4c-1.
- **Dead-allocation removal.** A `New` none of whose projections is read is unobservable. Most of
  this already falls out: killing the last projection drops the New's last use and the cascade
  takes it. What has to be *added* is dropping a store to an object that escapes nowhere.

Each one is irreversible, so each one must ask `n-ty-proven?` before it acts (D8). NOTE WHAT
THAT GUARD IS FOR ON A STRUCTURAL RULE, which 4c-1 had to work out: these rules read no type, so
the guard is not about disbelieving a provisional type. It is about not committing permanently to
a store that the sweep has not finished with. Each one is gated the same way: the differential
interpreter must return the identical result on the whole corpus before and after, and the seven
memory fixtures (19 to 25) are what make "the whole corpus" mean something for memory.

**4c-3 HAS ITS GATE ALREADY, AND IT IS `24-object-returned` AND ITS SCRATCH TWIN.** They are the
same program with and without an allocation nothing reads back, they are the first pair in the
corpus whose RESULT is an object, and they are already through `diff-pair`. That is deliberate:
before D14 the oracle compared objects by their index in the allocation stream, so removing a dead
allocation shifted every later index and the oracle would have reported `DIFFERENTIAL FAILURE` on a
pass that changed nothing. The pass now has a gate that will pass when it is right.

AND EVERY MEMORY FIXTURE THAT FORWARDS MUST PIN ITS MEMORY ACROSS THE FOLD. 4c-1 measured this:
when the forwarded load is the store's last reader, subsuming it runs the kill cascade over the
entire memory chain, and the `Return` built on the next line is wired to a corpse. `n-keep!` on
the memory value across the load's construction is the fix, and the corpus-wide verifier
(`VERR-DEAD-INPUT`) is what catches a missed one, a long way from the cause.

Two things are open from M3 and are worth reading, because the first one will bite around memory
edges:

1. **A dead node can be wired as an input** and nothing says so at the wiring site. `n-add-def!`
   accepts a corpse, and the resulting graph fails `g-verify` with `VERR-DEAD-INPUT` a long way from
   the line that caused it. A named panic was written and reverted because one existing test reuses
   an `Arg` after the peephole under test has killed it; closing this means fixing that test's
   hygiene first.
2. **`g-analyze!` followed by `iterate!`** deleted an entire `If` on 15 of 200 worklist seeds on the
   nested-guard fixture, from a fully analysed graph. It does not reproduce now, which means only
   that no gate covers that phase order. A gate for it is the first thing to write when it is picked
   up.
3. **The two memory access contracts are checked only by the interpreter**, by decision rather than
   by omission: `EV-MEM` (the memory edge carries the class this node names) and `EV-SHAPE` (the
   pointer's shape carries this word) are both statically checkable, and D13 records why they are
   not verifier rules today. What is closed is the LAUNDERING: no rewrite may turn either refusal
   into a plausible value, which is `load-compute`'s two answers and `load-idealize`'s
   `access-refused?`, gated by
   `a_miswired_access_is_not_laundered_by_the_constant_fold` and
   `forwarding_refuses_a_store_the_program_itself_refuses`. What is still open is whether M5 wants
   the static forms as well; that decision needs the `dyn` pointer rule in D13 settled first, and it
   is not free, because a shape rule strong enough to be worth having makes `EV-SHAPE` unreachable
   from a verifier-clean graph and so removes the project's own argument for having an oracle.
4. **An unread store's checks never run at all**, and that is a property of the demand-driven
   interpreter rather than of any rewrite: a `Store` whose memory result nothing reads is never
   evaluated, so its `ev-check-access` never happens, in EVERY build. `forwarding_refuses_a_store_the_program_itself_refuses`
   pins the part that IS a rewrite's fault (a store that was demanded becoming undemanded); the
   residual is a gap in the oracle's coverage of stores and is worth closing when 4c-3 makes stores
   disappear on purpose.

---

## M0. Foundations: the lattice

The type lattice, interned, with `meet`, `dual`, `join`, `isa`, and the constant/high
predicates. Simple types, the value axis (kind bitset with integer and float refinements),
tuples.

**Gate.** Lattice property tests pass: `meet` is commutative, associative and idempotent;
`dual` is involutive and order-reversing; `meet` is a lower bound and `join` an upper bound;
`isa` is reflexive and transitive; `x isa ALL` and `ANY isa x` for all `x`; every ascending
chain terminates.

**Status: DONE.** `src/ty.coil`, gated by `tests/ty-test.coil`: 26 tests, laws checked
exhaustively over 32 hand-listed types plus all 32 of their duals (every pair for the binary
laws, every triple for associativity and `isa` transitivity). The sample set was 28 until the M3
review found that no tuple in it had a tuple MEMBER, which is the one shape that made `meet` stop
being a lower bound.

Three things this milestone actually caught, which is the argument for gating on laws rather
than on examples:

- `isa` reads backwards from subtyping intuition (see DESIGN.md). Three laws were written the
  wrong way round and the property tests said so immediately.
- `ty-widen` needed `TY-WIDEN-MAX` steps from the low side but twice that from the dual side,
  because `dual` negates the counter. Termination was still guaranteed, but at the wrong
  bound, and only the "reaches a fixpoint in N steps" property exposed it.
- Two Coil compiler bugs, reported and fixed upstream (`docs/repro/impl-body-scope` and
  `docs/repro/sigbus-emit-ir-singleton-cycle` in the coil repo). One of them had been masking
  eight real type errors in this file.

## M1. Nodes, peepholes, GVN

Node arena, use-def and def-use edge maintenance, `compute`/`idealize`/hash/eq, `peephole`,
the worklist with explicit dependency tracking, global value numbering. Ops: `Start`, `Stop`,
`Return`, `Constant`, integer and float arithmetic, comparisons, `Not`, `Minus`.

**Gate.** Golden tests on printed IR for constant folding and algebraic identities; the same
program optimises to an identical graph under many worklist seeds; no node leaks (every
unreachable node is dead).

**Status: DONE.** `src/node.coil`, gated by `tests/node-test.coil`: 27 tests. Ops through
`Return`, plus an opaque `Arg` node standing in until `Proj` arrives in M2. (Every count in this
file is the suite's size TODAY, not its size when the milestone closed: this one was 16 at M1 and
grew as M3's two reviews added the proof-and-provisionality gates to the same file. A count that
records history rather than the present is a number nobody can check.)

The seed test is doubled up, because the obvious version of it is nearly vacuous. Peepholing
eagerly during construction leaves only 5 nodes on the worklist, so seeds barely explore
anything. So there is a second fixture built with **no** peepholing at all, which pushes
every node and lets `iterate!` do the whole job: 18 items entering, 35 steps, sampled over 24
seeds. That version also pins something M7 will depend on directly, namely that **eager and
deferred peepholing reach the same graph**.

Four real bugs this gate caught, none of which a smaller test would have:

- **`Const` was excluded from value numbering**, so no constant ever deduplicated and
  `(arg+2)` and `(2+arg)` stayed distinct nodes. Deduplicating constants is most of what GVN
  buys.
- **The kill cascade ate the replacement.** `x+0` rewrites to `x`, and killing the `Add`
  released its inputs, which killed the `x` being returned. `return arg+0;` printed
  `return DEAD;`. Fixed with a pin, which is what Simple's `deadCodeElim` is for.
- **Operand canonicalisation ran inside `idealize` and returned the node itself**, so the
  identity rules never saw the new order and `0+x` stayed `0+x`. Normalising *before*
  `idealize` means each identity rule only handles one form.
- **The pin was first implemented as a fake `NO-NODE` out edge** and assumed to stay last,
  which stopped being true the moment `subsume` appended real uses after it. It is a counter
  now, and `outs` no longer contains a value every walker has to special-case.

## M2. Control flow

`If`, `Region`, `Phi`, `Proj`, `CProj`, `Loop`, `Cast`, `XCtrl`. Dead control flow
elimination. JavaScript truthiness deciding reachability.

**Gate.** Golden tests covering Simple chapters 5 through 8 translated to hand-built graphs;
unreachable code provably removed; a `Phi` never survives with a dead input; a loop phi reaches
a fixpoint, identically under every worklist seed.

**Status: DONE.** `tests/control-test.coil`, 19 tests. Reachability is not a separate pass: an
`If`'s type is a tuple of its two control outputs, so an untaken branch is `~ctrl` in one slot
and dead-code elimination falls out of type propagation. `Cast` is in, which is the mechanism
[D4](DECISIONS.md#d4-guards-are-control-flow-not-a-node-kind) rests on, and a guard with an
unboxed fast path plus a generic fallback is now a fixture in the diagram gallery.

**The rule this milestone produced**, which was wrong in three separate places: **an analysis
may act on a provisional type; a transformation may not.** `compute` skips any path whose
control is high, including `ANY`, because an optimistic answer gets revisited. Every
*irreversible* rewrite must instead require `~ctrl`, which is proven. The three sites were
`phi-single-input`, `cproj-idealize` and `stop-idealize`. A fourth, constant folding, needed a
different guard: it now refuses to fold while any input is still unanalysed.

**The in-progress contract**, now documented in `src/node.coil` and partly checked. A loop body
must be built *and peepholed* while the merge is still open, and the phis closed before control.
Get it wrong and the phi momentarily reads `int=0`, `i + 1` folds to the literal `1`, and the
entire loop is deleted with no error reported. `n-set-def!` hard-errors on the one part of this
that is cheap to detect (closing control while a phi is still open); the general defence is M7's
phase structure, which analyses to a fixpoint before transforming anything.

**Deferred out of M2, deliberately:**

- **Lazy phi creation** is a front-end concern, not an IR one. It is the trick where a parser
  creates a phi only when a variable is actually assigned in a branch. With no parser yet there
  is nothing to be lazy about, so it moves to the `dyn` front end.
- **`Never` and never-exit loop breaking.** `Never` exists solely to serve the rewrite that
  gives an infinite loop an exit edge, and that rewrite exists solely so global code motion can
  assume every block reaches `Stop`. Adding the op now, with nothing implementing its purpose,
  would be worse than not having it. It lands with GCM in M10.
- **The loop tree** (`idom`, `idepth`, nesting depth). **Now DONE**, gated by
  `tests/loop-tree-test.coil` (5 tests: straight line, diamond, loop, nested loops, staleness).
  Natural loop bodies by backward reachability from each back edge; nesting depth is the number
  of bodies containing a node. A loop header's immediate dominator is its entry and never its
  back edge, which is both correct and what makes the idepth relaxation converge.

  Both tables are rebuilt wholesale by `g-build-loop-tree!` with no incremental maintenance, and
  a `cfgver` epoch makes reading a stale one a hard error. Dominators computed against a stale
  CFG are a classic scheduling bug that only appears under optimisation, and it is much cheaper
  to refuse the read than to debug the schedule. Asking for the idepth of a *data* node is also
  an error rather than a `-1`: a data node has no dominator depth until a scheduler gives it one.

---

## M3. Tooling: textual IR, verifier, evaluator

The textual IR printer and parser, round-trip tested. Graphviz output. The graph verifier
(well-formedness, edge symmetry, type monotonicity). An **IR interpreter** over the ideal
graph, which becomes the differential oracle for everything after this point.

**Gate.** `print` then `parse` is the identity on the whole corpus; the verifier is green on
the corpus and *red* on a set of deliberately corrupted graphs (a verifier that never fails is
not a verifier); the interpreter agrees with expected results on the corpus, before and after
`Opto`.

**Status: DONE**, all four slices. **138 tests green across eight suites** (`tools/gate.sh`):
ty 26, node 27, control 19, loop-tree 5, verify 19, eval 23, text 12, gtext 7, plus 19 diagram
fixtures rendered and the gallery page rebuilt. M3 stayed open through 3a on purpose, because
the milestone's gate says "`print` then `parse` is the identity on the whole corpus" and 3a proved
that for types only; slice 3b is the clause itself, and `src/gtext.coil` gated by
`tests/gtext-test.coil` closes it. Four items were still open at its close; the two that remain are
under [The next concrete piece of work](#the-next-concrete-piece-of-work), and all four are written
up in full in [JOURNAL.md](JOURNAL.md).

**What M3 taught**, four things, because the detail below is long and these are the parts that
outlive it:

- **ANY is the ABSENCE of information; every other high type is a CLAIM someone computed.** M2 had
  already produced "an analysis may act on a provisional type, a transformation may not", and M3
  found that six rewrites were satisfying it with a test that proved nothing (`= (t-xctrl)`, on an
  axis whose high element IS XCtrl). The durable form is `n-ty-proven?`: transitive, and a FIXPOINT
  test rather than a "not ANY" test, because a stale LOW type is exactly as provisional as ANY.
- **A tool is only a tool if it can FAIL.** A verifier that never reports, a printer whose text
  cannot be told apart from a shorter one, and a round trip that would pass for a printer emitting
  only op names are all worth nothing. Every check in this milestone therefore has a NAMED code and
  a test asserting that specific code fires, and every identity claim has a counted floor under it
  saying how much text or how many nodes it actually compared.
- **The oracle has to RUN the program.** Golden strings and `g-verify` both stayed green through a
  merge deleted from a diamond, a Return dropped from Stop, and a discharged type check; the
  interpreter is what noticed, because a graph missing an arm is still a structurally valid graph.
- **A fixture can be correct by accident, and its CONSTRUCTION ORDER is part of it.** The flat
  `if (0)` passed while the same construct one level down miscompiled; the first raw diamond
  reproduced nothing on 200 seeds until it was built the way the witness was built. So a gate says
  which shape and which order it needs, and reverting each fix one at a time is the only way to
  learn whether the gate tests what it claims.

**Open at M3's close**, collected here so M4 does not have to find them at the bottom of a long
entry. All four are written up in full in [JOURNAL.md](JOURNAL.md); nothing is red.

1. **`g-analyze!` then `iterate!` deleted a whole If on 15 of 200 seeds** on the nested-guard
   fixture, from a FULLY ANALYSED graph, and does not reproduce now. It is the only item on this
   list that is a suspected live miscompile rather than a design decision, and it is not an instance
   of the law above. **A gate for it is the first thing to write when it is picked up**; "it does not
   reproduce" means only that no current gate covers that phase order.
2. **`region-dead-path` acting on a provisional `~ctrl`**, now formally M7's phase structure.
3. **A dead node can be wired as an input** and nothing says so at the wiring site. The named panic
   in `n-add-def!` was written and reverted, because one existing test reuses an Arg the peephole
   under test has already killed; closing it means fixing that test's hygiene.
4. **The `int=[min..max]` versus `w0` spelling wart** in the type format: a decision to make, not a
   fix, and the graph format inherits whichever way it goes.

---

## M4. Memory SSA and shapes

Alias classes, `New`, `Load`, `Store`, `MemMerge`, memory `Phi`. Object shapes with a
compile-time transition graph, shape sets in the lattice, fixed-offset field access through a
known shape.

**Gate.** Load-after-store forwarding, store-after-store elimination and dead-allocation
removal all demonstrated by golden tests; differential tests green; a shape-polymorphic site
keeps two inline paths instead of collapsing.

Where the clauses stand after 4c-1:

| Clause | State |
|---|---|
| the ops exist, verify, print, reparse and RUN | done (`tests/mem-test.coil`) |
| differential tests green | done, on seven memory fixtures in both build modes, and the comparison is now the reachable HEAP rather than an object's allocation index (D14) |
| a shape-polymorphic site does not collapse | done for the LATTICE half: the merged pointer keeps both shape bits, the two arms keep distinct alias classes, and the `MemMerge` describes their union. The two INLINE PATHS are M9's, since a guard is what makes a path, and 4c's forwarding is what makes each path worth having. It is now RUN on both arms and the merged memory is read back through the returned object, so swapping either memory phi's arms is red; before that, no seed ever took the else arm and nothing read the merge, so the canonical LAW 5 miscompile left the whole gate green |
| load-after-store forwarding | **done (4c-1)**. `load-idealize`: the load's memory input is a Store on the SAME alias class whose pointer is the SAME NODE, so the load is that store's value. Decided structurally and never from a type; Simple's offset-overlap rule is deliberately not ported, because it reads a type and then rewrites irreversibly. `fx-object` (19) forwards; `fx-object-two` (23) is the negative witness, two allocations of one shape where the pointer check is the only thing between the right answer and 2 |
| store-after-store elimination | 4c-2 |
| dead-allocation removal | partly, and for free: killing a `New`'s last projection drops its last use and the cascade collects it. A store to an object nothing reads still survives |

## M5. Dynamic values end to end

NaN boxing, `Box`/`Unbox`, `TypeTest`, the full value lattice with unions. `dyn` grows objects,
closures and dynamic dispatch.

**Gate.** `(+ a b)` on values inferred as integers compiles to an unboxed integer add with no
box, unbox or branch; the same expression on `Dyn` compiles to the generic path and still
returns the right answer under differential test; a `Box` immediately consumed by an `Unbox`
never survives.

## M6. Functions, calls, closures

`Fun`, `Parm`, `Call`, `CallEnd`, `Return`, the function-pointer axis in the lattice with
`fidx` sets, the linker table, inlining.

**Gate.** Recursion, mutual recursion, higher-order calls and closure capture all correct
under differential test; an indirect call with a single reaching target is devirtualised.

## M7. Closed-world inference

The optimistic interprocedural dataflow pass: reset to `TOP`, propagate to a fixpoint, build
the call graph as function-pointer types stabilise, under an explicit whole-world assumption.
TypeScript-style annotations discharged against the inferred types.

**Gate.** An inference-quality corpus: a table of programs with the type each interesting
value must be inferred to. Regressions in inference precision fail the build, not just
regressions in correctness. Plus: the optimistic pass never produces a type below what the
pessimistic pass produced (asserted per node), and reaches a fixpoint on every corpus program.

## M8. The GC contract

`Safepoint` with relocating projections, abstract `Barrier`, safepoint placement at `New`,
`Call` and loop back edges. The R1/R2 verifier.

**Gate.** The verifier rejects a hand-written graph that keeps an interior pointer live across
a safepoint, and one that uses a pre-safepoint reference after it; redundant barrier
elimination demonstrated; all earlier gates still green with safepoints present.

## M9. Specialisation and guards

Cloning per argument-type tuple, guard chains at call sites, the retained generic version,
cold-path marking, the specialisation cost model.

**Gate.** A polymorphic call site produces a monomorphic clone plus a generic fallback, both
reachable and both correct under differential test; the guard folds away entirely when
inference proves the type; specialisation never changes observable behaviour.

## M10. arm64 backend

Instruction selection, global code motion, list scheduling, graph-colouring register
allocation with splitting, encoding, Mach-O output. Adapts Coil's `a64.coil` and `macho.coil`.

**Gate.** Every corpus program compiles to a native binary whose output matches the IR
interpreter exactly; the register allocator terminates on all of them with no unassigned live
range; disassembly of a chosen kernel reviewed by hand against expectation.

## M11. A real collector

Bump allocation with a slow-path call, stack map emission, a moving generational collector,
and GC stress testing.

**Gate.** A collect-on-every-allocation stress mode passes the whole corpus; heap invariants
verified after every collection; no reference survives a collection unrelocated.

## M12. TypeScript front end

A real TypeScript parser and lowering to the core IR. Annotations become boundary casts.
Structural typing, generics erased to shapes, unions to lattice unions.

**Gate.** A published TypeScript test corpus runs correctly; annotations demonstrably reduce
guard count relative to the same code unannotated.

## M13. Performance

Benchmark suite against node/V8, with full result tables (every axis, raw samples, ratios).
Profile input to the specialisation cost model.

**Gate.** Published numbers per benchmark against V8, honestly reported including the losses.

---

## Deferred, deliberately

- **Exceptions and `try`/`catch`.** Calls will need exception edges and landing pads become
  ordinary CFG merges. This is a known, planned shape rather than an oversight; it lands
  between M6 and M7 if the `dyn` corpus needs it sooner, and it must be designed before M8
  because an exception edge is a safepoint.
- **Additional CPU targets** (x86-64, RISC-V). The `MachNode` interface is kept narrow
  precisely so this stays a contained job, and porting one is the real test of whether it is
  narrow enough.
- **ELF output** for the Linux build host.
- **Open-world mode.** Keeps the generic versions, drops the guard fold-away, supports `eval`.
- **`defnode`**, the metaprogram that generates per-op tables from one declaration. Lands once
  the set of per-op behaviours has stopped changing, which will be around M5.
