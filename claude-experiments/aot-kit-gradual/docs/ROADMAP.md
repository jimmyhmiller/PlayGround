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

## M0. Foundations: the lattice

The type lattice, interned, with `meet`, `dual`, `join`, `isa`, and the constant/high
predicates. Simple types, the value axis (kind bitset with integer and float refinements),
tuples.

**Gate.** Lattice property tests pass: `meet` is commutative, associative and idempotent;
`dual` is involutive and order-reversing; `meet` is a lower bound and `join` an upper bound;
`isa` is reflexive and transitive; `x isa ALL` and `ANY isa x` for all `x`; every ascending
chain terminates.

**Status: DONE.** `src/ty.coil`, gated by `tests/ty-test.coil`: 25 tests, laws checked
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

**Status: DONE.** `src/node.coil`, gated by `tests/node-test.coil`: 16 tests. Ops through
`Return`, plus an opaque `Arg` node standing in until `Proj` arrives in M2.

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

**Status: DONE.** `tests/control-test.coil`, 17 tests. Reachability is not a separate pass: an
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

## M3. Tooling: textual IR, verifier, evaluator

The textual IR printer and parser, round-trip tested. Graphviz output. The graph verifier
(well-formedness, edge symmetry, type monotonicity). An **IR interpreter** over the ideal
graph, which becomes the differential oracle for everything after this point.

**Gate.** `print` then `parse` is the identity on the whole corpus; the verifier is green on
the corpus and *red* on a set of deliberately corrupted graphs (a verifier that never fails is
not a verifier); the interpreter agrees with expected results on the corpus, before and after
`Opto`.

**Slice 1, the verifier: DONE** (`src/verify.coil`, `src/corpus.coil`, 19 tests). One named code
per check, which is what stops the corrupted-graph half from being vacuous: "verify returned
non-zero" is satisfied by a verifier that always fails, whereas "corrupting the phi/region lockstep
reports `VERR-PHI-ARITY`" is a claim about a specific check noticing. It runs `compute` and never
`idealize`, because unlike Simple's equivalent our `idealize` MUTATES (`region-remove-path!`
deletes edges, `stop-idealize` deletes an input), so a reducibility check would silently transform
the graph it was verifying.

**Slice 2, the IR interpreter: DONE** (`src/eval.coil`, 12 tests). The differential oracle is live:
every corpus program that exists in both a raw and an optimised build produces the same observable
result, over 12 argument bindings. From here on that is the gate for every optimisation.

Three things worth keeping from it:

- **Arguments are bound to values INSIDE their declared types**, keyed by type rather than node id.
  Feeding an argument something its declaration forbids does not test the optimisation, it tests
  what happens when you lie to the compiler. Keying by id would break because optimisation deletes
  arguments and shifts every later id.
- **It refuses to guess.** Integer overflow reports `EV-OVERFLOW` rather than wrapping, because
  JavaScript promotes to a double and doing that properly needs M5's value domain. A wrapped answer
  would make the oracle quietly disagree with the language it exists to define.
- **A failing `Cast` is a compiler bug, not a program condition**, so it has its own status. A guard
  placed where it was not proven is now caught by running the program.

It also found two things immediately. A bodyless `Loop` whose back edge is itself is **not** an
infinite loop: control's only forward step from the header is the `Return`, so nothing ever
iterates. The corpus loop fixture was exactly that shape, drawn for the diagram and never run, so
it has a real exit test now and counts to 5. And the interpreter's truthiness is checked directly
against `ty-can-be-truthy?`, because an interpreter that takes a different branch than the compiler
predicted is worse than no oracle at all.

**Slice 3a, the exact textual TYPE form: DONE** (`src/text.coil`). Printer and parser live in one
file because they are one format. `ty-print-exact` decides every axis BY FIELD VALUE, `ty-parse`
rebuilds through `val-make`/`ty-intern` so no second canonicalisation exists, `ty-render` writes to
a caller buffer, and `ty-injective?` answers "does this type's text decode back to it" — which
`ty-print-exact` consults, hard-erroring rather than emitting an ambiguous string.

Four things worth keeping from it:

- **The debug printer is measurably non-injective, and it is now labelled as such.** `ty-print`
  consults a refinement's KIND BIT before printing it, so an axis whose bit is off is invisible.
  Two types *currently in the interning table* collide: `+0.0` and `-0.0` both print `flt=0.0`, and
  `int=[0..10]` at widening counters 0, 1 and 2 are three types with one string. The exact form
  therefore prints float constants as their signed decimal BIT PATTERN (`{f}` is a fixed 6-digit
  display and NaN is not reflexive under `fcmp` while interning needs exact identity) and prints
  the widening counter as `w<n>` whenever it differs from the default the interval implies.
- **`fmt`'s `{d}` renders INT64_MIN as "-0".** Cosmetic in a dump, a correctness bug in an exact
  format: `t-fun-con 63` is the bitset INT64_MIN and `fun#-0` decodes to 0. There is now one
  decimal writer, `dec-print`, used by both printers.
- **A tuple's members and a value's refinements cannot share a separator.** With a space for both,
  `[int flt=b1]` is simultaneously a 1-tuple whose member carries a float constant and a 2-tuple,
  and no lookahead can fix a printer that emits one string for two types. Refinements keep the
  space (so every value's spelling, and therefore every golden string in the project, is identical
  under both printers); members take a comma.
- **It found a live miscompile in the lattice.** `ty-meet-tuples` pushed each member's meet into
  the shared kid array as it computed it, but meeting a member INTERNS, and interning a tuple
  appends to that same array, so a tuple of tuples ended up pointing at its inner tuple's members:
  `[[ctrl int] flt] meet [[ctrl dyn] flt]` returned `[ctrl dyn]`, and `meet` was not a lower
  bound. Nothing caught it because no tuple in the corpus has a tuple member. Fixed by making
  every producer reserve its member window first (`ty-tuple-open`/`ty-tuple-put!`/`ty-tuple-close`),
  which is also what lets a parsed tuple have unbounded arity — needed for M6's `Start`, one output
  per parameter.

Gate coverage the corpus alone would not give: the corpus contains no partial widening, no
off-kind refinement and no extreme bitset, so a round-trip test written only over it would pass
with the widening counter unprinted. The adversarial fixtures and the axis cross-product are the
defence, and the cross-product is now a test rather than a sentence:
`every_axis_combination_is_injective` in tests/text-test.coil enumerates all 1024 kind sets x 3
intervals x 3 widening counters x 3 float sizes x 2 function sets x 2 shape sets, each with its
dual, and pins the count at 221184 with `XPROD-FLOOR`. It runs in 1.4s. It fails if any axis stops
being printed, because that axis's two settings then collide on one string and `ty-injective?`
rejects both; verified by disabling the widening-counter print, which takes it and five other
tests red. This entry previously CLAIMED that cross-product as existing coverage when the largest
injectivity gate actually running was the 120-type sweep over the interning table.

**The M3 adversarial review: eight findings, all fixed.** Worth keeping, because six of the eight
were invisible to a green gate and three were live miscompiles:

- **A Cast was discharged on an unanalysed input.** `ty-isa a b` is `meet(a,b) == b` and
  `meet(ANY, t) == t`, so `ty-isa ANY t` is TRUE FOR EVERY TARGET. A raw graph starts at ANY, so
  every Cast on it discharged itself. What that deletes is a TYPE CHECK, D4's only guard mechanism,
  so it is LAW 3 in its most expensive form: `if (p) a = 1; else a = null; return (int)a;` answered
  `null` when built merge-first and `EV-CAST` when the analysis ran first. Two answers for one
  program. `t-top` alone is NOT the guard either: a phi whose declared `int` meets a `null` arm
  reports `~dyn`, which is high but not ANY, and `ty-isa ~dyn int` is also true. The guard is
  `ty-high?`, which is what `cast-compute` twelve lines away already used. Gated three ways: 200
  worklist seeds (unguarded it discharged on 108, i.e. the answer depended on the seed), the
  contradictory-phi structural case, and the differential run reporting EV-CAST rather than `null`.
- **A multi node is IN PROGRESS until all of its projections exist**, and that is LAW 4 for a multi
  node rather than for a merge. Peepholing an If's projections ONE AT A TIME folds the first to
  XCtrl, which drops the If's last use, so the kill cascade takes the If itself; the sibling is
  then peepholed against a corpse whose `ins` `n-kill!` cleared but whose `ty` it did not, so the
  stale tuple still matched and the rewrite read index 0 of a zero-length list. On the flat
  `if(0)` fixture that read returned Start, which M3 pins, so the wrong answer was accidentally the
  right node and the corpus stayed green. Nested inside a live branch it returned a DEAD node: a
  live Return whose control is dead, `VERR-DEAD-INPUT`, and `EV-STUCK` on the arm that should have
  returned 3. Three layers now: `n-multi-open!`/`n-multi-close!` pin the multi across the window
  (and `n-if-arms!` is the builder every call site uses), `cproj-idealize` aborts BY NAME on a dead
  multi instead of reading stale state, and `n-in`/`n-out` are bounds-checked so an out-of-range
  read can never again return a plausible node id. `15-nested-dead-branch` is the fixture, and it
  found a SECOND layer while being written: the surviving arm is the If's own control input, so it
  is used by nothing but the dying If at close time and has to be pinned across the collect too.
- **Four arithmetic identities were false for their own operands.** `x+0`, `x-0`, `x*1` and `-(-x)`
  all returned `x` with nothing established about `x`'s type, in a language where `"3"+0` is `"30"`
  and `-` `*` and unary `-` coerce. Each replaced a node whose type the lattice had computed with a
  node of a DIFFERENT type. Their siblings in the same block (`x*0`, `x==x`) already carried
  `ty-int-only?`, so this was an inconsistency, not a decision. It is observable TODAY and not only
  at M5, because `bool` is non-numeric and does exist in the interpreter's value domain: `ev-arith`
  reports EV-TYPE while the unguarded peephole answered `true`.
- **`meet` was not a lower bound for a tuple of tuples, and nothing in the suite could see it.**
  The fix (reserve the member window) was already in, but `ty-sample` had no tuple with a tuple
  member, so reverting `ty-meet-tuples` to the pushing form left the whole gate green. `ty-sample`
  now carries the exact witness pair plus a 1-tuple and a 0-tuple, which puts them under all the
  M0 laws and their duals: on the reverted producer, three laws break at once (associativity,
  `a isa meet(a,b)`, `join(a,b) isa a`) with those two as the printed witnesses.
- **Printing mutated the type table.** `ty-print-exact` round-trips through the PARSER to decide
  whether it may emit, the parser reserves a member window for every tuple, and the kid array is
  append-only, so printing an interned tuple leaked its reservation: unbounded in the number of
  PRINTS, with `ty-count` flat the whole time. Same leak on every repeat construction, and
  `if-compute` builds its result with `t-tuple2` on every visit, so an M3 fixpoint leaked per If
  revisit. `ty-tuple-close` now looks the tuple up BEFORE interning and drops the whole tail on a
  hit. Sound because a pre-existing outer tuple implies every member id pre-existed, hence nothing
  in that subtree was newly pushed; checked rather than trusted by comparing `ty-count` across the
  window, and it simply leaves the window in place if the table grew (leaking is the safe answer
  there; a panic would be worse than the bug it guards).
- **A legally interned type could be unprintable, and the abort blamed the wrong file.**
  `ty-print-exact` reported "the exact form of this type is not injective" for a 300-deep tuple
  whose 604-byte form is perfectly unambiguous; the real cause was the parser's TEXT-MAX-DEPTH
  budget. Both halves are fixed. `ty-round-trip-check` returns a NAMED cause (RT-OK,
  RT-PARSE-FAILED, RT-DIFFERENT-ID) and each gets its own message naming the file to read, so
  "the format is ambiguous" is now said only when the text really decoded to a different type.
  And the policy question the depth bound raises is answered rather than left open: the format is
  part of the type's contract, so `TY-MAX-TUPLE-DEPTH` lives in src/ty.coil, `ty-tuple-close`
  refuses to MINT a type deeper than the format can represent (at construction, with the caller on
  the stack), and TEXT-MAX-DEPTH is defined as the same constant with a gate pinning them equal and
  round-tripping a tuple at exactly the bound. Depth is recorded once per interned type, so the
  check is O(arity). Arity stays unbounded, which is what M6's `Start` actually needs.

**Open, found while gating the above: an irreversible rewrite still acts on a provisional `~ctrl`.**
`region-dead-path` requires XCtrl exactly rather than merely high, which is the letter of LAW 3.
But XCtrl is itself an optimistic answer when the If's PREDICATE is unanalysed: `if-compute` reads
a high predicate as "no value can be here", reports `[~ctrl ~ctrl]`, and a Region peepholed at that
moment loses both paths and takes the whole branch with it. Building a diamond FULLY raw and then
running `iterate!` does exactly that; it is why the corpus's raw fixtures use `g-analyze!` (which
changes no edges) and why the eager path never sees it. The honest guard is not local: the
projection's own type is not ANY, so `n-inputs-analysed?` does not catch it, and the property
needed is transitive. This is the case M7's phase structure exists for (analyse to a fixpoint, then
transform), and it is recorded here rather than papered over with a partial local check. Slice 3b
must not build a diamond fully raw until it is resolved.

**Slice 3b, the graph round trip: still to do.** LAWS 3, 4 and 5 do not bite in 3a, because nothing
in it constructs, rewrites or peepholes a graph. They bite in 3b: the graph parser has to build a
loop's region and phis inside the in-progress window without letting a peephole fire, has to keep
region and phi arity as one operation, and has to build a multi node's projections inside
`n-multi-open!`/`n-multi-close!` (the review above turned forgetting that into a named abort rather
than a miscompile).

## M4. Memory SSA and shapes

Alias classes, `New`, `Load`, `Store`, `MemMerge`, memory `Phi`. Object shapes with a
compile-time transition graph, shape sets in the lattice, fixed-offset field access through a
known shape.

**Gate.** Load-after-store forwarding, store-after-store elimination and dead-allocation
removal all demonstrated by golden tests; differential tests green; a shape-polymorphic site
keeps two inline paths instead of collapsing.

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
