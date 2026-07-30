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

**Status: DONE**, all four slices (`tools/gate.sh`). M3 stayed open through 3a on purpose, because
the milestone's gate says "`print` then `parse` is the identity on the whole corpus" and 3a proved
that for types only; slice 3b is the clause itself, and `src/gtext.coil` gated by
`tests/gtext-test.coil` closes it. Two items are still open and are carried forward at the end of
this entry rather than dropped: the type format's one spelling wart, and `region-dead-path` acting
on a provisional `~ctrl`, which is now formally M7's.

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

**Slice 3a, the exact textual TYPE form: DONE** (`src/text.coil`, gated by
`tests/text-test.coil`: 12 tests; the whole gate is 114). Printer and parser live in one
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
injectivity gate actually running was the sweep over the interning table (131 types, and that
sweep is still the one that proves the format works on the types the project actually builds).

**What the gate deliberately does NOT assert.** Four clauses of the slice's brief are not provable
as written, and recording that is cheaper than a test that pretends otherwise:

- **`ty-injective?` returning FALSE is unreachable, so `ty-print-exact`'s hard-error branch is
  gated in one direction only.** That is the property being gated rather than a hole in the tests:
  no type whose exact form is ambiguous exists, and after the review moved the depth bound into
  `ty-tuple-close` no type the format cannot represent can be MINTED either, so the last remaining
  handle on the false branch closed. Liveness is therefore measured the only way it can be: forcing
  the predicate always-false takes 9 of the 12 tests red, forcing it always-true keeps all 12 green.
  Instead the guard is asserted TRUE over every interned id and every adversarial fixture. Making
  the ambiguity branch reachable would need an injection point that nothing else in the design wants.
- **TERR-TUPLE-COUNT is not provoked, and the code says why.** It is an assertion between two
  implementations of "how many members are here" (the comma scan and the member loop), not a report
  about input, and every early stop in the member loop sets a different code first. If some input
  ever does reach it, that is a real find about the grammar, not a badly written test.
- **`ty-print` and `ty-print-exact` do NOT agree on `~dyn`, on `[ctrl int]`, or on a float
  constant**, and cannot: VK-NONE makes the debug printer return before any refinement, members must
  take a comma to stay injective, and a float has to print its bit pattern. All three are pinned as
  required DIFFERENCES with both spellings written out. The intent behind the clause, that no golden
  string and no gallery label moved, is gated instead by 19 named forms pinned in both printers and
  by 74 labelled types across all 15 gallery fixtures with 0 disagreements.
- **The debug printer's collisions are asserted by TYPE, not by id.** Hardcoding an id would make
  the test break on any change to interning order, which would teach nothing. So the colliding pairs
  are built constructively (one debug string, two exact ones, printed), and the scale is measured
  globally: the debug form fails to decode back for 85 of 131 interned ids, and 41 of those decode
  to the WRONG type rather than failing.

**Open, a wart in the format rather than a bug in either half of it.** The parser defaults the
widening counter off whether the INT axis was WRITTEN; the printer decides by whether the INTERVAL
is full. They agree everywhere except on an explicitly written full interval, so `int=[min..max]`
and `int w0` are two texts for one type, which is exactly what the format refuses for kind sets
(`num|int` is TERR-DUP-KIND). Nothing is red, because print-then-parse is still the identity: the
printer only ever emits the second. Closing it is a decision and not a fix, either reject an
explicit full interval or derive the default from the interval on both sides, which is why it is
recorded here instead of asserted either way.

**Two bounds, both reporting by name rather than truncating.** Tuple NESTING at
`TY-MAX-TUPLE-DEPTH` (256, because the parser recurses on the C stack and a type nested that deep
is a fuzzer artefact, not IR), and the injectivity scratch buffer at 64 KiB, whose overflow is its
own named panic and deliberately NOT "not injective", because those are two different bugs in two
different files. Tuple ARITY has no bound at all: a printer that refuses to print a legally
constructed type is worse than no printer, and M6's `Start` gets one output per parameter.

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
  **This one was only PARTLY closed, and the second review found the rest**: `not ty-high?` is not
  the proven answer either, so the same Cast was still discharged one nesting level deeper. See
  "the second review" below; the fix is `n-ty-proven?` and the gate is
  `a_cast_nothing_proved_survives_a_merge_of_merges`.
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

**CLOSED, by the second review below: an irreversible rewrite acting on a provisional `~ctrl`.**
This was recorded here as open, with the note that "the honest guard is not local, and the property
needed is transitive". That was right, and the property now has a name: `n-ty-proven?`. It is also
recorded here that it was one instance of something larger; the entry below replaces it.

## The second adversarial review: seven findings, one law, all closed

Every one of the seven was a different consequence of ONE mistake, and it is worth stating the
mistake in one sentence because five separate "fixes" had already been written against instances of
it and none of them had named it:

> ANY is what an UNANALYSED node reports; every other high type is a CLAIM someone computed. A
> producer must not mint a claim out of the absence of information, and a consumer must not read a
> claim as a PROOF.

The three earlier repairs (`phi-single-input`, `cproj-idealize`, `stop-idealize`, and the constant
fold) all moved from `ty-high?` to `= (t-xctrl)` and believed that was the proven answer. On the
control axis it is not: **XCtrl IS the high element of that axis**, so `meet(~ctrl, ctrl)` is `ctrl`
and a node reporting `~ctrl` may still fall. `region-compute` made that concrete by seeding its meet
with `~ctrl`, so a merge none of whose paths had a type reported PROVABLY UNREACHABLE out of nothing.

What was found, in the order the damage runs:

- **A merge whose input is a merge was miscompiled to a single wrong constant** on 55 of 200
  worklist seeds, using the construction order the project itself documents as mandatory (arms
  eager, merges raw, worklist, `iterate!`). `g-verify` reported 0 violations, because a graph with an
  arm deleted is a structurally valid graph.
- **Two merges feeding a merge deleted the entire program** on 31 of 200 seeds: `stop-idealize`
  dropped the only Return and the kill cascade did the rest. Stop with no inputs at all, `g-verify`
  clean, and the printer omits dead nodes so the diagram showed Start and Stop with nothing between
  them. Note Simple does NOT delete a Return; that rewrite is this project's own addition.
- **The Cast finding from the first review was only half closed.** `phi-compute` legitimately SKIPS
  a region path whose control is still high, so a Phi momentarily reports the type of the single arm
  that happens to have been analysed. That type is LOW, it satisfies the Cast, and the type check
  was deleted on 110 of 200 seeds; the surviving graph printed the Phi's settled type as `null|int`,
  so the analysis proved the Cast unsatisfiable AFTER the Cast was gone.
- **`if-compute` reported "neither arm is taken" for an If whose own CONTROL was unanalysed**
  (`(and (!= c (t-ctrl)) (!= c (t-bot)))` is satisfied by ANY), and `cproj-idealize` then replaced
  BOTH live projections with XCtrl.
- **The constant fold's guard asked the wrong question.** `n-inputs-analysed?` is one level deep,
  and for a Phi the relevant fact is two or more levels away. `if (p) x=1; else { if (q) x=true;
  else x=true; } return x + 0;` folded to `Const int=1` on 108 of 200 seeds, and with a
  non-constant live arm the `x + 0` identity alone still lost the Add on 99 of 200 and answered
  `true` where `ev-arith` correctly reports a type error.
- **The dep mechanism `phi-compute` relies on was inert.** Two independent halves: the dependency
  was recorded on the REGION rather than on the region's control input (Simple does `addDep(r.in(i))`
  and the port had lost the `.in(i)`), and `g-analyze!` never called `n-move-deps!` at all. A region
  already at `ctrl` does not change type when a SECOND path goes live, so nothing re-queued the phi
  and `if (p) x=8; else x=0;` published `int=8`. That is the too-SPECIFIC direction, which is the
  miscompile direction, and `n-set-ty-falling!` cannot catch it because the stale type never changes.
- **`ty-high?` answered a per-axis question with a single-axis test.** `join(int=8, int=[0..5])` is
  `int=~[5..8]`: the INT bit present, the range inverted, which is the high element of that axis and
  which `ty-print` even spells with a `~`. Reading it as LOW made four computes answer with their
  axis BOTTOM and then RISE when the input fell, and `g-analyze!`'s own monotonicity assertion
  aborted the compiler on a 14-node program. It also disabled `cast-compute`'s freeze rule for every
  Cast that disagrees with its input on the RANGE rather than on the kind.

### What closed them

**`n-ty-proven?`, and every irreversible rewrite asks it.** The answer is "no node in this node's
input cone is unanalysed, and every type in the cone is at its fixpoint". Both halves are needed and
the second was found the hard way: a first version tested only for ANY, and a FLAT diamond still
compiled to `return 1` on 9 of 40 seeds, because the phi had met `int=0` with an as-yet-unanalysed
`8` (ANY is the identity for meet) and the `8` was analysed one step before the Add was popped. A
stale LOW type is exactly as provisional as ANY. The test is therefore the fixpoint condition
itself, `n-ty (n-compute)` on every node in the cone; being on the worklist is the wrong proxy,
because `iterate!` sweeps by pushing every live node.

It has to be TRANSITIVE, and that is forced by `phi-compute`, whose optimism is load-bearing: a loop
phi must be able to report its entry value while the back edge is unanalysed, or the phi, the exit
test and the back edge sit at ANY forever waiting on each other. So a Phi's type is provisional
while its own type and all of its inputs' types are low, which no local check can see.

**Six call sites, one predicate**: the constant fold, `n-int-con?`, `n-proven-int-only?` (the five
arithmetic identities), `cast-idealize`, `n-in-proven-xctrl?` (`region-dead-path`,
`phi-single-input`, `stop-idealize`), and `cproj-idealize`. `n-int-con?` is the one nobody had
suspected: it asks "is this operand the literal 0", a Phi that momentarily reports `int=0` says yes,
`should-swap?` moves it to the right because constants belong on the right, and `x + 0 = x` then
returns the OTHER operand as the whole program's value. No fold and no control rewrite is involved,
which is why the four guards on those did not see it.

**Two producers stopped manufacturing proof**: `region-compute` returns ANY while any path is
unanalysed, and `if-compute` returns ANY while its control or its predicate is. These are now
redundant with the consumer-side proof for every fixture in the corpus, and both are kept anyway,
because a compute that reports "provably unreachable" for a node nobody has looked at is a false
statement about the program regardless of who is reading it, and `g-analyze!` publishes those types
to the verifier, to the diagrams, and to M7's optimistic pass. Each has its own unit gate for
exactly that reason: the end-to-end gates cannot see them while the consumers refuse to act, so
without the unit gates the pair would silently degrade to one.

**`val-high?` is an INHABITATION test now**, per axis and per kind, and its old comment claiming
that erring toward `false` "never breaks correctness" is deleted, because erring toward false is
precisely what made four computes non-monotone. Note this is not a lattice-POSITION test:
`int=~[5..8] isa ~dyn` is false, since the product lattice does not identify "uninhabited" with "at
or above the value top".

**`iterate!` sweeps.** A proof-gated rewrite is DEFERRED rather than declined, and nothing re-queues
it: the blocker's type may never change again, so neither `wl-push-outs!` nor the deps list will fire.
One drain therefore reached a fixpoint of "what could be proven at the time", which is a function of
the pop order, and `1 + 2*3` stayed unfolded on some seeds. So: drain, and if that drain changed
anything, push every live node and drain again. `iterate_reaches_a_peephole_fixpoint_on_every_optimised_fixture`
is the gate, and it asks for the claim (one more sweep changes nothing) rather than for the mechanism.

### The gates, and how each one fails without its fix

Three new corpus fixtures (`16-merge-of-merge`, `17-merge-tree`, `18-diamond-raw`, floor raised
15 -> 18), of which the last is the FIRST raw-analysed fixture containing a merge: every other one is
straight-line, which is why nothing exercised `phi-compute` under the analysis-only pass. Twelve new
tests, 121 -> 138.

**Every fix was reverted one at a time and the gate re-run**, which is the only way to know a gate
tests what it claims. That exercise changed the work twice:

- **A fixture's CONSTRUCTION ORDER is part of the fixture.** The first version of the raw diamond
  created its constants inline and pushed with `wl-push-live!`, and it did not reproduce the
  miscompile on ANY of 200 seeds. `wl-pop!` picks a random INDEX, so which orders a seed can produce
  depends on how the list was filled. It is written the way the witness was written now, and says so.
- **Two producers and one consumer overlap**, so reverting any one of them left the gate green even
  though the miscompiles were real. That is why `region-compute` and `if-compute` have unit gates on
  their ANSWER (`an_unanalysed_merge_does_not_claim_to_be_unreachable`,
  `an_if_whose_control_is_unanalysed_does_not_claim_neither_arm`) rather than only end-to-end gates,
  and why the proven-XCtrl requirement has one too
  (`a_provisional_xctrl_is_not_a_licence_to_drop_a_control_path`, which builds a graph where a
  projection's stored type says `~ctrl` and the claim is a lie). Without those three, the pair would
  have degraded to one at the next refactor with nothing to say so.

**Five changes are NOT individually gated, and here is the honest reason for each.** Three are the
axis-level answer stated where the axis is computed, and are unreachable while `val-high?` is right,
because each is behind that same compute's own `ty-high?` test: `minus-compute`'s and
`not-compute`'s uninhabited-input branches and `if-compute`'s neither-truthy-nor-falsey branch.
(`int-arith`'s is reachable through a direct call and IS gated.) The other two are
`g-analyze!`'s `n-move-deps!` and `iterate!`'s sweep: both are the mechanism behind a property that
is gated (the analysis reaches a type fixpoint; `iterate!` reaches a peephole fixpoint), and with
`region-compute` reporting ANY there is no constructible graph today where either is the only thing
holding the property up. They are kept because the property should be true by construction rather
than by an argument about which pushes happen to reach which node.

### Still open

- **A dead node can be wired as an input**, and nothing says so at the wiring site. Peephole a raw
  merge whose only user is a Phi that then collapses, and the merge loses its last use and is killed;
  the id in your hand is now a corpse, and `n-add-def!` accepts it. The graph that results fails
  `g-verify` with VERR-DEAD-INPUT far from the line that caused it, and the interpreter reports
  EV-STUCK. A named panic in `n-add-def!` was written and reverted: `identity-peepholes-do-not-change-a-non-numeric-type`
  reuses an Arg after the peephole under test has killed it, so closing this means changing that
  test's hygiene, which is a separate change from these seven findings.
- **`g-analyze!` followed by `iterate!`** on the nested-guard fixture deleted the whole If on 15 of
  200 seeds while this review was being written. It reproduced from a FULLY ANALYSED graph, so it is
  not an instance of the law above and needs its own investigation. It does not reproduce now, which
  means only that no current gate covers that phase order; a gate for it is the first thing to write
  when it is picked up.

**Slice 3b, the GRAPH round trip: DONE** (`src/gtext.coil`, gated by `tests/gtext-test.coil`).
Printer and parser in one file because they are one format, the same reason 3a gives. One line per
LIVE node:

```
n0: Start <- : ctrl
n1: Stop <- n7 : ALL
n2: Const int=1 <- n0 : int=1
n4: If <- n0 n3 : [ctrl,ctrl]
n5: CProj.0 <- n4 : ctrl
n9: Phi dyn <- n7 n2 n8 : dyn
```

Aux is `.<dec>` for a projection slot and ` <exact type>` for Const/Arg/Cast/Phi, chosen by
`op-aux-kind`, a case over the NAMED op constants whose default is a hard error: a new op silently
inheriting "no aux" would print a Const whose value had vanished, which is the same failure
`v-check-arity`'s missing-entry panic exists for. The delimiters are the two bytes a type's grammar
cannot contain, `<` and `:`, so ` <-` and ` : ` terminate a field unambiguously even though a
type's text carries spaces, commas and brackets; the three fields are found by scanning for one
byte each, with no lookahead.

**IDENTITY IS EXACT UP TO A DENSE RENUMBERING OF LIVE NODES**, and that is a real weakening of the
gate's word "identity" rather than an implementation detail. `n<k>` is a PRINT INDEX, the node's
position in the listing, because the arena ids of DEAD nodes cannot be recreated (nothing recreates
a node in order to kill it again). Consistent with [D5](DECISIONS.md#d5-written-in-coil-node-and-type-handles-are-integer-ids),
which already says ids are dense arena indices and not identities, so no amendment was needed. It
is not a vacuous distinction: 6 of the 15 corpus fixtures have live nodes whose arena ids are not
their print indices (`02-fold-after` prints 4 lines out of a 10-node arena), so the renumbering is
exercised by the identity clause itself. A listing whose indices are not dense and ascending is
`GERR-INDEX-ORDER` and not a text the parser renumbers for you: a parser that quietly accepted `n7`
on line 3 would make every hand-written reduction filed against the optimiser ambiguous.

**THE PARSER PEEPHOLES NOTHING, AND THAT IS LAW 3 RATHER THAN A CHOICE.** When this was written, a
diamond built fully raw and handed to `iterate!` lost both arms; that is fixed (see the second
review above, and a fully raw diamond is now a gate), but the policy stands on the reasons below,
which have nothing to do with that bug. There is no `n-peephole`, no `iterate!` and no `g-analyze!`
in `src/gtext.coil`, and the gate does not iterate a parsed graph either. Everything the parser does instead is analysis-free:
types are RESTORED with `n-set-ty!` and checked with `g-verify`, which runs `compute` and never
`idealize`. Restoring is also the only exact choice, not merely the safe one: `cast-compute`'s
freeze rule and `phi-compute`'s widening fuel both read the node's PREVIOUS type, so re-analysing
from ANY is not guaranteed to reach the same fixpoint. Letting `g-verify`'s type pass then prove
`n-compute == n-ty` on every node is exact AND a stronger check, and it is what makes a text that
LIES about a type a named `VERR-STALE-TYPE` instead of a silently corrected one.

**Three passes, which is LAW 4 discharged rather than honoured.** A loop phi's back-edge value is a
node created later in the listing, so a one-pass parser needs a placeholder for a forward
reference, and any placeholder that is a plausible node id is LAW 2's forbidden stub in its purest
form. Pass 1 creates every node, pass 2 wires every input with `n-add-def!`, pass 3 restores every
type. Because nothing computes or rewrites during pass 2, the in-progress window and `n-if-arms!`
are not needed at all: there is no moment at which a half-wired node's type is consulted. That is a
STRONGER position than opening the window, and the file says so explicitly so that a later reader
does not "restore" the window and feel entitled to add a peephole inside it. A side effect worth
keeping: an IN-PROGRESS graph round-trips, `_` and all, which is exactly the reduced test case
someone debugging the in-progress contract will want to file.

**What the format does NOT carry**, listed because a reader will otherwise assume it does: `outs`
order, the GVN table, `deps`, `hash`, and `keeps` beyond the two pinned roots. The first four are
caches or unordered, and a parsed graph rebuilds `outs` as a by-product of wiring. Every current
reader of `outs` (`ev-ctrl-succ`, `ev-if-arm`, `ev-enter-merge!`, `v-check-edges`,
`region-has-phi?`) is order-insensitive, but that is an ARGUMENT and not a proof: if it is wrong,
the differential clause across the reparse is what reports it, and that would be a finding to write
down here rather than something to paper over. `keeps` is the one that is asserted instead of
dropped: `g-write` refuses to print a graph with a pin anywhere but the roots, because a live pin
means a construction window is still open and the caller is printing a half-built graph.

**What the gate proves.** The corpus round trip (text identity, live-count equality, `g-verify`
clean on every reparsed graph, with a counted floor on lines round-tripped); the differential
oracle across the reparse for 12 argument seeds per fixture; the field-sensitivity table; the
malformed-text table, one named code each plus a distinct-code floor; that `g-parse` aborts in a
forked child rather than handing back a partial graph; the "the parser optimises nothing" fixture;
and the structural loop check (region and phi arity equal, back edge in slot 2).

**What the gate deliberately does NOT assert.**

- **LAW 5 is not enforceable by the format, and no attempt is made to pretend otherwise.** A text
  whose Phi has one fewer input than its Region is lexically fine, and the printer emits the two
  arities independently, so a dropped input on one side round-trips happily. Duplicating
  `v-check-phi` inside the parser would give two implementations of one contract that could drift,
  so a short-phi text is instead a named case whose expected outcome is `VERR-PHI-ARITY`. The same
  split covers a Return with its inputs deleted, which parses cleanly and reports `VERR-ARITY`.
- **"Reparse, `iterate!` to a fixpoint, get the same text" is NOT gated**, and is the most
  attractive property that was left out. It is exactly the shape that trips the open
  `region-dead-path` item, so it waits for M7's phase structure rather than being half-attempted.
- **`GRT-DIFFERENT-TEXT` is provoked only by a hand-written text, never by a printed one.** Like
  `ty-injective?` returning false in 3a, the branch that says "the format is ambiguous" has no
  reachable witness among graphs the project builds, which is the property being gated rather than
  a hole in it.

**Why the identity clause is not self-fulfilling**, which is LAW 8 and is the whole reason this
slice is as large as it is. A print-then-parse-then-print identity would pass for a printer that
emitted only op names. Three separate defences: the field-sensitivity table deletes one printed
field at a time; the line-count floor pins how much text is actually being compared; and the
differential clause requires the reparsed graph to COMPUTE the same answers. Measured rather than
argued, by deleting each field from a real four-line graph and recording what notices: deleting the
print index reports `GERR-NO-NPREFIX`, the op name `GERR-UNKNOWN-OP`, a Const's aux type
`GERR-AUX-MISSING`, the computed type `GERR-NO-SEP` — and deleting the INPUT LIST is caught by
nothing in the parser at all, only by `g-verify` (`VERR-ARITY`), by text identity (the reparse
prints the shorter line), and by the interpreter. That last one is the honest measurement the table
exists to produce, and it is why the input list could not have been left to the parser to police.

**Truncation is a named panic and never a shorter answer.** A `FixBuf` clamps at its capacity and
the discarded `Result` hides it, so a text that did not fit would come back truncated — and a
truncated text compares EQUAL to another truncated text, which would make the identity clause pass
on nothing at all. `g-render` panics by name instead, the lesson `tests/text-test.coil` already
paid for once. Printing a type goes through the CHECKED `ty-print-exact`, so a type whose exact
form is not injective aborts by name rather than putting an ambiguous string into a graph line and
having the ambiguity attributed to this file.

**Twenty named failure codes, all twenty provoked and distinct**, including
`GERR-NO-NEWLINE` for a truncated text and `GERR-ROOT-DUP` for a second Start or Stop (without
which the parser would `n-new` a root that `g-start` does not name: a Start nothing anchors
constants to). The two type codes, `GERR-BAD-TYPE` and `GERR-BAD-AUX-TYPE`, leave `text-err`
readable so the type production that failed is still named; "the graph line is malformed" and "the
type on it is malformed" are two bugs in two files, and `g-parse` prints both.

**Still open, carried forward, not fixed here.**

- **The `int=[min..max]` versus `w0` wart in the TYPE format** (recorded under slice 3a above) is
  untouched. It is a decision, not a fix: either reject an explicitly written full interval or
  derive the widening default from the interval on both sides. Nothing is red, because the printer
  only ever emits the second spelling, so the graph round trip inherits an exact type form.
- **`region-dead-path` acting on a provisional `~ctrl`** (recorded above) is now formally assigned
  to **M7's phase structure**, which analyses to a fixpoint before transforming anything. Slice 3b
  did not need it resolved, because it neither peepholes nor iterates; the constraint it imposed
  was "do not build a diamond fully raw and then iterate it", and the parser satisfies that by
  never iterating at all.

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
