# Journal

The detailed record of how the compiler came to look the way it does: per-slice write-ups, the
adversarial reviews, the measurements behind each claim, and the arguments for choices that a
reader would otherwise assume were arbitrary.

**This is not the plan.** [ROADMAP.md](ROADMAP.md) is the plan and the current status;
[DESIGN.md](DESIGN.md) is the architecture; [DECISIONS.md](DECISIONS.md) is the law. This file
exists so those three can stay short while nothing is lost.

Read it when you want to know *why* something is the way it is, or before changing something that
looks gratuitous. Most of what is here was written immediately after a bug that a green test suite
could not see, and the reasoning is usually more valuable than the fix.

---

## M3, slice by slice

**Slice 1, the verifier: DONE** (`src/verify.coil`, `src/corpus.coil`, 19 tests). One named code
per check, which is what stops the corrupted-graph half from being vacuous: "verify returned
non-zero" is satisfied by a verifier that always fails, whereas "corrupting the phi/region lockstep
reports `VERR-PHI-ARITY`" is a claim about a specific check noticing. It runs `compute` and never
`idealize`, because unlike Simple's equivalent our `idealize` MUTATES (`region-remove-path!`
deletes edges, `stop-idealize` deletes an input), so a reducibility check would silently transform
the graph it was verifying.

**Slice 2, the IR interpreter: DONE** (`src/eval.coil`, `tests/eval-test.coil`, 23 tests; 12 when
the slice closed, and the two reviews put their differential witnesses here). The oracle is live:
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
`tests/text-test.coil`: 12 tests; the whole gate was 114 then and is 138 now). Printer and parser live in one
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


---

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


---

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
is not a vacuous distinction: 6 of the 18 corpus fixtures have live nodes whose arena ids are not
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

**What the gate proves, with the numbers it measures** (`tests/gtext-test.coil`, 7 tests, and every
number below is printed by the run rather than asserted in prose):

- the corpus round trip, text identity plus live-count equality plus `g-verify` clean on every
  reparsed graph: **207 node lines across 18 fixtures**, floor 130, fixture floor 15;
- the differential oracle across the reparse: **216 paired runs** (18 fixtures x 12 argument seeds)
  agreeing on both status and value, with each fixture's Arg-type signature compared across the
  renumbering first, because `ev-bind-args!` keys bindings by declared type and position;
- the field-sensitivity table: **10 one-field-apart pairs** that must render to different text, over
  **14 hand-shaped graphs**;
- the malformed-text table: 24 texts reaching **19 distinct named codes**, tied to
  `gtext-err-count`;
- that `g-parse` aborts (signal 6, observed in a forked child) rather than handing back a partial
  graph;
- the "the parser optimises nothing" fixture, kept by the parser and deleted by the eager builder in
  the same test;
- and the structural loop check: **9 reparsed Phis in lockstep with their merge**, back edge in
  slot 2, plus a short-phi text whose expected outcome is `VERR-PHI-ARITY`.

**A gallery specimen with no builder.** `19-text-only-add` in `tools/dot-dump.coil` is the first
fixture that arrives as TEXT: the `Add(x, 0)` graph the eager builder deletes on sight, parsed and
then drawn. Every fixture's listing now travels in its `.dot` file as a block comment written by
`g-render`, so the page shows the printer's own output beside the diagram for the same reason the
node counts come from the compiler: a listing typed into the page by hand is a claim nothing
re-checks. It also puts the 3a wart on screen, since the diagram's debug label reads
`int=[min..max]` where the exact listing reads `int w0` for the one type.

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
- **`GRT-DIFFERENT-TEXT` is never provoked, and neither is `GRT-PARSE-FAILED`.** Like
  `ty-injective?` returning false in 3a, the branch that says "the format is ambiguous" has no
  reachable witness among graphs the project builds, which is the property being gated rather than
  a hole in it. `g-round-trip-check` takes a GRAPH and not a text, so reaching either cause needs a
  graph whose print-parse-print differs, which IS the defect the check exists to report; only
  `GRT-OK` is ever observed.
- **"A malformed text must come back as the NO-GRAPH SENTINEL" is not assertable against this API,
  and was not faked.** `g-try-parse` returns a CODE, not a graph handle, and `src/gtext.coil` says
  outright that a failed parse leaves the graph in an unspecified partial state, because rolling
  back would be a second implementation of `graph-reset!`. Nor is "the graph after a failed parse is
  invalid" a true property: a text that fails on its third line can leave a perfectly legal
  Start+Stop graph behind. What is asserted instead is the guarantee a caller can actually rely on:
  no malformed text ever returns `GERR-NONE`, the code is the specific named first failure, and the
  strict entry point `g-parse` ABORTS, so a partial graph is never handed onward.
- **The differential clause has no witnessed catch of its own.** Every wiring defect that was
  injected while writing the gate is also visible to text identity, because every input is printed
  in slot order. Its unique territory is the part of the format that carries nothing (`outs` order,
  `deps`, `hash`, the GVN table), where an order-sensitive reader would only be observable by
  RUNNING the program. So the order-insensitivity of the five current `outs` readers stays an
  argument, and this clause is the measurement standing behind the argument rather than a catch.
- **The input-ORDER pair catches order-INSENSITIVITY, not a systematic REVERSAL.** A printer that
  sorted its inputs is what it was injected against and what it finds. Reversing every input list
  instead is caught by the identity clause, because reprinting an already-reversed parse yields a
  different text. Written down because the pair's name could be read as the stronger claim.
- **`GERR-COUNT`'s own self-check is exercised only in the passing direction.** `gtext-err-count`
  panics when the code table and the count disagree, and making that fire needs a source edit rather
  than an input, the same shape as `ty-injective?` in 3a. What the gate does instead is tie the
  distinct-code count to `gtext-err-count`, which is what turns a newly added code that nothing
  provokes into a red test.
- **Six printer-side panics are unreachable from any text or corpus graph**, and are gated in
  neither direction: `op-aux-kind`'s and `gaux-write`'s hard-error defaults (a new op), `g-write`'s
  "fewer than two live nodes" and "Start/Stop are not the first two live nodes", `g-render`'s
  truncation panic, and `gtext-check-pins`'s open-window panic. Each needs either a new op or a
  deliberately corrupted graph handed straight to the printer, which is precisely the input they
  exist to refuse. They stay because LAW 2 wants the refusal to exist before the op does.

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

**Nineteen named failure codes plus `GERR-NONE`, and all nineteen are provoked and distinct** (the
table is 24 malformed texts reaching 19 codes; `GERR-NONE` is the one a malformed text never
reaches, and it is asserted separately on the two-line minimum graph), including
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


---

## M4, in progress

**Slice 4a, the shape table and the memory type in the lattice: the lattice and table half is
written** (`src/shape.coil` new; `src/ty.coil`, `src/text.coil` extended; `src/node.coil` and
`src/eval.coil` gained the arms the new `Ty` variant forces). Gated by `tests/shape-test.coil`
plus additions to `tests/ty-test.coil` and `tests/text-test.coil`.

- `src/shape.coil`: shape 0 is the field-less root; `shape-transition s name ty` is memoised on
  `(s, name)`; the alias is allocated at the edge that introduces the field and inherited by every
  descendant, which is what makes a store through `{x}` and a load through `{x,y}` name the same
  word; offsets are `8 * (SHAPE-HEADER-WORDS + i)`; `shape-reset!` mirrors `graph-reset!`.
- `src/ty.coil`: `TMem(aliases, contents)` with the alias axis as a **bitset**, not Simple's
  collapse (see DESIGN.md §4 for the counterexample and why the samples are part of the same
  change); `SHAPE-MAX`/`ALIAS-MAX` as named bounds with named aborts; `ty-iwide`/`ty-with-iwide`/
  `ty-widen`/`ty-widen-from` now reach a memory type's contents; `ty-sample` grew 32 → 41.
- `src/text.coil`: `mem#<bits> <contents>`, printed and parsed, with `TERR-MEM-NO-HASH`,
  `TERR-MEM-NO-SPACE` and `TERR-MEM-CONTENTS`. The form contains neither `:` nor `<` because
  `gtext`'s line grammar scans for both.

**What 4a deliberately does NOT assert**, said plainly because "it typechecks and the lattice laws
pass" is available almost for free here and would prove close to nothing:

- **There is no memory OP.** `op-memory?` is still legitimately `false` for every op, which is the
  precedent this roadmap already set for it. Nothing produces or consumes a `TMem`, so no `compute`
  is exercised on one and `ty-high?`'s answer for a partly-empty alias set is untested against
  monotonicity. 4b is where that becomes a claim.
- **The reserve-then-fill discipline on the field window is prophylactic.** Nothing in 4a can
  re-enter the shape table while a window is open, so replacing it with push-as-you-go would not go
  red. What IS gated is the consequence that bit the type table: a repeat construction must not grow
  the side array, and a window with a hole panics by name.
- **`ALIAS-MAX` cannot fire in 4a.** A transition is the only alias allocator and the transition
  graph is a tree, so aliases = shapes − 1 and `SHAPE-MAX` reports one edge earlier. The tree
  relation is asserted (`shape-alias-invariant-ok?`) instead of the bound being claimed reachable,
  and the 65th-bit aborts that ARE provokable are `t-obj-shape 64` and `t-mem-alias 64`.
- **Laws 3, 4 and 5 barely apply**, and where they bite next is worth recording now rather than
  rediscovering. 4a adds no rewrite, no merge and no region/phi pairing. **Law 5 acquires a second
  instance in 4b**: a memory Phi's arity is locked to its region exactly as a value Phi's is, so
  `region-remove-path!` must drop memory-phi inputs in the same operation or a memory phi reads the
  wrong arm — a miscompile that typechecks. **Law 3 acquires new consumers in 4c**: load-after-store
  forwarding, store-after-store elimination and dead-allocation removal are all irreversible and
  must each ask `n-ty-proven?`. Simple's `Load.idealize` decides aliasing from pointer identity and
  from `neverAlias` on two distinct `New`s, both structural and safe, but its offset-overlap test
  reads a TYPE and is exactly the shape Law 3 forbids acting on provisionally.


---

**Slice 4b, the four memory ops: `New`, `Load`, `Store` and `MemMerge` exist, verify, print,
reparse and RUN** (`src/node.coil`, `src/eval.coil`, `src/verify.coil`, `src/gtext.coil`,
`src/corpus.coil`, `src/dot.coil` extended; `src/shape.coil` and `src/ty.coil` gained the readers
the graph needs). Gated by `tests/mem-test.coil`, 15 cases, plus four new corpus fixtures that put
every earlier milestone's corpus-wide gate over a program with a heap in it.

What was decided, as opposed to implemented:

- **Only `New` carries control.** A memory op is ordered by its memory edge alone, which is the
  property that makes 4c's forwarding ordinary dataflow rather than a pass. An allocation is the
  exception, because two executions of it are two objects.
- **`op-gvn?` excludes `New`,** for the same reason it excludes `Arg`. Two structurally identical
  allocations are two objects, and the graph that results from merging them is internally
  consistent, verifies clean, and has quietly lost an object. The witness is a program that
  compares two allocations and must answer false.
- **A Store meets the incoming contents rather than replacing it**, so a load reports the union of
  everything its alias class holds: `undefined|int`, not `int=5`. That number is pinned exactly, so
  4c's structural forwarding is measured against what 4b earned rather than against a hope. The
  alternative (contents = the last value stored) is precise for one pointer and a lie about the
  class, and it would let a load off a different object fold to a constant it never held.
- **`Return` gained an optional memory input.** Without it the kill cascade collects the whole heap
  as unused: nothing reads the last store, so 4c's store elimination would have been gated against
  a program whose stores were already gone for unrelated reasons.
- **The interpreter's heap is IMMUTABLE AND VERSIONED.** This is the one decision a reader is
  likely to want to argue with, and the argument is short: `ev-data` evaluates on demand, so a
  mutable heap would answer from whatever state the walk had reached, and the same graph would mean
  different things depending on which use asked first. Memory SSA already says a memory state is a
  value; the oracle takes it literally. A version records which alias classes it describes, every
  read checks that mask, and a `MemMerge`'s children must be disjoint — which is what makes running
  a memory graph evidence rather than exercise.
- **An allocation is executed when control ARRIVES, not when its value is demanded.** A data node is
  evaluated once per use, and an object's identity may not be computed twice; nor may it be cached
  forever, because a `New` in a loop must give a fresh object every trip. Both are right at once if
  identity is established when control enters the block, which is the mechanism and the environment
  a `Phi` already uses.

**Two gaps found by reviewing the slice against itself, both closed in it:**

1. **No loop had a memory phi.** 4a added widening through `TMem` on the strength of an argument
   about a bug nobody could yet provoke: a memory phi on a back edge meets a fresh memory type every
   trip, and with the widening counter invisible inside the memory type the contents creeps by one
   step per round for ever. The symptom is a hang with nothing reported. Fixture 22 is the witness
   (`o = {x}; while (i < 3) { o.x = i; i = i + 1 }`), it terminates, and the store is in the body
   purely by being on the memory back edge.
2. **Monotonicity of the four new computes was claimed in prose only.** `the_memory_computes_only_
   ever_fall` observes the handful of types one raw fixture visits; the law is about the whole
   lattice. `every_memory_compute_is_monotone_in_every_slot` now drives each compute over all 82
   samples and duals in every input slot with the other slots swept too, 356,454 checks. Reverting
   `ty-mem-contents`'s ANY arm to ALL makes it red with a printed witness (`Load: input fell from
   ANY to mem#0 ANY, result rose from ALL to ANY`), and independently trips `g-analyze!`'s falling
   check on the raw fixture.

**What 4b deliberately does NOT assert**, said plainly for the same reason 4a's list exists:

- **No memory rewrite fires.** There is no load forwarding, no store elimination and no
  escape-based dead-store removal, so none of Law 3's new consumers exists yet and the
  `n-ty-proven?` obligation on them is still only written down. Dead-ALLOCATION removal is partly
  present and was free: killing a `New`'s last projection drops its last use and the cascade
  collects it.
- **The shape-polymorphic clause is met for the lattice, not for code.** The merged pointer keeps
  both shape bits, the two arms keep distinct alias classes, and the `MemMerge` describes their
  union. "Two inline paths" needs a guard to make a path, which is M9.
- **A `Load`'s slot 0 is unconstrained and always null.** Nothing pins a load to a block yet; when
  a load has to be pinned to the block that proved its shape, that slot is where it goes.
- **The shape table is not part of the graph text.** A printed graph is meaningful only against the
  table that was live when it was printed. The parser refuses an id the current table does not
  define (`GERR-BAD-SHAPE`, `GERR-BAD-ALIAS`) so that mismatch is a named failure on the first line
  that mentions it, but serialising the table is a whole-program question and belongs with whichever
  milestone first has to ship one.
- **Nothing escapes.** There are no calls, no globals and no arrays, so every object is local to the
  program and `EV-MEM` on a load of a word this run never allocated is the correct answer rather
  than a limitation. M6 changes that, and the memory a function receives is already spelled: an
  `Arg` of memory type, which is how the initial heap enters today.

## M4 slice 4c-1: load-after-store forwarding, by pointer identity

`load-idealize` in `src/node.coil`, wired into `n-idealize` under `OP-LOAD`. If the load's memory
input is a live `Store` naming the SAME alias class whose pointer is the SAME NODE, the load is
that store's value. Nothing else forwards: a `Phi`, a `MemMerge`, a `Proj` of a `New`, a store on
another class, a store on another pointer all decline and record a dep on the memory node.

**Why it reads no type at all, and why it asks `n-ty-proven?` anyway.** Every other guarded
rewrite in `node.coil` asks for the proven answer because it read a type and a provisional type is
a lie. This one reads node identity and an alias number, and neither can be provisional. The guard
is there for a different reason and it is worth keeping the reason attached: the rewrite is
irreversible and the graph is not finished, so a store sitting in a still-settling cone is a store
the sweep may yet replace, and taking its value commits to its identity permanently. The fixpoint
of the load's input cone is the available proxy for "this part of the graph has stopped moving",
and being transitive it covers the store, its pointer and its value at once. It also keeps the
substitution a type NARROWING rather than a widening: `load-compute` reports `t-top` while the
pointer is high, and swapping in the stored value under that would widen the type at every use.

**Simple's offset-overlap rule is deliberately not ported.** Chapter 24's `LoadNode.idealize` also
forwards past a store whose offsets provably cannot overlap, which asks whether two types are
disjoint and then rewrites: the exact shape D8 forbids acting on provisionally. Its
distinct-allocation rule is structural and would be legitimate, but needs a predicate that does not
exist yet; its push-a-load-through-a-Phi rule builds a new Phi on the merge's region and lands in
LAW 5's arity lockstep. Both stay out of this slice.

**Two things measured rather than anticipated:**

1. **The forwarded load takes the whole heap with it if memory is not rooted first.** `fx-object`
   built the load as an argument to `ret-to-mem`, so the load forwarded, subsuming it dropped the
   `Store`'s only use, and the kill cascade collected the store, both projections and the `New`
   before the `Return` that was supposed to root them existed. `ret-to-mem` then wired a corpse.
   Nothing at the construction site says a word; the corpus-wide verifier reports `VERR-DEAD-INPUT`
   much later. The fix is `n-keep!` on the memory value across the load's construction, the same
   mechanism `n-peephole` uses to stop a cascade eating the replacement it is installing, and it
   works because `n-del-use!` reports a pinned node as still used. This is D9's still-open "a dead
   node can be wired as an input" clause showing up exactly where the roadmap said it would.
   It also hit two pre-existing tests in `tests/mem-test.coil` that build a load and then keep
   using its memory (`an_alias_is_the_same_word_through_a_longer_shape`, and
   `two_loads_of_one_word_are_one_load`, where the SECOND load is built against the corpse of the
   store the first one killed).
2. **"The load disappeared" is a vacuous gate.** A rule that deletes every load satisfies it. The
   weight is carried by `fx-object-two` (`a = {x}; a.x = 1; b = {x}; b.x = 2; return a.x;`): two
   allocations of one shape, so both stores name the same alias class and both pointers carry the
   identical type `obj@2`. Everything a type could look at says the two stores are interchangeable,
   and the only thing between the right answer and 2 is that the store's pointer is a different
   NODE. The graph that returns 2 is well formed, at a type fixpoint, clean under `g-verify`,
   survives print and reparse, and has every structural count unchanged. Only the interpreter tells
   them apart, which is what D12's oracle exists for.

**What 4c-1 deliberately does NOT do.** It does not eliminate the store it forwarded past: a read
being redundant says nothing about whether the write is observable, and `fx-object` keeps its
`Store` and its `New` rooted at the `Return`'s memory slot. Store-after-store elimination and
dead-store removal are 4c-2.

## The M4 adversarial review: eight findings, and what they had in common

Eight findings survived an attempt to refute them. Six of the eight are the same defect wearing
different clothes, and it is worth naming the pattern before the individual fixes: **a check written
on one path, with a second path into the same place that nobody checked.** The gate was green
throughout, and stayed green through the first draft of every fix, which is why each one below ends
with the measurement that made it red.

### The fold is a second door into forwarding, and it had no lock on it

`n-peephole` runs its constant fold BEFORE it calls `n-idealize`. `op-foldable?` is `>= OP-PROJ` and
`OP-LOAD` is 23, so whenever a Load's type is a constant the node becomes a `Const` and disappears,
and `load-idealize`'s alias comparison never runs. Clause 5 of `tests/mem-test.coil` was written
specifically to forbid laundering a miswired memory edge into a plausible value, and it measured the
idealize path only.

It survived by an accident of arithmetic. Its store sits on the `New`'s projection, so `new-compute`
has met `undefined` into the class's contents and the load is typed `undefined|int=5`, which is not
constant. Move the store onto the class's INCOMING state and the contents is a bare `int=5`: one
verifier-clean graph, `EV-MEM` before `iterate!` and `5` after it, produced by nothing but the fold.
The shape half needs no contrivance at all, because every freshly allocated field holds `undefined`,
which IS a constant, so an unguarded `o.y` on allocation-derived memory folds to `undefined` and
reports success.

**The fix is in `load-compute`, not in the fold**, and that distinction is the finding. Blocking the
fold for `OP-LOAD` alone leaves the bogus type in the graph, where an `Add` above the load folds to
`int=6` instead and the load is collected anyway. So the compute reports **ANY** when the memory
state's alias bitset does not carry the node's class, and **ALL** when some shape the pointer can be
does not carry the word.

ANY on one and ALL on the other is not a stylistic choice, and getting it backwards is caught by the
project's own monotonicity sweep. An alias bitset only GROWS as a memory type falls, so "the class
is absent" can only become "present"; a low answer there would have to RISE to the contents at that
moment, and `every_memory_compute_is_monotone_in_every_slot` says so by name. A shape SET also only
grows, so "some shape misses this word" can only become more true, and ALL is stable. The predicate
has to be "some shape misses it" rather than "no shape has it" for exactly that reason.

### A rewrite must not consume an access the program would refuse

The store-side twin of clause 5, and nothing gated it. Build a `Store` whose own access is bogus (an
object of shape `{y}`, a store naming class x) with a perfectly formed `Load` on top of it, and root
the memory somewhere else so the load is the store's only reader. The load forwards correctly, that
drops the store's last reader, the kill cascade collects the store, and the `EV-SHAPE` that store
owed the program vanishes with it. Same program, two builds, `EV-SHAPE` against `5`.

`load-idealize` now asks `access-refused?` of the store it is about to consume. Reading a type in
order to DECLINE is always sound, which is why this may look at an answer that an irreversible
rewrite may not (LAW 3). The negative control is the same graph shape with a sound access and the
same unrooted store: forwarding must still fire there, or the refusal would just be "declines
whenever the store is unrooted", which would kill every legitimate case 4c-3 wants.

### `t-mem` did not enforce the restriction three other comments said it enforced

`ty-xdepth` answers 0 for a memory type BECAUSE its contents must be a value; `shape-transition`
cites `t-mem` as the precedent for its own bounds check. `t-mem` checked nothing. Since `op-value?`
is `>= OP-CONST` and `Store` is 24, a memory state parked in another `Store`'s value slot passed
`v-need-value`, `store-compute` met it into the contents, and the result was `mem#1 mem#1 int`:
nesting without bound, `ty-depth` reporting 0 at every level so no depth guard applied, and
`ty-print-exact` aborting the process on a graph `g-verify` called clean.

Three changes, and all three are needed. `t-mem` panics by name. `store-compute` clamps its value
slot through `ty-as-mem-value`, because the monotonicity sweep deliberately feeds every lattice
element including memory types through that slot and a panic there would kill the sweep rather than
report anything. And `v-need-value` gained the mirror of `VERR-MEM-SLOT`: `VERR-VALUE-IS-MEM`.

Adding that mirror immediately found a second thing. `v-mem-producer?` said "a `Proj` of a `New`, at
any slot", which was harmless while the predicate had one direction and reports every store's own
POINTER as a memory state the moment it has two. Slot 0 is the object. The comment that said the
arity check would catch it was wrong; `v-check-proj` only checks the range.

### R1 was not machine-checked, and D2 said it was

Nothing required a `Load`'s or `Store`'s pointer to be a managed reference. `v-check-slots` asks only
`v-need-value`, and `load-compute` consulted the pointer only through `ty-high?`. So
`Load(mem, Const int=8)` and `Load(mem, Add(o, 8))` built, verified clean, round-tripped and ran. D2
claims both GC rules are machine-checked on every phase; that was true of R2 and not of R1, and it
held only because nothing in the IR was yet NAMED an address.

`VERR-PTR-SLOT`, in `v-pass-types`, which runs last and only on an otherwise clean graph. It has to
be a type check: unlike a memory state, a reference is an ordinary value and an `Arg`, a `Phi`, a
`Cast` and another `Load` (`o.y.x`) are all legitimate producers, so there is no finite producer set
to whitelist. The rule is "an object and NOTHING ELSE": `arith-compute` types `o + 8` as plain `dyn`
and `dyn` includes `VK-OBJ`, so "could be an object" accepts precisely the address arithmetic R1
exists to exclude. That puts an obligation on M5, and D13 records it rather than leaving it to be
rediscovered.

### The oracle could not see an object

`rt-eq?` compares an object by its index in the run's allocation stream. Inside one run that IS
`===`. Across two it is wrong in both directions. Removing an allocation nothing reads shifts every
later index, so the oracle reports `DIFFERENTIAL FAILURE` on 4c-3's dead-allocation removal, a
transformation `ev-enter-ctrl!`'s own comment calls unobservable. And two objects of DIFFERENT
SHAPES that land at the same index compare EQUAL, so the oracle also could not see a rewrite that
returned the wrong one. Nothing was red only because every `diff-pair` fixture returned an integer
and the one fixture that returns an object had no twin.

`ev-render-outcome` renders the reachable heap under the result, objects numbered by DISCOVERY
ORDER, shapes named by their FIELD NAMES. Discovery order is what keeps sharing observable
(`{a: o, b: o}` differs from `{a: o, b: o2}`) while an unreachable allocation is invisible: that is
isomorphism of the two heaps, which is identity modulo renaming. Field names rather than the shape
id for the same reason as the object index: `{x}` and `{y}` are both shape 1 in their own runs, and
the first draft of this renderer printed `s1` for both and could not tell them apart.

It renders to TEXT because it has to outlive `ev-reset!`. The second build cannot run until the
first run's heap is cleared, so a comparison holding two live heaps would need the interpreter to
keep both, which is the one thing `ev-reset!`'s comment is right to forbid.

### No else arm in the corpus had ever executed

`ev-value-in` answered `1 + seed%97` for an unbounded int, with a comment saying the bias existed
"so truthiness is exercised rather than every branch always going the same way". For a `dyn`
argument it guaranteed the opposite: 5001 seeds, 5001 truthy, 0 falsey. Every branch predicated on a
`dyn` `Arg` took its true arm for ever, which is eight corpus fixtures including the one M4 built to
have two arms.

That alone was not what hid the LAW 5 miscompile, and the second half is the more useful lesson.
With the binder fixed so one seed in three is falsey, swapping `fx-shape-poly`'s class-x memory
phi's arms STILL left the gate green, because the clause asserted `EV-OK` and PRINTED the result. A
printed object is a name. Nothing read the merged memory back, so no arm order was observable. It
went red only once `ev-outcome` read the returned object's field out of the state the `MemMerge`
produced, and the clause asserted the whole outcome under one truthy and one falsey seed. Two
independent gaps, and closing either one alone leaves the miscompile invisible.

### What was measured

Every fix above was reverted in turn and the corresponding gate confirmed red: eight fixes, eight
red gates, then green again at 185. The corpus grew by the pair `24-object-returned` and
`25-object-returned-scratch`, which is the same program with and without an allocation nothing reads
back and is 4c-3's gate, written before the pass it gates.

### What was deliberately NOT done

Two of the eight findings suggested making "the memory edge carries this class" and "the pointer's
shape carries this word" into verifier rules. They are not, and D13 records why rather than leaving
it as an omission. Both a memory type's alias bitset and a value type's shape set are unions over
PATHS, so carrying a class statically does not mean carrying it on the path taken; and a shape rule
strong enough to be worth having would make `EV-SHAPE` unreachable from any verifier-clean graph,
which would delete the project's own argument for having an oracle at all
(`running_it_catches_the_memory_contracts_the_verifier_cannot`). What is closed is the laundering,
which is the part that made one program give two answers. Whether M5 wants the static forms as well
is in ROADMAP.md's open list, with the `dyn` pointer question that has to be settled first.
