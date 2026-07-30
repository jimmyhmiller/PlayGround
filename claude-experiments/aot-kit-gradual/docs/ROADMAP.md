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

**Status: DONE.** `src/ty.coil`, gated by `tests/ty-test.coil`: 23 tests, laws checked
exhaustively over 28 hand-listed types plus all 28 of their duals (every pair for the binary
laws, every triple for associativity and `isa` transitivity).

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

`If`, `Region`, `Phi`, `Proj`, `CProj`, `Loop`, `Cast`, `XCtrl`, `Never`. Lazy phi creation.
Dead control flow elimination. Loop tree construction.

**Gate.** Golden tests covering Simple chapters 5 through 8 translated to `dyn`; unreachable
code provably removed; a `Phi` never survives with a dead input.

## M3. Tooling: textual IR, verifier, evaluator

The textual IR printer and parser, round-trip tested. Graphviz output. The graph verifier
(well-formedness, edge symmetry, type monotonicity). An **IR interpreter** over the ideal
graph, which becomes the differential oracle for everything after this point.

**Gate.** `print` then `parse` is the identity on the whole corpus; the verifier is green on
the corpus and *red* on a set of deliberately corrupted graphs (a verifier that never fails is
not a verifier); the interpreter agrees with expected results on the corpus, before and after
`Opto`.

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
