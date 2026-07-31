# aot-kit-gradual

An ahead-of-time compiler toolkit on a sea-of-nodes IR, aimed at dynamic languages and at languages
that layer types over a dynamic base (TypeScript). Written in [Coil](../../coil), a low-level Lisp.

The goal is V8-class performance **with no JIT**. What makes that plausible is a closed-world
assumption: it lets a TypeScript annotation be *proven* rather than trusted, which is what earns the
right to compile the check away. Where a value stays polymorphic, monomorphic specialisations are
emitted ahead of time behind guards, with a generic version retained as the fallback. There is no
deoptimisation machinery anywhere in the design.

## Getting oriented

```sh
tools/gate.sh          # THE gate: typecheck everything, run every suite, re-render the diagrams
tools/gate.sh --quick  # skip the diagram pipeline
```

Green means every milestone's gate still holds. **Nothing is marked done and nothing is committed
while it is red**, which is the contract the whole project runs on.

Read in this order:

1. **[docs/ROADMAP.md](docs/ROADMAP.md)** for where things are and what to do next. There is a
   status table at the top and a named next piece of work.
2. **[docs/DECISIONS.md](docs/DECISIONS.md)** is law. D1 to D7 were decided up front; D8 onward were
   learned by getting them wrong. **If you read only two things, read D8 and D9** (below).
3. **[docs/DESIGN.md](docs/DESIGN.md)** for the architecture: the pipeline, the lattice, memory, the
   GC contract, specialisation, the backend, the tooling.
4. **[docs/JOURNAL.md](docs/JOURNAL.md)** when you want to know *why* something looks the way it
   does, or before changing something that looks gratuitous.

## The module map

| File | What it is |
|---|---|
| `src/ty.coil` | The type lattice: interned, `meet`/`dual`/`join`/`isa`, the dynamic value axis, memory types |
| `src/node.coil` | The graph: nodes, edges, `compute`/`idealize`, peepholes, GVN, the worklist, control flow, dominators and the loop tree |
| `src/shape.coil` | Hidden classes as a transition tree, and the alias classes memory SSA is built on |
| `src/verify.coil` | The graph verifier, one named failure code per check |
| `src/eval.coil` | The IR interpreter, and therefore the **differential oracle**. Its heap is immutable and versioned, because a demand-driven walker cannot read a mutable one |
| `src/text.coil` | The exact textual form of a *type*, printed and parsed |
| `src/gtext.coil` | The textual form of a *graph*, printed and parsed |
| `src/corpus.coil` | The shared fixtures: the verifier, the diagrams and the differential tests all use these same graphs |
| `src/dot.coil` | Graphviz output |
| `tests/*-test.coil` | One suite per area; `tools/gate.sh` runs them all |
| `tools/dot-dump.coil`, `render-dot.sh`, `build-page.py` | The diagram pipeline and the gallery page |

There is no front end yet. Graphs are built through the node API, which is deliberate: the IR and its
tooling come first, and the `dyn` surface language arrives later (see
[D6](docs/DECISIONS.md#d6-the-first-driver-is-a-minimal-dynamic-core-language-in-s-expression-syntax)).

## The two things that will cost you a day

Both are in DECISIONS.md in full. Both fail in ways a completely green test suite cannot see, which
is why they are called out here.

**[D8](docs/DECISIONS.md#d8-a-rewrite-may-only-act-on-a-proven-type-and-proven-is-narrower-than-it-sounds):
a rewrite may only act on a PROVEN type.** `compute` is an optimistic analysis and may act on
whatever it currently holds. Any *irreversible* rewrite must first ask `n-ty-proven?`, which is
transitive over the node's whole input cone and is a fixpoint test. The insight underneath it:

> `ANY` is the ABSENCE of information. Every other high type is a CLAIM someone computed.

So "the type is exactly `~ctrl`" proves nothing, because `~ctrl` is merely the high element of the
control axis. Six rewrites once satisfied that test and were wrong. The cost, measured on a green
gate: one wrong constant on 55 of 200 worklist seeds, and on 31 of 200 the entire program deleted
with the verifier still clean.

**[D9](docs/DECISIONS.md#d9-construction-has-contracts-and-they-are-checked-where-they-can-be):
construction has contracts.** A merge under construction reports CONTROL and its phis report their
declared types; a loop body must be built *and peepholed* inside that window, with the phis closed
before the control back edge. An `If` is in progress until *all* of its projections exist, so use
`n-if-arms!` rather than peepholing them one at a time. And a region's path count and every phi's
value count are **one** invariant, changed together or not at all.

Get the loop window wrong and the loop is deleted with no error reported, every individual step
locally justified.

## Writing Coil

Run `coil guide` first. The gotchas that actually bite here:

- `if` requires both branches and they must have the same type. For effect-only, `(if c (do … 0) 0)`.
- `match` binds a struct payload **by value**, and `field` needs a place, so pass a matched struct to
  a function (which gives a read-only reference) rather than poking at it.
- `print-str` and `fmt` return `(Result i64 IoError)`. This project discards that through a named
  `ps` helper, so the discard is deliberate and greppable.
- `case` keys are **named constants**, never literal numbers. Literal keys are how a renumbering
  silently rewires every dispatch table at once.
- There is no struct literal: `(let [(mut v) (zeroed T)] (store! (field v f) x) (load v))`.
- No top-level mutable state: a singleton is `alloc-static` inside a zero-arg accessor.
- `coil check FILE` typechecks; `coil test FILE` runs its `deftest`s (exit 0 pass, 1 fail).

## Working on it

There is a workflow at `PlayGround/.claude/workflows/aotkit-milestone.js` that advances the roadmap
one slice at a time: scout a finishable slice, one serialized writer implements it, a *different*
agent writes the gate from the contract, a bounded repair loop drives it green, four adversarial
lenses review and their claims are refuted before anyone acts on them, then it records and commits.

The invariant that makes it safe to run unattended is that **only green work is ever committed**, so
the worst outcome of a bad slice is no progress rather than a broken tree.

Whether you use it or not, the habits that have actually caught bugs here:

- **Revert your fix and confirm the gate goes red.** If it stays green, the gate does not test what
  you think ([D11](docs/DECISIONS.md#d11-a-tool-is-only-a-tool-if-it-can-fail)).
- **Run the program, do not just inspect the graph.** Golden strings and the verifier both stayed
  green through a deleted arm, a dropped `Return` and a discharged type check
  ([D12](docs/DECISIONS.md#d12-the-oracle-has-to-run-the-program)).
- **Vary the worklist seed.** Several of the worst bugs appeared on a minority of seeds; one showed
  up on 15 of 200.
- **Construction order is part of a fixture.** A flat `if (0)` passed while the same construct one
  level deeper miscompiled.

## The reference implementation

Cliff Click's [SeaOfNodes/Simple](https://github.com/SeaOfNodes/Simple) is the reference for the
parts of this that are solved problems: the peephole-and-worklist engine, the lattice construction,
and the optimistic interprocedural dataflow pass in its chapter 24 `Opto`, which already carries an
explicit whole-world assumption and builds its own call graph.

```sh
git clone --depth 1 https://github.com/SeaOfNodes/Simple reference/Simple   # gitignored
```

What this project adds is summarised in
[DESIGN.md section 9](docs/DESIGN.md#9-deviations-from-simple-summarised): a dynamic value lattice
with first-class unions and shapes, annotations discharged by whole-program inference, speculative
specialisation with retained generic versions, and a GC contract enforced by a verifier from the
first commit.
