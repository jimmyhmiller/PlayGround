# aot-kit-gradual

An ahead-of-time compiler toolkit on a sea-of-nodes IR, aimed at dynamic languages and at
languages that layer types over a dynamic base (TypeScript). Written in
[Coil](../../coil). The goal is V8-class performance with no JIT.

Read in this order:

- **[docs/DECISIONS.md](docs/DECISIONS.md)** is law. Closed world with verified rather than
  trusted types, GC-abstract IR nodes, NaN boxing, guards as plain control flow.
- **[docs/DESIGN.md](docs/DESIGN.md)** is the architecture: the pipeline, the IR core, the type
  lattice, the GC contract, specialisation, the backend, the tooling.
- **[docs/ROADMAP.md](docs/ROADMAP.md)** is the plan, as milestones with executable gates.

## Layout

```
src/ty.coil          the type lattice        (M0, done)
tests/ty-test.coil   the M0 gate: lattice laws
reference/Simple/    upstream SeaOfNodes/Simple, gitignored, clone it yourself
```

## Running the gates

```sh
coil test tests/ty-test.coil
```

## The reference implementation

Cliff Click's [SeaOfNodes/Simple](https://github.com/SeaOfNodes/Simple) is the reference for
the parts of this that are solved problems: the peephole-and-worklist engine, the lattice
construction, and the optimistic interprocedural dataflow pass in its chapter 24 `Opto`, which
already carries an explicit whole-world assumption and builds its own call graph. Clone it if
you want to read along:

```sh
git clone --depth 1 https://github.com/SeaOfNodes/Simple reference/Simple
```

What this project adds is in [DESIGN.md §9](docs/DESIGN.md#9-deviations-from-simple-summarised):
a dynamic value lattice with first-class unions and shapes, annotations discharged by
whole-program inference, speculative specialisation with retained generic versions, and a GC
contract enforced by a verifier from the first commit.
