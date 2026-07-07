# microlang docs

Start with the top-level [README](../README.md) for the overview and the
runnable examples. These go deeper:

| Doc | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The pipeline, the two trait axes, the neutral `Ir`, the five execution engines, and where languages live. Read this first. |
| [CONTINUATIONS.md](CONTINUATIONS.md) | The continuation gradient (escape → full → delimited), why it is a representation choice not an interpreter/compiler one, the CEK machine, and how captured continuations survive a moving GC. |
| [SCHEME.md](SCHEME.md) | The R7RS-flavored Scheme frontend: pipeline, the 61/61 oracle-checked conformance suite, hygienic macros, the numeric tower, and honest edges. |
| [CODEGEN_AXES.md](CODEGEN_AXES.md) | The catalog of things a native code generator must let a *strategy* control, so swapping a strategy stays free. The design target the emit tier realizes. |
| [LIBRARY_LANGUAGE_SPLIT.md](LIBRARY_LANGUAGE_SPLIT.md) | The charter for building a language on the reusable core without the core bending to it. The boundary is discovered, not designed. |

## The one-paragraph version

One small, meaning-free `Ir` sits between two orthogonal trait axes — **value
representation** (`ValueModel`) and **execution strategy** (`CodeSpace`) — with a
moving GC underneath. Five execution engines (a tree-walking interpreter, a
closure compiler, a bytecode emitter, a stackless CEK machine, and a tracing
wrapper) all run the same `Ir`; any combination of value model × engine ×
dispatch × speculation is a valid program that computes the same answer (the
`matrix` example: 45 combinations, one answer each). A whole R7RS-flavored Scheme
— hygienic macros, arbitrary-precision integers, and multi-shot delimited
continuations that survive garbage collection — rides on top as a library that
touches only the public API.
