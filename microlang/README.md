# microlang

A dynamic-language toolkit re-cut at the axes a monolithic interpreter conflates:
**value representation** and **execution strategy** are orthogonal traits, one
axis-neutral IR sits between them, and a moving GC runs underneath. Several real
micro-languages exercise it — including an R7RS-flavored **Scheme** with hygienic
macros, a numeric tower, and multi-shot **delimited continuations that survive
garbage collection**.

The thesis: you should not have to choose your value layout, your executor
(interpreter vs compiler), your dispatch strategy, and your GC as one welded
bundle. Each is a plug-point; any combination is a valid program that computes
the same answer; and a whole language can ride on top as a *library*, touching
only the public API. The `matrix` example proves the orthogonality (45
combinations, one answer each); the Scheme frontend proves the library claim
(61/61 conformance, oracle-checked against Chicken Scheme).

## The two axes as traits

| Axis | Trait(s) | File | Carved at |
|---|---|---|---|
| Value model | `Repr`, `ValueModel` | `model.rs` | **immediacy** (`is_immediate(Cat)`), not float-vs-tagged |
| Execution | `CodeSpace` | `code.rs`, `compiled.rs`, `bytecode.rs`, `cek.rs` | meaning (`Ir`) vs strategy; re-entrant `invoke` |

**Five execution engines, one `Ir`, one `CodeSpace` contract.** Interpretation
vs compilation is itself a point on the axis, not a fork in the design:

| Engine | File | Kind | Notable |
|---|---|---|---|
| `TreeWalk` | `code.rs` | recursive interpreter | re-dispatches the `Ir` each run; escape continuations via unwind |
| `ClosureComp` | `compiled.rs` | closure compiler | compiles each `Ir` subtree to a Rust closure once, caches bodies |
| `BytecodeVm` | `bytecode.rs` | emit tier | value model emits model-specific bytecode (`ModelEmit`) |
| `CekMachine` | `cek.rs` | stackless abstract machine | full/delimited **multi-shot continuations**, GC-survivable |
| `Traced` | `code.rs` | wrapper | counts invocations (feeds speculation) |

`tiers_agree` pins that the general tiers produce identical results on every
program. `ClosureComp` proves the contract's two hard promises: compile-once
(`compiled_bodies() == 1` for a 6-deep recursion) and late binding (a compiled fn
calls one defined later). `CekMachine` proves the deepest one: a *fundamentally
different* evaluation strategy (explicit heap continuation, not the host stack)
slots in as just another `CodeSpace` over the same `Ir`.

`value.rs` holds the neutral vocabulary (`Cat`, `Val`, `Obj`) with `Int` a peer
of `Float`. `runtime.rs` ties it together: reader (code is data), `encode`/
`decode` (the boxing seam, `allocs`-counted), `macroexpand` (re-enters compiled
code), `analyze` (`Val` -> `Ir`), and the value-model-aware primitives.

## Run it

```
cargo run --example calc        # value axis: same engine, two models, measured allocs
cargo run --example microlisp   # eval axis: defmacro, recursion, macro-time re-entrancy
cargo run --example backends    # execution axis: interpreter vs compiler, same Ir
cargo run --example gc          # the fusion point: rooting forms across a macro GC
cargo run --example dispatch    # dispatch axis: megamorphic vs mono/poly inline cache
cargo run --example speculation # speculation/deopt axis: never/always/blacklist policies
cargo run --example bytecode    # THE EMIT TIER: value model emits model-specific bytecode
cargo run --example matrix      # PROOF: all 45 axis combinations run and agree
cargo run --example standard_ir # one axis-neutral IR, run by interpreter AND JIT
cargo test                      # locks every axis + the grand matrix + Scheme conformance
cargo run --bin scheme -p scheme    # the R7RS-flavored Scheme, running on the tiers
```

See [`docs/`](docs/) for the deep dives: [architecture](docs/ARCHITECTURE.md),
[continuations](docs/CONTINUATIONS.md), [the Scheme frontend](docs/SCHEME.md),
[the codegen axes](docs/CODEGEN_AXES.md), and [the library/language
split](docs/LIBRARY_LANGUAGE_SPLIT.md).

## One standard IR for interpretation and JIT

`Ir` is the standard, **axis-neutral** IR: it says `Prim(Add)`, `Call`,
`Global`, `Dispatch{site,method}` — semantic operations, not `AddRawLowBit` or a
resolved pointer. A frontend lowers to it once, and every execution strategy
consumes the same `Ir`: an interpreter (`TreeWalk`) and two compilers
(`ClosureComp`, `BytecodeVm`). `CodeSpace` is the formal name for "a way to
execute the IR," so interpreter and JIT are simply two `CodeSpace`s over it (the
`standard_ir` example hands the *same* `Ir` value to both).

The neutrality is load-bearing: because the IR commits to no value layout, no
dispatch, and no execution strategy — those are applied by the executor through
`ModelEmit`/`Dispatch`/the collector — one IR serves every axis combination. A
lower, per-configuration IR (the bytecode) sits *below* it as the shared
interpreter+JIT input for a given configuration; the axis-neutral standard IR
sits *above* the axis choices.

## Orthogonality: the grand matrix

The axes are independent, so their product space is all valid. The `matrix`
example and the `grand_matrix_all_combinations_agree` test run the full cross
product on two programs — arithmetic+recursion across {3 value representations} ×
{3 execution tiers including the bytecode emit tier}, and records+methods+
dispatch+a mid-program moving GC across {3 representations} × {2 general tiers} ×
{6 dispatch strategies} — a moving collector underneath all of it. **45
combinations, one answer each.** Any value layout, any execution strategy, any
dispatch/speculation strategy, freely combined, is a valid program that computes
the same result. That is the property the original toolkit never had.

`calc` prints integers free on `LowBit` / boxed on `NanBox`, and the exact
inverse for floats: the whole "value layout is a free choice" claim, made real.
`microlisp`'s `add2` macro calls the compiled `inc` during its own expansion.

## Two real models, ~30 lines apart

`LowBitModel` and `NanBoxModel` differ in one method (`is_immediate`) plus the
bit twiddling. Both drive the same generic `arith`. That is the discipline that
keeps the axis honest: a second real, tested consumer, not a fork.

## Design-tension #1: resolved (backends compose)

Originally `Runtime<M, C: CodeSpace<M>>` baked the backend into the runtime's
type. That swapped cleanly but did not *compose*: a wrapping backend
(`Traced<Inner>`) failed to typecheck, because `Inner::eval_ir` wanted
`Runtime<M, Inner>` while handed `Runtime<M, Traced<Inner>>`. Two changes fixed
it, and both were load-bearing:

1. **Backend is a value, not a type parameter.** `Runtime<M>` carries no
   backend; `eval_ir`/`invoke` take `&self` and you pass the code space in.
   `Traced` now holds a `Box<dyn CodeSpace<M>>` and delegates.

2. **Open recursion.** Passing the backend as a value is not enough: if a
   backend recurses through `self`, a wrapper only observes the calls the
   *runtime* initiates — the inner backend's own nested calls bypass it. So
   every method also takes `top`, the outermost backend, and recurses through
   `top`. Now `Traced` observes EVERY call (the test pins `invoke_count() == 5`
   for `(fact 5)`; a `self`-recursing wrapper would see `1`).

The lesson generalizes: "swappable" is cheap; "composable" needs the fixpoint.
Stopping at #1 would have left the exact "almost composes" seam that rots into a
hardcode — which is the failure mode this whole exercise is about.

## Slot resolution: the Ir/env cut, validated

Lexical variables resolve to `(up, idx)` slots at analyze time (`Ir::Local`), so
the runtime never searches a frame by name — `frame_get` is a pointer walk plus
an index. Globals stay `Ir::Global(Sym)`, resolved through the Var table at call
time (late binding preserved). Resolution lives in `analyze`, so both tiers get
it for free and share one frame-layout definition (`Runtime::build_call_frame`).

Doing this was the real test of whether the `Ir`/env split was cut in the right
place. Two findings:

- **The cut held.** `analyze` grew a compile-time scope stack; the runtime frame
  became `Vec<u64>`; both backends changed only their `Local`/`Let`/`Lambda`
  arms. No change to the value model, the `CodeSpace` contract, or the macro
  pipeline. The seam absorbed it.
- **It caught a real bug** (the point of the stress tests). `analyze` counts a
  `let` as one pushed frame, but the runtime initially ran the *first* binding's
  init under the parent env directly — off by one level, so an init referencing
  an outer variable walked past the root frame and panicked. The
  `slot_resolution_*` tests (deep capture, shadowing, on both tiers) surfaced it;
  the fix is that the first init also runs under an (empty) let frame. This is
  exactly the class of bug name-based lookup silently tolerates and slot
  addressing makes loud — which is a point in favor of resolving early.

## Moving GC + the handle discipline: where the two axes fuse

`gc.rs` is a **moving** (semi-space copying) collector with a shadow-stack root
set. Reachable objects are copied to fresh addresses; old slots become
`Obj::Moved` forwarding markers; every root is rewritten to the new address:
globals, the **constant pool**, the shadow stack, and the live environment.

Three design choices make this work, and each is a real finding:

- **Literals are constant-pool indices, not embedded pointers.** `Ir::Const` /
  `Quote` hold a `ConstId` into `Runtime::consts` (a GC root), so `Ir` carries no
  heap pointer a moving collector could not rewrite inside an immutable
  `Rc<Ir>`. This is the fix to the signal the mark-sweep version flagged about
  the `Ir` cut.
- **Frames stay `Rc`-managed with `Cell<u64>` slots.** The `Rc` pointer never
  moves, so the mutator's `locals` reference survives a collection; the GC
  rewrites the heap pointers *inside* the cells in place. Variable reads through
  `frame_get` therefore see relocated addresses automatically — env access is
  sound across a move for free.
- **Handles re-read.** The one thing a moving GC can't fix: a bare `u64` the
  mutator holds directly in a Rust local. After a move it points into from-space
  (a `Moved` marker) and dereferencing it is a loud use-after-move. The fix is
  the handle — `root(v)` publishes to the shadow stack, `Root::get` re-reads the
  relocated address.

The fusion: the compiler holds in-flight forms as bare `u64`s. `macroexpand`
roots the form and re-reads it through the root after each macro `invoke`;
`analyze` roots the form and re-derives every child from it (`child`) instead of
caching `list_to_vec(form)`. So a macro that triggers a collection mid-analysis
relocates the form, and sibling subforms are still read at their new addresses.
Caching the bare child list across the macro would be a use-after-move — that is
the clojure-jvm form-609 bug, now fixed by construction. The `gc` example shows
both: the relocation mechanism (handle re-reads, stale pointer dies) and the
compiler surviving `(f (firstof (40)) (h))` where `firstof` GCs mid-expansion.

**What is done and what is not, honestly:**

- Done and moving-safe: the relocation mechanism; the **compiler** (form-609);
  global and lexical-env access across a move; both execution tiers.
- Done on the **`CekMachine`**: the evaluator's operand temps *are* rooted across
  a move. The machine's continuation (`Kont`) holds its in-flight argument
  accumulators in `Cell` slots and is traced by `gc::walk_kont` at the `(gc)`
  safepoint, so `(+ heapval (do (gc) x))` — and captured continuations closing
  over relocated heap data — survive. This is the moving GC fused with the
  full-continuation tier, the combination the 45-way matrix deliberately excludes.
- Not yet on the **host-stack tiers** (`TreeWalk` etc.): a bare heap value held in
  an `argv` vec across a sub-expression that GCs would use-after-move. The same
  handle discipline applies; GC only fires at explicit `(gc)` safepoints, so this
  is reachable only by deliberately placing one there. Run such programs on the
  CEK tier, which handles it.
- Simplification vs. production: one growing arena with poisoned from-space
  (so stale reads stay loud) instead of flipping two fixed buffers and reusing
  from-space. The relocation semantics are faithful.

## Dispatch: the third axis (`dispatch.rs`)

Polymorphic method calls (`(area shape)`) resolve `(method, receiver type)` to an
implementation, and *how* they resolve is a swappable strategy, exactly like
value layout and GC. A minimal object model (records + `defmethod` + method-call
sites) gives something to dispatch on; three real strategies plug in behind one
`Dispatch` trait:

- `Megamorphic` — no cache, hit the registry every call.
- `MonomorphicIc` — one cached `(type -> impl)` per call site.
- `PolymorphicIc(k)` — up to k cached per site.

`set_dispatch` swaps the strategy and nothing else changes. All three compute the
same answer; they differ only in per-site cache behavior, which the `dispatch`
example measures: on an alternating-type call site the monomorphic cache thrashes
(all misses) while the polymorphic cache hits after warmup. This is the classic
inline-cache ladder, and it exercises the **dispatch⟺GC coupling** from the axes
graph: method impls are GC roots (the registry is forwarded), and inline caches
hold impl pointers that a move would dangle, so `collect` invalidates them. A test
locks that dispatch still resolves correctly across a moving collection.

Per-site cache state (`SiteId` assigned at analyze time) is where a native JIT
would attach the emitted guard chain; here it is a side table the strategy owns.

## Speculation + deopt: the fourth axis (`speculation.rs`)

A speculative tier assumes something and runs a guarded fast path; on a guard
failure it *deoptimizes* — abandons the assumption and finishes on a fallback,
correctly. `Speculative` expresses this with two swappable boundaries:

- It is a **`Dispatch` strategy** wrapping an inner (fallback) dispatch, so it
  rides the runtime hook both execution tiers call and composes with the
  interpreter AND the closure-compiler. Swap it in with `set_dispatch`.
- Inside it, a **`SpeculationPolicy`** decides what to speculate and when to
  give up: `NeverSpeculate`, `AlwaysMonomorphic` (re-arm on every deopt),
  `BlacklistAfter(n)` (give up on a site after n deopts).

The load-bearing invariant, in a test: **speculation never changes results.**
Every guard failure reconciles with the real receiver type, so all policies
compute the same answer; they differ only in spec-hits/deopts/fallbacks, which
the `speculation` example measures (poly site: always-thrash vs blacklist-give-up
vs never-speculate; mono site: speculate once then all hits).

Two honest points, both discovered building it:

- **It began as a `CodeSpace` wrapper and failed to compose** with the
  closure-compiler, which inlines the dispatch node and so escapes a node-level
  wrapper — the same erased-node boundary as `Traced` + `ClosureComp`. Making it
  a dispatch strategy (a runtime hook that survives compilation) fixed it, and
  is the more correct model.
- **In an interpreter, speculation-of-dispatch reduces to an adaptive dispatch
  strategy** — a guarded cached target plus a give-up policy. The guard and
  deopt are real and counted, but the DISTINCTIVE native payoff (inlining the
  impl body through the guard, unboxing across it, mid-frame deopt with SSA→frame
  state reconstruction) needs the emit tier. That is the honest edge, noted not
  hidden. Dispatch's inline-cache misses are exactly the type feedback that
  native tier would consume.

## The capstone: the emit tier (`bytecode.rs`)

Every axis so far supplied a *compute* form (run now on a value). The capstone is
a real **emit** tier: a bytecode compiler + stack VM — a distinct compile phase
producing an instruction stream a dispatch loop executes, genuinely different
from the tree-walker and closure-compiler — where the value model supplies the
**emit** half through `ModelEmit`. Given an op buffer, the model appends the
bytecode for `+ - * <`, and the emitted code differs by representation from one
source:

```
(+ (* 2 3) (* 4 5))     — same source, three representations:
  LowBit  => 26   [Const Const Sar 3 MulRaw  Const Const Sar 3 MulRaw  AddRaw Ret]
  HighBit => 26   [Const Const MulRaw        Const Const MulRaw        AddRaw Ret]
  NanBox  => 26   [Const Const Slow(Mul,2)   Const Const Slow(Mul,2)   Slow(Add,2) Ret]
```

LowBit shifts to untag before each multiply; HighBit (value unshifted under a
high tag) needs no shift; NanBox boxes integers, so arithmetic is a slow-path
runtime call — the boxing the `calc` example *measured* is now visible in the
*generated code*. The compiler is generic over `M: ModelEmit`; swap the
representation and the bytecode changes, the compiler does not. `fact 6` runs on
the VM (recursion via the same `top.invoke` composition). A test locks that all
three representations agree, and another that the emitted ops actually differ.

This is the interface boundary the codegen-axes doc promised, realized on the
foundational value axis. A machine-code tier is the same shape with a real ISA;
GC barriers, dispatch guards, and deopt each plug in the same way, as their own
emit forms. Scope: the tier covers arithmetic, control, calls/recursion,
closures, globals, and list prims (via a `Slow` runtime escape); `let`, records/
dispatch, and `(gc)` error clearly and run on the tree-walker. It is a focused
demonstration of the emit interface, not a complete backend.

## Continuations: the CEK tier (`cek.rs`)

The other tiers evaluate on the host (Rust) call stack, so "the current
continuation" is that native stack — uncapturable, gone once a call returns.
`CekMachine` makes the continuation an explicit, `Rc`-linked, heap-allocated data
structure (`Kont`) and evaluates as a loop over `Eval`/`Apply` states with no host
recursion. Because that continuation is first-class immutable data, it can be
reified and re-installed *any number of times*. This is the deepest validation of
the `CodeSpace` axis: a fundamentally different evaluation strategy slots in over
the same `Ir`.

What that unlocks, all as ordinary `Ir::Prim` nodes the machine interprets:

- **Full `call/cc`** (`%callcc`) — multi-shot, re-entrant undelimited continuations.
- **Delimited `shift`/`reset`** (`%shift`/`%reset`) — native `Prompt` delimiter +
  composable `PartialCont`; the captured slice splices onto the caller's
  continuation under a fresh prompt and *returns* (composes), any number of times.

The dividing line is not interpreter-vs-compiler — it is whether the continuation
is reifiable data or the host stack. The gradient is visible in the code:
`TreeWalk` (host stack) already supports *escape* continuations (`call/ec`, via a
typed panic that unwinds the stack), but cannot do multi-shot ones. On any
non-CEK tier `%callcc`/`%shift`/`%reset` reach a loud, specific error rather than
a wrong answer — some `Ir` nodes are only meaningful under some strategies.

**Continuations survive a moving GC.** A production language collects while
continuations are captured and live on the heap. Here they do: a `Kont`'s
argument accumulators are `Cell`-backed (rewritten in place like lexical frames),
`Ir` holds no heap pointers (constant pool), and `gc::walk_kont` traces both the
live continuation (at the CEK `(gc)` safepoint, via `collect_cek`) and
heap-captured `Obj::Cont`/`PartialCont`. `scheme/tests/delim_gc.rs` proves it: a
delimited continuation captured over heap data survives a collection that
relocates that data, then resumes correctly — twice. See
[docs/CONTINUATIONS.md](docs/CONTINUATIONS.md).

## A standards-flavored Scheme, as a library (`scheme/`)

The `scheme` crate is an R7RS-flavored Scheme built **entirely on the core's
public API** — the core has no idea Scheme exists. It is the live proof of the
library/language split: its own reader (`#t`, `#\c`, strings, quasiquote), a
`desugar` pass, `syntax-rules`, and then it hands plain core forms to `analyze`.
The same tiers that run the core's own Lisp run it.

`scheme/tests/conformance.rs` is a **61/61**-passing R7RS suite that doubles as
the roadmap, and every expected value is oracle-checked against Chicken Scheme
(`csi`) when installed, so a wrong expectation fails loudly. Coverage: arithmetic
and a numeric tower up through a dependency-free arbitrary-precision `BigInt`
(`bigint.rs`), lists/vectors/chars/strings, `let`/`letrec`/named-`let`/`cond`/
`case`, tail calls, `map`/`append`/`apply`/`call-with-values`, quasiquote,
**hygienic** `syntax-rules` (introduced bindings are alpha-renamed, so a macro's
`t` cannot capture the caller's), and the full continuation story above.

Honest edges: `dynamic-wind` is correct for non-escaping use but does not yet
re-run guards across continuation jumps; hygiene covers introduced bindings for
the common binding forms; the numeric tower stops at exact integers (no rationals
yet). See [docs/SCHEME.md](docs/SCHEME.md).

## What a real toolkit adds on top (not sketched here)

- A JIT `CodeSpace` whose `define`/`invoke` emit machine code (the `ClosureComp`
  tier already proves the incremental + late-bound contract; a native tier would
  reuse it). This is the genuinely hard engineering.
- Rooting the **host-stack** evaluators' operand temps across a move (the
  compiler and the CEK machine are moving-safe; `TreeWalk` et al. are not yet —
  see the GC section). Same handle discipline, applied pervasively.
- Making the CEK tier compose with **dispatch** (it currently errors on method
  dispatch), and automatic GC on allocation pressure (the mechanism is identical
  to the explicit `(gc)` safepoint, it just needs an allocation trigger).
- Fuller Scheme: rationals/the rest of the numeric tower, `dynamic-wind` across
  continuation jumps, full referential-transparency hygiene.
- The runtime library (persistent collections, seqs) generic over `ValueModel`.
