# Continuations

microlang supports the full gradient of continuations, and the codebase makes
the design line precise: **what a continuation costs is decided by how the
evaluator represents "the rest of the computation," not by interpreter-vs-compiler.**

## The gradient

| Kind | Needs | Works on a host-stack interpreter? | Engine here |
|---|---|---|---|
| Escape (one-shot, upward) | stack *unwinding* (exceptions) | ✅ yes | `TreeWalk` (`call/ec`) |
| Full `call/cc` (multi-shot) | continuation as *reifiable data* | ❌ no | `CekMachine` (`%callcc`) |
| Delimited `shift`/`reset` (multi-shot) | reifiable data + a prompt delimiter | ❌ no | `CekMachine` (`%shift`/`%reset`) |

### Escape continuations, on the host stack

`TreeWalk` evaluates on the Rust call stack, so "the current continuation" *is*
that stack. You cannot reify it, but you *can* unwind it. `Prim::CallEc`
(`call-with-escaping-continuation`) captures a one-shot, upward-only continuation
using a typed `EscapeSignal` panic: invoking the continuation throws, and the
enclosing `call/ec` catches it and unwinds to that point (`code.rs`). This is why
a host-stack interpreter can do escape continuations but not multi-shot ones —
you can throw an exception, but you cannot *un-unwind* a native stack you already
left.

### Full and delimited continuations, on the CEK machine

`CekMachine` (`cek.rs`) makes the continuation an explicit, `Rc`-linked,
heap-allocated data structure (`Kont`) and evaluates as a loop over
`Eval`/`Apply` states with no host recursion. Because `Kont` is immutable data,
it can be reified and re-installed any number of times.

- **`%callcc`** reifies the current `Kont` as an `Obj::Cont`. Invoking it discards
  the current continuation and re-installs the captured one (the abortive
  multi-shot jump).
- **`%reset body`** installs a `Kont::Prompt` delimiter and evaluates `body`
  under it.
- **`%shift f`** captures the `Kont` slice from the shift point up to the nearest
  enclosing prompt, reifies it as a composable `Obj::PartialCont`, and applies
  `f` to it under a re-established prompt. Invoking a `PartialCont` *splices* the
  captured slice onto the caller's continuation under a fresh prompt and returns
  (it composes) — `regraft` in `cek.rs` — so it works any number of times.

All three are ordinary `Ir::Prim` nodes. The IR is neutral about continuations;
the CEK machine is the only `CodeSpace` that gives those nodes meaning. On the
other tiers they reach a loud, specific error (`"%callcc requires the stackless
CekMachine"`) rather than a wrong answer.

## Why the other tiers *can't* (in principle vs in practice)

Nothing in principle stops an interpreter *or* a compiler from supporting full
continuations. The one requirement is that the continuation be represented as
first-class data the evaluator controls, instead of the host language's implicit
call stack. That is a structural choice — CPS, a defunctionalized abstract
machine (what CEK is), or an explicit-stack VM that snapshots its own stack —
available to interpreters and compilers alike. "Adding continuations to
`TreeWalk`" would mean rewriting it to thread an explicit continuation through
every step, at which point it *becomes* the CEK machine. That is exactly why full
continuations live in their own `CodeSpace` rather than being bolted onto the
tree-walker.

## Continuations survive a moving GC

A production language collects while continuations are captured and live on the
heap. Here they do, using the same discipline the moving GC already applies to
lexical frames:

- A `Kont`'s in-flight argument accumulators (`CallK`/`PrimK`'s `done`) are
  `Vec<Cell<u64>>` — the collector rewrites them in place, exactly like a
  `Frame`'s slots.
- `Ir` holds no heap pointers (constant pool), so a captured continuation's code
  needs no relocation.
- `gc::walk_kont` traces a `Kont` chain: it forwards the `done` cells and the
  captured frame environments, and follows `next`. It runs for both the **live**
  continuation (at the CEK `(gc)` safepoint, via `Runtime::collect_cek`) and any
  **heap-captured** `Obj::Cont` / `Obj::PartialCont` reached during the normal
  Cheney scan (`gc::scan_obj`).

`scheme/tests/delim_gc.rs` is the proof of concept for "you could build a
production language with delimited continuations on this kit": a delimited
continuation `k = λx. (* 2 (+ x (first data)))`, captured over a heap-allocated
`data`, survives a moving collection that relocates `data` (the test asserts
`relocated > 0`), then resumes **twice** — dereferencing `data` at its new address
each time — and computes the right answer. Without the collector tracing the
continuation's captured frame, that would be a use-after-move.

This is the moving GC fused with the full-continuation execution tier — the exact
combination the 45-way orthogonality matrix deliberately leaves out (the matrix
uses the three general tiers; the CEK tier is its own thing).

## Honest edges

- The CEK tier does not yet compose with method **dispatch** (`Dispatch` nodes
  error there) or with automatic GC on allocation pressure (only the explicit
  `(gc)` safepoint is wired; the mechanism for auto-GC would be identical).
- `dynamic-wind` (Scheme) is correct for non-escaping use but does not yet re-run
  its `before`/`after` guards when a continuation jumps across it — that needs a
  wind stack in the machine.
