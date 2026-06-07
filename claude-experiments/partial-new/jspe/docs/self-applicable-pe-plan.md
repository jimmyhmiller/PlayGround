# Plan: a self-applicable partial evaluator as powerful as jspe

## The problem, stated honestly

We have two evaluators and a recurring wall:

- **jspe** — an *online* PE in full Node. Powerful (compiles real BF directly), but written in big Node, so it can't be fed to itself.
- **mix / mix_imp** — PEs in jspe's subset, made only so *something* could be self-applied.

Every failure on the P2/P3 road has the **same root cause**: we are self-applying an **online** PE, and online self-application is the genuinely hard case. The symptoms:

1. **The "trivial projection"** — `jspe(mix,int)` produced a compiler that re-interprets, because online decisions can't be pre-committed.
2. **Binding-time confusion** — under self-application the value *tags* (`"s"`/`"d"`) were static while *payloads* were dynamic, so the PE took static branches on dynamic data.
3. **Non-termination** — `mix_imp` inlines the interpreter's recursion forever on a dynamic program (no memoization/whistle).
4. **OOM / scaling** — eager `heapCopy`/`envCopy` snapshots, plus the size of an online PE, blow up when specialized.

These are not four bugs. They are four faces of one fact: **online PEs decide binding times *during* specialization, using actual values — which is exactly what is unavailable when the PE is itself the subject of specialization.**

## The decision: go offline

The established, textbook route to clean Futamura P2/P3 (Jones–Gomard–Sestoft, "Partial Evaluation and Automatic Program Generation"; the `mix`/Similix/C-mix line) is an **offline** PE:

> A **binding-time analysis (BTA)** runs *first* and annotates every construct as **S** (static) or **D** (dynamic). The **specializer** then just *follows the annotations*: evaluate S, residualize D.

Why this fixes all four symptoms by construction:

- The specializer **dispatches on annotations**, which are static at *every* level — including when the specializer is itself being specialized. No binding-time confusion. (Fixes #2.)
- Annotations pre-commit the static/dynamic split, so `spec(spec, int)` is a clean compiler, not the trivial projection. (Fixes #1.)
- Termination is **polyvariant function memoization** keyed by static-argument values + annotation-directed loop handling — a finite, well-understood mechanism. (Fixes #3.)
- An offline specializer is **much smaller and lighter** than an online one (no partial-static abstract heap, no per-step generalization), so self-applying it is tractable — this is precisely why the field self-applies offline PEs and not online ones. (Fixes #4.)

"As powerful as jspe" — for **interpreter specialization** (the Futamura use case: BF, calc, etc.) an offline PE with a decent BTA is as powerful, and is *the* way these results are actually achieved. Online PEs win on precision in adversarial cases (partial-static structures decided by runtime values), but they do not self-apply cleanly. The standard recovery is **binding-time improvement** (restructure the interpreter / add a few annotations) rather than going online.

**Recommendation: build ONE offline PE in the subset. It does P1/P2/P3 by itself, run by Node. jspe stays only as a convenience for direct P1 (or is retired).**

## Architecture (one PE, two passes, both in the subset)

```
  source ──parse──▶ AST ──BTA(sig)──▶ annotated AST ──spec(static args)──▶ residual source
                          (offline, once)            (the part that self-applies)
```

- **`bta(funcs, sig)` → annotated funcs.** `sig` says which entry params are S/D. Produces, for every expression and statement, an `S`/`D` tag, plus per-function call-site divisions (polyvariant). This is the only genuinely new algorithm.
- **`spec(annFuncs, entry, staticArgs)` → residual program.** Annotation-driven. This is structurally similar to `mix_imp` but *simpler*, because it never decides binding times — it reads them.
- Both are **first-order subset programs** (no closures, no Maps; env = pairs array, memo = pairs array, residual = tagged-array AST → printed to JS). Same discipline `mix_imp.subset.js` already proved lowers under jspe.

### The specializer's core (annotation-directed)

| construct | annotation | action |
|---|---|---|
| literal / var | S | evaluate (return value) |
| any expr | D | residualize (emit residual expr) |
| `bin`, `idx`, … | S | compute; **D** | emit residual |
| `if` | S cond | take the branch | **D** cond | residualize both, emit `if` (join handled by BTA marking divergent vars D) |
| `while` | S cond | unroll | **D** cond | residualize loop body once over D-vars, emit `while` |
| call | — | **polyvariant memo**: key = (fn, values of its S args). Seen → emit call to existing residual fn. New → make residual fn, specialize body once, recurse |

The **memoization on (fn, static-args)** is the termination story and the residual-function generator, all in one — and it's the piece `mix_imp` lacked. Because the key is built from *static* values (annotation-guaranteed), it is well-defined under self-application.

### Why BTA makes the join/loops trivial for the specializer

- A `D`-conditioned `if` where a variable is assigned differently in the two branches: **BTA marks that variable `D`**. So at spec time it's already a residual var; both branches emit assignments to it; no phi/heap-merge gymnastics (the thing that made `mix_imp` need eager `heapCopy`).
- A `D`-controlled `while`: BTA marks the loop-carried vars `D`. The specializer emits one residual loop. No whistle needed — the *analysis* decided.
- The BF tape: BTA over the store marks cells `D` (input-dependent) and the pointer `S` (data-independent) → cells scalar-replace, dispatch unrolls, `[...]` becomes a residual `while`. Clean BF, and it self-applies.

## The data-structure point (your instinct, made concrete)

Offline removes most of the need for the persistent structure, but two structures still matter and should be **persistent / append-only** from the start (this is what bit us as eager copies):

1. **The specialization memo** — `(fn, static-arg-key) → residual-fn-name`. Append-only list; never copied. Self-application stays bounded because keys are finite (finitely many static-arg patterns for a static interpreter).
2. **The environment** — immutable cons-list (share the tail on extension) rather than the copied pairs array. Branch snapshots become O(1) (just hold the old head) instead of O(n) `envCopy`. This is the "persistent data structure like the Rust one," and it's the difference between self-application scaling and OOM.

No partial-static abstract heap is needed (offline doesn't have one), which is the single biggest source of jspe's self-application weight that simply disappears.

## Phases (each independently testable; the oracle is always "does the residual run == the interpreter?")

- **Phase 0 — scaffolding.** Decide the residual IR is the subset AST itself (so projections compose: target of one phase is a valid input to the next). Printer to JS. Reuse `parse.js`.
- **Phase 1 — specializer first, BTA by hand.** Write `spec` (annotation-driven), and hand-annotate a tiny interpreter. Verify P1 on calc-style programs. (De-risks the easy half early.)
- **Phase 2 — BTA, expression + control core.** Fixpoint S/D for var/bin/if/while/return/call over a first-order functional interpreter. Verify `spec(bta(int), src)` == hand-annotated result.
- **Phase 3 — self-application.** `spec(bta(spec), int)` = compiler; check `compiler(src)` == `spec(int,src)`. `spec(bta(spec), spec)` = cogen (P3). This is the milestone the whole plan exists for.
- **Phase 4 — store/array BTA + imperative.** Extend BTA to mutable arrays and the store (which cells are S/D, pointer S/D). This is the hard phase and the prerequisite for BF.
- **Phase 5 — real BF at P2.** `spec(bta(bf_interp), program)` = target (run by `spec`), and `spec(bta(spec), bf_interp)` = a standalone BF compiler. Differential-verify against the existing real BF interpreter.

## Honest risks

- **BTA for the imperative store (Phase 4)** is the hard, novel part — getting "pointer static, cells dynamic" right for BF is the crux. This is where C-mix-style analyses live; it's established but fiddly.
- **Self-application still has to scale** (Phase 3) — offline is far lighter than online, but the specializer specializing itself is still real work; the persistent env/memo above are load-bearing, not optional.
- **It's a rewrite, not a port.** We keep `mix_imp`'s proven *specializer* logic (much of it carries over to the annotation-driven `spec`), but BTA is new and the online machinery (whistle, abstract heap, partial-static values) is *dropped*, not ported.
- **Precision tradeoff.** A few interpreters that jspe handles by online precision may need a binding-time improvement (small interpreter restructuring) to specialize well offline. Acceptable and standard.

## What we keep

- `parse.js` (front-end), the residual printer / `backend.js` ideas, the subset itself (now with `typeof`/`!`/`%`/`/`/`null`), and `mix_imp`'s specializer skeleton (re-pointed at annotations).
- jspe stays usable for direct P1 (`bfc.js`) during the transition; it is no longer the path to P2/P3.

## One-line summary

Stop self-applying an online PE. Build **one offline PE** (BTA + annotation-driven specializer) in the subset, with a **persistent env + append-only memo**; it does P1/P2/P3 itself, compiles real BF, and self-applies cleanly — because the specializer only ever dispatches on static annotations.
