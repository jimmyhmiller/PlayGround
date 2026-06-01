# RESOLVED — was: "Reentrant `run_jit` loses direct-call return values"

## ✅ RESOLVED 2026-05-31 — it was NOT reentrancy, NOT the JIT executor

> **TL;DR of the actual fix.** The bug had nothing to do with reentrancy,
> nested `run_jit`, `x23`, the ControlAware outcome protocol, the call table,
> or GC. It was a plain analyzer logic error in `analyze_seq` (clojure-jvm
> only — no shared crate touched):
>
> A namespace-QUALIFIED head symbol whose namespace part contained a `.`
> (e.g. `clojure.core/inc`) was unconditionally rewritten as a *static method
> call* `(. clojure.core (inc 41))`, because the `looks_like_class` heuristic
> was `ns.contains('.')`. `clojure.core` is a NAMESPACE, not a class, so the
> rewrite found no host method → emitted a runtime `ThrowExpr` → the top-level
> caught it and returned `nil`. Bare `(inc 41)` worked because it took the
> normal var-invoke path. The var value was identical in both cases; only the
> qualified call path was broken.
>
> **Why it looked like a reentrancy bug.** The probe only exercised qualified
> calls (`clojure.core/inc`, `clojure.core/find-ns`) from inside a reentrant
> `RT/load`. The exact same `nil` reproduces **non-reentrantly**:
> `(clojure.core/inc 41)` returns `nil` at a bare top-level eval after core
> loads, while `(inc 41)` returns `42`. Reentrancy was a red herring; the
> minimal reentrant fn-call (`(myinc 41)` where `myinc` is a user var) always
> returned the correct value.
>
> **The fix.** In `analyze_seq`, the static-call sugar now skips the rewrite
> when the namespace part is a known namespace (directly or via an alias in
> the current ns) or when the qualified symbol resolves to a Var — those are
> ordinary fn invocations. Only true class names (`Math/ceil`,
> `clojure.lang.RT/load`) still take the static path.
>
> A second, latent issue surfaced once qualified calls reached the invoke
> path: `InvokeExpr::emit` panicked when its head expression *diverged*
> (a `ThrowExpr` for an unmodeled host-class reference in a macro body that
> only fires on a bad call, e.g. `defn`'s `(throw (IllegalArgumentException.
> …))`). It now returns `None` for a diverging head, mirroring the existing
> diverging-arg handling — the call is unreachable, the block already
> terminated.
>
> **Result.** `tests/interesting_features::probe_reentrant_refer` now asserts
> `(clojure.core/inc 41) == 42` and a non-nil `find-ns`. The upstream-core
> loader frontier moved from form **215** to form **430** (a different,
> unrelated blocker: a corrupt/unmodeled heap value reaches `RT.first`,
> `runtime.rs` panic — separate work).
>
> Everything below is the ORIGINAL (wrong) investigation, kept for the record.

---

## Summary (ORIGINAL — disproven)

When a JIT-compiled form is evaluated **inside** another running JIT frame
(a nested `run_jit`), the **return value of a direct JIT→JIT function call is
lost** — the caller reads `nil` even though the callee executed correctly and
returned a real value. Literals and var-derefs are unaffected; only *call
results* are dropped. The failure is **deterministic** (not GC-timing
dependent).

This is the current blocker behind `reduce`: `clojure/core_protocols.clj` is
loaded via `RT/load`, which runs reentrantly, and `clojure.core/refer`'s body
makes ordinary function calls (`find-ns`, `apply`, `ns-publics`, …) whose
results come back `nil`, so `refer` populates nothing and the loaded file's
unqualified `seq` never resolves.

## How we get into a nested `run_jit`

```
user eval / core load
  └─ run_jit(form)                         ← outer JIT frame
       └─ JIT code calls clojure.lang.RT/load
            └─ cljvm_rt_load (extern, runtime.rs)
                 └─ with_active_session_load_resource (compiler.rs)
                      └─ for each form: sess.eval_form(form)
                           └─ run_jit(form)   ← INNER (nested) JIT frame  ⟵ bug lives here
```

`load` is called from compiled code, so each loaded form compiles + runs via a
**second** `run_jit` while the first is still on the stack. This is the correct
Clojure load model (synchronous, in-order), and the architecture is meant to
support it.

## Isolation

Resource `"reentrant-refer-probe"` (`resource_source` in `compiler.rs`) loaded
via `RT/load`, read back afterward with a **direct `VarExpr`**
(`clojure.core/rr-lit`), *not* `(deref (var …))` — the latter does not read the
root and produced an early false "all nil".

Test: `tests/interesting_features.rs::probe_reentrant_refer` (ignored).

| Form (run reentrantly)                  | Result | |
|-----------------------------------------|--------|---|
| `(def rr-lit 4242)` (literal)           | `4242` | ✓ |
| `(def rr-done 1)` (literal)             | `1`    | ✓ |
| `(def rr-fnval clojure.core/inc)` (var deref) | fn handle | ✓ |
| `(def rr-call (clojure.core/inc 41))` (fn **call**) | `nil` | ✗ |
| `(def rr-findns (clojure.core/find-ns 'clojure.core))` (fn **call**) | `nil` | ✗ |

For `rr-findns` the callee provably runs: `cljvm_ns_find` fires, receives the
correct `'clojure.core` symbol, and **finds** the namespace. Only the value's
trip back to the caller is lost.

Same facts under `CLJVM_GC=every`, so it is **not** a GC/safepoint-rooting
timing issue. The *same compiled thunk* works when run non-reentrantly
(top-level `(def x (f a))` is used constantly during core load), so it is the
nested **execution context**, not codegen.

## Root cause (mechanism)

> **UPDATE 2026-05-31 — the x23 theory below is DISPROVEN. Read this first.**
>
> A parity iteration implemented the proposed x23 save/restore in
> `cljvm_rt_load` and **measured no effect**: `rr-call` still returns
> `nanbox_nil` with and without it. The mechanism cannot exist, because **`x23`
> is only saved/restored in the JIT prologue/epilogue** (`dynlower/src/backend.rs:390`
> stp, `:414` ldp) and is **never read or written in any lowered function body**;
> `batch_lower.rs` never consumes it. The `JitControlContext` is consumed by Rust
> handlers via the `ctx` Rust pointer (`dynlower/src/lib.rs:2379/2467`), not via
> `x23` at runtime — so `x23` cannot influence a JIT body's return value.
>
> **Where the bug actually lives:** the shared JIT execution machinery, not
> `clojure-jvm`. Under a nested `run_jit` (`RT/load` → `with_active_session_load_resource`
> → `eval_form` → `gc.run_jit` while the outer load's `run_jit` is live), a JIT
> call's **return value** is dropped (`nil`), deterministically, while literal
> defs and var-derefs survive. Suspect the nested session / suspended-frame
> interaction in `dynruntime/src/jit.rs` (`JitSafepointSession::with_installed`),
> `dynlang/src/gc.rs` (`run_jit`), and the call-table / `call_outcome` plumbing
> in `dynlower` under nested entry. **Fixing this touches SHARED crates
> (dynlang/dynruntime/dynlower) that beagle/lox also use — needs sign-off and
> careful investigation, not a blind edit.**
>
> The original (now-disproven) ControlAware/x23 hypothesis is preserved below for
> the historical record only.

The Session's JIT uses `CallMode::ControlAware` (`compiler.rs`, ~line 10780 and
~11470) — needed for the continuation/prompt machinery behind exceptions.

In that mode, a JIT entry returns an **outcome tuple**, not a plain value. The
arm64 entry shim `call_jit_regs_with_reg_limit` (`dynlower/src/lib.rs` ~2375)
passes a `JitControlContext*` in **x23** and reads the outcome back as:

```
x0 = result   x1 = kind   x2 = payload0   x3 = payload1
```

where `kind` ∈ `{ReturnValue, ReturnVoid, Exception, Deopt, CaptureSlice,
CloneSlice, ResumeSlice, AbortToPrompt}` (see `call_jit_outcome_with_reg_limit`).

Direct internal calls in ControlAware mode also use this protocol (caller emits
the call, then branches on the `kind` register; on `ReturnValue` it uses `x0`).
When such a call happens **inside a nested `run_jit` frame**, the inner thunk
was itself entered through `call_jit_regs_with_reg_limit` with its **own**
`JitControlContext` installed in x23. The control-aware return of a call made
within that nested frame is mis-handled and collapses to `nil`.

Suspected exact culprit (to confirm): the `kind`/`ctx` register convention
(x1 / x23) for an internal control-aware call return is clobbered or
mis-interpreted across the nested-entry boundary, so a normal `ReturnValue` is
read as a non-value outcome and the caller substitutes `nil`.

## Why earlier theories were wrong

- **Not** "instance-method receiver emits `None`" — that was a red herring from
  probe forms that themselves crashed.
- **Not** the `refer` infrastructure — the full chain (`find-ns` → `ns-publics`
  = 652 → `(. *ns* (refer sym v))` in a `doseq`) works **non-reentrantly**.
- **Not** GC — deterministic under `CLJVM_GC=every`.
- **Not** the bare-class mock-var bug — that was a *separate* real bug, now
  fixed (see below).

## Adjacent bug fixed along the way

`prelude.clj` defines mock vars (`(def String "#<MOCKED java.lang.String>")`)
so references to *unregistered* classes compile through. `analyze_symbol` ran
the var lookup **before** the host-class lookup, so bare/qualified *registered*
class names resolved to the mock var (a string) instead of the class. That
broke `(instance? String x)`, `(symbol "s")`, `(Exception. …)` for all code
compiled after prelude loads. Fixed by checking `host_class::lookup` **before**
`resolve_var` in `analyze_symbol`. Regression test: `probe_inst_args`.

## Proposed fixes (pick one)

1. **Fix the executor (correct, harder).** Make the ControlAware outcome
   protocol survive nested `run_jit`: ensure an internal call's `kind`/value
   registers (x1/x0) and the `ctx` register (x23) are handled correctly when the
   call occurs inside a nested entry frame. Likely localized to
   `call_jit_regs_with_reg_limit` / the lowerer's control-aware call-return
   sequence. Needs care + the executor author's eye.

2. **Avoid nesting (workaround).** Have `cljvm_rt_load` **queue** the resource's
   forms and let the top-level eval loop (`load_clojure_core` / the driver)
   drain the queue between forms, so each loaded form runs in a fresh top-level
   `run_jit` (non-nested). Ordering is preserved (forms still load before any
   subsequent user form), but it diverges from Clojure's strictly-reentrant
   model.

## Repro

```
cd crates/clojure-jvm
cargo test --test interesting_features probe_reentrant_refer -- --ignored --nocapture
# rr-lit / rr-done persist; rr-call / rr-findns come back nil.
```
