# Deobfuscating `simple.js`: lessons and next steps

`/Users/jimmyhmiller/Documents/Code/deob/simple.js` is a control-flow-flattened
bytecode VM (a `while (pc >= 0) switch (pc & mask)` interpreter plus a CPS
"trampoline" execution driver, anti-debug timing, and a hand-rolled
`TextDecoder`). Specializing it against its own static bytecode is a first
Futamura projection, and it turns out to be an excellent adversarial test for the
partial evaluator: it exercises nearly every soundness corner at once. This doc
records what it taught us, what we fixed, and what to learn and fix next.

The throughline: **almost every bug here is a single-evaluation / effect-ordering
soundness bug**, where the residual re-reads mutable state at the wrong time, or
re-orders effects, or returns from the wrong stack frame. None were
`simple.js`-specific; each fix generalized.

---

## The methodology: differential effect tracing

The one tool that made this tractable is `tools/difftrace.js`. The insight:

> A correct residual is *observationally equivalent* to the original. Partial
> evaluation only folds away pure, internal computation, so it can never change
> the ordered sequence of **effectful / nondeterministic external calls**
> (`Date.now`, `console.log`, IO) or how the program terminates. Therefore the
> first point where the two effect traces differ is the first place the residual
> diverges, which is exactly where the specialization bug bit.

Two modes:

- **External effect diff** (default): runs original vs residual with effectful
  globals wrapped as logging proxies; prints the first divergence. Pure folded
  calls (`String.fromCharCode` on constants) are deliberately *not* tracked,
  because folding legitimately removes them. It doubles as a regression oracle.
- **`--rf` internal trace**: when the divergence is *before* the first external
  effect, this injects entry logging into every generated residual function
  (`__rfN`) and prints the handler sequence with a compact summary of each
  `{value}` cell argument, up to the throw.

**Lesson:** for a code-emitting partial evaluator, the right correctness oracle
is *observational equivalence at the external boundary*, not internal-state
alignment (folding destroys internal alignment). Build the differential tracer
first; it pays for itself immediately. A complementary "compare final global
state" check (`/tmp/cmpstate.js` pattern) catches data divergences the effect
trace can miss, **but beware**: matching `myGlobal` only proved listener
*registration* matched, not that the handlers *ran*. Adding real output
(`console.log`) was what exposed the deepest divergence. **Always validate on
genuine observable output, not a proxy for it.**

---

## Bug classes found and fixed

Roughly in the order the tracer surfaced them.

### 1. Scale / termination (the heap)
- **Append-only abstract heap → quadratic blowup.** The VM allocates per
  iteration; dead objects never got reclaimed, so memory and the memo's
  per-state hash grew without bound. Fix: persistent heap (`im::OrdMap`) +
  **garbage collection** of unreachable objects at block boundaries, with
  **monotonic, never-reused addresses** so reclaiming a slot can never alias its
  `escape_var` residual name.
- **`next_addr` leaked into the memo key.** The allocation counter was part of
  `State`'s `Eq`/`Hash`, so no two states ever compared equal once anything was
  allocated, and loops could never converge. Fix: exclude it from identity. This
  flipped `simple.js` from non-terminating to terminating.

**Lesson:** the abstract store must be finite (bounded) for loops to converge
(this is the AAM "finitize the store" point), and implementation details like an
allocation counter must never be part of a memoization key.

### 2. Spurious dynamism (folding gaps)
A static computation that *looks* dynamic because an operator isn't folded makes
the engine explore branches the real program never takes (here, the VM's
`switch (pc & 31)` dispatch forked all 32 cases per iteration → exponential).
Fixes, all "keep static things static":
- **Bitwise / shift folding** with JS 32-bit (`ToInt32`) semantics, including
  mixed-type coercion (`"mousemove" | 0 === 0`).
- **Ternary `?:`** folds to the taken branch when the condition is static.
- **Unary + logical + loose-equality folding**: `!`, `~`, `void`, `typeof`,
  `&&`, `||`, `??`, `==`, `!=` over static operands.

**Lesson:** any pure operator left "opaque" is a folding gap that can turn linear
specialization into exponential branch exploration. Model every pure operator's
constant-folding, even the ones that seem rare.

### 3. Single evaluation / effect ordering (the deep, recurring class)
This is the heart of the test case. Every instance: a value that reads mutable
state is used *after* that state changes, and the residual re-reads it.
- **Postfix `x++` reordering** (`Instr::Snapshot`): the byte reader
  `v5[v6][v7++]` read at the *new* `v7` because the "old value" was an
  expression re-read after the increment. Fix: snapshot the old value to a temp
  before the store.
- **`v4 = v3 ^ v8; v8 = …; return v4`**: `v4`'s expression re-read the mutated
  `v8`. Generalized fix: `freeze_readers` — **before any residual write
  (`SetProp`/`SetIndex`/`PushOp`/`SetGlobal`/delete), freeze every live value in
  the frame's locals and operand stack whose expression reads the location being
  written**, binding it to a temp holding the pre-write value.
- **Residual `try` effect ordering**: a local write before a may-throw op in a
  `try` body was batched at the boundary and lost on throw. Fix: eager,
  in-source-order residual stores inside residual `try`/`catch` bodies.

**Lesson:** representing a local's value as an *expression* (`Dyn(RExpr)`) is
unsound the moment that expression reads mutable state and the state mutates.
The residual must preserve **single evaluation** and **program order** of reads
relative to writes. We solved it pointwise (snapshot + freeze-at-write); see
"Next" for the case we *haven't* covered yet (freezing before opaque calls).

### 4. Capture-by-reference analysis
- **`assigned_expr` ignored computed member properties.** A variable mutated
  *only* via `++` inside `arr[i++]` wasn't detected as "assigned", so it wasn't
  boxed and was captured by value, dropping the increment. (`simple.js`'s `v7`
  masked this because it's *also* written `v7 = v1._j` in a catch.) Fix:
  traverse computed properties in the mutation analysis.

**Lesson:** the boxing/escape analyses must visit *every* sub-expression that can
mutate a variable, including computed indices and update expressions.

### 5. Materialization soundness
- **Cyclic object graphs** (a boxed cell whose value transitively captures the
  cell) were emitted with their uses before their definition. Fix: two-phase
  materialization (empty **shell** first, then **fill** fields/captures) for
  cyclic graphs; the compact inline form is kept for acyclic graphs (so big data
  arrays don't explode into element-by-element assignment).

**Lesson:** reconstructing a heap graph in the residual is a serialization
problem; cycles need shell-then-fill, and you want a cheap cycle check so the
common acyclic case stays readable.

### 6. Residual `try`/`catch` and inlining
- **Real residual `try`/`catch`** (`Op::Try { body, catch_slot, catch_body }`)
  with a foldability probe: a `try` that fully specializes still compiles its
  exceptions away to control flow; only a body with a may-throw residual op
  residualizes.
- **`catch` normal-completion codegen**: a `Halt` in a `try`/`catch`
  sub-program was rendered as `return undefined`, so a `catch` that should fall
  through and loop returned from the enclosing function. Fix: break out of the
  sub-program's trampoline (`break <label>`), not return.
- **Never inline a function containing a `try`.** The trampoline `v21` got
  inlined into `main`, so its body `return;` became a `return` from `main`,
  exiting before the firing loop. Fix: functions with a `try` are residualized
  as functions, so their `return` is a real function return.

**Lesson:** a `return` inside an inlined function must continue the caller, not
return from it; rather than thread that through a residualized `try`, the simpler
sound rule is "don't inline functions whose `try` will residualize." And nested
residual programs need their own block-exit semantics (break, not return).

### 7. Built-in modeling
- A small registry models `new TextDecoder().decode(...)`, `new Uint8Array(...)`,
  `String.fromCharCode(...)` so they fold to constants on static inputs, and
  **residualize as real runtime calls when inputs are dynamic** (rather than
  erroring). A built-in instance can also escape and is reconstructed as
  `new TextDecoder()` etc.

**Lesson:** modeled built-ins need both a fold path (static inputs) and a
residualize path (dynamic inputs); a fold-only model is a latent crash.

---

## Current state

`main(0)` on the residual now:
- terminates (from the original non-terminating hang),
- runs its main loop and reaches the **firing** of registered listeners,
- matches the original's **first 4 `Date.now` calls** exactly,
- runs `console.log` and emits initial output.

It is **not yet bit-exact**. The whole Rust test suite (66 frontend + 12 engine)
is green, including regression tests for each fix above.

---

## The open thread (next to fix)

Once `v21` (the execution driver) became a residual function, the firing runs at
*runtime* instead of being folded, and the tracer shows the divergence has moved
into it:

- The VM operand stack holds **`undefined` where the `Date` object should be**
  (`Date.now` → `undefined.now.apply` throws).
- Listeners are registered with **`null` handlers** (`addEventListener("mousemove", null)`).

So **function/object references that should be live on the stack during the
residualized execution are coming through as `undefined`/`null`** — a
state-threading gap between the folded decode (which builds those values at
specialization time) and the now-runtime execution driver.

Concrete hypotheses to investigate, highest-leverage first:

1. **`arguments` is silently lowered as an undefined global** (confirmed).
   `simple.js` uses `arguments` twice, e.g. `v3 = function v3() { …
   arguments.length … }`. The lowerer has no case for `arguments`, so
   `lower_ident` falls through to its unknown-name rule and emits
   `Global("arguments")`, which is `undefined` at runtime — so `arguments.length`
   becomes `undefined.length`. This both violates the project's no-silent-stub
   rule and is a concrete source of the wrong/`null` handler values. **Action:**
   either model `arguments` (bind the call's argument list, the cleanest fix) or
   make an unbound `arguments` reference a hard, clear error; then re-trace.

2. **Freeze before opaque calls.** `freeze_readers` runs before residual
   *writes*, but an opaque **call** (`Op::Eval` of a `Call`/`New`) can mutate
   anything. A value read before such a call and used after is still stale.
   `Date.now.apply(...)` and the method-call handler are opaque calls. **Action:**
   freeze live volatile readers before opaque calls too (conservatively: all
   `Get`/`Index`/`Global`-reading values), and measure the residual-size cost.

3. **State threading into the residualized driver.** When a `try`-containing
   function is residualized (rule #6), the VM state it needs (decoded handler
   table, the `Date`/`window` references) is passed via captured cells. Verify
   those captures carry the *post-decode* values, not stale/empty ones, and that
   materialization of the handler closures didn't drop a reference to `null`.
   Trace a single `Date`/handler value from where the decode produces it to where
   the driver reads it.

---

## Larger lessons for the partial evaluator

1. **Single evaluation is the dominant soundness risk.** We fixed it pointwise
   (snapshot for postfix updates, `freeze_readers` at writes). A more principled
   design would treat any `Dyn` value that reads mutable state as needing a
   **barrier**: it must be materialized to a temp before *any* effect that could
   change what it reads (writes *and* opaque calls). Consider making this a
   single invariant enforced in one place rather than per-construct.

2. **Inlining and residual control flow interact subtly.** A `return`,
   `break`, or `throw` inside an inlined function must resolve to the inlined
   function's frame, not the host's. The "don't inline functions with `try`"
   rule is a pragmatic instance; the general principle is that any non-local
   exit crossing an inlining boundary needs explicit handling.

3. **The store must be finite and its addresses must be opaque.** Bound it (GC /
   allocation-site abstraction) so loops converge; never let address identity or
   an allocation counter leak into memoization keys or escape-variable names in a
   way that can alias.

4. **Validate on real observable output.** Internal-state proxies (like
   `myGlobal`) can match while behavior diverges. The differential effect trace
   plus genuine program output is the trustworthy oracle.

5. **Keep `simple.js` as a regression target.** Each fix here is general; this
   one file is worth more than a dozen unit tests because it stresses the
   interactions. The minimal reproducers we extracted (postfix-update-in-`try`,
   boxed-increment-in-computed-index, single-eval-before-write) are the unit
   tests; `simple.js` is the integration test.
