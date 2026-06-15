# coil — Elaboration, `build`, and Hygiene

This is the document that resolves the hard part. `DESIGN.md` asserted "building
IR is evaluation, and macros run with a live MLIR context." That sentence hides
three real hazards that would sink the implementation if left vague:

1. If a macro can `build` IR to inspect a type, and then returns a *form* that
   gets built again, you double-emit ops.
2. Sometimes a macro needs a type *without committing* an op (overload choice).
3. Hygiene must survive macro-defining-macros with nested quasiquote
   (`defstruct`), or the whole "everything is a library macro" claim collapses.

This file defines the elaboration model that makes all three sound, then
re-validates `prelude.coil` against it and records the corrections it forced.

---

## 1. There is no separate "expand everything, then build everything" phase

coil does **single-pass elaboration**, not classic two-phase
(expand-all → compile). Elaboration is one traversal, `elab(form, env) -> Val`:

```
elab(form, env):
  if atom            -> reflect (numbers, strings, type symbols → MlirType, …)
  if (quote x)       -> x
  if special-form    -> per KERNEL §3
  if (head … ) and head names a MACRO ->
        expanded = apply-macro(head, raw-args, env)   ; runs NOW, may build
        elab(expanded, env)                           ; re-elaborate result
  if (head …) and head is a dotted op symbol ->
        elab((op-call head args…), env)               ; → builds an Op, returns its result Value
  if (head …) otherwise ->
        f  = elab(head, env)
        vs = map elab args
        apply(f, vs)
```

Key consequence: **elaborating an op-form builds it** (inserts at the current
insertion point) and returns its result as an `MlirValue`. There is no later
"build phase" to double-emit into. A macro that runs during `elab` is running
*in the middle of building the surrounding function*, with `*builder*` live.

This is the Terra/Zig-`comptime` model, not the Common-Lisp model. It is the
only model in which "macros can see types" is both true and sound.

---

## 2. The macro return protocol (the anti-double-emit rule)

A macro returns a `Val`. How `elab` treats it:

| Macro returns | `elab` does | Use when |
|---|---|---|
| a **form** (List/Sym/…) | re-elaborates it (may build) | normal sugar (`defn`, `when`) |
| an **`MlirValue`** | splices it as-is (already in IR) | you already `build`-ed it |
| an **`MlirType`/`MlirAttr`** | splices as a literal | type/attr-computing macros |

**The rule:** *if you `build`, you must return the resulting `MlirValue` (or a
form that splices it), never the original operand form.* Returning the form would
re-build it. Quasiquote makes this natural — you splice the value with `~`:

```clojure
(defmacro widen-i64 [x]
  (let [v  (build x)]                 ; builds x ONCE, v : MlirValue
    (if (mlir/type= (mlir/value-type v) i64)
        v                             ; return the value → spliced, not rebuilt
        `(arith.extsi ~v))))          ; ~v splices the already-built value
```

Here `arith.extsi`'s operand is the value `v`, not the form `x`, so `x` is built
exactly once. This discipline is the whole answer to hazard (1).

---

## 3. Three ways to "see a type", in order of preference

Building to learn a type is the heavy hammer. Most macros want lighter tools:

1. **`(mlir/infer-results name attrs operands regions) -> [Type]`** — *pure*.
   Queries `InferTypeOpInterface` without creating or inserting anything. Use
   this whenever you only need the result type of an op you're *about* to emit.
   No double-emit hazard because nothing is built.

2. **`(mlir/value-type v) -> Type`** on an operand you already have — also pure.
   The operand was built by the caller before the macro ran (args are
   elaborated… *unless* the macro wants them raw — see §4).

3. **`(build form) -> Value`** — commits the op. Only when you actually want the
   value in the IR. Subject to the §2 splice-the-value rule.

4. **`(with-scratch (fn [] …)) -> a`** — opens a throwaway builder scope (a
   detached region in the same context). Ops built inside are discarded when the
   thunk returns; only the pure facts you extract (types, trait answers) escape.
   This is for genuinely *speculative* construction — e.g. "try building it as
   `arith.addi`; if that doesn't verify, build `arith.addf` instead." Hazard (2),
   solved.

Guidance baked into the prelude: reach for `infer-results` first, `value-type`
second, `with-scratch` for speculation, and bare `build` only when committing.

### Are macro arguments pre-built?

No. A macro receives its arguments **unevaluated** (as forms) — that's what makes
it a macro. So in `widen-i64`, `x` arrives as a form; the macro decides whether
to `build` it. This is deliberate: a syntactic macro like `when` must *not* build
its branches eagerly. The cost is that a macro wanting an operand's type must
`build`/`infer` it itself. That cost is exactly the power the old system lacked.

---

## 4. Effects, ordering, and compile-time state

Because elaboration is a single ordered pass, side effects have a defined order:
top-to-bottom, depth-first. That gives the mutation-based prelude features
(`intern-cstring!`, dialect registration) a well-defined semantics:

- **`eval-when :compile`** bodies run when elaboration reaches them — i.e. before
  any later form is elaborated. `require-dialect` therefore loads dialects before
  the ops that need them, with no separate pre-pass.
- **Compile-time atoms** (`atom`/`swap!`/`get`/`assoc`) are scoped to the
  *compilation unit* and live only during elaboration. `*interned*` is one such
  atom; its lifetime is the module build. They are not runtime values.
- **Idempotency:** `intern-cstring!` is idempotent by construction (it checks the
  atom before appending a global). Dialect/pattern registration must be too;
  `mlir/load-dialect` and `mlir/register-irdl` are defined to be no-ops on a
  second identical call.

There is no nondeterminism: one pass, one order, effects commute only when the
programmer arranges it.

---

## 5. Hygiene: scope-set algorithm

coil uses **scope sets** (Flatt's model, as in Racket), which is the only hygiene
algorithm known to compose correctly through macro-defining-macros.

- Every identifier carries a set of *scopes* (integers).
- Reading source adds the module scope. Each macro *expansion* mints a fresh
  scope and adds it to every identifier the macro *introduces* (template
  literals), but not to identifiers that flowed in from the call site (they were
  already marked at their own birth).
- Binding resolution: an identifier reference resolves to the binding whose scope
  set is the largest subset of the reference's scope set. Introduced and
  passed-in identifiers thus never collide even when spelled the same.
- `(gensym)` mints a globally-unique name (belt-and-suspenders for codegen-y
  cases). `(unhygienic e)` strips the introduction scope (intentional capture).
  `(datum->syntax ctx d)` adopts `ctx`'s scope set for `d`.

Worked check — `or`'s temp:

```clojure
(defmacro or [& xs] … `(let [t# ~(first xs)] (if t# t# (or ~@(rest xs)))))
```

`t#` is template-introduced → gets the expansion's fresh scope. A caller's own
`t` has a different scope set, so `(or t false)` cannot capture. ✔

### The `defstruct` stress test (macro-defining-macro, two quote levels)

The original prelude wrote the generated accessors with double-unquote
(`~~sty`, `~~i`), which is correct under nested-quasiquote semantics but is a
notorious footgun and interacts subtly with scope sets. The model says: *don't
nest quasiquote to smuggle values into a generated macro — close over them in a
plain `let` in the outer macro and splice once.* That is hygienic and obvious.
This forced a prelude correction (§6).

The deeper point hygiene must guarantee: the *generated* getter macro
`Name/field` expands, at the *use* site, to `llvm.getelementptr`/`llvm.load`
referencing the struct type and index captured at `defstruct` time — and its
introduced temporaries don't capture the user's names. Scope sets give this for
free because the captured `sty`/`i` are spliced as *values* (an `MlirType` and an
`Int`), not as identifiers, so there is nothing to capture.

---

## 6. Corrections this model forced on `prelude.coil`

Re-validating each prelude macro against §§1–5 surfaced exactly one real bug and
two clarifications:

1. **`defstruct` (bug).** Replaced the double-unquote (`~~sty`/`~~i`) generation
   with a single-level template that closes over `sty`/`idx` as values bound in
   the outer macro. Same behavior, no nested-quasiquote footgun, provably
   hygienic (values, not identifiers, are captured). *Applied in prelude.coil.*

2. **`op-call` (clarification).** Operands in an op-call are elaborated by `elab`
   in the normal head-application path *before* `mlir/op` runs — so by the time
   `mlir/op` sees them they are `MlirValue`s. `op-call` itself does no building;
   it only rearranges syntax. No change needed, but noted so the reader matches
   §1.

3. **`widen-i64` (clarification).** Confirmed it builds `x` once and splices the
   value (§2). It is the canonical example of a type-directed macro and now
   carries a comment pointing here.

No other prelude macro builds during expansion, so none can double-emit. `defn`,
`when`, `cond`, `and`, `or`, `->`, the `i*` arithmetic, `passes`, `emit`,
`require-dialect` are all pure syntax→syntax and trivially sound.

---

## 7. Summary

Single-pass elaboration (build = eval), the splice-the-value rule, the
infer/value-type/scratch ladder for seeing types, ordered compile-time effects,
and scope-set hygiene together make the design's headline — "macros are ordinary
compile-time functions sharing a live MLIR world" — actually buildable without
double-emission, nondeterminism, or capture. The `build` primitive and
`with-scratch` are now first-class in `KERNEL.md §4`; the one prelude bug the
model caught (`defstruct` double-unquote) is fixed.
