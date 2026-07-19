# Stage 3 — staged macros via interleaved elaboration (design + plan)

> **Status: this is the original design/plan doc.** Stage 3 shipped and the old
> macro Lisp was deleted; the wording below (`defmacro`, "alongside the old Lisp")
> reflects the plan, not the final shape. For how the implemented system actually
> resolves names — module scoping, `coil.core` auto-refer, trait/method
> namespacing, and the two-phase resolve with its `strict` gating — see
> **[NAMESPACING.md](NAMESPACING.md)**.

Goal: make **macros ordinary Coil code** that run during compilation, manipulate
**code as a first-class value**, and use the **real (typed) reflection** — replacing
the separate pre-type macro Lisp. One language, available at every stage.

This is a front-end rearchitecture, not a feature. It is built as a sequence of
individually-green steps; the existing macro Lisp keeps working until the last step
retires it.

## The end state

- A macro is a comptime function `[Code] -> Code`, run by the comptime interpreter
  (the one we already have). It can call any `defn`, recurse, use `=`/arithmetic,
  and use typed reflection (`field-count`, `field-type`, `impls?`, …).
- `Code` is a first-class comptime value (`CtVal::Code`, `Type::Code`): quoted with
  `quote`/quasiquote, taken apart with `code-*` accessors, built with quasiquote +
  unquote that splices comptime values.
- The pipeline becomes an **elaboration loop** over top-level forms: expand → check
  → extend the environment → continue, so a macro sees everything elaborated so
  far (including types).

```lisp
(defmacro derive-eq [T]                       ; T : Type (a real reflected type)
  (and-chain
    (map (fn [f]
           (if (impls? (field-type T f) Eq)
               `(= (field a ~f) (field b ~f)) ; nested type has Eq → recurse
               (error (concat "field " (field-name T f) " is not Eq"))))
         (fields T))))
```

## Architecture change: phases → an elaboration loop

Today: `read → expand_all (macro Lisp) → parse → resolve → check → mono → codegen`.

Stage 3: a single **elaborator** drives the front end incrementally —

```
env := empty            ; types, sigs, structs, sums, traits, macros, consts
for form in top_level_forms (in source order):
    form := macroexpand(form, env)        ; if head is a macro, run it (comptime) → Code, recurse
    case form:
      def/struct/sum/trait/impl/const/macro:
            resolve(form, env); check(form, env); env := env + form
      expr (top-level eval, rare): resolve/check/eval
mono(env.funcs); codegen
```

`macroexpand` runs a macro by **comptime-evaluating** its body (already checked,
because the macro was defined earlier in the loop) on the *quoted* argument forms,
yielding a `Code` value that is then resolved + checked like hand-written code.
Because expansion is interleaved with checking, a macro can query types.

## Code as a value

- `Type::Code` — opaque, comptime-only. A `Code` value escaping to runtime is a
  hard error (codegen has no representation).
- `CtVal::Code(Sexp)` — wraps raw syntax.
- `quote` / `` ` ``(quasiquote) at the **expression** level produce `Code`;
  `~`(unquote) splices a comptime value into the template. (Today these live only
  in the macro Lisp; Stage 3 lifts them into the runtime/comptime layer — the same
  syntax, now one mechanism.)
- Accessors (mirroring `lib/sexp.coil`, but over the compiler's `Sexp` at comptime):
  `code-list?` / `code-sym?` / `code-int?`, `code-count`, `code-nth`, `code-sym`
  (→ string), `code-int` (→ i64). `gensym` for hygiene.

## Hygiene

The macro Lisp already implements hygiene (template symbols resolve in the macro's
definition module; `gensym` for fresh names). That logic moves to the new layer:
quasiquoted identifiers carry their definition scope; `gensym` mints fresh ones.
This is the subtlest part to port and gets its own step + tests.

## Step plan (each step compiles + keeps the suite green)

1. **Code as a comptime value (read side).** `Type::Code`, `CtVal::Code`, `quote`,
   and the `code-*` accessors; comptime can quote and inspect code. (Additive; no
   pipeline change. Inert until later steps, but the foundation.)
2. **Quasiquote/unquote → Code.** Build `Code` from templates with comptime
   unquote splicing; `gensym`. Now comptime can *generate* code values.
3. **Incremental elaborator (no behavior change).** Refactor `check` to process
   top-level forms in source order against a growing environment, producing the
   same result as today. The riskiest step; pure refactor, guarded by the suite.
4. **`defmacro` as a comptime `[Code] -> Code`.** Hook macro definitions + calls
   into the elaborator: a macro call comptime-evaluates its body on quoted args,
   splices the result, checks it. New-style macros work alongside the old Lisp.
5. **Typed reflection in macros.** Expose `impls?`, `field-type` (as a `Type`
   value), etc. to macros — the payoff (generation that sees types).
6. **Migrate + retire the macro Lisp.** Rewrite `derive`, `case`, control-flow
   macros as Stage-3 macros; delete the old evaluator.

## Risks / rollback

- Step 3 (incremental elaborator) is the load-bearing refactor; if it can't be made
  behavior-preserving cleanly, the const/comptime/trait ordering we rely on must be
  re-expressed in the loop. Each step is a separate commit, so any step can be
  reverted without losing the rest.
- `Code`/quote unification with the existing reader prefixes must avoid
  double-handling: at the expression level `quote` is new; the macro-Lisp handling
  is removed only in step 6.

## What it unlocks (recap)

One language for compute + generate; typed reflection inside code generation
(trait-bound-aware `derive`, recursion into nested types, layout-aware codegen);
compute-then-generate (perfect-hash/state-machine generators); multi-stage
programming / partial evaluation as a library; `Code` as data for code-transform
libraries written in Coil.
