# Partial evaluation — specialising interpreters

tallyc's answer to Idris's [specialising interpreters](https://docs.idris-lang.org/en/latest/reference/partial-evaluation.html):
a recursive function applied to a **static constructor tree** (an interpreter
applied to a fixed program, a matcher applied to a fixed pattern, …) is
automatically specialised into straight-line residual code — the recursion, the
case dispatch, and the tree's `alloc`/`unbox` all vanish.

```
tally run examples/pe_interp.tal          → 33
tally dump examples/pe_interp.tal run     → \x. 1 + mul(2, x)    (the residual)
TALLY_PE_LOG=1 tally check <file>         → logs each `[pe] specialised <name>`
```

For `eval(prog(), x)` — a first-order interpreter over an owned `Expr` tree, at a
dynamic input `x` — the specialised `run` drops from **9 mallocs + tag dispatch**
to **0 mallocs, straight-line arithmetic**, byte-for-byte the hand-written C.

## The idea: PE is NbE with the dynamic input left symbolic

The kernel already normalises by evaluation (`eval` + `quote`) — it reduces
`Vec (2+2)` to `Vec 4` at the type level. Partial evaluation is the *same*
machinery pointed at runtime code: evaluate the function with the dynamic
arguments as **neutral variables**; everything static reduces; `quote`
residualises the result. The dynamic `x` stays symbolic and comes out as
`Var(0)` in the residual.

The kernel's `eval` deliberately withholds exactly two reductions (to keep
type-checking terminating and the memory layer opaque). The PE evaluator
(`peval` in `src/dep.rs`) adds them back:

1. **`Fix` unfolds.** A recursive function unfolds against its static argument
   (via a `VLamNative` self-closure). Termination is driven by the static tree
   shrinking, with a fuel backstop (2M steps) for any mis-fire.
2. **`unbox (alloc v) ⟶ v`.** The memory-layer β-rule: a statically-built owned
   tree is read through at compile time, so its `alloc`/`free` never reach the
   residual.

Both are meaning-preserving conversions (the fixpoint law and the intended
memory semantics), so the residual is genuinely equal to the original.

## Trigger and safety

The pass (`pe_reduce_body`) walks each elaborated def body and fires wherever a
`Fix` is applied to a **constructor/`alloc`-headed** static argument (a real
inductive structure — an AST, list, tree) plus at least one dynamic argument. A
bare `Nat` loop-counter is deliberately excluded: a counting recursion is not
bounded by it and would diverge (the fuel would catch it, but the type guard
avoids the wasted attempt).

The pass is **untrusted and self-verifying**, in the same discipline as every
other elaboration step:

- each rewritten body is **re-checked by the kernel** against the def's original
  type, and
- **reverted to the original on any failure**.

So PE can only make a well-typed program faster; it can never change what a
program means or reject a valid one. A bug in `peval` yields a residual that
fails the re-check and is dropped — never an unsound accepted program. This is
why the pass can run automatically on every program with no annotation.

## Higher order (defunctionalized)

A lambda-calculus interpreter with first-class functions — written in a language
with **no surface lambdas** — via a defunctionalized closure `Val = IntV | CloV
(body, env)` and mutually-recursive `eval`/`apply` over a runtime environment
(`examples/pe_higher_order.tal`). PE handles the mutual recursion, builds and
applies closures at compile time, and sees through an environment list
(`ECons(IntV(x), ENil)` — static list *spine*, dynamic *leaf* `x`). It removes
the interpretive **control** overhead (recursion, AST + closure dispatch); a
little value-boxing (the untyped `Val` union) remains, removable with a
dependently-typed tagless interpreter.

Two engine properties make this work: the pass threads binder **depth** and
specialises via NbE over the enclosing context as neutrals (so open static
arguments — dynamic leaves — are handled), and the trigger fires on static
**structure** (constructor-headed), not fully-static.

## Binding-time via erasure (`0`)

The program argument can be **erased** (`(0 e : Expr)`) — a pure compile-time
value with no runtime existence (`examples/pe_erased.tal`). Matching on it would
normally be `1 ⋢ 0`; tallyc accepts it because **`0` on the argument IS the
static/dynamic binding-time annotation**: the function is a *specialisation
template* (checked with its erased parameters bumped to available), PE runs
before the usage check and specialises every use away, and erasure then
*guarantees* it happened — an interpreter over a program that is not statically
known is **rejected** (you cannot dispatch on data that does not exist at
runtime). Stronger than Idris's `%static` (a hint you can ignore).

## Recursion over dynamic data (online termination)

When an object program recurses over *runtime* data (`examples`/gate
`pe_preserves_recursion_over_dynamic_data`: `run(x) = sumTo(x) + 1`), the
interpreter dispatch is specialised away but the genuine dynamic recursion is
**preserved as a residual call** to the recursive function — not inlined
forever. The rule (online PE): `fix_value` unfolds a recursion only when its
argument is a static constructor (incl. `alloc`-of-constructor for owned ASTs);
on a dynamic argument it residualises `App(Fix, arg)`. This ties the recursive
knot to the *existing* definition.

## What it does not do yet

- **Fresh residual recursive definitions (full memoisation).** Recursion is
  preserved as a call to the *original* function, and "static-config + dynamic-
  recursion" specialisation (e.g. `pow(3, n)` ⟶ a fresh `pow_3`) is not
  generated — the automatic trigger deliberately skips a bare-`Nat` config to
  stay safe; enabling it safely needs a memo table + fresh-definition generation
  (the classic online-PE knot-tying to *new* defs). Such cases fall back to the
  generic recursive function today (correct, just not specialised).
- **Value-representation overhead in higher-order residuals.** The untyped `Val`
  union leaves a little boundary boxing; a dependently-typed tagless interpreter
  (values unboxed by type) removes it.

## Key code

- `src/dep.rs`: `peval` / `pvapp` / `pvcase` / `fix_value` (the PE evaluator;
  `fix_value` unfolds on static constructors, residualises on dynamic args);
  `pe_reduce` (the depth-threaded pass); `is_static_data_root` (the trigger);
  `bump_leading_zero_pis` / `bump_template_body` (the erased-template check).
- `src/rust_surface.rs` `check_program`: runs PE **before** the usage check,
  with re-check-and-fallback and the erased-template dispensation.
- Examples: `pe_interp.tal`, `pe_erased.tal`, `pe_higher_order.tal`. Gates in
  `src/dep_codegen.rs`: `pe_specialises_interpreter_to_straight_line_code`,
  `pe_end_to_end_interpreter_overhead_removed`,
  `pe_erased_program_is_a_specialisation_template`,
  `pe_specialises_higher_order_defunctionalized_interpreter`,
  `pe_preserves_recursion_over_dynamic_data`.
