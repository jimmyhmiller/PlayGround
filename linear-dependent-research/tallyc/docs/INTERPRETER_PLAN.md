# The interpreter dogfood — scope (build on (A) sign-off)

**STATUS: SCOPED, build held for the (A) heap-recursion verdict** (the interpreter STACKS on
`Term::Case`/`Fix`; cheaper to fix a flaw before `eval` is built on it). The goal: write a
genuinely complicated program — an interpreter — in surface Tally, with the linear-dependent
types + manual memory, and let FRICTION drive what to build next (especially the view layer).

## 1. The owned AST (the data — positivity (a) already handles it)

A recursive structure of OWNED pointers — the bread-and-butter that (a) variance-aware
positivity was built for (`Own Expr` is a positive occurrence):

```
enum Expr {
  lit  : Nat -> Expr,
  add  : Own Expr -> Own Expr -> Expr,
  mul  : Own Expr -> Own Expr -> Expr,
  var  : Nat -> Expr,                    -- a variable (name as a Nat / de Bruijn)
  let_ : Own Expr -> Own Expr -> Expr,   -- let x = e1 in e2  (e2 sees x)
}
```

Each recursive child is `Own Expr` (a heap node). Strict positivity accepts it (a); each node
is allocated with `alloc`/`box`.

## 2. `eval` (the engine — (A) heap recursion makes it RUN)

```
eval : Env -> (1 e : Own Expr) -> Nat
```

`%partial` recursion on the owned AST (via `Case` + `Fix`, the (A) capability), CONSUMING the
tree as it evaluates — `unbox` each node, free it, recurse on the owned children, combine.
Each recursive result is `let`-SEQUENCED (the (A) discipline — `let va = eval(env, a); …`,
so the linear consumption is counted once, not ω-scaled into the combiner's argument). For
`add(a,b)`: `let va = eval(env,a); let vb = eval(env,b); plus(va, vb)`. So a one-shot eval
that builds, walks, and FREES the whole tree, memory-safe, zero-GC — the differentiator,
applied.

This is the MVP and it needs NOTHING new beyond (A): every sub-expression is evaluated EXACTLY
ONCE, so `unbox`-consume (free-as-you-go) is exactly right.

## 3. The environment (name → value)

For the arithmetic MVP, values are `Nat` (COPYABLE), so the env is an ω (non-linear) map:
the simplest is an assoc-list `List (Pair Nat Nat)` (a boxed, copyable list) with a recursive
lookup; `var n` looks `n` up. Copyable values ⇒ the env is freely reused — no linearity
pressure yet. (The env becomes LINEAR only when values can be OWNED — see §5.)

## 4. Build order (each step a runnable milestone)

- **MVP-0 — arithmetic, no env:** `lit`/`add`/`mul`; `eval` builds + walks + frees a tree,
  e.g. `eval(add(lit 1, mul(lit 2, lit 3))) = 7`. Proves `eval` RUNS on the owned AST.
- **MVP-1 — variables + env:** add `var`/`let_` + the copyable assoc-list env + lookup.
  Proves the env and scoping.
- **Then let friction decide** (§5).

## 5. WHERE THE VIEW LAYER GETS FORCED (the key friction — Leader's emphasis)

`unbox`-consume works iff each sub-expression is evaluated EXACTLY ONCE. The view layer
(non-consuming read, Phase C) is FORCED the moment a sub-expression must be evaluated MORE
THAN ONCE — because once you `unbox` a node you've FREED it and cannot re-walk it:

- **Functions / closures** — a closure captures an `Own Expr` body; applying it N times
  re-evaluates the SAME body N times. You cannot `unbox`-consume the body on the first call.
  `eval` must READ the body without consuming it ⇒ a non-consuming borrow `&(Own Expr)` with
  a read-back token (the design's §2 read-back, deferred to Phase C). ALSO forces a LINEAR
  env (a closure is an OWNED value → the env now maps to owned values → linear name→value map).
- **Loops** (`while`/recursion in the OBJECT language) — re-evaluate the loop body each
  iteration ⇒ same non-consuming-read need.
- **NOT forced by:** `if`/conditionals (eval one branch, FREE the other unused branch —
  the dead-arm-free discipline, already works); `let x=e1 in e2` with x used many times (x's
  VALUE is a copyable `Nat` in the env — the EXPR `e1` is still evaluated once).

So the precise friction: **re-evaluation of an expression** (functions, loops). The MVP
(one-shot eval) deliberately surfaces it — add a function-call or a loop and `eval` will
demand `&(Own Expr)` (read without consume). That is when we build the MINIMAL read-back view
(scoped to exactly that need, per the Leader's earlier steer), NOT the full Phase-C view
layer speculatively.

## 6. Soundness bar (every new piece, as the foundation got)

- `eval` is `%partial` (an interpreter SHOULD diverge on a divergent object program — totality
  on `eval` would be WRONG). Linearity STILL enforced: every `Own Expr` node freed exactly
  once on every path (incl. the unused `if` branch); a leak/double-free in `eval` is REJECTED.
- The view layer, when forced, gets its OWN maximal-bar review — non-consuming reads are
  where aliasing / use-after-free / a borrow outliving its source hide (the Leader's flag).
  The read-back TOKEN threading is the soundness crux (the borrow must be returned; no two
  live mutable views).

## Path: (A) sign-off → MVP-0 (eval arithmetic owned AST, runs) → MVP-1 (var + env) →
## first re-evaluation construct → the MINIMAL read-back view (Phase C, friction-driven).
