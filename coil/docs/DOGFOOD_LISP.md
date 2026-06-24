# Dogfood: a mini-Lisp interpreter (`examples/lisp.coil`)

Per FUTURE_WORK §14 ("write a real program in Coil and let the friction set the
agenda"), this is a reader + tree-walking evaluator with **closures**, **`define`**,
and **recursion** — pushing past calc/json into first-class functions:

```
(define fact (lambda (n) (if (= n 0) 1 (* n (fact (- n 1)))))) (fact 10)  => 3628800
(define fib  (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) (fib 15) => 610
((lambda (x) (* x x)) 9)                                                   => 81
```

It exercises, together: a recursive `Sexp` sum (reader output) + recursive `Value`
sum (with a closure variant holding a captured `(ptr Env)`), the allocator value,
`(slice u8)` symbols + `str-eq`, a string-keyed `HashMap` per environment frame,
`ArrayList` for list/param vectors, `loop`/`break`, `match`, and recursion in both
the interpreter *and* the interpreted language.

## Friction it surfaced

**1. Generic type-arg inference leaked a `ref` from an aggregate-by-ref argument — FIXED.**
`env-define` had `(v Value)` (a sum → passed by immutable reference, so its type
is `(ref Value)`), and stored it via `hm-put!`'s generic `(v V)` slot. The checker
unified `V` against `(ref Value)` and bound `V = (ref Value)`, conflicting with the
map's `V = Value` ("conflicting types for parameter 'V' (Value vs (ref Value))").
Fix (`src/check.rs` `unify`): a type parameter is a *value* type, so peel a
top-level `ref` from the actual argument before binding it — references are erased
value-semantically. (356 tests still green.)

**2. Forwarding a by-ref aggregate into a generic by-value slot — ABI gap, FIXED.**
With (1) fixed, `(hm-put! … v)` hit an LLVM-verifier error: the monomorphized
`hm-put!` takes `Value` by value, but `env-define`'s `v` is a `(ref Value)`
(pointer), so codegen passed a pointer where a by-value aggregate was expected.
`coerce_arg`'s `Type::Struct` branch passed the place pointer for *every* caller,
expecting codegen to load+coerce — which the `extern`/C-ABI call path does, but the
ordinary monomorphized-generic call path does not.

Fix (`src/check.rs` `coerce_arg`): pass the place pointer **only** for an `extern`
(C-ABI) call; for an ordinary Coil call whose parameter is a by-value aggregate
(the only non-extern way a parameter is a bare `Type::Struct` is a generic
instantiated with an aggregate — concrete aggregate params are by-reference), fall
through to `coerce`, which loads a `(ref T)`/place to a value so the by-value
callee matches. Verified: a by-ref aggregate param now stores into a generic
`HashMap`/container directly; 356 tests green.

The dogfood still stores `(ptr Value)` in the env — now a *choice* (it lets frames
and closures share a binding), not a necessity.
