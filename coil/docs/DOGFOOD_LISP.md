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

The dogfood still stores `Value` straight in the env now (by value) — the fix
makes the natural idiom work. (`examples/lisp.coil` keeps it by value to exercise
the fix; `(ptr Value)` would also be fine and lets frames share a binding.)

## Two ergonomics the dogfood drove out

**A runtime reader, as a library — `lib/sexp.coil`.** The first cut hand-rolled
~50 lines of tokenizing/parsing. That's the part *every* Lisp/config/serialization
format rewrites, so it's now a library: `read-all a "(…) (…)"` → an
`ArrayList<(ptr Sexp)>`, plus accessors (`sexp-num`/`sexp-sym`/`sexp-count`/
`sexp-nth`/`sexp-sym-is`) so a consumer never matches the representation. The Lisp
dropped its whole reader for one import + `read-all`. (Coil's *compiler* reader is
in Rust and can't be a zero-cost runtime primitive without a runtime lib + FFI;
a Coil-level library is the in-spirit "grow via libraries" answer.)

**A `case` macro — `lib/control.coil`.** The builtin dispatch was a tall
`(if (sexp-sym-is head "+") … (if (sexp-sym-is head "-") …))` ladder. `case` flattens
it, Clojure-style (flat `key body` pairs, lone trailing = default):

```
(case head sexp-sym-is
  "+" (VNum (iadd …))
  "-" (VNum (isub …))
  …
  (call-closure …))            ; default
```

Coil has **no universal runtime `=`** — equality is type-specific (`icmp-eq` for
ints, `str-eq` for slices, per-type structural for aggregates) because values are
unboxed with no RTTI, and there's no trait/typeclass resolution. So `case` takes
the **equality explicitly** as its second argument — `icmp-eq`, `str-eq`, or a
domain predicate like the Lisp's `sexp-sym-is`. (That predicate-as-argument form
is the right call precisely because the matched thing here is a `(ptr Sexp)`, not
a plain int/slice — auto-picking a comparison from the key literals couldn't know
to use `sexp-sym-is`.) Honest and zero-magic.
