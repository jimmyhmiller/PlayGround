# `SIGBUS` in code emission: recursive cycle through an `alloc-static` singleton that interns a sum variant with a struct payload

## Fixed

This was the same over-broad metaprogram closure: retaining every application impl
pulled `Eq Ty`, its helpers, and the otherwise-unreachable singleton cycle into the
compiler's macro engine. The closure now keeps impls by reachable owner module and
iterates their ordinary-function dependencies to a fixpoint. `check`, LLVM `emit-ir`,
and ARM64 `build` are regression-gated in `selfhost/oracle/gate-cli.sh`.

Reproduced with the `coil` on PATH, macOS arm64, 2026-07-29. Deterministic: 5 out of 5 runs.

```sh
coil check    docs/repro/sigbus-emit-ir-singleton-cycle/crash.coil   # exit 0
coil dump-mono docs/repro/sigbus-emit-ir-singleton-cycle/crash.coil  # exit 0
coil emit-ir  docs/repro/sigbus-emit-ir-singleton-cycle/crash.coil   # Bus error: 10
coil build    docs/repro/sigbus-emit-ir-singleton-cycle/crash.coil -o /tmp/x                  # Bus error: 10
coil build    docs/repro/sigbus-emit-ir-singleton-cycle/crash.coil -o /tmp/x --backend arm64  # Bus error: 10
```

Notes on where it is:

- `check` and `dump-mono` both succeed, so it is after monomorphization.
- **Both backends die**, so it is in something they share rather than in LLVM emission or in
  `codegen_a64` specifically.
- `emit-ir` writes zero lines before dying, so there is no partial output to point at a
  function.
- Raising the stack (`ulimit -s 65000`) does not help, so it is probably not simple
  unbounded recursion, though I did not confirm that.

## The trigger

`crash.coil` has a lazily initialised global singleton in the usual idiom:

```clojure
(defn types [] (-> (ptr Types))
  (let [p (alloc-static Types)]
    (if (load (field p inited)) 0
      (do (store! (field p inited) true)
          (store! (field p tab)    (al-new [Ty] (malloc-allocator)))
          …
          (ty-seed p)          ; <-- seeding re-enters `types`
          0))
    p))
```

`ty-seed` interns the well-known types, and interning calls `types` again. That cycle is what
crashes, and it needs the interned value to be a **sum variant carrying a struct payload**.

Four variants pin it down. All four are byte-identical to `crash.coil` except as noted:

| file | change | result |
|---|---|---|
| `crash.coil` | none | **SIGBUS** |
| `crash-minimal-seed.coil` | `ty-seed` body reduced to `(t-kinds 1023) 0`, which is one `(ty-intern (TVal (val-make …)))` | **SIGBUS** |
| `nocrash-seed-interns-payloadless.coil` | `ty-seed` body reduced to `(ty-intern (TBot)) 0`, a variant with **no payload** | compiles |
| `nocrash-seed-body-stubbed.coil` | `ty-seed` body is `0`, so no cycle | compiles |
| `nocrash-no-seed-call.coil` | `types` no longer calls `ty-seed`, so no cycle | compiles |

So the necessary conditions are, together:

1. a recursive cycle that passes through the `alloc-static` accessor, and
2. the cycle constructs and interns a `defsum` variant whose payload is a **struct**
   (`(TVal [(v Val)])`), not a payload-free variant.

Things that are **not** required (each was removed individually and it still crashed): the
dual-installation logic, `al-set!` on a second `ArrayList`, and the nested
`ty-xdual` -> `ty-tuple-of-duals` -> `ty-dual` -> `types` sub-cycle.

## What I could not shrink

I could not reproduce it by *growing* a small file. `works`-style probes with the same
ingredients (a 4-variant sum with a 3-field struct payload, `ArrayList` + `HashMap` in a
lazily-initialised singleton, a `seed` in the cycle interning a `TVal`, and keyops callbacks
that re-enter the singleton) all compile fine. So some further ingredient in `crash.coil`
matters that I have not identified. `crash.coil` is 537 lines and is the smallest reproducer
I have; the variant table above is the sharpest characterisation.

`crash.coil` is a truncated copy of the type lattice from
`claude-experiments/aot-kit-gradual/src/ty.coil`, cut at the line where the printing code
begins and given a trivial `main`, which is why the module is named `cutmod`.
