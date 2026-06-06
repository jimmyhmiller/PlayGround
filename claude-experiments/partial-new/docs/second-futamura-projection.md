# Second Futamura Projection — a compiler derived from an interpreter

`compiler = mix(peval, int)`: specialize a partial evaluator with respect to an
interpreter, get a compiler. Done for real. Artifacts in `docs/p2real/`.

## Retraction of the first attempt (`docs/p2`)

An earlier version (`docs/p2`, the `gmix` files) was **not** P2. `gmix` was a
hand-written *compiler* whose `clauses` table was a hand-written *codegen spec* —
the actual interpreter never appeared, and RustPE only inlined the table. The
interpreter→compiler step (deriving codegen from executable semantics) was done
by hand, which is the whole thing P2 is supposed to do automatically. Kept for
history; superseded by this.

## The genuine setup (all in the JS subset)

- **`int(prog, input)`** — a real interpreter for a stack/accumulator bytecode.
  It *runs* programs and *produces values*: a loop over the program, dispatch on
  opcode, arithmetic on an accumulator. Expressed as **data** in a tiny first-order
  language `L` (`docs/p2real/int_L.js`).
- **`peval`** — a real partial evaluator for `L`, written in the subset
  (`docs/p2real/peval2.js`). It is **generic**: its only codegen rule is, in
  `pevalArith`, "a primitive whose operands are all static folds; otherwise it
  residualizes to the *same* primitive." Nothing in `peval` mentions the source
  language's opcodes. So when it specializes `int`, the interpreter's `+`/`*`/`-`
  become compiled `radd`/`rmul`/`rsub` **automatically** — the codegen *falls out*.

`peval` is a correct P1 specializer on its own: `peval(INT, prog)` produces a
compiled-code AST equal to interpreting `prog`, verified over 14,000 checks, with
the codegen derived rather than hand-written.

## The projection

`RustPE(main(prog) = peval(INT, prog))` with `INT` static and `prog` **dynamic**
yields a residual that **loops over a source program, dispatches each opcode, and
emits compiled code** — i.e. a compiler:

```js
// docs/p2real/p2_peval.js  ──RustPE──▶  the derived compiler (excerpt)
while (prog.length > 0) {
  instr = prog[0];
  if (instr[0] === "addlit") { acc = ["radd", acc, ["rlit", instr[1]]]; }
  else if (instr[0] === "mullit") { acc = ["rmul", acc, ["rlit", instr[1]]]; }
  else if (instr[0] === "addin")  { acc = ["radd", acc, ["rin"]]; }
  else if (instr[0] === "subin")  { acc = ["rsub", acc, ["rin"]]; }
  prog = prog.slice(1);
}
return acc;
```

(The real residual is `docs/p2real/` → run the commands below; it is 49 blocks of
the above shape.) The opcode set is inlined, the interpreter's `L`-AST walk has
folded away, and the `radd`/`rmul` constructors are the compiled output.

**Verified two ways** (`docs/p2real/`): the derived compiler is *structurally
identical* to `peval(INT, ·)` over 3,000 random programs, and its output
*evaluates equal to `int`* over 21,000 checks.

```
compile([["addin"],["mullit",3],["addlit",1]])
   →  ["radd",["rmul",["radd",["rlit",0],["rin"]],["rlit",3]],["rlit",1]]   //  (0+in)*3+1
```

## Why it folds (the two prerequisites)

1. **Engine fixes landed this session** — test-at-top dispatch (a changed
   `drives_dispatch` slot blocks generalization) and if-without-else loop
   termination (`consume_pending_joins` at block entry) — make a dispatch loop
   over dynamic data tie cleanly.
2. **`peval` walks the program with an explicit `while` loop** (the `fold` case in
   `peval2.js`), so the loop-carried program and accumulator are *locals*. The
   growth whistle fires on loop back-edges over locals; it ties `peval`'s loop
   over the dynamic program. The first attempt used inlined recursion, which the
   engine expanded into unbounded straight-line growth the whistle never saw
   (`GEN count = 0`, runaway budget). Restructuring as a loop fixed it.

## Reproduce

```bash
# (re)generate the combined input and derive the compiler
node -e 'const fs=require("fs");let pe=fs.readFileSync("docs/p2real/peval2.js","utf8");
  const INT=require("./docs/p2real/int_L.js");
  fs.writeFileSync("docs/p2real/p2_peval.js", pe+"\nfunction main(prog){var INT="+JSON.stringify(INT)+";return peval(INT,prog);}\n");'

SPEC_WEIGHT_BUDGET=8000000 ./target/release/js-frontend --js docs/p2real/p2_peval.js 2>/dev/null \
  | sed -n '/residual as JavaScript/,/^---/p' | grep -v '^---' > /tmp/p2_compiler.js
# /tmp/p2_compiler.js : main(prog) is the derived compiler; see docs/p2real/ for the verifier
```

## Files

`docs/p2real/peval2.js` (the partial evaluator), `int_L.js` (interpreter as data),
`p2_peval.js` (input to RustPE), `PLAN.txt`.

## Real-JS version (`docs/p2real/`): interpreter written in plain JavaScript → compiler

The `peval2`/L version above uses an interpreter encoded as nested-array data. The
real-JS version closes that gap: the interpreter is written in a **reasonable JS
subset** and parsed.

- `int.js` — the interpreter in plain JS: `interp(prog,input) = prog.reduce((acc,instr)=>step(instr,acc,input), 0)` with `step` a ternary dispatch on opcode.
- `parse.js` — a small JS-subset parser (front-end): source text → AST.
- `jsmix.js` — a genuine partial evaluator **for the JS subset, written in the subset**. Generic: the `+`/`*`/`-` codegen falls out of the interpreter's operators (`pevalArith`), never hand-written. Loops via `.reduce` walked structurally.
- `RustPE(main(prog) = jsmix(INT, "interp", prog))`, `INT` static + `prog` dynamic → an **83-block compiler**: loops over the source program, dispatches each opcode (inlined), emits `["radd"/"rmul"/"rsub"]`.

Verified: derived compiler ≡ reference `jsmix` structurally over 3,000 programs, and ≡ `interp` semantically over 21,000 checks. This is `compiler = mix(jsmix, int)` for an interpreter in **real JavaScript**.

### Three compounding fixes it took (all sound, fuzz-verified)

Making a partial-evaluator-in-the-subset specialize under RustPE is the crux of
self-application; three issues stacked, each found by instrumentation:

1. **Fixed-size environment.** `jsmix`'s variable lookup `while (i < env.length)`
   runs away once the env materializes (dynamic loop bound). Fix: a fixed 8-slot
   frame scanned with a **literal** bound (`while (i < 8)`) so it unrolls even
   when materialized.
2. **Canonical heap memo key** (engine-adjacent, but in the JS client's `key()`,
   not `engine.rs`). `canonicalize_state` renumbers heap addresses in reachability
   order from the roots and drops unreachable garbage, so states that are
   heap-*isomorphic* (a PE allocates fresh objects every iteration; addresses
   drift) memo-tie. Sound, and it even improves folding elsewhere (bfdyn 11→10).
3. **Inline the reducer frame.** `jsmix` stored the per-iteration env in a
   loop-carried `var`, so it got materialized and the input code-constant
   (`["rin"]`) flickered abstract-vs-materialized, breaking the tie. Passing the
   frame inline (only `acc`/`list` loop-carried, like `peval2`) fixed it.

Gate: 124 unit tests, simple.js 13/13 + 4113 lines, fold suite unchanged, 4000
fuzz programs with 0 divergences.

## Next: P3

P3 = `mix(peval, peval)` = a compiler generator. It needs `peval` to specialize
itself. `peval` is now a real specializer in the subset and RustPE can specialize
it (that is exactly what P2 did) — so the foundation is in place; P3 is the next
target.
