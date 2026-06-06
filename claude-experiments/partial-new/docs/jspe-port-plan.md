# Porting the partial evaluator to JavaScript — composable plan

Goal: a capable partial evaluator written in JS (`jspe`), strong enough to turn
real interpreters (Brainfuck, …) into compilers — and, eventually, to self-apply
(the third Futamura projection). Staged: **Stage A** = capability (port RustPE's
core, verified against it); **Stage B** = self-application.

## The thing that makes this safe: a perfect oracle

RustPE already works, is fuzz-verified, and specializes the **same language** a JS
port would. So every module an agent writes is checked by **differential testing
against the Rust original**: take a random program `P` in the object subset, get
`RustPE(P)` and `jspe(P)`, run both residuals on random inputs, compare outputs.
Divergence = a bug, found automatically. This reuses the existing fuzzer's program
generator. **Build the differential harness first**; it gates every module.

## Architecture (mirrors RustPE: generic engine + client)

```
ir.js        residual IR + emit(program)->JS            [no deps]
state.js     Abs values, State, heap, gc, canonicalize  [no deps]
engine.js    worklist/memo/whistle driver (port engine.rs) [no deps; client injected]
lower.js     object-language source -> Instr[]          [extends parse.js]
step/        the client stepper (the bulk):
  arith.js   push/load/store, Bin (fold|residualize), cmp, unary
  mem.js     New{Object,Array}, Get/Set {Prop,Index}, partial-escape/materialize
  control.js Jmp, JmpIfFalsy (decide|Branch), loop-head handling
  call.js    Call (inline|residual fn), Ret, frames
whistle.js   whistle, generalize, loopDiverges, materialize, loop analyses
harness.js   differential fuzzer vs RustPE (THE verification backbone)
```

### Module interfaces (freeze these before Wave 2 so agents don't collide)

- **`ir.js`**: constructors for `RExpr | Op | Cond | Block | Terminator | Program`;
  `emitProgram(program) -> string` (valid JS). Pure.
- **`state.js`**: `Abs` (`Num|Str|Bool|Undef|Null|Ref|Dyn`); `State {frames, heap,
  nextAddr, ...}`; `alloc`, `gc`, `weight`; `canonicalize(state) -> State`;
  `keyOf(state) -> string` (canonical, for the memo). No engine/step deps.
- **`engine.js`**: `specialize(client, initState) -> Program`. `client = {key,
  point, step, whistle, generalize}`. Direct port of `engine.rs`.
- **`lower.js`**: `lower(source) -> {funcs, code: Instr[], entries, ...}`. `Instr`
  is the bytecode the stepper runs (define the enum here; it is the contract
  between `lower` and `step`).
- **`step/*`**: each exports handlers keyed by `Instr` tag; `step(state, out,
  atEntry) -> Step` dispatches. Shared contract = the `Instr` enum (`lower.js`) +
  the `Abs`/`State` API (`state.js`).
- **`whistle.js`**: `whistle(seen, cand)`, `generalize(seen, from, out)`,
  `materialize(state, slots, out)`. Consumes loop analyses computed in `lower.js`.

## Scope — port the CORE, skip the rest

INCLUDE (enough for real interpreters incl. BF): integer/string/bool arithmetic &
comparisons; `if`/`while`; `var` + assignment (MUTABLE store); arrays & objects as
**partially-static** data (alloc, indexed/keyed get & set, escape/materialize);
function calls (inline + residual-function generation); the whistle + canonical
key + generalization.

SKIP for now: `TextDecoder`/`Uint8Array`/host builtins, the simple.js-deob-specific
closure residualization, `try`/`catch` (defer — add later if an interpreter needs
exception-based control flow). Keep the object subset clean and documented.

## Build waves (parallelizable)

- **Wave 0 (me):** write `harness.js` skeleton — random-program generator (port
  `fuzz.rs`'s generator), spawn RustPE for the oracle, diff residual outputs. Also
  freeze the `Instr` enum and the `Abs`/`State`/IR interfaces. This is the spec
  agents code against.
- **Wave 1 (parallel agents):** `ir.js`, `state.js`, `engine.js`, `lower.js`.
  Independent. Each ships with unit tests + a stub `step` so the harness can run a
  trivial end-to-end (e.g. constant folding) the moment Wave 1 lands.
- **Wave 2 (parallel agents):** `step/arith.js`, `step/mem.js`, `step/control.js`,
  `step/call.js`. Share frozen interfaces. Each gated by the differential harness
  restricted to programs using that construct family (+ earlier families).
- **Wave 3 (1 agent, supervised):** `whistle.js` — termination/generalization, the
  subtle part. Gated by loop programs tying the same way as RustPE.
- **Milestone A-done:** `jspe` matches RustPE on the full fuzzer (0 divergences),
  and `jspe(bf_interpreter, bf_program)` compiles Brainfuck. Then derive a BF
  compiler via the projection: `RustPE(jspe, bf_interp)`.

## Stage B — self-application (P3)

P3 = `mix(mix, mix)` = a compiler generator. It needs `jspe` to take **itself** as
input, which requires:

> `jspe` must be written in the subset `jspe` specializes.

Two paths (decide after Stage A, when we know how heavy the online state is):

- **B1 — discipline the port.** Keep `jspe.js` within the object subset it
  supports (mutable vars, arrays, `while`, calls — yes; closures-as-data, host
  builtins — no). If the subset is rich enough (it will be post-Stage-A) and
  `jspe` stays inside it, then `jspe(jspe, int)` typechecks and `RustPE(jspe, jspe)`
  → a cogen.
- **B2 — offline PE on the proven pieces.** If the online, heap-based design proves
  too heavy to specialize cleanly (likely — the literature self-applies *offline*
  binding-time-analysis PEs precisely because they're simpler to specialize), build
  a second, small offline PE reusing `ir.js`/`engine.js`/`lower.js`. This is the
  textbook self-applicable design (mix/Similix).

Verification for Stage B: `cogen(int)` must produce the same compiler `jspe(int)`
produces; check structurally + differentially.

## Honest risk read

- **Stage A is low-risk, high-parallelism.** It's a large but mechanical
  translation with a golden oracle at every step. Weeks with agents, not months.
- **Stage B is the research part.** Real uncertainty; the online→self-application
  path may push us to B2. But B1/B2 both stand on Stage A's proven IR/engine, so
  Stage A is never wasted.

## Through-line: the oracle is everything

Stage A oracle = RustPE (differential fuzz). Stage B oracle = `jspe(int)` for the
cogen. Never write a module without its differential check; that discipline is
what made the Rust side sound, and it's what makes farming modules to agents safe.
