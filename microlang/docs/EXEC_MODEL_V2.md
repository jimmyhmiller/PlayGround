# Execution Model v2 — flat closures, native calling convention, arena heap

Status: IN PROGRESS. This doc is the plan of record for the performance
re-architecture. It is written so work can resume mid-stage.

## Why (measured, 2026-07)

Matched microbenchmarks (microclj `--jit`, startup-subtracted, vs warmed JVM
Clojure) and a `sample` profile of `loop`+`conj` vector-build showed:

| workload | microclj | JVM | ratio |
|---|---|---|---|
| raw loop arithmetic | ~3ns/iter | 0.7ns | ~4× |
| non-capturing defn call | ~10ns | 4.7ns | ~2× |
| capturing closure call | ~40ns | 5ns | ~8× |
| reduce+map pipeline | ~270ns/el | 9ns | ~30× |
| loop+conj vecbuild | ~2.7µs/op | 26ns | ~100× |

Profile attribution: ~40% Rust allocator (fat `Obj` enums + inner `Vec` mallocs
+ heap lock), ~25% shim-call machinery (`resolve_call`/`alloc_frame`/
`run_trampoline`/let shims), ~10% decode/tag dispatch. JIT'd code is a sliver.

Three compounding fixes, in order:

## Stage A — flat closures + register-arg calling convention

### A1. Core Ir change: closure conversion (new pass, `src/flatten.rs`)

Input: today's chain-scoped Ir (`Local{up,idx}` resolved against a frame chain;
`Let` pushes frames; `Lambda` captures the whole env; `Try` catch binds thrown
value in a fresh 1-slot frame). Both frontends (Sexpr, clojure-stub compile.rs)
produce this shape; they call `flatten::flatten(ir)` after lowering.

Output: FLAT Ir — the only shape tiers execute:
- One activation frame per call, size `nslots`, no parent chain. All
  `Local`/`SetLocal` have `up == 0`; idx is a function-level slot.
  Slot layout: `[params.., rest?, let/catch slots..]` assigned by the pass.
- `Ir::Let` is GONE from output: inits become `SetLocal{0, slot}` in a `Do`.
- `Lambda { nparams, variadic, nslots, captures: Vec<CapSrc>, body }` where
  `CapSrc::Slot(i)` = copy enclosing activation slot i at closure-creation
  time; `CapSrc::Cap(i)` = copy from enclosing closure's captures (nested).
- `Ir::Capture(u16)` reads the current closure's capture array.
- `Try` gains `cslot: u16` — catch stores the thrown value into an activation
  slot of the SAME frame (no fresh frame).
- Assignment conversion: a var that is BOTH `SetLocal`-assigned somewhere AND
  crosses a lambda boundary (referenced or assigned from an inner lambda) is
  boxed: init wraps in a cell (reuses `Obj::Atom` until Stage B gives a lean
  cell), reads become deref, writes become reset. Everything else stays a plain
  slot. Clojure emits no local `set!` (loop is a self-parameterized fn), so
  this only triggers for Scheme.
- Top-level forms that need slots are wrapped in `Call(Lambda{nparams:0,...},[])`
  so every eval_ir call site keeps working unchanged (activation built by the
  normal invoke path).

### A2. Runtime + tier migration

- `Obj::Closure { nparams, variadic, nslots, body, caps: Arc<[AtomicU64]> }`
  (captured VALUES; AtomicU64 so the moving GC forwards them in place).
- `Frame { slots: Vec<AtomicU64>, caps: Arc<[AtomicU64]> }` — parent field
  DELETED. `build_call_frame` sizes to nslots, fills params (+rest list),
  attaches caps Arc (1 atomic inc per call, no copy).
- gc.rs: `update_env` loses the chain walk; `scan_obj` Closure arm forwards
  the caps array directly. Kont walking unchanged otherwise.
- TreeWalk/ClosureComp/BytecodeVm/Cek: `Capture(i)` = read frame.caps[i];
  Lambda arm builds caps per CapSrc list; Let arms become
  unreachable-for-flat-input (kept or removed); Try uses cslot.
- Gate: `cargo test` (45-combo matrix), scheme suite, clojure oracle suite.

### A3. JIT v2 (jit_cranelift.rs rewrite of the call path)

- Fast body = no Try/Gc/Await/CallCc and no non-self tail call (same
  trampoline rule as today; Cranelift `Tail` callconv + return_call is a
  possible later upgrade). Everything else — capturing, variadic-CALLEE via
  rest-build at entry, multi-arity — is now fast-eligible.
- Entry signature (fast): `fn(rc: *mut RunCtx, closure_bits: u64,
  caps: *const u64, a0..a{k-1}) -> u64` — args in registers, k = nparams.
  Locals = Cranelift SSA variables (regalloc'd), NO heap frame, NO atomics,
  NO per-call JitCtx construction (rc is the shared per-run context).
- Fast-call table keyed by heap id: `{code, nparams|flags, caps_base}` filled
  on first slow call; call site: bounds-check id, load entry, check
  nparams==k && !variadic, `call_indirect` with the k-ary sig; else shim_call.
- Self tail call: compare callee bits == own closure_bits → refill SSA loop
  vars, branch to header (as today, now for capturing bodies too).
- Closure creation: spill capture values to a stack array,
  `shim_make_closure(rc, template_id, caps_ptr, ncaps)`.
- Slow bodies keep (a simplified version of) the current heap-frame path.
- Hot prims get monomorphic extern "C" shims with register args (no array
  spill, no giant `Runtime::prim` match on the hot path).
- Dispatch: per-site monomorphic cache inside the shim now; code-level IC
  needs Stage B's readable type headers.

## Stage B — payload word-arena (hybrid; landed design)

Status note (mid-flight): stages A1–A3 and C are DONE and committed
(flat closures; register-arg SSA JIT; speculative inlining; per-arity
MultiFn). vecbuild went 2.75µs → 290ns/op. The remaining vecbuild profile is
~55% malloc/free/memmove from the INNER `Vec` payloads of hot objects.

Hybrid design (this stage): keep the `Vec<Obj>` object table and the enum,
but move the HOT payloads into a chunked bump WORD ARENA (`Vec<Box<[u64]>>`,
stable addresses, atomic bump, no per-alloc malloc/free/lock):
  * `Obj::Vector { off, len, cap }` (cap allows in-place `%apush` growth by
    re-spanning — object identity is the Obj slot, so growth just updates
    off/cap), `Obj::Values { off, len }`, `Obj::Record { type_id, off, len }`.
  * Spans are exclusively owned by their Obj; GC's `scan_obj` copies the live
    span to the arena top and updates `off` (append-style semi-space, same
    discipline as the object heap).
  * `Str` stays `String` this round (string paths already have native bulk
    prims; revisit).
Full header-arena (objects themselves inline, readable type tags for
JIT-inline dispatch ICs + inline bump allocation) remains the follow-up.

## Stage C — var-guarded speculative inlining

At JIT-compile time, resolve `Global` callees; a small already-compiled body
inlines behind a guard (var slot still holds the same closure bits → inlined
body, else call). Depth budget. Beta-reduce direct `Call(Lambda)`. This is
what erases the per-call floor inside `-pv-conj`-style prim-doctrine code and
the transducer step-fn cluster.

## Invariants / gates

- Every stage lands green on: `cargo test`, `cargo test --features jit`,
  scheme crate suite, clojure-stub crate suite (oracle), before benchmarks.
- No stubs that silently return wrong values — placeholders throw loudly.
- Continuations: CEK stays the fallback for callcc/shift bodies (Tiered);
  flat closures must keep multi-shot continuation tests passing.
- Explicit-only GC is unchanged in Stage A/B; stack maps are future work and
  deliberately NOT required (fast bodies exclude Gc/Await, as today).
