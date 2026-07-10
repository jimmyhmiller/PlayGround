# Codegen axes

The map of things a native code generator must let a *strategy* control, so that
swapping a strategy stays free (write one impl, change nothing downstream). Value
representation and GC are the first two of a coupled set; this catalogs the rest.

## The unifying principle

Every operation the interpreter routes through an interface is a codegen axis.
The interpreter needs the **compute** form (run now on a value); the JIT needs the
**emit** form (append machine instructions that will run later). One strategy owns
both, and the two forms are a differential spec: run interpreter and JIT on the
same inputs, compare, and a mismatch is a bug in the strategy.

```
compute:  fn op(&self, v: u64) -> u64;                 // interpreter tier
emit:     fn emit_op(&self, b: &mut Builder, v: Val) -> Val;   // native tier
```

Corollary: **the map of axes is the map of runtime hooks the mutator funnels
through.** If the interpreter calls a hook to do it, the JIT must emit it, and
different strategies emit it differently. So to find the axes, list the runtime
operations. In microlang those are `prim` (arithmetic), `invoke` (calls),
`alloc`/`collect`/cell access (GC), global lookup, and the `(gc)` safepoint —
already the core of this list.

Non-negotiable consequence: the mutator must route **every** such operation
through the interface from day one. You cannot retrofit a write barrier, a
dispatch cache, or a tag check into code that pokes memory directly, any more
than you can retrofit `Repr` into code that pokes bits.

---

## The axes

### Data plane — how values and memory are shaped

**1. Value representation** (`Repr`, built). Tag/untag, box/unbox, immediacy.
- Strategies: low-bit tag, NaN-box, high-bit tag, boxed-everything.
- Emit: `emit_tag_of`, `emit_box_int`, `emit_unbox_float`, `emit_is_immediate`.
- Optimizer leak: *representation selection* — the JIT picks a machine rep per SSA
  value (unboxed i64/f64, tagged, raw pointer) and wants to keep values unboxed
  across whole loops; redundant-guard elimination needs to know immediacy facts.

**2. Numeric / overflow policy.** What `+` does at the boundary.
- Strategies: wrap (fixed 64-bit), promote to bignum (Clojure `+`, Python),
  trap/error, saturate. Note a language can offer several (`+` vs `unchecked-add`).
- Emit: bare add, or add + overflow guard + slow path to bignum promotion.
- Coupling: promotion needs a bignum heap type → value rep + GC.

**3. Aggregate representation + safety checks.** Strings, arrays, the persistent
collection layout, and the checks around access.
- Strategies: UTF-8 vs UTF-16 vs rope; array vs HAMT vs RRB; bounds-check
  explicit-branch vs implicit-via-signal (protect a guard page); nil-check
  explicit vs implicit.
- Emit: `emit_element_load`, `emit_bounds_check`, `emit_null_check`, `emit_len`.
- Optimizer leak: bounds-check hoisting/elimination, nil-check folding.

**4. Object model / header.** The bits every heap object carries.
- Strategies: where the type/shape id lives, hash slot, mark/forwarding bits,
  lock/monitor word, alignment, field offsets (packed vs slotted).
- Emit: `emit_type_of`, `emit_field_offset`, `emit_shape_check`.
- Coupling: type-id location ↔ dispatch; lock word ↔ concurrency; mark/forward
  bits ↔ GC.

### Memory management

**5. GC + allocation + barriers + roots** (moving collector built; barriers/roots
partial). See `gc.rs` and the README GC section.
- Strategies: STW copying (built), mark-sweep, generational, concurrent-copy, RC.
- Compute: `alloc`, `read_field`, `write_field`, `collect`.
- Emit: `emit_alloc` (inline bump-pointer vs call), `emit_write_barrier`
  (no-op / card-mark / SATB), `emit_read_barrier` (no-op / forwarding resolve),
  `emit_safepoint_poll`.
- Root protocol is its own sub-axis: precise stack maps vs shadow stack vs
  conservative scan.
- Barriers are the price of generational partitioning and concurrency; STW
  non-generational needs none.

### Control plane — how execution flows

**6. Calling convention + closures + tail calls.**
- Strategies: register/stack arg passing, closure-invoke (self ptr + captured
  env), fixed vs variadic vs multi-arity dispatch, and tail calls: proper TCO
  (jump reusing the frame) vs trampoline (Clojure `recur`) vs none (Python).
- Emit: `emit_call`, `emit_prologue` (bind params/captures), `emit_tailcall`.
- Coupling: frame strategy (TCO reuses the frame; root maps must survive it).

**7. Dispatch + inline caches.** The heart of dynamic-language performance.
- Strategies: monomorphic IC, polymorphic IC, megamorphic dictionary/vtable,
  prototype chain (JS), class + vtable (Ruby/Python), protocol dispatch (Clojure),
  multimethod. Or guard-and-deopt.
- Emit: `emit_dispatch` = shape/type guard + cached target + inline body + slow
  path + cache-fill stub.
- Coupling: value rep (the tag test is the first guard), object model (where the
  shape id is), deopt (megamorphic falls back), type feedback (fills the cache).

**8. Speculation + deoptimization + OSR.**
- Strategies: what to speculate (this is always an int / this site is
  monomorphic); guard-on-failure to a lower tier; on-stack replacement to enter
  optimized code mid-loop.
- Emit: `emit_guard`, plus deopt metadata mapping SSA values → interpreter frame
  state.
- Coupling: **requires an interpreter tier as the deopt target** (this is why
  `CodeSpace` is interpreter-first); frame strategy (reconstruct the frame); GC
  (every live value at a deopt point must be a root, with correct pointer-ness).

**9. Exceptions / continuations / effects.** Non-local control.
- Strategies: result-propagation (control-aware calls — what the parent toolkit's
  `abort_to_prompt` and microlang's earlier `throw` did), table-based zero-cost
  unwinding, setjmp/longjmp, stack copy/segment for continuations (one-shot vs
  multi-shot), CPS, algebraic effect handlers.
- Emit: `emit_throw`, `emit_landingpad`, `emit_prompt_install`, `emit_capture`,
  `emit_resume`.
- Coupling: frame strategy (stack copying needs heap-relocatable frames), calling
  convention (how the non-normal outcome propagates across a call).

**10. Concurrency + memory model + safepoints.**
- Strategies: single-threaded, shared-memory + atomics, actor/message. Safepoint
  polling: page-protect trap vs counter vs loop-back-edge poll.
- Emit: `emit_atomic`, `emit_fence`, `emit_cas`, `emit_safepoint`.
- Coupling: concurrent GC → barriers + all-thread safepoints; memory model →
  fence placement.

### Binding / environment

**11. Global / Var access.**
- Strategies: direct pointer (early binding), indirection cell (Clojure Vars,
  redefinable at runtime), dictionary lookup (Python globals), inline cache.
- Emit: `emit_global_ref` / `emit_var_deref`.
- Coupling: incremental compilation + late binding (call a fn defined later —
  the `ClosureComp` late-binding property); dispatch (a Var deref can be cached).

**12. Frame / stack layout + root placement.**
- Strategies: native-stack vs heap frames (for continuations/coroutines), spill
  slots, GC roots via stack map vs shadow stack (microlang uses a shadow stack),
  segmented/growable stacks for deep recursion or green threads.
- Emit: `emit_prologue`/`emit_epilogue`, root-slot placement, frame-size patch
  (the parent toolkit's `emit_frame_size_patch` + frame-zeroing lived here).
- Coupling: GC roots, tail calls, continuations, deopt all read this.

**13. FFI boundary.**
- Strategies: marshalling values across the C ABI, un/boxing at the edge,
  callback trampolines, GC transitions (native code can't hold moving pointers).
- Emit: `emit_ffi_call` + arg/ret marshalling per the C ABI and value model.
- Coupling: value rep, calling convention, GC (pin/transition across the call).

### Policy / meta

**14. Tiering + profiling + type feedback.** Controls *what* is emitted.
- Strategies: interpreter-only, baseline + optimizing, profile-guided; when to
  compile, when to reopt/deopt.
- Emit: profiling hooks in lower tiers (invocation/back-edge counters, type
  feedback collection at call/access sites).
- Coupling: deopt (shared metadata), dispatch (feedback fills ICs), `CodeSpace`
  (the tiers themselves).

---

## The coupling graph (the actually-hard part)

These are not independent knobs. The design difficulty is the constraint web, not
any single interface. The load-bearing edges:

```
moving GC ─── requires ──▶ precise roots ─── needs ──▶ frame strategy emits stack maps / shadow stack
     │                                                        │
     └── objects relocate ──▶ deopt & safepoints must know exactly which words are pointers
                                          │
value representation ── must make pointer-ness decidable ─────┘

concurrent GC ──▶ write + read barriers ──▶ safepoint polling ──▶ thread model ──▶ memory-model fences

speculation / deopt ──▶ interpreter tier exists (CodeSpace) + frame-state map + GC-roots the deopt live set

overflow = promote ──▶ bignum heap type ──▶ GC + value representation

dispatch IC ──▶ value-rep tag as first guard + object-model shape id + deopt on megamorphic + type feedback

tail calls ──▶ calling convention + frame reuse — and can conflict with stack-based root maps
```

So the toolkit's real job is not just exposing each emit-interface. It is a
**coherence layer**: valid combinations constructible, incoherent ones a compile
error. Examples of combinations that must NOT typecheck:

- moving GC + conservative stack scan (can't rewrite an ambiguous root)
- concurrent GC without barriers
- speculation without an interpreter tier to deopt into
- proper TCO with a root protocol that assumes a growing native stack

This is the `dynexec::SoundRoots` sealed-trait pattern generalized: an
`ExecutionConfig` bundle whose constructor rejects incoherent tuples. The parent
toolkit already shipped `CallingConvention`, `RootStrategy`, `FrameStrategy`,
`SafepointStrategy`, `CodegenConfig`, and `SoundRoots` — it identified axes
6, 5(roots), 12, 10(safepoints) and the coherence idea. The additions here are
the *emit* half of each, plus dispatch (7), deopt (8), control flow (9), and the
data-plane axes (2, 3, 4), plus drawing the full coupling graph.

---

## The meta-discipline (same three rules as value and GC)

1. **Compute-form + emit-form, one strategy owns both.** The two are a
   differential spec that cross-checks the strategy.
2. **Route every operation through the interface from day one.** Barriers,
   dispatch caches, tag checks, safepoints cannot be retrofitted.
3. **Two real strategies per axis or it rots**, and enforce the couplings at the
   type level. A single strategy leaves the interface untested exactly where a
   second one would break it — the original toolkit's failure mode.

---

## Where microlang stands

Grounding, honestly. microlang's interpreter already funnels through the core
hooks, so the axis map is visible in its code:

- Value rep (1): `Repr`, three real strategies; emit-half built BOTH as bytecode
  (`ModelEmit`) and as native code (`jit_cranelift.rs` lowers the same recipe).
- Overflow (2): the interpreter promotes to BigInt; the native JIT emits the
  guarded emit form — fixnum fast path + range check + fall back to the promoting
  `prim` (`jit_cranelift.rs`, `emit_guarded_arith`), so compiled code has the full
  tower too. (The bytecode tier still wraps — the emit form lives only in the JIT.)
- GC (5): moving collector built; roots via shadow stack; **barriers are no-ops
  because the GC is STW non-generational** — correct, but the barrier interface
  isn't a hook yet.
- Calling convention (6): the interpreter and the JIT do proper tail calls (a
  trampoline reusing the frame); the JIT emits a tail `Call` as a trampoline
  bounce, so unbounded tail recursion is O(1) native stack. The bytecode and
  closure tiers still recurse. No non-tail frame reuse / explicit `recur`.
- Frame/roots (12): shadow stack + `Cell` frames; no stack maps.
- Var access (11): indirection through the globals table; late binding works.
- Tiering (14): interpreter + closure-compile + bytecode + a native Cranelift JIT,
  and `Tiered` = JIT-with-CEK-fallback picking a strategy per body (the
  interpreter-fallback the deopt design needs, chosen statically by feature rather
  than by profile). No profiling-driven reopt/deopt yet.
- Dispatch (7): `dispatch.rs`, three swappable strategies (megamorphic, mono IC,
  poly IC) over records + `defmethod`; per-site caches; the dispatch⟺GC coupling
  wired (impls are roots, caches invalidate on collect). Emit-half not built.

- Speculation + deopt (8): `speculation.rs`, three swappable policies
  (never / always-monomorphic / blacklist-after-n) inside a `Speculative`
  dispatch strategy that wraps a fallback dispatch. Guard + deopt are real and
  counted; the invariant "speculation never changes results" is tested. Built as
  an adaptive dispatch strategy, not a `CodeSpace` wrapper, because a compiling
  backend inlines the dispatch node and escapes a node-level wrapper. The
  distinctive native payoff (inline-through-guard, unbox-across-guard, mid-frame
  deopt + SSA→frame state reconstruction) still needs the emit tier.

What microlang has **none** of, and would have to grow — the remaining frontier,
each where a real dynamic language spends its performance budget:

- Exceptions / continuations (9): existed in an earlier revision, removed here.
- Concurrency (10) and its barriers.

The **emit half** is real, and now at two levels. `bytecode.rs` is a bytecode
emit tier (compiler + stack VM) where the value model supplies `ModelEmit` and
the SAME source compiles to representation-specific instruction streams (LowBit
shifts, HighBit does not, NanBox boxes to slow calls). And `jit_cranelift.rs`
(opt-in `--features jit`) lifts that exact op stream to a real machine ISA: it
lowers the *same* `ModelEmit` recipe to Cranelift IR and compiles to host machine
code, proving the value-axis emit interface is ISA-neutral (one recipe →
bytecode *or* native). Adding it surfaced a latent bug the positive-only matrix
never hit — HighBit's top-bit tag corrupts under a raw *signed* subtraction whose
result is negative, so both emit tiers fail there while the re-masking
interpreter does not. That is precisely the differential-spec payoff this doc
opened with: a second emit strategy is what catches the first's silent gap.

The JIT has since grown past a demo: `let`/`set!`, proper tail calls (axis #6),
and guarded overflow→bignum arithmetic (axis #2) make it run **all of Scheme
except first-class continuations**. `Tiered` composes it with the `CekMachine`
(the interpreter fallback the tiering design always promised), and the full R7RS
conformance suite passes on that pairing — 54/61 live cases fully native,
oracle-checked against Chicken.

What remains is the emit half of the OTHER axes (a GC write barrier emitted
inline, a dispatch guard chain, a deopt point with a frame-state map) and the
first coupling to close: making native temporaries GC roots (the frame/roots
axis) so the JIT tier can model the `(gc)` safepoint the interpreter tiers do.

The through-line: value and GC were the right first two because they are the
foundation every other axis sits on. Dispatch is the right third — it is where
dynamic languages are slow, and it couples value rep, object model, deopt, and
type feedback all at once, so building it honestly would exercise the coherence
layer harder than anything so far.
