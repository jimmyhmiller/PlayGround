# Stage E — GC stack maps FOR REAL (allocation-driven collection)

Decision (Jimmy, 2026-07-15, emphatic): no more explicit-`(gc)`-only. Native
frames get real root maps; collections are driven by allocation pressure.
This retires EXEC_MODEL_V2 gap "GC stack maps" AND the carried
caps_base/SSA-staleness gap for good.

STATUS (2026-07-15): **SHIPPED, E1–E4.** Pressure GC is ON by default
(`MICROLANG_PRESSURE_GC=0` opts out; `MICROLANG_GC_STRESS=1` collects at
every safepoint). The SP arithmetic was pinned empirically on arm64:
`slot = fp − active_size + entry_off` (a live SSA value and a live capture
both survived 9 mid-loop moving collections under the verify heap before the
formula was trusted). Notes vs the plan: the safepoint shim reads NO ctx
fields beyond `rc` (a fast-call child ctx is a minimal 3-store; its `cur` is
uninitialized memory), memory-mode frames ride the dynamic env chain
(`run_ctx_entry` pushes them), the trampoline's bounce loop polls too (the
back-edge of mutual/variadic tail loops), spawn roots the thunk in the
child's shadow before the thread exists, and the AllocWindow's limit sits at
the SOFT trigger so inline allocations drive pressure through the slow shim.
Gates: all four suites + the gc-stress battery
(`cargo test --features jit --test gc_stress -- --ignored`) green.
Measured cost: back-edge poll puts raw loop arithmetic at ~4ns/iter (was
~2); calls/captures unchanged (4/5 ns); reduce+map ~56ns/el (was 51),
vecbuild ~157ns/el (was 150).

## The key insight that makes this tractable

Collections ALREADY only happen with every thread at a safepoint (the STW
park rendezvous), and the interpreter tiers' rooting discipline (shadow
stack around Call/Prim arg evaluation, env publication at park) was already
built and tested for "another thread collects while I'm parked". So
PRESSURE-TRIGGERED collection at existing safepoints needs NO new rooting in
the Rust runtime — bare u64s held inside a prim are safe because that thread
cannot reach a safepoint mid-prim. The ONLY missing machinery is on the JIT
side: native frames neither poll nor publish their live values. That is
what Cranelift user stack maps provide.

## Cranelift 0.133 facts (verified in the vendored source)

- `FunctionBuilder::declare_value_needs_stack_map(val)` — the value is
  spilled to a sized stack slot before EVERY safepoint (= every non-tail
  call) it is live across, and RELOADED afterwards; the doc comment says
  explicitly: "between spilling and reloading, the stack can be updated to
  facilitate moving GCs". Values ≤ 16 bytes. This is exactly the moving-GC
  contract we need — no manual spill code.
- `CompiledCode::buffer.user_stack_maps() -> &[(CodeOffset, u32, UserStackMap)]`
  where CodeOffset is the RETURN ADDRESS (start of the instruction after the
  call) and the u32 is the frame's `active_size()` in bytes.
  `UserStackMap::entries() -> (ir::Type, u32)` yields SP-relative byte
  offsets of each root slot AT THAT PC.
- `preserve_frame_pointers` shared flag (default false) must be set so an
  FP-chain walk works.

## Design

### E1. Heap pressure (runtime, no JIT)
- `heap::Heap` gets a soft threshold (fraction of space, default ~50%;
  `MICROLANG_GC_TRIGGER_PCT`); `alloc` sets a `gc_pressure` flag (Relaxed
  store, checked cheaply) when the bump crosses it. Allocation itself NEVER
  collects and NEVER fails until the hard wall (unchanged loud panic).
- `Runtime::safepoint(locals)` (and the CEK step loop + `invoke` entries):
  if `gc_pressure && !gc_requested`, claim the collector role and collect
  exactly as the explicit `(gc)` prim does today (same stw_collect). The
  `(gc)` prim stays (tests, benchmarking).
- gc-stress mode (`MICROLANG_GC_STRESS=1`): pressure permanently true →
  collect at EVERY safepoint, with the verify heap armed. This is the bug
  hammer: the whole behavior matrix + scheme + clojure-stub suites must
  pass under it.
- IMPORTANT ORDERING: E1 can only be ENABLED-BY-DEFAULT after E2, because a
  pressure-triggered collection while another thread sits in a native frame
  would corrupt it exactly like concurrent `(gc)` does today. Land E1
  behind the env flag, flip the default in E4.

### E2. JIT stack maps + safepoint polls
- Set `preserve_frame_pointers = true` in the ISA builder.
- Mark every SSA value holding TAGGED VALUE BITS with
  `declare_value_needs_stack_map`: block params for loop vars, the values
  bound to `vars` (locals), call results, closure bits (`self_closure`),
  spilled arg arrays… NOT untagged intermediates (untagged ints/floats mid
  arith), NOT derived pointers. The compile() plumbing: a `def_value`
  helper that declares + returns, used everywhere a value-producing node
  yields tagged bits.
- caps_base DIES as a cached ctx field: `Ir::Capture` re-derives
  `mask(self_closure) + CLOSURE_CAPS_OFF` at each read (self_closure is
  stack-mapped, so post-safepoint reads see the moved object). The 4-store
  child ctx shrinks by one word. (Loop-hoisting for capture reads can come
  back later via a "no safepoint in loop body" analysis — not now.)
- SAFEPOINT POLLS: at every fast-body function ENTRY and at every self-tail
  BACK-EDGE: load `gc_requested||gc_pressure` (one shared AtomicU8 "poll
  word" in RunCtx so it is a single readonly-base load + brif), cold block
  calls `shim_safepoint(rc)` (a normal call → a safepoint with a stack
  map). This bounds time-to-park for native-only loops — the thing that
  makes concurrent GC + native code actually correct rather than
  "eventually the loop exits".
- Every existing shim call site is automatically a safepoint (it is a
  call): the stack maps for live values there come for free from
  declare_value_needs_stack_map.

### E3. The native frame walker (runtime side)
- Code registry: when a body is compiled, record
  `(code_start, code_end, Vec<(ret_addr_off, active_size, entries: Vec<u32>)>)`
  sorted; a global (per-JitCranelift) lookup: RA → (function, map).
- Entry/exit anchoring: the JIT entry trampoline stores the current native
  SP/FP into the RunCtx ("anchor") before jumping into JIT code; shims do
  NOT need anchors (they are Rust frames between JIT frames — the FP chain
  crosses them; we only INTERPRET frames whose RA lands in JIT code).
- At park time (rt.park / shim_safepoint / await_future), before publishing:
  walk the FP chain from the current frame to the anchor; for each frame
  whose return address resolves in the code registry, compute
  `SP = fp_pair_addr - active_size` (pin the exact arithmetic empirically
  on arm64 with one instrumented test before trusting it) and publish
  `SP + entry_off` for each entry as *mut u64 root slots (into
  `me.native_roots: Mutex<Vec<usize>>`).
- gc.rs collect_inner: visit every published native root slot like any
  other root (idempotent; the collector rewrites the spill slots; Cranelift
  emitted code reloads them after the call returns).
- The collecting thread itself may be inside JIT frames (a shim called
  collect): scan OUR OWN native frames the same way (walker runs on self).

### E4. Flip the default + kill the carried gap
- Pressure-triggered GC ON by default; `MICROLANG_HEAP_MB` still caps.
- Remove the "concurrent explicit (gc) unsound with native frames" caveat
  from docs/memory: with polls + maps it is simply correct.
- New tests: (a) gc-stress across the 45-combo matrix + jit tests;
  (b) allocation-pressure loop that provably collects mid-JIT-loop
  (assert collections>0 with no explicit (gc), values intact after);
  (c) thread test: one thread in a native arith loop, another (gc)-ing —
  must terminate (polls) with intact heap; (d) capture-read-after-move.
- Gates: all four suites (default + jit + scheme + clojure-stub), plus the
  full gc-stress matrix run, plus the bench suite (no regression on the
  1-3× core band; some regression on capture-heavy loops is acceptable and
  measured).

## Non-goals (now)
- Precise maps for the SLOW (memory-mode) JIT bodies beyond what they get
  free from their heap Frame usage (their locals live in Frames already —
  traced via env publication; only their Rust-local temporaries need the
  same treatment as fast bodies — verify during E2 which values those are).
- Concurrent/incremental collection. STW stays.
