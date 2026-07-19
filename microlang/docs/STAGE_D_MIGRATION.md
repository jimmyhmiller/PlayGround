# Stage D migration spec — the real heap (working notes)

Status: D1 landed (src/heap.rs, commit 3ed239050). D2–D4 IN FLIGHT: core
design files are rewritten (model.rs, value.rs, gc.rs, runtime.rs infra);
the remaining work is the mechanical sweep of match/alloc sites. This file
pins every design rule so the sweep is unambiguous.

## The design (decided; do not re-litigate)

- A reference IS an address: `Repr::as_ref(bits) -> heap::Gc` (mask),
  `enc_ref(Gc) -> u64`. `HeapId` is GONE. Each Repr also impls
  `heap::PtrPolicy` (`Repr: PtrPolicy` supertrait) for the collector.
- Objects: 8-byte header `[type_id u16 | spare u16 | aux u32]` + inline
  fields, allocated from `heap::Heap` (two bump spaces, Cheney evacuation,
  explicit/safepoint GC only — allocation NEVER collects; exhaustion panics).
  Layouts per type: `heap::kind` + `heap::type_table()`.
- `value::Obj` is now the ALLOCATION REQUEST enum (constructor vocabulary):
  `Runtime::alloc(Obj) -> Gc` lowers it. Hot paths use typed constructors:
  `alloc_vector/_cap/_nil`, `alloc_values`, `alloc_record`, `alloc_closure`.
  All return `Gc`; wrap with `M::R::enc_ref(g)` for value bits.
- READS go through `Runtime::view(bits) -> ObjView<'_>` (or `view_gc(Gc)`),
  which reconstructs the old enum shapes as borrows:
  - `Obj::Str(s)` (String) → `ObjView::Str(&str)`
  - `Obj::Vector{off,len,cap}` + `rt.words(off,len)` →
    `ObjView::Vector{elems: &[u64], gc, cap}` (elems IS the live slice; no
    more word arena. For mutation use `rt.arr_slice_mut(gc)`; for growth
    `rt.arr_extend(gc, xs)`; capacity/logical-len live in blob/handle aux.)
  - `Obj::Values{off,len}` → `ObjView::Values(&[u64])`
  - `Obj::Record{type_id,off,len}` → `ObjView::Record{type_id, fields: &[u64]}`
  - `Obj::Closure{nparams,variadic,nslots,body,caps}` →
    `ObjView::Closure{nparams,variadic,nslots,template,gc}`; the BODY is
    `rt.template(template).clone()` (append-only registry); captures are
    inline in the object (`value::closure_cap(gc, i)`).
  - `Obj::MultiFn{fixed,variadic}` → `ObjView::MultiFn{fixed: &[u64], variadic}`
  - `Obj::Atom(Arc<AtomicU64>)` → `ObjView::Atom(&AtomicU64)` (the in-object
    slot; CAS directly on it).
  - `Obj::Cont/PartialCont(Arc<Kont>)` → same, via the kont registry.
  - `Obj::Future(arc)` → same, via the future registry.
  - `Obj::Moved` is GONE: stale pointers hit the poisoned/INVALID header
    check inside `view`/`decode`/`as_cons` and panic loudly.
- `Frame { slots, caps_src: AtomicU64 }`: `caps_src` holds the RUNNING
  closure's bits (0 = none) and is a traced root slot; capture reads decode
  it and load off the closure object (`value::frame_cap::<M::R>(env, idx)` —
  now takes the Repr type param). `build_caps::<M::R>(captures, env) ->
  Vec<u64>`; `Caps`/`no_caps()` are GONE. `build_call_frame` must set
  `caps_src` to the CALLEE's bits (pass callee bits instead of a caps Arc).
- GC: `gc.rs` enumerates roots into `heap.collect::<M::R>`; per-object
  scanning is generic (`scan_object` over TypeInfo) — there is NO per-type
  GC code anymore. Roots: globals, shadow stacks, dyn stacks, parked
  mutators' published roots/envs, consts, method impls, arglists, the `()`
  singleton (`Shared.empty_list`), the kont registry (walk chains), future
  results, live envs, live kont. IC epoch = `shared.relocated` (bumped once
  per collection now, not per object).
- Closure object ABI (for the JIT): `[hdr(aux=ncaps) | meta | code | caps…]`
  at offsets `CLOSURE_META_OFF=8`, `CLOSURE_CODE_OFF=16`, `CLOSURE_CAPS_OFF=24`.
  meta = `closure_meta(template, nparams, nslots, variadic)`:
  bits 0..32 template, 32..48 nparams, 48..63 nslots, bit63 variadic.
  code = native fast entry or 0. THIS RETIRES the JIT's heap-id-keyed
  `fast_targets` table: the emitted call site now masks the callee to an
  address, checks `type_id == kind::CLOSURE`, checks meta arity + !variadic,
  loads the code word, and `call_indirect`s with `caps_base = addr + 24`.
  `JitCtx.fast_base/fast_len` die; `emit_ref_id` becomes an address mask.
  Compiled code per TEMPLATE: a `template_code: Vec<AtomicPtr>` style map
  (reserved TABLE_CAP) lets `shim_make_closure` stamp the code word at
  closure creation; the slow-call path compiles, stamps the template map AND
  the callee object.
- `%spawn`/futures: `FutureSlot` Arc lives in `Shared.futures` (registry =
  OS-resource table); object holds the index. `%callcc`/`%shift`: Arc<Kont>
  in `Shared.konts`; object holds the index. Registry entries leak by
  design (append-only) — reification is rare; note if it ever matters.
- `enc_empty_list` returns the per-runtime SINGLETON (Shared.empty_list,
  CAS-initialized, GC root).
- Strings: `rt.str_view(bits) -> Option<&str>` is the fast typed accessor.
- HugeInt: sign raw word + base-10^9 limbs as LE bytes in the tail;
  `BigInt::from_parts(neg, mag)` reassembles (cold path).
- Heap sizing: `MICROLANG_HEAP_MB` per space (default 4096 MiB, virtual /
  lazily committed). Verify mode (`MICROLANG_GC_VERIFY`, default ON in debug
  builds): poisons evacuated space (0x5A), checks traced slots point at real
  headers. Type id 0 = INVALID by construction.

## Sweep rules (mechanical)

- `&rt.heap()[M::R::as_ref(bits) as usize]` → `rt.view(bits)`; match arms
  per the table above. `rt.heap()[id]` with a saved id: ids are `Gc` now.
- `rt.alloc(Obj::X(...))` still works for request-shaped variants; it
  returns `Gc` — wrap in `enc_ref` where bits are needed (grep for
  `M::R::enc_ref(id)` which still compiles unchanged).
- `Obj::Vector{off,len,cap}` PATTERNS in prims: rewrite on `ObjView::Vector
  {elems, gc, cap}`; `rt.words/words_mut/word_alloc*` are GONE.
- Sites that CLONED heap Strings (`Obj::Str(s) => s.clone()`) can borrow
  (`ObjView::Str(s)`) or `to_string()` when ownership is needed. Watch
  borrowck: a view borrow of `self` blocks `&mut self` calls — copy out
  (`let s = s.to_string()`) or destructure to owned values first, exactly
  like the old code did with `.clone()`.
- Multi-thread invariants unchanged: allocation is lock-free; `heap_lock`
  now only serializes array growth + collection entry; publication ordering
  rides the existing Acquire/Release slot discipline.
- Suites: `cargo test`, `cargo test --features jit`, scheme crate, and
  clojure-stub crate (its `src/{lib,reader,data}.rs` need the same sweep)
  must all pass before a phase commits.

## What remains after the sweep (D5)

- JIT inline tag tests + header type_id loads for dispatch ICs; inline
  field/aget; AllocWindow inline bump allocation (`heap.window`, gc-stress
  via limit=0). Until then the JIT correctness path = shims through the new
  accessors; only `emit_call`'s fast path (object code-word load) is
  REQUIRED to compile/run.
- KNOWN CARRIED GAP (pre-existing): SSA-held bits + derived `caps_base` in
  native frames go stale if a callee triggers `(gc)` mid-body — same class
  as before (stack maps are the fix, EXEC_MODEL_V2 gap #4).
