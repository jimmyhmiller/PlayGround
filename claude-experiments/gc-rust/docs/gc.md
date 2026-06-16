# The garbage collector and the codegen↔GC contract

gc-rust does not have its own collector. It reuses ai-lang's GC module verbatim
(`crates/gcrust-rt/src/gc/`), a self-contained, precise, relocating collector.
This document describes what it is and the exact ABI compiled code must honor so
collections are correct. `tests/gc_abi_smoke.rs` proves this contract end-to-end
with hand-built LLVM IR; `tests/generational.rs` verifies the generational path.

## The collector in one paragraph

gc-rust runs the **generational** mode by default (see "Generational mode" at
the end). At its core is a semi-space **copying** collector — two equal
half-spaces; allocation bumps a cursor in *from-space*; when it fills, a
collection copies every live object to *to-space* (Cheney scan), leaving a
forwarding pointer in each moved object's header word, then swaps the spaces.
Because objects **move**, every live pointer must be found and updated — which
is why roots are precise and why compiled code must spill pointers across
allocations. Cycles are collected for free (a copying collector doesn't care
about cycles). The generational layer puts a bump-allocated nursery in front of
this, collected cheaply and often by minor GCs, with a card-table write barrier.
(A concurrent-marking path with SATB barriers also exists in the runtime but is
not turned on.)

## Object layout

Every heap object starts with a 16-byte `Full` header (`gc_word` + `type_id`).
The `type_id` indexes the runtime *type table* of `gc::TypeInfo`, which
describes the object's shape:

```
[ Full header (16B) ][ ptr fields: N×8B (GC-traced) ][ raw bytes (untraced) ][ optional varlen tail ]
```

**Pointer fields come first**, always. The scanner (`gc::scan_object`) traces
exactly the leading `value_field_count` 8-byte slots (plus a varlen-Values tail
if present) and skips the raw bytes. Codegen's `Layout` → `TypeInfo` lowering
must put every GC reference in the pointer-field region and every scalar in the
raw region. (See `docs/core-ir.md`.)

## The ABI (`crates/gcrust-rt/src/runtime.rs`)

Compiled code is passed a `*mut Thread` as its first argument:

```
Thread { state: u8, top_frame: *mut Frame, heap: *mut Heap,
         dyna_thread: *const ThreadState, alloc_window: *const AllocWindow }
```

fields at fixed byte offsets (`thread_offsets`). Per call it builds a `Frame`:

```
Frame { parent: *mut Frame, origin: *const FrameOrigin, roots: [*mut u8; N] }
FrameOrigin { num_roots: u32, name: *const u8 }   // one static const per fn
```

### What each compiled function must do

1. **Prologue.** Alloca `{ parent, origin, [roots; N] }`. Set `origin` to the
   function's static `FrameOrigin`. Set `parent = thread.top_frame`, then
   `thread.top_frame = &frame`. Zero all N root slots.
2. **Spill before every allocation/call.** Any GC reference that is live across
   an allocation (or any call that might allocate/collect) must be stored into a
   frame root slot *before* the call, and reloaded *after* — because the call
   may move it. The slot is what the GC updates.
3. **Allocate** via `ai_gc_alloc_fixed(thread, type_id)` /
   `ai_gc_alloc_varlen(thread, type_id, n)`. These publish the frame chain, call
   `Heap::alloc_obj::<Full>` (which stamps the header `type_id` — *not* bare
   `alloc`, which would leave it 0), then clear the published fp.
4. **Safepoint poll** at loop back-edges: load `thread.state` (volatile); if
   non-zero, call `ai_gc_pollcheck_slow(thread)`, which parks at a safepoint
   with the frame published until the collection finishes.
5. **Epilogue.** Restore `thread.top_frame = frame.parent`.

### How the GC finds roots

At collection, the GC walks each registered thread's published `parked_jit_fp`
(its top `Frame`) via `runtime::walk_gc_frames`: follow `parent` links, and for
each frame scan `origin.num_roots` slots. Each slot is visited as a `*mut u64`,
so the collector both *traces* (keeps alive) and *updates* (relocates) it. This
is the entire root set — there is no stack scanning and no false retention.

### Triggering a collection

A mutator that needs to collect (allocation exhaustion, or stress mode) does
**not** call `stw_collect` — that deadlocks (it would wait for itself to reach a
safepoint). It publishes its frame and calls
`Heap::mutator_triggered_gc::<IdentityPtrPolicy>(dyna_thread)`: the caller
becomes the GC thread, parks every *other* mutator, scans all parked frames,
copies, swaps, and resumes. See `RuntimeContext::force_collect`.

## What Phase 0 proved

`tests/gc_abi_smoke.rs` JITs a function implementing exactly the prologue →
spill → allocate → build graph → **collect mid-function** → reload → epilogue
sequence above, and asserts that after the collection the relocated object's
field still points at the relocated child of the correct shape. Green ⇒ the
contract codegen will generate is sound.

## Generational mode (active)

gc-rust now runs a **generational** collector by default (JIT and AOT):

- **Nursery (young gen)**: 16 MB. New allocations land here. When it fills, a
  cheap **minor GC** scavenges only the nursery, promoting survivors to tenured.
- **Tenured (old gen)**: 256 MB per space. A **major GC** (the semi-space
  copying collector) reclaims it when it fills.
- **Write barrier**: `ai_gc_write_barrier` runs after every pointer store into a
  heap object/array. If a tenured object gains a nursery pointer, the covering
  card is marked dirty so the minor GC finds the old→young edge without scanning
  all of tenured. Object *construction* needs no barrier (a fresh object is
  always young). The barrier no-ops for non-pointer stores and on the
  non-generational heap.

Most objects die young, so minor GCs do the bulk of the work and major GCs are
rare — e.g. `binary_trees` runs 12 minor + 0 major collections. Set
`GCR_GC_STATS=1` to print the counts.

`--gc-stress` deliberately uses the **single-generation semi-space** collector
with collect-on-every-allocation: its relocation invariants are semi-space
specific, so it torture-tests that collector while normal runs get the
generational speedup. Both produce identical results on every example.
