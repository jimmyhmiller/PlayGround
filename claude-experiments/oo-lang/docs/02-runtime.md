# 02 — Runtime Architecture

This is the design for the bytecode VM, object model, and per-type arena allocator, all
implemented in Coil (`coil guide`, `apps/clox/clox.coil` as precedent). Every structure
below is written as if it were about to be typed into a `.coil` file — this is meant to be
buildable, not aspirational.

The one idea everything else serves: **a Scry program's live object graph must be
walkable, from the outside, without stopping the world for longer than a query takes.**
That requirement shapes the allocator, the GC, and the thread model, in that order.

**Concurrency is now real OS threads, day one** (`DECISIONS.md` #4b, superseding this doc's
earlier cooperative-single-interpreter-thread position). Every Scry-level thread — each
demo agent, concretely — is a genuine `pthread`, with its own call-frame array and value
stack; the heap, globals, and type/method tables are shared across all of them. That
single change is why §7 below is the deepest rewrite in this document: a lock-free
single-writer allocator and a one-thread safepoint check both assumed exactly one mutator,
and neither assumption survives contact with N of them.

## 1. Position: no NaN-boxing

clox NaN-boxes `Value` into one tagged 64-bit word because Lox is dynamically typed —
every stack slot, every field, every argument can hold *any* type, so every `+`, every
`OP_GET_PROPERTY`, every arithmetic op has to ask "what actually is this?" at runtime
(`is-number`, `is-obj-kind`, table lookups on `ObjInstance.fields`).

Scry is statically typed with no inheritance. The compiler knows, at every single
bytecode-emission site, the exact static type of every value it's emitting code for. That
changes the right representation entirely: **values are untagged, and the bytecode itself
carries the type information via which opcode is chosen, not via a runtime tag.**

- A local variable of type `Int` lives in a stack slot as a raw `i64`. A `Float` local
  lives in a slot as the bit pattern of an `f64` (round-tripped through memory the same
  way clox's `bits-to-f64` does — Coil's `(cast f64 i64)` is a numeric conversion, not a
  bitcast, so reading a float slot means `(load (cast (ptr f64) slot-addr))`). A `Bool`
  local is a raw `i8`/`i64` 0-or-1. A class-typed local is a raw pointer.
- `a + b` where both operands are statically `Int` compiles to `OP_ADD_I64`. Where both
  are `Float`, `OP_ADD_F64`. There is no `OP_ADD` that branches on a runtime tag — the
  type checker already proved which one is correct, so the bytecode says so directly.
  This is exactly how JVM bytecode works (`iadd` vs `fadd` vs `ladd`) and it is why a
  typed VM can be simpler *and* faster than clox's dynamically-tagged one, not just faster.
- This is a real divergence from the clox port and needs to be said plainly: **Scry's
  bytecode is a typed bytecode** (à la JVM), clox's is a **dynamically-tagged** bytecode
  (à la a scripting language). We are reusing clox's *architecture* (dispatch loop, object
  header, mark-sweep GC, call frame shape) but not its *value representation*, because the
  entire reason clox needs NaN-boxing (uniform representation for a dynamically-typed
  stack) doesn't apply here.

Where dynamism does show up — generics — it's handled by monomorphization at compile time
(Coil already does this for its own `[T]` generics: `(defn id [T] [(x T)] …)` is specialized
per instantiation, not boxed), not by falling back to a tagged `Value`. A `List<Agent>` and
a `List<Task>` are different compiled types with different, statically-known element
layouts.

**Interfaces are IN** (`DECISIONS.md` #4, superseding this doc's earlier "fat pointer if
interfaces land" hedge in §5) — and, having actually designed them (§5 below), they turn
out **not** to need a fat pointer at all. An interface-typed field or local is the same
single raw pointer a class-typed one is; the one new thing it needs is a *per-instance*
type tag, because unlike a class-typed field (whose concrete type the compiler already
knows) an interface-typed field's pointee could be any of several implementing classes.
That tag lives in `InstanceHeader` (§3), not in the reference itself — so dispatch costs a
header load plus a table lookup, and storage costs nothing extra over a plain pointer. See
§5 for the full design and its honest cost.

## 2. Bytecode and the dispatch loop

Reused wholesale from clox: a flat `u8` bytecode array per method, a giant dispatch loop
that reads one opcode and jumps to its handler, and Coil's `case` compiling a dense
integer switch to a real jump table (`coil guide`: "a dense integer `case` compiles to a
JUMP TABLE") — the same property that makes clox's `run()` fast.

```
(const OP_LOAD_LOCAL_I64   0)   (const OP_LOAD_LOCAL_F64   1)  (const OP_LOAD_LOCAL_REF 2)
(const OP_STORE_LOCAL_I64  3)   (const OP_STORE_LOCAL_F64  4)  (const OP_STORE_LOCAL_REF 5)
(const OP_LOAD_FIELD_I64   6)   (const OP_STORE_FIELD_I64  7)  ; …one pair per stored kind
(const OP_ADD_I64  10) (const OP_ADD_F64 11) (const OP_LT_I64 12) (const OP_LT_F64 13)
(const OP_NEW           20)     ; operand: type-id  -> pushes a raw ptr to a fresh instance
(const OP_CALL_STATIC   21)     ; operands: fn-address-const-index, argc (no dispatch, direct call)
(const OP_CALL_VIRTUAL  22)     ; operands: interface-id, method-slot, argc — REAL now, §5
(const OP_RETURN        23)
(const OP_JUMP 24) (const OP_JUMP_IF_FALSE 25) (const OP_POP 26)
```

Object field access compiles to `OP_LOAD_FIELD_I64`/`_F64`/`_REF`/… with a **compile-time
constant byte offset** as the operand, not a name. This is the other big static-typing
win: clox's `ObjInstance.fields` is a hash `Table` because Lox instances can grow fields
dynamically at runtime; Scry classes have a fixed shape decided at compile time, so
`self.status = AgentStatus.Paused` is `OP_STORE_FIELD_I64 <offset=16>` — a pointer add and
a store, no hashing, no probing, comparable to JVM `putfield` with a constant-pool-resolved
offset.

**Call frames** — same shape as clox's, with one addition (the ref-slot bitmap, needed for
precise GC roots, §6):

```coil
(defstruct MethodInfo
  [(name        (ptr u8)) (name-len i64)
   (arity       i64)
   (local-count i64)                 ; total stack slots this method needs
   (ref-slots   (ptr u64))           ; bitmap: bit i set => local slot i holds a ref/ptr
   (code        (ptr u8)) (code-len i64)
   (fn-address  (fnptr c [(ptr u8)] i64))])   ; for CALL_STATIC / viewer invoke, same fn

(defstruct CallFrame [(method (ptr MethodInfo)) (ip (ptr u8)) (base i64)])
```

**One `VMThread` per Scry-level thread — this is the change §7 forces back up into this
section.** clox, and every earlier draft of this doc, had exactly one `frames` array and
one `stack`, living directly on a singleton `VM`. Real OS threads (`DECISIONS.md` #4b)
mean N independent call stacks can be executing at once, so the frame array and value
stack move off the shared `VM` and onto a per-thread struct; only the things that are
genuinely shared — globals, the type/method tables, the arenas, GC and safepoint state —
stay on `VM`:

```coil
(defstruct VMThread
  [(thread-id   i64)                 ; index into VM.threads — assigned once, at spawn
   (os-thread   Thread)              ; lib/thread.coil's pthread handle wrapper
   (frames      (array CallFrame 256))
   (frame-count i64)
   (stack       (array i64 65536))   ; THIS thread's value stack, untagged (see §1)
   (stack-top   i64)
   (magazines   (ptr (ptr Magazine))) ; per-type allocation caches, THIS thread's own — §4
   (parked      i64)])               ; atomic 0/1 — set while this thread is stopped at a
                                      ; safepoint for another thread's GC/migration/eval, §7

(defstruct VM
  [(threads       (array (ptr VMThread) 64))  ; fixed capacity, same style as CallFrame's 256
   (thread-count  i64)                         ; atomic — bump-reserved at spawn, §7
   (stop-flag     i64)                         ; atomic 0/1 — a global stop is requested, §7
   ; … globals table, current-class-table (the generation root — §3), arenas, GC state — §5/§6
   ])
```

`VMThread.frames`/`stack` are sized the same as before per thread — the memory budget is
now `(256 CallFrames + 65536 i64 slots) × live thread count`, not a fixed constant; for the
PoC's handful of agent threads this is a few hundred KiB per thread, a non-issue, but it's
worth naming as the first real cost of "N stacks instead of one."

The dispatch loop is clox's loop verbatim in shape (hoist the mutable `frame` cell outside
the `loop`, per the clox port's own comment: never `alloc-stack` inside a hot loop) —
**the only change from a single-threaded VM is that `run` takes its `VMThread` as an
explicit parameter instead of reading a singleton**, because Coil has no implicit
thread-local storage: a spawned thread's pthread body function receives its `VMThread*`
as the raw `(ptr i8)` argument `pthread_create` hands it (§7), and that pointer is simply
threaded through every call in the interpreter's own call chain from there — the same
explicit-parameter style Coil already uses everywhere else, not a new mechanism:

```coil
(defn run [(vt (ptr VMThread))] (-> i64)
  (let [(mut frame) (index (field vt frames) (- (load (field vt frame-count)) 1))]
    (loop
      (safepoint-poll vt)                             ; §7 — one branch, almost always false
      (let [instr (read-byte (load frame))]
        (case instr
          OP_LOAD_LOCAL_I64 (do (push-i64 (load (index stack-base (read-byte …)))) 0)
          OP_ADD_I64        (do (push-i64 (+ (pop-i64) (pop-i64))) 0)
          OP_STORE_FIELD_I64 (let [off (read-byte (load frame)) v (pop-i64) obj (pop-ptr)]
                               (do (store! (cast (ptr i64) (+ (cast i64 obj) off)) v) 0))
          OP_CALL_STATIC     (do (push-frame …) 0)
          OP_CALL_VIRTUAL    (do (call-virtual vt …) 0)   ; §5
          OP_RETURN          (if (pop-frame) (continue) (break (pop-i64)))
          ; … 
          )))))
```

### What `ref-slots` actually covers: locals *and* operand-stack temporaries

This dispatch loop is, like clox's, a **stack** VM: `push-i64`/`pop-i64`/`push-ptr` push and
pop against the *same* physical array (`VMThread.stack`, §2 above) that declared locals
live in — an intermediate result like `OP_NEW`'s freshly-allocated pointer sits on that
stack, above the frame's declared locals, for however many instructions elapse before the
following `OP_STORE_FIELD_REF`/`OP_STORE_LOCAL_REF` consumes it. `safepoint-poll` (§7) runs
at the top of *every* iteration of this loop — once per instruction, not once per call — so
it absolutely can fire in the gap between `OP_NEW` pushing that pointer and the store that
retires it. If `MethodInfo.ref-slots` only tracked *declared* locals, `mark-roots` (§6)
would have no idea that transient value exists, and a GC triggered by a *different* thread
at exactly that instant would collect the object out from under the very frame that just
allocated it — a use-after-free on first GC, not a threading nuance.

**Resolved by widening what a "slot" is, not by restricting where safepoints may fire.**
`MethodInfo.local-count` (§2) is not "how many declared locals this method has" — it is the
method's **entire frame extent**: declared arity + locals, *plus* the method's
statically-computed maximum operand-stack depth (the same `max_stack` a JVM verifier
computes — a static property of well-formed stack bytecode, since every control-flow merge
point has one consistent stack depth no matter which runtime path reached it). `ref-slots`
is sized and populated over that *whole* extent. Concretely: every operand-stack push the
compiler emits targets a specific, numbered slot at a specific depth-from-base, exactly the
way a declared local does — "the operand stack" is not a separate, untracked region, it's
the same slot-numbering scheme continued past the declared locals.

One compiler invariant this requires, stated explicitly because it's what keeps
`ref-slots` a single flat, whole-method bitmap instead of needing a per-program-point stack
map: **a given slot index's kind (ref vs. non-ref) is fixed for the entire method, never
reused across kinds.** Two operand-stack temporaries with non-overlapping live ranges may
still share a slot index — that's the whole point of tracking stack depth instead of
handing out a fresh slot per push — but only if they're the *same* kind; a ref-typed
temporary is never assigned a slot index that some other point in the method uses for an
`Int`/`Float` temporary. This costs a handful of extra slots in the rare method whose peak
ref-temp depth and peak non-ref-temp depth would otherwise have overlapped — an honest,
small, one-time cost paid at compile time, not a correctness compromise, and it's what lets
`mark-roots` keep reading one bitmap per method rather than one per instruction.

The other half of the fix: every slot in `[declared-locals, local-count)` — the
temp-slot region — is zeroed when `push-frame` sets up a new `CallFrame` (the same
"zero once, cheaply" move §4 already makes for fresh slabs), so a safepoint landing before
a ref-temp slot's *first* write in this invocation sees a null pointer sitting in that slot,
never garbage left over from whatever this same stack memory held for some earlier, unrelated
frame. `mark-object` already has to treat null as "nothing to mark" (every optional
reference field needs that case regardless), so this adds no new case to the marker — just
a guarantee about what an as-yet-unwritten ref slot contains.

**Worked example.** Compiling `self.tasks[i] = Task.new()`, in the shape §2's dispatch loop
above already uses:

```
OP_NEW <type-id=Task>          ; allocates, stamps the header (§4), and PUSHES the fresh
                                ; (ptr Task_Instance) into this frame's temp slot T —
                                ; ref-slots bit T is set at compile time, unconditionally
                                ; <-- safepoint-poll runs again right here, at the top of the
                                ;     loop's next iteration, before OP_STORE_FIELD_REF runs
OP_STORE_FIELD_REF <offset>    ; pops slot T (the new Task) and the receiver ptr, stores
```

If another thread's `request-global-stop` (§7) lands in the gap marked above, *this*
thread's `safepoint-poll` parks it with the freshly-allocated `Task` sitting in slot `T` and
nowhere else yet — no local variable has been assigned it, no field has been stored into.
Because `T < local-count` and `ref-slots` bit `T` is set, `mark-roots`'s scan of this frame
(§6) reads exactly that slot and marks the object live, exactly as if it were sitting in a
declared local. The object survives the collection that's about to run, and
`OP_STORE_FIELD_REF` resumes and retires it into `self.tasks[i]` once every thread is
released — this is the property the rest of this design depends on, made concrete.

## 3. Object layout for class instances

Every instance is: a fixed **header**, then fields in declaration order at compile-time
offsets. No per-instance kind tag is needed the way clox's `Obj.kind` needs one — **which
arena you're in already tells you the type** (§5). That's a real header-size win over
clox: one arena per type means "type" is a property of the *pointer's origin*, not
something every object has to carry itself.

```coil
(defstruct InstanceHeader
  [(slot-index i64)     ; this instance's index within its type's arena — needed to hand
                         ; the viewer a stable id (§4), and to return this slot to the
                         ; free list without a reverse lookup
   (generation u32)      ; bumped when this slot is freed — invalidates viewer handles (§4)
   (type-id    u32)      ; NEW, driven by interfaces (§5) — see note below
   (flags      u32)])    ; bit0 live, bit1 gc-marked (bit2+ reserved)

; class Agent { name: String, model: String, status: AgentStatus, conversation: Conversation, tools: List<Tool> }
(defstruct Agent_Instance
  [(header       InstanceHeader)
   (name         (ptr StringObj))
   (model        (ptr StringObj))
   (status       i64)                 ; enum -> i64 tag, no boxing
   (conversation (ptr Conversation_Instance))   ; raw pointer — see §4 for why this is safe
   (tools        (ptr ListObj))])
```

**Why `type-id` is new.** Earlier drafts of this doc didn't need a per-instance type tag —
"which arena you're in already tells you the type" (below) was true for every pointer in
the system, because every reference was either a class-typed field (compiler already
knows the concrete type) or reached via an arena walk (the walk already knows). Interfaces
(`DECISIONS.md` #4) break that: an interface-typed field like `Agent.tools`'s element type
`Tool` can point at a `ShellTool` or a `SearchTool`, two different arenas, and — critically
— the field itself carries no static hint of which. Given only the raw pointer, the only
way to find out is to ask the object itself, so the tag has to live in the one place every
instance shares regardless of concrete type: the header. It is used for exactly two things,
both detailed in §5/§6: resolving `OP_CALL_VIRTUAL`'s itable lookup, and telling
`blacken-object` which `TypeInfo` to recurse into when it reaches an object through an
interface-typed field rather than a class-typed one. It costs 4 bytes on every instance of
every type, whether or not that type ever implements an interface — a real, small,
always-paid cost, stated honestly rather than making it conditional on a per-type flag
(which would in turn need its own per-instance test on every dereference to know whether
the tag is present, defeating the point).

`InstanceHeader` is now 20 bytes (padded to 24 for 8-byte struct alignment, since
`slot-index` is `i64`). For an `Agent` with the fields above that's ~64 bytes total —
overhead is real but small, and paid once per instance, not per field access.

**Type metadata.** Every concrete `class` gets one compile-time-generated `TypeInfo`. This
single artifact is what both the GC (which fields are pointers to trace) and the viewer
(what to render, what to invoke) hang off of — worth calling out explicitly, because it
means reflection is not a separate subsystem bolted on for the viewer; it's the same table
the GC already needs to exist.

```coil
(const FIELD_I64 0) (const FIELD_F64 1) (const FIELD_BOOL 2) (const FIELD_STRING 3)
(const FIELD_REF  4) (const FIELD_LIST_REF 5)   ; REF/LIST_REF = fields the GC must trace
(const FIELD_IFACE_REF 6)   ; NEW, §5 — an interface-typed field; same one word as FIELD_REF,
                             ; but the concrete TypeInfo to trace into is read from the
                             ; pointee's OWN header (type-id), not from elem-type-id below

(defstruct FieldInfo
  [(name (ptr u8)) (name-len i64) (offset i64) (kind i64) (elem-type-id i64)])  ; elem-type-id: which
                                                 ; TypeInfo a REF points at — for FIELD_IFACE_REF
                                                 ; this is the *interface's* id, not a concrete
                                                 ; class's (there may be several); see §5

(defstruct MethodSig [(param-kinds (ptr i64)) (param-count i64) (ret-kind i64)])

(defstruct TypeInfo
  [(type-id       i64)
   (name          (ptr u8)) (name-len i64)
   (instance-size i64)
   (fields        (ptr FieldInfo)) (field-count i64)
   (methods       (ptr MethodInfo)) (method-count i64)
   (sigs          (ptr MethodSig))                 ; parallel to methods, for arg marshalling
   (itables       (ptr (ptr ITable)))               ; NEW, §5 — indexed by interface-id, null
                                                     ; for interfaces this class doesn't implement
   (arena         (ptr TypeArena))])                ; -> §4

(defn type-table [] (-> (ptr (ptr TypeInfo))) (alloc-static (ptr (ptr TypeInfo))))  ; indexed by type-id
```

### The class-table root — what `03-live-semantics.md`'s generation swap actually swaps

`03-live-semantics.md` requires "one atomic pointer swap, no torn class table ever
observed" for *every* accepted edit, even though every edit whole-program-recompiles (not
just the touched class) — its own words: `current_generation` is one pointer, swapped with
a single atomic store, and "no thread ever sees a torn class table." `type-table` just
above, as sketched, is a single static array **filled in once** and never described as
swappable — fine for a program that never redefines anything, but it has no notion of
"replace the whole thing with a new one," so as written it cannot be what 03 depends on.
This is the missing piece, defined here since 03 explicitly treats this doc's structures as
its own foundation:

```coil
(defstruct ClassTable
  [(generation i64)
   (entries    (ptr (ptr TypeInfo)))    ; same shape as `type-table` above, indexed by type-id
   (type-count i64)])

; VM (§2) gains one more field: (current-class-table (ptr ClassTable))

(defn vm-current-class-table [] (-> (ptr ClassTable))
  (atomic-load-ptr (field (vm) current-class-table)))   ; lib/atomic.coil's generic ptr family
```

`VM.current-class-table` is one pointer field, read with `atomic-load-ptr` and written with
a single `atomic-store-ptr` — never mutated field-by-field, never patched entry-by-entry.
`type-table`'s earlier sketch collapses into `(field (vm-current-class-table) entries)`;
every call site elsewhere in this doc that resolves `type-id -> TypeInfo` reads through
this root, so which `TypeInfo` a given `type-id` names is a property of *which
generation's* `ClassTable` a thread happens to be holding at the moment it reads it, not a
fixed slot in a table that's filled in once and never touched again.

**Accepting an edit (03's `compile_and_link` + `swap`) builds one full new `ClassTable`,
then swaps the root in a single atomic store:**

1. Compile the whole program (03's whole-program recompile, unconditionally — every
   method, every class, not just the one that was edited).
2. **Type-ids are stable identifiers, assigned by class name and carried forward across
   generations** — `Agent` is type-id 3 in generation 5 and stays type-id 3 in generation
   6, whether or not `Agent`'s own source changed at all, so that `InstanceHeader.type-id`
   (stamped once at `OP_NEW` time, §3, and never rewritten by a generation swap) keeps
   meaning the same thing across the swap. A class present in the old generation but absent
   from the new one keeps its type-id slot (03's "Removed type" case) so its live instances
   stay resolvable.
3. **For every type-id whose `ShapeInfo` is structurally unchanged** from the previous
   generation (same fields, same order, same types — whether method *bodies* changed is
   irrelevant here), the new generation's `TypeInfo.arena` field is set to **the exact same
   `(ptr TypeArena)`** the previous generation's `TypeInfo` used for that type-id — never a
   freshly allocated arena. This is what makes "recompile the whole program from scratch"
   compatible with "17 live `Agent`s must survive a body-only edit to `Agent.summarize`":
   the arena, and every live slot inside it, is completely untouched by the swap; only the
   `TypeInfo`/`ShapeInfo`/`methods` pointers wrapped around it are new.
4. **For every type-id whose shape *did* change**, the new `TypeInfo.arena` field is
   *also* the same `(ptr TypeArena)` as before — the subsection below ("Shape migration")
   mutates that same struct in place (populating `old-slabs`/`pending-shape`, then
   draining), it never swaps in a different `TypeArena`. The generation swap is what
   *schedules* the migration (03's `schedule_migration`); the quiescence gate and rewrite
   pass below are what actually carry it out, against that one persistent arena.
5. Build a fresh `entries` array (one `(ptr TypeInfo)` per type-id, freshly allocated) and a
   fresh `ClassTable` wrapping it, then `atomic-store-ptr` `VM.current-class-table` to point
   at it — one instruction, the entire generation becomes visible at once.

Because the swap is a single pointer store, a thread reading `vm-current-class-table`
mid-swap sees either the whole old `ClassTable` or the whole new one, never a mix — 03's
"no torn class table" guarantee is grounded in exactly this, the same one-atomic-pointer
shape `VM.stop-flag`/`VMThread.parked` already use elsewhere in this doc (§7), not a new
kind of atomicity invented for this purpose. The old `ClassTable` struct and its `entries`
array are not reclaimed the instant the swap lands — a thread that read the pointer just
before the swap may still be mid-dispatch against it (every `CALL` resolves through
whichever `ClassTable` it loaded fresh, never a cached one, exactly as 03's "Threads and
the generation boundary" section requires); the old `ClassTable`'s memory is safe to free
once every `VMThread`'s next `CALL` has passed — in practice, reclaimed at the next
safepoint stop, the same coordination point §7 already brings every thread to for GC and
migration, rather than inventing a second reclamation mechanism just for this.

## 4. The centerpiece: per-type arena / slab allocator

This is decision #6 (`DECISIONS.md`), and it's the thing the entire viewer rests on:
**"list every live `Agent`" must be a slab walk, not a heap crawl.**

### Slab layout

Each type gets one growable arena, made of fixed-capacity slabs (chosen so a slab is
about 64 KiB, and its capacity is rounded down to a power of two — so slot→(slab, local)
is a shift and a mask, never a division):

```coil
(defstruct TypeArena
  [(type-id        i64)
   (slot-size      i64)              ; sizeof(InstanceHeader) + sizeof(fields), padded to align
   (slab-shift     i64)              ; capacity = 1 << slab-shift, sized so slab ≈ 64 KiB
   (slabs          (ptr (ptr u8)))   ; growable array of slab base pointers — index = slab#
   (slab-count     i64)              ; atomic — only advances under `growing` below
   (bump-cursor    i64)              ; atomic — flat frontier of ever-reserved slots (NEW:
                                      ; collapses the old (bump-slab, bump-offset) pair — both
                                      ; were already redundant with a single flat index, since
                                      ; arena-slot-ptr below derives slab#/local from one index
                                      ; via shift+mask; one field is what lets reservation be a
                                      ; single atomic-add instead of a two-field CAS loop, §4b)
   (free-head      i64)              ; atomic — index of first free slot, or -1
   (growing        i64)              ; atomic 0/1 spinlock — held only while malloc'ing a new
                                      ; slab and appending to `slabs`, §4b
   (live-count     i64)])            ; atomic — coarse, for cheap "how many Agents" summaries
```

A slot's global index is `slab# * capacity + local#`. Resolving `slot-index → pointer`:

```coil
(defn arena-slot-ptr [(a (ptr TypeArena)) (idx i64)] (-> (ptr u8))
  (let [cap    (ishl 1 (load (field a slab-shift)))
        slab#  (ishr idx (load (field a slab-shift)))
        local  (iand idx (- cap 1))]
    (index (load (index (load (field a slabs)) slab#)) (* local (load (field a slot-size))))))
```

### Thread-safe allocation: per-thread magazines over the shared arena

This is `DECISIONS.md` #4b's "thread-safe per-type arena allocation" — and the design the
brief's magazine language was named for (Bonwick's slab-allocator paper puts a small
per-CPU cache, a *magazine*, in front of a shared slab layer specifically so the common
case never touches a lock; here the cache is per-**thread** instead of per-CPU, same idea).
With N real interpreter threads (§7) all calling `OP_NEW`/instance-free concurrently, the
single-writer `arena-alloc`/`arena-free` an earlier draft of this doc had is simply wrong —
plain, non-atomic reads and writes to `free-head`/`bump-*` racing across threads is a
corrupted free list, not a slow path. Two layers fix it without putting a lock on the hot
path:

**Layer 1 — the shared arena, touched only in *batches*, never per-object.** The free list
is a lock-free Treiber stack over slot indices — the exact CAS-loop shape `examples/
lockfree.coil` already proves in this codebase (`lf-push`/`lf-pop`), just pushing/popping
`i64` slot indices instead of node pointers, using the freed slot's own first word as the
link (the same trick as before):

```coil
(defn arena-free-push [(a (ptr TypeArena)) (idx i64)] (-> i64)     ; one CAS-loop push
  (let [p (cast (ptr i64) (arena-slot-ptr a idx))]
    (loop (let [old (atomic-load (field a free-head))]
            (do (store! p old)
                (if (icmp-eq (atomic-cas (field a free-head) old idx) old) (break) (continue)))))
    0))
; arena-free-pop-batch(a, out, want) -> got: same CAS-loop shape as lf-pop, called `want`
; times (or until the list is empty); elided here since the mechanism is identical to
; lockfree.coil's, not repeated for the second time in this doc.
```

Bump reservation needs no CAS loop at all, because the frontier only ever moves forward —
one `atomic-add` reserves a whole **batch** of fresh slots in a single atomic op:

```coil
(defn arena-bump-batch [(a (ptr TypeArena)) (n i64)] (-> i64)   ; -> first reserved slot index
  (atomic-add (field a bump-cursor) n))
```

A batch can span into a slab that hasn't been `malloc`'d yet. Growing `slabs` — appending
a new 64 KiB block's base pointer — is the one operation that genuinely needs mutual
exclusion (it's not a per-object rate operation, so a spinlock is the right tool, and it's
the only lock anywhere in this allocator): a thread whose reserved range crosses into an
unmapped slab CAS-acquires `growing` (0→1), mallocs and appends the slab(s), then releases
(store 0). A thread whose entire reserved range already falls inside existing slabs — the
common case — never touches `growing` at all, just an `atomic-load` of `slab-count` to
check.

**Layer 2 — the per-thread magazine, the actual hot path, zero synchronization:**

```coil
(const MAGAZINE_CAP 32)

(defstruct Magazine [(slots (array i64 32)) (count i64)])   ; a small LIFO cache of free slot
                                                              ; indices, owned by ONE VMThread
```

`VMThread` (§2) gains one more field: `(magazines (ptr (ptr Magazine)))`, indexed by
type-id and lazily created on first use — since it's reached only through this thread's own
`VMThread*`, nothing else ever touches it, so no atomics are needed inside a magazine at
all:

```coil
(defn thread-alloc [(vt (ptr VMThread)) (type-id i64)] (-> (ptr u8))
  (let [a (type-arena type-id) mag (thread-magazine-for vt type-id)]
    (if (icmp-gt (load (field mag count)) 0)
      ; fast path: pop THIS thread's own cache — a decrement and a load, nothing shared touched
      (let [c (- (load (field mag count)) 1)]
        (do (store! (field mag count) c) (arena-slot-ptr a (load (index (field mag slots) c)))))
      (do (magazine-refill a mag) (thread-alloc vt type-id)))))   ; slow path, batches from Layer 1

(defn magazine-refill [(a (ptr TypeArena)) (mag (ptr Magazine))] (-> i64)
  (let [(mut got) (arena-free-pop-batch a (field mag slots) MAGAZINE_CAP)]
    (do (if (icmp-lt got MAGAZINE_CAP)
          ; free list came up short (or empty) — top the rest up from the bump frontier
          (let [first (arena-bump-batch a (- MAGAZINE_CAP got))]
            (for [i 0 (- MAGAZINE_CAP got)]
              (do (ensure-slab-mapped a (+ first i))   ; may CAS-acquire `growing`, see above
                  (store! (index (field mag slots) (+ got i)) (+ first i)))))
          0)
        (store! (field mag count) MAGAZINE_CAP))))
```

`OP_NEW`'s handler stamps the header (`slot-index`, `type-id`, `flags = live`) at the
moment bytecode actually takes ownership of the slot — **not** when it's merely sitting in
a magazine. This is what keeps a magazine's cached slots invisible to enumeration (below)
for free: a freshly `malloc`'d slab is zeroed once, at creation (`grow-arena` now `calloc`s,
not `malloc`s — one `memset` per 64 KiB slab, not per object), so every never-yet-used
slot's `flags` already reads 0 before `OP_NEW` touches it, identical to a properly-freed
slot's `flags`. No extra bookkeeping is needed to hide "reserved but not yet constructed"
slots from a live-instance walk.

Freeing is the mirror image — push onto the thread's own magazine first, only spilling to
the shared free list (Layer 1) when the local cache is full:

```coil
(defn thread-free [(vt (ptr VMThread)) (type-id i64) (idx i64)] (-> i64)
  (let [p (cast (ptr InstanceHeader) (arena-slot-ptr (type-arena type-id) idx))
        mag (thread-magazine-for vt type-id)]
    (do (store! (field p generation) (+ (load (field p generation)) 1))
        (store! (field p flags) 0)
        (if (icmp-lt (load (field mag count)) MAGAZINE_CAP)
          (do (store! (index (field mag slots) (load (field mag count))) idx)   ; local push
              (store! (field mag count) (+ (load (field mag count)) 1)))
          (magazine-drain-half (type-arena type-id) mag)))))   ; return half to Layer 1, keep half
```

`live-count` moves from a plain increment/decrement to `atomic-add`/`atomic-sub`, updated
only by `OP_NEW`/free (never by a magazine refill/drain, which only moves slots between
"shared free" and "thread-cached free" — neither of which is live) — it stays a coarse,
eventually-consistent counter, exactly the "cheap per-type summary" §8's table already
asks of it, not a linearizable one.

Same-size slots per type still means **no cross-type fragmentation is possible** — a freed
`Agent` slot can only ever become a new `Agent`, and that remains true regardless of which
thread's magazine it passes through. Internal fragmentation is still absorbed by the free
list; it's never returned to the OS.

### Enumeration — "list all live Agents"

```coil
(defn arena-for-each-live [(a (ptr TypeArena)) (f (fnptr c [(ptr u8)] i64))] (-> i64)
  (let [cap (ishl 1 (load (field a slab-shift)))
        hwm (atomic-load (field a bump-cursor))    ; snapshot the frontier once, up front
        slab-count (atomic-load (field a slab-count))]
    (for [slab# 0 slab-count]
      (for [local 0 (min cap (- hwm (* slab# cap)))]     ; last slab is partial, others full
        (let [p (index (load (index (load (field a slabs)) slab#)) (* local (load (field a slot-size))))]
          (if (= (iand (load (field (cast (ptr InstanceHeader) p) flags)) 1) 1)
            (call-ptr f p) 0))))))
```

**Cost, stated honestly:** this is `O(high-water-mark)`, not `O(live)`, exactly as before —
that part of the story is unchanged by threading. A type that churns heavily — created and
destroyed thousands of times while only a handful are alive at any moment — still costs
one header-flag check per ever-allocated slot on every enumeration, because slabs are
never compacted or returned. For the demo's entity types (`Agent`, `Conversation`, `Task`)
this is a non-issue: they're created once and live for the process. For a hypothetical
high-churn type (e.g. a `ToolCall` per tool invocation in a long-running agent), this is the
first thing that would show up in a profile, and the fix (§6, compaction) is deferred, not
solved. Magazines add one honest wrinkle to the constant, not the complexity class: slots
sitting in an idle per-thread magazine still count toward `bump-cursor` (they were bump-
reserved to get there) even though they hold no live instance, so the high-water mark can
now run up to `MAGAZINE_CAP × thread-count` ahead of where a single-threaded allocator
would have left it — bounded and small (32 × a handful of agent threads), not unbounded.

**What "consistent" means for a concurrent walk, and who's allowed to do one (§7 decides
this, this is where the mechanism lives):** `arena-for-each-live` never touches `free-head`
or the magazine layer — only `slab-count`, `bump-cursor` (both read via `atomic-load`
above, as shown), and each slot's own `flags` word. `flags`, like `type-id`/`generation`/
`shape-id`, is a `u32` field, and Coil's `lib/atomic.coil` has no `u32` atomic-load
primitive to reach for — only `(ptr i64)` atomics plus the generic pointer-typed family
(`atomic-load-ptr`/`atomic-cas-ptr` over `(ptr T)`) — so `flags` is read with a **plain
load**, as the code above actually does. That's fine, not a hole: a `u32` load/store is a
single, naturally-aligned 4-byte access, inherently non-tearing on this architecture — the
same non-tearing argument §7 already makes for ordinary field reads during a read-only
eval. A walk can therefore observe a slot's `flags` transition (0 → live, or live → 0 on
free) at any point mid-scan without ever seeing a torn, half-written value; it just isn't
linearizable against the rest of the object's fields, which the walk never reads anyway.
That means a walk is safe to run **while other threads are still allocating and
freeing**, with one exception: it must not race `grow-arena`
appending a new pointer into `slabs` mid-read of that array. §7's read-only evals take the
same per-arena `growing` spinlock allocation already uses for exactly that reason — a
short, per-type, rarely-contended lock, not a global stop. A full stop-the-world (GC sweep,
§6; shape migration, §4's subsection below) makes the question moot by construction: no
other thread is touching this arena at all while it runs.

### Identity, handles, and why they matter

**Internal object-graph references are raw pointers**, exactly like clox's `(ptr Obj)`.
`Agent_Instance.conversation` points directly at a `Conversation_Instance`. This is safe
*by construction*: a mark-sweep collector never frees an object that's still reachable
through a field, so a live object can never hold a dangling pointer to a dead one. Internal
field access is one dereference, same cost as clox, no regression.

This is a deliberate, load-bearing choice, made jointly with `03-live-semantics.md`:
Scry does **not** put a handle table between class-typed fields/locals and the objects
they refer to — not permanently, not lazily, not ever. That would tax every single field
access (the hottest path in the VM, per §2/§3) to buy something only one feature actually
needs — shape migration relocating live instances to a new-stride slab without corrupting
every other object's pointer to them. Instances move *only* during a shape migration, and
identity across that move is preserved by same-index copy, not by indirection — see
"Shape migration" below, later in this section, for the mechanism (it reuses §6's
graph-rewrite pass, run once at a safepoint, and leaves the resting representation
untouched).

**The viewer is different.** A handle like `"Agent#7"` can sit in a browser tab for
minutes. In that window the referenced `Agent` might be freed and its slot **reused by a
different Agent**. A raw pointer can't tell you that happened; a stale pointer would
silently show you the wrong live object. So the *only* place identity needs to be more than
a pointer is exactly the boundary DECISIONS.md calls out: **"stable ids the viewer can hold
across GC."**

The wire-facing handle is `{type-id, slot-index, generation}` — packed as a `type-id` (from
`type-table`) plus the `slot-index`/`generation` already sitting in `InstanceHeader`. To
resolve a handle back to a pointer: `arena-slot-ptr(type-table[type-id].arena, slot-index)`,
then compare `generation` against what's actually stored there. Mismatch means "that
instance is gone" — a clean, detectable failure mode, not a crash:

```json
→ {"id": 12, "source": "Agent#7"}
← {"id": 12, "value": {"live": false}}
```

Note the honest refinement to `00-vision.md`'s wire examples: the human-visible id stays
`"Agent#7"` (type name + slot index), but the serialized instance-ref value that actually
crosses the wire carries `generation` too (as a hidden field the viewer round-trips, not
something a person types) — without it, "click a link, the object was freed and the slot
reused, you silently see someone else's data" is a real bug, not a corner case, given the
demo explicitly churns `Message`/`ToolCall` instances during the live session.

**One more consequence worth stating plainly: a handle held by the viewer does not keep
the object alive.** If nothing in the running program still references `Agent#7`, GC
collects it whether or not a browser tab has its detail pane open. The viewer must treat
"live: false" as a normal, expected state, not an error — this falls directly out of the
"no image, ordinary process" stance in `00-vision.md`: the viewer is a **weak**, external
observer, never a root.

**`List<T>` is not an entity type.** Only `class` declarations get an arena. A `List<Tool>`
is an ordinary growable buffer (`lib/arraylist.coil`'s shape: malloc'd backing array, `len`,
`cap`), living on the general heap, not in a slab. It is not independently browsable in the
viewer (you browse it through the `Agent` that owns it); its elements, if reference-typed,
are raw pointers the GC must trace when it traces the object that holds the list (§6, same
as clox's `mark-array` over `ValueArray`).

### Shape migration — the structures `03-live-semantics.md` assumes exist here

`03-live-semantics.md`'s entire field-add/remove design is written "as if `02-runtime.md`
already existed" — it needs shape-versioned method tables and a per-instance shape id,
none of which has a home above: `InstanceHeader` has no shape field, and
`TypeInfo`/`TypeArena` model exactly one fixed `instance-size` for a type's whole
lifetime, with no notion of two shapes or two method tables coexisting. This subsection is
that home. The vocabulary is shared: `TypeInfo`/`ShapeInfo`/`FieldInfo`/`TypeArena` below
are the same structures `03-live-semantics.md` names, one definition, defined here.

**A caveat this forces on the "raw pointers" position above:** the "internal references are
raw pointers, safe by construction" claim a few paragraphs up is only true at rest — for a
type that has never had a live instance moved out from under it. A field-add/remove
migration (03's mechanism) copies each live instance from the old-stride slab set into the
new-stride one at the *same slot index*; a raw pointer captured before the copy would now
point at stale, about-to-be-freed memory. This doc does **not** resolve that by making
every class-typed field a handle (an extra indirection on *every* field access, forever,
to guard against a migration that most types will never undergo). Instead, migration reuses
the mechanism §6 already named and deferred for compaction: *"moving a live instance means
rewriting every pointer to it, which requires exactly the same 'walk the graph with
`TypeInfo`' pass mark already does, plus a second pass to fix up."* A field-add/remove
migration is exactly that pass, scoped to one type, run at a safepoint so no bytecode is
mid-read while it happens. Ordinary field access stays a raw pointer dereference, always;
the cost of a move is paid once, at migration time, not on every read.

**`TypeInfo` and `ShapeInfo`.** A type's field layout is pulled out of `TypeInfo` into its
own struct so two can exist for one type during a drain window:

```coil
(defstruct ShapeInfo
  [(shape-id       i64)
   (instance-size  i64)              ; stride for this shape specifically
   (fields         (ptr FieldInfo)) (field-count i64)
   (methods        (ptr MethodInfo)) (method-count i64)])  ; this shape's compiled bytecode

(defstruct TypeInfo
  [(type-id         i64)
   (name (ptr u8)) (name-len i64)
   (shape           (ptr ShapeInfo))       ; current shape — the flat, common case
   (old-shape       (ptr ShapeInfo))       ; null unless a migration is draining
   (pending-shape   (ptr ShapeInfo))       ; null unless a migration is draining
   (active-frames   i64)                   ; §3-live-semantics' quiescence counter — atomic
                                            ; now (NEW): every thread's ENTER_METHOD/RETURN
                                            ; touches it, not just one interpreter's, §7
   (itables         (ptr (ptr ITable)))    ; carried over from §3's TypeInfo — a shape
                                            ; migration never touches interface dispatch,
                                            ; since `implements` is fixed at compile time
   (arena           (ptr TypeArena))])
```

When `old-shape` is null, dispatch and field offsets are exactly what §2/§3 already
describe — one flat `shape.methods[slot]`, one flat set of `FieldInfo` offsets, zero extra
cost. `old-shape`/`pending-shape` are populated only for the duration of one migration.

**`InstanceHeader` gains a shape id**, on top of the `type-id` §3 already added for
interfaces — every instance has to say which shape it's on, because during a drain window
two live instances of the same `type-id` can disagree:

```coil
(defstruct InstanceHeader
  [(slot-index i64)
   (generation u32)
   (type-id    u32)      ; from §3, unchanged by migration — identifies the CLASS
   (shape-id   u32)      ; NEW here — which ShapeInfo this instance's fields are laid out per
   (flags      u32)])
```

**`TypeArena` holds two live slabs during a migration window.** Enumeration (`arena-for-
each-live`, §4 above) and allocation must both know there can be an old, frozen, draining
slab and a new, growing one at once:

```coil
(defstruct TypeArena
  [(type-id i64)
   (slot-size i64) (slab-shift i64) (slabs (ptr (ptr u8))) (slab-count i64)
   (bump-cursor i64) (free-head i64) (growing i64) (live-count i64)   ; §4's thread-safe fields
   ; migration-only fields — all zeroed/null when no migration is pending:
   (old-slot-size  i64)             ; 0 when not migrating
   (old-slabs      (ptr (ptr u8)))  ; the frozen, draining slab set — no new allocs here,
                                     ; from ANY thread's magazine, while it drains
   (old-slab-count i64)
   (old-live-count i64)])           ; atomic — counts down to 0 as instances are copied over
```

The flip happens the instant a field-changing edit is accepted: the existing slab set moves
to the `old-*` fields, and a fresh current side is allocated at `pending-shape`'s stride.
New instances always allocate from the *current* (`slot-size`/`slabs`) side — with its bump
pointer started *above* the old side's high-water mark, so slot indices `0..old-HWM` stay
reserved for the same-index copy the migration pass below performs (free slots among them
rejoin the free list after the copy completes). Existing instances on the old shape stay
exactly where they are — read, written, and enumerated at their old stride — until the
quiescence gate below fires.

**The quiescence gate and the migration pass**, using `TypeInfo.active-frames` and the
bytecode this doc's dispatch loop already has to emit (this is 03's mechanism; it belongs
here because it's bytecode, not semantics):

```
ENTER_METHOD <type-id>   -> atomic-add(&TypeInfo[type-id].active-frames, 1)   ; NEW: atomic —
RETURN                   -> if atomic-sub(&TypeInfo[type-id].active-frames, 1) == 1:  ; every
                              if pending-shape != null: request-migration(type-id)     ; thread's
                                                                                        ; own calls
```

`active-frames` is now touched by every language thread that calls or returns from a method
on this type, concurrently, so the plain increment/decrement an earlier, single-threaded
draft of this doc used is exactly the kind of thing §7 forbids — it becomes `atomic-add`/
`atomic-sub`, and "drops to zero" is read off the value `atomic-sub` itself returns (1,
meaning the counter *was* 1 and is now 0), not a separate load-after-store that another
thread could race.

`run-migration(type-id)` runs at a **full stop-the-world safepoint** — §7's generalized,
N-thread version of the mechanism, not a single-interpreter one. The thread whose `RETURN`
drove `active-frames` to zero becomes the coordinator: it calls §7's `request-global-stop`,
which blocks until every *other* live `VMThread` has parked itself at its own
`safepoint-poll` (or is itself the coordinator and skips waiting on itself), then runs the
two steps below with the entire object graph quiescent — no other thread is executing
bytecode, allocating, or freeing anywhere — then calls `release-global-stop`. It does, in
order:

1. **Same-index copy.** For each live instance at slot index `i` in the old slab set,
   compute its new field values (default or migration-fn, per 03) and copy into slot index
   `i` of the current-side slab set — the slot reserved for it above, **never a fresh
   allocation** — stamping the new `shape-id` into its header.
   `InstanceHeader.slot-index` is unchanged and `generation` is **not** bumped (only
   freeing bumps it, §4), so the wire-facing identity `{type-id, slot-index,
   generation}` survives the move untouched: `Agent#7` is still literally `Agent#7`, with
   no lookup table anywhere.
2. **Pointer rewrite** — the pass §6 defers for compaction, done here for one type, now
   over **every thread's** roots (this is the change real threads force on this step,
   `DECISIONS.md` #4b): walk every arena's live instances via `TypeInfo`/`FieldInfo` exactly
   like `blacken-object` (§6) does for GC marking, plus, for **every** `VMThread` in
   `vm.threads[0..thread-count]` (not just one), that thread's own stack's ref slots
   (`MethodInfo.ref-slots`, the same bitmap §6's `mark-roots` reads, generalized the same
   way), plus the globals table; for every class-typed slot holding an address inside the
   migrating type's old slab set, rewrite it to the same-index address in the new set. Old
   and new addresses are both pure functions of `slot-index` (§4's `arena-slot-ptr`), so the
   rewrite is address arithmetic — recover `i` from the old address, emit the new-set
   address for `i` — no lookup table, no remembered set. This is safe precisely because
   every thread is parked: no stack is being pushed to, no register holds a not-yet-spilled
   copy of a rewritten pointer, mid-rewrite.

Old-shape slabs are freed once every live instance has been copied; `old-live-count`
reaching 0 is what lets `run-migration` tear the old slab set down. Handle resolution
stays `arena-slot-ptr` plus the `generation` compare through any number of migrations,
with one drain-window wrinkle: while `old-slot-size != 0`, a `slot-index` below the old
high-water mark resolves into `old-slabs` (that's where those instances still physically
are); the whole copy happens inside `run-migration`'s safepoint, so from the wire's point
of view resolution flips old-set → new-set in one atomic step, same triple throughout.

**Dispatch during the drain window.** §2's `OP_CALL_STATIC` comment ("no dispatch, direct
call") is accurate only when `old-shape` is null. The moment a migration is pending,
`OP_CALL_STATIC`'s operand is a method-slot index, not a baked-in address, and the callee
is resolved as `instance.header.shape-id == current.shape-id ? shape.methods[slot] :
old-shape.methods[slot]` — one branch and one indirect load, not a hash lookup, and it
collapses back to the flat, zero-cost form the instant the drain completes and `old-shape`
goes null again. This is the same "pay only while it's actually in flight" shape as §7's
`safepoint-poll` check: every other type, and this type once its drain finishes, never
pays it.

**Cost, stated honestly, matching this doc's own convention (§4's enumeration-cost note):**
a migration is `O(live instances of that type)` for the copy, plus `O(total live objects
across every arena, plus every thread's stack slots, plus globals)` for the pointer-rewrite
pass — the whole live heap **and now every language thread's stack**, honestly — run once
per accepted field-add/remove edit — not on any hot path, and bounded by how often a
developer saves that kind of edit, not by request rate. Real threads add one more honest
line item this doc didn't previously have to pay: every migration (and every GC, §6) now
also pays the cost of bringing every other thread to a safepoint before it can start, on
top of the walk itself — for a handful of demo agent threads this is not a real cost, but
it is a real, structural one that would matter at a larger thread count. This is the real,
buildable project §6 already flagged as deferred for general compaction; shape migration is
the first concrete customer for it.

## 5. Interfaces and generics — how they touch the arena model

**Interfaces are IN** (`DECISIONS.md` #4, Jimmy ruling), nominal and explicit (`class
ShellTool implements Tool { … }`), interface types usable anywhere a type can appear
(`Agent.tools: List<Tool>`), with dynamic dispatch through them — the polymorphism story
for the PoC, superseding the enum-dispatch fallback earlier drafts of `01-language.md`
proposed for `Agent.tools`. This section makes `OP_CALL_VIRTUAL` (§2, reserved until now)
real.

### Position: plain pointers + a per-instance type tag, not a fat pointer

An earlier draft of this doc assumed that *if* interfaces landed, an interface-typed value
would need to become a fat `{data, vtable}` pointer — Rust's trait-object shape — because
that's the standard answer when a reference has to carry its own dispatch information.
Having actually designed it, that's the wrong call for this object model specifically, and
the reason is §3's arena-origin trick generalizing one step further than it first looks
like it does:

- A **class**-typed field's concrete type is always known at compile time — no tag needed,
  a plain pointer suffices, dispatch is `OP_CALL_STATIC` to a baked-in address.
- An **interface**-typed field's concrete type is *not* known at compile time (a
  `Tool`-typed slot might hold a `ShellTool` or a `SearchTool`) — but it *is* known at
  *runtime*, the instant you have the pointer, **if the pointee can tell you its own type**.
  Every instance already has a header at a fixed offset (§3); adding one `type-id` word to
  it (already done in §3, motivated jointly by this section and by GC, below) means **any**
  pointer, interface-typed or not, can self-report its concrete type with one load. That
  makes the fat pointer's second word redundant — it would only ever be recomputing
  something the pointee already carries.
- So: an interface-typed field or local is **the same single-word raw pointer** a
  class-typed one is. Nothing about storage changes size, alignment, or field layout for
  holding an interface-typed value. The entire cost of interfaces lands on `InstanceHeader`
  (4 bytes, paid once, §3) and on the dispatch instruction itself (below), not on every
  reference that happens to be interface-typed.

### The itable: computed per (class, interface) pair at compile time

Because `implements` is nominal and fully known at compile time — no structural/duck typing,
no runtime interface discovery — the compiler can assign every interface method a stable
**slot number** (declaration order in the `interface` body, exactly like a C++ vtable slot
or a Java itable slot) and, for every `class C implements I`, emit one `ITable` mapping
`I`'s slots to `C`'s compiled method addresses:

```coil
(defstruct ITable [(methods (ptr (fnptr c [(ptr u8)] i64)))])  ; methods[slot] for interface I,
                                                                 ; as implemented by ONE class

; TypeInfo.itables (§3) is indexed directly by interface-id — flat, not a search structure:
; itables[iface-id] is null if this class doesn't implement that interface, else -> ITable.
```

A flat, interface-id-indexed array trades a little static memory (most entries null, for a
class implementing few interfaces out of however many the whole program declares — a
non-issue at the PoC's scale of one or two interfaces) for **O(1)** dispatch: no per-call
search over "which interfaces does this class implement," unlike JVM `invokeinterface`,
which historically resolves via a runtime search (real JVMs cache it per call site; this
doc gets the same end state — one indexed load — without needing a cache to get there,
because the whole table is static).

### `OP_CALL_VIRTUAL`, made real

```coil
; operands: iface-id (constant), method-slot (constant), argc
OP_CALL_VIRTUAL (let [self (peek-ptr argc)                                   ; the receiver,
                                                                                ; already on
                                                                                ; the stack
                       tid  (load (field (cast (ptr InstanceHeader) self) type-id))
                       it   (load (index (load (field (type-table-entry tid) itables)) iface-id))
                       fn   (load (index (load (field it methods)) method-slot))]
                   (call-ptr fn self …args…))
```

Four memory operations beyond an ordinary call: one header load (`type-id`), one indexed
load into `TypeInfo.itables` (`iface-id`), one indexed load into the `ITable` (`method-
slot`), one indirect call — genuinely comparable to a C++ virtual call, not to a hash-based
or linear-scan dispatch, precisely *because* the nominal, compile-time-checked `implements`
relationship (`DECISIONS.md` #4) is what makes static slot assignment possible in the first
place. This is real, machine-level cost `OP_CALL_STATIC` doesn't pay — stated honestly as
the price of polymorphism, same spirit as every other honest-cost note in this doc.

### Interface-typed fields, storage, and GC

`Agent.tools: List<Tool>` is a `ListObj` (§4's "not an entity type" note, unchanged) whose
elements are single-word pointers, `kind = FIELD_IFACE_REF` in the list's element metadata
(the list itself isn't an instance with a `FieldInfo`, but the same kind tag applies to its
elements — see §3's `FIELD_IFACE_REF`). Tracing one is: dereference to the header (fixed
layout, works on *any* instance regardless of concrete type), read `type-id`, look up
`TypeInfo` in `type-table`, and recurse via that type's *own* `FieldInfo`s — this is exactly
why the header needed `type-id` for a second, independent reason beyond dispatch (§3
already flagged this): `blacken-object` reaching an object through an interface-typed field
has no other way to know which `TypeInfo` to iterate, unlike a class-typed field, whose
`FieldInfo.elem-type-id` already says so at compile time.

### Generics need no new runtime representation

Unchanged from the earlier position: generics are handled by monomorphization at compile
time (Coil's own `[T]` model, reused directly — `(defn id [T] [(x T)] …)` is specialized per
instantiation, not boxed). `Inventory<Tool>` and `Inventory<Agent>` are two distinct
compiled types with two distinct arenas, same as any other class. This keeps the "one arena
per concrete type" property intact, and it composes cleanly with interfaces: `Inventory<T>`
where `T` is later instantiated as an interface type (`Inventory<Tool>`) still stores plain
single-word pointers per element, per the position above — no interaction effect, no
special case.

### Default methods (`DECISIONS.md`: OPEN, lean no for PoC)

Not built. If added later, the itable model above accommodates them without a redesign: a
default method is simply a method the interface itself provides an implementation for, so
an implementing class's `ITable` slot for it, if the class doesn't override, points at the
interface's own default body instead of a per-class one — same `ITable` shape, populated
differently at compile time. Flagged here, not decided, per the ruling.

## 6. Garbage collection

Mark-sweep, per DECISIONS.md, fanned out over N per-type arenas instead of clox's one
global object list — and, with real threads (`DECISIONS.md` #4b), now a genuine **stop-
the-world** collector over every `VMThread`, not just the one interpreter earlier drafts
of this doc assumed. Whichever thread's allocation crosses the trigger below becomes the
GC coordinator: it calls §7's `request-global-stop`, which blocks until every *other* live
thread has parked at its own `safepoint-poll`, runs mark + sweep below with the whole heap
quiescent, then calls `release-global-stop`. This is the same protocol §4's shape migration
uses (§7 defines it once, both consume it) — GC and migration are two different *triggers*
for the identical "stop everyone, do a bounded amount of graph work, resume everyone" shape.

**Roots are precise, not conservative, and now span every thread.** clox has to treat
every stack slot as a potential `Value` and check `is-obj` on it, because Lox has no
static types. Scry knows, per method (`MethodInfo.ref-slots`, §2), exactly which stack
slots hold a pointer — declared locals *and* operand-stack temporaries alike, since §2's
"What `ref-slots` actually covers" subsection sizes `local-count`/`ref-slots` over the
method's whole frame extent, not just its declared locals, precisely so a scan like this
one also catches a value like a freshly-`OP_NEW`'d pointer still sitting on the operand
stack (§2's worked example). Marking roots walks **every `VMThread`'s** live call frames —
the one change real threads force on this function, since each thread now owns its own
frame array and stack (§2) instead of there being a single shared one — and only looks at
the bits set in `ref-slots`:

```coil
(defn mark-roots [] (-> i64)
  (for [t 0 (atomic-load (field (vm) thread-count))]
    (let [vt (index (load (field (vm) threads)) t)]
      (for [i 0 (load (field vt frame-count))]
        (let [fr (index (field vt frames) i) m (load (field fr method))]
          (for [slot 0 (load (field m local-count))]
            (if (bitmap-test (load (field m ref-slots)) slot)
              (mark-object (load (cast (ptr (ptr u8)) (index (field vt stack) (+ (load (field fr base)) slot)))))
              0))))))
  (mark-globals-table)
  0)
```

This is safe to run exactly *because* every thread is parked when it runs — no frame is
being pushed to, no `ref-slots` bit is changing underfoot, for the entire duration of the
walk. It is also, honestly, the most direct cost real threads add to a GC pause: a pause
now stops N threads' worth of useful work simultaneously, not one — see the trigger note
below for the corresponding upside/downside tradeoff this doc is making for v1.

**Marking follows `TypeInfo`, not a hand-written visitor per class** — another payoff of
building the reflection table anyway: `blacken-object(p)` looks up `p`'s own arena's
`type-id` in `type-table` when reached via a class-typed field (the concrete type is
already known statically, per `FieldInfo.elem-type-id`), **or reads `type-id` straight off
`p`'s own header** when reached via a `FIELD_IFACE_REF` field (§5 — the concrete type is
*not* known statically there, that's the entire reason the header carries it), then
iterates that `TypeInfo`'s `FieldInfo`s and, for every `FIELD_REF`/`FIELD_LIST_REF`/
`FIELD_IFACE_REF` field, loads the pointer at that offset and marks it. One generic tracer
for every class, instead of clox's per-`ObjKind` `switch` in `blacken-object` — and, notably,
the *same* generic tracer whether the field it's tracing is a plain class reference or an
interface reference, because both are, at rest, a single pointer (§5).

**Sweep is enumerate-and-conditionally-free**, per arena, reusing §4's
`arena-for-each-live` shape: walk each type's slabs in slot order; a slot whose `mark` bit
is set gets the bit cleared (ready for next cycle) and survives; a slot that was live but
unmarked is pushed straight onto that arena's shared free list (`arena-free-push`, §4) —
its `generation` bumps, invalidating any viewer handle that pointed at it. Sweep pushes to
the *shared* free list directly, not through any one thread's magazine, deliberately: the
collector isn't "on behalf of" any particular language thread, and every thread is parked
anyway, so there's no locality to preserve and no contention to avoid — the CAS-loop push
is uncontended, a single winner every time, during a sweep. Global mark phase runs once
across all arenas (it has to — the object graph crosses arena boundaries), then sweep runs
arena-by-arena, independently.

**Trigger:** one global `bytes-allocated` / `next-gc` pair, same heuristic as clox
(`GC_HEAP_GROW_FACTOR`), summed across all arenas — simplicity over per-type precision for
v1; a type that's unusually hot could get its own threshold later if profiling calls for
it, but that's a tuning knob, not an architecture change. Checked opportunistically on a
magazine refill (§4), not on every single fast-path allocation, for the same reason the
fast path avoids atomics at all: precise triggering isn't worth taxing the hot path for.

**Compaction: not in v1, and here's why it's not a soundness gap.** Same-size slots per
type mean external fragmentation across types is structurally impossible (§4), and
internal fragmentation is absorbed by the free list. Compaction would only buy back
enumeration cost on high-churn types (the honest cost flagged in §4) and slab memory
returned to the OS. It's not free to add later: moving a live instance means rewriting
every pointer to it, which requires exactly the same "walk the graph with `TypeInfo`" pass
mark already does, plus a second pass to fix up. That's a real, buildable project — the
metadata to do it precisely already exists — it's just not worth building *for GC's own
sake* before there's a profiled workload that needs it. Ship mark-sweep-with-free-lists
first. **Parallel marking is likewise deferred, and now a real, not hypothetical, gap:**
with N threads parked and idle during a GC pause, only the coordinator thread does any
marking or sweeping work — the other N−1 threads' CPU cores sit unused for the pause's
duration. Splitting the root set or the per-arena sweep across the now-parked threads
(they could mark instead of merely waiting) is a legitimate follow-up once a profiled
workload shows GC pause time actually matters; v1 ships the simple, single-coordinator
version and states plainly that it leaves parallelism on the table.

**This is a different thing from shape migration, which *is* in v1.** GC compaction
(deferred above) is "move objects around to reclaim memory/enumeration cost, driven by
the collector's own heuristics, whenever it wants." `03-live-semantics.md`'s shape
migration is narrower and unavoidable for the PoC: driven by a specific field add/remove
edit, for one class, at a quiescence point that class's own `active_frames` counter
already gates. It reuses the identical rewrite-pass mechanism described above (walk the
live graph via `TypeInfo`, find every pointer into the class's old slab, repoint it at the
new one), but it ships as part of live-edit support (`DECISIONS.md` #9), not as a general
GC feature — so "compaction is deferred" above should be read as "the *collector*
triggering a move on its own initiative is deferred," not as "nothing in v1 ever moves a
live instance." See `03-live-semantics.md`'s "Field add / remove" section for the
migration-specific trigger and per-instance failure handling; the pointer-rewrite
mechanics are this section's.

## 7. Interpreter threads, safepoints, and the embedded HTTP/WebSocket server

**Position: N real interpreter threads, one separate server thread, and a generalized
safepoint protocol that any of them can use to bring every *other* one to a stop.**
`DECISIONS.md` #4b (Jimmy ruling: "proper threads") supersedes this doc's earlier
single-interpreter-thread position outright — every Scry-level thread (the demo app spawns
one per `Agent`) is a genuine `pthread`, running §2's dispatch loop against its own
`VMThread`. This section is the deepest rewrite in the document because almost everything
else in it — the allocator (§4), the GC (§6), shape migration (§4) — was built assuming a
single mutator and now has to cope with N.

### `VMThread` spawn/join — grounded directly in Coil's pthread wrapper

`lib/thread.coil`'s `Thread`/`thread-spawn`/`thread-join` (a thin wrapper over
`pthread_create`/`pthread_join`) are exactly the primitive a language-level `Thread.spawn`
needs, with one twist: the pthread body function's `(ptr i8)` argument is how the new OS
thread finds *its own* `VMThread*` — Coil has no implicit thread-local storage, so rather
than invent one, the pointer is passed down explicitly, the same explicit-parameter style
`run` already uses (§2):

```coil
(defn spawn-language-thread [(entry (fnptr c [(ptr u8)] i64)) (arg (ptr u8))] (-> i64)
  (let [vt (unwrap-ptr [VMThread] (create [VMThread] (heap-allocator)))]
    (do (store! (field vt frame-count) 0) (store! (field vt stack-top) 0)
        (store! (field vt parked) 0)
        (register-thread vt)                          ; reserve a slot in vm.threads, below
        (push-entry-frame vt entry arg)                ; frame 0: call entry(arg), OP_CALL_STATIC's
                                                         ; own setup, reused verbatim
        (thread-spawn (field vt os-thread) (fnptr-of interpreter-thread-main) (cast (ptr u8) vt)))))

(defn interpreter-thread-main [(arg (ptr i8))] (-> (ptr i8))
  (do (run (cast (ptr VMThread) arg)) (cast (ptr i8) 0)))   ; §2's dispatch loop, this thread's own

(defn register-thread [(vt (ptr VMThread))] (-> i64)         ; lock-free: one atomic-add reserves
  (let [idx (atomic-add (field (vm) thread-count) 1)]         ; this thread's slot, same bump-
    (do (store! (field vt thread-id) idx)                     ; reservation pattern as §4's arena
        (store! (index (field (vm) threads) idx) vt) 0)))

(defn join-language-thread [(vt (ptr VMThread))] (-> i64) (thread-join (field vt os-thread)))
```

`vm.threads` is a fixed-capacity array (64, matching the style of `CallFrame`'s fixed 256 —
this doc's convention throughout is bounded arrays over dynamic growth wherever the bound
is small and known), so spawning is genuinely lock-free: no CAS loop, no lock, just one
`atomic-add` to claim a slot, identical in spirit to §4's arena-bump reservation.

### The generalized safepoint protocol — one stop-flag, N parked threads

Every interpreter thread's `safepoint-poll` (§2, called once per bytecode instruction) is
no longer "check the viewer queue" — it's "park myself if anyone has requested a global
stop, and don't move until they're done":

```coil
(defn safepoint-poll [(vt (ptr VMThread))] (-> i64)
  (if (icmp-eq (atomic-load (field (vm) stop-flag)) 1)
    (do (atomic-store (field vt parked) 1)
        (loop (if (icmp-eq (atomic-load (field (vm) stop-flag)) 0) (break) (spin)))
        (atomic-store (field vt parked) 0) 0)
    0))
```

The requester's side — used identically by GC (§6), shape migration (§4), and the eval
thread below, one mechanism for all three coordinators:

```coil
(defn request-global-stop [(self (ptr VMThread))] (-> i64)
  (loop (if (icmp-eq (atomic-cas (field (vm) stop-flag) 0 1) 0) (break) (spin)))  ; no nested
                                                    ; stops in v1 — see the honest gap below
  (for [i 0 (atomic-load (field (vm) thread-count))]
    (let [vt (index (field (vm) threads) i)]
      (if (!= vt self) (loop (if (icmp-eq (atomic-load (field vt parked)) 1) (break) (spin))) 0))))

(defn release-global-stop [] (-> i64) (atomic-store (field (vm) stop-flag) 0))
```

This is a plain spin-wait, not a futex/condvar-based park — deliberately: Coil's
concurrency surface (`lib/thread.coil`, `lib/atomic.coil`) is `pthread_create`/`join` plus
`atomicrmw`/`cmpxchg`/atomic load-store, with no mutex or condition-variable wrapper today.
A spin loop over an atomic flag is the correct, minimal thing to build on top of exactly
that primitive set — the same choice `examples/lockfree.coil` already makes for its
CAS-retry loops — and is fine at the PoC's scale (a handful of agent threads, human-paced
viewer/GC events, not a hot loop under contention).

**Honest gap: no nested/reentrant global stop in v1.** If a GC trigger fires while a shape
migration (or a viewer-triggered definition eval) already holds `stop-flag`, the later
requester simply spins on the CAS until the earlier one releases — safe, but a real
serialization point this doc doesn't try to make concurrent. Two coordinators never do
their walks at once; they queue.

### What this means for `01-language.md`'s `async`/`await` — now resolved, not proposed

An earlier draft of this section spent considerable space arguing that a single shared
`frames`/`stack` couldn't represent several independently-suspended call stacks, and picked
"the PoC never suspends mid-call" as the way out, pending a future fiber design if one were
ever needed. Real threads (`DECISIONS.md` #4b) resolve this a different way than either
option that draft considered: **each Scry-level thread already gets its own real stack, for
free, from the OS** (`VMThread`, §2) — there is no shared stack for three concurrently-live
agents to contend over in the first place. `async`/`await` as a *surface-syntax* concern is
explicitly **post-PoC** (`DECISIONS.md` #4b: "async stuff can come later") — not because
this VM can't represent suspension, but because the PoC's answer to "how do three agents
run concurrently" is simply "three real threads," full stop, no cooperative scheduler
underneath it to design. If/when `async`/`await` lands later as sugar *within* a single
Scry-level thread (e.g. non-blocking I/O multiplexed onto fewer OS threads than agents),
that would need something like the old draft's fiber design (suspending a logical task
independently of its underlying pthread) — a genuinely different, harder problem than "give
each thread its own stack," and out of scope here.

### Grounding the server thread — and where that grounding runs out

Grounded in what Coil actually supports (`lib/thread.coil`: `pthread_create`/`pthread_join`
wrappers; `lib/atomic.coil`: `atomic-cas`/`atomic-load`/`atomic-store`;
`examples/lockfree.coil`: a proven CAS-loop Treiber stack) — the primitives for "several
OS threads talking through lock-free structures" already exist and are already exercised in
this codebase, and generalize cleanly from "one worker thread" to "N agent threads plus one
server thread," since nothing about `pthread_create`/atomics is inherently one-writer.

**Where that grounding runs out: there is no HTTP or WebSocket precedent anywhere in
Coil.** A search across `lib/`, `examples/`, and `apps/` for socket, HTTP, WebSocket,
SHA-1, or base64 support turns up nothing but unrelated README URLs. Unlike
threads/atomics above, "the runtime embeds a small HTTP + WebSocket server" — the claim
`00-vision.md`, this doc, and `04-viewer.md` all rest on — has no equivalent existing
precedent to point to. Concretely, still to build, from nothing:

- **Raw BSD socket externs** (`socket`/`bind`/`listen`/`accept`/`read`/`write`/`close`) via
  Coil's `extern` FFI — the same mechanism `lib/thread.coil` already uses for `pthread_*`,
  so mechanically routine, but not written anywhere in this codebase yet.
- **An HTTP/1.1 request-line + header parser**, from scratch — no existing Coil parser to
  extend.
- **The WebSocket handshake, RFC 6455** — requires **SHA-1** (roughly a hundred lines of
  bit ops over the message schedule; feasible with Coil's bitwise/metal ops per `coil
  guide`, but it does not exist in any Coil library or example today) and **base64**
  encoding of the resulting digest, neither of which exists anywhere in this codebase.
- **WebSocket frame parsing, including client-to-server masking/unmasking** per RFC 6455 —
  a wire-correctness bug here (e.g. a masking-key off-by-one) is silent data corruption,
  not a crash.

This is a hand-rolled crypto primitive plus a wire-protocol parser, sitting underneath
every milestone from M1 onward — a substantially different risk profile than
threads/atomics, which are proven precedent, not new build. It deserves the same explicit
risk treatment `05-milestones.md` already gives M2's safepoint mechanism, not a single
undifferentiated IN bullet; and a plain HTTP long-polling or SSE transport (no handshake
crypto at all) is a legitimate way to de-risk M1 by deferring the WebSocket upgrade to a
later milestone once the simpler transport is proven. Both of those are
`05-milestones.md`'s call to make, not this doc's, but the gap belongs on the record here,
next to the grounding that *is* solid.

**Why sharing arenas across N interpreter threads is fine now, when an earlier draft said
it wasn't:** that draft's reasoning was correct for what it described — a lock-free
`arena-alloc`/`arena-free` doing plain, non-atomic reads and writes really is only safe
single-writer. The fix isn't "add a lock and eat the cost on every allocation," it's the
layered design §4 now builds: the shared arena (bump-cursor, free-list) is touched only in
batches, through a handful of atomics and one narrow spinlock, and the actual hot path —
what every `OP_NEW`/free pays — is a per-thread magazine with zero synchronization at all.
N threads sharing the arenas is exactly what §4's magazine layer is *for*; it isn't a
retreat from the single-writer design, it's the thing that makes multi-writer viable
without paying a lock on every object.

### The server thread, and the eval channel's interleaving with N running threads (`DECISIONS.md`: OPEN — position taken here)

`DECISIONS.md` leaves open how the viewer's eval channel interleaves with running threads,
suggesting reads can likely run at a partial safepoint while definitions need full STW.
Having worked through §4's concurrent-enumeration argument above, this doc takes a concrete
position rather than leaving it purely to `04-viewer.md`:

**Read-only evals do *not* need a full stop.** `Agent.instances()`, `Agent#7`, a field
read — these only call `arena-for-each-live` or a single `arena-slot-ptr` + generation
check (§4), both of which are safe to run concurrently with every language thread's
ordinary alloc/free/field-write traffic, *because* every individual field write is a
single aligned word store (§1: untagged, fixed-width slots) — inherently non-tearing at
that granularity on this architecture — and enumeration itself only reads
`atomic-load`-guarded counters plus per-slot `flags`. **The server thread runs these
directly, on its own stack, with no `VMThread` of its own** — no queue round-trip, no
waiting on any agent thread — taking only the one lock that genuinely matters: §4's
per-arena `growing` spinlock, held briefly if a concurrent allocation is in the middle of
appending a new slab. The honest cost of this choice: a multi-field read (e.g. serializing
an entire `Agent` instance) is **not a linearizable snapshot** — two fields read a
microsecond apart could reflect the state before and after a concurrent write lands between
them. Given this design's own stated philosophy (§8: "no dirty-bit machinery, refresh is
just re-eval"), that's an acceptable, already-assumed tradeoff, not a new one — but it's
worth stating plainly here since it's the direct consequence of *not* stopping the world
for reads.

**Method-invoke and definition evals do need a full stop**, for the reason §4's shape
migration already establishes: running arbitrary Scry code, or swapping a method table,
concurrently with N other mutating threads is not sound at any granularity this design
offers short of a global stop — there is no partial-safepoint level between "this one
arena" and "everything," and an invoked method can touch any arena. These evals reuse
§4/§6's `request-global-stop`/`release-global-stop` and run on **one dedicated,
always-present `VMThread`** — registered once at server startup, not spawned per request —
so there is always somewhere for `OP_CALL_STATIC`'s machinery to push a real `CallFrame`
onto, exactly the way a bytecode-triggered call would, with no agent thread's own stack
borrowed or disturbed.

**The server thread does transport, plus the read-eval fast path above.** For anything
past a read, it parses the incoming `{"id": ..., "source": "..."}` payload — the *only*
shape it ever sees — and pushes a `ViewerRequest` onto a bounded queue for the eval thread:

```coil
(defstruct ViewerRequest
  [(id          i64)                   ; echoed back verbatim in the response
   (source      (ptr u8)) (source-len i64)  ; raw source text — parsed/typechecked/run on
                                             ; the eval thread, never on the server thread
   (conn        (ptr u8))               ; opaque connection handle, for the reply
   (next        i64)])                  ; CAS-linked queue node, Treiber-stack style

(defn viewer-enqueue [(req (ptr ViewerRequest))] (-> i64)
  (loop (let [head (atomic-load (queue-head))]
          (do (store! (field req next) head)
              (if (= (atomic-cas (queue-head) head (cast i64 req)) head) (break 0) (continue))))))
```

**The eval thread drains the queue at its own safepoint**, then calls
`request-global-stop`, and — with every language thread parked — does exactly what an
earlier, single-threaded draft of this doc described (unchanged, just relocated onto its
own `VMThread` instead of borrowing the one shared interpreter's):

- **An expression that calls a method** (`Agent#7.resume()`) resolves the target handle,
  looks up the `MethodInfo` by name in the target's `TypeInfo`, marshals the evaluated
  argument values into typed stack values per `MethodSig`, and pushes a real `CallFrame`
  using the exact same machinery `OP_CALL_STATIC` uses — a viewer-triggered `pause()` is
  indistinguishable, once it's running, from a bytecode-triggered one.
- **A definition** (a new method body, per `03-live-semantics.md`) typechecks against the
  live class and, if it passes, swaps the method-table entry while still stopped; if it
  fails, the response's `error` carries the type error and nothing about the running
  program changes. Live code change is not a separate wire feature — it's `eval` of a
  definition instead of an expression.

Then `release-global-stop` runs, and every parked thread resumes. The cost is that an
invoke or a definition "cuts the line" for **every** language thread at once, not just one:
for a `pause()` that's imperceptible; for a slow method it's a real, visible pause across
the whole demo, the same category of cost as a GC pause (§6), and should be budgeted the
same way.

**The genuinely hard part, revisited for N threads: a blocking foreign call now delays a
stop, not "the whole system."** This is the one place real threads make a risk *better*,
not worse, but not free. An agent thread parked inside an `extern` call — a synchronous
HTTP call to an LLM provider — cannot poll its own safepoint, so `request-global-stop`
blocks waiting for it specifically. Unlike the single-thread design's failure mode ("the
whole system froze"), **every other agent thread keeps running** during that wait — only
operations that need *everyone* stopped (GC, shape migration, a viewer invoke or
definition) are delayed, and only until the blocked call returns. This is real and worth
designing around, not hand-waved:

- **What this doc requires of the language's I/O model** (`01-language.md`'s call, OPEN
  per `DECISIONS.md`'s thread-API-surface item): no Scry thread should be parked in a
  blocking syscall for an unbounded time, because doing so caps how promptly *any* GC,
  migration, or live-edit can land while that call is outstanding — not a correctness bug
  (nothing races), a latency one.
- **Concrete proposal for the PoC** (proposing, not deciding): run genuinely long blocking
  I/O on a *separate* worker OS thread that touches **no** Scry heap at all (it computes
  with plain bytes — an HTTP response body — nothing arena-allocated) and hands the raw
  result back to the owning agent thread through the same lock-free queue shape the viewer
  uses; the agent thread's own `VMThread` stays inside the dispatch loop, polling
  safepoints normally, the whole time. This keeps the "no thread blocks longer than a
  safepoint interval" property that both GC/migration promptness and viewer responsiveness
  actually depend on, while still giving each agent the real-thread simplicity
  `DECISIONS.md` #4b asked for.

### A pacing primitive — the demo needs one

The language surface for this is `01-language.md` §1.7's `Clock.sleep(ms)`
(`__builtin_sleep_millis`); this section designs its runtime implementation.
The demo's most emphasized visual beats depend on it existing.
`00-vision.md`'s 0:30 beat requires `Message`/`ToolCall` counts "visibly climbing" and
status lines "ticking" over the ~5-minute script, not completing instantly. M4's agents are
driven by a synchronous, deterministic `ScriptedModel` with no real network latency, and —
per `DECISIONS.md` #4b/`05-milestones.md`'s M5 — each agent runs its scripted steps on its
own real `VMThread` (§7), not on a shared cooperative scheduler; there is no single
dispatcher left to hand a "not ready yet" task to. Left unpaced, an agent's own thread
simply races through its entire scripted task list in milliseconds — by the time a
presenter opens the browser at 0:30 there is nothing left to watch climb.

A naive blocking `sleep()` is not the fix, for the same reason this section already forbids
an agent thread from parking in a blocking foreign call for longer than a safepoint-poll
interval: a thread parked in a real blocking syscall cannot poll its own `safepoint-poll`,
so it delays every GC/migration/viewer-invoke that needs a full stop for as long as the
sleep runs.

**Proposal: a bounded, safepoint-cooperative `sleep` primitive that each agent's own
`VMThread` calls directly between scripted steps** — no shared scheduler, no second queue;
each agent already is its own real thread (§7), so pacing it is a local concern of that
thread's own dispatch loop, the same way the "run genuinely long blocking I/O on a separate
worker thread" proposal above keeps an agent thread inside its own loop rather than
blocking it. `sleep(millis)` compiles to a tight loop the interpreter runs in place of a
blocking syscall, polling its own safepoint on every pass:

```coil
(defn language-sleep [(vt (ptr VMThread)) (millis i64)] (-> i64)
  (let [deadline (+ (now-millis) millis)]
    (loop (do (safepoint-poll vt)                        ; same poll every instruction
                                                           ; already takes (§2/§7), just also
                                                           ; reachable from inside a sleep
              (if (icmp-ge (now-millis) deadline) (break 0) (do (spin-hint) (continue)))))))
```

Because `safepoint-poll` runs on every pass through this loop, exactly as it does on every
ordinary bytecode instruction, a `sleep` call never blocks a stop-the-world for longer than
one pass through the loop — the same granularity every other instruction already offers,
not a special case carved out for sleeping. `spin-hint` is a cheap CPU pause (or, if
profiling later shows busy-waiting to be a real cost even at the PoC's handful of agent
threads, a short bounded OS sleep between polls — a tuning knob, not an architecture
change; what matters is that the thread comes back up for air and re-checks its safepoint
on some short, bounded period, not the exact wait mechanism inside that period). Demo
pacing is human-visible-seconds scale, not microsecond scale, so even a coarse,
millisecond-granularity poll loop is more than fine here — this is not a hot loop under
real contention.

This mechanism belongs here because it's a small, direct addition to machinery §7 already
owns (`safepoint-poll`, now also called from inside a sleep, not just the dispatch loop);
naming the primitive in `01-language.md`'s stdlib sketch (so demo code can actually call
it) and budgeting it in `05-milestones.md`'s M4 IN list (so each agent's pacing is
designed, not assumed to fall out of an unpaced loop) is those docs' follow-up, flagged
here since the gap surfaces first as a runtime-mechanism question.

## 8. Introspection hooks

The runtime exposes exactly **one** service to the outside world: `eval`. There is no
per-feature op table — every viewer pane in `00-vision.md` is sugar over sending
`{id, source}` and rendering the `{id, value}` (or `{id, error}`) that comes back. What
follows is "here's what running different shapes of source does with machinery already
built above," not a new op menu:

| What the viewer sends as `source` | Resolves via | Notes |
|---|---|---|
| `Agent.instances()` (or any per-type summary the type index needs) | walk `type-table`, read `arena.live-count` per `TypeInfo`, or `TypeInfo` by name → `arena-for-each-live` | O(type count) for a summary, O(live) for a listing; the arena enumeration this whole doc is built around (§4) is exactly what makes this cheap enough to re-run on every eval |
| a filtered listing (`Agent.instances().filter(...)` or equivalent) | same as above, plus evaluating simple field-equality predicates using `FieldInfo.offset`/`kind` — no reflection cost beyond a pointer add, since offsets are static | search is a linear scan over the arena (§5's honest cost); a real query surface is `04-viewer.md`'s problem, this doc only guarantees per-field predicates are O(1) to evaluate |
| `Agent#7` / a field read | parse `Type#slot`, `arena-slot-ptr`, compare `generation` | evaluates to a clean `{"live": false}`-shaped value on a stale handle (§5) — never a crash |
| `Agent#7.resume()` (a method call) | `TypeInfo.methods` by name, `MethodSig` for arg marshalling, push a real `CallFrame` | serialized with the main program via the safepoint queue (§7); the response's `value` is whatever the method returns |
| a definition (a new method body) | typecheck against the live class, swap the method-table entry at the same safepoint (`03-live-semantics.md`, M3) | a rejected definition returns `{id, error}`; the running program is unaffected |

**There is no dirty-bit, tick, or publish machinery anywhere in this design.** Nothing
marks an instance "watched," nothing fires when a field is written, and no background
clock scans for changes. `OP_STORE_FIELD_*` does exactly what §2 already says it does and
nothing more — nothing about the viewer touches the field-write path at all. A viewer pane
goes stale the instant something changes in the running program and stays stale until the
client sends the same `eval` again; refresh is the client re-asking — on an interval timer
in the browser, on focus, after an action — never the runtime noticing a mutation and
telling anyone. This is why the safepoint-queue machinery in §7 is the *entire* mechanism:
there is no second, outbound-push half to build.

**Method invocation from outside the main loop is not a special code path** — it is
`OP_CALL_STATIC`'s own machinery, entered from the queue-drain point instead of from
another bytecode instruction. This is the same simplification that made GC roots and field
marking uniform (§6): reuse the one real mechanism, don't build a parallel "viewer version"
of call, of field access, or of iteration.

## 9. What's still open, here

- **Interface default methods** (`DECISIONS.md`: OPEN, lean no for PoC) — the only part of
  interfaces still open. The representation and dispatch mechanism itself is decided and
  built, not open: an interface-typed value is a single pointer, exactly like a class-typed
  one, with dispatch resolved through a per-instance header `type-id` plus a static itable
  (§5) — no fat pointer, superseding this doc's earlier "would need one if interfaces land"
  hedge. If default methods are added later, §5's itable model accommodates them without a
  redesign (an implementing class's unfilled slot points at the interface's own default body
  instead of a per-class one) — flagged here, not decided, per the ruling.
- **Concurrency model of the runtime** (`DECISIONS.md` #4b) — decided and built, not open.
  §7 runs N real `VMThread`s, one genuine OS `pthread` per Scry-level thread, each with its
  own frame array and value stack (§2), polling a generalized safepoint protocol that any of
  them can use to bring every *other* one to a stop for GC (§6), shape migration (§4), or a
  viewer invoke/definition eval (§7). What's still open is `DECISIONS.md`'s own open item —
  the *language-level* thread API surface (spawn/join shape, which synchronization
  primitives — mutex? channels? atomics? — get exposed to Scry code for the PoC) — not the
  runtime mechanism underneath it, which this section already specifies and this doc treats
  as settled.
- **GC-driven compaction** (§6): deliberately deferred, not designed away — the collector
  itself never moves a live instance on its own initiative in v1. The underlying
  walk-and-rewrite-via-`TypeInfo` mechanism is *not* hypothetical, though: it ships in v1
  regardless, because `03-live-semantics.md`'s shape migration (field add/remove) is a
  narrower trigger for the identical pass. "Compaction deferred" means "nothing beyond
  migration moves objects yet," not "the rewrite machinery doesn't exist yet."
- **Query language for instance search** (§8): this doc guarantees field predicates are
  cheap to evaluate as part of an `eval`; the actual query surface (what expression shape
  the viewer generates for a search box) is `04-viewer.md`'s scope.
- **Multiple concurrently-suspended call stacks**: not a gap this doc still has open — each
  Scry-level thread already gets its own real frame array and value stack, for free, from
  the OS (`VMThread`, §2/§7), so there is no shared stack for several concurrently-live
  agents to contend over in the first place. What remains genuinely open is
  `01-language.md`'s `async`/`await` as *surface syntax within a single Scry-level thread*
  (e.g. multiplexing non-blocking I/O onto fewer OS threads than logical tasks) —
  explicitly post-PoC (`DECISIONS.md` #4b) — which would need a materially different
  fiber/stack-suspension mechanism if it ever lands (§7's "What this means for
  `async`/`await`" subsection), not a hole in what this doc already builds for N real
  threads today.
- **HTTP/WebSocket implementation risk** (§7): sockets, an HTTP/1.1 parser, SHA-1, base64,
  and WS frame masking have zero precedent anywhere in Coil today, unlike the
  thread/atomic primitives §7 grounds the server thread in. Whether M1 ships the full
  WebSocket handshake or a simpler long-poll/SSE transport first is `05-milestones.md`'s
  call; this doc only guarantees the gap is named, not that it's closed.
- **Demo pacing primitive** (§7): a bounded, safepoint-cooperative `sleep` — each agent's
  own `VMThread` polling its own safepoint between scripted steps — is proposed, but it
  isn't named yet in `01-language.md`'s stdlib or `05-milestones.md`'s M4 scope — needed so
  each agent's pacing is designed rather than assumed to fall out of an unpaced loop.
