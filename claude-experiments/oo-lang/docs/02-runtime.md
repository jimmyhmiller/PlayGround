# 02 — Runtime Architecture

This is the design for the bytecode VM, object model, and per-type arena allocator, all
implemented in Coil (`coil guide`, `apps/clox/clox.coil` as precedent). Every structure
below is written as if it were about to be typed into a `.coil` file — this is meant to be
buildable, not aspirational.

The one idea everything else serves: **an oo-lang program's live object graph must be
walkable, from the outside, without stopping the world for longer than a query takes.**
That requirement shapes the allocator, the GC, and the thread model, in that order.

## 1. Position: no NaN-boxing

clox NaN-boxes `Value` into one tagged 64-bit word because Lox is dynamically typed —
every stack slot, every field, every argument can hold *any* type, so every `+`, every
`OP_GET_PROPERTY`, every arithmetic op has to ask "what actually is this?" at runtime
(`is-number`, `is-obj-kind`, table lookups on `ObjInstance.fields`).

oo-lang is statically typed with no inheritance. The compiler knows, at every single
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
- This is a real divergence from the clox port and needs to be said plainly: **oo-lang's
  bytecode is a typed bytecode** (à la JVM), clox's is a **dynamically-tagged** bytecode
  (à la a scripting language). We are reusing clox's *architecture* (dispatch loop, object
  header, mark-sweep GC, call frame shape) but not its *value representation*, because the
  entire reason clox needs NaN-boxing (uniform representation for a dynamically-typed
  stack) doesn't apply here.

Where dynamism does show up — generics, and interfaces if/when they land (**OPEN**, see
§4 and `01-language.md`) — it's handled by monomorphization at compile time (Coil already
does this for its own `[T]` generics: `(defn id [T] [(x T)] …)` is specialized per
instantiation, not boxed), not by falling back to a tagged `Value`. A `List<Agent>` and a
`List<Task>` are different compiled types with different, statically-known element
layouts. If interfaces are added, a value of interface type is the one place a fat pointer
(`{data: ptr, vtable: ptr}`, exactly Rust's trait-object shape) becomes necessary — flagged
here as the future extension point, not built now.

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
(const OP_CALL_VIRTUAL  22)     ; reserved for interfaces (§4) — unused until they land
(const OP_RETURN        23)
(const OP_JUMP 24) (const OP_JUMP_IF_FALSE 25) (const OP_POP 26)
```

Object field access compiles to `OP_LOAD_FIELD_I64`/`_F64`/`_REF`/… with a **compile-time
constant byte offset** as the operand, not a name. This is the other big static-typing
win: clox's `ObjInstance.fields` is a hash `Table` because Lox instances can grow fields
dynamically at runtime; oo-lang classes have a fixed shape decided at compile time, so
`self.status = AgentStatus.Paused` is `OP_STORE_FIELD_I64 <offset=16>` — a pointer add and
a store, no hashing, no probing, comparable to JVM `putfield` with a constant-pool-resolved
offset.

**Call frames** — same shape as clox's, with one addition (the ref-slot bitmap, needed for
precise GC roots, §5):

```coil
(defstruct MethodInfo
  [(name        (ptr u8)) (name-len i64)
   (arity       i64)
   (local-count i64)                 ; total stack slots this method needs
   (ref-slots   (ptr u64))           ; bitmap: bit i set => local slot i holds a ref/ptr
   (code        (ptr u8)) (code-len i64)
   (fn-address  (fnptr c [(ptr u8)] i64))])   ; for CALL_STATIC / viewer invoke, same fn

(defstruct CallFrame [(method (ptr MethodInfo)) (ip (ptr u8)) (base i64)])

(defstruct VM
  [(frames      (array CallFrame 256))
   (frame-count i64)
   (stack       (array i64 65536))    ; shared value stack, untagged, per-slot meaning
                                       ; comes from the static type at that slot — see §1
   (stack-top   i64)
   ; … globals table, arenas, GC state — §5/§6
   ])
```

The dispatch loop is clox's loop verbatim in shape (hoist the mutable `frame` cell outside
the `loop`, per the clox port's own comment: never `alloc-stack` inside a hot loop):

```coil
(defn run [] (-> i64)
  (let [(mut frame) (index (field (vm) frames) (- (load (field (vm) frame-count)) 1))]
    (loop
      (safepoint-poll)                                ; §7 — one branch, almost always false
      (let [instr (read-byte (load frame))]
        (case instr
          OP_LOAD_LOCAL_I64 (do (push-i64 (load (index stack-base (read-byte …)))) 0)
          OP_ADD_I64        (do (push-i64 (+ (pop-i64) (pop-i64))) 0)
          OP_STORE_FIELD_I64 (let [off (read-byte (load frame)) v (pop-i64) obj (pop-ptr)]
                               (do (store! (cast (ptr i64) (+ (cast i64 obj) off)) v) 0))
          OP_CALL_STATIC     (do (push-frame …) 0)
          OP_RETURN          (if (pop-frame) (continue) (break (pop-i64)))
          ; … 
          )))))
```

## 3. Object layout for class instances

Every instance is: a fixed **header**, then fields in declaration order at compile-time
offsets. No per-instance kind tag is needed the way clox's `Obj.kind` needs one — **which
arena you're in already tells you the type** (§5). That's a real header-size win over
clox: one arena per type means "type" is a property of the *pointer's origin*, not
something every object has to carry itself.

```coil
(defstruct InstanceHeader
  [(slot-index i64)     ; this instance's index within its type's arena — needed to hand
                         ; the viewer a stable id (§5), and to return this slot to the
                         ; free list without a reverse lookup
   (generation u32)      ; bumped when this slot is freed — invalidates viewer handles (§5)
   (flags      u32)])    ; bit0 live, bit1 gc-marked, bit2 watched-by-viewer (§8)

; class Agent { name: String, model: String, status: AgentStatus, conversation: Conversation, tools: List<Tool> }
(defstruct Agent_Instance
  [(header       InstanceHeader)
   (name         (ptr StringObj))
   (model        (ptr StringObj))
   (status       i64)                 ; enum -> i64 tag, no boxing
   (conversation (ptr Conversation_Instance))   ; raw pointer — see §5 for why this is safe
   (tools        (ptr ListObj))])
```

`InstanceHeader` is 16 bytes. For an `Agent` with the fields above that's ~56 bytes total —
overhead is real but small, and paid once per instance, not per field access.

**Type metadata.** Every concrete `class` gets one compile-time-generated `TypeInfo`. This
single artifact is what both the GC (which fields are pointers to trace) and the viewer
(what to render, what to invoke) hang off of — worth calling out explicitly, because it
means reflection is not a separate subsystem bolted on for the viewer; it's the same table
the GC already needs to exist.

```coil
(const FIELD_I64 0) (const FIELD_F64 1) (const FIELD_BOOL 2) (const FIELD_STRING 3)
(const FIELD_REF  4) (const FIELD_LIST_REF 5)   ; REF/LIST_REF = fields the GC must trace

(defstruct FieldInfo
  [(name (ptr u8)) (name-len i64) (offset i64) (kind i64) (elem-type-id i64)])  ; elem-type-id: which
                                                                                 ; TypeInfo a REF points at

(defstruct MethodSig [(param-kinds (ptr i64)) (param-count i64) (ret-kind i64)])

(defstruct TypeInfo
  [(type-id       i64)
   (name          (ptr u8)) (name-len i64)
   (instance-size i64)
   (fields        (ptr FieldInfo)) (field-count i64)
   (methods       (ptr MethodInfo)) (method-count i64)
   (sigs          (ptr MethodSig))                 ; parallel to methods, for arg marshalling
   (arena         (ptr TypeArena))])                ; -> §5

(defn type-table [] (-> (ptr (ptr TypeInfo))) (alloc-static (ptr (ptr TypeInfo))))  ; indexed by type-id
```

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
   (slab-count     i64)
   (bump-slab      i64) (bump-offset i64)   ; frontier of ever-allocated slots
   (free-head      i64)              ; index of first free slot, or -1
   (live-count     i64)])
```

A slot's global index is `slab# * capacity + local#`. Resolving `slot-index → pointer`:

```coil
(defn arena-slot-ptr [(a (ptr TypeArena)) (idx i64)] (-> (ptr u8))
  (let [cap    (ishl 1 (load (field a slab-shift)))
        slab#  (ishr idx (load (field a slab-shift)))
        local  (iand idx (- cap 1))]
    (index (load (index (load (field a slabs)) slab#)) (* local (load (field a slot-size))))))
```

### Allocation fast path

```coil
(defn arena-alloc [(a (ptr TypeArena))] (-> (ptr u8))
  (let [(mut idx) (load (field a free-head))]
    (if (!= idx -1)
      ; pop the free list — O(1), no syscall
      (do (store! (field a free-head) (load (cast (ptr i64) (arena-slot-ptr a idx)))) idx)
      ; bump inside the current slab, or grow a new one
      (let [cap (ishl 1 (load (field a slab-shift)))]
        (do
          (if (= (load (field a bump-offset)) cap)
            (grow-arena a)   ; malloc a new 64KiB slab, push its base ptr onto `slabs`
            0)
          (let [new-idx (+ (* (load (field a bump-slab)) cap) (load (field a bump-offset)))]
            (do (store! (field a bump-offset) (+ (load (field a bump-offset)) 1)) new-idx)))))
    (let [p (arena-slot-ptr a idx)]
      (do (store! (field (cast (ptr InstanceHeader) p) slot-index) idx)
          (store! (field (cast (ptr InstanceHeader) p) flags) 1)   ; live
          (store! (field a live-count) (+ (load (field a live-count)) 1))
          p))))
```

Freeing bumps `generation`, clears the live flag, and pushes the slot back onto the free
list (the freed slot's own memory is reused to store the free-list link — no extra
allocation for the free list itself, the same trick a real slab allocator uses):

```coil
(defn arena-free [(a (ptr TypeArena)) (idx i64)] (-> i64)
  (let [p (cast (ptr InstanceHeader) (arena-slot-ptr a idx))]
    (do (store! (field p generation) (+ (load (field p generation)) 1))
        (store! (field p flags) 0)
        (store! (cast (ptr i64) p) (load (field a free-head)))   ; overlay free-next
        (store! (field a free-head) idx)
        (store! (field a live-count) (- (load (field a live-count)) 1)))))
```

Same-size slots per type means **no cross-type fragmentation is possible** — a freed
`Agent` slot can only ever become a new `Agent`. Internal fragmentation (holes left by
frees inside a slab) is exactly what the free list absorbs; it's never returned to the OS.

### Enumeration — "list all live Agents"

```coil
(defn arena-for-each-live [(a (ptr TypeArena)) (f (fnptr c [(ptr u8)] i64))] (-> i64)
  (let [cap (ishl 1 (load (field a slab-shift)))]
    (for [slab# 0 (load (field a slab-count))]
      (for [local 0 (if (= slab# (load (field a bump-slab))) (load (field a bump-offset)) cap)]
        (let [p (index (load (index (load (field a slabs)) slab#)) (* local (load (field a slot-size))))]
          (if (= (iand (load (field (cast (ptr InstanceHeader) p) flags)) 1) 1)
            (call-ptr f p) 0))))))
```

**Cost, stated honestly:** this is `O(high-water-mark)`, not `O(live)`. A type that churns
heavily — created and destroyed thousands of times while only a handful are alive at any
moment — still costs one header-flag check per ever-allocated slot on every enumeration,
because slabs are never compacted or returned. For the demo's entity types (`Agent`,
`Conversation`, `Task`) this is a non-issue: they're created once and live for the process.
For a hypothetical high-churn type (e.g. a `ToolCall` per tool invocation in a long-running
agent), this is the first thing that would show up in a profile, and the fix (§6,
compaction) is deferred, not solved.

### Identity, handles, and why they matter

**Internal object-graph references are raw pointers**, exactly like clox's `(ptr Obj)`.
`Agent_Instance.conversation` points directly at a `Conversation_Instance`. This is safe
*by construction*: a mark-sweep collector never frees an object that's still reachable
through a field, so a live object can never hold a dangling pointer to a dead one. Internal
field access is one dereference, same cost as clox, no regression.

This is a deliberate, load-bearing choice, made jointly with `03-live-semantics.md`:
oo-lang does **not** put a handle table between class-typed fields/locals and the objects
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
→ {"op": "instance", "target": "Agent#7"}
← {"op": "instance", "target": "Agent#7", "live": false}
```

Note the honest refinement to `00-vision.md`'s wire examples: the human-visible id stays
`"Agent#7"` (type name + slot index), but the request/response payload that actually
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
   (active-frames   i64)                   ; §3-live-semantics' quiescence counter
   (arena           (ptr TypeArena))])
```

When `old-shape` is null, dispatch and field offsets are exactly what §2/§3 already
describe — one flat `shape.methods[slot]`, one flat set of `FieldInfo` offsets, zero extra
cost. `old-shape`/`pending-shape` are populated only for the duration of one migration.

**`InstanceHeader` gains a shape id.** Every instance has to say which shape it's on,
because during a drain window two live instances of the same `type-id` can disagree:

```coil
(defstruct InstanceHeader
  [(slot-index i64)
   (generation u32)
   (shape-id   u32)      ; NEW — which ShapeInfo this instance's fields are laid out per
   (flags      u32)])
```

**`TypeArena` holds two live slabs during a migration window.** Enumeration (`arena-for-
each-live`, §4 above) and allocation must both know there can be an old, frozen, draining
slab and a new, growing one at once:

```coil
(defstruct TypeArena
  [(type-id i64)
   (slot-size i64) (slab-shift i64) (slabs (ptr (ptr u8))) (slab-count i64)
   (bump-slab i64) (bump-offset i64) (free-head i64) (live-count i64)
   ; migration-only fields — all zeroed/null when no migration is pending:
   (old-slot-size  i64)             ; 0 when not migrating
   (old-slabs      (ptr (ptr u8)))  ; the frozen, draining slab set — no new allocs here
   (old-slab-count i64)
   (old-live-count i64)])           ; counts down to 0 as instances are copied over
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
ENTER_METHOD <type-id>   -> TypeInfo[type-id].active-frames += 1   (single-thread: plain add)
RETURN                   -> TypeInfo[type-id].active-frames -= 1
                            if active-frames == 0 and pending-shape != null:
                              run-migration(type-id)
```

`run-migration(type-id)` runs at a **full safepoint** — it fires on the `RETURN` that
drops `active-frames` to zero, so the interpreter thread itself is the one executing it,
no bytecode is mid-read, and the server thread's queries are held off exactly as they are
for any other safepoint-serviced work (§7). It does, in order:

1. **Same-index copy.** For each live instance at slot index `i` in the old slab set,
   compute its new field values (default or migration-fn, per 03) and copy into slot index
   `i` of the current-side slab set — the slot reserved for it above, **never a fresh
   `arena-alloc`** — stamping the new `shape-id` into its header.
   `InstanceHeader.slot-index` is unchanged and `generation` is **not** bumped (only
   `arena-free` bumps it, §4), so the wire-facing identity `{type-id, slot-index,
   generation}` survives the move untouched: `Agent#7` is still literally `Agent#7`, with
   no lookup table anywhere.
2. **Pointer rewrite** — the pass §6 defers for compaction, done here for one type. Walk
   every arena's live instances via `TypeInfo`/`FieldInfo` exactly like `blacken-object`
   (§6) does for GC marking, plus the VM stack's ref slots (`MethodInfo.ref-slots`, the
   same bitmap §6's `mark-roots` reads) and the globals table; for every class-typed slot
   holding an address inside the migrating type's old slab set, rewrite it to the
   same-index address in the new set. Old and new addresses are both pure functions of
   `slot-index` (§4's `arena-slot-ptr`), so the rewrite is address arithmetic — recover
   `i` from the old address, emit the new-set address for `i` — no lookup table, no
   remembered set.

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
goes null again. This is the same "pay only while it's actually in flight" shape as §8's
`watched`-bit check: every other type, and this type once its drain finishes, never pays it.

**Cost, stated honestly, matching this doc's own convention (§4's enumeration-cost note):**
a migration is `O(live instances of that type)` for the copy, plus `O(total live objects
across every arena, plus stack slots and globals)` for the pointer-rewrite pass — the
whole live heap, honestly — run once per accepted field-add/remove
edit — not on any hot path, and bounded by how often a developer saves that kind of edit,
not by request rate. This is the real, buildable project §6 already flagged as deferred for
general compaction; shape migration is the first concrete customer for it.

## 5. Interfaces and generics — how they touch the arena model (OPEN)

**OPEN** (`DECISIONS.md`): interfaces/traits are undecided. If added, an interface-typed
field or local needs a fat value (`{ptr, vtable}`), and calling a method through it needs
`OP_CALL_VIRTUAL` (already reserved, §2, unused). None of this touches per-type arenas —
an interface value still points at a real instance living in its concrete type's arena; the
fat pointer is purely a dispatch mechanism, not a storage mechanism. Generics need no
runtime representation at all if monomorphized (Coil's own `[T]` model, reused directly):
`Inventory<Tool>` and `Inventory<Agent>` are two distinct compiled types with two distinct
arenas, same as any other class. This keeps the "one arena per concrete type" property
intact regardless of how §5's OPEN question resolves.

## 6. Garbage collection

Mark-sweep, per DECISIONS.md, fanned out over N per-type arenas instead of clox's one
global object list.

**Roots are precise, not conservative.** clox has to treat every stack slot as a
potential `Value` and check `is-obj` on it, because Lox has no static types. oo-lang knows,
per method (`MethodInfo.ref-slots`, §2), exactly which stack slots hold a pointer. Marking
roots walks live call frames and only looks at the bits set in `ref-slots`:

```coil
(defn mark-roots [] (-> i64)
  (for [i 0 (load (field (vm) frame-count))]
    (let [fr (index (field (vm) frames) i) m (load (field fr method))]
      (for [slot 0 (load (field m local-count))]
        (if (bitmap-test (load (field m ref-slots)) slot)
          (mark-object (load (cast (ptr (ptr u8)) (index stack (+ (load (field fr base)) slot)))))
          0))))
  (mark-globals-table)
  0)
```

**Marking follows `TypeInfo`, not a hand-written visitor per class** — another payoff of
building the reflection table anyway: `blacken-object(p)` looks up `p`'s arena's `type-id`
in `type-table`, iterates `FieldInfo`, and for every `FIELD_REF`/`FIELD_LIST_REF` field,
loads the pointer at that offset and marks it. One generic tracer for every class, instead
of clox's per-`ObjKind` `switch` in `blacken-object`.

**Sweep is enumerate-and-conditionally-free**, per arena, reusing §5's
`arena-for-each-live` shape: walk each type's slabs in slot order; a slot whose `mark` bit
is set gets the bit cleared (ready for next cycle) and survives; a slot that was live but
unmarked is freed via `arena-free` (§5) — its `generation` bumps, invalidating any viewer
handle that pointed at it. Global mark phase runs once across all arenas (it has to — the
object graph crosses arena boundaries), then sweep runs arena-by-arena, independently.

**Trigger:** one global `bytes-allocated` / `next-gc` pair, same heuristic as clox
(`GC_HEAP_GROW_FACTOR`), summed across all arenas — simplicity over per-type precision for
v1; a type that's unusually hot could get its own threshold later if profiling calls for
it, but that's a tuning knob, not an architecture change.

**Compaction: not in v1, and here's why it's not a soundness gap.** Same-size slots per
type mean external fragmentation across types is structurally impossible (§5), and
internal fragmentation is absorbed by the free list. Compaction would only buy back
enumeration cost on high-churn types (the honest cost flagged in §5) and slab memory
returned to the OS. It's not free to add later: moving a live instance means rewriting
every pointer to it, which requires exactly the same "walk the graph with `TypeInfo`" pass
mark already does, plus a second pass to fix up. That's a real, buildable project — the
metadata to do it precisely already exists — it's just not worth building *for GC's own
sake* before there's a profiled workload that needs it. Ship mark-sweep-with-free-lists
first.

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

## 7. The interpreter thread and the embedded HTTP/WebSocket server

**Position: one interpreter thread, one separate server thread, communicating through a
lock-free request/response queue drained at safepoints.** Not N interpreter threads, not a
single thread doing both jobs cooperatively.

### One call stack, not N — what this means for `01-language.md`'s `async`/`await`

Before grounding the server thread, a gap in what this doc has already committed to needs
naming. §2's `VM` struct is unambiguous: one `frames (array CallFrame 256)` and one shared
`stack (array i64 65536)` — a single execution context, matching clox's single-call-stack
interpreter, extended (§1) only in value representation, never in how many stacks exist.

`01-language.md` §5 proposes `async fn`/`await`/`spawn` as the language's concurrency
primitive, and its own demo program suspends `runTurn` at `await Llm.complete(...)` —
three agents, each potentially parked mid-turn, several frames deep, while a scheduler
runs other tasks in between. One 256-frame array and one 65536-slot stack cannot hold
three independently-suspended, independently-resumable call stacks at once: there is
nowhere to keep agent A's frames 12–15 parked while agent B's frames 3–9 run, because both
would have to live in the same array at the same time. This is *not* the same gap as the
already-flagged OPEN "concurrency model of the language" item (§9) — that item is about
which primitive the language exposes to a programmer. This is that the VM structures this
doc already commits to cannot represent the state any such primitive needs, once "suspend
mid-call, resume later, while other tasks also run concurrently" is the semantics.

Two honest ways to close this, and this doc picks the second for the PoC:

1. **Real per-task stacks (fibers/green threads).** Give every `spawn`ed task its own
   frame array and stack slab — `Task { frames: (array CallFrame 256), frame-count: i64,
   stack: (array i64 65536), stack-top: i64, status: i64 }` — and have `VM` hold a
   `current-task (ptr Task)` the scheduler swaps on every cooperative yield, the same way
   an OS context-switches a green thread. Coil's plain structs represent this fine (no
   `pthread` needed for this part), but it is a materially different VM than §2 describes:
   every read of `(vm frames)`/`(vm stack)` has to go through `(current-task)` instead, and
   256 frames × 65536 slots *per concurrently-live task* is a memory budget this doc has
   not sized.
2. **State plainly that M0–M4 never suspend mid-call.** The turn scheduler this section
   and `05-milestones.md` M2 actually build is a cooperative *turn* queue: each queued turn
   runs to completion, synchronously, on the one shared stack, before the next turn is
   dequeued. `Llm.complete` inside `ScriptedModel`'s turns never blocks — its result comes
   back through the completion queue below and is picked up at the *next* safepoint, not by
   resuming a parked stack mid-call — so nothing ever needs frames 12–15 of one task parked
   while frames 3–9 of another run. `await`, for the PoC, is sugar for "yield the rest of
   this turn back to the scheduler at a turn boundary," never a true stackful suspension
   mid-call.

**Position for the PoC: option 2.** Option 1 is a real, buildable project — Coil's structs
represent it fine — but it is not needed for M0–M4 as scoped, and building it speculatively
now would repeat the mistake §6 explicitly avoids with compaction. `01-language.md` §5's
`await` should be read, for the PoC, as: *the turn function runs synchronously end-to-end;
`await` marks a turn boundary the scheduler can round-robin between, not a point this VM
can resume execution from mid-stack.* If a later milestone needs mid-call suspension (an
agent turn blocking deep inside nested helper calls, not just at the top of `runTurn`),
option 1 is the concrete shape that closes it, and it should be scoped as its own
milestone, not assumed to fall out of the safepoint/queue machinery below.

### Grounding the server thread — and where that grounding runs out

Grounded in what Coil actually supports (`lib/thread.coil`: `pthread_create`/`pthread_join`
wrappers; `lib/atomic.coil`: `atomic-cas`/`atomic-load`/`atomic-store`;
`examples/lockfree.coil`: a proven CAS-loop Treiber stack) — the primitives for "one
worker OS thread talking to a main thread through a lock-free structure" already exist and
are already exercised in this codebase.

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

**Why not multiple interpreter threads sharing the arenas:** the entire allocator fast path
(§5) is lock-free specifically *because* only one thread ever touches it — `arena-alloc`
and `arena-free` do plain (non-atomic) reads and writes. Making arenas thread-safe means a
CAS or a lock on every allocation, which is the exact cost the per-type-arena design exists
to avoid. So: single-writer discipline on the whole object graph, enforced by construction
— everything that isn't the interpreter thread talks to it through the queue, never
touches an arena directly.

**The server thread does transport only.** It accepts connections, parses an incoming
JSON op (`{"op":"instances","type":"Agent",...}`), and pushes a `ViewerRequest` onto a
bounded queue; it never dereferences an oo-lang pointer itself, because there is no safe
moment for it to do so — the interpreter could be mutating or the GC could be sweeping the
exact arena it would read.

```coil
(defstruct ViewerRequest
  [(kind (ptr u8)) (kind-len i64)      ; raw JSON body — parsed on the interpreter side
   (conn (ptr u8))                     ; opaque connection handle, for the reply
   (next i64)])                        ; CAS-linked queue node, Treiber-stack style

(defn viewer-enqueue [(req (ptr ViewerRequest))] (-> i64)
  (loop (let [head (atomic-load (queue-head))]
          (do (store! (field req next) head)
              (if (= (atomic-cas (queue-head) head (cast i64 req)) head) (break 0) (continue))))))
```

**The interpreter drains the queue at a safepoint** — a check at the top of the dispatch
loop (§2's `safepoint-poll`), which is one atomic load and a predicted-not-taken branch per
bytecode instruction. Viewer traffic is human-paced (clicks, not microseconds), so in
steady state this check is false almost every time — the cost is in the noise next to the
several branches clox's own dispatch already does per instruction.

When a request is pending, the interpreter services it **synchronously, inline in the
dispatch loop, before executing its next instruction**:

- **Read-only ops** (`types`, `instances`, `instance`) walk the relevant arena directly
  (§5's `arena-for-each-live`, or a single `arena-slot-ptr` + generation check) and format
  a JSON response — safe, because the interpreter is the only thing that ever touches
  arenas, so there's no concurrent mutation to race against its own read.
- **`invoke`** resolves the target handle, looks up the `MethodInfo` by name in the target's
  `TypeInfo`, marshals JSON args into typed stack values per `MethodSig`, and pushes a real
  `CallFrame` using the exact same machinery `OP_CALL_STATIC` uses — a viewer-triggered
  `pause()` is indistinguishable, once it's running, from a bytecode-triggered one. It runs
  to completion before the interpreter returns to whatever bytecode it was executing before
  the safepoint fired.

This means a viewer invoke and the program's own next instruction are **strictly
serialized** — never concurrent — so nothing inside the interpreter needs new
synchronization beyond the safepoint check itself. The cost is that an invoke "cuts the
line": the program's own next instruction waits for the viewer's call to finish. For a
`pause()` that's imperceptible; for a slow method it's a real, visible pause — the same
category of cost as a GC pause, and should be treated as one when reasoning about latency.

**The genuinely hard part: a blocking foreign call closes the window.** The interpreter can
only poll the safepoint while it's executing bytecode. If it's parked inside an `extern`
call — the demo's agent turn making a synchronous HTTP call to an LLM provider — it is not
in the dispatch loop and cannot service the queue. From the viewer's side, this reads as
"the whole system froze," for the duration of that one call, even though every other
`Agent` is idle and uninvolved. This is real and worth designing around rather than
hand-waving:

- **What this doc requires of the language's I/O model (01-language.md, OPEN — this is
  the concurrency-model question DECISIONS.md flags):** the interpreter thread must never
  be parked in a blocking syscall for longer than a safepoint-poll interval. Concretely,
  that means an agent's LLM call cannot be `curl`-style synchronous FFI on the interpreter
  thread.
- **Concrete proposal for the PoC** (proposing, not deciding — this is 01-language.md's
  call): run the actual blocking I/O on a *separate* worker OS thread that touches **no**
  oo-lang heap at all (it computes with plain bytes — an HTTP response body — nothing
  arena-allocated), and hand the raw result back to the interpreter through the *same*
  lock-free queue the viewer uses. The interpreter, at its next safepoint, allocates the
  real `Message` instance from the raw bytes and resumes the agent's logical "turn." One
  completion-queue mechanism serves both viewer requests and async I/O completions — no
  second concurrency primitive to build. Whatever scheduling abstraction 01-language.md
  puts on top (explicit `async`/`await`, cooperative "turns" a scheduler round-robins,
  green-thread-style resumable calls) has to bottom out here, because this is the only
  place oo-lang objects are ever allowed to be touched.

### A pacing primitive — the demo needs one and nothing above provides it

Nothing in this doc, `01-language.md`, or `05-milestones.md` defines a sleep, timer, or
delay primitive, and the demo's most emphasized visual beats depend on one existing.
`00-vision.md`'s 0:30 beat requires `Message`/`ToolCall` counts "visibly climbing" and
status lines "ticking" over the ~5-minute script, not completing instantly. M4's agents
are driven by a synchronous, deterministic `ScriptedModel` with no real network latency,
looped by a plain cooperative turn scheduler with no described delay anywhere in this
design. On a bytecode interpreter, a turn loop with no pacing mechanism races through the
entire scripted task list in milliseconds — by the time a presenter opens the browser at
0:30 there is nothing left to watch climb.

A naive `sleep()` is not the fix: this section already forbids the interpreter thread ever
parking in a blocking foreign call for longer than a safepoint-poll interval, and a literal
sleep on the interpreter thread is exactly that — it would also freeze the viewer's own
safepoint-serviced queue for the sleep's duration.

**Proposal: a scheduler-yield/delay primitive that reuses the completion-queue mechanism
above, rather than a second concurrency primitive.** A pending delay is just another
"not ready yet" entry the turn scheduler already has to check:

```coil
(defstruct TimerRequest [(fire-at-millis i64) (task (ptr u8)) (next i64)])
```

At each safepoint (or a coarser, dedicated low-frequency check — demo pacing is
human-visible-seconds scale, not microsecond scale, so this need not run every
instruction), the interpreter compares `now() >= fire-at-millis` for any pending
`TimerRequest` and, once true, pushes the same shape of completion record the async-I/O
path above already produces onto the shared queue, marking that task ready. The turn
scheduler treats a still-pending timer exactly like a still-pending I/O completion: the
task isn't ready, so the scheduler runs a different ready task instead of blocking. No
worker thread is even required for a coarse, seconds-scale pace — a monotonic-clock
comparison at the safepoint is enough.

This mechanism belongs here because it is exactly the completion-queue machinery this
section already owns; naming the primitive in `01-language.md`'s stdlib sketch (so demo
code can actually call it) and budgeting it in `05-milestones.md`'s M4 IN list (so the
demo app's turn cadence is designed, not assumed) is those docs' follow-up, flagged here
since the gap surfaces first as a runtime-mechanism question.

## 8. Introspection hooks

Every op in `00-vision.md`'s wire protocol maps onto machinery already built above — this
section is mostly "here's the lookup," not new design:

| Wire op | Resolves via | Notes |
|---|---|---|
| `types` | walk `type-table`, read `arena.live-count` per `TypeInfo` | O(type count); pushes a delta whenever any `arena-alloc`/`arena-free` fires for a type the client is watching |
| `instances` (+ `query`) | `TypeInfo` by name → `arena-for-each-live`; evaluate simple field-equality predicates using `FieldInfo.offset`/`kind` — no reflection cost beyond a pointer add, since offsets are static | search is a linear scan over the arena (§5's honest cost); a real query language is `04-viewer.md`'s problem, this doc only guarantees per-field predicates are O(1) to evaluate |
| `instance` | parse `Type#slot`, `arena-slot-ptr`, compare `generation` | returns `{"live": false}` cleanly on a stale handle (§5) — never a crash |
| `invoke` | `TypeInfo.methods` by name, `MethodSig` for arg marshalling, push a real `CallFrame` | serialized with the main program via the safepoint queue (§7); pushes a `changed` event afterward if any mutated field is on a `watched` instance |

**`watched` and the `changed` push.** `InstanceHeader.flags` bit2 is set when the viewer
opens an instance's detail pane (an explicit `watch` op, not shown above — implied by
`00-vision.md`'s live-update requirement) and cleared when the pane closes. `OP_STORE_FIELD_*`
checks a single global "any viewer attached" flag first (near-zero cost when no viewer is
connected — the common case for an ordinary run of the program), and only if that's set,
checks the target instance's `watched` bit before enqueueing a `changed` event onto the
outbound side of the same queue. This keeps the cost of being watchable at "one branch per
field write," paid only while a viewer is actually attached, and "one more branch per
field write on a watched instance," paid only for the handful of instances someone's
actually looking at — not a cost that scales with total live object count.

**Method invocation from outside the main loop is not a special code path** — it is
`OP_CALL_STATIC`'s own machinery, entered from the queue-drain point instead of from
another bytecode instruction. This is the same simplification that made GC roots and field
marking uniform (§6): reuse the one real mechanism, don't build a parallel "viewer version"
of call, of field access, or of iteration.

## 9. What's still open, here

- **Interfaces/traits** (`DECISIONS.md`): affects §4's opcode reservation and would add a
  fat-pointer value kind; doesn't touch the arena/handle model.
- **Concurrency model of the language** (`DECISIONS.md`): §7 pins the *runtime's* answer
  (single interpreter thread, safepoints, a shared completion queue) and states a hard
  requirement (never block the interpreter thread in a foreign call) that whatever
  01-language.md proposes for agent "turns" must satisfy.
- **GC-driven compaction** (§6): deliberately deferred, not designed away — the collector
  itself never moves a live instance on its own initiative in v1. The underlying
  walk-and-rewrite-via-`TypeInfo` mechanism is *not* hypothetical, though: it ships in v1
  regardless, because `03-live-semantics.md`'s shape migration (field add/remove) is a
  narrower trigger for the identical pass. "Compaction deferred" means "nothing beyond
  migration moves objects yet," not "the rewrite machinery doesn't exist yet."
- **Query language for `instances` search** (§8): this doc guarantees field predicates are
  cheap to evaluate; the actual query surface is `04-viewer.md`'s scope.
- **Multiple concurrently-suspended call stacks** (§7): this doc's VM has exactly one
  frame array and one stack; `01-language.md`'s `async`/`await` is treated, for the PoC, as
  never suspending mid-call (option 2 in §7's "One call stack, not N"). A real per-task
  fiber/stack design (option 1, same section) is deferred, not built, and would be a
  materially different VM if a later milestone needs true mid-call suspension.
- **HTTP/WebSocket implementation risk** (§7): sockets, an HTTP/1.1 parser, SHA-1, base64,
  and WS frame masking have zero precedent anywhere in Coil today, unlike the
  thread/atomic primitives §7 grounds the server thread in. Whether M1 ships the full
  WebSocket handshake or a simpler long-poll/SSE transport first is `05-milestones.md`'s
  call; this doc only guarantees the gap is named, not that it's closed.
- **Demo pacing primitive** (§7): a non-blocking scheduler-yield/delay is proposed, reusing
  the completion queue, but it isn't named yet in `01-language.md`'s stdlib or
  `05-milestones.md`'s M4 scope — needed so the demo's turn cadence is designed rather than
  assumed to fall out of an unpaced turn loop.
