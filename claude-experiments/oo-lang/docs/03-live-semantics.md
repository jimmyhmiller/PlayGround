# 03 — Live Semantics

> Reads against `00-vision.md` and `DECISIONS.md`. Grounds every claim in a concrete VM
> mechanism — this doc is written as if `02-runtime.md` already existed; where it invents
> runtime structure to make a semantics claim implementable, that structure is a
> requirement `02-runtime.md` must adopt, not a suggestion.

## The core mechanism: generations, not diffs

Every accepted edit produces a new **generation** of the whole program: a fresh class
table, a fresh shape table, a fresh compiled bytecode image for every method in the
program — not just the ones you touched. The runtime does not try to detect "you edited
`summarize`, leave everything else alone." It recompiles everything from source and
atomically swaps one pointer: `current_generation`.

This is the single idea the rest of this doc is a consequence of. It falls directly out
of the task the brief and `DECISIONS.md` both hand us: *"the whole program re-typechecks
against the new code."* If typechecking is whole-program, compilation might as well be
too — trying to diff-and-patch bytecode is extra machinery in service of an optimization
(faster edits) that the PoC's program sizes don't need yet. Recompiling a few hundred
lines is sub-millisecond; that's a compiler-speed concern for later, not a semantics
concern now.

Consequence: "stale code" is not "the file you didn't save yet." It's *any frame whose
instruction pointer was captured from a generation older than the current one*. A frame
that's inside `Agent.summarize` when generation 6 swaps in is running generation 5's
bytecode for the rest of that call, full stop — including if generation 5's `summarize`
happens to be byte-identical to generation 6's.

## VM structures this depends on

```
TypeInfo {
  type_id: u32
  name: Sym
  shape: ShapeId                     // current field layout
  methods: [FnPtr; N]                // indexed by compile-time method slot
                                      // (keyed by shape_id too during a drain — see below)
  active_frames: u32                 // atomic per §7's concurrency requirement, OPEN below
  pending_shape: Option<ShapeId>
}

ShapeInfo {
  shape_id: u32
  fields: [(name: Sym, ty: Type, offset: u32)]
  stride: u32                        // size of one instance, this shape
}
```

This reuses `02-runtime.md`'s object model as-is, not a parallel one: `TypeInfo` and
`ShapeInfo` here *are* `02-runtime.md §4`'s structs of the same names (a shape change is a
new `ShapeInfo` for the same `TypeInfo`, whose `old-shape`/`pending-shape` slots are
populated only while a migration drains), and per-instance identity is `02-runtime.md
§4`'s existing `InstanceHeader { slot-index, generation, shape-id, flags }` plus the arena
it lives in — there is no separate `ObjectHeader`/`HandleTable` layer on top of that.

One structural decision worth calling out because it isn't free:

- **Class-typed fields and locals are raw pointers, exactly as `02-runtime.md §4` pins
  them** — `self.conversation` is a direct pointer chase, not a handle-table lookup, on
  every access. This doc initially proposed a universal handle indirection instead (to
  make "an instance can move" cheap to reason about); that's wrong for this codebase's
  priorities — it would tax the hottest path in the VM (every field access, §2/§3 of
  `02-runtime.md`) to buy something only shape migration needs. What migration actually
  needs — "move a live instance without leaving a dangling pointer anywhere in the heap
  or on the stack" — is provided instead by a **one-time graph-rewrite pass**, not a
  standing indirection: `02-runtime.md §6` already has to build a full-graph walk driven
  by `TypeInfo` for GC marking (`blacken-object`); migration reuses that exact traversal
  to find every `FIELD_REF`/`FIELD_LIST_REF` pointing into the class's old slab and
  repoint it at the new one, once, at the quiescence point below. See
  `02-runtime.md §6`'s "shape migration is not GC compaction" note for how the two
  relate. The viewer's `Agent#7` identity doesn't need a handle table either — it's
  already `02-runtime.md §4`'s `{type-id, slot-index, generation}` triple, and migration
  preserves it directly (below) by keeping an instance's `slot-index` and `generation`
  unchanged across the move.
- **Method calls are always an indirect call through `methods[slot]`, never inlined.**
  Because there's **no inheritance** (`DECISIONS.md` #4), this indirection is *not*
  buying polymorphism — a call site's target class is always statically known, there is
  no subtype to resolve. The only reason it's indirect is so a redefinition is a single
  pointer swap that reaches every future caller with zero call-site patching. This is a
  real payoff of the no-inheritance decision: we get vtable-style live patchability
  with none of the override/diamond-problem baggage a hierarchy would add. (If
  interfaces/traits land later — **OPEN**, `01-language.md` — an interface-typed call
  site becomes genuinely polymorphic and needs its own dispatch layer on top of this
  one. Orthogonal, not yet decided.)

## What can change live

| Change | Needs shape migration? | Takes effect | Rejected when |
|---|---|---|---|
| Method body | no | next call (new frames only) | new body fails typecheck |
| Method signature | no | next call, at every call site | any call site (old or new) fails typecheck under the new signature |
| Field add | yes | after class drains | no default and no migration fn, and live instances exist |
| Field remove | yes | after class drains | never (dropping data is always representable) — but see migration-fn note below |
| Class add | no | immediately | name collision |
| Class remove | no | immediately for *new* instantiation | never — see "Class removal" |

### Method body redefinition

You save a new body for `Agent.summarize`. Generation N+1 compiles a new `FnPtr`,
typechecks it against the class's *current* shape (unchanged), and on success does:

```
class_table[Agent].methods[SUMMARIZE_SLOT] = gen_N+1.summarize_code
```

one write, atomically visible to every thread. What happens to frames already inside the
old `summarize`:

- **The frame that's already running finishes on the old bytecode.** It already has a
  code pointer and a program counter into it; nothing forces it to re-fetch.
- **Any *new* call it makes — including a recursive call to itself — resolves through
  `methods[slot]` again, and gets generation N+1.** Dispatch is dynamic at every `CALL`
  instruction, not pinned at frame-entry. So a stale frame's straight-line code is
  old, but anything it calls (including itself, recursively) is current. This is the
  right rule because it's the *only* one that doesn't require tracking "which generation
  is this call tree allowed to see" — every call site just asks "what's live right now,"
  which is also what a fresh call from outside would ask.

Developer-facing:

```
method Agent.summarize replaced — next call uses new code
  2 frames currently executing old Agent.summarize(); they will run to completion
```

**Why not stop-the-world until they drain, or force them to restart on new code?**
Restarting means re-running a method whose first half may have already had side effects
(an HTTP call, a file write) — you cannot safely rewind an effect. Blocking the edit
until frames drain means editing an agent that's mid-tool-call (which can be seconds to
minutes) makes your save hang; that's fatal to the "instant edit" demo. Finishing old
frames on old code, non-blocking, with the drain state observable in the viewer, is the
only option that's both safe and fast. The failure mode you accept: for up to one call's
duration, two agents can be executing semantically different code for the same method —
but it's bounded, self-resolving, and visible, not silent.

### Method signature changes

Same mechanism as body redefinition — a signature change is still just a new `FnPtr` at
the same slot — plus one more check: **every call site in the whole program**, not just
the definition, is re-typechecked against the new signature, because whole-program
typecheck was already going to catch this. If you widen `summarize(maxLen: Int)` to
`summarize(maxLen: Int, tone: Tone)` without updating a caller, the *entire* edit is
rejected, not just the signature change — there is no partial application of an edit.

```
✗ change rejected
  Agent.summarize(maxLen: Int) -> String
    changed to Agent.summarize(maxLen: Int, tone: Tone) -> String
  1 call site does not typecheck against the new signature:
    agents.oo:142  self.summarize(20)   — missing argument `tone`
  program unaffected — still running the previous generation
```

No shape/layout migration is needed for a signature change — it never touches field
storage, only the calling convention — so it's exactly as cheap as a body swap and has
the same stale-frame behavior. A stale frame mid-old-`summarize(maxLen)` runs to
completion with one argument; nothing else in the system needs to know it used to have a
different arity.

**OPEN:** first-class function values (a `Fn(Int) -> String` stored in a field, e.g.
`Tool.execute`) are not decided in `01-language.md` yet. If they land, a stored function
value closes over a signature at the moment it was assigned; redefining the underlying
method changes what future *dispatch-by-name* calls see but does not retroactively
change an already-captured function value's type. That's a reasonable default but is
genuinely unresolved until closures are designed.

### Field add / remove — shape versioning and the quiescence gate

This is the case where live STATE and live EXECUTION actually collide, and it's worth
spelling out exactly where.

A class's arena is a slab of fixed-stride chunks (`DECISIONS.md` #6) — that uniform
stride is what makes "walk every live `Agent`" an O(instances) operation instead of a
heap crawl. Adding or removing a field changes the stride. You cannot flip the stride
under instances that are still being read by code compiled against the old one:
`GETFIELD` offsets are baked into bytecode at compile time (that's what makes field
access free — the metadata "every field's name, type, offset" the viewer promises is
free *because* it's static, not looked up every access). A stale frame executing old
`Agent` bytecode is reading `self.notes` at offset 40; if the arena has already been
repacked to the new stride, offset 40 is now garbage or another field entirely. That is
memory corruption, not a type error, and it is the one failure mode this design cannot
tolerate at all.

So: **a class's shape only changes at a quiescence point for that class** — the moment
no frame anywhere in the program is executing a method compiled against the class's
current shape. Mechanically, this rides on bytecode the interpreter already has to emit:

```
ENTER_METHOD Agent   → class_table[Agent].active_frames += 1   (atomic)
RETURN                → class_table[Agent].active_frames -= 1   (atomic)
                         if active_frames == 0 and pending_shape:
                           run_migration(Agent)
```

When a field-changing edit is accepted:

- if `active_frames == 0` right now, migrate synchronously, in the same instant the
  generation swaps in.
- if `active_frames > 0`, mark `pending_shape = shape N+1` and let it fire on the
  return that brings the counter to zero — no polling, no global stop-the-world, and the
  gate is scoped to *this one class*, not the process. Every other class's methods keep
  running uninterrupted the whole time.

Migration itself: allocate a new slab at the new stride, and for each live instance
either (a) copy unaffected fields across and fill an added field from its declared
default, or (b) run a user-supplied migration function that receives the *entire old
instance* (all its old fields, including any about to be removed) and returns the new
field's value. Removed fields are simply not copied. This also gives field **rename**
for free without a separate mechanism — a rename is a remove + an add in the same edit,
and the migration function bridges them using the old value before it's gone:

```
class Agent {
  name: String
  model: String
  status: AgentStatus
  notes: String              // new
  conversation: Conversation
  tools: List<Tool>
}

migrate Agent v1 -> v2 {
  notes: ""                                        // default form
}

migrate Agent v2 -> v3 {
  summaryTone: (old) => old.status == AgentStatus.Paused
    ? Tone.Terse
    : Tone.Normal            // derived from an old field, arbitrary expression
}
```

Migration allocates the new slab with the *same slot-index numbering* as the old one —
live instance at old index `i` is written to new index `i` — so an instance's
`InstanceHeader.slot-index` never changes and `generation` is **not** bumped by a
migration (only `arena-free`, §5, bumps it). That's what keeps `Agent#7` literally
`Agent#7` across the move without a handle table: the viewer's identity triple
(`type-id`, `slot-index`, `generation`) is unaffected. What *does* need fixing up is
every other raw pointer in the live graph that pointed at the old slab's memory (e.g. an
`Agent.conversation` field aimed at a `Conversation` that just migrated) — one pass over
the whole object graph, driven by `TypeInfo` exactly like GC marking (`02-runtime.md §6`),
rewrites each stale `FIELD_REF`/`FIELD_LIST_REF` from the old address to the new one at
the same slot-index. Once that pass completes, the old slab is freed, and the old
shape/bytecode for that class become unreachable and are collected normally. There is no
permanent "shape history" kept around — versioning here is a
**transient artifact of draining**, not a standing feature. `02-runtime.md`'s class table
is mutated in place; nothing is versioned once a migration finishes. This is a deliberate
answer to the "versioned classes vs in-place mutation" question the task poses: in-place
is the resting state, and a bounded two-version window (old shape draining, new shape
accepting) is the only exception, collapsing back to one version as soon as it's safe.

**Per-instance migration failure — this is what "predates field X" means.** A default
value cannot fail. A migration function can — it can throw, or (if it references another
entity that's itself mid-invalid-state) produce a value that fails a type/invariant
check. When that happens for a specific instance, that instance alone is **quarantined**:
left on the old shape, flagged, and any method compiled against the new shape refuses to
run on it:

```
⚠ Agent#12 did not migrate to shape v3
  migrate Agent v2 -> v3 threw: "status was null during migration"
  Agent#12 remains on shape v2 — field `summaryTone` unavailable
  calling Agent.summarize() (v3) on Agent#12 will raise:
    Agent#12 is on shape v2; summarize() requires v3 — no `summaryTone`
```

This is the viewer's "this instance predates field X" surface: an instance badge in the
detail pane, greyed-out invoke buttons for any method that needs the missing field, and
a link to the migration error. It is a real, first-class runtime state, not a UI
affordance bolted onto something the runtime doesn't know about — the object header's
`shape_id` literally differs from `class_table[Agent].shape`, and every dispatch checks
that.

**Dispatch during the drain window**, precisely: for a class currently mid-migration,
`methods[slot]` is not a single `FnPtr` but keyed by shape — `methods[shape_id][slot]`.
An instance still on the old shape calls old bytecode (correct offsets for its actual
layout); an instance already migrated calls new bytecode. This two-shape dispatch table
only exists while a migration is pending; classes untouched by a field change never pay
for it — they keep the flat `methods[slot]` from the "method body" case.

**Rejected alternative: lazy, permanent per-instance migration** (migrate on next access,
let two shapes coexist indefinitely). This looked attractive but breaks the enumeration
guarantee the whole memory model exists for: a permanently heterogeneous-stride arena
means "list all `Agent`s" stops being a uniform slab walk and becomes "walk N different
strides, dispatching per-instance." The bounded, self-collapsing drain window gets the
same flexibility (old and new can coexist briefly) without that permanent cost.

**Rejected alternative: refuse any field change on a class with live instances.** Too
restrictive for a system whose whole point is evolving code while it runs — this is
exactly the demo the vision doc leads with. Reserved for the *only* case where refusal
is actually correct: an add with no default and no migration function.

**OPEN: stuck migrations.** If a stale frame never returns — blocked forever on a tool
call, or a genuine infinite loop — `active_frames` for that class never reaches zero,
and no field edit to that class can ever land again. The honest behavior (never
forcibly reinterpret memory a live frame is reading) means this is a real operational
dead-end, not a bug to paper over. Whether there should be an operator-facing "kill this
frame" capability from the viewer to unblock it is undecided; it also needs a "kill a
frame" primitive that doesn't exist elsewhere in this design yet.

**OPEN: concurrency primitive for `active_frames`.** The increment/decrement above needs
to be atomic if the concurrency model (OPEN, `DECISIONS.md`) allows more than one OS
thread inside the interpreter at once. If agent turns are cooperatively scheduled on a
single interpreter thread, a plain counter suffices and this is a non-issue. Pinning this
down belongs to `02-runtime.md` / the concurrency doc; the invariant this section depends
on — "count reaches zero ⇒ safe to migrate" — must hold under whatever is decided, but
the exact primitive isn't chosen here.

### Class add

No live state to reconcile — add a `TypeInfo`, an empty `ShapeInfo`, an
empty arena. Effective the instant the generation swaps in. The only rejection is a name
collision, caught by ordinary whole-program typecheck.

### Class removal

Live instances of a removed class do not evaporate. Removing `class Task { ... }` from
source means: the new generation has no `Task` constructor reachable from any code path,
so `new Task(...)` can no longer be typechecked anywhere in the program — but any `Task`
instances that already exist keep their arena, keep their methods, keep answering to the
viewer and to any surviving handle. They're simply un-instantiable going forward, and the
viewer marks the type "removed — N instances remain" instead of offering a "new" action.
The arena is reclaimed only when its last instance is collected (or the process exits).

**Why not hard-delete on removal?** It would silently invalidate every surviving handle
into that arena and destroy live state the moment someone edited a class — directly
against the project's core promise that the runtime is a lens, never a shredder, over
what's actually running. A removed class becoming permanently un-instantiable, while its
existing citizens live out their process lifetime, is the version of "delete" that
doesn't contradict "your program's live state is never silently thrown away."

## The acceptance algorithm

Putting the two checks — static typecheck and live-state safety — together, the whole
edit-acceptance path is:

```
on save(new_source):
  new_program = parse(new_source)
  errors = typecheck_whole_program(new_program)        // ordinary static check,
  if errors: reject(errors); return                    // includes every call site

  diff = structural_diff(current_program, new_program) // per class: methods/fields
                                                         //   added/removed/changed

  live_errors = []
  for class_diff in diff.classes where live_instance_count(class_diff.class) > 0:
    for f in class_diff.fields_added:
      if not f.has_default and not diff.migration_fn_for(class_diff.class, f):
        live_errors.push(NoDefaultOrMigration(class_diff.class, f))

  if live_errors: reject(live_errors); return           // still nothing changes —
                                                         // current generation keeps running

  generation += 1
  new_class_table = compile_and_link(new_program, generation)
  swap(current_class_table, new_class_table)            // atomic — every new call sees it
  for class_diff where class_diff.shape_changed:
    schedule_migration(class_diff.class)                 // gated on active_frames == 0
```

Two independent kinds of rejection, and the developer-facing distinction matters:

- **Static rejection** ("`tone` is not defined") — an ordinary type error, nothing to do
  with liveness, would fail even against an empty program.
- **Live-state rejection** ("17 live `Agent`s exist and `notes` has no default") — the
  new program is *perfectly valid on its own*; it only fails because instances exist
  that it doesn't know how to become. This is the "type error against live state" the
  vision doc promises, and it's important that the message says *which* — a developer
  who sees "17 live instances" immediately understands why a program with zero syntax
  errors was refused, instead of hunting for a typo.

In both cases the guarantee is the same: **rejection is a no-op.** The current generation
keeps running, unaffected, exactly as if you'd never hit save. Nothing partially lands.

## What the viewer shows

- **Normal state**: entity type row, live count, nothing unusual.
- **Migrating**: `Agent — migrating to v3 (2 frames draining)` — transient, disappears
  the instant the drain completes and the arena flips.
- **Quarantined instance**: an instance-detail badge — `predates field summaryTone (shape
  v2) — migration failed, see error` — with the affected methods' invoke buttons
  disabled and a tooltip explaining why.
- **Removed type**: greyed out in the type index — `Task — removed, 8 instances remain` —
  still fully browsable, searchable, invokable; just not constructible.
- **Rejected edit**: shown at the point of editing (viewer code pane or external editor
  round-trip), not in the type index — the running program never enters a state the
  index needs to reflect, because rejection is a no-op.

## Summary of positions taken

1. One mechanism — whole-program recompilation into a new **generation**, swapped by a
   single pointer — explains method bodies, signatures, and (with an added migration
   step) fields. Three "different features" are one mechanism plus one extra gate.
2. Method/signature changes: instant, no coordination, because they never touch object
   layout. Stale frames finish on old bytecode; every *new* call, including recursion,
   re-dispatches to whatever's current.
3. Field changes: gated on a per-class quiescence counter, not a global stop-the-world.
   Migration is eager and total once safe, with per-instance quarantine as the failure
   path — not permanent dual-shape coexistence.
4. Class table mutation is in-place at rest; shape/version coexistence is a bounded,
   self-collapsing artifact of draining, never a standing multi-version model.
5. Rejection — static or live-state — is always a full no-op on the running program.
6. No inheritance removes the hardest part of most live-patching stories (override
   consistency across a hierarchy) before it starts; the method-table indirection this
   doc relies on exists purely for live redefinition, not dispatch.

**Flagged OPEN** (do not silently resolve elsewhere): stuck-migration operator override;
the atomicity primitive for `active_frames` under the not-yet-decided concurrency model;
first-class function values across a signature change.
