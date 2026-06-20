# Runtime reflection & heap exploration

The GC's `TypeInfo` table describes an object's **shape** — pointer-slot count,
raw bytes, varlen tail — which is all the collector needs to trace it. It carries
no names: two structs with the same shape are indistinguishable to the GC.

Reflection adds a parallel, *cold* **metadata** table (`gc::reflect::TypeMeta`,
one entry per `type_id`) holding the **nominal** information the shape omits:
source type name, field names, field byte offsets, field types, and — for enums
— per-variant tags and payloads. This is what turns `<type 7: 2 fields>` into
`Point { x: 3, y: 4 }`.

## How it flows

The metadata is built once, during layout lowering (`src/layout.rs`), and travels
inside each `core::Layout` as its `meta` field, guaranteed 1:1 with the layout.
It reaches the runtime by two routes that share one encoder/decoder
(`gc::reflect::encode`/`decode`) so they can't drift:

- **JIT** (`gcr run`): `layouts_to_type_meta` hands the table straight to
  `Heap::set_type_meta`.
- **AOT** (`gcr build`): the table is encoded into a byte blob baked into the
  executable as `gcrust_type_meta`, and `gcr_runtime_main` decodes it at startup.

Field offsets are **absolute from the object pointer** (they include the 16-byte
header), so a tool with only an object pointer and a `FieldMeta` can read the
field directly. Enum payload offsets match codegen's placement exactly (pointer
payloads in the shared pointer slots, scalar payloads after the tag word).

## Surfaces

### `gcr emit reflect <file>`

Dumps the full metadata table as JSON — exactly what is installed into the heap
(JIT) or baked into the binary (AOT). One entry per `type_id`, with `struct` /
`enum` / `opaque` kinds. Useful for tooling and agents.

```
gcr emit reflect prog.gcr
```

### `GCR_HEAP_DUMP=1` — host-side heap dump

Set the env var and the runtime renders the live object graph at program end
(works for both `gcr run` and AOT binaries). Pointers resolve to per-dump object
ids so the output reads as a graph:

```
── gc-rust heap dump: 5 objects ──
   by type: List×4  Point×1

#0 Point { x: 3, y: 4 }
#1 List::Nil
#2 List::Cons(12, #1)
#3 List::Cons(20, #2)
#4 List::Cons(10, #3)
```

Strings show their contents (`String "Ada"`); other varlen builtins show their
element count. The dump walks all *allocated* objects in the live spaces (tenured
from-space + nursery), which between collections may include unreclaimed garbage
in the nursery — trigger a collection first for a strictly-live snapshot.

Host tooling can call `gc::dump_heap_text(heap)` directly, or build on
`Heap::walk_live_objects` + `Heap::type_meta_by_id` for custom renderers.

### `GCR_HEAP_DUMP=json` — structured snapshot

Set the value to `json` for a machine-readable snapshot (`gc::dump_heap_json`):
a `summary` (object/byte totals, a per-type histogram, and `roots` — the
in-degree-0 top-level objects with their transitive `reachable_bytes`) plus the
full object graph (each object's `id`, `type`, `bytes`, human `render`, and
outgoing `refs` edges). Edges come from the GC's own `scan_object`, so they
include interior pointers in flattened value fields and varlen elements.

```json
{
  "summary": { "objects": 5, "bytes": 184,
    "by_type": [{"name":"List","count":3,"bytes":120}, {"name":"Point","count":2,"bytes":64}],
    "roots": [{"id":4,"reachable_bytes":184}],
    "top_retainers": [{"id":4,"retained_bytes":184}, …] },
  "objects": [
    {"id":4, "type":"List", "bytes":40, "retained_bytes":184, "render":"List::Cons(#0, #3)", "refs":[0,3]}, …
  ]
}
```

Each object carries an **exclusive retained size** (`retained_bytes`): the total
bytes that become unreachable if it is removed — the size of the subtree it
dominates. Computed with the Cooper-Harvey-Kennedy dominator algorithm over the
snapshot graph (virtual super-root → the in-degree-0 proxy roots, with an entry
injected into any pure cycle). A shared object is charged to its nearest common
dominator, not double-counted under each referrer — the standard heap-profiler
metric for "what's actually holding memory." `summary.top_retainers` lists the
biggest.

It's a program-end snapshot (no live stack root set), so it covers all
currently-allocated objects; roots/retention approximate from the top-level
structures.

### In-language builtins

All require a heap (reference) value as the first argument.

- `type_id_of(x) -> i64` — the runtime `type_id` from `x`'s header.
- `type_name_of(x) -> String` — the source type name.
- `field_count(x) -> i64` — number of reflectable fields: a struct's fields, or
  the **active** variant's payload for an enum (0 for opaque builtins).
- `field_name(x, i) -> String` — name of field `i` (`"0"`,`"1"`,… for positional).
- `field_kind(x, i) -> i64` — `0`=ref, `1`=int, `2`=float, `3`=bool, `4`=char,
  `5`=value-aggregate.
- `field_i64(x, i) -> i64` — field `i` widened to i64: a scalar decoded by its
  kind (sign-extended ints, float *bits* for floats) or a ref's pointer bits.
  Pair with `field_kind` to interpret it.

An index out of `0..field_count` is a hard error (call `field_count` first).
Example — a generic field dump written in the language itself:

```rust
let p = Point { x: 3, y: 4 };
print(type_name_of(p));                       // "Point"
let n = field_count(p);
let mut i = 0;
while i < n {
    print(field_name(p, i)); print("=");
    print(to_string(field_i64(p, i))); print(" ");   // x=3 y=4
    i = i + 1;
}
```

## Value types (flattened `#[value]` aggregates)

A `#[value]` struct is stored *flattened* inline in a heap object's field (no
header, no separate allocation) — the same model as Java's Project Valhalla,
which carries a value's type in the enclosing field/array metadata rather than a
per-instance header. gc-rust mirrors this: a flattened field is a
`FieldTy::Value(value_id)` pointing into a parallel **value-metadata table**
(`gc::ValueMeta`, indexed by `value_id`), and the renderer recurses into it,
composing the value's (value-relative) field offsets with the field's own offset.
So the heap dump shows the real contents, recursively:

```
#0 Line { a: Point { x: 3, y: 4 }, b: Point { x: 5, y: 6 } }
```

The GC also has the machinery for references *embedded inside* a flattened value:
`TypeInfo.interior_ptrs` lists absolute byte offsets of GC pointers in the raw
region, and `scan_object` traces them. (`gcr emit reflect` exposes the value
table alongside the type table.)

## References inside flattened value structs

A `#[value]` **struct** may contain GC references and still be flattened into a
heap object — both halves of GC correctness are wired:

- **Heap side**: each embedded ref's absolute byte offset is recorded in
  `Layout.interior_ptrs` → `gc::TypeInfo::interior_ptrs`, so `scan_object` traces
  and relocates it. Computed in `layout.rs` (recursing nested value structs),
  carried to the JIT type table directly and baked into AOT binaries via the
  metadata blob.
- **Stack side**: a value-with-ref *local* lives in a plain alloca, so its refs
  sit in untraced stack memory. Codegen registers each interior ref's *address*
  as an **indirect frame root** (`FrameOrigin.num_indirect` + a trailing slot
  array); `walk_gc_frames` dereferences it and relocates the pointer in place
  inside the alloca. Locals and params are covered.

## GC-temporary rooting (ANF)

A precise moving GC can only find pointers in frame root slots (declared locals).
A GC value computed as an *inline temporary* — held in a register while a later
operand or the enclosing allocation runs — is invisible to the collector. The
`src/anf.rs` pass rewrites every function body so that every non-atomic GC-valued
subexpression is let-bound to a fresh local before use; combined with the
allocation-time **reload** in `gen_alloc` (a GC field cached before the alloc
safepoint is reloaded from its relocated slot after), no live pointer is ever
stranded across a collection. This closes a whole-language latent gap (it applied
to reference temporaries too), not just value types. Validated with
`GcRunMode::SemiSpace(small)` — a tiny heap that forces real collections during a
program's own construction.

Both `gen_alloc` (heap objects) and `gen_make_closure` (closure environments)
apply this reload, since both allocate. A missing reload in `gen_make_closure`
was the real cause of a thread-spawn crash under stress: `Thread::spawn` wraps the
user closure, and the wrapper env's captured closure pointer went stale across the
env allocation. (The cross-thread env *handoff* itself was already rooted via a
global slot.)

`--gc-stress` now genuinely collects on **every** allocation (it had been
ignoring the flag) and is sound single- and multi-threaded — the strongest
validation of precise rooting.

## Value enums with references

A `#[value]` enum carrying a reference now works as a heap field and as a local,
via a **pointers-first layout** that mirrors reference enums: every variant's
pointer payloads share the leading slots at fixed offsets `0, 8, …`, with the tag
and raw (scalar/POD-value) payloads after. So an embedded ref's offset is
independent of the runtime tag, expressible in the static interior-pointer (heap)
and indirect-root (stack) lists. Ref-free value enums keep their compact
`{ tag, raw }` form unchanged. (A *nested* value-with-reference payload inside a
variant is still rejected — its refs would sit at tag-dependent sub-offsets.)

## Limitations (current)

- A nested value-with-reference payload inside a value-enum variant is rejected
  (clear compile error, never corruption).
- Value enums flattened in heap fields render as `Name <value>` in dumps (their
  tag/union offsets aren't decoded yet); value structs render fully.
- Bare scalars and value aggregates that are never stored in a heap object have
  no `type_id` and aren't reachable from a heap dump.
- In-language reflection currently exposes name/id only; structural field
  iteration from within the language can be layered on the same metadata next.
