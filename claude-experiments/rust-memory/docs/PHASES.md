# Phases / spans: tagging allocations with program semantics

Today an allocation is tied to its **call stack** + recovered **type**. That
answers "what code allocated this." It can't answer "what was my program *doing*"
— loading? indexing? serving request #42? The same `Vec::with_capacity` runs in
all three. **Phases** add that semantic axis: you sprinkle a few macros to mark
regions of execution, and every allocation made inside one is attributed to it.

This is the `tracing`-span idea, but the payload is allocations instead of log
events.

## What you'd write

### Scoped guard (the common case)

```rust
fn run() {
    let _p = memscope::phase!("load");          // enter "load"; exits on drop
    let data = load_dump(path);                  // allocs here -> phase: load

    {
        let _p = memscope::phase!("index");      // nested span
        build_index(&data);                      // allocs here -> phase: load > index
    }                                            // "index" exits

    serve(&data);
}                                                // "load" exits
```

### On a function (attribute macro)

```rust
#[memscope::phase]                  // phase name defaults to "build_index"
fn build_index(d: &Data) { … }

#[memscope::phase("index")]         // …or name it explicitly
fn build_index(d: &Data) { … }
```

### With metadata

```rust
let _p = memscope::phase!("request", id = req.id, route = req.path);
// allocations tagged: phase=request{ id: 42, route: "/pets/9" }
```

### Free, if you already use `tracing`

A `memscope::TracingLayer` maps existing spans to phases, so a codebase already
annotated with `#[tracing::instrument]` gets allocation-phases with **zero** new
macros:

```rust
tracing_subscriber::registry().with(memscope::TracingLayer).init();
```

## How it works — phases are an event stream, not a hot-path field

Key design choice: **the allocation hot path does not change at all.** A phase is
recorded as its own lightweight event.

`phase!` / the guard call `enter_phase(id)` / `exit_phase(id)`, which push a
`PhaseEnter` / `PhaseExit` marker onto the *same per-thread ring* the allocator
already uses — carrying an interned phase id, the thread, and a timestamp. That's
one ring push per *phase boundary* (coarse — thousands of them, not billions), so
it's free. Allocation records gain **nothing**: no extra field, no extra TLS read
on the hot path.

At analysis time the reader replays each thread's events *in order* and keeps a
phase stack:

```
for ev in thread_events_in_order:
    PhaseEnter(id) -> phase_stack.push(id)
    PhaseExit(id)  -> phase_stack.pop()
    Alloc          -> alloc.phase_path = phase_stack.snapshot()   // e.g. ["load","index"]
```

Because the ring preserves causal order per thread (it already does — that's how
the live set is reconstructed), the phase active at each allocation is exact.

> Alternative considered: stamp the current phase id on every `RawEvent` in the
> hot path (one TLS read + one field). Simpler reader, but it touches the hot path
> and the POD layout, and costs something per allocation. The event-stream model
> is free for allocations and fully backward compatible, so it wins.

## Plumbing (small, additive)

- **memscope-core**: `phase_id(&'static str) -> PhaseId` (interned, cached at the
  call site so each enter is just a `u32`); `PhaseGuard::enter(id)` /
  `Drop`; a thread-local `Vec<PhaseId>` for the live nesting; `emit_phase(kind,
  id)` pushes the marker event.
- **memscope-proto**: two new `EventKind`s — `PhaseEnter = 3`, `PhaseExit = 4`.
  The phase id rides in the existing `site` field (free for these kinds), so
  `RawEvent` is unchanged. A new `recfmt` record `TAG_PHASE { id, name, kvs }`
  written once per phase (exactly like the interned site records). Old readers
  just skip event kinds they don't recognize → backward compatible.
- **memscope-macros** (new, tiny): `phase!` (a `macro_rules!` wrapping the guard +
  call-site interning) and the `#[phase]` attribute (a small proc-macro that drops
  a guard at the top of the function body).
- **tools**: the reader gains phase correlation; the views gain phase awareness
  (below).

## What you get in the views

- **`flamegraph --by-phase`** — phases become the *root* levels of the flame:
  `load → index → <call stack> → [Type]`. One glance shows `index` cost 900 MB vs
  `query` 1.8 GB, each still drillable into its stacks.
- **`flamegraph --phase index`** — filter to a single phase's allocations.
- **`flamechart`** — phases render as labeled **background spans / lanes** across
  the time axis (Perfetto-style), with the per-allocation flame underneath: you
  literally see the allocations happening *inside* the `index` span from t1..t2,
  and a leak shows as bytes that enter during `request` and never leave.
- **`replay` / `perfetto`** — group `live_bytes` by phase: a per-phase memory
  curve over time, and "bytes still live, by the phase that allocated them" (leak
  attribution by program stage, not just by stack).

## Overhead

- Per allocation: **zero** added cost.
- Per phase boundary: one ring push (~ns). Phases are coarse, so negligible.

## Why phases beat "just use the stack"

Phases are *orthogonal* to the call stack:

1. The same leaf (`Vec::with_capacity`, `HashMap::insert`) allocates in every
   phase — the stack can't separate them once entry points converge.
2. Phases are often **time-shaped**, not call-shaped: a "tick", a "frame", a
   "request" is a region of *time* that spans many unrelated call trees.
3. Phases carry **domain metadata** (request id, file being parsed, tenant) that
   no stack frame knows.

You keep the stack (how) and gain the phase (what for).

## Minimal first slice (if we build it)

1. `enter_phase`/`exit_phase` + `PhaseGuard` + `phase!` (macro_rules, no
   proc-macro yet) in core; the two new event kinds + `TAG_PHASE` in proto/recfmt;
   the file recorder already writes whatever the ring carries.
2. Reader: phase correlation + `flamegraph --by-phase` and `--phase <name>`.
3. Demo: tag `serve`'s workload with two or three phases and show the split.

Everything after — the `#[phase]` attribute, dynamic kv metadata, the `tracing`
layer, flamechart phase lanes — layers on without reworking the core.
