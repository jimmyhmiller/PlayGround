# Arbitrary metadata on allocations

An allocation already knows its **call stack** (how) and recovered **type**
(what). Metadata adds *your* axis: attach arbitrary key/value context to a region
of execution, and every allocation made inside it inherits that context. Then the
views can group, filter, and color by **any key you define**.

"Phase" is not a built-in concept — it's just `meta!(phase = "index")`. So is
`meta!(request = 42, tenant = "acme")`, `meta!(subsystem = "parser", file = path)`,
`meta!(frame = n)`. The library ships the *mechanism*; the keys are yours.

## What you'd sprinkle

### Scoped guard — arbitrary keys/values

```rust
let _m = memscope::meta!(subsystem = "parser", file = path);  // push context
parse(input);                                                 // allocs here inherit it
// ... drops at end of scope -> context pops

let _m = memscope::meta!(request = req.id, tenant = req.tenant, route = req.path);
handle(req);
```

Contexts **nest and merge** — an allocation inside both sees the union:

```rust
let _a = memscope::meta!(subsystem = "index");
{
    let _b = memscope::meta!(shard = 3);
    insert(x);     // tagged { subsystem: "index", shard: 3 }
}
```

### On a function (attribute)

```rust
#[memscope::meta(subsystem = "query")]
fn run_query(q: &Q) { … }
```

### Free, via `tracing`

A `memscope::TracingLayer` maps `tracing` span **fields** straight to metadata —
so `#[tracing::instrument(fields(tenant = %id))]` already-annotated code gets
allocation metadata with no new macros.

### One allocation, not a scope (extension)

```rust
let cache = memscope::tagged(HashMap::new(), [("role", "cache")]);  // tags this value's backing alloc
```

## Data model

- **Keys**: `&'static str`, interned to a small id space (you have a handful of
  distinct keys, reused everywhere).
- **Values**: a small typed enum — `Str | Int(i64) | Uint(u64) | F64 | Bool`.
  Covers the common cases and stays flat; anything structured is stringified at
  the call site.
- A **context** is a set of `(key_id, value)` pairs.

Dynamic values (`request = 42`, where 42 changes every call) are fine: the value
rides **inline** in the scope-enter event, so high-cardinality values don't bloat
any intern table — only the *keys* are interned.

## How it works — a context event stream (zero hot-path cost)

Same insight as everything else here: **the allocation hot path does not change.**

`meta!` pushes a frame onto a thread-local context stack and emits a `MetaEnter`
event onto the *same per-thread ring* the allocator uses, carrying the inline
key/value pairs + thread + timestamp. The guard's `Drop` emits `MetaExit`. That's
one ring push per *scope boundary* (coarse), not per allocation. Allocation
records gain nothing — no extra field, no extra TLS read on the hot path.

At read time the reader replays each thread *in causal order* (the ring already
guarantees this) and keeps the context stack:

```
MetaEnter(kvs) -> ctx_stack.push(kvs)
MetaExit       -> ctx_stack.pop()
Alloc          -> alloc.meta = merge(ctx_stack)     // { subsystem:"index", shard:3 }
```

So every allocation carries the exact merged context that was live on its thread
at that instant.

> Why not stamp metadata on each `RawEvent` in the hot path? It would cost a TLS
> read + a variable-length payload per allocation, and metadata is far coarser
> than allocations. Recording it as its own scope-level stream keeps allocations
> free and is backward compatible (old readers skip unknown event kinds).

## Plumbing (small, additive)

- **memscope-core**: a thread-local `Vec<ContextFrame>`; `push_meta(&[(KeyId,
  Value)]) -> MetaGuard` / `Drop`; `key_id(&'static str)` interner;
  `emit_meta(Enter/Exit, …)` ring push. No change to the allocator path.
- **memscope-proto**: two new `EventKind`s, `MetaEnter`/`MetaExit`. `MetaEnter`'s
  payload is a variable-length kv list — written as a new `recfmt` record
  (`TAG_META { thread, ts, [(key_id, value)…] }`), with a small `TAG_KEY` table
  interning key names. `RawEvent` itself is unchanged. Old readers skip the new
  kinds → backward compatible.
- **memscope-macros** (new, tiny): `meta!{ k = v, … }` (a `macro_rules!` that
  interns keys at the call site and builds the guard) and the `#[meta(...)]`
  attribute (proc-macro dropping a guard at the top of the body).
- **tools**: the reader merges the context stack onto each allocation; the views
  query it generically (below).

## What you get in the views — generic over keys

Because every allocation ends up with a `Map<key, value>`, the views take the key
to operate on as an argument — nothing is hard-coded to "phase":

- **`flamegraph --group-by subsystem`** — the values of `subsystem` become the
  root levels of the flame (`parser → <stack> → [Type]`, `index → …`). Swap the
  key to re-pivot the same data.
- **`flamegraph --filter tenant=acme`** — restrict to allocations whose context
  matches; combine filters (`--filter subsystem=index --filter shard=3`).
- **`flamechart --color-by request`** / **`--lane subsystem`** — color frames by a
  key, or render each value of a key as its own background lane over the timeline.
- **`replay --by subsystem`** — live-bytes (and leaked-bytes) broken down by any
  key: leak attribution by *your* semantics, e.g. "3 MB still live, all tagged
  `request`, none freed."

## Cardinality & overhead

- Per allocation: **zero** added cost.
- Per scope boundary: one ring push + the inline kv bytes (~tens of bytes). Scopes
  are coarse, so negligible.
- Keys interned (tiny). Values inline (no table to blow up). A `meta!` with a
  unique value per call (a request id) is fine — it's one small event per scope,
  not per allocation.

## Why this is the right primitive

It's one mechanism — *scoped key/value context recorded as events and merged at
read time* — and everything else is a use of it:

| you want | you write |
|---|---|
| phases | `meta!(phase = "index")` |
| per-request attribution | `meta!(request = id)` |
| subsystem / module tagging | `meta!(subsystem = "parser")` |
| per-frame (game loop) | `meta!(frame = n)` |
| tenant / multi-tenant leaks | `meta!(tenant = id)` |

You keep the stack (*how*) and the type (*what*), and add an open-ended number of
your-own axes (*for what*) — at no per-allocation cost.

## Minimal first slice (if we build it)

1. `key_id` + context stack + `MetaGuard` + `meta!` (macro_rules) in core; the two
   event kinds + `TAG_KEY`/`TAG_META` records in proto/recfmt; the file recorder
   already streams whatever the ring carries.
2. Reader: context merge onto allocations + `flamegraph --group-by <key>` and
   `--filter <key>=<val>`.
3. Demo: tag the `serve` workload with two keys (e.g. `subsystem`, `phase`) and
   show re-pivoting the same recording by either.

The `#[meta]` attribute, the `tracing` layer, per-allocation `tagged`, and
flamechart lanes/coloring layer on without reworking the core.
