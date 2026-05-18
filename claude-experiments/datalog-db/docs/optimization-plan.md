# Optimization Plan

Companion to `performance-roadmap.md`. The roadmap focuses on the
query executor; this doc covers the broader optimization surface —
storage layer, write path, schema knobs, executor — and tracks what
has actually been done versus what remains.

## What's been done so far

Six rounds of work landed. Every round expanded the test suite; none
of them were benchmarked.

| Round | Change | Tests added | Measured? |
|---|---|---|---|
| 1 | `Durability` options (Sync / Buffered / MemoryOnly) | 5 | No |
| 2 | RocksDB tuning knobs (block cache, compression, bloom filter, write buffer) | 8 | No |
| 3 | `CachePolicy` (None / Bounded LRU / Unbounded) | 11 | No |
| 4 | Write-path refactor (BatchOverlay, `execute_batch`, in-memory counters, Rust mutex) | 0 new — verified existing 99 still pass | No |
| 5 | `transact_many` for explicit group commit | 7 | No |
| 6 | Background writer thread with auto-batching | 4 | No |

Net effect on the codebase:

- `StorageBackend::execute_txn` was replaced with `execute_batch`. The
  RocksDB `TransactionDB` lock manager is no longer engaged on the write
  path; coordination moved to a Rust `Mutex` on `Database`.
- `Database` carries in-memory `AtomicU64` counters for `tx_counter` and
  `entity_counter`, initialized from storage at open and rewritten into
  every batch.
- `BatchOverlay` adds read-your-own-writes (and read-your-prior-tx-writes
  within a group) by layering an in-memory map over the snapshot.
- `execute_group_batch` lets multiple user transactions share one
  RocksDB `WriteBatch` — one WAL append, one fsync.
- `transact_many` and a background writer thread are both built on top
  of that.

## Honest assessment of measurement

**No benchmarks have been run.** The performance claims attached to each
round are reasoned from architecture, not observed. The `benches/comparative.rs`
suite exists and would establish a baseline against PostgreSQL on 13
workloads, but it hasn't been re-run since these changes landed.

What the work *does* establish:

- **Correctness invariants** are tested: 113 tests covering durability
  round-trips, cache eviction order, per-tx rollback in groups,
  read-your-prior-writes across group members, asOf ordering preserved,
  20-thread unique-email race resolved to exactly one winner, clean
  writer-thread shutdown.
- **Tunable surface area** exists where it didn't before: callers can
  pick durability, set RocksDB tuning knobs, bound cache memory, opt into
  group commit.

What the work does *not* establish: that anything got faster.

**To actually measure:**

1. `cargo bench --bench comparative` on the current tree. Compare ratios
   to those documented in `performance-roadmap.md` (which is a snapshot
   of earlier numbers).
2. Add a concurrent-writer benchmark. The existing comparative is
   single-threaded — the writer-thread work is invisible to it. Need
   an N-thread `transact()` storm with and without `group_commit`
   enabled.
3. `cargo flamegraph` on Graph 2-Hop (the 11.7× workload). The roadmap's
   diagnosis is reasoning-from-code; a profile would either confirm or
   redirect the priority order below.

## Remaining work

### A. Storage-layer wins (format-affecting)

#### A1. Attribute interning

Every index key inlines the attribute name as `[u16 len][bytes]`. For
`Metric/value` that's 14 bytes of attribute string per key, repeated
across 5 keys per datom (EAVT + AEVT + AVET + CURRENT_AEVT +
CURRENT_AVET, plus VAET when the value is a ref). For a 1M-datom DB
on one attribute, that's ~70 MB of just repeated attribute names in
keys before compression.

**Mechanism:**
- A new in-memory `AttrInterner` maps `String ↔ u32` (forward and
  reverse).
- Persisted as meta keys `attr:<name>` → `u32` (big-endian). Forward is
  the source of truth; reverse is built in memory at open.
- IDs allocated on first write, atomically with the datom that uses
  them (the `attr:<name>` put goes into the same WriteBatch as the
  data datom).
- Index encoding replaces `encode_attr(buf, &str)` with
  `encode_attr_id(buf, u32)` — 4 bytes instead of `[u16 len][bytes]`.
- Decode functions return an `AttrId` instead of a `String`; callers
  resolve back to a name via the interner.

**Migration:** breaking format change. Old databases can't be read by
new code unless we ship a one-time scan-and-rewrite or version-gate the
storage layout. For this codebase (experimental, no shipped data), we
can just accept the break.

**Plausible impact (unmeasured):** ~20-40% smaller index keys for
typical schemas. SSTs shrink proportionally; block cache fits more
keys; read bandwidth from disk drops. This is a real-world Datomic
technique because it pays back consistently.

#### A2. Column family split

Today everything lives in RocksDB's default column family —
CURRENT_AEVT/CURRENT_AVET (hot, point-read shaped), EAVT/AEVT/AVET/VAET
(history, append shaped, scanned only for time-travel queries), meta
keys (tiny, hot). Different access patterns, same SSTs, same block
cache, same compaction.

**Mechanism:**
- Three CFs: `current` (CURRENT_*), `history` (EAVT/AEVT/AVET/VAET),
  `meta` (counters, timestamps, schema, attr table).
- Per-CF options: `current` gets bloom filters + LZ4 + cache priority;
  `history` gets Zstd + larger SSTs + no bloom; `meta` is tiny.
- Storage backend dispatches puts by inspecting the key's prefix byte.
- Indexes that scan range stay within one CF — no cross-CF iteration
  needed.

**Migration:** same breaking-format problem as A1. Could batch with A1.

**Plausible impact (unmeasured):** point lookups (most of the unique
constraint and update path) get bloom filters → 2-5× faster on
negative lookups. Type cache loads stop competing with cold history
data for block cache space.

### B. Executor wins (no format change)

These come from `docs/performance-roadmap.md` and don't touch on-disk
state. Order is roughly highest-leverage first.

#### B1. Inline tuple slots

`Tuple` is `Vec<Option<Value>>` — heap allocation per tuple, full clone
on every join merge. Graph 2-Hop generates ~10k intermediate tuples;
each is one Vec alloc plus N `Value::String` clones during merges.

Replace with:
```rust
struct Tuple {
    slots: [Option<Value>; 16],
    len: u8,
}
```

Stack-resident, fixed-size memcpy for the container (string contents
still allocate, but the wrapper doesn't).

**Mechanism:** change the `Tuple` definition in `query/executor.rs`,
update every site that does `tuple.clone()` / `tuple.get()` /
`tuple.set()` / `merge_tuples()`. Mechanical but touches many functions.

**Plausible impact:** roadmap claims 20-40% on Graph 2-Hop, smaller on
scans.

#### B2. `Arc<str>` for `Value::String`

`Value::String(String)` is a heap-allocated owned string. Every
`tuple.clone()` clones the inner string. Switch to `Arc<str>` — clone
is a refcount bump.

**Mechanism:** change `Value::String` payload. Cascades through every
constructor (`"...".to_string()` → `Arc::from("...")`) and every match
pattern. Probably 30-50 sites.

**Plausible impact:** significant on any join-heavy workload. Hard to
estimate without measuring.

#### B3. Columnar cache layout

`QueryCache::types` is `HashMap<String, Arc<HashMap<EntityId, HashMap<String, Value>>>>` —
a map of maps. Scanning all entities of a type means iterating a
HashMap (random memory access), and for each entity, hashing the
field-name string per lookup.

Replace per-type with:
```rust
struct TypeData {
    entity_ids: Vec<EntityId>,                       // sorted
    columns: HashMap<String, Vec<Option<Value>>>,    // one Vec per attr
}
```

Scanning `salary > 80000` becomes iterating `columns["salary"]`
linearly. Sequential access, SIMD-friendly, no per-entity hash lookups.

**Mechanism:** rewrite `load_type_data` to produce the new shape;
rewrite the executor's cached-path inner loops; entity-by-id lookups
become binary search on `entity_ids` + index into columns.

**Plausible impact:** roadmap claims 2-3× on scan workloads.

#### B4. Pre-resolved field offsets

With B3 in place, attribute names can be resolved to column indices at
plan time. The hot loop replaces a `HashMap<String, Pattern>` lookup
per field per entity with an array index.

**Mechanism:** planner records `Vec<(column_idx, Pattern)>` on each
`ClauseScan`. Executor uses `td.columns[i][entity_idx]`.

**Plausible impact:** 20-30% on scan-heavy queries. Compounds with B3.

#### B5. Plan cache eviction

`Database.plan_cache` is unbounded. Apps building queries dynamically
leak plans into the cache forever. Same machinery as the
`QueryCache::Bounded { max_types }` work, applied to a smaller map.

### C. Deferred — only after A/B land

#### C1. Volcano/iterator execution model

Replace "each plan node returns a `Vec<Tuple>`" with "each plan node
has `next() -> Option<Tuple>`". The win is eliminating intermediate
vec materialization on Graph 2-Hop and enabling streaming / LIMIT.
Several thousand lines of churn. Only worth it after the cheaper
executor wins are exhausted. (Roadmap item 7.)

#### C2. Bytecode / compiled expression evaluation

Replace interpreted pattern matching with a flat instruction sequence
or compiled Rust closures. Big lever for repeated queries, also big
code. Same advice: defer. (Roadmap item 8.)

#### C3. Batch clause evaluation

Fold sequential `evaluate_clause` calls into one batched call. Pairs
naturally with B3 (columnar). Wait until B3 lands.

## Recommended next step (revised, after this doc)

Run `cargo bench --bench comparative` to establish a baseline, *then*
decide between A1/A2 (storage) and B1-B4 (executor) based on what the
benchmark actually shows is slow. Pretending we know the priority
without measuring is the same mistake the past six rounds made.

User has picked **A1 (attribute interning)** as the next round
regardless of benchmark results. That is what gets implemented next.
