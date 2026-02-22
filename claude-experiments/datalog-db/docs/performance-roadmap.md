# Performance Roadmap

Current benchmark results after implementing the type-level entity cache:

| Workload | datalog-db | PostgreSQL | Ratio |
|---|---|---|---|
| Single inserts (1000) | 0.096s | 0.153s | 0.63x |
| Batch insert (1000) | 0.051s | 0.078s | 0.65x |
| Point query (1000x) | 0.033s | 0.095s | 0.35x |
| Range query (1000x) | 0.479s | 0.286s | 1.68x |
| Join query (1000x) | 0.136s | 0.311s | 0.44x |
| Mixed 4-thread (2000 ops) | 0.537s | 0.117s | 4.57x |
| 3-Way Join (500x) | 0.069s | 0.075s | 0.92x |
| 4-Way Star Join (500x) | 1.861s | 1.160s | 1.60x |
| Large Fan-Out (200x) | 1.153s | 0.342s | 3.37x |
| Graph 2-Hop (500x) | 3.365s | 0.288s | 11.68x |
| Selective Scan (1000x) | 0.758s | 0.191s | 3.97x |
| Many-to-Many Join (500x) | 0.188s | 0.202s | 0.93x |
| Unfiltered Scan (500x) | 0.977s | 0.258s | 3.79x |

We beat PostgreSQL on writes, point queries, simple joins, and many-to-many joins. The remaining gaps fall into three categories: graph traversal (11.68x), scans (3-4x), and fan-out/mixed workloads (3-4x).

## Execution Model: How We Differ from PostgreSQL

Two fundamental architectural differences explain most of our performance gap.

### Materializing Tree-Walk vs. Pipelined Volcano

PostgreSQL uses a **volcano/iterator model**: each plan node exposes a `next()` method that produces one row at a time. Rows flow through the join pipeline without ever being collected into intermediate lists. A nested loop join calls `left.next()` to get one row, then repeatedly calls `right.next()` to find matches, emitting joined rows one at a time. Memory usage is O(1) per node regardless of how many rows pass through.

We use a **materializing tree-walk**: `execute_node` pattern-matches on the `PlanNode` enum, recurses into children, and each child returns its **entire result set** as a `Vec<Tuple>`. A nested loop join calls `execute_node(left)` to get *all* left tuples, then for each one calls `evaluate_clause` which returns *all* matches as another Vec.

```rust
// Our model: materialize everything at every stage
fn execute_node(node: &PlanNode, ...) -> Vec<Tuple> {
    match node {
        PlanNode::Scan(scan) => evaluate_clause(...),  // returns full Vec
        PlanNode::Join { left, right, .. } => {
            let left_tuples = execute_node(left, ...);   // materialized
            let right_tuples = execute_node(right, ...);  // materialized
            hash_join(left_tuples, right_tuples)           // third Vec
        }
    }
}
```

For Graph 2-Hop, this means three fully materialized Vecs (400 + ~400 + ~400 tuples) where PostgreSQL holds at most one row per node. The allocation and cloning cost of building these Vecs is a significant chunk of the gap.

Moving to a volcano/iterator model would let us:
- Stream rows through joins without intermediate allocation
- Short-circuit early (e.g., LIMIT) without computing the full result
- Reduce peak memory from O(left × right) to O(left + right) for nested loop joins

### Interpreted Field Matching vs. Bytecode/JIT Expression Evaluation

PostgreSQL compiles WHERE clause predicates into a flat bytecode instruction sequence (and optionally JIT-compiles them to native code via LLVM for complex expressions). Evaluating `department = 'Engineering' AND salary > 80000` is a tight loop over a few opcodes.

We interpret field patterns dynamically. `evaluate_clause` iterates over a `HashMap<String, Pattern>`, does a string-keyed lookup per field, then matches each `Pattern` variant:

```rust
for (field_name, pattern) in &clause.field_patterns {
    match pattern {
        Pattern::Constant(val) => { /* string lookup + equality check */ }
        Pattern::Predicate(op, val) => { /* string lookup + comparison */ }
        Pattern::Variable(slot) => { /* string lookup + slot assignment */ }
    }
}
```

Per entity, this is: HashMap lookup by string key → pattern match → value comparison. PostgreSQL does: array index by column offset → direct comparison. The string hashing and pattern dispatch add overhead that compounds across thousands of entities.

A bytecode interpreter or compiled expression evaluator would replace the HashMap + pattern match with a flat instruction sequence operating on slot offsets, eliminating per-field hashing and dynamic dispatch.

## Where Time Goes

### Tuple Representation

Every intermediate result tuple is a `Vec<Option<Value>>`. This means:

- 24-byte Vec allocation overhead per tuple
- Full deep clone on every join match (`tuple.clone()` in `evaluate_clause`, `merge_tuples`, hash join probe)
- `Value::String` cloning allocates new heap strings

For a Graph 2-Hop query that produces ~10,000 intermediate tuples with 8 slots each, we're doing thousands of Vec allocations and string clones just to pass data between join stages. PostgreSQL uses fixed-offset tuple slots in shared buffers with zero-copy access.

Key sites: `executor.rs` Tuple struct, `merge_tuples()`, every `tuple.clone()` call in `evaluate_clause`.

### Graph 2-Hop (11.68x)

The query self-joins Employee three times: find engineers, look up their managers, look up their managers' managers. The execution plan nests three `evaluate_clause` calls:

1. ~400 engineers via AVET department="Engineering"
2. For each, look up manager by ref — 400 cache lookups, each producing a new tuple
3. For each manager, look up *their* manager — another 400+ lookups and tuple clones

The core issue is that each hop through the nested loop creates and clones tuples one-at-a-time. There's no batching — the inner loop calls `evaluate_clause` once per outer tuple. PostgreSQL's executor pipelines rows through join nodes without materializing intermediate results.

Additionally, the hash join builds a `HashMap<JoinKey, Vec<&Tuple>>` where `JoinKey` is `Vec<HashableValue>` — itself an allocation. Every probe computes a fresh key, hashes it, and clones matching tuples during merge.

### Scan Workloads (3-4x)

For unfiltered scans (all 2000 employees), the type cache is already warm after the first iteration. The remaining cost is:

1. Iterating over `HashMap<EntityId, HashMap<String, Value>>` — random memory access pattern, poor cache locality
2. Building a `Tuple` per entity — Vec allocation + field cloning
3. Projecting output — another Vec<Value> per result row

PostgreSQL does a sequential heap scan with fixed-width tuples in page-aligned buffers. Our HashMap-of-HashMaps layout scatters data across the heap.

For selective scans (department + salary filter), the AVET index intersection allocates a `HashSet` per pattern to intersect candidates. Then field values are pulled from the type cache one-by-one.

### Fan-Out and Multi-Thread (3-4x)

Large fan-out (5000 posts, 1000 users, join on author) produces many intermediate tuples. The hash join builds a full hash table of all posts, then probes for each user. Each match clones both tuples and merges them.

Mixed 4-thread is bottlenecked on RwLock contention on the type cache during concurrent writes + reads, plus per-query overhead that doesn't amortize well.

## What To Work On

### 1. Inline Tuple Slots

Replace `Vec<Option<Value>>` with a fixed-size array:

```rust
struct Tuple {
    slots: [Option<Value>; 16],  // stack-allocated, covers all current queries
    len: u8,
}
```

This eliminates the Vec allocation per tuple. For joins producing 10k+ tuples, that's 10k fewer heap allocations. Clone becomes a fixed-size memcpy instead of a heap-allocating Vec clone (Value::String still allocates, but the container doesn't).

Go further with an arena allocator: allocate all tuples for a join stage from a single contiguous buffer. This gives cache-friendly sequential access and batch deallocation.

**Target**: 30-40% improvement on join-heavy queries, 20% on scans.

### 2. Reduce Cloning in Joins

The hash join probes and merges like this:

```rust
for right_tuple in &right_tuples {
    let key = join_key(right_tuple, &right_slots);  // allocates Vec<HashableValue>
    if let Some(left_matches) = hash_table.get(&key) {
        for left_tuple in left_matches {
            results.push(merge_tuples(left_tuple, right_tuple));  // clones both
        }
    }
}
```

Improvements:
- **Inline join keys**: Hash the slot values directly instead of collecting into a Vec. Use a custom hasher that processes slots in-place.
- **Lazy merge**: Don't materialize merged tuples. Instead, store (left_idx, right_idx) pairs and resolve slots on demand during the next stage.
- **Build-side selection**: Always build the hash table on the smaller side. Currently the planner doesn't consider this.

**Target**: 40-50% improvement on Graph 2-Hop, 20-30% on star joins.

### 3. Columnar Cache Layout

The type cache stores `HashMap<EntityId, HashMap<String, Value>>` — a map of maps. For scan queries that iterate all entities, this is the worst possible layout: random pointer chasing through nested hash buckets.

Replace with column-oriented storage:

```rust
struct TypeCache {
    entity_ids: Vec<EntityId>,              // dense, sorted
    columns: HashMap<String, Vec<Value>>,   // one Vec per field, aligned with entity_ids
}
```

Scanning "all employees' names" becomes iterating a single contiguous `Vec<Value>` — sequential memory access, SIMD-friendly, no hash lookups. Entity lookup by ID uses binary search on the sorted `entity_ids` vec.

This also enables vectorized predicate evaluation: instead of checking `salary > 80000` per entity in a loop, compare an entire `&[Value]` column at once.

**Target**: 2-3x improvement on scan workloads, 1.5x on filtered queries.

### 4. Batch Clause Evaluation

Currently `evaluate_clause` is called once per input tuple in a nested loop join. For Graph 2-Hop with 400 engineers, we call it 400 times for the manager lookup.

Instead, batch all input bindings and evaluate the clause once:

```rust
// Current: 400 individual calls
for tuple in &left_results {
    let bound_eid = tuple.get(manager_slot);
    results.extend(evaluate_clause(cache, clause, Some(bound_eid), ...));
}

// Better: single batch call
let bound_eids: Vec<EntityId> = left_results.iter()
    .map(|t| t.get(manager_slot).as_ref())
    .collect();
let batch_results = evaluate_clause_batch(cache, clause, &bound_eids, ...);
```

With a columnar cache, the batch lookup becomes a single pass over the entity_ids column with a HashSet probe — no per-entity function call overhead.

**Target**: 2-3x improvement on Graph 2-Hop specifically.

### 5. Smarter Join Strategy Selection

The planner uses simple heuristics (exact=1, range=count/3, full=count) to estimate cardinality and pick join order. It doesn't consider:

- **Build-side selection** for hash joins (always builds right, should build smaller)
- **Semi-join reduction**: For Graph 2-Hop, the second clause only needs entity IDs from the first, not full tuples. A semi-join passes just the join key, reducing intermediate tuple traffic.
- **Index nested loop join**: When the inner side has an index on the join attribute (e.g., manager_id via AVET), use it instead of building a hash table. This is what PostgreSQL does for Graph 2-Hop and it's why it's fast.

Concrete improvement: detect when a join key matches an indexed attribute and use index-nested-loop instead of hash join. For Graph 2-Hop, this means 400 AVET lookups for manager_id instead of hashing all 2000 employees.

**Target**: 3-5x improvement on Graph 2-Hop, 1.5x on star joins.

### 6. Parallel Execution

Once per-query overhead is reduced (Phases 1-5), parallelism becomes worth the coordination cost:

- **Parallel hash join probe**: Partition probe tuples across threads with `rayon::par_chunks`
- **Parallel type cache loading**: Load multiple fields concurrently during `load_type_data`
- **Parallel independent clauses**: In a star join, evaluate independent arms concurrently

This is low priority until single-threaded performance is solid. Parallelism won't help Graph 2-Hop (pipeline dependency between hops) but will help fan-out and star joins.

**Target**: 1.5-2x on fan-out and star joins.

### 7. Volcano/Iterator Execution Model

Replace the materializing tree-walk with a pipelined iterator model. Each `PlanNode` becomes a stateful iterator with a `next() -> Option<Tuple>` method. Joins pull one row at a time from their children instead of collecting full result sets.

```rust
trait PlanIterator {
    fn next(&mut self) -> Result<Option<Tuple>, String>;
}

struct NestedLoopJoin {
    left: Box<dyn PlanIterator>,
    right: Box<dyn PlanIterator>,
    current_left: Option<Tuple>,
}

impl PlanIterator for NestedLoopJoin {
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        loop {
            if self.current_left.is_none() {
                self.current_left = self.left.next()?;
                if self.current_left.is_none() { return Ok(None); }
                self.right.reset(&self.current_left)?;
            }
            if let Some(right_row) = self.right.next()? {
                return Ok(Some(merge(&self.current_left, &right_row)));
            }
            self.current_left = None;  // exhausted right side, advance left
        }
    }
}
```

This eliminates the intermediate `Vec<Tuple>` allocations that currently dominate Graph 2-Hop. It also enables streaming results back to the client and short-circuiting for LIMIT queries.

The main challenge is hash joins — they inherently need to materialize at least the build side. But the probe side can still stream, and the build side can use an arena allocator to reduce per-tuple overhead.

**Target**: 2-3x on Graph 2-Hop and multi-way joins, enables LIMIT/streaming.

### 8. Bytecode/Compiled Expression Evaluation

Replace the interpreted `HashMap<String, Pattern>` field matching with a flat instruction sequence. Two levels of ambition:

**Level 1 — Slot-based evaluation**: At plan time, resolve field names to integer offsets into the type cache. Replace per-field HashMap lookups with array indexing.

```rust
// Current: HashMap lookup per field
let val = fields.get("department");  // hashes string, probes buckets

// Better: pre-resolved offset
let val = field_values[slot_offsets.department];  // direct array index
```

**Level 2 — Bytecode interpreter**: Compile predicates into a small instruction set (LOAD_SLOT, CONST, CMP_EQ, CMP_GT, AND, BIND_VAR). Evaluate with a tight loop over a `&[Instruction]` array instead of pattern-matching each `Pattern` variant.

**Level 3 — Closure generation**: Generate a specialized Rust closure at plan time that hard-codes the join structure, slot offsets, and comparisons. This lets the Rust compiler inline and optimize the entire query hot loop.

Level 1 is low-effort and composes well with columnar cache (item 3). Levels 2-3 are longer-term and pay off most on repeated queries.

**Target**: Level 1: 20-30% on scans. Level 2-3: additional 2x on repeated queries.

## Priority Order

| Priority | Change | Effort | Impact | Targets |
|---|---|---|---|---|
| 1 | Inline tuple slots | Small | Medium | All join queries |
| 2 | Reduce cloning in joins | Small | Medium | Graph 2-Hop, star joins |
| 3 | Columnar cache layout | Medium | High | Scans, filtered queries |
| 4 | Batch clause evaluation | Medium | High | Graph 2-Hop |
| 5 | Smarter join strategies | Medium | High | Graph 2-Hop, star joins |
| 6 | Parallel execution | Small | Medium | Fan-out, star joins |
| 7 | Volcano/iterator model | Large | High | Graph 2-Hop, multi-way joins |
| 8 | Bytecode/compiled expressions | Medium-Large | High | Scans, repeated queries |

Items 1-2 are localized changes that can be done independently. Item 3 requires reworking the cache module. Items 4-5 require changes to both planner and executor. Item 7 is a significant rewrite of the executor but addresses the root cause of our worst benchmark (Graph 2-Hop). Item 8 level 1 (slot-based offsets) is moderate effort and pairs well with item 3; levels 2-3 are longer-term.

## Target Performance

After implementing items 1-5:

| Workload | Current | Target |
|---|---|---|
| Graph 2-Hop | 11.68x | 2-3x |
| Selective Scan | 3.97x | 1.5x |
| Unfiltered Scan | 3.79x | 1.5-2x |
| Large Fan-Out | 3.37x | 1.5-2x |
| 4-Way Star Join | 1.60x | 1.0-1.2x |
| Mixed 4-thread | 4.57x | 2-3x |

The remaining gap after these changes would be structural: PostgreSQL's compiled C executor with decades of optimization vs. our interpreted Rust executor. Closing that last mile requires query compilation (item 7).
