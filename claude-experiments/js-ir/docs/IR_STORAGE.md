# JSIR Storage — a data-oriented layout

> **Status — DONE for the supported scope (build + print).** By decision, the SoA
> layout only needs to *build* the IR and *print* it byte-exact; the analysis /
> transform consumers (DCE, tree-shaking, dataflow) and the React/JSLIR dialect
> are explicitly **out of scope** and stay on the AoS `Op`. Implemented:
> - `crates/jsir-ir/src/traits.rs` — `IrRead` / `IrBuild`.
> - `soa.rs` — the columnar `Module` + `Module::from_op` (build by lowering).
> - `print.rs` `*_via` fns — one generic printer, shared with the AoS path.
> - `build.rs` — `rebuild`, the worked example of constructing a `Module`
>   through the full `IrBuild` API.
> - `jsir-swc::source_to_module` — public "source → SoA IR" entry point.
>
> Oracle (all green): `Module::from_op(&op).print()`, `rebuild(&module).print()`,
> and `source_to_module(src).print()` are byte-exact vs the AoS printer on all 46
> conversion fixtures + both dialects. The AoS `Op` remains the construction path;
> SoA is reached by lowering. Optional future polish (not needed for scope):
> opcode→enum (currently interned u32), `AttrKey` interning, trivia→`SecondaryMap`.


## Performance vs oxc and swc (measured)

A no-AST front end (`crates/jsir-parse`) parses a JS subset **straight to the SoA
`Module`** (text → SIMD tokens → JSIR, no syntax tree), then DCE runs on it.
Benchmarks (`jsir-parse/examples/bench_parse.rs`, `bench_dce.rs`;
`simd-lang/aot-bench/pipeline` for the real AOT `.simd` kernel). M-series macOS,
~stable medians.

**Front end — text → structure (20k stmts / ~0.6 MiB):**

| | time | speed | output |
|---|---:|---:|---|
| **ours: SIMD → RPN tape** | **~2.7 ms** | **~220 MiB/s** | flat AST-equivalent tape |
| oxc: lex+parse → AST | 3.3 ms | 177 MiB/s | bare AST |
| oxc: parse + semantic | 5.9 ms | 99 MiB/s | rich (scopes/symbols) |
| ours: SIMD → JSIR Module | 7.5 ms | 78 MiB/s | rich IR (SSA + operands + attrs) |
| swc: lex+parse → AST | 9.0 ms | 65 MiB/s | bare AST |

**The flat RPN tape (`parse_to_tape`) beats oxc bare AST by ~1.23×** (2.2× its
`parse+semantic`) — *the* fast path. Each node is a 12-byte
`(kind,flags,nargs,start,end)` record in post-order; `nargs` + spans reconstruct
the tree, with no interning/ValueIds/regions. **Building the structured `Module`
instead doubles+ parse time** (78 vs 220 MiB/s) — that build is the entire cost,
so JSIR's `Module` is constructed from the tape *only when the structured IR is
actually needed*, not on the hot parse path. The columnar-`Module` front end was
itself optimized 21 → ~85 MiB/s (~4×: arena strings, parser-mode lexing, pull
parser, inline operands, key dedup, source-borrowed attrs, `#[inline]`), but the
tape is what reaches oxc parity. Our SIMD lexer (real AOT `.simd` kernel) is
faster than oxc's; stage-1 is ~1.3% of the pipeline. (simd-lang's `tape.rs` is
the parallel canonical implementation with a much larger grammar, ~2.3× oxc.)

**DCE — text → dead-code-eliminated (10k stmts, half dead), DCE-only each:**

| | time | speed |
|---|---:|---:|
| **ours: JSIR + in-place dce** | **4.4 ms** | **77 MiB/s** |
| oxc: `Minifier::dce` | 4.85 ms | 69 MiB/s |
| swc: `simplify::dce` | 7.7 ms | 43 MiB/s |
| ours: dce (copying rebuild) | 9.3 ms | 36 MiB/s |

In-place DCE (`Module::compact_block_ops` + `build::dce_in_place`) makes us
fastest — the copying rebuild was the whole gap. Caveat: our DCE is a lighter
analysis (read-scan + drop unread top-level `var`s, no scope resolution) than
oxc/swc's scope-correct passes, so part of the win is doing less; in-place vs
rebuild is the other part. Output verified identical to the copying DCE.

---

How we represent and store our IR. This is a design doc for moving `jsir-ir`
from its current array-of-structs (AoS) `Op` tree to a struct-of-arrays (SoA),
index-addressed store — without changing the IR's *semantics* or the byte-exact
textual printer.

Sizes for OXC comparisons are ground-truth from `oxc` rev `7b0380d`.

---

## 1. What the IR is (this does not change)

JSIR is an **MLIR-generic operation model**. The whole program is a tree of
*operations*; an op has:

- a **name** (e.g. `jsir.binary_expression`, `jslir.cond_br`),
- ordered **operands** (SSA value uses),
- an **attribute dictionary** (alphabetically printed),
- nested **regions**, each a list of **blocks**, each a list of ops,
- 0+ **results** (jsir ops have ≤1),
- on terminators only, **successor** edges (target block + block-argument values),
- carried **trivia** (loc / offsets / comments / scope / symbols) used by
  `hir2ast` to rebuild AST nodes; the printer ignores it.

Two dialects ride on this one core:

- **AST dialect** — single block per region, no terminators, no block args.
  This is what `ast2hir` emits and what the byte-exact oracle checks.
- **JSLIR dialect** — CFG/SSA: multiple blocks, block arguments (= phis),
  terminators with successors. Dialect-specific analysis state hangs off ops as
  **opaque attributes**.

**Hard invariants the storage must preserve:**

1. The textual printer is **byte-exact** vs upstream MLIR's generic format,
   including its SSA value numbering: `%N` numbered region-scoped, block-level
   ops first, then descending into child regions, the counter scoped per region
   so sibling regions reuse numbers.
2. `hir2ast` must reconstruct each AST node's base fields, so **trivia must
   survive** verbatim — it just doesn't have to be *fast* or *inline*.
3. Dialects must be able to attach arbitrary typed state without the core
   understanding it (the opaque-attr escape hatch).

The storage rewrite changes *how the bytes are laid out in memory*, nothing
about what prints or round-trips.

---

## 2. Current layout and why it hurts

```rust
pub struct Op {
    pub name: String,                 // 24B + 1 heap alloc PER OP
    pub operands: Vec<ValueId>,       // 24B + heap
    pub attrs: Vec<(String, Attr)>,   // 24B + heap; each key is a String alloc
    pub regions: Vec<Region>,         // 24B + heap → Vec<Block> → Vec<Op>  (the tree)
    pub results: Vec<ValueId>,        // 24B + heap; but jsir ops have ≤1 result
    pub successors: Vec<Successor>,   // 24B + heap; empty for the entire AST dialect
    pub trivia: Option<Trivia>,       // ~150B inline; comments/symbols; printer IGNORES it
    pub node_id: Option<u32>,
}
// Region { blocks: Vec<Block> }   Block { id, args: Vec<ValueId>, ops: Vec<Op> }
```

`sizeof(Op)` ≈ 300 B and building one chases **6+ heap allocations**. The
problems, in order of severity:

- **`name: String`** — a heap allocation per op to store one of ~150 fixed
  opcode strings.
- **The tree** — `Vec<Region> → Vec<Block> → Vec<Op>` is pointer-chased; every
  traversal (printer, every analysis pass) hops through owned `Vec`s.
- **`trivia` inline** — the single biggest field, sitting in the hot cache line
  of every op, and the printer never reads it.
- **`results` / `successors` Vecs** — paid on every op even though results are
  ≤1 and successors are empty for the entire AST dialect.

---

## 3. The DOD layout: a columnar op store

Stop storing ops as a tree of owned structs. Store the module as a handful of
parallel arrays (columns) indexed by a dense **`OpId`**, with all structure
expressed as **integer ranges into shared pools**, not nested `Vec`s.

```rust
/// 4-byte op identity. NonMax so `Option<OpId>` is also 4 bytes (niche), exactly
/// like OXC's `NodeId`. This id IS the identity — there is no pointer.
pub struct OpId(NonMaxU32);

/// A half-open slice into a pool: 8 bytes, no heap.
#[derive(Clone, Copy)]
pub struct Range32 { start: u32, len: u32 }

pub struct Module<'src> {
    // ── HOT columns (touched by the printer and every pass), indexed by OpId ──
    kind:     Vec<OpKind>,    // u16 interned opcode, replaces `name: String`   (2B)
    operands: Vec<Range32>,   // slice into `operand_pool`                       (8B)
    regions:  Vec<Range32>,   // slice into `region_table` (len 0 for leaf ops)  (8B)
    // result is IMPLICIT: the result value of op i is ValueId(i). No results column.

    // ── shared flat pools (append-only, amortized growth) ──
    operand_pool: Vec<ValueId>,            // every op's operands, contiguous
    attr_pool:    Vec<(AttrKey, AttrVal)>, // AttrKey = u8 enum, NOT a String
    attr_index:   Vec<Range32>,            // op i's attrs = attr_pool[attr_index[i]]

    // ── region / block tables (only ops with regions touch these) ──
    region_table: Vec<Range32>,  // a region → range of BlockId in block_table
    block_table:  Vec<Block>,    // Block { id: BlockId, args: Range32, ops: Range32 }
    block_ops:    Vec<OpId>,     // the op ordering within blocks (printer order)
    succ_pool:    Vec<Successor> // terminator successors, ranged per terminating op

    // ── COLD columns (never in the hot loop), same OpId index space ──
    loc:      Vec<PackedLoc>,                 // (start:u32, end:u32) byte offsets
    trivia:   SecondaryMap<OpId, TriviaId>,   // ONLY ops that carry comments/symbols
    node_id:  Vec<u32>,                       // origin id for IR→IR mapping

    // ── strings ──
    src: &'src str,        // identifier/literal raws are (u32,u32) ranges into here
    interner: Interner,    // only SYNTHESIZED names (introduced by passes)
}

pub enum AttrVal {              // the typed attribute payloads, unboxed where small
    Str(Range32),               // slice of `src` or interner — no String
    Bool(bool), F64(f64), I64(i64),
    Array(Range32),             // into a side attr-array pool
    Opaque(OpaqueId),           // dialect payload, kept in a side Vec<Arc<dyn ..>>
    // … the structured #jsir<…> attrs become small structs in side pools …
}
```

Hot per-op footprint: `kind(2) + operands(8) + regions(8) = 18 B`, **zero
per-op allocation** — everything lands in a few growable pools.

### Field-by-field mapping from today's `Op`

| current field | becomes |
|---|---|
| `name: String` | `kind: OpKind` (u16); printer holds `static OP_NAMES: [&str; N]` |
| `operands: Vec<ValueId>` | `Range32` into `operand_pool` |
| `attrs: Vec<(String, Attr)>` | `Range32` into `attr_pool`, keys are `AttrKey` u8 |
| `regions: Vec<Region>` | `Range32` into `region_table` |
| `results: Vec<ValueId>` | **gone** — `ValueId(i) == OpId(i)` |
| `successors: Vec<Successor>` | `Range32` into `succ_pool`, empty range for non-terminators |
| `trivia: Option<Trivia>` | `loc` cold column + `SecondaryMap` for the heavy part |
| `node_id: Option<u32>` | `node_id` cold column |
| `Region { blocks }` | range in `region_table` → `block_table` |
| `Block { id, args, ops }` | `Block { id, args: Range32, ops: Range32 }` over `block_ops` |

### The four moves that do the work

1. **Opcode interning.** `name: String` → `OpKind` (u16). ~150 opcode strings →
   0 allocations; opcode matching in passes is a jump table, not a `str` compare.
   The printer reverses it through a static name table.
2. **Implicit value numbering.** jsir ops have ≤1 result and the printer numbers
   `%N` in op-emission order, so make **`ValueId(i) == OpId(i)`** and delete the
   `results` column. A use names the defining op's id directly; def-before-use
   becomes a structural invariant of append order.
3. **Ranges replace nested Vecs.** Operands, attrs, regions, blocks, successors
   are all `Range32` slices into one flat pool each. "All operands of all ops" is
   now a single contiguous `Vec<ValueId>` scan — exactly what dataflow wants.
4. **Hot/cold split.** Byte-offset `loc` moves to a parallel cold column (line/col
   computed lazily from a line-start table). The heavy `Trivia` (comments,
   symbols, scope) moves to a `SecondaryMap` populated *only* for ops that have
   it. The printer's hot path and every pass stop carrying ~150 B of dead weight.

---

## 4. Printing and numbering over columns

The byte-exact printer (`print.rs`) is unaffected in output; only its traversal
changes from "recurse the `Vec<Region>` tree" to "walk `region_table` /
`block_table` / `block_ops` by index." The MLIR value-numbering algorithm maps
cleanly:

- For each region, number block-level results first, then descend into child
  regions, counter scoped per region. With implicit numbering (`ValueId ==
  OpId`) this is a pass that assigns the *printed* `%N` from the op's position;
  the underlying `OpId` stays stable and dense.
- Block labels (`^bbN`), block args (`^bb3(%3: !jsir.any)`), and terminator
  successor lists read from `block_table` / `succ_pool` exactly as before.

`hir2ast` reads the cold `loc` column + `trivia` `SecondaryMap` to rebuild each
node's base fields. Nothing about reconstruction needs trivia to be inline.

---

## 5. How passes use it

The whole point: downstream passes (SSA, dataflow, DCE, constprop) are the real
consumers, and they want index-addressable instructions, not a tree.

- **DCE / liveness:** a backward scan over `operand_pool` with a `live: BitVec`
  indexed by `OpId`. No traversal — just array sweeps.
- **Constprop:** a `value: Vec<Option<Const>>` lattice column indexed by `OpId`,
  scanned forward; results read straight off the column.
- **SSA / phis (JSLIR):** block args and successors are already index ranges —
  the form the SSA builder wants natively.
- **IR→IR rewrites:** append new ops to the columns and repoint `Range32`s; the
  stable `OpId` means a pass can hold an id across edits (the reason `node_id`
  exists today, now generalized).

A pass touching only `kind` + `operands` reads ~10 B/op **contiguously**, versus
chasing owned `Vec`s through a 300 B struct.

---

## 6. Why this is the right target (and the OXC proof point)

OXC bolted a `node_id: Cell<NodeId>` (a `NonMaxU32`) onto every AST node, and its
semantic layer (`oxc_semantic/src/node/nodes.rs`) stores analysis as **parallel
columns keyed by that id**:

```rust
pub struct AstNodes<'a> {
    nodes:      IndexVec<NodeId, AstNode<'a>>,
    parent_ids: IndexVec<NodeId, NodeId>,   // structure as integer links, SoA
    flags:      IndexVec<NodeId, NodeFlags>,
    cfg_ids:    IndexVec<NodeId, BlockNodeId>,
}
```

That is exactly the layout above. OXC built it as a **second** structure on top
of a pointer-tree AST, because *its* consumers (linters, transformers,
formatters) want an ergonomically mutable tree, so the tree has to stay a tree
and the columnar store is derived from it.

**Our consumers are the opposite** — everything downstream of JSIR is the
equivalent of `oxc_semantic`. So JSIR should be *born* columnar and never pay
for the pointer tree at all. OXC validates the destination; it just can't adopt
it for its own AST.

Head-to-head on the layout (the `1 + 2` example, 3 nodes/ops):

| | OXC AST | ideal JSIR |
|---|---|---|
| identity / handle | 16 B enum (tag+`Box` ptr) **+** 4 B bolted `NodeId` | 4 B `OpId`, *is* the identity |
| node payload | 48 B boxed `BinaryExpression` | ~10 B in columns (kind + operand range) |
| structure | pointer tree (`Box` graph) | `Range32` indices into pools |
| strings | `&src[..]` / `Ident` w/ packed hash | `(u32,u32)` ranges into `src` |
| spans | 8 B `u32` offsets, line/col lazy | same: cold `PackedLoc` column |
| allocations | 1 bump/node, drop arena O(1) | 0/op, ~8 column Vecs total, drop O(1) |
| pass working set | 16 B handle → deref 48 B node | ~10 B contiguous, no deref |

We tie OXC on allocation discipline and come out ahead on handle size, hot
footprint, and cache behavior — because we keep only the columnar half OXC is
forced to duplicate.

---

## 7. Migration plan (no oracle regressions)

Stand this up *behind* the existing byte-exact printer so the oracle never moves:

1. Add `OpKind` (+ `static OP_NAMES`) and `AttrKey`; keep the existing `Attr`
   variants, just key them by enum.
2. Build the columnar `Module` as a new module (`jsir-ir` SoA storage) with the
   same public construction surface the converter uses.
3. Reimplement `print.rs` traversal over columns; gate it behind the oracle
   diff against all 44 golden fixtures — output must be byte-identical.
4. Port `hir2ast` to read the cold `loc`/`trivia` columns.
5. Migrate one pass (DCE is the cleanest) to the columnar form and confirm
   parity, then the rest.
6. Benchmark build + DCE against the current `Op` to quantify the win.

The semantics, the printed bytes, and the round-trip are invariant throughout;
only the in-memory representation changes.

---

## 8. One-line summary

Store the IR as a few parallel columns keyed by a 4-byte `OpId`, with all
structure as integer ranges into shared pools, opcodes interned to a `u16`,
value numbering made implicit (`ValueId == OpId`), and locations/trivia split
into cold columns — the columnar analysis layout OXC only reaches as a derived
side-store, adopted as JSIR's native form.
