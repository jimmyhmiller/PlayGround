# AI-facing memory analysis: snapshots, diffs, and findings

This spec turns memscope from a tool that produces *renderings for humans* (flame
graphs, Perfetto traces, the monitor TUI) into one that also produces *artifacts
for an AI*: ranked, source-located **findings**; **diffs** between heap states;
and a small **query** surface to drill in. Everything here is built on data
memscope already captures — typed allocations, full stacks with source
locations, lifetimes (alloc→free), realloc chains, retained sizes + dominators,
and `meta!` context. No new hot-path capture is required except one cheap marker
event for labeled checkpoints.

The design principle throughout: **a flame graph asks a human to visually
pattern-match a problem; an AI needs the conclusion stated as structured data,
plus a handle to pull the one detail that matters.** Recordings are 50 MB–346 MB;
no model reads that. Every output here is budget-bounded and references raw data
by handle rather than inlining it.

---

## Part A — Snapshots & diffs

### A.1 What exists today

- `Snapshot` (memscope-proto) is already self-contained: `live: Vec<LiveAlloc>`,
  `sites`, `types`, `total_live_bytes`, `taken_at_nanos`, `sample_scale`.
  `LiveAlloc = { addr, size, align, site, ts_nanos, thread }`.
- `recorder.snapshot()` reconstructs the live set off-thread and dumps it.
- `replay` / the flamegraph readers already reconstruct a live set by streaming
  the binary record format: `live: HashMap<addr, (size, site)>`, applying
  `A`/`R` (insert) and `D` (remove) events in timestamp order.
- The binary format (`recfmt.rs`) interleaves `TAG_SITE` (interned site → frames)
  and `TAG_EVENTS` (batched 34-byte `RawEvent`s) plus `TAG_KEY`/`TAG_META` for
  `meta!`.

What's missing: a notion of **named checkpoints** and **diffs between two heap
states**. Almost all real memory debugging is differential ("it grew between
request 1 and request 100 — what's the delta, and who retains it?"). We have the
stream to answer it; we just don't expose it.

### A.2 Labeled checkpoints (`mark`)

Add a semantic marker to the event stream so a state can be named at a meaningful
moment instead of by raw nanosecond.

**API (memscope facade):**

```rust
memscope::mark("after_warmup");        // checkpoint the current instant
memscope::mark(format!("req:{}", id)); // dynamic labels fine (interned)
```

**Wire format:** reuse the `meta!` machinery — a marker is just a label string
interned like a meta key, emitted as one event on the per-thread ring.

- New `EventKind::Mark` (proto). Carries the interned label id in the `site`
  slot (same trick `MetaEnter`/`MetaExit` already use for the meta-context id)
  and the timestamp in `ts_nanos`.
- New record tag `TAG_MARK = b'M'` in `recfmt.rs`, written like `TAG_KEY`:
  `[TAG_MARK][u32 label_id][u32 len][utf8 label]` once per distinct label, then
  the `Mark` event references it by id. (Labels are few; interning keeps it tiny.)
- Cost: one ring push, identical to a `meta!` enter. Nothing on the alloc hot
  path changes.

A `mark` does **not** snapshot eagerly. It drops a timestamped fencepost; the
live set *at that fencepost* is reconstructable from the stream by replaying up
to that `ts_nanos`. This keeps `mark` free and lets you diff any two marks
posthoc from a single recording. (Live agent: `ClientMsg::Snapshot { label }`
can still force an eager `Snapshot` dump when there's no recording file.)

### A.3 Reconstructing a state at a mark

A reusable reader (new `memscope-replay` crate, or a module in the CLI promoted
to a lib) exposes:

```rust
/// Replays a .mscope/.jsonl recording, materializing the live set at each mark.
pub struct Timeline {
    pub marks: Vec<Mark>,              // { label, ts_nanos, seq }
    // internal: site table, meta table, key table
}
pub struct LiveState {
    pub at: Mark,
    pub allocs: Vec<LiveAlloc>,        // exactly what Snapshot.live holds
    pub sites: SiteTable,              // id -> frames + resolved type/shape
    pub meta_of: HashMap<u64 /*addr*/, MetaCtx>, // correlated meta! context
    pub total_live_bytes: u64,
}

impl Timeline {
    pub fn open(path: &Path) -> Result<Timeline>;
    pub fn state_at(&self, label: &str) -> Result<LiveState>;   // replay ≤ mark ts
    pub fn state_at_end(&self) -> Result<LiveState>;
}
```

`state_at` is the existing replay loop (`live.insert` on A/R, `live.remove` on D)
stopped at the mark's `ts_nanos`, plus meta-context correlation (reuse the
flamegraph reader's per-thread meta-scope stack). A reconstructed `LiveState` is
identical in shape to a `Snapshot`, so the graph crate and every existing
consumer accept it unchanged.

### A.4 `memscope diff A B` — the killer feature

Set-diff two heap states by `(type, site)` identity. This isolates a leak to a
*phase* and names the retainer — the single most actionable artifact for an AI.

```sh
memscope diff rec.mscope after_warmup end_of_request --out diff.json
memscope diff a.snapshot b.snapshot --out diff.json        # two saved dumps
memscope diff --sock <P> --baseline after_warmup           # live: now vs a mark
```

**Grouping key:** `(resolved_type, site_id)`. Site id (not just type) so two
call sites allocating the same type don't merge — the AI needs the *site* to
locate the fix.

**Output schema (`DiffReport`):**

```jsonc
{
  "v": 1,
  "a": { "label": "after_warmup",    "ts_nanos": 1200000000, "live_bytes": 4200000 },
  "b": { "label": "end_of_request",  "ts_nanos": 9600000000, "live_bytes": 6300000 },
  "window_ms": 8400,
  "sample_scale": 1.0,
  "net_retained_delta": 2100000,
  "grew": [
    {
      "type": "serve::Session",
      "shape": "Boxed",
      "site": "cache::insert (cache.rs:88)",
      "site_id": 41,
      "delta_count": 4200,            // live in B not in A
      "delta_bytes": 2100000,
      "born_in_window": 4200,         // allocated after A.ts
      "freed_in_window": 0,           // ← 0 freed = monotonic; strong leak signal
      "still_live": 4200,
      "stack_handle": "site:41"       // pull full stack via `query`
    }
  ],
  "shrank": [
    { "type": "alloc::vec::Vec<u8>", "site": "...", "delta_count": -300, "delta_bytes": -38400 }
  ],
  "unchanged_types": 57,             // summarized, not enumerated
  "truncated": { "grew": false, "shrank": false }   // honest caps, never silent
}
```

**Computation** (all from data on hand):

- `delta_count`/`delta_bytes`: per-`(type,site)` `B.live − A.live`.
- `born_in_window`: count of `A` events with `ts_nanos > A.ts && ≤ B.ts` for the
  group (from the stream between the two marks).
- `freed_in_window`: count of `D` events in the window whose addr was live at A
  *or* born in window. `freed_in_window == 0 && delta_count > 0` is the canonical
  "born and never died" leak fingerprint.
- `net_retained_delta`: `B.total_live_bytes − A.total_live_bytes`.

Rank `grew` by `delta_bytes` desc; cap to top-N (default 50) and set
`truncated.grew` honestly when cut — never silently drop.

### A.5 Heap-graph diff (structural root cause)

`diff --graph` additionally builds the reference graph (memscope-graph) for both
states and reports **which retention edge appeared**:

```jsonc
"new_retention": [
  { "dominator": "serve::World.cache (HashMap<u64, Session>)",
    "now_dominates": "serve::Session",
    "retained_delta_bytes": 2100000,
    "edge_offset": 24 }                 // byte offset of the pointer field
]
```

This is the MAT "who started keeping this alive" answer, mapped to a concrete
field+offset an AI can trace to source. It reuses `HeapGraph` (nodes, edges,
`idom`, `retained_size`) verbatim — the diff is over two graphs' dominator trees.

### A.6 Snapshot timeline ("git log for the heap")

`memscope marks rec.mscope` lists checkpoints with a coarse `live_bytes`-by-type
series so the AI can see the *shape* of growth in a few hundred tokens and pick
which two marks to `diff` — then `diff` two "commits."

```jsonc
{ "marks": [
  { "label": "after_warmup",   "ts_ms": 1200, "live_bytes": 4200000,
    "top": [["serve::Particle", 1300000], ["serve::Session", 520000]] },
  { "label": "end_of_request", "ts_ms": 9600, "live_bytes": 6300000,
    "top": [["serve::Session", 2620000], ["serve::Particle", 1300000]] }
]}
```

---

## Part B — `analyze`: from data to findings

### B.1 Shape

```sh
memscope analyze rec.mscope --out findings.json     # ranked findings, tens of KB
memscope analyze --sock <P>                          # live process
memscope analyze rec.mscope --only leak,churn        # subset of detectors
memscope analyze rec.mscope --budget 30000           # token budget for the report
```

`analyze` runs a set of **detectors** over the reconstructed stream + final live
state (+ optionally the graph) and emits a ranked `Findings` document. Each
finding is self-describing: evidence, a hypothesis, source locations, and a
**fix class** an agent can act on.

### B.2 Finding schema

```jsonc
{
  "v": 1,
  "target": { "exe": "/path/serve", "pid": 1234, "duration_ms": 9600,
              "sample_scale": 1.0, "mode": "full" },
  "findings": [
    {
      "id": "leak:site41:Session",
      "detector": "monotonic-growth",
      "severity": 0.92,                 // 0..1, used for ranking
      "confidence": 0.8,
      "title": "Session leak: 4,200 live, never freed, retained by World.cache",
      "type": "serve::Session",
      "shape": "Boxed",
      "site": "cache::insert (cache.rs:88)",
      "site_id": 41,
      "evidence": {
        "live_count": 4200, "live_bytes": 2100000,
        "freed": 0, "allocated": 4200,
        "growth": "+18%/mark over 5 marks",
        "retained_size": 2100000,
        "dominated_by": "serve::World.cache",
        "median_lifetime_ms": null      // still alive
      },
      "hypothesis": "Entries inserted per request into World.cache are never evicted; the HashMap retains every Session for the process lifetime.",
      "fix": {
        "class": "unbounded-cache",     // see B.4 taxonomy
        "locations": ["cache.rs:88", "world.rs:integration site of World.cache"],
        "suggestion": "Bound the cache (LRU/TTL) or drain entries when the request completes."
      },
      "handles": { "stack": "site:41", "samples": "type:serve::Session" }
    }
  ],
  "summary": { "total_findings": 12, "shown": 12, "live_bytes": 6300000 },
  "truncated": false
}
```

`severity` drives ranking; `confidence` is reported separately so the agent can
weigh a high-impact-but-uncertain finding against a small-but-certain one.
`handles` are opaque tokens for `query` (Part C) — the AI pulls the full stack or
a lifetime histogram only for the findings it decides to act on.

### B.3 Detector set (v1)

All computable from the event stream + final state; none need new capture.
Each detector is a pure pass with an explicit formula and a ranking score.

| detector | fingerprint | score ∝ | fix class |
|---|---|---|---|
| **monotonic-growth** (leak) | per-`(type,site)` live set grows across marks; `freed_in_window ≈ 0` | live_bytes × growth_slope | leak / unbounded-cache |
| **churn-storm** | high `total_bytes`, near-zero `live_bytes` (alloc+free in a tight window) | `total_bytes / max(live_bytes,1)`, weighted by total_bytes | hoist-alloc / reuse-buffer / arena |
| **realloc-thrash** | N `R` events on one address chain (Vec growing 1-at-a-time) | Σ reallocated bytes; `realloc_count` per final object | `with_capacity` / `reserve` |
| **short-lived-box** | `Box<T>` with median lifetime < T_ms and high count | count × (1/median_lifetime) | stack / `SmallVec` / arena |
| **oversized-alloc** | allocation `size ≫ size_of::<T>()` (over-reserved capacity) | wasted bytes (size − used) | right-size capacity |
| **retention-surprise** | small object with large `retained_size` (dominator tree) | retained_size / self_size | restructure ownership |
| **transient-peak** | `live_bytes` peak ≫ end value within one mark window | peak − trough | streaming / chunking |
| **fragmentation-hint** | many small live allocs of one type, high count, scattered addrs | count of sub-cacheline allocs | pool / `SmallVec` inline |

Detector inputs available today:

- **Lifetime**: pair each `D` with its `A` by addr → `lifetime = D.ts − A.ts`.
  Histogram per `(type,site)`. (Replay already tracks addr→alloc; just record
  the delta on free.)
- **Realloc chains**: `R` events carry the address; the readers already treat
  `R` as alloc. Track per-final-object realloc count + grown bytes.
- **Retained size / dominator**: `HeapGraph.nodes[*].retained_size` and `idom`.
- **Growth slope**: live set per `(type,site)` sampled at each mark (Part A.6).
- **Used vs allocated**: `oversized` needs the *used* portion. For `Vec`/`String`
  we know element size and capacity from the allocation size + shape; flag when
  successive growth implies the final capacity dwarfs the high-water content.
  (Honest limitation: we see capacity bytes, not len — flagged as a *hint*, and
  the finding says so.)

### B.4 Fix-class taxonomy

A closed vocabulary so the agent maps a finding to a concrete code transform:

`leak` · `unbounded-cache` · `hoist-alloc` (move out of hot loop) ·
`reuse-buffer` (clear + reuse vs realloc) · `with_capacity` · `reserve` ·
`box-to-stack` · `smallvec` · `arena` · `pool` · `cow` · `intern` ·
`right-size-capacity` · `streaming` · `restructure-ownership`.

Each finding's `fix.suggestion` is prose; `fix.class` is from this set so
downstream tooling (or an agent's own logic) can branch on it deterministically.

### B.5 Honesty rules (mirrors the global no-silent-stub rule)

- Every cap is reported (`truncated`, `shown` vs `total_findings`). Never drop a
  finding silently.
- Sampled mode: every count is scaled by `sample_scale` and the finding carries
  `"sampled": true` so the agent discounts confidence.
- A detector that *can't* compute its input (e.g. no marks → no growth slope)
  emits nothing for that axis rather than guessing — and `analyze` notes the
  missing capability in `summary.skipped_detectors` with the reason.

---

## Part C — `query`: bounded drill-down

Findings carry `handles`; `query` resolves a handle to exactly the detail an
agent asks for, bounded by `--top`.

```sh
memscope query rec.mscope --site 41 --field stack            # full call stack, app frames
memscope query rec.mscope --type serve::Session --field lifetimes   # histogram
memscope query rec.mscope --type serve::Session --field stacks --top 5
memscope query rec.mscope --addr 0x... --field paths         # paths-to-roots (reuses `paths`)
memscope query rec.mscope --site 41 --field reallocs         # realloc chain detail
```

`--field stack` reuses the `--no-std` frame collapser so the AI sees *its own
code*, with `file:line` in each frame. This is the agent loop: read `findings`
→ `query` the one site that matters → propose a patch.

---

## Part D — Surfacing to an agent (MCP)

memscope already speaks newline-JSON over a Unix socket. A thin MCP server wraps
the three verbs so a Claude session can debug a live or recorded heap directly:

| MCP tool | maps to |
|---|---|
| `memscope_analyze(source, only?, budget?)` | `analyze` → findings JSON |
| `memscope_diff(source, a, b, graph?)` | `diff` → diff JSON |
| `memscope_query(source, handle, field, top?)` | `query` → bounded detail |
| `memscope_marks(source)` | `marks` → heap timeline |

`source` is a recording path or a live socket. The agent's natural workflow:
`marks` to see the shape → `diff` two marks to localize a phase → `analyze` for
ranked findings → `query` for the one stack → edit code. All outputs are
token-budgeted JSON, so the loop stays inside a context window.

---

## Build plan (phased, each independently shippable)

**Phase 1 — `diff` (highest leverage, smallest surface). ✅ SHIPPED.**
1. ✅ Promoted the CLI's binary/JSON replay reader into a `memscope-replay` lib
   (`Recording` parser + `Timeline` / `LiveState`); the CLI's flamegraph/perfetto
   commands now share the one canonical parser.
2. ✅ `EventKind::Mark` + `TAG_MARK` + `memscope::mark()`; emitted by the file
   recorder (binary + JSON), read back self-contained.
3. ✅ `memscope diff <src> <A> <B> [--json]` — `(type,site)` set-diff with
   `born/freed_in_window` attribution (Dealloc carries no site, so frees are
   attributed by seeding `addr→site` from state A and tracking through the
   window). `start`/`end` resolve to the stream ends; rows carry the exact app
   source line; top-N capped with honest `truncated`.
4. ✅ `memscope marks <src> [--json]` — checkpoint list + heap size + top types.

Not yet done from the original Phase-1 sketch (deferred, non-blocking): the full
`DiffReport` schema includes a per-`(type,site)` `stack_handle`; today the row
carries the source location inline instead. Per-allocation `meta!` correlation
onto `LiveState` (the spec's `meta_of`) is also deferred — `diff` groups by
`(type,site)`, not yet by meta key.

**Phase 3 — `analyze`. ✅ SHIPPED** (done before Phase 2 — it's pure stream
analysis over a recording, whereas a heap *graph* needs live memory to walk
pointer fields, so `diff --graph` is inherently a live-only feature).
- ✅ `memscope-replay::analyze` — one-pass `SiteStats` (alloc/free/realloc counts,
  bytes, lifetimes via addr-paired A/D, final live set) feeding detectors
  `monotonic-growth` (leak / unbounded-cache), `churn-storm`, `realloc-thrash`,
  `short-lived-box`. Ranked by severity, each with a closed-vocabulary `fix_class`.
- ✅ Findings are **merged by application boundary location + type** (loop-unrolled
  call sites sharing a source line collapse into one), and sites whose allocation
  originates *inside* the profiler/DWARF machinery are **dropped** as measurement
  overhead (new `memscope-replay::frames` module: `is_std_frame` / `is_profiler_*`
  / `boundary_frame`, now the single home for frame classification — the CLI's
  flamegraph `--no-std` shares it).
- ✅ `memscope analyze <FILE> [--json] [--top N]` — text + AI-ready JSON with
  source-located, ranked findings; honest `truncated` + `shown`/`total`.
- Deferred (need the heap graph / type sizes, not in a recording):
  `retention-surprise`, `oversized-alloc`. Marks-aware growth-slope refinement of
  the leak detector is also a follow-up (today it keys on whole-run freed-fraction).

**Phase 2 — `diff --graph` (LIVE only).** Build `HeapGraph` for two in-memory
states; dominator-tree diff → `new_retention`. Reuses memscope-graph unchanged.
Deferred: requires reading live process memory (a recording has no heap contents),
so it attaches to the live-agent path, not the posthoc recording flow.

**Phase 4 — `query` + MCP. ✅ SHIPPED.**
- ✅ `memscope query <FILE> (--site N | --type T) [--field stack|lifetimes|stats|sites] [--json]`
  — bounded drill-down over `memscope-replay::site_stats`: the full call stack
  (app frames marked), a freed-allocation **lifetime histogram**, aggregate
  **stats**, or every call **site** of a type. `--type` aggregates across sites.
- ✅ **`memscope-mcp`** — a minimal MCP stdio JSON-RPC server (no SDK; handles
  `initialize` / `tools/list` / `tools/call`) exposing `memscope_marks` /
  `memscope_diff` / `memscope_analyze` / `memscope_query`. It's a thin wrapper:
  each tool shells out to the `memscope` CLI with `--json` (located via
  `$MEMSCOPE_BIN`, a sibling binary, or `PATH`) and returns the structured result
  as MCP text content, with `isError` on failure. So a Claude session runs the
  loop `analyze` → `query`/`diff` → edit → re-`diff` directly, all token-bounded.

All four phases that run off a **recording** (1, 3, 4) are shipped; the live-only
graph diff (Phase 2) remains the one deferred piece.

---

## What this deliberately does not do

- It does not replace the flame graph / Perfetto / monitor views — those stay
  the *human* surface. `analyze`/`diff`/`query` are the *agent* surface over the
  same captured data.
- It does not add hot-path cost beyond one `mark` marker (≈ a `meta!` enter).
- It does not infer fixes it can't ground: a finding names a fix *class* and
  *locations*; applying the patch is the agent's job, verified by re-running
  `diff` before/after (the natural regression check — did the leak's
  `delta_count` go to zero?).
