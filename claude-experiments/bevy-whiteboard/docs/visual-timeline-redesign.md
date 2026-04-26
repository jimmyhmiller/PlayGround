# Visual Timeline — current shape and where it should go

Snapshot of analysis from a perf pass on `examples/life_30x30_random.whiteboard`. The work itself was deferred — captured here so future sessions don't re-derive.

## Context

The visual timeline lives in `crates/flow-bevy/src/visual.rs`. It's the layer between the discrete-event sim and the packet-cloud renderer. Sim emits `PacketEmitted` events; the timeline turns each one into a `VisualPacket` with `(emit_real, arrive_real)` mapped through the F12 formalism (real-time anchored, k-scaled, causally clamped). The renderer (`update_packet_cloud` in `packet_cloud.rs`) walks the timeline's `Vec<VisualPacket>` every frame and packs visible packets into a GPU storage buffer.

This doc covers what the timeline does, why its current shape costs more than it should, and what an optimal replacement would look like. It does **not** change any code.

## The three jobs hidden in one struct

`VisualTimeline` mixes three responsibilities with very different lifecycle requirements:

1. **Ingest mapping** — sim event → `VisualPacket`. Pure, stateless transformation per call.
2. **Causal-trigger memory** — `HashMap<NodeId, NodeArrivals>`. Records per-node arrival history so the next emit *from* that node can clamp its `emit_real` to "after the trigger arrived." Without this, a router's response visually starts the same frame the request arrived. This is the only part of the timeline that genuinely needs persistent state across ingests.
3. **Renderable list** — `Vec<VisualPacket>`. Single source of truth that `update_packet_cloud` walks every frame.

| Job | Storage | Pruning | Live consumer |
|---|---|---|---|
| ingest | (none — pure compute) | — | — |
| causal triggers | parallel `Vec<u64>` + `Vec<f64>` per node | `gc_before(now, 2.0)` | only the next ingest |
| renderable list | `Vec<VisualPacket>` | `gc_before(now, 2.0)` | `update_packet_cloud` per frame |

Both persistent stores share one GC system (`gc_timeline`) with one retention policy: **2 seconds past arrival**.

## Why the 2-second tail is wasted work

For the **renderable list**: the renderer never draws a packet with `arrive_real < now`. It checks `is_visible_at(now)` and skips them. Every retained-arrived packet is dead weight — walked, filtered, discarded, every frame, for two seconds after it ceased mattering.

For **causal triggers**: when the next emit from node N looks up `trigger_for(at_ns)`, it wants the most-recent arrival with `arrives_ns ≤ at_ns`. Older entries dominated by a later one with greater `arrives_ns` are unreachable. Most of what the 2-second GC retains is dominated junk.

The 2-second tail was originally there for "tests and debugging." After the entity-removal refactor, **nothing in production code reads arrived packets**. The remaining tests all read current state through `is_visible_at(now)`, which by definition excludes arrived ones.

### Measured cost (Life 30×30 random, M2 Max, vsync off)

```
phase                          p50         p95         p99         max
edges.gc_timeline             656 µs      1.3 ms      1.6 ms      1.9 ms
packet_cloud.update            1.5 ms     2.3 ms      2.4 ms      2.5 ms
```

Steady-state `timeline.packets.len() ≈ 14,500`. Most of those are in the 2-second tail. The gc_timeline retain() does ~14,500 comparisons every frame. `update_packet_cloud` does ~14,500 visibility checks every frame.

## Optimal shape

Three separate structures, each sized by what its job actually needs.

### Renderable list — sized by "in flight"

```rust
packets: VecDeque<VisualPacket>   // ordered roughly by arrive_real
```

- Push at the back (insertion order is approximately `arrive_real` for in-order ingestion).
- At the head of `update_packet_cloud`, pop the front while `front.arrive_real <= now`.
- Steady-state size: `emit_rate × mean_animation_duration`. For Life: ~7000/s × 0.41s ≈ **3,000 packets**, vs current 14,500. ~5× smaller.
- Not perfectly sorted (causal clamp can reorder by ~tens of ms); occasional out-of-order packets sit one or two slots past the head until they fall off naturally. Doesn't need to be sorted, just needs O(1) drop of dead-front-or-near-front.

### Causal triggers — sized by "what's in flight at this node"

```rust
node_triggers: HashMap<NodeId, BoundedRing<(at_ns, arrive_real)>>
```

- Bounded ring per node, cap ≈ 16 entries. Push appends; oldest falls off when full.
- `trigger_for(at_ns)` does a linear scan over ≤16 entries — equal to or faster than `partition_point` on a vec with 50+ entries.
- **No periodic GC.** Self-bounded.
- Memory for Life: ~900 nodes × 16 entries × 16 bytes ≈ 230 KB.

### Ingest — pure function

```rust
fn ingest(ev, real_now, k, &mut node_triggers) -> Option<VisualPacket>
```

Reads/writes `node_triggers`. Returns the packet to push onto the renderable list. No GC concerns — handled by the two structures it touches.

## Differences from today

| | Today | Optimal |
|---|---|---|
| renderable list size | ~14,500 | ~3,000 |
| renderable list pruning | `retain()` over 14.5k each frame, ~670 µs | pop_front while expired, <5 µs |
| trigger memory size | unbounded → GC'd to 2s tail | bounded ring per node, ~16 entries |
| trigger pruning | per-node linear scan + drain, called from `gc_timeline` | none — self-bounds at insert |
| `gc_timeline` system | exists, 670 µs/frame median | **deleted** |
| `update_packet_cloud` scan | 14.5k iterations | 3k iterations |
| Memory steady state | `2s × emit_rate` | bounded by in-flight count |

Estimated frame impact: gc_timeline saves ~670 µs, update_packet_cloud saves ~1 ms (5× smaller scan). About **1.7 ms/frame at p50**, more during sim bursts (where the timeline grows fastest).

## Subtle concern: NodeArrivals binary search may be unsound

`NodeArrivals.trigger_for` uses `partition_point(|ns| *ns <= at_ns)`, which requires `arrives_ns` to be sorted ascending. The doc-comment on the field claims monotonicity under sim-order ingestion. This is **not generally true**:

Events are ingested in `at_ns` (emit-time) order, but `arrives_at_ns = at_ns + latency`. A later-emitted packet with short latency can arrive at the same node before an earlier-emitted packet with long latency. Concretely:

```
emit at t=10, latency 50, → N (arrives_ns=60)   pushed first
emit at t=11, latency  4, → N (arrives_ns=15)   pushed second
```

`node_arrivals[N].arrives_ns` is now `[60, 15]` — not sorted, and `partition_point` returns wrong results.

For Life this never matters because all latencies are uniform. For real flow scenarios with heterogeneous edge weights, it could silently mis-clamp. **Worth a separate investigation** — fix the binary search (insert in sort order, or use a different lookup), independent of the GC work.

## Recommended implementation order, when we get to it

1. **Drop arrived packets immediately** in `update_packet_cloud` (or a tiny eager-drop system). Keep the rest as-is. Quickest win, smallest blast radius. ~1.5 ms/frame saved.
2. **Replace `node_arrivals` with bounded ring buffers**. Delete `gc_after` / `gc` on it. Contained to `visual.rs`.
3. **Delete `gc_timeline` system entirely** once 1+2 are done.

This order minimizes risk to causal-clamp correctness — (1) only touches the render-feeding side, (2) preserves the same lookup interface but with a more disciplined size bound.
