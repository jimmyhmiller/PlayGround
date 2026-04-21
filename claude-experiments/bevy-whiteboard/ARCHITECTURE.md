# Architecture — what we have, what's awkward

This is a stock-take of the simulation core (ignoring the Bevy UI) so
we can decide where to go next. It maps what's in `src/sim/mod.rs`
today to what it does, then calls out the awkward bits and the
features that aren't there yet.

## 1. Core data model

```
Sim
 ├── nodes: HashMap<NodeId, Node>
 ├── edges: HashMap<EdgeId, Edge>
 ├── outbound: HashMap<NodeId, Vec<EdgeId>>   // source → outgoing edges
 ├── now_ns: TimeNs                            // sim clock (nanoseconds)
 └── next_node_id / next_edge_id / next_packet_id
```

### Node

A component. The fields fall into four groups:

- **Identity** — `name`, `kind: NodeKind`, `color`
- **Program** — `mode: RuntimeMode`, `program: Vec<Instruction>`
- **Runtime state** — differs by mode:
  - per-packet: `holding: Option<Packet>`, `started_at_ns`, `buffer: VecDeque<Packet>`, `cursor_per_color`
  - sequential: `cursor: Option<Vec<usize>>`, `cursor_started_ns`, `cursor_awaiting: Option<PacketId>`
  - source: `emit_scheduled`
- **Stats** — `emitted / processed / dropped / total_in / total_out / lost / max_depth / sink_total / sink_per_color / sent / received / rtt_sum_ns / rtt_count / outstanding`
- **Composite membership** (legacy grouping) — `contains / input_port / output_port / parent`

### Edge

```rust
struct Edge {
    from, to: NodeId,
    mode: EdgeMode,           // Push | Pull
    from_row: Option<usize>,  // source-side row anchor for Steps containers
    last_sent_ns: TimeNs,     // used by PortKey::LastSentAt for round-robin
}
```

### Packet

```rust
struct Packet {
    id: PacketId,
    color: Color,
    reply: Option<ReplyAddress>,  // Some = request; return_path records path
}
```

## 2. Instruction set

One enum drives every node's behavior. Today:

**Entry / source**
- `EmitAtRate { period_ns, color, one_way }` — timer-driven source
- `Emit { color, one_way }` — cursor-driven one-shot emit (sequential mode)
- `AcceptInbound` — push entry point
- `PullInbound` — pull entry point (node asks upstream when idle)

**Gating / timing**
- `MatchColor { color }` — drop if packet color doesn't match
- `Buffer { capacity }` — FIFO
- `Process { duration_ns }` — hold a packet for a duration (per-packet)
- `Hold { duration_ns }` — hold the cursor for a duration (sequential)
- `AwaitResponse` — block cursor until a request's response returns
- `Require { reason }` — drop packet if current port set is empty

**Port pipeline** (routing by composition)
- `Filter { pred: PortPredicate }` — `Ready` or `ColorMatches`
- `Sort { key: PortKey }` — `LastSentAt | QueueDepth | EdgeOrder | Random`
- `Take { n: usize }` — keep first N ports
- `Send` — dispatch to remaining ports (broadcast if >1)

**Side effects**
- `Respond` — synthesize response to the reply-address
- `Consume` — terminal absorb (sink)

**Composition**
- `Sequence { label, body: Vec<Instruction> }` — named sub-program;
  cursor descends into `body`, ascends when done

## 3. Execution

### `advance_ns(dt_ns)`

Event-driven. The loop:

1. Ask `next_event_ns()` for the next scheduled time.
2. If past the deadline or no events: advance clock to deadline and
   return.
3. Else: set `now_ns` to that time and call `process_due`.
4. Repeat up to 10M iterations as a runaway guard.

### `next_event_ns()`

Walks every node and returns the earliest scheduled event among:
- Timer sources: `(emit_scheduled + 1) * period_ns`
- Workers with `holding`: `started_at_ns + processing_ns`
- Sequential-mode cursors:
  - `Hold` instruction: `cursor_started_ns + duration_ns`
  - `Emit` not awaiting and has outbound: `now` (fire immediately)
  - `AwaitResponse` not waiting: `now` (advance)
  - no-op leaf or Sequence descent: `now`
  - stuck (no edge, or awaiting response): nothing

### `process_due`

Runs, in order, each frame's due work:

1. `complete_processing` — packets whose `Process` dwell ended
   advance past the `Process` step via `run_steps_from`.
2. `drain_buffers` — any buffered packet whose next-ready-downstream
   is ready is popped and forwarded. Uses a separate
   `cursor_per_color` for round-robin — **does not run the pipeline
   in the program**.
3. `emit_due` — timer sources synthesize packets. Each new packet
   walks the program from the instruction after `EmitAtRate`.
4. `service_pullers` — any node with a Pull-mode inbound edge and a
   ready program tries to fetch one packet from its buffered upstream.
5. `tick_steps_nodes` — advance every sequential-mode cursor through
   its program. Caps at 1024 iterations per node per tick; runaway
   stalls the cursor.

### Per-packet execution — `run_steps_from(nid, packet, idx, events)`

Walks the program from `idx`. Each instruction either:
- advances `idx` and continues (instant),
- returns an outcome (`Forwarded | Consumed | Buffered | Processing
  | Dropped`), or
- mutates the port set (pipeline primitives), then continues.

The port set is lazy-initialised to "all push outbound edges" on the
first pipeline primitive. `Send` dispatches:
- 1 port → normal forward
- >1 ports → broadcast with fresh packet ids for clones

### Sequential execution — `tick_steps_nodes`

For each `Sequential`-mode node, walks `cursor: Vec<usize>` through
the program tree. Each iteration:
- resolves the current instruction via `instr_at(program, path)`
- if None, wraps back to `[0]` and emits `StepsLooped`
- if Sequence with body, descends (push 0 to path)
- if Hold, waits or advances by deadline
- if Emit, fires the request on the row's outbound edge (matched by
  `from_row`) and blocks on `cursor_awaiting`
- if AwaitResponse, blocks until `deliver_response` clears `cursor_awaiting`
- otherwise (leaf no-op) advances

If the inner loop runs its full 1024 iterations without breaking,
the cursor is cleared (`None`) — the program has no blocking step
and would otherwise spin. `push_instruction` reawakens by setting
`cursor = Some(vec![0])`.

### Request / response

A request packet carries `ReplyAddress { client, return_path, sent_at }`.
`travel_forward` appends the traversed edge to `return_path`.
`Respond` takes the address, calls `deliver_response` which walks
the path in reverse emitting `Traveled { is_response: true }` events.
At the originating node:
- `NodeKind::Client`: RTT is recorded, `received` tick.
- `NodeKind::Steps` sequential cursor: clears `cursor_awaiting`.

## 4. What works cleanly

- **Per-packet execution** is linear and easy to reason about.
- **Port-pipeline routing** decomposes cleanly. The 5 common
  strategies (broadcast / round-robin / content / least-loaded /
  failover) all drop out of combinations of `Filter`, `Sort`, `Take`,
  `Send`.
- **Sequential cursor with nested Sequences** — descent / ascent /
  auto-loop in a single recursive-ish walker.
- **Request/response edge tracing** — return path recorded on the
  way out, replayed on the way back. Stats update in one place.
- **Preset construction** — `add_generator`, `add_client`,
  `add_worker`, `add_queue`, `add_router`, `add_sink` are short
  programs. `client_step` / `worker_step` factor common sequential
  sub-programs.

## 5. Awkward bits — existing tech debt

### 5.1 Queue drain bypasses the pipeline

`drain_buffers` pops from a node's `buffer` and picks an outbound
edge using hardcoded `ready_outbound_candidates` + `cursor_per_color`
round-robin. The post-`Buffer` pipeline primitives in a Queue's
program are never executed — they're dead code there.

Implication: if the user edits a Queue's program to change its
drain strategy, nothing happens. The Queue always round-robins its
ready outbound.

Fix would be to run the remaining pipeline on each drained packet.
Non-trivial because drain happens outside the normal per-packet
walker. Also because the drained packet has no original `idx` to
resume from — we'd need to resume from "the instruction after
`Buffer`" explicitly.

### 5.2 `Process` vs `Hold` — parallel concepts

- `Process { duration_ns }` holds a **packet** (per-packet mode).
- `Hold { duration_ns }` holds the **cursor** (sequential mode).

Same duration argument, different semantics, two arms in every
match. An honest unification is "`Wait { duration }` does whichever
is applicable in context" — needs the executor to branch at
execution time, not enum-variant time.

### 5.3 `Emit` and `EmitAtRate` — similar overlap

`EmitAtRate` is "every period_ns, emit one packet" — timer-driven.
`Emit` is "at cursor arrival, emit one packet" — cursor-driven.
They could be unified as `Emit { trigger: Timer | Cursor }`, but
the timer path is routed through `emit_due` while cursor is routed
through `tick_steps_nodes`. Unifying the data doesn't unify the
code path.

### 5.4 `AcceptInbound` vs `PullInbound`

Also parallel. Both are "entry points," one for push and one for
pull. Could be `Accept { mode: Push | Pull }` without losing
information. Would collapse a few match arms.

### 5.5 Worker still takes push-inbound as a fallback

`deliver_push` treats a node with no `AcceptInbound` as "first
`MatchColor/Process/Buffer` is the entry point." This is a
compatibility crutch from when Generators pushed directly into
Workers. Feels like a hidden rule and is easy to forget about.

### 5.6 Composite nodes

`NodeKind::Custom` + `group_into_composite`. Lets you drag a box
around a few nodes and treat them as a unit. Members still live in
the outer sim. The composite has its own program (`Accept → Sort →
Take 1 → Send`) that fans inbound traffic to members. Mostly
independent from Steps containers — two parallel "grouping"
mechanisms. Unclear which one survives.

### 5.7 Stats explosion on `Node`

`emitted / processed / dropped / total_in / total_out / lost /
max_depth / sent / received / sink_total / sink_per_color / rtt_sum_ns /
rtt_count / outstanding` — roughly one stat per kind's primary
metric, all on every node. Most are zero most of the time. A
cleaner factoring would be a `NodeStats` sub-struct or per-kind
typed stats.

### 5.8 Sequential mode is only used by one kind

`RuntimeMode { PerPacket, Sequential }` is an explicit field, but
`Sequential` is only ever set by `add_steps`. `NodeKind::Steps` and
`RuntimeMode::Sequential` carry the same information. Mostly
harmless, but the enum exists in anticipation of a second
sequential kind that hasn't shown up.

### 5.9 Dead / vestigial code

- `ReplyMode` enum in `Packet` doc — defined, not used.
- `Packet::is_response` method — always returns false.
- `no_ready_reason_for` — unreachable after the pipeline migration.
- Several `Node` methods (`buffer_capacity`, `buffer_len`,
  `is_source`, `set_match_color`) — not called from anywhere.
- `cursor_per_color` field on `Node` — only still used by
  `drain_buffers` (see 5.1); everything else uses per-edge
  `last_sent_ns`.

## 6. Known lies

### 6.1 Analytical rates don't cover the new model

`Sim::analytical_edge_rates` pattern-matches on `EmitAtRate`,
`Process`, `MatchColor`, and used to match on `ForwardOut`.
`ForwardOut` is gone. Custom pipelines (`Filter / Sort / Take /
Send`) and Steps containers aren't modelled at all. Probes on
edges leaving those nodes silently report 0 or wrong numbers.

Realistic fix is to stop trying to solve it analytically and
measure instead — rolling window of `SimEvent::Traveled` events
per edge.

## 7. Missing: controllable randomness

`PortKey::Random` uses `xor_shift(now_ns.wrapping_mul(0x9E37_79B1)
^ edge_id)` as a sort key. This is deterministic for identical
inputs, so in practice it's reproducible as long as the sim clock
advances identically between runs. But:

- There's no **seed** parameter, so one sim's "random" order is
  always the same order.
- Any future randomness source (probabilistic drops, jitter, random
  service times) would have no shared RNG state.
- If the user wants to test "another run with a different seed," we
  have no mechanism.

What we need:
- A `rng_seed: u64` field on `Sim`, optionally settable.
- A `rng_state` that advances on each use (xoshiro, pcg, whatever).
- All "randomness" in the sim routes through that state.
- Reset: at sim creation or explicit `Sim::reset_rng(seed)`.

Notably, this also matters for **rewind** (below): reproducible
randomness is a precondition for replaying any segment of sim time.

## 8. Missing: rewind

We don't have rewind. The sim runs forward only. The Bevy layer's
visible packets are spawned from events and then discarded — there's
no history.

### What rewind would need

Two viable approaches:

**A. Snapshot + replay.** Every N sim ms (or every K events), clone
the whole `Sim` into a bounded ring buffer. Rewind = pop a snapshot
and replay forward to the exact target time, emitting the intervening
events so the UI can animate them again.

Pros: easy to reason about. Sim state is self-contained.

Cons:
- `Sim` derives `Clone`, so snapshots are cheap memory-wise for
  small graphs, but grow linearly with packet count (buffers +
  holding + outstanding).
- Events that cascaded synchronously within a single `advance_ns`
  tick are atomic — rewind can step between ticks, not within.
- The Bevy side (visual packets, probe histories) needs to
  roll back in sync, or else reset on rewind.

**B. Reverse-step.** Instead of snapshots, record an inverse for
every state mutation (packet pushed → packet popped, buffer
enqueued → dequeued, etc.). Rewind just plays the inverse.

Pros: constant memory, exact.

Cons: pervasive. Every mutation site has to emit an inverse;
easy to miss one and corrupt state silently. Composite
operations like `travel_forward` would need their inverses too.

I'd go with (A). It's the lower-risk option and snapshot cost is
bounded for the scale we're targeting (hand-drawn whiteboards, not
thousand-node simulations).

### Interaction with RNG

For (A) to replay identically, every RNG consumption has to be
deterministic from the snapshot. This means the RNG state must
live on `Sim` (fix in §7) and be included in the snapshot.

### Interaction with the Bevy layer

Sim rewind should also rewind visuals. The simplest model:
- On rewind, despawn all `TravelingPacket` / `StepsLoopDot` /
  `PaletteGhost` entities, clear `TickEvents`, `StepsRowActivity`.
- Replay forward from the target snapshot; events re-spawn visuals.

This means the Bevy layer should treat sim events as the single
source of truth for transient visuals. It mostly already does.

## 9. Open design questions

- **Should a user-defined preset be more than a Sequence snapshot?**
  Today a preset captures `label + Vec<Instruction>` at save time.
  No parameterisation. A "Client for color X" preset is bound to
  X. If we want parameterised presets ("a Client, parameterised on
  color and rate"), the preset model needs holes.

- **Ports vs edges.** `from_row` is a Step-container concept — it
  tags an outbound edge as "emerging from row N." Should this
  generalise: every node has a set of typed output ports, and edges
  carry the source port? Today only Steps containers have multiple
  semantically distinct outputs; other kinds have one.

- **Per-row inbound.** We don't support incoming edges anchored to
  a specific row of a Steps container. Everything flows into the
  container as a whole (row 0 by convention). Realistic programs
  might want "row N is the entry point for inbound on this edge."

- **What even is a response from a Steps container?** When a user
  builds a custom Client-like step with `Emit` + `AwaitResponse`,
  and someone sends a request TO that Steps container, what should
  happen? Today the container has no Respond step and the request
  hangs. The whole bi-directional request/response story is
  tailored to the preset Clients and hasn't been generalised.

## 10. Short list, ranked

If I were picking what to address next, in priority order:

1. **Measured probes** (§6.1). Stops the lying.
2. **Seeded RNG on `Sim`** (§7). Prerequisite for rewind; small
   change on its own.
3. **Snapshot-based rewind** (§8). The big one. Also the
   highest-leverage feature — unlocks "what happens if I change
   this mid-run?" workflows.
4. **Unify Queue drain with the pipeline** (§5.1). Makes Queue
   routing actually editable.
5. **Unify `Process`/`Hold`, `EmitAtRate`/`Emit`, `Accept`/`Pull`**
   (§5.2–5.4). Reduces match surface; cleaner primitive list for
   the step library palette.

Everything else in §5 is small cleanup.
