# Decoupling sim from render

Design doc for moving the flow simulator off the Bevy render thread, with a snapshot interface between them. Captures the analysis so a fresh session can pick this up without re-deriving the context.

Companion: [`visual-timeline-redesign.md`](./visual-timeline-redesign.md). The timeline GC waste is a separate concern; this doc is about the bigger architectural decoupling.

## Why this matters

Measured on `examples/life_30x30_random.whiteboard`, M2 Max, vsync off, after the entity-removal refactor:

```
phase                  p50         p95         p99         max
sim.run_until.total    1.5 µs      22.9 ms     23.8 ms     24.5 ms
sim.fire_rules         0.2 µs      20.7 ms     21.7 ms     22.2 ms
```

Sim runs synchronously inside the Bevy `Update` schedule. When it spikes to 22 ms (Life ticks where most cells flip simultaneously), the entire frame waits for it. The full 92 ms p99 frame tail is *the sim blocking everything else*, not 22 ms of sim plus ordinary rendering.

Decoupling sim from render moves that 22 ms onto a separate thread so the render frame proceeds independently. p99 frame time should drop close to p50.

This is also a structural improvement: the sim and renderer become independent systems with an explicit interface, instead of the renderer reaching directly into sim state every frame.

## Current architecture

The sim is `flow::Sim`. The Bevy resource `FlowSim` wraps it. `bridge::advance_sim` calls `flow.sim.run_until(target)` inline each frame.

**Render systems read sim state directly.** Every frame:

| Reader | What it reads |
|---|---|
| `sync_node_state_labels` (nodes.rs) | `flow.sim.nodes[id].slots` |
| `sync_binary_slot_paint` (nodes.rs) | `flow.sim.nodes[id].slots[paint.slot]` |
| `draw_edges` (edges.rs) | iterates `flow.sim.edges` |
| `bridge::collect_new_events` | drains `flow.sim.log` since last frame |
| inspector / probes / hud | various `flow.sim.*` reads |

These are sync reads of the live `Sim`. Moving sim onto a worker without a snapshot would either deadlock (worker holds `Sim`, renderer waits) or require locking (defeats the point).

**Writes also go straight at the sim.** Inspector slot edits, palette drops, canvas seeds, scenario loads — all currently call `flow.sim.method(...)` from Bevy systems.

## Proposed architecture

Three parts:

```
┌──────────────────┐                    ┌──────────────────┐
│  Bevy main       │   commands         │  sim worker      │
│  thread          │  ──────────────►   │  thread          │
│  (renderer)      │                    │  (owns Sim)      │
│                  │   snapshot         │                  │
│                  │  ◄──────────────   │                  │
└──────────────────┘                    └──────────────────┘
       reads                                 owns + mutates
```

### 1. `SimSnapshot` — what the renderer reads

```rust
pub struct SimSnapshot {
    pub now_ns: u64,
    /// Per-node slot maps. Cloned at snapshot time — the renderer
    /// never sees mid-tick mutation.
    pub node_slots: BTreeMap<NodeId, BTreeMap<String, Value>>,
    /// Edge topology view. Rarely changes.
    pub edges: BTreeMap<EdgeId, EdgeView>,
    /// Events recorded since the previous snapshot. Drained by the
    /// host into `NewEvents` for the visual timeline.
    pub new_events: Vec<Event>,
    pub error_counts: BTreeMap<String, u64>,
    pub perf_samples: Vec<(&'static str, f64)>,
}
```

Worker publishes a fresh `Arc<SimSnapshot>` after each tick. Bevy resource holds the latest `Arc`. Read path is a single atomic load — no lock, no copy. Use `arc-swap::ArcSwap` (the standard crate for this) or a simple `Arc<Mutex<Arc<SimSnapshot>>>` indirection.

`EdgeView` is a renderer-friendly subset of `Edge` — fields the UI actually reads, no expression bodies or other heavy state.

### 2. Sim worker

```rust
fn sim_worker(
    mut sim: Sim,
    snapshot_out: Arc<ArcSwap<SimSnapshot>>,
    commands_in: mpsc::Receiver<SimCommand>,
    multiplier: Arc<AtomicF64>,
    paused: Arc<AtomicBool>,
) {
    let mut last_tick = Instant::now();
    let mut prev_log_index = sim.log.total_recorded;
    loop {
        // Drain commands first so Pause / LoadCanvas take effect
        // before we advance time.
        while let Ok(cmd) = commands_in.try_recv() {
            apply_command(&mut sim, cmd);
        }
        if paused.load() {
            std::thread::park_timeout(Duration::from_millis(10));
            last_tick = Instant::now();
            continue;
        }
        let now = Instant::now();
        let elapsed = now - last_tick;
        let dt_sim_ns = (elapsed.as_secs_f64() * multiplier.load() * 1e9) as u64;
        if dt_sim_ns > 0 {
            sim.run_until(sim.now_ns + dt_sim_ns);
            last_tick = now;
            let snap = make_snapshot(&sim, &mut prev_log_index);
            snapshot_out.store(Arc::new(snap));
        } else {
            std::thread::yield_now();
        }
    }
}
```

Notes:

- The worker tracks wall clock itself. The renderer doesn't drive sim time.
- Tick rate is "as fast as the worker can run while keeping `dt_sim_ns > 0`" — typically yields most of the time. During bursts it runs hot.
- Multiplier and pause state are atomics (cheap reads on the hot path; cheap writes from the host).
- A more sophisticated version could enforce a fixed tick rate, but the simple wall-clock approach preserves current semantics.

### 3. Commands main → sim

Every write that today goes straight to `flow.sim` becomes a command:

```rust
enum SimCommand {
    LoadCanvas(Box<Canvas>),
    EditSlot { node: NodeId, slot: String, value: Value },
    Inject { to: NodeId, payload: Value },
    AddNode { /* ... */ },
    AddEdge { /* ... */ },
    DespawnNode(NodeId),
    SetMultiplier(f64),  // also writes the AtomicF64 directly
    Pause,
    Resume,
    StepOnce(u64),
    Reset(u64), // seed
}
```

Sent via `mpsc::Sender` from Bevy systems. Worker drains at the top of each tick.

`Pause` / `Resume` / multiplier are duplicated as atomics so the worker reacts within one yield, not the full command-drain interval.

## Migration

### Read sites

Every `flow.sim.foo` access in Bevy systems becomes `snapshot.foo`. Mostly mechanical:

- `nodes.rs::sync_node_state_labels` — `&Res<FlowSim>` → `&Res<SimSnapshotRes>`
- `nodes.rs::sync_binary_slot_paint` — same
- `edges.rs::draw_edges` — same
- `bridge::collect_new_events` — drains `snapshot.new_events` instead of `flow.sim.log`
- inspector / probes / hud / examples / canvas — same pattern, ~30 sites total

A `SimSnapshotRes(pub Arc<SimSnapshot>)` resource wraps the snapshot. A small system at the start of `Update` reads `arc_swap.load_full()` into the resource so all downstream systems see a consistent snapshot for the frame.

### Write sites

Anywhere Bevy systems mutate `flow.sim.*`:

- inspector slot edits → `EditSlot` command
- palette drops → `AddNode` + `AddEdge` commands
- `examples::handle_load_example` → `LoadCanvas` command
- `canvas::seed_from_path` → `LoadCanvas` command
- `palette` hotkeys (multiplier, pause, step) → atomic writes / commands

### Spawn / despawn boundary

`LoadExample` and `seed_from_path` currently:

1. Call `flow.sim.add_node(...)` repeatedly.
2. Synchronously, in the same system, spawn Bevy entities for each new node.

Under the snapshot model, step 1 sends a command. The worker applies it. The next snapshot reflects the new sim state. *Then* a Bevy system spawns entities for nodes that appear in `snapshot.node_slots` but don't have a corresponding `Entity` yet (and despawns the inverse).

So entity-spawning becomes a *reconciliation* pass: walk the snapshot, ensure each node has an entity. Same pattern Bevy uses for any data-driven world.

This shifts the load flow:
- old: synchronous "load canvas → sim populated → entities spawned" all in one frame
- new: command sent → worker loads → snapshot publishes → reconciliation system spawns entities, possibly one frame later

That latency is fine. The user sees it as "loading" for one frame.

## Tests

Tests today drive the sim synchronously: `world.resource_mut::<FlowSim>().sim.run_until(...)` then inspect `flow.sim.nodes`.

Under the worker model, that pattern doesn't work — `flow.sim` doesn't exist as a Bevy resource any more. Two options:

**A. Direct mode for tests.** Introduce a `SimDriver`:

```rust
enum SimDriver {
    Direct(Sim),                         // synchronous, used by tests
    Worker { handle: WorkerHandle,       // separate thread, used by app
              snapshot: Arc<ArcSwap<SimSnapshot>> },
}
```

Tests use `Direct`. The app uses `Worker`. Both publish snapshots; both accept commands. The trait-like surface is identical for read paths. Test helper builds an app with `Direct`.

This is the right answer. Determinism for tests + threading for app.

**B. Single-step worker for tests.** Worker has a "tick once" mode where the test thread calls `step()`. Less natural; the worker thread machinery still exists.

Go with A. The Direct/Worker split is small and contained.

## Where the work goes

Rough scope, ordered:

1. **`SimSnapshot` type + `make_snapshot(&Sim) -> SimSnapshot`.** ~80 lines. Build it from current `Sim` state; just clones.
2. **Worker thread + command channel.** ~150 lines. Spin it up at `App::build`; tear down at exit.
3. **`SimSnapshotRes` resource + per-frame load system.** ~30 lines.
4. **Spike: migrate the two hottest read sites** (`sync_binary_slot_paint`, `sync_node_state_labels`). Run the bench. Confirm p99 cliff drops and tests still pass for the migrated systems.
5. **Migrate remaining read sites.** ~30 sites, mostly mechanical.
6. **Migrate write sites to commands.** Inspector / palette / examples / canvas. The `LoadCanvas` flow is the hardest; everything else is straightforward.
7. **`SimDriver::Direct` for tests.** Update `crates/flow-bevy/tests/common/mod.rs` to construct an app with direct driver.
8. **Delete the old `FlowSim` resource.** Once nothing reads it directly.

Steps 1–4 are the spike — that's where we de-risk. If the spike confirms the architecture, 5–8 are mechanical.

Estimated 2–3 days focused. Bigger than the entity-removal refactor; smaller than rewriting the renderer.

## Hard parts / gotchas

### Snapshot cost

Each tick clones every node's slot map. For Life: 901 nodes × ~3 slots × small `Value`s. Probably under 200 µs per snapshot. Watch for it in the bench after migration.

If it shows up as hot:
- **Incremental snapshots**: only changed slots since last snapshot. Requires tracking dirty bits in the sim.
- **Structural sharing**: `im::OrdMap` instead of `BTreeMap`. Trades sharded mutation cost for cheap clones. Probably overkill for our scale.

Don't over-engineer until measured.

### Determinism

Sim is currently deterministic given the same RNG seed + scenario + command sequence. Under the worker model, *command ordering* is what tests need to control. `Direct` mode applies commands inline → identical to today. `Worker` mode applies them in receive order; for the app this is fine, for tests it'd be flaky.

Conclusion: tests use Direct; only the app uses Worker. Solved.

### Snapshot freshness

A snapshot is published after each worker tick. If the worker is mid-tick when a render frame starts, the renderer sees the *previous* tick's snapshot. That's intentional (no waiting). The visible effect is up to one tick of staleness — at typical sim multiplier and tick rate, well under a frame.

### Render systems that should *write* the sim

A few render systems today write sim state during interaction (e.g., dragging a node updates its position in some pseudo-state). Audit each: most are pure-visual (Bevy `Transform`, no sim slot involved). Where they do write, convert to a command.

### Probes

Probes capture sim state for inspection panels. Today they read `flow.sim` directly. Easy migration: read `snapshot.node_slots` instead.

### Step-once mode

The current `SimClock::step_once_ns` flow means "advance the sim by exactly N ns this frame." Becomes a `StepOnce(N)` command. Worker checks the command queue, runs once, publishes snapshot, then the renderer sees the result.

For tests, `Direct` mode handles step-once trivially.

### Multiple snapshots in flight

The worker can produce snapshots faster than the renderer consumes them (especially at vsync-locked 60 fps with a fast sim). `ArcSwap::store` simply replaces the latest; the renderer always reads the most recent. Older `Arc`s drop when the renderer's previous reference is released. No queue buildup.

## Decisions to validate during the spike

1. `arc-swap` vs `Arc<Mutex<Arc<SimSnapshot>>>`. ArcSwap is cleaner and lock-free; Mutex works fine and avoids a dep. Pick one.
2. Whether perf samples should still be drained from snapshots (yes, probably — it's already part of the sim's data we want to see).
3. Whether the worker should yield, sleep, or park when idle. `yield_now` is fine for a busy-loop equivalent at our scale; `park_timeout(1ms)` is more polite if the sim multiplier is low.
4. Snapshot sample throughput. After the spike, check if making a snapshot every tick is the right cadence, or if it should be throttled (e.g., max 240 Hz).

## Out of scope for this work

- Replacing the sim with a multi-threaded one. The sim itself is still single-threaded inside the worker. Parallelizing rule firing is a different project.
- Network sync / replay. Snapshots could be the basis for both, but adding either is a separate scope.
- Visual timeline redesign — see [`visual-timeline-redesign.md`](./visual-timeline-redesign.md). Independent of this work.

## Starting checklist for a fresh session

1. Re-read `bridge.rs::advance_sim` and the list of read/write sites in this doc to confirm nothing has shifted.
2. Run the bench, confirm p99 frame is still ~92 ms with sim spike at ~22 ms (or note new numbers).
3. Build the spike: `SimSnapshot`, worker, `SimSnapshotRes`, two migrated read sites.
4. Re-run bench. Confirm p99 drops and the new sim phase appears in `bevy diagnostics` (the worker is still timing itself; perf samples flow through the snapshot).
5. If confirmed, work through the migration in the order listed above.
6. Delete `FlowSim` last.
