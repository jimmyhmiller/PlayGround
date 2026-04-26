# Spatial behavior via three orthogonal primitives

## The idea

Don't add "space" to the simulator. Add three small generic features and let
spatial behavior fall out as a library on top of them.

1. **Dynamic edges.** `Effect`/`Action::Connect { from, to, latency }` and
   `Disconnect { edge_id }`. Edges become first-class mutable topology, not
   a fixed declaration.

2. **Recurring timeline ("tickline").** Today, `Timeline` actions fire at
   specific times with god-view access (read/write any slot, trigger any
   action). Add `every: Duration` so an entry can fire on a cadence.

3. **Position as metadata.** The engine doesn't learn about `x`/`y`. They're
   just numbers in node slots (or packet metadata). Whatever reads them
   decides what they mean.

## Why this composes into proximity

Proximity is no longer a sim feature. It's a tickline action:

```
every 16ms:
  for each pair (a, b) of nodes with x/y metadata:
    if distance(a, b) < R and not connected(a, b):
      Connect(a, b, latency=...)
    if distance(a, b) >= R and connected(a, b):
      Disconnect(edge(a, b))
```

Once edges exist, normal packet flow over those edges does the actual
communication. The engine never has a concept of "space" — it just has
nodes, edges, packets, and a god-view cron job that reshapes topology.

## Why this is the right factoring

- **Each primitive is independently useful.** Dynamic edges matter for
  markets, gossip, trust networks — proximity or no proximity. Recurring
  timeline matters for any periodic global behavior (rebalancing, cleanup,
  scheduled snapshots). Metadata is already there.
- **The formalism stays pure.** Rules remain local; only timeline has
  god-view access, and timeline already does. No new "spatial mode" fork.
- **Other interpretations of "distance" come free.** Trust distance, graph
  hops, 3D, hyperbolic — same machinery. The tickline decides what
  "distance" means.
- **Pheromone / environment fields reduce to the same primitives.** A
  field cell is a gadget at (x, y) with a slot; proximity edges form to
  nearby cells; deposits and reads are normal packets. So diffusion
  systems compose without adding an `Environment` resource.

## Things to think through before building

- **Cadence picking.** The engine is event-driven (`run_until` jumps to
  the next scheduled event). "Every tick" means "a recurring scheduled
  action." The cadence is a knob — too coarse and proximity lags reality;
  too fine and you spam Connect/Disconnect. Probably configurable per
  tickline entry.
- **Cross-node read access.** Tickline needs to scan all nodes' metadata
  to compute neighborhoods. Timeline already has this; rules deliberately
  don't. Keep this capability on timeline, not on rules — protects rule
  locality.
- **Event log noise.** Emitting Connect/Disconnect every cadence will
  swamp the log. Log only *deltas* (edges that actually changed this
  tick), not the full neighborhood scan.
- **Determinism / replay.** Same metadata → same neighborhoods → same
  edges. Replay stays intact. Good.
- **Cost.** Naive O(N²) is fine for ~100 nodes. Beyond that, the
  tickline implementation can keep its own spatial index (grid hash,
  k-d tree). The engine doesn't need to know.
- **Movement.** Just slot writes. A per-node rule, a global tickline
  action, or a scenario action can update `x`/`y`. No new mechanism.

## Engine changes required

Small:

- `Action::Connect { from, to, latency }` and `Action::Disconnect { edge }`
  in the timeline/scenario action enum.
- Corresponding `Event::EdgeAdded` / `Event::EdgeRemoved` in the event log
  so the Bevy layer and rewind tests stay correct.
- An `every: Option<Duration>` field on timeline entries (or a
  self-rescheduling action — same outcome).
- `Effect::ScheduleSelf { delay, payload }` is a parallel small win — gives
  rules access to delays without needing the scheduler to be a tickline.

That's the whole spatial story, expressed as three small features that
each pay rent independently.
