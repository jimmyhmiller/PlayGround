# RL Loop Scenario & Design Criteria

This document specifies a concrete high-concurrency workload — an IMPALA-style
distributed reinforcement learning loop — in full detail, then derives the set
of design criteria the runtime and IR must satisfy for it to work well.

We use this scenario as the demanding case the design is judged against. If a
realistic RL workload can be expressed inside the constraints, and the runtime
can execute it with competitive throughput, inspectability, and recovery
properties, the design works.

---

## 1. Reference workload

### 1.1 The system we are specifying

A distributed actor-learner RL system, IMPALA-shaped:

- **Many concurrent environment instances** (the actors) generating
  experience.
- A **shared sharded replay buffer** absorbing transitions.
- A **learner** sampling batches from the buffer and updating a neural network
  policy.
- A **versioned policy** broadcast to actors; actors use the latest version
  they can read.
- An **eval pool** periodically running deterministic rollouts for metrics.
- **Metric streams** aggregated and exposed.

The same model is intended to support PPO (synchronous, on-policy), SAC
(off-policy, replay-driven), and population-based variants without changing
the runtime.

### 1.2 Scale targets

These are the numbers the design must defensibly hit. They are not the
ceiling; they are the threshold for "this works for serious RL."

| Quantity                              | Target                       |
|---                                    |---                           |
| Concurrent env instances              | 256 – 4,096                  |
| Aggregate env-step throughput         | 50,000 steps/sec             |
| Per-env-step scheduler overhead (p99) | < 50 µs                      |
| Policy-inference batch size           | 32 – 256 (adapter-batched)   |
| Replay buffer size                    | 100K – 1M transitions        |
| Replay write throughput               | linear in shard count        |
| Learner gradient step rate            | 5 – 50 / sec                 |
| Policy version freshness lag (p50)    | < 4 gradient steps           |
| Distributed nodes                     | 1 (v1) → up to 16 (later)    |
| Wall-clock duration of a run          | hours to days                |
| MVCC memory overhead                  | bounded, no unbounded growth |

### 1.3 Hardware shape we design for

- One GPU node hosting the learner and the inference batcher.
- N CPU nodes hosting the actors and replay shards.
- Standard 10–25 Gbps cross-node networking.
- v1 may run all of this on a single machine; the design must not preclude
  splitting later.

### 1.4 What "success" looks like

- The full system can be expressed in the IR without escape hatches or
  bespoke runtime extensions.
- The runtime hits the throughput targets above.
- The inspector exposes every observability property listed in §9 without the
  algorithm code containing any instrumentation.
- A crash mid-run can be recovered to a consistent state by log replay.
- Switching between sequential and concurrent schedulers requires no IR
  changes.

---

## 2. Decomposition into sub-programs

The full system is composed of five sub-programs, each owning its own state
cells and handlers, communicating only via declared events and effects.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          rl_run  (parent program)                          │
│                                                                            │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │   actor_pool     │   │   replay_buf     │   │     learner      │        │
│  │                  │◄─►│                  │◄─►│                  │        │
│  │ envs, last_pv    │   │ shards, indexes  │   │ params, opt_state│        │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│         ▲                                              │                   │
│         │            ┌──────────────────┐              │                   │
│         │            │     evaluator    │              │                   │
│         │            │ eval_envs, score │              │                   │
│         │            └──────────────────┘              │                   │
│         │                                              │                   │
│         └────────── PolicyUpdated (broadcast) ─────────┘                   │
│                                                                            │
│  ┌──────────────────┐                                                      │
│  │  metrics_sink    │   ← consumes EmitMetric from every sub-program       │
│  └──────────────────┘                                                      │
└────────────────────────────────────────────────────────────────────────────┘
```

Each sub-program is fully self-contained: its state cells are private, its
internal events are not visible to others, and its surface to the parent is
the declared `forward_events`, `surface_events`, and `surface_effects`.

This decomposition is not a runtime trick. It is part of the IR. The
inspector renders these as collapsible regions of the program map.

---

## 3. Schemas

### 3.1 Shapes (structural; aliases for readability only)

```
EnvId       = string                          # opaque
ShardId     = u32                             # hash(EnvId) % num_shards
PolicyVer   = u64                             # monotonic
TransitionId = u128

Observation = { ... env-specific tensor shape ... }
Action      = { ... env-specific action shape  ... }

Transition = {
  env_id:         EnvId,
  obs:            Observation,
  action:         Action,
  reward:         f32,
  next_obs:       Observation,
  done:           bool,
  policy_version: PolicyVer,
  step_index:     u64,
  ts:             Timestamp,
}

PolicyParams = {
  version: PolicyVer,
  blob:    Bytes,                  # opaque; layout known only to adapters
}

Gradients = {
  version_basis: PolicyVer,
  blob:          Bytes,
}

Batch = {
  batch_id:    BatchId,
  rows:        list<Transition>,
  drawn_from:  list<ShardId>,
}
```

### 3.2 Events

```
event StepCompleted    { env_id, obs, reward, done, step_index }
event ActionDecided    { env_id, action, policy_version }
event BatchSampled     { batch_id, batch }
event GradientsReady   { batch_id, gradients }
event PolicyPublished  { new_version }
event Tick             { now }
event EvalRequested    { eval_id, policy_version }
event EvalCompleted    { eval_id, score, episode_count }
event AdapterFailure   { effect_kind, request_id, reason }
```

All effect responses (success or failure) arrive as events of some
corresponding kind. `AdapterFailure` is the uniform failure-variant.

### 3.3 State cells

```
# Owned by actor_pool
state envs:        Map<EnvId, EnvState>
state pending_act: Map<EnvId, PendingAct>          # awaiting PolicyInfer
state pending_env: Map<EnvId, PendingEnv>          # awaiting StepEnv

# Owned by replay_buf
state shards:        Map<ShardId, RingBuffer<Transition>>
state shard_indexes: Map<ShardId, ShardIndex>      # for sampling

# Owned by learner
state params:        Versioned<PolicyParams>       # MVCC cell
state opt_state:     OptimizerState
state train_step:    u64
state in_flight:     Map<BatchId, BatchMeta>

# Owned by evaluator
state eval_envs:     Map<EnvId, EnvState>          # disjoint from train envs
state recent_scores: RingBuffer<EvalRecord>

# Owned by metrics_sink
state streams:       Map<MetricName, MetricStream>
```

`Versioned<T>` is a runtime-provided cell kind that retains multiple
versions and serves snapshot reads. It is part of the IR's primitive cell
vocabulary, not an extension.

### 3.4 Effects

```
effect StepEnv          { env_id, action }       -> StepEnvResp
effect PolicyInfer      { obs, policy_version }  -> PolicyInferResp
effect SampleBatch      { count, shard_hint? }   -> SampleBatchResp
effect ComputeGradients { batch }                -> ComputeGradientsResp
effect ApplyGradients   { gradients }            -> ApplyGradientsResp
effect StoreTransition  { transition }           -> ack
effect EmitMetric       { name, value, tags }    -> ack
effect PublishParams    { params }               -> ack
```

Each `-Resp` is a sum: success variant + failure variant. The runtime
translates effect-adapter outcomes into the appropriate event kind.

---

## 4. Handlers

Each handler below lists `on`, `read`, `write`, `emit`, and a body sketch.
The body is pseudocode; in the real system it is a WASM component
implementing a generated WIT world.

### 4.1 actor_pool

```
handler on_env_step {
  on:    StepCompleted { env_id, obs, reward, done, step_index }
  read:  envs[env_id],
         params.version                       # version pointer only
  write: envs[env_id],
         pending_act[env_id]
  emit:  PolicyInfer, StoreTransition, EmitMetric

  body:
    let e  = read envs[env_id]
    let pv = read params.version              # snapshot pointer

    emit StoreTransition { transition: Transition {
      env_id, obs: e.last_obs, action: e.last_action,
      reward, next_obs: obs, done,
      policy_version: e.last_pv,
      step_index, ts: now()
    } }

    let next = if done then reset_env_state(env_id, e)
               else { ...e, last_obs: obs, last_pv: pv, step_index: step_index + 1 }
    write envs[env_id] = next

    let req_id = derive_id(env_id, step_index)
    write pending_act[env_id] = { req_id, policy_version: pv }
    emit PolicyInfer { obs, policy_version: pv } as req_id

    emit EmitMetric { name: "env_step", value: 1, tags: { env_id } }
}

handler on_action {
  on:    ActionDecided { env_id, action, policy_version }
  read:  pending_act[env_id], envs[env_id]
  write: pending_act[env_id],   # delete
         envs[env_id],
         pending_env[env_id]
  emit:  StepEnv

  body:
    delete pending_act[env_id]
    let e = read envs[env_id]
    write envs[env_id] = { ...e, last_action: action }

    let req_id = derive_id(env_id, e.step_index)
    write pending_env[env_id] = { req_id }
    emit StepEnv { env_id, action } as req_id
    # StepEnvResp arrives as StepCompleted -> closes the actor loop
}
```

### 4.2 replay_buf

```
handler on_store {
  on:    StoreTransition { transition }
  read:  # none
  write: shards[shard_of(transition.env_id)],
         shard_indexes[shard_of(transition.env_id)]
  emit:  # none

  body:
    let sid = shard_of(transition.env_id)
    append shards[sid] with transition
    update shard_indexes[sid]
}

handler on_sample {
  on:    SampleBatchRequested { batch_id, count, shard_hint? }
  read:  shards[*], shard_indexes[*]          # read-only across all shards
  write: # none — sampling is pure read
  emit:  BatchSampledEffect                   # response routed back to learner

  body:
    let rows = sample_uniform(count, shards, shard_indexes, shard_hint?)
    emit BatchSampledEffect { batch_id, batch: { batch_id, rows,
                                                 drawn_from: shards_used } }
}
```

Note that `shards[*]` is a read-only declaration across all shards. The
scheduler permits this to run concurrently with `on_store` invocations
because reads do not conflict with writes under MVCC.

### 4.3 learner

```
handler train_tick {
  on:    Tick
  read:  train_step
  write: in_flight[*]
  emit:  SampleBatch

  body:
    let bid = fresh_id()
    write in_flight[bid] = { requested_at: now(),
                              version_basis: read params.version }
    emit SampleBatch { count: 256 } as bid
}

handler on_batch {
  on:    BatchSampled { batch_id, batch }
  read:  in_flight[batch_id]
  write: in_flight[batch_id]
  emit:  ComputeGradients

  body:
    write in_flight[batch_id].sampled_at = now()
    emit ComputeGradients { batch } as batch_id
}

handler on_grads {
  on:    GradientsReady { batch_id, gradients }
  read:  in_flight[batch_id]
  write: in_flight[batch_id]
  emit:  ApplyGradients

  body:
    write in_flight[batch_id].grads_at = now()
    emit ApplyGradients { gradients } as batch_id
}

handler on_params {
  on:    PolicyPublished { new_version }
  read:  train_step
  write: train_step, in_flight[*]   # cleanup any matching in-flight rows
  emit:  EmitMetric

  body:
    write train_step = read train_step + 1
    cleanup_in_flight_for(new_version)
    emit EmitMetric { name: "policy_version", value: new_version, tags: {} }
}
```

The learner is a chain: `Tick → SampleBatch → ComputeGradients →
ApplyGradients → PolicyPublished → next Tick`. Each link is a separate
recorded handler invocation.

### 4.4 evaluator

```
handler on_eval_request {
  on:    EvalRequested { eval_id, policy_version }
  read:  # none (uses fresh envs)
  write: eval_envs[eval_id_keyed]
  emit:  PolicyInfer, StepEnv

  body:
    # spawn N eval episodes, similar to actor loop but deterministic
    ...
}

handler on_eval_done {
  on:    EvalCompleted { eval_id, score, episode_count }
  read:  # none
  write: recent_scores[*]
  emit:  EmitMetric

  body:
    append recent_scores with { eval_id, score, episode_count, at: now() }
    emit EmitMetric { name: "eval_score", value: score, tags: { eval_id } }
}
```

### 4.5 metrics_sink

```
handler on_metric {
  on:    MetricArrived { name, value, tags, ts }
  read:  # none
  write: streams[name]
  emit:  # none

  body:
    append streams[name] with { value, tags, ts }
}
```

The metrics sink is a single sub-program subscribed to `EmitMetric` from
every other sub-program. This is the canonical observability sink, but
note that nothing in the algorithm depends on it — the inspector reads the
event log directly.

---

## 5. Effect adapters

Adapters are the host-side fulfillers of effects. They are where
non-determinism, batching, and resource limits live.

### 5.1 StepEnv

- Backing: vectorized env stack (Gym/Brax/Isaac/whatever).
- Concurrency: N vector workers; each owns a shard of envs.
- Failure modes: env crash, observation NaN, timeout.
- On failure: emits `AdapterFailure { effect_kind: StepEnv, request_id, reason }`.

### 5.2 PolicyInfer

- Backing: GPU inference batcher.
- Behavior: buffers incoming requests for up to `window_us`
  (default 2ms) or `max_batch` (default 128), whichever comes first;
  issues one batched forward pass; demultiplexes responses.
- Picks the requested `policy_version` from the MVCC params store if
  recent enough; otherwise falls back to latest (and tags the response
  with the actual version used).
- Failure modes: GPU OOM, NaN logits, unknown version.

### 5.3 ComputeGradients

- Backing: GPU learner step.
- Behavior: one batch at a time; computes loss, backward pass, gradients.
- Failure modes: NaN loss, OOM. NaN-loss → adapter discards and emits
  failure; learner retries with next batch.

### 5.4 ApplyGradients

- Behavior: optimizer step; writes new params version into MVCC store;
  emits `PublishParams` internally; emits `PolicyPublished` event.
- After write, **retains last K versions** (configurable; default 8).
  Older versions are eligible for GC once no reader pins them.

### 5.5 SampleBatch

- Backing: replay_buf's `on_sample` handler — modeled as an effect to
  preserve the event-driven shape.

### 5.6 StoreTransition

- Routes to replay_buf's `on_store`. Modeled as an effect from
  actor_pool's perspective.

### 5.7 EmitMetric

- Backing: metrics_sink. Modeled as effect emission to keep metrics
  cross-cutting.

### 5.8 PublishParams

- Internal to the learner sub-program; updates MVCC params cell.

---

## 6. Concurrency model

### 6.1 Per-slice ordering

The runtime guarantees:

- For any single state slice (e.g., `envs[env_id=42]`), handler
  invocations that touch it are serialized.
- For slices that do not overlap, no ordering guarantee is provided.

This is the only ordering primitive. Anything stronger (e.g., total
order across all events) is opt-in for testing and not used in
production RL.

### 6.2 MVCC for `Versioned<T>` cells

- Reads pin the current version at scheduling time.
- Writes produce a new version; old versions persist until no reader
  pins them and the retention policy allows GC.
- Readers and writers do not block each other.

`params` is the only `Versioned` cell in this scenario. Everything else
is partitioned-by-key (`envs[env_id]`, `shards[shard_id]`) and serializes
per-key.

### 6.3 Sharded keyed state

For any cell of shape `Map<K, V>`:

- A handler footprint `cell[k]` (with `k` bound from event/reads) is
  treated as a slice keyed on `k`.
- A footprint `cell[*]` is treated as the entire cell.
- The scheduler runs handlers with disjoint key sets concurrently.

### 6.4 Worker affinity

Handlers can declare a soft `affinity: by(<key>)` hint. The scheduler
tries to dispatch invocations with the same key to the same worker
thread to keep state hot in cache. This is a hint, not a correctness
requirement.

### 6.5 Optimistic concurrency + retry

Concurrent writes to the same slice may race. The runtime uses CAS-style
apply: a handler invocation reads at version `V`, computes, attempts to
apply writes at version `V`. If a conflicting write committed in the
interim, the invocation is **discarded and re-run** with a fresh
snapshot. This is safe because handler bodies are pure functions of
their declared inputs.

Retry count is bounded; after the bound, the invocation is marked
failed and an `AdapterFailure`-style event is enqueued.

### 6.6 Backpressure

Effects can be back-pressured by the adapter:

- `PolicyInfer`: if the batch queue is full, the adapter delays
  accepting new emits. The runtime queues the effect emit; the
  handler invocation completes normally.
- `StoreTransition`: if the buffer is full, the oldest transition in
  that shard is evicted (standard ring-buffer semantics).
- `ComputeGradients` / `ApplyGradients`: serial by design; the
  learner's chain is one-at-a-time.

Backpressure is observable in the inspector as queue depth on the
effect's adapter.

### 6.7 Determinism modes

Two modes, switchable per program:

- **Strict**: Total event order; one handler invocation at a time.
  Used for debugging.
- **Loose**: Per-slice order only. Used for production. The recorded
  log captures the actual scheduling decisions, so a loose-mode run
  can be replayed in strict mode to step through.

---

## 7. Data flow & timing

### 7.1 Actor loop, single env

```
t=0    StepCompleted(env=42, step=N)
       └─ on_env_step
            ├─ snapshot envs[42] and params.version  (~1 µs)
            ├─ emit StoreTransition                  (~1 µs queue)
            ├─ write envs[42]                        (~2 µs CAS)
            └─ emit PolicyInfer(req=A)               (~1 µs queue)

t=~2ms ActionDecided(env=42, action, policy_version)
       └─ on_action
            ├─ delete pending_act[42]
            ├─ write envs[42] (action)
            └─ emit StepEnv(req=B)

t=~3ms StepCompleted(env=42, step=N+1)
       └─ loop
```

The body of `on_env_step` is the hot path. It is single-digit
microseconds excluding the actual `PolicyInfer` round-trip (which is
the bulk of wall-clock latency but is the GPU's time, not the
scheduler's).

### 7.2 Steady state

256 envs each in a loop with ~3 ms cycle time (dominated by
batched inference) = ~85,000 env-step events per second. The
scheduler must dispatch and apply that many handler invocations per
second, distributed across CPU cores. At 50 µs scheduler overhead per
invocation and 32 cores, the runtime can sustain ~640,000
invocations/sec. **Headroom is 7×.**

### 7.3 Learner chain

```
Tick → SampleBatch (≈1 ms)
     → ComputeGradients (≈50 ms on GPU)
     → ApplyGradients (≈10 ms)
     → PolicyPublished (new version K+1)
     → next Tick
```

One full learner iteration is ~60 ms, giving ~16 grad steps/sec.
Sample-to-publish lag is recorded per `BatchMeta` row.

### 7.4 Critical paths

- **Actor critical path**: `on_env_step → PolicyInfer adapter →
  on_action → StepEnv adapter → on_env_step`. Scheduler overhead must
  stay well under the adapter latencies.
- **Learner critical path**: `train_tick → sample → grads → apply →
  publish`. Adapter latencies dominate; scheduler is irrelevant.
- **Policy lag**: distance between `policy.version` read by actors and
  `policy.version` written by learner. Recorded per Transition.

---

## 8. Failure modes

| Failure                  | Detection                          | Recovery                                   |
|---                       |---                                 |---                                         |
| Handler body trap        | WASM trap                          | Log failure event; runtime continues       |
| Adapter failure          | Adapter emits `AdapterFailure`     | Failure event routed to originating emit's handler chain |
| Env crash                | `StepEnv` returns failure variant  | Reset env; emit fresh `StepCompleted`      |
| GPU OOM                  | Inference/grad adapter returns failure | Reduce batch size; retry                |
| NaN loss                 | Grad adapter returns failure       | Discard batch; learner emits next `SampleBatch` |
| Replay buffer overflow   | `on_store` evicts oldest           | Normal ring-buffer behavior                |
| Runtime crash            | Process exit                       | Replay event log from snapshot             |
| Network partition (dist) | Cross-node effect timeout          | Failure variant; sub-program responsible for retry |
| MVCC version exhaustion  | Retention policy                   | GC oldest unreferenced version             |
| Retry storm              | Retry counter exceeds limit        | Mark invocation failed; log event          |

The core principle: **all failures are events**. There are no
exceptions, no try/catch, no out-of-band error channels. A handler
deals with a failure-variant event the same way it deals with any
other event.

---

## 9. Inspectability properties

For a running or completed RL job, the inspector exposes (without any
instrumentation in the algorithm code):

- **Per-env trajectory replay**. For any env, show the sequence of
  observations, actions, rewards, and which policy version produced
  each action.
- **Policy lag distribution**. Histogram of `(current_version −
  transition.policy_version)` across recent transitions.
- **Inference batch size distribution**. Plot of batch sizes the
  `PolicyInfer` adapter actually executed.
- **Replay buffer fill and write rate per shard**. Heatmap of writes
  over time; flags contention if shards skew.
- **Learner pipeline timing**. For each `BatchMeta`, show
  `requested_at → sampled_at → grads_at → published_at` waterfalls.
- **Throughput**. Env-steps/sec, grad-steps/sec, both from log counts.
- **Conflict graph**. Live view of which handler instances contended
  for which slices, with retry counts.
- **Worker utilization**. Per-worker timeline of handler invocations.
- **Effect failure rates**. Per-adapter failure counts and recent
  reasons.
- **Counterfactual replay**. Fork at any logged event; substitute
  inputs; replay forward. Useful for "what if the learner had skipped
  this batch?"

---

## 10. Design criteria

These are the falsifiable claims the design must satisfy. They are the
acceptance contract for the runtime, IR, and inspector. Each criterion
has a name, statement, and a means of verification.

### Correctness criteria

**C1 — Read isolation.** A handler invocation observes no state slice
outside its declared `read` footprint. *Verified by*: WIT linking
(no import for undeclared cells); runtime trap on disallowed access.

**C2 — Write isolation.** A handler invocation modifies no state slice
outside its declared `write` footprint. *Verified by*: WIT linking;
runtime trap.

**C3 — Effect isolation.** A handler invocation emits no effect type
outside its declared `emit` set. *Verified by*: WIT linking.

**C4 — Per-slice serialization.** For any single slice, no two
handler invocations write it concurrently. *Verified by*: stress
test with synthetic conflicts; absence of torn writes; scheduler
unit tests.

**C5 — Determinism under replay.** Replaying a recorded log produces
a state functionally equivalent to the original. *Verified by*:
post-replay snapshot equality of all state cells (modulo MVCC GC).

**C6 — Effect-response routing.** Each effect response is delivered as
the event tagged with the originating emit's id. *Verified by*: log
inspection invariant — every response event has a matching emit row
with the same id.

**C7 — Retry safety.** A handler invocation that is retried due to
optimistic-concurrency conflict produces the same writes/emits as a
single successful invocation. *Verified by*: bodies are pure
functions of their declared inputs, plus runtime fuzz test that
injects spurious retries.

**C8 — Failure as event.** Every adapter failure surfaces as an event
in the same inbox as successes; handlers do not see exceptions.
*Verified by*: adapter API has no exception channel; integration
tests inject failures and confirm event delivery.

### Performance criteria

**P1 — Scheduler overhead bound.** Per-invocation overhead, excluding
body work and effect latency, is < 50 µs at p99 under a load of
100 invocations/sec/worker. *Verified by*: benchmark with no-op
bodies and recorded percentiles.

**P2 — Inference batching.** The `PolicyInfer` adapter produces
batches of size ≥ 32 at p50 when offered ≥ 256 concurrent in-flight
requests with a 2 ms window. *Verified by*: batch size histogram
from log.

**P3 — Replay buffer scaling.** Write throughput scales linearly in
the number of shards up to (num_workers) shards. *Verified by*:
throughput vs shard count benchmark.

**P4 — MVCC memory bound.** Steady-state MVCC memory is bounded by
`K × sizeof(params)` for retention `K`. *Verified by*: heap
profiling under steady-state.

**P5 — Throughput target.** Aggregate env-step throughput ≥ 50,000
steps/sec on a single 32-core node with a 1-GPU learner.
*Verified by*: end-to-end benchmark.

**P6 — Sequential→concurrent equivalence.** A program developed and
tested under the sequential scheduler runs under the concurrent
scheduler with no IR changes. *Verified by*: integration test runs
the same fixture under both schedulers.

**P7 — Log throughput.** Event log append sustains ≥ 1 M
records/sec on a single node. *Verified by*: log microbenchmark.

### Observability criteria

**O1 — Historical reconstruction.** For any past time `t` in the
recorded log, the inspector can reconstruct the value of any state
cell at `t`. *Verified by*: snapshot-from-replay round-trip test.

**O2 — Zero instrumentation.** None of the properties listed in §9
require code in the algorithm. *Verified by*: handlers contain no
metric, log, or trace calls; all observability is derived from IR
+ event log.

**O3 — Conflict graph from IR.** The handler conflict graph (which
handlers can race on which slices) is derivable from declared
footprints alone. *Verified by*: scheduler exposes it via API and
the inspector renders it without runtime telemetry.

**O4 — Per-invocation record.** Every handler invocation is
recorded with: input event, read snapshot, output writes, output
emits, retry count, wall-clock duration, worker id.
*Verified by*: log schema enforcement.

**O5 — Counterfactual replay.** A user can fork the log at any
event, substitute inputs, and replay forward. *Verified by*:
inspector exposes fork; result is a distinct logical run.

### Composability criteria

**K1 — Sub-program contract expressible.** A sub-program's
contract surface (events forwarded in, events surfaced out, effects
exposed to the parent) is fully expressed in the parent's manifest.
*Verified by*: the rl_run parent manifest is sufficient to
generate parent WIT worlds.

**K2 — Sub-program state isolation.** A handler in the parent cannot
declare a read or write footprint against a sub-program's internal
state cell. *Verified by*: IR validator rejects such manifests.

**K3 — Sub-program inspector rendering.** Nested programs appear as
collapsible regions of the program map; per-sub-program timelines
filter to that region. *Verified by*: inspector UI fixture test.

**K4 — Effect routing across sub-programs.** An effect emitted by a
handler in one sub-program is fulfilled by the appropriate adapter
or sub-program handler without the emitter knowing.
*Verified by*: integration test where a sub-program is swapped for
an equivalent one with different internals.

### Failure & recovery criteria

**F1 — Body trap containment.** A WASM trap in a handler body does
not crash the runtime; it produces a failure event and the runtime
continues. *Verified by*: fault-injection integration test.

**F2 — Adapter failure delivery.** Adapter failures produce
failure-variant events delivered to the originating emit's
continuation. *Verified by*: §C8 test.

**F3 — Crash recovery.** Restarting the runtime against a complete
event log reproduces the same in-memory state (modulo MVCC
retention). *Verified by*: kill-and-restart test in steady state.

**F4 — Retry bound.** Retry storms on a contended slice are bounded
by a per-invocation retry limit; exceeded invocations fail and are
logged. *Verified by*: synthetic contention test.

**F5 — Partial node loss (deferred to distributed v2).** Loss of a
non-learner node degrades throughput but does not corrupt the
event log or in-memory state on remaining nodes.

### Distribution criteria (deferred to v2)

**D1 — Sub-program placement.** Sub-programs can be placed on
different runtime nodes without IR changes.

**D2 — Cross-node effect latency bound.** Effects routed across
nodes add ≤ 5 ms at p99 on a local network.

**D3 — Cross-node log reconciliation.** Per-node logs reconcile
into a single causal log by logical clock.

---

## 11. What this scenario tells us about the design

Reviewing the criteria above against the architecture from
`GOALS.md`:

- **Most correctness criteria fall out of WIT linking and pure
  bodies.** No runtime cleverness required beyond enforcement.
- **Most performance criteria depend on runtime implementation
  choices** — MVCC, sharded log, worker affinity, adapter batching —
  but none require IR changes.
- **All observability criteria derive from the existence of a
  complete event log + IR.** They are properties of the design, not
  features that need to be added.
- **Composability criteria are the most under-specified.** The
  parent-manifest surface (forward_events / surface_events /
  surface_effects) needs to be nailed down precisely; this is the
  single largest open IR question.
- **Failure criteria all reduce to "failures are events."** No
  exception channels in the body language.

### Implications for the IR

The IR must explicitly support:

1. `Versioned<T>` as a primitive cell kind (for MVCC reads).
2. `Map<K, V>` cells with key-bound access paths (`cell[k]`,
   `cell[*]`).
3. `RingBuffer<T>` and similar runtime-managed cells (sharded).
4. Soft scheduler hints (`affinity: by(key)`) attached to handlers.
5. Sub-program embedding with declared exposure surface.
6. Failure-variant response types on every effect.
7. Effect-emit ids for response routing.

### Implications for the runtime

The runtime must implement:

1. Optimistic concurrency with pure-body retry.
2. MVCC versioned cells with retention policy and GC.
3. Sharded append-only event log with logical clock.
4. Effect adapters as plug-in components with backpressure and
   batching support.
5. Both sequential and concurrent schedulers with identical
   semantics modulo timing.
6. Worker pool with optional key-affinity dispatch.
7. Replay-from-log starting from any snapshot.
8. Counterfactual fork.

### Implications for the inspector

The inspector must provide:

1. Program map with nested sub-program regions.
2. State cell view with reader/writer/conflict graph.
3. Handler card with full footprint and latest invocation stats.
4. Per-handler and per-invocation timelines.
5. Filterable event log view (by handler, by slice, by time range).
6. Counterfactual replay UI.
7. Live and historical modes (live = streaming SSE; historical =
   served from log).

---

## 12. Open questions surfaced by this scenario

These are decisions the rest of the design must answer before the RL
case can be implemented end-to-end:

- **MVCC retention policy shape.** Last-K versions? Time-based?
  Reader-pinned with timeout? The right answer affects memory
  behavior under stalled readers.
- **Shard count as IR data vs runtime config.** Is `shards`'
  number-of-shards fixed in the manifest, or runtime-tunable? Tunable
  is more flexible but complicates the conflict graph.
- **Effect emit ids: runtime-generated or body-supplied?** The body
  shows `emit ... as req_id` with body-supplied ids. Runtime-supplied
  is simpler; body-supplied lets you correlate manually. Probably
  body-supplied with an "auto" default.
- **Backpressure surface.** Does the body see a backpressure signal,
  or does the runtime just queue and acknowledge? Visible
  backpressure lets the body shed load; invisible keeps the body
  simple.
- **`Versioned<T>` write semantics.** Must a writer always produce a
  new version, or can it overwrite the current one? Always-new is
  cleaner; overwrite-current allows certain optimizations.
- **Per-sub-program inboxes vs single global inbox.** Single global
  is simpler; per-sub-program lets sub-programs be paused/resumed
  independently.
- **Affinity as IR data or scheduler config.** Probably IR — the
  affinity hint is a property of the handler, not the deployment.
- **Eval pool isolation.** Is the eval pool's effect routing
  separated from training (separate adapters / GPU sharing rules), or
  is it the same `PolicyInfer` adapter with eval requests tagged?
