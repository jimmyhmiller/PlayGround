//! Pre-built scenarios the user can load with one click.
//!
//! Each [`Example`] is a topology + parameter set that demonstrates a
//! specific behavior of the flow formalism. Loading an example wipes the
//! current canvas (sim + entities + associated maps) and rebuilds from
//! scratch, so examples don't have to worry about what was there before.
//!
//! To add a new one: append a variant, give it a label + description,
//! and implement `build`. The palette's Examples section picks up every
//! variant in [`Example::ALL`].

use bevy::prelude::*;
use flow::{NodeId, Value};

use crate::bridge::{EntityMaps, FlowEdgeRef, FlowNodeRef, FlowSim};
use crate::edges::{HiddenEdges, TravelingPacket};
use crate::gadgets::{self, Kind, spawn as spawn_gadget};
use crate::nodes::{NodeAssetCache, NodeCounter, spawn_node_entity};
use crate::probes::Probe;
use crate::theme::Theme;
use crate::tool::NodeColors;

// ────────────────────────────────────────────────────────────
// Plugin + event
// ────────────────────────────────────────────────────────────

pub struct ExamplesPlugin;
impl Plugin for ExamplesPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<LoadExample>()
            .add_systems(Update, handle_load_example);
    }
}

/// Fire this message to clear the canvas and load the chosen example.
/// Consumed by [`handle_load_example`]. (Bevy 0.18 renamed events →
/// messages; non-entity cross-system signals live in `Messages<_>`.)
#[derive(Message, Clone, Copy, Debug)]
pub struct LoadExample(pub Example);

// ────────────────────────────────────────────────────────────
// Example catalog
// ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Example {
    /// The original three-colour fan-out demo — Gen → Router → Queue →
    /// Worker(s) → Sink in three lanes.
    ThreeLaneFanout,
    /// Minimal request/response: Client → Worker, Worker replies via
    /// `return_path.head()` after popping. The client sees RTT.
    ClientWorker,
    /// Client → Router → Worker, showing return_path preserved through
    /// a transparent forwarder. The user gets to see the Router is NOT
    /// on the response's path — the reply goes directly Worker → Client.
    ClientRouterWorker,
    /// Two independent clients send requests to a single Worker.
    /// Each client pushes *itself* onto `return_path` before emitting,
    /// so the Worker's pop-and-reply sends each response back to the
    /// originating client — never crossing streams. This is the demo
    /// that pins the load-bearing property of `return_path`.
    TwoClientsOneWorker,
    /// Client submits requests to a Queue. The Queue acks each one
    /// back to the Client immediately (bumping `len`, popping
    /// `return_path`), then a Worker pulls from the Queue on its own
    /// cadence. Models "async submit + ack" — the Client's RTT is
    /// the submission round-trip, not end-to-end processing time.
    ClientQueueWorker,
    /// Twenty BackoffClients sharing a single Worker. When the Worker
    /// is up, the clients' staggered base periods keep retries
    /// spread out. Toggle the Worker `up → down`: every client's
    /// next request gets a `resp_error` and they all seed
    /// `backoff_ns = period_ns` at roughly the same instant. From
    /// there the fixed-backoff doubling locks them into synchronised
    /// 1s / 2s / 4s / 8s waves — the thundering herd. Bring the
    /// Worker back up and watch a wall of simultaneous reqs hit it.
    BackoffHerd,
}

impl Example {
    pub const ALL: &'static [Example] = &[
        Example::ThreeLaneFanout,
        Example::ClientWorker,
        Example::ClientRouterWorker,
        Example::TwoClientsOneWorker,
        Example::ClientQueueWorker,
        Example::BackoffHerd,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Example::ThreeLaneFanout    => "3-lane fanout",
            Example::ClientWorker       => "Client→Worker",
            Example::ClientRouterWorker => "Client→Router→Worker",
            Example::TwoClientsOneWorker => "2 Clients→1 Worker",
            Example::ClientQueueWorker  => "Client→Queue→Worker",
            Example::BackoffHerd        => "Backoff herd",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Example::ThreeLaneFanout    => "Three colour lanes through a router to workers of varying capacity.",
            Example::ClientWorker       => "Smallest request/response demo — Client pushes self, Worker pops and replies.",
            Example::ClientRouterWorker => "Router forwards the request transparently; Worker still replies directly to Client.",
            Example::TwoClientsOneWorker => "Two clients share one worker. Each client's return_path steers its own response back — they never cross.",
            Example::ClientQueueWorker  => "Client submits to a queue, queue acks immediately; a worker pulls from the queue on its own schedule.",
            Example::BackoffHerd        => "Twenty BackoffClients hammering one Worker. Toggle the Worker down/up to see fixed exponential backoff synchronise into a thundering herd.",
        }
    }
}

// ────────────────────────────────────────────────────────────
// The load pipeline
// ────────────────────────────────────────────────────────────

/// Consume [`LoadExample`] events: first wipe the canvas, then
/// dispatch to the variant-specific builder. If multiple events fire
/// in one frame we only apply the last — loading is idempotent, so
/// the earlier ones would just get overwritten anyway.
fn handle_load_example(
    mut events: MessageReader<LoadExample>,
    mut commands: Commands,
    mut cache: ResMut<NodeAssetCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut flow: ResMut<FlowSim>,
    mut maps: ResMut<EntityMaps>,
    mut counter: ResMut<NodeCounter>,
    mut node_colors: ResMut<NodeColors>,
    mut hidden: ResMut<HiddenEdges>,
    mut timeline: ResMut<crate::edges::VisualTimelineRes>,
    theme: Res<Theme>,
    nodes_q: Query<Entity, With<FlowNodeRef>>,
    edges_q: Query<Entity, With<FlowEdgeRef>>,
    probes_q: Query<Entity, With<Probe>>,
    packets_q: Query<Entity, With<TravelingPacket>>,
) {
    let Some(LoadExample(example)) = events.read().last().copied() else { return; };

    // ── wipe ──────────────────────────────────────────────
    for e in nodes_q.iter() { commands.entity(e).despawn(); }
    for e in edges_q.iter() { commands.entity(e).despawn(); }
    for e in probes_q.iter() { commands.entity(e).despawn(); }
    for e in packets_q.iter() { commands.entity(e).despawn(); }

    // Fresh sim — keeps the formalism honest. Param defaults are
    // re-installed because tests and examples rely on them (emit_period,
    // service_mean, …).
    let seed = 1;
    let mut new_sim = flow::Sim::new(seed);
    gadgets::install_default_params(&mut new_sim);
    flow.sim = new_sim;
    flow.consumed_log_index = 0;

    // Reset the visual timeline: drop any in-flight visual packets
    // (their entities were just despawned above) and clear the
    // per-node causal arrival history. F12 uses `real_now` freshly
    // at each ingestion so no anchor update is needed here — the
    // next event after load will emit near the current wall clock.
    timeline.0.reset();

    maps.node_to_entity.clear();
    maps.entity_to_node.clear();
    maps.edge_to_entity.clear();
    maps.entity_to_edge.clear();
    node_colors.0.clear();
    hidden.set.clear();
    counter.0 = 0;

    // ── build ─────────────────────────────────────────────
    let mut ctx = BuildCtx {
        commands: &mut commands,
        cache: &mut cache,
        meshes: &mut meshes,
        materials: &mut materials,
        flow: &mut flow,
        maps: &mut maps,
        counter: &mut counter,
        node_colors: &mut node_colors,
        hidden: &mut hidden,
        theme: &theme,
    };
    match example {
        Example::ThreeLaneFanout    => build_three_lane(&mut ctx),
        Example::ClientWorker       => build_client_worker(&mut ctx),
        Example::ClientRouterWorker => build_client_router_worker(&mut ctx),
        Example::TwoClientsOneWorker => build_two_clients_one_worker(&mut ctx),
        Example::ClientQueueWorker  => build_client_queue_worker(&mut ctx),
        Example::BackoffHerd        => build_backoff_herd(&mut ctx),
    }
}

// ────────────────────────────────────────────────────────────
// Builder helpers (shared scaffolding)
// ────────────────────────────────────────────────────────────

struct BuildCtx<'a, 'w, 's> {
    commands: &'a mut Commands<'w, 's>,
    cache: &'a mut NodeAssetCache,
    meshes: &'a mut Assets<Mesh>,
    materials: &'a mut Assets<ColorMaterial>,
    flow: &'a mut FlowSim,
    maps: &'a mut EntityMaps,
    counter: &'a mut NodeCounter,
    node_colors: &'a mut NodeColors,
    hidden: &'a mut HiddenEdges,
    theme: &'a Theme,
}

impl BuildCtx<'_, '_, '_> {
    /// Place a gadget of the given kind: creates the sim node, the
    /// Bevy entity at `pos`, and records its data-palette colour
    /// (unless Router, which is neutral).
    fn place(&mut self, kind: Kind, slot: usize, pos: Vec2) -> NodeId {
        self.counter.0 += 1;
        let name = format!("{}_{}", kind.label(), self.counter.0);
        let fid = spawn_gadget(&mut self.flow.sim, kind, &name, slot);
        spawn_node_entity(
            self.commands,
            self.cache,
            self.meshes,
            self.materials,
            self.maps,
            self.theme,
            fid,
            kind,
            None,
            name,
            pos,
        );
        if !matches!(kind, Kind::Router) {
            self.node_colors.0.insert(fid, self.theme.data[slot]);
        }
        fid
    }

    /// Wire an edge the same way the Connect tool does — delegates to
    /// the shared helper so pull semantics (Worker→Queue reverse
    /// data edge) and Client→Worker auto-response-edges happen
    /// uniformly with manual drawing.
    fn wire(&mut self, from: NodeId, fk: Kind, to: NodeId, tk: Kind) {
        crate::edges::wire_flow_edge(
            &mut self.flow.sim,
            self.maps,
            self.hidden,
            self.commands,
            from, Some(fk),
            to,   Some(tk),
        );
    }

    /// Overwrite a slot on a sim node. Used to tune rates / service
    /// times from their gadget defaults.
    fn set_slot(&mut self, nid: NodeId, slot: &str, value: Value) {
        if let Some(n) = self.flow.sim.nodes.get_mut(&nid) {
            n.slots.insert(slot.into(), value);
        }
    }
}

// ────────────────────────────────────────────────────────────
// Individual scenario builders
// ────────────────────────────────────────────────────────────

/// The original fan-out demo — moved here verbatim from the old
/// `seed_demo_graph`, minus scaffolding that's now in `BuildCtx`.
fn build_three_lane(ctx: &mut BuildCtx) {
    // Generators — one per colour.
    let gens: [NodeId; 3] = [
        ctx.place(Kind::Generator, 0, Vec2::new(-600.0,  260.0)),
        ctx.place(Kind::Generator, 1, Vec2::new(-600.0,    0.0)),
        ctx.place(Kind::Generator, 2, Vec2::new(-600.0, -260.0)),
    ];
    // One router fans the three streams out.
    let router = ctx.place(Kind::Router, 0, Vec2::new(-300.0, 0.0));
    // Queues — one per colour.
    let queues: [NodeId; 3] = [
        ctx.place(Kind::Queue, 0, Vec2::new(-50.0,  260.0)),
        ctx.place(Kind::Queue, 1, Vec2::new(-50.0,    0.0)),
        ctx.place(Kind::Queue, 2, Vec2::new(-50.0, -260.0)),
    ];
    // Workers — 1 / 2 / 3 per lane. Red lane is the bottleneck.
    let workers_r: [NodeId; 1] = [ctx.place(Kind::Worker, 0, Vec2::new(230.0, 260.0))];
    let workers_y: [NodeId; 2] = [
        ctx.place(Kind::Worker, 1, Vec2::new(230.0,  40.0)),
        ctx.place(Kind::Worker, 1, Vec2::new(230.0, -40.0)),
    ];
    let workers_b: [NodeId; 3] = [
        ctx.place(Kind::Worker, 2, Vec2::new(230.0, -200.0)),
        ctx.place(Kind::Worker, 2, Vec2::new(230.0, -260.0)),
        ctx.place(Kind::Worker, 2, Vec2::new(230.0, -320.0)),
    ];
    // Sinks — one per colour, workers fan in.
    let sinks: [NodeId; 3] = [
        ctx.place(Kind::Sink, 0, Vec2::new(520.0,  260.0)),
        ctx.place(Kind::Sink, 1, Vec2::new(520.0,    0.0)),
        ctx.place(Kind::Sink, 2, Vec2::new(520.0, -260.0)),
    ];

    // Rates: 30/s demand vs per-lane capacity (20/40/60 per-second).
    for g in gens { ctx.set_slot(g, "period_ns", Value::Int(33_000_000)); }
    for w in workers_r.iter().chain(workers_y.iter()).chain(workers_b.iter()) {
        ctx.set_slot(*w, "service_ns", Value::Int(50_000_000));
    }

    // Edges. Pull semantics (Worker→Queue reverse edge) is in wire().
    for g in gens { ctx.wire(g, Kind::Generator, router, Kind::Router); }
    for (slot, q) in queues.iter().enumerate() {
        ctx.wire(router, Kind::Router, *q, Kind::Queue);
        let workers: &[NodeId] = match slot {
            0 => &workers_r,
            1 => &workers_y,
            _ => &workers_b,
        };
        for w in workers {
            ctx.wire(*w, Kind::Worker, *q, Kind::Queue);
            ctx.wire(*w, Kind::Worker, sinks[slot], Kind::Sink);
        }
    }
}

/// The smallest demo that exercises return_path: Client pushes self
/// on `req`, Worker pops and replies on `resp`. The Connect-helper
/// auto-creates the hidden Worker→Client response edge.
fn build_client_worker(ctx: &mut BuildCtx) {
    let client = ctx.place(Kind::Client, 0, Vec2::new(-250.0, 0.0));
    let worker = ctx.place(Kind::Worker, 0, Vec2::new(150.0, 0.0));
    // Slower client so responses are visible, faster worker so no queue.
    ctx.set_slot(client, "period_ns", Value::Int(200_000_000));
    ctx.set_slot(worker, "service_ns", Value::Int(40_000_000));
    ctx.wire(client, Kind::Client, worker, Kind::Worker);
}

/// Request/response through a Router — two edges, no triangle.
///
/// The Router opts into the reply path: its `forward_req` rule
/// pushes `self` onto the packet's return_path, so the Worker's
/// pop lands on the Router (not the Client directly). The Router's
/// new `forward_resp` rule then pops itself and relays the resp to
/// whoever's now at head — the Client. Both hops use the engine's
/// reverse-route along the user-drawn edges.
///
/// Trace:
///   Client → Router (req pushing self)    return_path = [client]
///   Router → Worker (req pushing self)    return_path = [router, client]
///   Worker pops, emits resp to Router     reverses Router→Worker edge
///   Router pops, emits resp to Client     reverses Client→Router edge
fn build_client_router_worker(ctx: &mut BuildCtx) {
    let client = ctx.place(Kind::Client, 0, Vec2::new(-400.0,  0.0));
    let router = ctx.place(Kind::Router, 0, Vec2::new(-100.0,  0.0));
    let worker = ctx.place(Kind::Worker, 0, Vec2::new( 250.0,  0.0));

    ctx.set_slot(client, "period_ns",  Value::Int(250_000_000));
    ctx.set_slot(worker, "service_ns", Value::Int(60_000_000));

    ctx.wire(client, Kind::Client, router, Kind::Router);
    ctx.wire(router, Kind::Router, worker, Kind::Worker);
}

/// Two clients, one worker. Both clients push `self` onto the packet's
/// `return_path` before emitting — the Worker's `serve` rule pops the
/// head and replies there. Without return_path-by-packet (i.e. under
/// the old `reply_to` model that got clobbered on every hop), this
/// scenario would be ambiguous: who does the Worker reply to? With
/// the new mechanism the answer is intrinsic to each packet.
///
/// Visible effect: both clients' `completed` counters grow in step
/// with their emit rates. No cross-talk. Spatial separation (one
/// above, one below) distinguishes the two streams on the canvas;
/// both share the same data-palette colour because the Worker does
/// no colour filtering (it serves any `req(_)`). Using mismatched
/// colours here would falsely imply the Worker rejects some requests.
fn build_two_clients_one_worker(ctx: &mut BuildCtx) {
    let client_a = ctx.place(Kind::Client, 0, Vec2::new(-400.0,  180.0));
    let client_b = ctx.place(Kind::Client, 0, Vec2::new(-400.0, -180.0));
    let worker   = ctx.place(Kind::Worker, 0, Vec2::new( 250.0,    0.0));

    // Distinct cadences make it obvious the two clients are independent
    // — responses arrive at their respective rates, not interleaved by
    // round-robin coincidence. The Worker's `busy` gate means
    // coincident arrivals get rejected with `resp_error`; that's
    // correct saturation behaviour and the tests account for it.
    ctx.set_slot(client_a, "period_ns", Value::Int(300_000_000));
    ctx.set_slot(client_b, "period_ns", Value::Int(500_000_000));
    ctx.set_slot(worker,   "service_ns", Value::Int(100_000_000));

    // Single Client→Worker wire carries the reply too — the engine's
    // reverse-route fallback walks the same edge backward when the
    // Worker's `emit resp(_) popping to (head(return_path))` targets
    // a client that has no outbound edge to them.
    ctx.wire(client_a, Kind::Client, worker, Kind::Worker);
    ctx.wire(client_b, Kind::Client, worker, Kind::Worker);
}

/// Async submit-and-ack: Client → Queue → Worker → Sink.
///
/// The Client fires `req(color) pushing self`. The Queue's
/// `enqueue_req` rule bumps `len` (so the canvas's "N queued" label
/// ticks up) AND emits `resp(nil) popping to (head(return_path))`.
/// That ack travels back along the *same* Client→Queue wire the
/// request came in on — no hidden reply edge — because the engine
/// reverse-routes when no outbound edge to the target exists.
///
/// A Worker is wired Worker→Queue (the existing pull-pattern
/// wiring injects a `pull(worker_id)` into the Queue at startup).
/// When the queue has depth and a pending consumer, its
/// `wake_tick_flush` emits `packet(color)` to the worker, which
/// services for `service_ns` then emits `packet(color)` to the
/// downstream Sink. The Sink's `count` grows.
fn build_client_queue_worker(ctx: &mut BuildCtx) {
    let client = ctx.place(Kind::Client, 0, Vec2::new(-500.0,  0.0));
    let queue  = ctx.place(Kind::Queue,  0, Vec2::new(-150.0,  0.0));
    let worker = ctx.place(Kind::Worker, 0, Vec2::new( 200.0,  0.0));
    let sink   = ctx.place(Kind::Sink,   0, Vec2::new( 500.0,  0.0));

    // Client emits faster than the worker can drain the queue (5/s
    // vs worker's 10/s → OK, actually worker is faster; switch:
    // client 10/s, worker 80ms → 12.5/s — roughly balanced). Pick
    // rates where queue depth oscillates gently rather than
    // monotonically growing.
    ctx.set_slot(client, "period_ns",  Value::Int(100_000_000));  // 10/s
    ctx.set_slot(worker, "service_ns", Value::Int(120_000_000));  // ~8/s

    ctx.wire(client, Kind::Client, queue,  Kind::Queue);
    ctx.wire(worker, Kind::Worker, queue,  Kind::Queue);
    ctx.wire(worker, Kind::Worker, sink,   Kind::Sink);
}

/// Thundering-herd demo. A crowd of BackoffClients (`N = 20`) all
/// talk to one Worker. Base periods are staggered across a ~40%
/// spread so the initial req stream is desynchronised — you can
/// see that in the probes: `emitted` counters climb at slightly
/// different rates.
///
/// The interesting behaviour appears when the user flips the
/// Worker's `up` toggle 1 → 0 in the inspector:
///   - Every in-flight request fails back as `resp_error`.
///   - Each client's `on_resp_error_seed` rule fires the first time
///     (backoff_ns = 0 → period_ns ≈ 500ms).
///   - Subsequent failures double the backoff in lockstep.
///   - Because there's no jitter, every client's retry lands at
///     the same instant — `1s`, `2s`, `4s`, `8s` after the outage.
///
/// When the user flips `up` back 0 → 1, all clients that happened
/// to land their retry in that window get served simultaneously.
/// The visible effect: a wall of concurrent req packets hitting
/// the Worker at each backoff tier, then silence, then another
/// wall — the textbook thundering-herd shape.
fn build_backoff_herd(ctx: &mut BuildCtx) {
    // 4 columns × 5 rows, centred horizontally on x ≈ -400 so the
    // Worker sits comfortably to the right with room for the many
    // arrows to fan in.
    const COLS: i32 = 4;
    const ROWS: i32 = 5;
    const X_STEP: f32 = 110.0;
    const Y_STEP: f32 = 110.0;
    const ORIGIN_X: f32 = -620.0;

    let worker = ctx.place(Kind::Worker, 0, Vec2::new(200.0, 0.0));
    // Service time is picked to be visually substantial — you want to
    // see each req "dwell" on the Worker before the resp comes out.
    // The Worker's `serve` rule has no busy gate, so many services
    // run concurrently; 200ms work vs 500ms client cadence keeps the
    // animation readable without saturating anything.
    ctx.set_slot(worker, "service_ns", Value::Int(200_000_000));

    for row in 0..ROWS {
        for col in 0..COLS {
            let idx = (row * COLS + col) as usize;
            // Spread positions in a grid, centred vertically.
            let x = ORIGIN_X + col as f32 * X_STEP;
            let y = (row as f32 - (ROWS as f32 - 1.0) * 0.5) * Y_STEP;
            let client = ctx.place(Kind::BackoffClient, 0, Vec2::new(x, y));

            // Stagger base periods from 400ms up to ~580ms in 10ms
            // steps so the initial stream is visibly desynchronised.
            // All within 30–40% of each other — once a failure seeds
            // backoff to `period_ns`, the doublings converge anyway.
            let period_ns = 400_000_000 + (idx as i64) * 10_000_000;
            ctx.set_slot(client, "period_ns", Value::Int(period_ns));

            // Cap backoff at 4s — fast enough to see multiple
            // backoff tiers in a short session before saturation.
            ctx.set_slot(client, "max_backoff_ns", Value::Int(4_000_000_000));

            ctx.wire(client, Kind::BackoffClient, worker, Kind::Worker);
        }
    }
}
