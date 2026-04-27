//! Property tests at the gadget / example-scenario level.
//!
//! Categories:
//!   M — color strictness (Worker/Queue/Sink reject mismatches)
//!   A — accounting/conservation per scenario
//!   V — visual invariants (canvas counts, ordering)
//!
//! Many tests generate random colours, rates, and sim durations to
//! exercise a space the hand-written examples can't. Where a test
//! asserts *post-drainage*, it uses `quiesce_and_drain` to be sure
//! in-flight packets aren't polluting the count.

mod common;

use std::collections::{BTreeMap, HashMap};

use bevy::prelude::*;
use common::make_app;
use proptest::prelude::*;

use flow::event::Event;
use flow::expr::Expr;
use flow::sim::{NodeId, Sim};
use flow::value::Value;

use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::edges::{HiddenEdges, wire_flow_edge};
use flow_bevy::gadgets::{Kind, spawn as spawn_gadget};
use flow_bevy::nodes::NodeCounter;
use flow_bevy::theme::Theme;
use flow_bevy::tool::NodeColors;

// ─────────────────────────────────────────────────────────────
// Shared scenario helpers
// ─────────────────────────────────────────────────────────────

/// Drop a gadget into the sim (no Bevy entity) and return the id.
/// Mirrors `common::spawn_node` but skips visual plumbing so these
/// sim-level tests stay headless.
fn drop_gadget(app: &mut App, kind: Kind, slot: usize, name: &str) -> NodeId {
    let world = app.world_mut();
    let data_color = world.resource::<Theme>().data[slot];
    let mut flow = world.resource_mut::<FlowSim>();
    let id = spawn_gadget(&mut *flow, kind, name, slot);
    drop(flow);
    world.resource_mut::<NodeCounter>().0 += 1;
    if !matches!(kind, Kind::Router) {
        world.resource_mut::<NodeColors>().0.insert(id, data_color);
    }
    id
}

fn wire(app: &mut App, from: NodeId, fk: Kind, to: NodeId, tk: Kind) {
    let world = app.world_mut();
    let mut sys_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<FlowSim>,
        ResMut<EntityMaps>,
        ResMut<HiddenEdges>,
    )> = bevy::ecs::system::SystemState::new(world);
    {
        let (mut commands, mut flow, mut maps, mut hidden) = sys_state.get_mut(world);
        wire_flow_edge(
            &mut *flow,
            &mut maps,
            &mut hidden,
            &mut commands,
            from, Some(fk),
            to,   Some(tk),
        );
        sys_state.apply(world);
    }
}

fn advance_sim_ns(app: &mut App, duration_ns: u64) {
    let world = app.world_mut();
    let mut flow = world.resource_mut::<FlowSim>();
    let target = flow.now_ns + duration_ns;
    flow.run_until(target);
}

/// Quiesce all clients (by name prefix) and iteratively drain the
/// sim until no event is scheduled within the next 1 s. Identical
/// contract to the flow-crate helper of the same name.
fn quiesce_and_drain(app: &mut App) {
    let world = app.world_mut();
    let mut flow = world.resource_mut::<FlowSim>();
    let client_ids: Vec<NodeId> = flow.nodes.iter()
        .filter(|(_, n)| n.name.starts_with("Client_"))
        .map(|(id, _)| *id)
        .collect();
    for c in client_ids {
        flow.nodes.get_mut(&c).unwrap().slots
            .insert("period_ns".into(), Value::Int(i64::MAX / 4));
    }
    for _ in 0..1000 {
        let Some(next) = flow.next_event_time_ns() else { break; };
        if next > flow.now_ns.saturating_add(1_000_000_000) { break; }
        flow.run_until(next.saturating_add(1_000_000));
    }
}

fn slot_int(sim: &Sim, nid: NodeId, slot: &str) -> i64 {
    match sim.nodes[&nid].slots.get(slot) {
        Some(Value::Int(i)) => *i,
        _ => 0,
    }
}

// ─────────────────────────────────────────────────────────────
// M1 — Worker strict-colour
// ─────────────────────────────────────────────────────────────

/// Build: one Client at `client_slot`, one Worker at `worker_slot`.
/// Client→Worker edge. Returns (app, client, worker).
fn build_client_worker(client_slot: usize, worker_slot: usize) -> (App, NodeId, NodeId) {
    let mut app = make_app();
    let client = drop_gadget(&mut app, Kind::Client, client_slot, "Client_1");
    let worker = drop_gadget(&mut app, Kind::Worker, worker_slot, "Worker_1");
    // Slow the client down so the test runs cheaply.
    app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&client)
        .unwrap().slots.insert("period_ns".into(), Value::Int(200_000_000));
    wire(&mut app, client, Kind::Client, worker, Kind::Worker);
    (app, client, worker)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 30, ..ProptestConfig::default() })]

    /// M1 — Worker serves ONLY reqs whose colour matches its own.
    /// Mismatched reqs are rejected with `color_mismatch` errors
    /// instead of producing resp's.
    #[test]
    fn m1_worker_strict_color(
        client_slot in 0usize..3,
        worker_slot in 0usize..3,
        duration_ms in 500u64..2500,
    ) {
        let (mut app, client, worker) = build_client_worker(client_slot, worker_slot);
        advance_sim_ns(&mut app, duration_ms * 1_000_000);
        quiesce_and_drain(&mut app);

        let sim = &app.world().resource::<FlowSim>();
        let emitted   = slot_int(sim, client, "emitted");
        let completed = slot_int(sim, client, "completed");
        let served    = slot_int(sim, worker, "served");
        let errors = sim.error_counts.get("color_mismatch").copied().unwrap_or(0) as i64;

        if client_slot == worker_slot {
            // Matching colour: every req gets served, every client
            // sees a reply, no color errors.
            prop_assert_eq!(served, emitted,
                "matching color: worker should serve every req (emitted={}, served={})",
                emitted, served);
            prop_assert_eq!(completed, emitted,
                "matching color: client should get every reply (emitted={}, completed={})",
                emitted, completed);
            prop_assert_eq!(errors, 0,
                "matching color: no color_mismatch errors (got {})", errors);
        } else {
            // Mismatched colour: NO reqs are served, client gets
            // zero replies, one color_mismatch per req.
            prop_assert_eq!(served, 0,
                "mismatched color: worker should reject all reqs (served={})", served);
            prop_assert_eq!(completed, 0,
                "mismatched color: client should get no replies (completed={})", completed);
            prop_assert_eq!(errors, emitted,
                "mismatched color: one color_mismatch per req (emitted={}, errors={})",
                emitted, errors);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// M1 — Queue strict-colour
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 30, ..ProptestConfig::default() })]

    /// M1 — Queue only enqueues req's whose colour matches.
    /// Mismatched reqs → color_mismatch, `len` stays 0, no ack
    /// reaches the client.
    #[test]
    fn m1_queue_strict_color(
        client_slot in 0usize..3,
        queue_slot in 0usize..3,
        duration_ms in 500u64..2500,
    ) {
        let mut app = make_app();
        let client = drop_gadget(&mut app, Kind::Client, client_slot, "Client_1");
        let queue = drop_gadget(&mut app, Kind::Queue, queue_slot, "Queue_1");
        app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&client)
            .unwrap().slots.insert("period_ns".into(), Value::Int(200_000_000));
        wire(&mut app, client, Kind::Client, queue, Kind::Queue);

        advance_sim_ns(&mut app, duration_ms * 1_000_000);
        quiesce_and_drain(&mut app);

        let sim = &app.world().resource::<FlowSim>();
        let emitted   = slot_int(sim, client, "emitted");
        let completed = slot_int(sim, client, "completed");
        let len       = slot_int(sim, queue, "len");
        let errors = sim.error_counts.get("color_mismatch").copied().unwrap_or(0) as i64;

        if client_slot == queue_slot {
            prop_assert_eq!(completed, emitted,
                "matching: every req acked (emitted={}, completed={})",
                emitted, completed);
            prop_assert_eq!(len, emitted,
                "matching: every req left in queue (emitted={}, len={})",
                emitted, len);
            prop_assert_eq!(errors, 0);
        } else {
            prop_assert_eq!(completed, 0,
                "mismatched: no acks (completed={})", completed);
            prop_assert_eq!(len, 0,
                "mismatched: queue stays empty (len={})", len);
            prop_assert_eq!(errors, emitted,
                "mismatched: one color_mismatch per req (emitted={}, errors={})",
                emitted, errors);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// A — per-scenario conservation
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, ..ProptestConfig::default() })]

    /// A1 — ClientWorker: post-drainage, `client.emitted ==
    /// completed == worker.served`, `in_flight == 0`.
    #[test]
    fn a1_client_worker_conservation(
        slot in 0usize..3,
        period_ms in 50u64..500,
        duration_ms in 500u64..2500,
    ) {
        let (mut app, client, worker) = build_client_worker(slot, slot);
        app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&client)
            .unwrap().slots.insert("period_ns".into(), Value::Int((period_ms * 1_000_000) as i64));
        advance_sim_ns(&mut app, duration_ms * 1_000_000);
        quiesce_and_drain(&mut app);

        let sim = &app.world().resource::<FlowSim>();
        let emitted   = slot_int(sim, client, "emitted");
        let completed = slot_int(sim, client, "completed");
        let in_flight = slot_int(sim, client, "in_flight");
        let served    = slot_int(sim, worker, "served");

        prop_assert_eq!(emitted, completed,
            "emitted {} != completed {} (client)", emitted, completed);
        prop_assert_eq!(in_flight, 0,
            "in_flight should be 0 after drain, got {}", in_flight);
        prop_assert_eq!(served, emitted,
            "worker.served {} != client.emitted {}", served, emitted);
        prop_assert!(emitted > 0, "test misconfigured: nothing emitted");
    }
}

// ─────────────────────────────────────────────────────────────
// R4 — no cross-talk: per-client resp count matches completed
// ─────────────────────────────────────────────────────────────

fn build_n_clients_one_worker(
    app: &mut App,
    n: usize,
    period_ms_each: &[u64],
    slot: usize,
) -> (Vec<NodeId>, NodeId) {
    let worker = drop_gadget(app, Kind::Worker, slot, "Worker_1");
    let mut clients = Vec::with_capacity(n);
    for i in 0..n {
        let name = format!("Client_{}", i + 1);
        let c = drop_gadget(app, Kind::Client, slot, &name);
        app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&c)
            .unwrap().slots.insert("period_ns".into(),
                Value::Int((period_ms_each[i] * 1_000_000) as i64));
        wire(app, c, Kind::Client, worker, Kind::Worker);
        clients.push(c);
    }
    (clients, worker)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, ..ProptestConfig::default() })]

    /// R4 — N clients sharing a worker, each gets exactly its own
    /// reqs' responses back. With the Worker's busy gate, some reqs
    /// coming in while the worker is still servicing another get
    /// rejected with `resp_error` (client increments `failed`) —
    /// that's correct saturation behaviour. The no-cross-talk
    /// property is: per client, successful resp's from the worker
    /// match `completed`, and resp_error's match `failed`; no
    /// response of either kind lands on the wrong client.
    #[test]
    fn r4_n_clients_no_cross_talk(
        n in 2usize..5,
        periods in prop::collection::vec(100u64..400u64, 2..5),
        duration_ms in 500u64..2500,
    ) {
        // Proptest can hand us mismatched lengths — clip.
        let n = n.min(periods.len());
        prop_assume!(n >= 2);

        let mut app = make_app();
        let (clients, worker) = build_n_clients_one_worker(&mut app, n, &periods, 0);
        advance_sim_ns(&mut app, duration_ms * 1_000_000);
        quiesce_and_drain(&mut app);

        let sim = &app.world().resource::<FlowSim>();

        // Every resp / resp_error emitted by the worker targeted
        // someone; count per client, split by tag.
        let mut resp_ok_to: HashMap<NodeId, i64> = HashMap::new();
        let mut resp_err_to: HashMap<NodeId, i64> = HashMap::new();
        for ev in sim.log.iter() {
            if let Event::PacketEmitted { from, to, payload, .. } = ev {
                if *from != worker { continue; }
                if let Value::Variant { tag, .. } = payload {
                    match tag.as_str() {
                        "resp"       => *resp_ok_to.entry(*to).or_insert(0) += 1,
                        "resp_error" => *resp_err_to.entry(*to).or_insert(0) += 1,
                        _ => {}
                    }
                }
            }
        }

        for c in &clients {
            let emitted   = slot_int(sim, *c, "emitted");
            let completed = slot_int(sim, *c, "completed");
            let failed    = slot_int(sim, *c, "failed");
            let in_flight = slot_int(sim, *c, "in_flight");
            let ok_here   = resp_ok_to.get(c).copied().unwrap_or(0);
            let err_here  = resp_err_to.get(c).copied().unwrap_or(0);

            prop_assert_eq!(in_flight, 0,
                "client {:?}: in_flight should drain to 0, got {}", c, in_flight);
            prop_assert_eq!(completed + failed, emitted,
                "client {:?}: completed ({}) + failed ({}) != emitted ({})",
                c, completed, failed, emitted);
            prop_assert_eq!(ok_here, completed,
                "client {:?}: worker emitted {} resp's to it, completed={} — cross-talk",
                c, ok_here, completed);
            prop_assert_eq!(err_here, failed,
                "client {:?}: worker emitted {} resp_error's to it, failed={} — cross-talk",
                c, err_here, failed);
        }

        // `served` ticks only on successful `serve` — should equal
        // the sum of each client's `completed`.
        let served = slot_int(sim, worker, "served");
        let total_completed: i64 =
            clients.iter().map(|c| slot_int(sim, *c, "completed")).sum();
        prop_assert_eq!(served, total_completed,
            "worker.served ({}) != sum(client.completed) ({})",
            served, total_completed);
    }
}

// ─────────────────────────────────────────────────────────────
// C2 — visual-layer causality sanity: the SIM's event log should
// always show the req consume before the corresponding resp
// emit. If this test fails, the formalism is broken; if it passes
// and the user still sees "reply before request" on the canvas,
// it's a pure visualization issue (see reply_visual.rs).
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 30, ..ProptestConfig::default() })]

    /// At the event-log level, the ordering invariant: within a
    /// rule firing, the consume precedes the emit of the resp the
    /// consume caused. No "worker emits before it consumed the
    /// triggering req."
    #[test]
    fn c2_event_log_consume_before_emit(
        slot in 0usize..3,
        period_ms in 50u64..400,
        duration_ms in 500u64..2500,
    ) {
        let (mut app, _client, worker) = build_client_worker(slot, slot);
        app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&_client)
            .unwrap().slots.insert("period_ns".into(), Value::Int((period_ms * 1_000_000) as i64));
        advance_sim_ns(&mut app, duration_ms * 1_000_000);
        quiesce_and_drain(&mut app);

        let sim = &app.world().resource::<FlowSim>();
        let mut consumed: u64 = 0;
        let mut emitted:  u64 = 0;
        for ev in sim.log.iter() {
            match ev {
                Event::PacketConsumed { by, rule, .. } if *by == worker && rule == "serve" => {
                    consumed += 1;
                    prop_assert!(emitted <= consumed);
                }
                Event::PacketEmitted { from, payload, .. } if *from == worker => {
                    if let Value::Variant { tag, .. } = payload {
                        if tag == "resp" {
                            emitted += 1;
                            prop_assert!(emitted <= consumed,
                                "resp emit #{} happened before its matching consume (only {} consumes by this point)",
                                emitted, consumed);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// S3 — after LoadExample, error_counts starts empty
// ─────────────────────────────────────────────────────────────

#[test]
fn s3_load_example_resets_error_counts() {
    use flow_bevy::examples::{Example, LoadExample};

    let mut app = make_app();
    // Pre-dirty: inject a color mismatch.
    let client = drop_gadget(&mut app, Kind::Client, 0, "Client_1");
    let worker = drop_gadget(&mut app, Kind::Worker, 1, "Worker_1");
    app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&client)
        .unwrap().slots.insert("period_ns".into(), Value::Int(100_000_000));
    wire(&mut app, client, Kind::Client, worker, Kind::Worker);
    advance_sim_ns(&mut app, 500_000_000);

    assert!(!app.world().resource::<FlowSim>().error_counts.is_empty(),
        "pre-condition: we expected mismatch errors to be recorded");

    // Load an example; error_counts should be cleared.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(Example::ClientWorker));
    app.update();
    app.update();

    let sim = &app.world().resource::<FlowSim>();
    assert!(
        sim.error_counts.is_empty(),
        "LoadExample should reset error_counts, got {:?}",
        sim.error_counts
    );
}

// ─────────────────────────────────────────────────────────────
// R5 — two-edge Client→Router→Worker round-trip
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, ..ProptestConfig::default() })]

    /// Client → Router → Worker with ONLY those two edges (no
    /// hidden / manual Worker→Client reply edge). The Router's
    /// forward_req pushes self and forward_resp pops itself, so the
    /// reply walks back via two reverse-route hops. Invariant:
    /// every req eventually produces a resp delivered to the
    /// Client, no runtime errors, no cross-talk if we scale to
    /// multiple clients.
    #[test]
    fn r5_two_edge_router_round_trip(
        slot in 0usize..3,
        period_ms in 100u64..400,
        duration_ms in 500u64..2500,
    ) {
        let mut app = make_app();
        let client = drop_gadget(&mut app, Kind::Client, slot, "Client_1");
        let router = drop_gadget(&mut app, Kind::Router, slot, "Router_1");
        let worker = drop_gadget(&mut app, Kind::Worker, slot, "Worker_1");
        app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&client)
            .unwrap().slots.insert("period_ns".into(),
                Value::Int((period_ms * 1_000_000) as i64));

        wire(&mut app, client, Kind::Client, router, Kind::Router);
        wire(&mut app, router, Kind::Router, worker, Kind::Worker);

        advance_sim_ns(&mut app, duration_ms * 1_000_000);
        quiesce_and_drain(&mut app);

        let sim = &app.world().resource::<FlowSim>();
        let emitted   = slot_int(sim, client, "emitted");
        let completed = slot_int(sim, client, "completed");
        let served    = slot_int(sim, worker, "served");

        prop_assert_eq!(emitted, completed,
            "emitted {} != completed {} — reply never got back through router",
            emitted, completed);
        prop_assert_eq!(served, emitted,
            "worker served {} but client emitted {} — reqs dropped at router",
            served, emitted);
        prop_assert!(emitted > 0, "scenario didn't run");
        prop_assert!(sim.error_counts.is_empty(),
            "unexpected errors: {:?}", sim.error_counts);

        // The sim graph has exactly 2 CROSS-NODE edges (ignoring
        // self-loops the gadgets use for tick / service timers).
        let cross_edges = sim.edges.values().filter(|e| e.from != e.to).count();
        prop_assert_eq!(cross_edges, 2,
            "scenario has {} cross-node edges, expected 2 (no triangle reply edge)",
            cross_edges);
    }
}

// ─────────────────────────────────────────────────────────────
// Cross-talk through router (N clients sharing one router +
// worker). Each client's response must reach only that client.
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 15, ..ProptestConfig::default() })]

    /// N clients all go through a shared Router to one Worker.
    /// Every resp that comes back must land on the client that
    /// originated the request, not another. Router must pop its
    /// own frame correctly on each response.
    #[test]
    fn r5_multi_client_through_router_no_cross_talk(
        n in 2usize..4,
        periods in prop::collection::vec(150u64..400u64, 2..4),
        duration_ms in 500u64..2000,
    ) {
        let n = n.min(periods.len());
        prop_assume!(n >= 2);

        let mut app = make_app();
        let router = drop_gadget(&mut app, Kind::Router, 0, "Router_1");
        let worker = drop_gadget(&mut app, Kind::Worker, 0, "Worker_1");
        wire(&mut app, router, Kind::Router, worker, Kind::Worker);

        let mut clients = Vec::with_capacity(n);
        for i in 0..n {
            let name = format!("Client_{}", i + 1);
            let c = drop_gadget(&mut app, Kind::Client, 0, &name);
            app.world_mut().resource_mut::<FlowSim>().nodes.get_mut(&c)
                .unwrap().slots.insert("period_ns".into(),
                    Value::Int((periods[i] * 1_000_000) as i64));
            wire(&mut app, c, Kind::Client, router, Kind::Router);
            clients.push(c);
        }

        advance_sim_ns(&mut app, duration_ms * 1_000_000);
        quiesce_and_drain(&mut app);

        let sim = &app.world().resource::<FlowSim>();
        // Worker's bounded accept queue can legitimately fire
        // `worker_full` + `request_failed` under saturation —
        // those aren't cross-talk. Assert no OTHER error kinds
        // appeared.
        let unexpected: BTreeMap<_, _> = sim.error_counts.iter()
            .filter(|(k, _)| k.as_str() != "worker_full" && k.as_str() != "request_failed")
            .collect();
        prop_assert!(unexpected.is_empty(), "unexpected errors: {:?}", unexpected);

        // Count resp's and resp_error's the Router emitted per
        // client (Router is the direct sender via `forward_resp` /
        // `forward_resp_error`).
        let mut resp_ok_to: HashMap<NodeId, i64> = HashMap::new();
        let mut resp_err_to: HashMap<NodeId, i64> = HashMap::new();
        for ev in sim.log.iter() {
            if let Event::PacketEmitted { from, to, payload, .. } = ev {
                if *from != router { continue; }
                if let Value::Variant { tag, .. } = payload {
                    match tag.as_str() {
                        "resp"       => *resp_ok_to.entry(*to).or_insert(0) += 1,
                        "resp_error" => *resp_err_to.entry(*to).or_insert(0) += 1,
                        _ => {}
                    }
                }
            }
        }
        for c in &clients {
            let emitted   = slot_int(sim, *c, "emitted");
            let completed = slot_int(sim, *c, "completed");
            let failed    = slot_int(sim, *c, "failed");
            let in_flight = slot_int(sim, *c, "in_flight");
            let ok_here   = resp_ok_to.get(c).copied().unwrap_or(0);
            let err_here  = resp_err_to.get(c).copied().unwrap_or(0);

            prop_assert_eq!(in_flight, 0,
                "client {:?}: in_flight should drain to 0, got {}", c, in_flight);
            prop_assert_eq!(completed + failed, emitted,
                "client {:?}: completed + failed != emitted", c);
            prop_assert_eq!(ok_here, completed,
                "client {:?}: router relayed {} resp's to it, completed={} — cross-talk",
                c, ok_here, completed);
            prop_assert_eq!(err_here, failed,
                "client {:?}: router relayed {} resp_error's to it, failed={} — cross-talk",
                c, err_here, failed);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Q — per-node packet book-balance
// ─────────────────────────────────────────────────────────────

/// For each Queue in a running scenario, count:
///   - inbound `packet(_)` events (things arriving to be
///     enqueued),
///   - outbound `packet(_)` events (things dispatched to a
///     consumer via on_pull / wake_tick_flush).
///
/// Post-drainage, `out + final_len == in + ack_failures`. Any
/// other ratio means the Queue is conjuring packets out of thin
/// air (or losing them). This catches what proptest missed when
/// the user reported "extra thing coming out of queue_7."
#[test]
fn q_queue_packet_conservation_three_lane() {
    let mut app = make_app();
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<flow_bevy::examples::LoadExample>>()
        .write(flow_bevy::examples::LoadExample(
            flow_bevy::examples::Example::ThreeLaneFanout,
        ));
    app.update();
    app.update();

    advance_sim_ns(&mut app, 3_000_000_000);
    quiesce_and_drain(&mut app);

    let sim = &app.world().resource::<FlowSim>();
    let queue_ids: Vec<NodeId> = sim.nodes.iter()
        .filter(|(_, n)| n.name.starts_with("Queue_"))
        .map(|(id, _)| *id)
        .collect();

    for q in queue_ids {
        let mut packets_in  = 0i64;
        let mut packets_out = 0i64;
        for ev in sim.log.iter() {
            if let Event::PacketEmitted { from, to, payload, .. } = ev {
                if let Value::Variant { tag, .. } = payload {
                    if tag != "packet" { continue; }
                } else { continue; }
                if *to == q   { packets_in += 1;  }
                if *from == q { packets_out += 1; }
            }
        }
        let final_len = slot_int(sim, q, "len");
        let qname = sim.nodes[&q].name.clone();
        assert_eq!(
            packets_in, packets_out + final_len,
            "queue {} book-balance broken: packets_in={}, packets_out={}, \
             final_len={} — expected in == out + len (every packet is \
             either dispatched to a consumer or still queued)",
            qname, packets_in, packets_out, final_len,
        );
    }
}

// Dummy to keep the imports referenced.
#[allow(dead_code)]
fn _unused() { let _ = BTreeMap::<String, Expr>::new(); }
