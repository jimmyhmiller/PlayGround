//! Auto Scaling Group — end-to-end sim behaviour for the top-level,
//! pluggable design:
//!   * the ASG is one leaf node that spawns real gadget instances
//!     (its `template`, default `WorkerComposite`) at the TOP LEVEL;
//!   * it load-balances by dispatching `req(pushing self)` and forwards
//!     the worker's reverse-routed `resp` to its downstream sink;
//!   * it grows/shrinks the fleet from its own in-flight bookkeeping;
//!   * `paused` freezes auto-scaling and `nudge` adds/removes by hand.

use flow::sim::Sim;
use flow::{NodeId, Value};
use flow_bevy::gadgets::GADGETS_DSL;

fn sim_with_gadgets() -> Sim {
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, GADGETS_DSL)
        .expect("gadget DSL (incl. autoscaling) must compile");
    flow_bevy::gadgets::install_back_compat_aliases(&mut sim);
    sim
}

/// Count live worker instances (top-level `WorkerComposite` shims spawned
/// by the ASG — their names look like `WorkerComposite_<n>`).
fn worker_count(sim: &Sim) -> usize {
    sim.nodes
        .values()
        .filter(|n| {
            n.name.starts_with("WorkerComposite_") && !n.name.contains("::")
        })
        .count()
}

fn set_slot(sim: &mut Sim, nid: NodeId, slot: &str, v: i64) {
    sim.nodes.get_mut(&nid).unwrap().slots.insert(slot.into(), Value::Int(v));
}

/// Build `Generator → ASG → Sink` entirely in the sim and return the ids.
fn build(sim: &mut Sim, gen_period_ns: i64) -> (NodeId, NodeId, NodeId) {
    let asg = sim.instantiate("AutoScalingGroup", "ASG").unwrap();
    let source = sim.instantiate("GeneratorComposite", "Gen").unwrap();
    let sink = sim.instantiate("SinkComposite", "Sink").unwrap();
    // Tune via inner Tick + ASG slots.
    if let Some(tick) = sim.node_by_name("Gen::T") {
        set_slot(sim, tick, "period_ns", gen_period_ns);
    }
    set_slot(sim, asg, "tick_ns", 20_000_000);
    set_slot(sim, asg, "max_workers", 6);
    set_slot(sim, asg, "min_workers", 1);
    // Generator `output` out-port → ASG; ASG → Sink `input` in-port.
    sim.add_edge_ports(source, Some("output".into()), asg, None, flow::Expr::int(1_000_000));
    sim.add_edge_ports(asg, None, sink, Some("input".into()), flow::Expr::int(1_000_000));
    (source, asg, sink)
}

#[test]
fn asg_spawns_real_workers_at_top_level_and_scales() {
    let mut sim = sim_with_gadgets();
    let (_gen, _asg, _sink) = build(&mut sim, 25_000_000); // 40/s
    sim.run_until(4_000_000_000);

    let workers = worker_count(&sim);
    assert!(
        workers > 1,
        "fleet should scale up under load, got {workers} workers"
    );
    assert!(workers <= 6, "must respect max_workers, got {workers}");

    // Workers are real WorkerComposite instances at the top level (no
    // `::` prefix on the shim) — i.e. the same gadget you'd drop.
    let a_worker = sim
        .nodes
        .values()
        .find(|n| n.name.starts_with("WorkerComposite_") && !n.name.contains("::"))
        .expect("a worker shim");
    assert_eq!(
        sim.compound_class_of.get(&a_worker.id).map(String::as_str),
        Some("WorkerComposite"),
        "spawned worker must be the real WorkerComposite"
    );

    // Replies must come back: `completed` proves the worker's `resp`
    // reverse-routed to the ASG (without it `inflight` would grow without
    // bound and the fleet would pin to `max` forever).
    let completed = match sim.read_slot_resolved(_asg, "completed") {
        Some(Value::Int(c)) => *c,
        other => panic!("completed slot: {:?}", other),
    };
    assert!(completed > 0, "workers' replies must reach the ASG (completed={completed})");
}

/// No worker inner node (`WorkerComposite_n::…`) should ever be orphaned
/// at the top level — despawning a worker must take its whole interior.
fn orphan_inner_count(sim: &Sim) -> usize {
    sim.nodes
        .values()
        .filter(|n| {
            n.name.starts_with("WorkerComposite_")
                && n.name.contains("::")
                && !sim
                    .nodes
                    .values()
                    .any(|s| s.is_compound() && n.name.starts_with(&format!("{}::", s.name)))
        })
        .count()
}

/// Force every live worker's inner Service to be slow (simulates the
/// user dragging worker service-time up), so backlog builds in worker
/// inboxes — the condition that exposes the despawn inflight-leak.
fn make_workers_slow(sim: &mut Sim, service_ns: i64) {
    let svs: Vec<NodeId> = sim
        .nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with("WorkerComposite_") && n.name.ends_with("::Sv"))
        .map(|(id, _)| *id)
        .collect();
    for id in svs {
        sim.nodes.get_mut(&id).unwrap().slots.insert("service_ns".into(), Value::Int(service_ns));
    }
}

fn slot(sim: &Sim, asg: NodeId, s: &str) -> i64 {
    match sim.read_slot_resolved(asg, s) {
        Some(Value::Int(i)) => *i,
        _ => -1,
    }
}

#[test]
fn despawn_during_backlog_does_not_inflate_load() {
    // Reproduces "never scales down": with backlog, despawning a worker
    // used to leave its outstanding requests counted in `inflight` while
    // `count` dropped — inflating `load` and stalling all further
    // scale-down. The proportional `inflight` adjustment keeps `load`
    // stable across a despawn.
    let mut sim = sim_with_gadgets();
    let (_source, asg, _sink) = build(&mut sim, 8_000_000); // ~125/s overload
    set_slot(&mut sim, asg, "max_workers", 8);
    for _ in 0..25 {
        sim.run_until(sim.now_ns + 100_000_000);
        make_workers_slow(&mut sim, 300_000_000);
    }
    let load_before = slot(&sim, asg, "load");
    let count_before = worker_count(&sim);
    assert!(count_before >= 3, "need a multi-worker backlog, got {count_before}");
    assert!(load_before > 0, "expected real backlog load, got {load_before}");

    // Pause auto-scaling so only the manual nudge moves the fleet (else
    // scale_up immediately re-adds what we shed), then shed one worker
    // while the backlog is still high.
    sim.nodes.get_mut(&asg).unwrap().slots.insert("paused".into(), Value::Bool(true));
    sim.nodes.get_mut(&asg).unwrap().slots.insert("nudge".into(), Value::Int(-1));
    sim.run_until(sim.now_ns + 60_000_000); // a few control ticks

    assert_eq!(worker_count(&sim), count_before - 1, "one worker should be shed");
    let load_after = slot(&sim, asg, "load");
    // The leak would make load jump UP after the despawn. With the fix it
    // stays in the same ballpark (allow some drift from ongoing traffic).
    assert!(
        load_after <= load_before + (load_before / 4) + 5,
        "load must not spike after a despawn (leak): before={load_before}, after={load_after}"
    );
}

#[test]
fn fleet_recovers_to_min_after_overload() {
    // End-to-end: overload to max, then go idle — the fleet must return
    // all the way to min_workers (not stall partway).
    let mut sim = sim_with_gadgets();
    let (_source, asg, _sink) = build(&mut sim, 8_000_000); // ~125/s
    set_slot(&mut sim, asg, "max_workers", 8);
    sim.run_until(2_000_000_000);
    assert!(worker_count(&sim) > 2, "should scale up under overload");

    // Go essentially idle.
    if let Some(tick) = sim.node_by_name("Gen::T") {
        set_slot(&mut sim, tick, "period_ns", 2_000_000_000); // 0.5/s
    }
    sim.run_until(sim.now_ns + 6_000_000_000);
    assert_eq!(
        worker_count(&sim),
        1,
        "idle fleet must return to min_workers, not stall"
    );
}

#[test]
fn fleet_tracks_load_up_and_down() {
    // The whole point: the fleet should grow under load and SHRINK when
    // the load drops. This is what breaks if replies don't come back.
    let mut sim = sim_with_gadgets();
    let (_source, _asg, _sink) = build(&mut sim, 25_000_000); // 40/s — hot
    if let Some(tick) = sim.node_by_name("Gen::T") {
        set_slot(&mut sim, tick, "period_ns", 25_000_000);
    }
    sim.run_until(4_000_000_000);
    let hot = worker_count(&sim);
    assert!(hot > 1, "fleet should grow under heavy load, got {hot}");

    // Drop the load way down; the fleet should shed workers.
    if let Some(tick) = sim.node_by_name("Gen::T") {
        set_slot(&mut sim, tick, "period_ns", 4_000_000_000); // ~0.25/s — idle
    }
    sim.run_until(sim.now_ns + 8_000_000_000);
    let cool = worker_count(&sim);
    assert!(
        cool < hot,
        "fleet should scale DOWN when load drops: hot={hot}, cool={cool}"
    );
    // Scaling down must not leave a worker's inner nodes loose on the
    // top-level canvas.
    assert_eq!(
        orphan_inner_count(&sim),
        0,
        "despawning workers must take their whole compound interior"
    );
}

#[test]
fn completions_reach_the_sink() {
    // End-to-end throughput: the ASG must forward worker completions to
    // its downstream sink (which counts them).
    let mut sim = sim_with_gadgets();
    let (_source, _asg, sink) = build(&mut sim, 25_000_000);
    sim.run_until(2_000_000_000);
    let absorbed = match sim.read_slot_resolved(sink, "count") {
        Some(Value::Int(c)) => *c,
        other => panic!("sink count: {:?}", other),
    };
    assert!(absorbed > 0, "sink should absorb completed work, got {absorbed}");
}

#[test]
fn asg_dispatch_edges_render_at_top_level() {
    // Each spawned worker has a top-level edge from the ASG (so the
    // fan-out is visible on the canvas, not hidden in a compound).
    let mut sim = sim_with_gadgets();
    let (_gen, asg, _sink) = build(&mut sim, 25_000_000);
    sim.run_until(2_000_000_000);

    let worker = sim
        .nodes
        .values()
        .find(|n| n.name.starts_with("WorkerComposite_") && !n.name.contains("::"))
        .map(|n| n.id)
        .expect("a worker");
    let has_dispatch_edge = sim.edges.values().any(|e| e.from == asg && e.to == worker);
    assert!(has_dispatch_edge, "expected a top-level ASG → worker edge");
    // The worker shim is top-level (membership is name-prefix based).
    assert!(
        !sim.nodes[&worker].name.contains("::"),
        "worker must be a top-level node"
    );
}

#[test]
fn paused_freezes_the_fleet() {
    let mut sim = sim_with_gadgets();
    let (_gen, asg, _sink) = build(&mut sim, 25_000_000);
    // Let it reach min, then pause.
    sim.run_until(500_000_000);
    sim.nodes.get_mut(&asg).unwrap().slots.insert("paused".into(), Value::Bool(true));
    let frozen = worker_count(&sim);
    sim.run_until(5_000_000_000);
    assert_eq!(
        worker_count(&sim),
        frozen,
        "paused ASG must not change fleet size"
    );
}

#[test]
fn manual_nudge_adds_and_removes_workers() {
    let mut sim = sim_with_gadgets();
    // No load; pause auto-scaling so only nudges move the fleet.
    let (_gen, asg, _sink) = build(&mut sim, 10_000_000_000); // ~idle generator
    sim.run_until(300_000_000);
    sim.nodes.get_mut(&asg).unwrap().slots.insert("paused".into(), Value::Bool(true));
    let base = worker_count(&sim);

    // +2 nudge → two more workers.
    set_slot(&mut sim, asg, "nudge", 2);
    sim.run_until(sim.now_ns + 500_000_000);
    let after_up = worker_count(&sim);
    assert!(
        after_up >= base + 1,
        "nudge up should add workers: base {base}, after {after_up}"
    );

    // −1 nudge → one fewer.
    set_slot(&mut sim, asg, "nudge", -1);
    sim.run_until(sim.now_ns + 500_000_000);
    assert!(
        worker_count(&sim) < after_up,
        "nudge down should remove a worker"
    );
}

#[test]
fn pluggable_template_spawns_the_named_class() {
    // Point the ASG at a different class and confirm it clones THAT.
    let mut sim = sim_with_gadgets();
    let asg = sim.instantiate("AutoScalingGroup", "ASG").unwrap();
    sim.nodes
        .get_mut(&asg)
        .unwrap()
        .slots
        .insert("worker_class".into(), Value::Str("QueueComposite".into()));
    set_slot(&mut sim, asg, "tick_ns", 20_000_000);
    set_slot(&mut sim, asg, "min_workers", 2);
    sim.run_until(500_000_000);

    let queues = sim
        .nodes
        .values()
        .filter(|n| n.name.starts_with("QueueComposite_") && !n.name.contains("::"))
        .count();
    assert!(
        queues >= 2,
        "ASG should have cloned its `template` (QueueComposite), got {queues}"
    );
    assert!(
        sim.error_counts.is_empty(),
        "no runtime errors expected, got {:?}",
        sim.error_counts
    );
}

#[test]
fn autoscaling_whiteboard_example_loads_and_scales() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/autoscaling.whiteboard");
    let mut canvas = flow_bevy::canvas::load_canvas(&dir, 0)
        .expect("autoscaling.whiteboard should load");

    canvas.sim.run_until(3_000_000_000);
    let workers = canvas
        .sim
        .nodes
        .values()
        .filter(|n| n.name.starts_with("WorkerComposite_") && !n.name.contains("::"))
        .count();
    assert!(
        workers > 1,
        "the whiteboard fleet should scale up under load, got {workers}"
    );
    assert!(
        canvas.sim.error_counts.is_empty(),
        "no runtime errors expected, got {:?}",
        canvas.sim.error_counts
    );
}
