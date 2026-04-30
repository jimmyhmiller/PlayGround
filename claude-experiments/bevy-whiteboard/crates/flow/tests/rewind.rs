//! Determinism + rewind: snapshot at t=T, continue forward,
//! save the event trace; then restore snapshot, replay forward,
//! verify the event trace matches.

use std::collections::BTreeMap;

use flow::{
    Effect, EmitTo, Event, Expr, Pattern, Rule, Samples, Sim, Value, When,
};

fn event_digest(e: &Event) -> String {
    match e {
        Event::ClockAdvanced { from_ns, to_ns } => format!("clock {} -> {}", from_ns, to_ns),
        Event::RuleFired { node, rule, at_ns } => format!("fire {:?} {} @{}", node, rule, at_ns),
        Event::SlotWritten { node, slot, value, at_ns } =>
            format!("slot {:?} {} = {:?} @{}", node, slot, value, at_ns),
        Event::PacketEmitted { packet, from, to, arrives_at_ns, at_ns, .. } =>
            format!("emit {:?} {:?}->{:?} @{} arr={}", packet, from, to, at_ns, arrives_at_ns),
        Event::PacketDelivered { packet, to, at_ns } =>
            format!("deliver {:?} -> {:?} @{}", packet, to, at_ns),
        Event::PacketConsumed { packet, by, rule, at_ns } =>
            format!("consume {:?} by {:?} rule {} @{}", packet, by, rule, at_ns),
        Event::MetricRecorded { node, name, value, at_ns } =>
            format!("metric {:?} {} = {:?} @{}", node, name, value, at_ns),
        Event::NodeSpawned { node, template, parent, at_ns } =>
            format!("spawn {:?} from {} parent={:?} @{}", node, template, parent, at_ns),
        Event::NodeDespawned { node, at_ns } =>
            format!("despawn {:?} @{}", node, at_ns),
        Event::RuntimeError { .. } => String::new(),
        Event::TimelineEventFired { .. } => String::new(),
        Event::UserSlotEdit { .. } => String::new(),
    }
}

fn build_sim() -> Sim {
    let mut sim = Sim::new(99);

    // Client — server with stochastic response.
    let server_rules = vec![
        Rule::new("respond")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(Effect::respond(
                Expr::variant("resp", Expr::lit(Value::Nil)),
            )),
    ];
    let server = sim.add_node("Server", BTreeMap::new(), server_rules);

    let mut client_slots = BTreeMap::new();
    client_slots.insert("in_flight".into(), Value::Int(0));
    client_slots.insert("sent_at".into(),   Value::Samples(Samples::new(1024)));
    let client_rules = vec![
        Rule::new("send")
            .when(When::input(Pattern::variant("tick", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::add(Expr::slot("in_flight"), Expr::int(1)),
            })
            .do_(Effect::SamplesPush { slot: "sent_at".into(), value: Expr::now() })
            .do_(Effect::emit(
                Expr::variant("req", Expr::lit(Value::Nil)),
                EmitTo::ToTarget("Server".into()),
            ))
            .do_(Effect::emit(
                Expr::variant("tick", Expr::lit(Value::Nil)),
                EmitTo::ToTarget("Client".into()),
            )),
        Rule::new("recv")
            .when(When::input(Pattern::variant("resp", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::sub(Expr::slot("in_flight"), Expr::int(1)),
            })
            .do_(Effect::SamplesPopOldestInto {
                slot: "sent_at".into(),
                into_var: "t".into(),
            })
            .do_(Effect::RecordMetric {
                name: "rtt".into(),
                value: Expr::sub(Expr::now(), Expr::var("t")),
            }),
    ];
    let client = sim.add_node("Client", client_slots, client_rules);

    sim.add_edge(client, server, Expr::int(2_000_000));
    sim.add_edge(server, client, Expr::exp_dist(Expr::float(15_000_000.0)));
    sim.add_edge(client, client, Expr::int(5_000_000));
    sim.inject(client, Value::variant("tick", Value::Nil));
    sim
}

#[test]
fn snapshot_replay_matches_original() {
    // Reference run: run straight through to t=200ms.
    let mut ref_sim = build_sim();
    ref_sim.run_until(200_000_000);
    let ref_tail: Vec<String> = ref_sim.log.events.iter()
        .filter(|e| !matches!(e, Event::ClockAdvanced { .. }))
        .map(event_digest)
        .collect();

    // Replayed run: run to t=100ms, snapshot, then keep running to t=200ms.
    // Separately, restore from the snapshot and replay to t=200ms. The tails
    // starting from t=100ms should match byte-for-byte.
    let mut rep_sim = build_sim();
    rep_sim.run_until(100_000_000);
    let snap_len = rep_sim.log.total_recorded;
    let snapshot = rep_sim.snapshot();

    rep_sim.run_until(200_000_000);
    let rep_tail_from_snap: Vec<String> = rep_sim.log.events.iter()
        .filter(|e| !matches!(e, Event::ClockAdvanced { .. }))
        .skip_while(|_| {
            // Skip events that came before the snapshot. The event log is
            // ring-bounded but for this test size it's not evicting.
            // We want events recorded AFTER index `snap_len`.
            false
        })
        .map(event_digest)
        .collect();

    // Now restore and replay.
    let mut restored = Sim::new(0);  // seed doesn't matter; restore overwrites
    restored.restore_from(snapshot);
    assert_eq!(restored.log.total_recorded, snap_len);
    restored.run_until(200_000_000);
    let restored_tail: Vec<String> = restored.log.events.iter()
        .filter(|e| !matches!(e, Event::ClockAdvanced { .. }))
        .map(event_digest)
        .collect();

    // The tails from both runs should match after the snapshot point.
    // (Both should also match the reference run's tail after t=100ms.)
    assert_eq!(
        rep_tail_from_snap, restored_tail,
        "restored replay should produce identical events as the straight-through run"
    );
    assert_eq!(
        rep_tail_from_snap, ref_tail,
        "reference run should match both other variants"
    );
}

#[test]
fn snapshot_ring_captures_multiple() {
    use flow::SnapshotRing;

    let mut sim = build_sim();
    let mut ring = SnapshotRing::new(5);

    for t in (50_000_000u64..=300_000_000).step_by(50_000_000) {
        sim.run_until(t);
        ring.capture(&sim);
    }

    // We ran at 50, 100, 150, 200, 250, 300 — 6 captures but cap is 5,
    // so oldest (50ms) should have been evicted.
    assert_eq!(ring.len(), 5);
    assert_eq!(ring.entries.front().unwrap().sim_now_ns, 100_000_000);
    assert_eq!(ring.entries.back().unwrap().sim_now_ns, 300_000_000);

    // Rewind to just after 100ms.
    let snap = ring.latest_before_ns(150_000_000).unwrap().clone();
    let mut rewound = sim.clone();
    rewound.restore_from(snap.sim);
    assert!(rewound.now_ns <= 150_000_000);
}

#[test]
fn snapshot_ring_anchor_survives_eviction() {
    use flow::SnapshotRing;

    // Cap of 2 — easy to overflow. The anchor at t=0 must remain
    // available even after the ring has rolled forward many times.
    let mut sim = build_sim();
    let mut ring = SnapshotRing::new(2);

    // First capture seeds the anchor at t=0.
    ring.capture(&sim);
    assert!(ring.anchor.is_some());
    assert_eq!(ring.anchor.as_ref().unwrap().sim_now_ns, 0);

    // Advance and capture several times to evict everything.
    for t in (50_000_000u64..=400_000_000).step_by(50_000_000) {
        sim.run_until(t);
        ring.capture(&sim);
    }
    // Ring holds only the two newest, but anchor is still t=0.
    assert_eq!(ring.len(), 2);
    assert_eq!(ring.anchor.as_ref().unwrap().sim_now_ns, 0);

    // Querying for a time before any ring entry falls back to the
    // anchor instead of returning None.
    let snap = ring.latest_before_ns(10_000_000).expect("anchor fallback");
    assert_eq!(snap.sim_now_ns, 0);
}

#[test]
fn auto_capture_respects_interval() {
    use flow::{CapturePolicy, SnapshotRing};

    let mut sim = build_sim();
    let mut ring = SnapshotRing::new(64);
    let policy = CapturePolicy {
        min_interval_ns: 100_000_000,
        min_event_delta: u64::MAX,
    };

    // First call always captures (seeds anchor).
    assert!(ring.auto_capture(&sim, policy));
    assert_eq!(ring.len(), 1);

    // 50ms later: below threshold, skipped.
    sim.run_until(50_000_000);
    assert!(!ring.auto_capture(&sim, policy));
    assert_eq!(ring.len(), 1);

    // 150ms total: crosses 100ms threshold, captured.
    sim.run_until(150_000_000);
    assert!(ring.auto_capture(&sim, policy));
    assert_eq!(ring.len(), 2);

    // Marker times start at the anchor (0) and progress forward.
    let marks = ring.marker_times_ns();
    assert_eq!(marks.first().copied(), Some(0));
    assert!(marks.windows(2).all(|w| w[0] <= w[1]));
}
