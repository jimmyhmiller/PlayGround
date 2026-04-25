//! Sim-level timeline. The timeline lives on `Sim` and fires through
//! `Sim::run_until` directly — no Bevy, no UI framework, no system
//! schedule. This test asserts that property.

use std::collections::BTreeMap;

use flow::{Sim, Value};

fn make_sim_with_float_slot() -> (Sim, flow::NodeId) {
    let mut sim = Sim::new(0);
    let nid = sim.add_node(
        "Test",
        BTreeMap::from([("hit_rate".to_string(), Value::Float(0.5))]),
        Vec::new(),
    );
    (sim, nid)
}

#[test]
fn fires_at_scheduled_time_through_run_until() {
    let (mut sim, nid) = make_sim_with_float_slot();
    sim.timeline.schedule(1_000_000_000, nid, "hit_rate".into(), Value::Float(0.9));

    // Drive past the scheduled time. No app.update() — pure sim.
    sim.run_until(1_500_000_000);

    let v = match sim.nodes[&nid].slots.get("hit_rate") {
        Some(Value::Float(f)) => *f,
        other => panic!("hit_rate not Float: {:?}", other),
    };
    assert!((v - 0.9).abs() < 1e-9);
    assert_eq!(sim.timeline.pending(), 0);
    assert!(sim.timeline.events[0].fired);
}

#[test]
fn does_not_fire_before_at_ns() {
    let (mut sim, nid) = make_sim_with_float_slot();
    sim.timeline.schedule(5_000_000_000, nid, "hit_rate".into(), Value::Float(0.9));

    sim.run_until(1_000_000_000);

    let v = match sim.nodes[&nid].slots["hit_rate"] {
        Value::Float(f) => f, _ => unreachable!(),
    };
    assert!((v - 0.5).abs() < 1e-9);
    assert_eq!(sim.timeline.pending(), 1);
}

#[test]
fn next_pending_at_ns_drives_engine_advancement() {
    // The engine's tick loop must include timeline events when computing
    // the next instant to advance to. If it didn't, run_until would
    // skip past the firing time without ever stopping there.
    let (mut sim, nid) = make_sim_with_float_slot();
    sim.timeline.schedule(2_000_000_000, nid, "hit_rate".into(), Value::Float(0.1));
    sim.timeline.schedule(4_000_000_000, nid, "hit_rate".into(), Value::Float(0.9));

    sim.run_until(10_000_000_000);

    // Final state reflects the LAST event (chronological order).
    let v = match sim.nodes[&nid].slots["hit_rate"] {
        Value::Float(f) => f, _ => unreachable!(),
    };
    assert!((v - 0.9).abs() < 1e-9);
    assert!(sim.timeline.events.iter().all(|e| e.fired));
}

#[test]
fn remove_before_fire_skips_event() {
    let (mut sim, nid) = make_sim_with_float_slot();
    let id = sim.timeline.schedule(1_000_000_000, nid, "hit_rate".into(), Value::Float(0.9));

    assert!(sim.timeline.remove(id));
    sim.run_until(2_000_000_000);

    let v = match sim.nodes[&nid].slots["hit_rate"] {
        Value::Float(f) => f, _ => unreachable!(),
    };
    assert!((v - 0.5).abs() < 1e-9);
    assert!(sim.timeline.events.is_empty());
}

#[test]
fn type_mismatch_silently_skipped_but_marked_fired() {
    let (mut sim, nid) = make_sim_with_float_slot();
    // Schedule an Int into a Float slot.
    sim.timeline.schedule(1_000_000_000, nid, "hit_rate".into(), Value::Int(42));

    sim.run_until(2_000_000_000);

    // Slot untouched.
    let v = match sim.nodes[&nid].slots["hit_rate"] {
        Value::Float(f) => f, _ => unreachable!(),
    };
    assert!((v - 0.5).abs() < 1e-9);
    // But event still marked fired so the queue makes progress.
    assert!(sim.timeline.events[0].fired);
    assert_eq!(sim.timeline.pending(), 0);
}

#[test]
fn timeline_fires_alongside_rules_at_same_instant() {
    // Mix a timeline event with normal rule firing — the engine
    // should advance to the same instant via either gate, then
    // process both. Build a tiny self-ticking node and schedule a
    // slot write at the same time as one of its ticks.
    use flow::rule::{Effect, EmitTo, Rule, When};
    use flow::value::Pattern;
    use flow::expr::Expr;

    let mut sim = Sim::new(0);
    let nid = sim.add_node(
        "Ticker",
        BTreeMap::from([
            ("ticks".to_string(), Value::Int(0)),
            ("flag".to_string(), Value::Bool(false)),
        ]),
        vec![Rule::new("on_tick")
            .when(When::input(Pattern::variant("tick", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "ticks".into(),
                value: Expr::add(Expr::slot("ticks"), Expr::int(1)),
            })
            .do_(Effect::emit(
                Expr::variant("tick", Expr::lit(Value::Nil)),
                EmitTo::DefaultOut,
            ))],
    );
    sim.add_edge(nid, nid, Expr::int(500_000_000)); // 500ms self-loop
    sim.inject(nid, Value::variant("tick", Value::Nil));

    sim.timeline.schedule(1_500_000_000, nid, "flag".into(), Value::Bool(true));

    sim.run_until(2_000_000_000);

    // Ticker should have ticked several times AND the flag should be set.
    let ticks = match sim.nodes[&nid].slots["ticks"] {
        Value::Int(i) => i, _ => unreachable!(),
    };
    let flag = match sim.nodes[&nid].slots["flag"] {
        Value::Bool(b) => b, _ => unreachable!(),
    };
    assert!(ticks >= 3, "ticks={}", ticks);
    assert!(flag, "timeline event should have fired");
}
