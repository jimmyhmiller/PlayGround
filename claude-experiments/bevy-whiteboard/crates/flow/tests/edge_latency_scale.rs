//! `Sim::edge_latency_scale` slows packet transit without altering
//! anything else. Two facets to verify:
//!
//! 1. New emits scale: with `edge_latency_scale = 2.0`, an edge whose
//!    declared latency is `100ms` schedules arrival at `now + 200ms`.
//! 2. In-flight rescale: bumping the scale mid-flight rescales every
//!    `Scheduled` packet's *remaining* travel time so the change feels
//!    uniform — already-emitted packets slow down too.

const SRC: &str = r#"
    node Src {
        slots { dummy: Int = 0 }
        rule fwd {
            on go(_)
            do { emit_each packet(nil) to out_neighbors() }
        }
    }
    node Dst { slots { x: Int = 0 } }
    edges {
        Src -> Dst : 100ms
    }
"#;

#[test]
fn future_emits_scale() {
    let mut sim = flow::dsl::load(SRC, 0).unwrap();
    sim.set_edge_latency_scale(2.0);

    let src_id = sim.nodes.values().find(|n| n.name == "Src").unwrap().id;
    sim.inject(src_id, flow::Value::variant("go", flow::Value::Nil));
    sim.run_until(1); // fire the rule, schedule the emit

    let arr = sim.in_flight.peek().expect("emitted").0.arrives_at_ns;
    assert_eq!(arr, 200_000_000, "100ms × 2.0 should land at 200ms");
}

#[test]
fn in_flight_rescale_doubles_remaining_travel() {
    let mut sim = flow::dsl::load(SRC, 0).unwrap();
    let src_id = sim.nodes.values().find(|n| n.name == "Src").unwrap().id;
    sim.inject(src_id, flow::Value::variant("go", flow::Value::Nil));
    sim.run_until(1);
    let arr0 = sim.in_flight.peek().unwrap().0.arrives_at_ns;
    assert_eq!(arr0, 100_000_000);

    sim.run_until(40_000_000); // 60ms remaining
    sim.set_edge_latency_scale(2.0);
    let arr1 = sim.in_flight.peek().unwrap().0.arrives_at_ns;
    // remaining 60ms × 2 → 120ms; from now=40ms → 160ms
    assert_eq!(arr1, 160_000_000);
}

#[test]
fn scaling_back_down_compresses_remaining_travel() {
    let mut sim = flow::dsl::load(SRC, 0).unwrap();
    sim.set_edge_latency_scale(4.0);
    let src_id = sim.nodes.values().find(|n| n.name == "Src").unwrap().id;
    sim.inject(src_id, flow::Value::variant("go", flow::Value::Nil));
    sim.run_until(1);
    // 100ms × 4 = 400ms
    assert_eq!(sim.in_flight.peek().unwrap().0.arrives_at_ns, 400_000_000);

    sim.run_until(100_000_000); // 300ms remaining
    sim.set_edge_latency_scale(1.0); // factor 0.25
    // 300 × 0.25 = 75 → arrives at 100 + 75 = 175ms
    assert_eq!(sim.in_flight.peek().unwrap().0.arrives_at_ns, 175_000_000);
}

/// 0-latency edges should still slow down when scale > 1, otherwise
/// the user's "slow edges" knob has no effect on broadcast/wire-style
/// edges.
#[test]
fn zero_latency_edges_pick_up_floor_at_high_scale() {
    const ZERO_SRC: &str = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule fwd {
                on go(_)
                do { emit_each packet(nil) to out_neighbors() }
            }
        }
        node Dst { slots { x: Int = 0 } }
        edges {
            Src -> Dst : 0
        }
    "#;
    let mut sim = flow::dsl::load(ZERO_SRC, 0).unwrap();
    let src_id = sim.nodes.values().find(|n| n.name == "Src").unwrap().id;

    // At scale=1, 0-latency edge stays 0-latency.
    sim.inject(src_id, flow::Value::variant("go", flow::Value::Nil));
    sim.run_until(1);
    assert!(sim.in_flight.is_empty(),
        "0-latency edge at scale=1 should deliver in same instant");

    // At scale=4, 0-latency edge should arrive at (4 - 1) * 50ms = 150ms.
    sim.set_edge_latency_scale(4.0);
    sim.inject(src_id, flow::Value::variant("go", flow::Value::Nil));
    let emit_ns = sim.now_ns;
    sim.run_until(sim.now_ns + 1);
    let arr = sim.in_flight.peek().expect("emitted at scale=4").0.arrives_at_ns;
    let expected = emit_ns + 3 * flow::sim::EDGE_LATENCY_SCALE_FLOOR_NS;
    assert_eq!(arr, expected,
        "0-latency edge at scale=4 should land at emit + 3*FLOOR ({}ms)",
        3 * flow::sim::EDGE_LATENCY_SCALE_FLOOR_NS / 1_000_000);
}

/// Non-zero edges with natural latencies above the floor are unaffected
/// by the floor — they scale purely.
#[test]
fn non_zero_edges_above_floor_still_scale_purely() {
    let mut sim = flow::dsl::load(SRC, 0).unwrap();
    sim.set_edge_latency_scale(2.0);
    let src_id = sim.nodes.values().find(|n| n.name == "Src").unwrap().id;
    sim.inject(src_id, flow::Value::variant("go", flow::Value::Nil));
    sim.run_until(1);
    // 100ms × 2 = 200ms, floor at scale=2 is 50ms — pure scale wins.
    assert_eq!(sim.in_flight.peek().unwrap().0.arrives_at_ns, 200_000_000);
}

/// Self-loops carry a node's own clock (Generator/Client schedule
/// next-tick via `self -> self : period_ns`). The edge-latency scale
/// is "transit time between nodes" — internal clocks must not slow
/// down. Otherwise the user can't separate "slow lines" from
/// "slow emit rate".
#[test]
fn self_loops_ignore_edge_latency_scale() {
    const SELF_LOOP_SRC: &str = r#"
        node Tick {
            slots { count: Int = 0, period_ns: Int = 50000000 }
            rule beat {
                on go(_)
                do {
                    count := count + 1
                    emit go(nil) to self
                }
            }
        }
        edges {
            Tick -> Tick : period_ns
        }
        scenario {
            at 0ns: inject Tick <- go()
        }
    "#;
    let mut sim = flow::dsl::load(SELF_LOOP_SRC, 0).unwrap();
    sim.set_edge_latency_scale(10.0);

    // Run for 1s. At 50ms period, expect ~20 ticks regardless of scale.
    sim.run_until(1_000_000_000);
    let tick_id = sim.nodes.values().find(|n| n.name == "Tick").unwrap().id;
    let count = match sim.nodes[&tick_id].slots.get("count") {
        Some(flow::Value::Int(n)) => *n,
        _ => panic!("missing count"),
    };
    assert!(
        (19..=21).contains(&count),
        "self-loop emitter at 50ms period should produce ~20 ticks/s \
         regardless of edge_latency_scale; got {} at scale=10",
        count
    );
}

#[test]
fn invalid_scales_are_ignored() {
    let mut sim = flow::Sim::new(0);
    sim.set_edge_latency_scale(0.0);
    assert_eq!(sim.edge_latency_scale, 1.0);
    sim.set_edge_latency_scale(-1.0);
    assert_eq!(sim.edge_latency_scale, 1.0);
    sim.set_edge_latency_scale(f64::NAN);
    assert_eq!(sim.edge_latency_scale, 1.0);
}
