//! Render the mutex_contention sim to a standalone HTML player.
//!
//! After running: `open flow/out/mutex.html`
//!
//! You'll get a timeline scrubber, play/pause, and an SVG with
//! nodes/edges/packets plus a side panel showing live slot values.
//! The entire visualization is driven by replaying the event log —
//! no sim re-execution, no Bevy, no build step.

use std::collections::BTreeMap;

use flow::{Effect, EmitTo, Expr, Pattern, Rule, Sim, Value, When};

fn main() -> std::io::Result<()> {
    let mut sim = Sim::new(42);

    // --- Mutex ---
    let mut mutex_slots = BTreeMap::new();
    mutex_slots.insert("holder".into(), Value::variant("None", Value::Nil));
    let mutex_rules = vec![
        Rule::new("grant")
            .when(When::input(Pattern::variant("acquire", Pattern::var("who"))))
            .when(When::slot("holder", Pattern::variant("None", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "holder".into(),
                value: Expr::variant("Some", Expr::var("who")),
            })
            .do_(Effect::Respond {
                payload: Expr::variant("granted", Expr::lit(Value::Nil)),
            }),
        Rule::new("release")
            .when(When::input(Pattern::variant("release", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "holder".into(),
                value: Expr::variant("None", Expr::lit(Value::Nil)),
            }),
    ];
    let mutex = sim.add_node("Mutex", mutex_slots, mutex_rules);

    // --- 3 Workers ---
    for i in 0..3 {
        let wname = format!("W{}", i);
        let mut slots = BTreeMap::new();
        slots.insert("idle".into(), Value::Bool(true));
        slots.insert("completed".into(), Value::Int(0));

        let rules = vec![
            Rule::new("request")
                .when(When::slot("idle", Pattern::lit(Value::Bool(true))))
                .do_(Effect::SetSlot { slot: "idle".into(), value: Expr::bool(false) })
                .do_(Effect::Emit {
                    payload: Expr::variant("acquire", Expr::lit(Value::str(wname.clone()))),
                    to: EmitTo::ToTarget("Mutex".into()),
                }),
            Rule::new("on_granted")
                .when(When::input(Pattern::variant("granted", Pattern::wild())))
                .do_(Effect::SetSlot {
                    slot: "completed".into(),
                    value: Expr::add(Expr::slot("completed"), Expr::int(1)),
                })
                .do_(Effect::RecordMetric {
                    name: "completed".into(),
                    value: Expr::slot("completed"),
                })
                .do_(Effect::Emit {
                    payload: Expr::variant("release", Expr::lit(Value::Nil)),
                    to: EmitTo::ToTarget("Mutex".into()),
                })
                .do_(Effect::SetSlot { slot: "idle".into(), value: Expr::bool(true) }),
        ];
        let w = sim.add_node(&wname, slots, rules);
        sim.add_edge(w, mutex, Expr::int(1_000));
        // Service time on the grant-reply path, so holder is visibly locked during work.
        sim.add_edge(mutex, w, Expr::exp_dist(Expr::float(25_000_000.0)));
    }

    // Run for 300 ms — long enough to see rotation; short enough to scrub comfortably.
    sim.run_until(300_000_000);

    std::fs::create_dir_all("flow/out")?;
    sim.write_html("Mutex Contention (3 workers)", "flow/out/mutex.html")?;
    println!("wrote flow/out/mutex.html ({} events, {} ms sim time)",
        sim.log.total_recorded, sim.now_ns / 1_000_000);
    println!("open with:\n  open flow/out/mutex.html");
    Ok(())
}
