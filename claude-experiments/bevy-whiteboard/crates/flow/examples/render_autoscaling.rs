//! Render the pool_least_busy sim to HTML — shows *spawned nodes
//! appearing* over time (bootstrap rule spawns 3 workers at t=0)
//! plus packets flowing between them.

use std::collections::BTreeMap;

use flow::{
    EdgeEnd, Effect, EmitTo, Expr, Pattern, Rule, Samples, Sim,
    Template, Value, When,
};

fn main() -> std::io::Result<()> {
    let mut sim = Sim::new(42);

    let worker = Template::new("Worker").with_prefix("W")
        .rule(Rule::new("take_work")
            .when(When::input(Pattern::variant("work", Pattern::wild())))
            .do_(Effect::Emit {
                payload: Expr::variant("done", Expr::lit(Value::Nil)),
                to: EmitTo::ToTargetExpr(Expr::self_ref()),
            }))
        .rule(Rule::new("finish")
            .when(When::input(Pattern::variant("done", Pattern::wild())))
            .do_(Effect::Emit {
                payload: Expr::variant("ready", Expr::self_ref()),
                to: EmitTo::ToTarget("Pool".into()),
            }))
        .edge(EdgeEnd::Parent, EdgeEnd::ThisInstance, Expr::int(1_000))
        .edge(EdgeEnd::ThisInstance, EdgeEnd::Parent, Expr::int(1_000))
        .edge(EdgeEnd::ThisInstance, EdgeEnd::ThisInstance,
            Expr::exp_dist(Expr::float(25_000_000.0)));
    sim.register_template(worker);

    let mut pool_slots = BTreeMap::new();
    pool_slots.insert("ready".into(), Value::Samples(Samples::new(32)));
    pool_slots.insert("dispatched".into(), Value::Int(0));

    let pool_rules = vec![
        Rule::new("worker_ready")
            .when(When::input(Pattern::variant("ready", Pattern::var("w"))))
            .do_(Effect::SamplesPush { slot: "ready".into(), value: Expr::var("w") }),
        Rule::new("dispatch")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .guard(Expr::gt(Expr::samples_len("ready"), Expr::int(0)))
            .do_(Effect::SamplesPopOldestInto { slot: "ready".into(), into_var: "w".into() })
            .do_(Effect::SetSlot {
                slot: "dispatched".into(),
                value: Expr::add(Expr::slot("dispatched"), Expr::int(1)),
            })
            .do_(Effect::RecordMetric { name: "dispatched".into(), value: Expr::slot("dispatched") })
            .do_(Effect::Emit {
                payload: Expr::variant("work", Expr::lit(Value::Nil)),
                to: EmitTo::ToTargetExpr(Expr::var("w")),
            }),
        Rule::new("spawn_and_enroll")
            .when(When::input(Pattern::variant("kick", Pattern::wild())))
            .do_(Effect::Spawn { template: "Worker".into(), into_var: Some("w".into()) })
            .do_(Effect::SamplesPush { slot: "ready".into(), value: Expr::var("w") }),
    ];
    let pool = sim.add_node("Pool", pool_slots, pool_rules);

    // Schedule staggered spawning so the visualization shows nodes appearing over time.
    use flow::{Action, Scenario};
    let scenario = Scenario::new()
        .at(10_000_000, Action::Inject { node: pool, payload: Value::variant("kick", Value::Nil), reply_to: None })
        .at(40_000_000, Action::Inject { node: pool, payload: Value::variant("kick", Value::Nil), reply_to: None })
        .at(70_000_000, Action::Inject { node: pool, payload: Value::variant("kick", Value::Nil), reply_to: None });
    sim.load_scenario(scenario);

    let client_rules = vec![
        Rule::new("fire")
            .when(When::input(Pattern::variant("tick", Pattern::wild())))
            .do_(Effect::Emit {
                payload: Expr::variant("req", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Pool".into()),
            })
            .do_(Effect::Emit {
                payload: Expr::variant("tick", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Client".into()),
            }),
    ];
    let client = sim.add_node("Client", BTreeMap::new(), client_rules);
    sim.add_edge(client, pool, Expr::int(1_000_000));
    sim.add_edge(client, client, Expr::int(10_000_000));
    sim.inject(client, Value::variant("tick", Value::Nil), None);

    sim.run_until(400_000_000);

    std::fs::create_dir_all("flow/out")?;
    sim.write_html("Autoscaling pool (staggered spawn)", "flow/out/autoscaling.html")?;
    println!("wrote flow/out/autoscaling.html ({} events, {} ms sim time)",
        sim.log.total_recorded, sim.now_ns / 1_000_000);
    Ok(())
}
