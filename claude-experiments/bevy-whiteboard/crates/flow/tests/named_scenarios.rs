//! Named scenarios in the DSL: an unnamed `scenario { }` auto-runs under
//! the name "main" for back-compat; a `scenario foo { }` lives in the
//! library until the caller picks it.

use flow::{Sim, Value};

const SRC: &str = r#"
node Gen {
    slots {
        hits: Int = 0
    }
    rule on_ping {
        on ping(_)
        do { hits := hits + 1 }
    }
}

scenario warmup {
    at 0ns: inject Gen <- ping(nil)
    at 0ns: inject Gen <- ping(nil)
}

scenario {
    at 0ns: inject Gen <- ping(nil)
}
"#;

#[test]
fn unnamed_scenario_auto_runs_as_main() {
    let mut sim = flow::dsl::load(SRC, 0).unwrap();
    // Auto-scheduled: one inject from the unnamed block.
    sim.run_until(1);
    let id = sim.node_by_name("Gen").unwrap();
    assert_eq!(sim.nodes[&id].slots["hits"], Value::Int(1));
}

#[test]
fn named_scenario_does_not_auto_run() {
    let sim = flow::dsl::load(SRC, 0).unwrap();
    // "warmup" is registered but not scheduled.
    assert!(sim.scenarios.contains_key("warmup"));
    assert!(sim.scenarios.contains_key("main"));
}

#[test]
fn run_scenario_activates_named() {
    // Stack warmup on top of main before advancing time: both fire at
    // 0ns so we observe their combined effect. Scheduling past-time
    // actions after run_until advances the clock isn't exercised here
    // (the engine pops those in `apply_action` only when now_ns hits
    // their at_ns — already-past actions stall the loop).
    let mut sim = flow::dsl::load(SRC, 0).unwrap();
    sim.run_scenario("warmup").unwrap();
    sim.run_until(1);
    let id = sim.node_by_name("Gen").unwrap();
    // main (1 ping) + warmup (2 pings) = 3 hits
    assert_eq!(sim.nodes[&id].slots["hits"], Value::Int(3));
}

#[test]
fn duplicate_scenario_names_rejected() {
    const DUP: &str = r#"
        node N { slots { x: Int = 0 } }
        scenario foo { at 0ns: inject N <- tick(nil) }
        scenario foo { at 0ns: inject N <- tick(nil) }
    "#;
    let err = match flow::dsl::load(DUP, 0) {
        Err(e) => e,
        Ok(_) => panic!("expected duplicate scenario error"),
    };
    assert!(err.contains("duplicate scenario"), "got: {}", err);
}

#[test]
fn unknown_run_scenario_errors() {
    let mut sim = Sim::new(0);
    let err = sim.run_scenario("nope").unwrap_err();
    assert!(err.contains("no scenario"), "got: {}", err);
}
