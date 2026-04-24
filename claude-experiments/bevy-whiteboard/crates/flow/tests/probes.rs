//! Probes declared in DSL → readable values at runtime.

use flow::{Sim, Value};

const SRC: &str = r#"
node Counter {
    slots {
        hits: Int = 0
        period_ns: Int = 500000000
    }
    probes {
        rate:  "{1s / period_ns}/s"
        hits:  "{hits}"
        label: "counter"
    }
}
"#;

#[test]
fn probe_labels_and_readings() {
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, SRC).unwrap();
    let id = sim.instantiate("Counter", "c").unwrap();

    let labels = sim.probe_labels(id);
    assert_eq!(labels, vec!["rate".to_string(), "hits".to_string(), "label".to_string()]);

    assert_eq!(sim.probe_reading(id, "rate").as_deref(), Some("2/s"));
    assert_eq!(sim.probe_reading(id, "hits").as_deref(), Some("0"));
    assert_eq!(sim.probe_reading(id, "label").as_deref(), Some("counter"));

    // Bump a slot; probe reflects it on next read.
    sim.nodes.get_mut(&id).unwrap().slots.insert("hits".into(), Value::Int(42));
    assert_eq!(sim.probe_reading(id, "hits").as_deref(), Some("42"));

    // Unknown label returns None.
    assert!(sim.probe_reading(id, "nope").is_none());
}

#[test]
fn probes_default_to_empty() {
    const NO_PROBES: &str = r#"
        node Plain { slots { x: Int = 0 } }
    "#;
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, NO_PROBES).unwrap();
    let id = sim.instantiate("Plain", "p").unwrap();
    assert!(sim.probe_labels(id).is_empty());
}
