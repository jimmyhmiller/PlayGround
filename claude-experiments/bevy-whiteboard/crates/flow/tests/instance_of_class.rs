//! `node NAME : CLASS { slot: expr }` — instance-of-class DSL form.
//!
//! Lets a canvas spin up multiple instances of a registered class
//! without redeclaring the class. Without this, a canvas with three
//! `SagaStep`s would have to triple-paste the entire DSL.

use flow::Sim;
use flow::Value;

const CLASS_DSL: &str = r#"
node Step {
    slots {
        done:        Int   = 0
        compensated: Int   = 0
        fail_prob:   Float = 0.0
        color:       Int   = 0
    }
}
"#;

#[test]
fn instance_clones_class_with_overrides() {
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, CLASS_DSL).unwrap();

    let main = r#"
        node S1 : Step { fail_prob: 0.0; color: 0 }
        node S2 : Step { fail_prob: 1.0; color: 0 }
        node S3 : Step { color: 0 }
    "#;
    let file = flow::dsl::parse(main).unwrap();
    flow::dsl::lower_into(&mut sim, &file).unwrap();

    let s1 = sim.nodes.values().find(|n| n.name == "S1").unwrap().id;
    let s2 = sim.nodes.values().find(|n| n.name == "S2").unwrap().id;
    let s3 = sim.nodes.values().find(|n| n.name == "S3").unwrap().id;

    assert_eq!(sim.class_name(s1), Some("Step"));
    assert_eq!(sim.class_name(s2), Some("Step"));
    assert_eq!(sim.class_name(s3), Some("Step"));

    let s1 = sim.nodes.values().find(|n| n.name == "S1").unwrap();
    let s2 = sim.nodes.values().find(|n| n.name == "S2").unwrap();
    let s3 = sim.nodes.values().find(|n| n.name == "S3").unwrap();

    assert_eq!(s1.slots.get("fail_prob"), Some(&Value::Float(0.0)));
    assert_eq!(s2.slots.get("fail_prob"), Some(&Value::Float(1.0)));
    // S3 had no fail_prob override → defaults from class
    assert_eq!(s3.slots.get("fail_prob"), Some(&Value::Float(0.0)));

    // Each instance has its own `done` counter, starting from class default.
    assert_eq!(s1.slots.get("done"), Some(&Value::Int(0)));
    assert_eq!(s2.slots.get("done"), Some(&Value::Int(0)));
}

#[test]
fn instance_unknown_class_errors() {
    let mut sim = Sim::new(0);
    let main = r#"
        node X : DoesNotExist { }
    "#;
    let file = flow::dsl::parse(main).unwrap();
    let err = match flow::dsl::lower_into(&mut sim, &file) {
        Ok(_) => panic!("expected error"),
        Err(e) => e,
    };
    assert!(err.contains("DoesNotExist"), "expected error mentioning class, got: {}", err);
}

#[test]
fn instance_unknown_slot_errors() {
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, CLASS_DSL).unwrap();
    let main = r#"
        node X : Step { not_a_slot: 5 }
    "#;
    let file = flow::dsl::parse(main).unwrap();
    let err = match flow::dsl::lower_into(&mut sim, &file) {
        Ok(_) => panic!("expected error"),
        Err(e) => e,
    };
    assert!(err.contains("not_a_slot"), "expected error mentioning bad slot, got: {}", err);
}
