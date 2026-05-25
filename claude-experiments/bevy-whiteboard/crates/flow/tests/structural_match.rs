//! Structural-match form: rules can bind a whole packet and read its
//! kind/value parts instead of writing one rule per variant name.

use flow::{dsl, Value};

#[test]
fn filter_passes_any_kind_via_structural_match() {
    let src = r#"
        node Filter {
            slots { match: Int = 0 }
            rule pass   { on p when p.value == match do { emit p to port pass } }
            rule reject { on p                       do { emit p to port reject } }
        }
        node PassOut   { slots { hits: Int = 0 } rule any { on _ do { hits := hits + 1 } } }
        node RejectOut { slots { hits: Int = 0 } rule any { on _ do { hits := hits + 1 } } }
        node F : Filter { match: 5 }
        edges {
            F.pass   -> PassOut   : 1
            F.reject -> RejectOut : 1
        }
        scenario s1 {
            at 0ns: inject F <- packet(5)
            at 0ns: inject F <- packet(3)
            at 0ns: inject F <- total(5)
            at 0ns: inject F <- count(7)
        }
    "#;
    let mut sim = dsl::load(src, 0).unwrap();
    sim.run_scenario("s1").unwrap();
    sim.run_until(1_000_000_000);
    let pass = sim.node_by_name("PassOut").unwrap();
    let rej  = sim.node_by_name("RejectOut").unwrap();
    // packet(5) and total(5) match; packet(3) and count(7) reject.
    assert_eq!(sim.nodes[&pass].slots["hits"], Value::Int(2));
    assert_eq!(sim.nodes[&rej].slots["hits"],  Value::Int(2));
    assert!(sim.error_counts.is_empty(), "no errors: {:?}", sim.error_counts);
}

#[test]
fn rule_can_read_kind_via_field_access() {
    let src = r#"
        node Sniffer {
            slots { saw_kind: String = "" }
            rule any { on p do { saw_kind := p.kind } }
        }
        scenario s1 {
            at 0ns: inject Sniffer <- foobar(nil)
        }
    "#;
    let sim = dsl::load(src, 0);
    if let Err(e) = &sim {
        panic!("load failed: {}", e);
    }
    let mut sim = sim.unwrap();
    sim.run_scenario("s1").unwrap();
    sim.run_until(1);
    let id = sim.node_by_name("Sniffer").unwrap();
    assert_eq!(sim.nodes[&id].slots["saw_kind"], Value::Str("foobar".into()));
}
