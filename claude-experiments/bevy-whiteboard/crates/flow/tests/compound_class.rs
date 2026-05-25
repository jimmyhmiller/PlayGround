//! Compound classes registered like leaf classes. `register_classes`
//! stores a `compound { ... }` block unexpanded; `sim.instantiate`
//! expands it on demand into the live sim, name-prefixing every inner
//! node with the instance name. Multiple instances coexist with
//! distinct prefixes.

use flow::dsl;

#[test]
fn compound_class_can_be_registered_and_instantiated_twice() {
    // Tiny compound: one inner node bumps a counter on each tick.
    // Two instances share the class but keep separate counters.
    let src = r#"
        node Bumper {
            slots { count: Int = 0 }
            rule any { on _ do { count := count + 1 } }
        }

        compound DoubleBumper {
            in  { input: A }
            out { sum:    A }
            node A : Bumper { }
            node B : Bumper { }
            edges { A -> B : 1 }
        }
    "#;
    let mut sim = flow::Sim::new(0);
    let names = dsl::register_classes(&mut sim, src).expect("register");
    assert!(names.iter().any(|n| n == "Bumper"));
    assert!(names.iter().any(|n| n == "DoubleBumper"));
    assert!(sim.has_class("DoubleBumper"), "compound should be a class");
    assert!(sim.has_compound_class("DoubleBumper"));
    assert!(!sim.has_compound_class("Bumper"));

    let _id1 = sim.instantiate("DoubleBumper", "left").expect("inst left");
    let _id2 = sim.instantiate("DoubleBumper", "right").expect("inst right");

    // Both instances should have produced two prefixed inner nodes
    // (left::A, left::B, right::A, right::B) plus the two port shims.
    let names: Vec<&str> = sim.nodes.values().map(|n| n.name.as_str()).collect();
    assert!(names.contains(&"left::A"),  "names = {:?}", names);
    assert!(names.contains(&"left::B"),  "names = {:?}", names);
    assert!(names.contains(&"right::A"), "names = {:?}", names);
    assert!(names.contains(&"right::B"), "names = {:?}", names);
    assert!(names.contains(&"left"),     "port shim left missing: {:?}", names);
    assert!(names.contains(&"right"),    "port shim right missing: {:?}", names);
}
