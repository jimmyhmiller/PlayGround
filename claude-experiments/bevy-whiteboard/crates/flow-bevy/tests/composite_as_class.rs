use flow_bevy::gadgets::{GADGETS_DSL, validate_gadget_dsl};

#[test]
fn workercomposite_can_be_instantiated() {
    let mut sim = flow::Sim::new(0);
    flow::dsl::register_classes(&mut sim, GADGETS_DSL).expect("register");
    assert!(sim.has_compound_class("WorkerComposite"),
        "WorkerComposite should now be a registered compound class");
    let _id = sim.instantiate("WorkerComposite", "w1").expect("instantiate WorkerComposite");
    // Inner nodes should be prefixed.
    let names: Vec<&str> = sim.nodes.values().map(|n| n.name.as_str()).collect();
    assert!(names.contains(&"w1::L"),  "names = {:?}", names);
    assert!(names.contains(&"w1::F"),  "names = {:?}", names);
    assert!(names.contains(&"w1::Sv"), "names = {:?}", names);
    assert!(names.contains(&"w1::R"),  "names = {:?}", names);
    assert!(names.contains(&"w1"),     "port shim missing: {:?}", names);
    let _ = validate_gadget_dsl;
}
