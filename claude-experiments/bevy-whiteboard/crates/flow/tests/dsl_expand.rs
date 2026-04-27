//! Direct unit tests of the expansion pass — exercise `for`,
//! `Cell_{x}_{y}` interpolation, nested compounds (`Outer::Inner::Leaf`
//! qualified names), and negative paths (compile-time evaluator
//! rejects runtime constructs in `for` bounds, etc.).
//!
//! These tests work on the AST level rather than the eventual `Sim`,
//! so they don't depend on engine semantics — they pin the
//! construction-time contract directly.

use flow::dsl::{
    self,
    ast::{EdgeBodyItem, Item, NameTpl},
};

fn parse_and_expand(src: &str) -> Result<Vec<Item>, String> {
    let file = dsl::parse(src)?;
    Ok(dsl::expand::expand(&file)?.items)
}

fn node_names(items: &[Item]) -> Vec<String> {
    items
        .iter()
        .filter_map(|it| match it {
            Item::Node(n) => n.name.as_plain().map(str::to_string),
            Item::Instance(i) => i.name.as_plain().map(str::to_string),
            _ => None,
        })
        .collect()
}

fn edge_pairs(items: &[Item]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for it in items {
        if let Item::Edges(es) = it {
            for e in es {
                let EdgeBodyItem::Edge(d) = e else { panic!("residual edges should be flat: {:?}", e) };
                out.push((
                    d.from.node.as_plain().unwrap().to_string(),
                    d.to.node.as_plain().unwrap().to_string(),
                ));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------

#[test]
fn for_at_top_level_expands_into_named_nodes() {
    let src = r#"
        for x in 0..3 {
            node Cell_{x} { slots { v: Int = x } }
        }
    "#;
    let items = parse_and_expand(src).expect("expand should succeed");
    let names = node_names(&items);
    assert_eq!(names, vec!["Cell_0", "Cell_1", "Cell_2"]);
}

#[test]
fn nested_for_cartesian_product_with_two_holes() {
    let src = r#"
        for y in 0..2, x in 0..3 {
            node C_{x}_{y} { slots { v: Int = 0 } }
        }
    "#;
    let items = parse_and_expand(src).expect("expand should succeed");
    let names = node_names(&items);
    assert_eq!(
        names,
        vec!["C_0_0", "C_1_0", "C_2_0", "C_0_1", "C_1_1", "C_2_1"]
    );
}

#[test]
fn for_inside_edges_block_expands_to_concrete_edges() {
    let src = r#"
        node A { slots { v: Int = 0 } }
        node B { slots { v: Int = 0 } }
        node C { slots { v: Int = 0 } }
        edges {
            for i in 0..3 {
                A -> A : 1ms
            }
        }
    "#;
    let items = parse_and_expand(src).expect("expand should succeed");
    let edges = edge_pairs(&items);
    assert_eq!(edges.len(), 3, "expected 3 expanded edges, got {:?}", edges);
    for (a, b) in &edges {
        assert_eq!(a, "A");
        assert_eq!(b, "A");
    }
}

#[test]
fn nested_compound_qualifies_inner_class_with_double_colon() {
    // Inner.Leaf inside Outer should appear as "Outer::Inner::Leaf".
    // Both compounds also get a port-shim entry at residual time.
    let src = r#"
        compound Outer(n: Int = 1) {
            compound Inner(m: Int = 1) {
                node Leaf { slots { v: Int = 0 } }
            }
        }
    "#;
    let items = parse_and_expand(src).expect("expand should succeed");
    let names = node_names(&items);
    assert!(names.contains(&"Outer::Inner::Leaf".to_string()),
        "expected nested name `Outer::Inner::Leaf`, got {:?}", names);
}

#[test]
fn instance_class_ref_rewrites_to_compound_local_class() {
    // The `MyService` instance refers to `Worker` (declared inside the
    // same compound), so after expansion the instance's class field
    // should be `MyService::Worker`, not bare `Worker`.
    let src = r#"
        compound MyService(n: Int = 1) {
            node Worker { slots { jobs: Int = 0 } }
            node W1 : Worker { jobs: 5 }
        }
    "#;
    let items = parse_and_expand(src).expect("expand should succeed");
    let inst_class = items.iter().find_map(|it| match it {
        Item::Instance(i) if i.name.as_plain() == Some("MyService::W1") => Some(i.class.clone()),
        _ => None,
    }).expect("instance MyService::W1 must exist after expansion");
    assert_eq!(inst_class, "MyService::Worker");
}

// ---------------------------------------------------------------------------
// Negative paths: the CT evaluator must reject constructs that aren't
// pure compile-time data.

#[test]
fn slot_reference_in_for_bound_is_rejected_at_expand_time() {
    let src = r#"
        node N { slots { width: Int = 5 } }
        for x in 0..width { node Cell_{x} { slots { v: Int = 0 } } }
    "#;
    let err = parse_and_expand(src).expect_err("expansion must reject slot ref in for bound");
    assert!(
        err.contains("compile-time eval") || err.contains("unknown name"),
        "unexpected error: {}",
        err
    );
}

#[test]
fn unknown_ct_var_in_name_template_is_rejected() {
    let src = r#"
        for x in 0..2 { node Cell_{y} { slots { v: Int = 0 } } }
    "#;
    let err = parse_and_expand(src).expect_err("`y` is not bound");
    assert!(err.contains("unknown name `y`"), "unexpected error: {}", err);
}

#[test]
fn float_in_for_bound_is_rejected() {
    let src = r#"
        for x in 0..3.5 { node Cell_{x} { slots { v: Int = 0 } } }
    "#;
    let err = parse_and_expand(src).expect_err("float bounds must be rejected");
    assert!(err.contains("floats are not supported"), "unexpected error: {}", err);
}

#[test]
fn compound_param_with_no_default_and_no_args_is_rejected() {
    let src = r#"
        compound Pool(width: Int) {
            node X { slots { v: Int = 0 } }
        }
    "#;
    let err = parse_and_expand(src).expect_err("missing default must surface at expand time");
    assert!(
        err.contains("no default") && err.contains("Pool") && err.contains("width"),
        "unexpected error: {}",
        err
    );
}

#[test]
fn name_template_substitutes_compound_params_into_node_names() {
    // `period_ns` is a compound param; it should be substituted into
    // the slot init expression of the inner node.
    let src = r#"
        compound Tick(period_ns: Int = 5000000) {
            node Clock { slots { p: Int = period_ns } }
        }
    "#;
    let items = parse_and_expand(src).expect("expand should succeed");
    let clock = items.iter().find_map(|it| match it {
        Item::Node(n) if n.name.as_plain() == Some("Tick::Clock") => Some(n),
        _ => None,
    }).expect("Tick::Clock must exist");
    let init = clock.slots.iter().find(|s| s.name == "p").unwrap().init.as_ref().unwrap();
    // After expansion + folding, the init should be a literal Int.
    use flow::dsl::ast::Expr;
    assert!(matches!(init, Expr::Int(5_000_000)),
        "expected literal Int(5000000), got {:?}", init);
}

#[test]
fn plain_name_tpl_round_trips_through_expansion() {
    let n = NameTpl::plain("Foo");
    assert_eq!(n.as_plain(), Some("Foo"));
    assert!(n.is_plain());
}
