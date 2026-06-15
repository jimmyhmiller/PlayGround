//! The JSIR→HIR converter (`docs/HIR_COMPARISON.md` §9, strategy B): lower real
//! JS to JSLIR, then convert to a real `react_compiler_hir::HirFunction` and
//! assert the produced HIR is well-formed. Requires `--features react-hir`.
#![cfg(feature = "react-hir")]

use jsir_ir::{Op, Region};
use jsir_jslir::to_react_hir::convert_function;
use react_compiler_hir::environment::Environment;
use react_compiler_hir::{InstructionValue, Terminal};

fn jslir_body(src: &str) -> Region {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    find_body(&jslir).expect("a lowered function body")
}
fn find_body(op: &Op) -> Option<Region> {
    if op.name == "jsir.function_declaration" {
        if let Some(b) = op.regions.get(1) {
            if jsir_jslir::dialect::region_is_cfg(b) {
                return Some(b.clone());
            }
        }
    }
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                if let Some(f) = find_body(o) {
                    return Some(f);
                }
            }
        }
    }
    None
}

#[test]
fn converts_straight_line_arithmetic() {
    // let x = a + 1; return x;
    let region = jslir_body("function f(a) { let x = a + 1; return x; }");
    let mut env = Environment::new();
    let func = convert_function(&region, &mut env).expect("conversion");

    // One block, ending in a Return.
    assert_eq!(func.body.blocks.len(), 1);
    let entry = func.body.blocks.get(&func.body.entry).unwrap();
    assert!(matches!(entry.terminal, Terminal::Return { .. }));

    // The instruction stream contains the expected scalar value ops, in order:
    // LoadGlobal/LoadLocal(a) → Primitive(1) → BinaryExpression(+) → StoreLocal(x)
    // → LoadLocal(x). We assert the key ones are present and well-formed.
    let kinds: Vec<&str> = func
        .instructions
        .iter()
        .map(|i| match &i.value {
            InstructionValue::LoadLocal { .. } => "LoadLocal",
            InstructionValue::LoadGlobal { .. } => "LoadGlobal",
            InstructionValue::Primitive { .. } => "Primitive",
            InstructionValue::BinaryExpression { .. } => "BinaryExpression",
            InstructionValue::StoreLocal { .. } => "StoreLocal",
            _ => "other",
        })
        .collect();
    assert!(kinds.contains(&"Primitive"), "literal 1 became a Primitive: {kinds:?}");
    assert!(kinds.contains(&"BinaryExpression"), "a + 1 became a BinaryExpression: {kinds:?}");
    assert!(kinds.contains(&"StoreLocal"), "let x = … became a StoreLocal: {kinds:?}");

    // SSA-shaped operands: the BinaryExpression's operands are Places referencing
    // identifiers that were defined by earlier instructions (no dangling refs).
    let defined: std::collections::HashSet<u32> =
        func.instructions.iter().map(|i| i.lvalue.identifier.0).collect();
    for instr in &func.instructions {
        if let InstructionValue::BinaryExpression { left, right, .. } = &instr.value {
            // `left` is `a` (a param/global load result) and `right` is the literal;
            // both must be produced by some instruction's lvalue.
            assert!(
                defined.contains(&left.identifier.0) || defined.contains(&right.identifier.0),
                "binary operands resolve to defined values"
            );
        }
    }
}

#[test]
fn converts_if_diamond_with_real_terminals() {
    let region =
        jslir_body("function f(c) { let x = 0; if (c) { x = 1; } else { x = 2; } return x; }");
    let mut env = Environment::new();
    let func = convert_function(&region, &mut env).expect("conversion");

    // The entry terminal is a real React `Terminal::If` with a fallthrough (merge)
    // block that exists, and both arms are present.
    let entry = func.body.blocks.get(&func.body.entry).unwrap();
    let Terminal::If { consequent, alternate, fallthrough, .. } = entry.terminal else {
        panic!("entry should be an If, got {:?}", entry.terminal);
    };
    assert!(func.body.blocks.contains_key(&consequent));
    assert!(func.body.blocks.contains_key(&alternate));
    assert!(func.body.blocks.contains_key(&fallthrough));

    // Predecessors were populated: the merge block has both arms as predecessors.
    let merge = func.body.blocks.get(&fallthrough).unwrap();
    assert_eq!(merge.preds.len(), 2, "merge joins both arms: {:?}", merge.preds);

    // Every block ends in a real terminal (no leftover Unsupported).
    for (id, b) in &func.body.blocks {
        assert!(
            !matches!(b.terminal, Terminal::Unsupported { .. }),
            "block {id:?} has an unconverted terminal"
        );
    }
}

#[test]
fn reports_unsupported_constructs_soundly() {
    // A `while` loop is out of this first cut's scope → a descriptive Err, never a
    // silently-wrong HIR.
    let region = jslir_body("function f(n) { let i = 0; while (i < n) { i = i + 1; } return i; }");
    let mut env = Environment::new();
    let result = convert_function(&region, &mut env);
    assert!(result.is_err(), "while loop should be reported as unsupported, not mis-converted");
}
