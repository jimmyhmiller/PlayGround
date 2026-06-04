//! CFG verifier tests.

use jsir_ir::{Block, BlockId, Op, Region};
use jsir_jslir::dialect;
use jsir_jslir::verify::verify_cfg;

/// Build a JSLIR function via the real pipeline and pull out its body CFG.
fn body_cfg(src: &str) -> Region {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    find_first_function_body(&jslir).expect("a lowered function body")
}

fn find_first_function_body(op: &Op) -> Option<Region> {
    if op.name == "jsir.function_declaration" {
        if let Some(body) = op.regions.get(1) {
            if dialect::region_is_cfg(body) {
                return Some(body.clone());
            }
        }
    }
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                if let Some(found) = find_first_function_body(o) {
                    return Some(found);
                }
            }
        }
    }
    None
}

#[test]
fn built_cfgs_are_well_formed() {
    for src in [
        "function f(){ let x = 1; return x; }",
        "function f(x){ if (x) { return 1; } return 2; }",
        "function f(x){ if (x) { g(); } else { h(); } return 0; }",
        "function f(n){ let i = 0; while (i < n) { i = i + 1; } return i; }",
        "function f(x){ if (x) { while (x) { x = x - 1; } } return x; }",
    ] {
        let cfg = body_cfg(src);
        assert!(verify_cfg(&cfg).is_empty(), "CFG not well-formed for: {src}\n{:?}", verify_cfg(&cfg));
    }
}

#[test]
fn verifier_catches_missing_terminator() {
    // A block whose last op is not a terminator.
    let region = Region {
        blocks: vec![Block {
            id: BlockId(0),
            args: vec![],
            ops: vec![Op::new("jsir.identifier")],
        }],
    };
    let errs = verify_cfg(&region);
    assert!(errs.iter().any(|e| e.contains("no terminator")), "{errs:?}");
}

#[test]
fn verifier_catches_dangling_successor() {
    let region = Region {
        blocks: vec![Block {
            id: BlockId(0),
            args: vec![],
            ops: vec![dialect::br(BlockId(99))], // ^bb99 doesn't exist
        }],
    };
    let errs = verify_cfg(&region);
    assert!(errs.iter().any(|e| e.contains("does not exist")), "{errs:?}");
}
