//! The pass pipeline skeleton: every upstream React Compiler pass is present and
//! ordered, with a first-class implemented/stub status, and the driver runs the
//! implemented subset over a function body.

use jsir_ir::Op;
use jsir_jslir::pipeline::{pipeline, run_pipeline, PassStatus, Stage};

#[test]
fn full_pipeline_is_present_and_ordered() {
    let passes = pipeline();
    // The complete upstream pipeline (lowering → … → codegen_function).
    assert_eq!(passes.len(), 62, "every upstream pass must have a placeholder");

    // Front and back anchors match upstream.
    assert_eq!(passes.first().unwrap().name, "lower");
    assert_eq!(passes.last().unwrap().name, "codegen_function");

    // The four ported passes appear, in the right relative order.
    let names: Vec<&str> = passes.iter().map(|p| p.name).collect();
    let idx = |n: &str| names.iter().position(|x| *x == n).unwrap();
    assert!(idx("enter_ssa") < idx("eliminate_redundant_phi"));
    assert!(idx("eliminate_redundant_phi") < idx("constant_propagation"));
    assert!(idx("constant_propagation") < idx("dead_code_elimination"));
    // …and the big unported spine is present as placeholders.
    for p in [
        "infer_mutation_aliasing_effects",
        "infer_reactive_scope_variables",
        "propagate_scope_dependencies_hir",
        "build_reactive_function",
        "codegen_function",
    ] {
        assert!(names.contains(&p), "missing pipeline placeholder: {p}");
    }
}

#[test]
fn status_split_is_honest() {
    let passes = pipeline();
    let implemented: Vec<&str> = passes
        .iter()
        .filter(|p| p.status == PassStatus::Implemented)
        .map(|p| p.name)
        .collect();
    // `lower` (frontend) + the four real HIR passes.
    assert_eq!(
        implemented,
        vec![
            "lower",
            "enter_ssa",
            "eliminate_redundant_phi",
            "constant_propagation",
            "dead_code_elimination",
        ],
        "exactly the ported passes are marked Implemented"
    );
    let stubs = passes.iter().filter(|p| p.status == PassStatus::Stub).count();
    assert_eq!(stubs, 57);
}

#[test]
fn every_reactive_scope_pass_is_accounted_for() {
    // The reactive-scopes spine (the bulk of the remaining work) is fully listed.
    let passes = pipeline();
    let reactive = passes.iter().filter(|p| p.stage == Stage::ReactiveScopes).count();
    assert!(reactive >= 17, "reactive-scope + codegen passes present, got {reactive}");
}

#[test]
fn driver_reproduces_fold_prune_dce() {
    // Running the whole pipeline (implemented passes + no-op stubs) over a body
    // gives exactly the constant-fold + branch-prune + DCE result.
    let src = "function f() { let x = 4; let y = x + 1; if (true) { return y; } return 0; }";
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (mut jslir, _) = jsir_jslir::build_jslir(&file);
    let base = max_value(&jslir) + 1;
    let report = run_first_fn(&mut jslir, base);
    let js = jsir_swc::ir_to_source(&jsir_jslir::lift_jslir(&jslir)).unwrap();

    // `x + 1` folds to `5` (so `y = 5`); the dead `x` is DCE'd; `if (true)` is
    // pruned, dropping the unreachable `return 0`. (Copy-propagating `y` into the
    // `return` is a later pass, so `return y;` is expected here.)
    assert!(js.contains("let y = 5;"), "x+1 folds to 5, got: {js}");
    assert!(js.contains("return y;"), "got: {js}");
    assert!(!js.contains("let x"), "dead `x` eliminated, got: {js}");
    assert!(!js.contains("return 0"), "unreachable branch pruned, got: {js}");

    let report = report.expect("a function body ran");
    assert_eq!(report.implemented, 5);
    assert_eq!(report.total, 62);
}

fn run_first_fn(op: &mut Op, base: u32) -> Option<jsir_jslir::pipeline::PipelineReport> {
    if op.name == "jsir.function_declaration" {
        if let Some(b) = op.regions.get_mut(1) {
            if jsir_jslir::dialect::region_is_cfg(b) {
                return Some(run_pipeline(b, base));
            }
        }
    }
    for r in &mut op.regions {
        for b in &mut r.blocks {
            for o in &mut b.ops {
                if let Some(rep) = run_first_fn(o, base) {
                    return Some(rep);
                }
            }
        }
    }
    None
}

fn max_value(op: &Op) -> u32 {
    let mut m = op.results.iter().map(|v| v.0).max().unwrap_or(0);
    for r in &op.regions {
        for b in &r.blocks {
            m = m.max(b.args.iter().map(|v| v.0).max().unwrap_or(0));
            for o in &b.ops {
                m = m.max(max_value(o));
            }
        }
    }
    m
}
