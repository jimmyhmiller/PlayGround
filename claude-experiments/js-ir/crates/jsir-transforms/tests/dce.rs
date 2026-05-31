//! End-to-end DCE tests: source -> IR -> DCE -> IR -> source. We check both the
//! resulting JS and the elimination stats, and (most importantly) that the
//! rewrite preserves a valid round-trip.

use jsir_transforms::eliminate_dead_code;

/// Run the full pipeline and return `(js, stats)`.
fn dce(src: &str) -> (String, jsir_transforms::Stats) {
    let ir = jsir_swc::source_to_ir(src).expect("lower");
    let (out, stats) = eliminate_dead_code(&ir);
    let js = jsir_swc::ir_to_source(&out).expect("lift");
    (js, stats)
}

#[test]
fn folds_constant_if_keeps_consequent() {
    let (js, s) = dce("if (2 > 1) { a(); } else { b(); }");
    assert!(js.contains("a()"), "kept consequent: {js}");
    assert!(!js.contains("b()"), "dropped alternate: {js}");
    assert_eq!(s.if_taken_consequent, 1);
}

#[test]
fn folds_constant_if_keeps_alternate() {
    let (js, s) = dce("if (0) { a(); } else { b(); }");
    assert!(js.contains("b()"), "kept alternate: {js}");
    assert!(!js.contains("a()"), "dropped consequent: {js}");
    assert_eq!(s.if_taken_alternate, 1);
}

#[test]
fn drops_if_with_no_alternate() {
    let (js, s) = dce("if (false) { gone(); } keep();");
    assert!(!js.contains("gone"), "dropped dead branch: {js}");
    assert!(js.contains("keep()"), "kept following stmt: {js}");
    assert_eq!(s.if_taken_alternate, 1);
}

#[test]
fn removes_while_false() {
    let (js, s) = dce("while (false) { spin(); } go();");
    assert!(!js.contains("spin"), "removed dead loop: {js}");
    assert!(js.contains("go()"));
    assert_eq!(s.while_removed, 1);
}

#[test]
fn keeps_while_true() {
    let (_js, s) = dce("while (true) { spin(); }");
    assert_eq!(s.while_removed, 0, "infinite loop is not dead");
}

#[test]
fn drops_unreachable_after_return() {
    let (js, s) = dce("function f(){ return 1; console.log('dead'); } use(f);");
    assert!(!js.contains("dead"), "dropped post-return: {js}");
    assert!(s.unreachable_statements >= 1);
}

#[test]
fn keeps_hoisted_decl_after_return() {
    // `var`/`function` after a terminator still create hoisted bindings; we must
    // not drop them.
    let (js, _s) = dce("function f(){ return g(); function g(){ return 7; } } use(f);");
    assert!(js.contains("function g"), "kept hoisted fn: {js}");
}

#[test]
fn completion_through_nested_block() {
    // The `return` is nested in a block, yet code after the block is unreachable.
    let (js, s) = dce("function f(){ { return 1; } console.log('dead'); } use(f);");
    assert!(!js.contains("dead"), "completion analysis through block: {js}");
    assert!(s.unreachable_statements >= 1);
}

#[test]
fn completion_through_if_both_branches() {
    let (js, s) = dce("function f(x){ if (x) { return 1; } else { return 2; } nope(); } use(f);");
    assert!(!js.contains("nope"), "both branches return: {js}");
    assert!(s.unreachable_statements >= 1);
}

#[test]
fn if_one_branch_returns_falls_through() {
    let (js, _s) = dce("function f(x){ if (x) { return 1; } stillHere(); } use(f);");
    assert!(js.contains("stillHere"), "one-armed if falls through: {js}");
}

#[test]
fn unknown_condition_is_left_alone() {
    let (js, s) = dce("if (window.x) { a(); } else { b(); }");
    assert!(js.contains("a()") && js.contains("b()"), "both kept: {js}");
    assert_eq!(s.total_eliminations(), 0);
}

#[test]
fn removes_unused_var() {
    let (js, s) = dce("function f(){ var unused = 99; return 1; } use(f);");
    assert!(!js.contains("unused"), "removed unused var: {js}");
    assert_eq!(s.unused_vars_removed, 1);
}

#[test]
fn keeps_read_var() {
    let (js, s) = dce("function f(){ var x = 1; return x; } use(f);");
    assert!(js.contains("var x"), "kept read var: {js}");
    assert_eq!(s.unused_vars_removed, 0);
}

#[test]
fn keeps_var_with_impure_init() {
    // Never read, but the initializer has a side effect — must keep it.
    let (js, s) = dce("function f(){ var x = sideEffect(); return 1; } use(f);");
    assert!(js.contains("sideEffect"), "kept impure init: {js}");
    assert_eq!(s.unused_vars_removed, 0);
}

#[test]
fn keeps_reassigned_var() {
    let (js, _s) = dce("function f(g){ var x = 1; g(function(){ x = 2; }); } use(f);");
    assert!(js.contains("var x"), "kept reassigned var: {js}");
}

#[test]
fn cascade_unused_vars() {
    // `b` is unused; removing it makes `a` unused too.
    let (js, s) = dce("function f(){ var a = 1; var b = a; return 0; } use(f);");
    assert!(!js.contains("var a") && !js.contains("var b"), "cascaded: {js}");
    assert_eq!(s.unused_vars_removed, 2);
}

#[test]
fn removes_unused_function() {
    let (js, s) = dce("function used(){return 1;} function dead(){return 2;} used();");
    assert!(js.contains("function used"), "kept called fn: {js}");
    assert!(!js.contains("function dead"), "removed uncalled fn: {js}");
    assert_eq!(s.unused_fns_removed, 1);
}

#[test]
fn cascade_unused_functions() {
    // `dead` calls `deadHelper`; neither is reachable from `used`.
    let src = "function used(){return helper();} function helper(){return 1;} \
               function dead(){return deadHelper();} function deadHelper(){return 2;} used();";
    let (js, s) = dce(src);
    assert!(js.contains("function used") && js.contains("function helper"), "{js}");
    assert!(!js.contains("function dead") && !js.contains("function deadHelper"), "{js}");
    assert_eq!(s.unused_fns_removed, 2);
}

#[test]
fn keeps_recursive_function_conservatively() {
    // Self-reference counts as a use; we don't prove the whole cycle dead.
    let (js, _s) = dce("function loop(){ return loop(); } use(loop);");
    assert!(js.contains("function loop"), "{js}");
}

#[test]
fn guarded_dead_code_cascades_to_function_removal() {
    // The bundler scenario: a const-false guard makes the functions it calls
    // unreachable, which then get eliminated.
    let src = "function a(){return 1;} function b(){return 2;} \
               globalThis.r = a(); if (false) { globalThis.x = b(); }";
    let (js, s) = dce(src);
    assert!(js.contains("function a"), "kept used fn: {js}");
    assert!(!js.contains("function b"), "guarded-dead fn removed: {js}");
    assert_eq!(s.if_taken_alternate, 1);
    assert_eq!(s.unused_fns_removed, 1);
}

#[test]
fn string_compare_guard_folds() {
    let (js, s) = dce("if ('production' !== 'production') { dev(); } else { prod(); }");
    assert!(js.contains("prod()") && !js.contains("dev()"), "{js}");
    assert_eq!(s.if_taken_alternate, 1);
}

#[test]
fn nested_folding() {
    let (js, s) = dce("if (1) { if (2 > 3) { no(); } else { yes(); } }");
    assert!(js.contains("yes()") && !js.contains("no()"), "{js}");
    assert_eq!(s.if_taken_consequent, 1);
    assert_eq!(s.if_taken_alternate, 1);
}
