//! Comprehensive property-based and hand-written tests for the partial evaluator
//!
//! Key invariant: For any expression `e` and dynamic variables bound to values `v`,
//! evaluating the residual program `PE(e)` with `v` gives the same result as
//! evaluating `e` directly with `v`.

use proptest::prelude::*;

use partial3::ast::Expr;
use partial3::eval::eval;
use partial3::parse::parse;
use partial3::partial::{new_penv, partial_eval, residualize, PValue};
use partial3::value::{new_env, Value};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Parse and partially evaluate with given dynamic variables
fn pe(s: &str) -> PValue {
    let expr = parse(s).unwrap();
    partial_eval(&expr, &new_penv())
}

/// Parse and partially evaluate with specified dynamic variables
fn pe_with_dynamic(s: &str, dynamic_vars: &[&str]) -> PValue {
    let expr = parse(s).unwrap();
    let env = new_penv();
    for var in dynamic_vars {
        env.borrow_mut()
            .insert(var.to_string(), PValue::Dynamic(Expr::Var(var.to_string())));
    }
    partial_eval(&expr, &env)
}

/// Evaluate expression with given variable bindings
fn eval_with_bindings(s: &str, bindings: &[(&str, i64)]) -> Result<Value, String> {
    let expr = parse(s)?;
    let env = new_env();
    for (var, val) in bindings {
        env.borrow_mut().insert(var.to_string(), Value::Int(*val));
    }
    eval(&expr, &env)
}

/// Evaluate residual expression with given variable bindings
fn eval_residual_with_bindings(pv: &PValue, bindings: &[(&str, i64)]) -> Result<Value, String> {
    let residual = residualize(pv);
    let env = new_env();
    for (var, val) in bindings {
        env.borrow_mut().insert(var.to_string(), Value::Int(*val));
    }
    eval(&residual, &env)
}

/// Check semantic equivalence: PE then eval == eval directly
fn check_semantic_equivalence(code: &str, dynamic_vars: &[&str], bindings: &[(&str, i64)]) {
    // Partially evaluate with dynamic vars
    let pv = pe_with_dynamic(code, dynamic_vars);

    // Evaluate residual with concrete bindings
    let residual_result = eval_residual_with_bindings(&pv, bindings);

    // Evaluate original with concrete bindings
    let direct_result = eval_with_bindings(code, bindings);

    assert_eq!(
        residual_result, direct_result,
        "Semantic mismatch!\nCode: {}\nDynamic: {:?}\nBindings: {:?}\nResidual: {:?}\nResidual result: {:?}\nDirect result: {:?}",
        code, dynamic_vars, bindings, residualize(&pv), residual_result, direct_result
    );
}

// ============================================================================
// PROPTEST STRATEGIES
// ============================================================================

/// Generate a random integer in reasonable range
fn arb_int() -> impl Strategy<Value = i64> {
    -1000i64..1000
}

/// Generate a random boolean
fn arb_bool() -> impl Strategy<Value = bool> {
    any::<bool>()
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // -------------------------------------------------------------------------
    // Basic semantic equivalence properties
    // -------------------------------------------------------------------------

    #[test]
    fn prop_static_arithmetic_fully_reduces(a in arb_int(), b in arb_int(), c in arb_int()) {
        // (+ (+ a b) c) with all static should fully reduce
        let code = format!("(+ (+ {} {}) {})", a, b, c);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a + b + c);
            }
            other => prop_assert!(false, "Expected static, got {:?}", other),
        }
    }

    #[test]
    fn prop_static_subtraction_fully_reduces(a in arb_int(), b in arb_int()) {
        let code = format!("(- {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a - b);
            }
            other => prop_assert!(false, "Expected static, got {:?}", other),
        }
    }

    #[test]
    fn prop_static_multiplication_fully_reduces(a in -100i64..100, b in -100i64..100) {
        let code = format!("(* {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a * b);
            }
            other => prop_assert!(false, "Expected static, got {:?}", other),
        }
    }

    #[test]
    fn prop_static_comparison_fully_reduces(a in arb_int(), b in arb_int()) {
        let code = format!("(< {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a < b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_static_equality_fully_reduces(a in arb_int(), b in arb_int()) {
        let code = format!("(== {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a == b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_dynamic_add_semantic_equivalence(a in arb_int(), b in arb_int()) {
        // (+ x y) with dynamic x, y should give same result when evaluated
        check_semantic_equivalence("(+ x y)", &["x", "y"], &[("x", a), ("y", b)]);
    }

    #[test]
    fn prop_mixed_static_dynamic_add(a in arb_int(), b in arb_int()) {
        // (+ x 5) with dynamic x
        let code = format!("(+ x {})", b);
        check_semantic_equivalence(&code, &["x"], &[("x", a)]);
    }

    #[test]
    fn prop_let_static_binding(val in arb_int(), offset in arb_int()) {
        // (let x <val> (+ x <offset>)) should fully reduce
        let code = format!("(let x {} (+ x {}))", val, offset);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, val + offset);
            }
            other => prop_assert!(false, "Expected static, got {:?}", other),
        }
    }

    #[test]
    fn prop_let_dynamic_body_semantic_equiv(val in arb_int(), dyn_val in arb_int()) {
        // (let x <val> (+ x y)) with y dynamic
        let code = format!("(let x {} (+ x y))", val);
        check_semantic_equivalence(&code, &["y"], &[("y", dyn_val)]);
    }

    #[test]
    fn prop_if_true_branch_elimination(a in arb_int(), b in arb_int()) {
        // (if true a b) should reduce to a
        let code = format!("(if true {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a);
            }
            other => prop_assert!(false, "Expected static {}, got {:?}", a, other),
        }
    }

    #[test]
    fn prop_if_false_branch_elimination(a in arb_int(), b in arb_int()) {
        // (if false a b) should reduce to b
        let code = format!("(if false {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, b);
            }
            other => prop_assert!(false, "Expected static {}, got {:?}", b, other),
        }
    }

    #[test]
    fn prop_if_static_cond_eliminates_branch(a in arb_int(), b in arb_int(), x_val in arb_int()) {
        // (if (< a b) (+ x 1) (+ x 2)) with static condition
        let code = format!("(if (< {} {}) (+ x 1) (+ x 2))", a, b);
        check_semantic_equivalence(&code, &["x"], &[("x", x_val)]);

        // Also verify the residual only contains the taken branch
        let pv = pe_with_dynamic(&code, &["x"]);
        let residual = residualize(&pv);
        let residual_str = residual.to_string();
        if a < b {
            prop_assert!(!residual_str.contains("(+ x 2)"), "Should not contain else branch");
        } else {
            prop_assert!(!residual_str.contains("(+ x 1)"), "Should not contain then branch");
        }
    }

    #[test]
    fn prop_if_dynamic_cond_semantic_equiv(cond_val in arb_bool(), x_val in arb_int()) {
        // (if c (+ x 1) (+ x 2)) with dynamic c
        let cond_int = if cond_val { 1 } else { 0 };
        // We'll test with a comparison since c needs to be bool
        let code = "(if (< c 1) (+ x 1) (+ x 2))";
        check_semantic_equivalence(code, &["c", "x"], &[("c", cond_int), ("x", x_val)]);
    }

    #[test]
    fn prop_array_static_index(elements in prop::collection::vec(arb_int(), 1..5)) {
        if elements.is_empty() {
            return Ok(());
        }
        let idx = 0; // Always use first element to avoid out of bounds
        let array_str = elements.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" ");
        let code = format!("(index (array {}) {})", array_str, idx);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, elements[idx]);
            }
            other => prop_assert!(false, "Expected static {}, got {:?}", elements[idx], other),
        }
    }

    #[test]
    fn prop_array_len(elements in prop::collection::vec(arb_int(), 0..10)) {
        let array_str = elements.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" ");
        let code = format!("(len (array {}))", array_str);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, elements.len() as i64);
            }
            other => prop_assert!(false, "Expected static {}, got {:?}", elements.len(), other),
        }
    }

    #[test]
    fn prop_function_inlining(arg in arb_int()) {
        // (let f (fn (x) (+ x 1)) (call f arg)) should fully reduce
        let code = format!("(let f (fn (x) (+ x 1)) (call f {}))", arg);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, arg + 1);
            }
            other => prop_assert!(false, "Expected static, got {:?}", other),
        }
    }

    #[test]
    fn prop_closure_captures_static(captured in arb_int(), arg in arb_int()) {
        // (let a <captured> (let f (fn (x) (+ x a)) (call f <arg>)))
        let code = format!("(let a {} (let f (fn (x) (+ x a)) (call f {})))", captured, arg);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, arg + captured);
            }
            other => prop_assert!(false, "Expected static, got {:?}", other),
        }
    }

    #[test]
    fn prop_closure_with_dynamic_arg_semantic_equiv(captured in arb_int(), arg_val in arb_int()) {
        // (let a <captured> (let f (fn (x) (+ x a)) (call f y))) with y dynamic
        let code = format!("(let a {} (let f (fn (x) (+ x a)) (call f y)))", captured);
        check_semantic_equivalence(&code, &["y"], &[("y", arg_val)]);

        // Verify residual is correct
        let pv = pe_with_dynamic(&code, &["y"]);
        let residual = residualize(&pv).to_string();

        // When captured is 0, (+ y 0) folds to just y due to identity optimization
        if captured == 0 {
            prop_assert_eq!(&residual, "y", "When captured is 0, should fold to y: {}", residual);
        } else {
            prop_assert!(
                residual.contains(&captured.to_string()),
                "Residual should contain captured value {}: {}",
                captured,
                residual
            );
        }
    }

    #[test]
    fn prop_begin_returns_last(a in arb_int(), b in arb_int(), c in arb_int()) {
        let code = format!("(begin {} {} {})", a, b, c);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, c);
            }
            other => prop_assert!(false, "Expected static {}, got {:?}", c, other),
        }
    }

    #[test]
    fn prop_nested_let_shadowing(outer in arb_int(), inner in arb_int()) {
        // (let x outer (let x inner (+ x 0))) should return inner
        let code = format!("(let x {} (let x {} (+ x 0)))", outer, inner);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, inner);
            }
            other => prop_assert!(false, "Expected static {}, got {:?}", inner, other),
        }
    }

    #[test]
    fn prop_logical_and(a in arb_bool(), b in arb_bool()) {
        let code = format!("(&& {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a && b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_logical_or(a in arb_bool(), b in arb_bool()) {
        let code = format!("(|| {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a || b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_complex_expression_semantic_equiv(
        a in -10i64..10,
        b in -10i64..10,
        x_val in -100i64..100,
        y_val in -100i64..100
    ) {
        // (let c (+ a b) (+ (+ x c) (+ y c)))
        let code = format!("(let c (+ {} {}) (+ (+ x c) (+ y c)))", a, b);
        check_semantic_equivalence(&code, &["x", "y"], &[("x", x_val), ("y", y_val)]);
    }

    // -------------------------------------------------------------------------
    // New operators: bitwise, comparison, division, modulo
    // -------------------------------------------------------------------------

    #[test]
    fn prop_bitwise_and(a in -1000i64..1000, b in -1000i64..1000) {
        let code = format!("(& {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a & b);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }

    #[test]
    fn prop_bitwise_or(a in -1000i64..1000, b in -1000i64..1000) {
        let code = format!("(| {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a | b);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }

    #[test]
    fn prop_bitwise_xor(a in -1000i64..1000, b in -1000i64..1000) {
        let code = format!("(^ {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a ^ b);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }

    #[test]
    fn prop_shift_left(a in -1000i64..1000, b in 0i64..31) {
        let code = format!("(<< {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                // JS-style: convert to 32-bit, shift, result as i64
                let expected = ((a as i32) << (b as u32 & 0x1f)) as i64;
                prop_assert_eq!(result, expected);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }

    #[test]
    fn prop_shift_right(a in -1000i64..1000, b in 0i64..31) {
        let code = format!("(>> {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                // JS-style signed shift
                let expected = ((a as i32) >> (b as u32 & 0x1f)) as i64;
                prop_assert_eq!(result, expected);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }

    #[test]
    fn prop_unsigned_shift_right(a in 0i64..1000, b in 0i64..31) {
        let code = format!("(>>> {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                // JS-style unsigned shift
                let expected = ((a as u32) >> (b as u32 & 0x1f)) as i64;
                prop_assert_eq!(result, expected);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }

    #[test]
    fn prop_greater_than(a in arb_int(), b in arb_int()) {
        let code = format!("(> {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a > b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_greater_than_or_equal(a in arb_int(), b in arb_int()) {
        let code = format!("(>= {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a >= b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_less_than_or_equal(a in arb_int(), b in arb_int()) {
        let code = format!("(<= {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a <= b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_not_equal(a in arb_int(), b in arb_int()) {
        let code = format!("(!= {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Bool(result)) => {
                prop_assert_eq!(result, a != b);
            }
            other => prop_assert!(false, "Expected static bool, got {:?}", other),
        }
    }

    #[test]
    fn prop_division(a in arb_int(), b in 1i64..100) {
        // Avoid division by zero
        let code = format!("(/ {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a / b);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }

    #[test]
    fn prop_modulo(a in arb_int(), b in 1i64..100) {
        // Avoid modulo by zero
        let code = format!("(% {} {})", a, b);
        match pe(&code) {
            PValue::Static(Value::Int(result)) => {
                prop_assert_eq!(result, a % b);
            }
            other => prop_assert!(false, "Expected static int, got {:?}", other),
        }
    }
}

// ============================================================================
// HAND-WRITTEN EDGE CASE TESTS
// ============================================================================

mod edge_cases {
    use super::*;

    // -------------------------------------------------------------------------
    // Let binding edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_deeply_nested_lets() {
        let code = "(let a 1 (let b 2 (let c 3 (let d 4 (+ (+ a b) (+ c d))))))";
        match pe(code) {
            PValue::Static(Value::Int(10)) => {}
            other => panic!("Expected Static(Int(10)), got {:?}", other),
        }
    }

    #[test]
    fn test_let_shadowing_multiple_levels() {
        // x = 1, then x = 2, then x = 3, should return 3
        let code = "(let x 1 (let x 2 (let x 3 x)))";
        match pe(code) {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    #[test]
    fn test_let_uses_outer_before_shadowing() {
        // (let x 10 (let y x (let x 5 (+ x y)))) = 5 + 10 = 15
        let code = "(let x 10 (let y x (let x 5 (+ x y))))";
        match pe(code) {
            PValue::Static(Value::Int(15)) => {}
            other => panic!("Expected Static(Int(15)), got {:?}", other),
        }
    }

    #[test]
    fn test_let_dynamic_value_static_body() {
        // (let x dyn 5) with dynamic dyn - body doesn't use x, should just return 5
        let code = "(let x dyn 5)";
        match pe_with_dynamic(code, &["dyn"]) {
            PValue::Static(Value::Int(5)) => {}
            other => panic!("Expected Static(Int(5)), got {:?}", other),
        }
    }

    #[test]
    fn test_let_unused_dynamic_var_eliminated() {
        // (let x dyn 42) - x is not used, so let should be eliminated
        let pv = pe_with_dynamic("(let x dyn 42)", &["dyn"]);
        let residual = residualize(&pv).to_string();
        assert!(
            !residual.contains("let"),
            "Unused let should be eliminated: {}",
            residual
        );
    }

    // -------------------------------------------------------------------------
    // Function and closure edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_identity_function() {
        let code = "(let id (fn (x) x) (call id 42))";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_constant_function() {
        let code = "(let const (fn (x) 99) (call const 0))";
        match pe(code) {
            PValue::Static(Value::Int(99)) => {}
            other => panic!("Expected Static(Int(99)), got {:?}", other),
        }
    }

    #[test]
    fn test_two_argument_function() {
        let code = "(let add (fn (a b) (+ a b)) (call add 3 4))";
        match pe(code) {
            PValue::Static(Value::Int(7)) => {}
            other => panic!("Expected Static(Int(7)), got {:?}", other),
        }
    }

    #[test]
    fn test_zero_argument_function() {
        let code = "(let thunk (fn () 42) (call thunk))";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_higher_order_function_make_adder() {
        let code = "(let make-adder (fn (n) (fn (x) (+ x n)))
                      (let add5 (call make-adder 5)
                        (call add5 10)))";
        match pe(code) {
            PValue::Static(Value::Int(15)) => {}
            other => panic!("Expected Static(Int(15)), got {:?}", other),
        }
    }

    #[test]
    fn test_higher_order_function_with_dynamic() {
        let code = "(let make-adder (fn (n) (fn (x) (+ x n)))
                      (let add5 (call make-adder 5)
                        (call add5 y)))";
        match pe_with_dynamic(code, &["y"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "(+ y 5)");
            }
            other => panic!("Expected Dynamic (+ y 5), got {:?}", other),
        }
    }

    #[test]
    fn test_function_captures_correct_scope() {
        // Proper lexical scoping: f captures a=100 at definition time
        let code = "(let a 100
                      (let f (fn (x) (+ x a))
                        (let a 1
                          (call f 5))))";
        // f uses a=100 from definition-site scope, not a=1 from call-site
        match pe(code) {
            PValue::Static(Value::Int(105)) => {}
            other => panic!("Expected Static(Int(105)), got {:?}", other),
        }
    }

    #[test]
    fn test_function_captures_correct_scope_matches_eval() {
        // Verify PE and eval have identical lexical scoping behavior
        let code = "(let a 100
                      (let f (fn (x) (+ x a))
                        (let a 1
                          (call f 5))))";
        let expr = parse(code).unwrap();
        let eval_result = eval(&expr, &new_env()).unwrap();
        let pe_result = pe(code);

        assert_eq!(eval_result, Value::Int(105));
        match pe_result {
            PValue::Static(Value::Int(105)) => {}
            other => panic!("PE should match eval: {:?}", other),
        }
    }

    #[test]
    fn test_curried_function() {
        let code = "(let curry-add (fn (a) (fn (b) (fn (c) (+ (+ a b) c))))
                      (let f1 (call curry-add 1)
                        (let f2 (call f1 2)
                          (call f2 3))))";
        match pe(code) {
            PValue::Static(Value::Int(6)) => {}
            other => panic!("Expected Static(Int(6)), got {:?}", other),
        }
    }

    #[test]
    fn test_function_returned_as_value() {
        // Return a function, then call it
        let code = "(let mk (fn () (fn (x) (+ x 1)))
                      (let f (call mk)
                        (call f 10)))";
        match pe(code) {
            PValue::Static(Value::Int(11)) => {}
            other => panic!("Expected Static(Int(11)), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // If/conditional edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_nested_if_all_static() {
        let code = "(if true (if false 1 2) (if true 3 4))";
        match pe(code) {
            PValue::Static(Value::Int(2)) => {}
            other => panic!("Expected Static(Int(2)), got {:?}", other),
        }
    }

    #[test]
    fn test_if_with_computed_condition() {
        let code = "(if (< 3 5) 1 2)";
        match pe(code) {
            PValue::Static(Value::Int(1)) => {}
            other => panic!("Expected Static(Int(1)), got {:?}", other),
        }
    }

    #[test]
    fn test_if_condition_from_let() {
        let code = "(let flag (< 10 5) (if flag 100 200))";
        match pe(code) {
            PValue::Static(Value::Int(200)) => {}
            other => panic!("Expected Static(Int(200)), got {:?}", other),
        }
    }

    #[test]
    fn test_if_with_side_effects_in_both_branches() {
        // With static condition, only one branch should be evaluated
        let code = "(let x 0
                      (if true
                        (begin (set! x 10) x)
                        (begin (set! x 20) x)))";
        match pe(code) {
            PValue::Static(Value::Int(10)) => {}
            other => panic!("Expected Static(Int(10)), got {:?}", other),
        }
    }

    #[test]
    fn test_if_dynamic_condition_preserves_both_branches() {
        let code = "(if cond 1 2)";
        let pv = pe_with_dynamic(code, &["cond"]);
        let residual = residualize(&pv).to_string();
        assert!(residual.contains("if"), "Should have if: {}", residual);
        assert!(residual.contains("1"), "Should have then branch: {}", residual);
        assert!(residual.contains("2"), "Should have else branch: {}", residual);
    }

    #[test]
    fn test_deeply_nested_if() {
        let code = "(if (< 1 2) (if (< 2 3) (if (< 3 4) (if (< 4 5) 999 0) 0) 0) 0)";
        match pe(code) {
            PValue::Static(Value::Int(999)) => {}
            other => panic!("Expected Static(Int(999)), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Array edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_array_len() {
        let code = "(len (array))";
        match pe(code) {
            PValue::Static(Value::Int(0)) => {}
            other => panic!("Expected Static(Int(0)), got {:?}", other),
        }
    }

    #[test]
    fn test_single_element_array() {
        let code = "(index (array 42) 0)";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_array_index_last_element() {
        let code = "(index (array 1 2 3 4 5) 4)";
        match pe(code) {
            PValue::Static(Value::Int(5)) => {}
            other => panic!("Expected Static(Int(5)), got {:?}", other),
        }
    }

    #[test]
    fn test_nested_array_creation() {
        // Array containing results of expressions
        let code = "(array (+ 1 2) (- 5 1) (* 2 3))";
        match pe(code) {
            PValue::Static(Value::Array(elements)) => {
                let borrowed = elements.borrow();
                assert_eq!(borrowed.len(), 3);
                assert_eq!(borrowed[0], Value::Int(3));
                assert_eq!(borrowed[1], Value::Int(4));
                assert_eq!(borrowed[2], Value::Int(6));
            }
            other => panic!("Expected Static Array, got {:?}", other),
        }
    }

    #[test]
    fn test_array_with_dynamic_element() {
        let code = "(array 1 x 3)";
        let pv = pe_with_dynamic(code, &["x"]);
        match pv {
            PValue::Dynamic(e) => {
                let s = e.to_string();
                assert!(s.contains("array"), "Should be array: {}", s);
                assert!(s.contains("x"), "Should contain x: {}", s);
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_array_static_with_dynamic_index() {
        let code = "(index (array 10 20 30) i)";
        let pv = pe_with_dynamic(code, &["i"]);
        let residual = residualize(&pv).to_string();
        // Array should be preserved, index should remain
        assert!(residual.contains("10"), "Should contain 10: {}", residual);
        assert!(residual.contains("20"), "Should contain 20: {}", residual);
        assert!(residual.contains("30"), "Should contain 30: {}", residual);
        assert!(residual.contains("i"), "Should contain i: {}", residual);
    }

    #[test]
    fn test_array_len_with_computed_elements() {
        let code = "(len (array (+ 1 2) (+ 3 4) (+ 5 6)))";
        match pe(code) {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // While loop edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_while_false_never_executes() {
        let code = "(while false (set! x 999))";
        // Should not error even with undefined x since body is never executed
        match pe(code) {
            PValue::Static(Value::Undefined) => {}
            other => panic!("Expected Static(Undefined), got {:?}", other),
        }
    }

    #[test]
    fn test_while_simple_counter() {
        let code = "(let i 0 (begin (while (< i 3) (set! i (+ i 1))) i))";
        match pe(code) {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    #[test]
    fn test_while_sum_accumulator() {
        // Sum 1+2+3+4+5 = 15
        let code = "(let i 0
                      (let sum 0
                        (begin
                          (while (< i 5)
                            (begin
                              (set! i (+ i 1))
                              (set! sum (+ sum i))))
                          sum)))";
        match pe(code) {
            PValue::Static(Value::Int(15)) => {}
            other => panic!("Expected Static(Int(15)), got {:?}", other),
        }
    }

    #[test]
    fn test_while_with_dynamic_condition() {
        let code = "(while (< x 5) (set! x (+ x 1)))";
        let pv = pe_with_dynamic(code, &["x"]);
        match pv {
            PValue::Dynamic(e) => {
                assert!(e.to_string().contains("while"), "Should contain while");
            }
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_while_nested() {
        // Nested while: outer 0..2, inner 0..3 = 2 * 3 = 6 iterations
        let code = "(let i 0
                      (let j 0
                        (let count 0
                          (begin
                            (while (< i 2)
                              (begin
                                (set! j 0)
                                (while (< j 3)
                                  (begin
                                    (set! count (+ count 1))
                                    (set! j (+ j 1))))
                                (set! i (+ i 1))))
                            count))))";
        match pe(code) {
            PValue::Static(Value::Int(6)) => {}
            other => panic!("Expected Static(Int(6)), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Set!/mutation edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_set_simple() {
        let code = "(let x 1 (begin (set! x 42) x))";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_set_multiple_times() {
        let code = "(let x 0 (begin (set! x 1) (set! x 2) (set! x 3) x))";
        match pe(code) {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    #[test]
    fn test_set_computed_value() {
        let code = "(let x 10 (begin (set! x (+ x 5)) x))";
        match pe(code) {
            PValue::Static(Value::Int(15)) => {}
            other => panic!("Expected Static(Int(15)), got {:?}", other),
        }
    }

    #[test]
    fn test_set_in_inner_scope() {
        // NOTE: This test documents ACTUAL PE behavior.
        // In the PE, the inner let creates a new environment via clone, so set! in
        // the inner scope doesn't propagate to the outer scope. The outer x remains 1.
        let code = "(let x 1 (begin (let y 2 (set! x 99)) x))";
        match pe(code) {
            PValue::Static(Value::Int(1)) => {}
            other => panic!("Expected Static(Int(1)) [PE behavior], got {:?}", other),
        }
    }

    #[test]
    fn test_set_in_inner_scope_matches_eval() {
        // Both PE and eval have the same scoping behavior for set!
        // Inner let creates a clone of the environment, so set! doesn't propagate out
        let code = "(let x 1 (begin (let y 2 (set! x 99)) x))";
        let expr = parse(code).unwrap();
        let eval_result = eval(&expr, &new_env()).unwrap();
        let pe_result = pe(code);

        // Both should return 1 (set! in inner scope doesn't affect outer x)
        assert_eq!(eval_result, Value::Int(1));
        match pe_result {
            PValue::Static(Value::Int(1)) => {}
            other => panic!("PE should match eval: {:?}", other),
        }
    }

    #[test]
    fn test_set_to_dynamic_value() {
        let code = "(let x 0 (begin (set! x y) x))";
        let pv = pe_with_dynamic(code, &["y"]);
        let residual = residualize(&pv).to_string();
        assert!(residual.contains("y"), "Should contain y: {}", residual);
    }

    #[test]
    fn test_set_dynamic_then_return_emits_variable_reference() {
        // When a variable is set to a dynamic expression, subsequent references
        // should emit the variable name, not the expression.
        // This is a regression test for a bug where (set! x (+ y 1)) followed by x
        // would emit (+ y 1) instead of x.
        let code = "(let x 0 (begin (set! x (+ y 1)) x))";
        let pv = pe_with_dynamic(code, &["y"]);
        let residual = residualize(&pv).to_string();
        // The final reference to x should be just "x", not "(+ y 1)"
        // The residual should look like (let x 0 (begin (set! x (+ y 1)) x))
        assert!(residual.ends_with(" x))"), "Final expr should be variable x, got: {}", residual);
    }

    #[test]
    fn test_while_with_dynamic_set_preserves_assignments() {
        // When a while loop body contains dynamic set! statements,
        // those statements should be preserved in the residual.
        // With aggressive unrolling, the loop is fully unrolled when the condition
        // becomes statically false (state = -1, so state > 0 is false).
        let code = "(let arr y
                      (let result 0
                        (let state 100
                          (begin
                            (while (> state 0)
                              (begin
                                (set! result (index arr 0))
                                (set! state -1)))
                            result))))";
        let pv = pe_with_dynamic(code, &["y"]);
        let residual = residualize(&pv).to_string();
        // The residual should contain the set! for result, not have it optimized away
        assert!(residual.contains("(set! result"),
            "Residual should preserve set! result, got: {}", residual);
        // With aggressive unrolling, state becomes static (-1) so no set! state is needed
        // The loop is fully unrolled since the condition becomes statically false
        assert!(!residual.contains("(while"),
            "Loop should be fully unrolled since condition becomes static false, got: {}", residual);
        // And the final expression should be the variable "result", not the index expression
        assert!(residual.contains(" result))"),
            "Final expr should be variable result, got: {}", residual);
    }

    // -------------------------------------------------------------------------
    // Begin/sequencing edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_begin_empty() {
        let code = "(begin)";
        match pe(code) {
            PValue::Static(Value::Bool(false)) => {}
            other => panic!("Expected Static(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_begin_single() {
        let code = "(begin 42)";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_begin_with_side_effects() {
        let code = "(let x 0 (begin (set! x 1) (set! x (+ x 1)) (set! x (+ x 1)) x))";
        match pe(code) {
            PValue::Static(Value::Int(3)) => {}
            other => panic!("Expected Static(Int(3)), got {:?}", other),
        }
    }

    #[test]
    fn test_begin_dynamic_at_end() {
        let code = "(begin 1 2 x)";
        match pe_with_dynamic(code, &["x"]) {
            PValue::Dynamic(e) => {
                assert_eq!(e.to_string(), "x");
            }
            other => panic!("Expected Dynamic x, got {:?}", other),
        }
    }

    #[test]
    fn test_begin_dynamic_in_middle() {
        let code = "(let r 0 (begin (set! r x) r))";
        let pv = pe_with_dynamic(code, &["x"]);
        match pv {
            PValue::Dynamic(_) => {}
            other => panic!("Expected Dynamic, got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Binary operation edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_identity() {
        // (+ 0 x) should fold to just x
        let code = "(+ 0 x)";
        let pv = pe_with_dynamic(code, &["x"]);
        let residual = residualize(&pv).to_string();
        assert_eq!(residual, "x", "Should fold to x: {}", residual);
    }

    #[test]
    fn test_multiply_by_zero() {
        // Static 0 * dynamic x should fold to Static(Int(0))
        let code = "(* 0 x)";
        let pv = pe_with_dynamic(code, &["x"]);
        match pv {
            PValue::Static(Value::Int(0)) => {}
            other => panic!("Expected Static(Int(0)), got {:?}", other),
        }
    }

    #[test]
    fn test_multiply_by_one() {
        // (* 1 x) should fold to just x
        let code = "(* 1 x)";
        let pv = pe_with_dynamic(code, &["x"]);
        let residual = residualize(&pv).to_string();
        assert_eq!(residual, "x", "Should fold to x: {}", residual);
    }

    #[test]
    fn test_subtract_from_self() {
        let code = "(let x 42 (- x x))";
        match pe(code) {
            PValue::Static(Value::Int(0)) => {}
            other => panic!("Expected Static(Int(0)), got {:?}", other),
        }
    }

    #[test]
    fn test_negative_numbers() {
        let code = "(+ -5 -3)";
        match pe(code) {
            PValue::Static(Value::Int(-8)) => {}
            other => panic!("Expected Static(Int(-8)), got {:?}", other),
        }
    }

    #[test]
    fn test_comparison_equal_values() {
        let code = "(< 5 5)";
        match pe(code) {
            PValue::Static(Value::Bool(false)) => {}
            other => panic!("Expected Static(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_equality_of_bools() {
        let code = "(== true true)";
        match pe(code) {
            PValue::Static(Value::Bool(true)) => {}
            other => panic!("Expected Static(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_chain_of_comparisons() {
        // (< 1 2) && (< 2 3) && (< 3 4)
        let code = "(&& (&& (< 1 2) (< 2 3)) (< 3 4))";
        match pe(code) {
            PValue::Static(Value::Bool(true)) => {}
            other => panic!("Expected Static(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_or_short_circuit_would_matter() {
        // (|| true anything) - but we still evaluate both statically
        let code = "(|| true false)";
        match pe(code) {
            PValue::Static(Value::Bool(true)) => {}
            other => panic!("Expected Static(Bool(true)), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Complex/weird scenarios
    // -------------------------------------------------------------------------

    #[test]
    fn test_function_returning_function_returning_function() {
        let code = "(let f (fn () (fn () (fn () 42)))
                      (call (call (call f))))";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_mutual_reference_through_closure() {
        // Proper lexical scoping: f captures a=10 at definition time
        // (call f 5) = 5 + 10 = 15, plus current a=20 = 35
        let code = "(let a 10
                      (let f (fn (x) (+ x a))
                        (let a 20
                          (+ (call f 5) a))))";
        match pe(code) {
            PValue::Static(Value::Int(35)) => {}
            other => panic!("Expected Static(Int(35)), got {:?}", other),
        }
    }

    #[test]
    fn test_mutual_reference_through_closure_matches_eval() {
        // Verify PE and eval have identical lexical scoping behavior
        let code = "(let a 10
                      (let f (fn (x) (+ x a))
                        (let a 20
                          (+ (call f 5) a))))";
        let expr = parse(code).unwrap();
        let eval_result = eval(&expr, &new_env()).unwrap();
        let pe_result = pe(code);

        assert_eq!(eval_result, Value::Int(35));
        match pe_result {
            PValue::Static(Value::Int(35)) => {}
            other => panic!("PE should match eval: {:?}", other),
        }
    }

    #[test]
    fn test_array_in_let() {
        let code = "(let arr (array 1 2 3) (+ (index arr 0) (index arr 2)))";
        match pe(code) {
            PValue::Static(Value::Int(4)) => {}
            other => panic!("Expected Static(Int(4)), got {:?}", other),
        }
    }

    #[test]
    fn test_object_with_static_named_property_allows_prop_access() {
        // When an object contains a StaticNamed property (like an array bound to a variable),
        // property access should still be optimized since the values are known at PE time.
        let code = "(let arr (array 1 2 3)
                      (let obj (object (data arr) (count 42))
                        (+ (len (prop obj \"data\")) (prop obj \"count\"))))";
        match pe(code) {
            // len([1,2,3]) = 3, count = 42, total = 45
            PValue::Static(Value::Int(45)) => {}
            other => panic!("Expected Static(Int(45)), got {:?}", other),
        }
    }

    #[test]
    fn test_object_with_closure_property_allows_call() {
        // Object containing a closure - accessing and calling should work
        let code = "(let obj (object (add (fn (x y) (+ x y))))
                      (call (prop obj \"add\") 10 20))";
        match pe(code) {
            PValue::Static(Value::Int(30)) => {}
            other => panic!("Expected Static(Int(30)), got {:?}", other),
        }
    }

    #[test]
    fn test_len_in_condition() {
        let code = "(if (< (len (array 1 2)) 5) 100 200)";
        match pe(code) {
            PValue::Static(Value::Int(100)) => {}
            other => panic!("Expected Static(Int(100)), got {:?}", other),
        }
    }

    #[test]
    fn test_array_index_in_function() {
        let code = "(let get-second (fn (arr) (index arr 1))
                      (call get-second (array 10 20 30)))";
        match pe(code) {
            PValue::Static(Value::Int(20)) => {}
            other => panic!("Expected Static(Int(20)), got {:?}", other),
        }
    }

    #[test]
    fn test_while_with_array_length() {
        let code = "(let arr (array 1 2 3)
                      (let i 0
                        (let sum 0
                          (begin
                            (while (< i (len arr))
                              (begin
                                (set! sum (+ sum (index arr i)))
                                (set! i (+ i 1))))
                            sum))))";
        match pe(code) {
            PValue::Static(Value::Int(6)) => {}
            other => panic!("Expected Static(Int(6)), got {:?}", other),
        }
    }

    #[test]
    fn test_factorial_iterative() {
        // factorial(5) = 120
        let code = "(let n 5
                      (let result 1
                        (let i 1
                          (begin
                            (while (< i (+ n 1))
                              (begin
                                (set! result (* result i))
                                (set! i (+ i 1))))
                            result))))";
        match pe(code) {
            PValue::Static(Value::Int(120)) => {}
            other => panic!("Expected Static(Int(120)), got {:?}", other),
        }
    }

    #[test]
    fn test_fibonacci_iterative() {
        // NOTE: Due to PE's scoping behavior with set!, we can't use let inside while body.
        // Instead, we use a temp variable defined at the outer scope.
        // fib(10) = 55
        let code = "(let n 10
                      (let a 0
                        (let b 1
                          (let i 0
                            (let temp 0
                              (begin
                                (while (< i n)
                                  (begin
                                    (set! temp b)
                                    (set! b (+ a b))
                                    (set! a temp)
                                    (set! i (+ i 1))))
                                a))))))";
        match pe(code) {
            PValue::Static(Value::Int(55)) => {}
            other => panic!("Expected Static(Int(55)), got {:?}", other),
        }
    }

    #[test]
    fn test_compose_functions() {
        // compose(f, g)(x) = f(g(x))
        let code = "(let compose (fn (f g) (fn (x) (call f (call g x))))
                      (let add1 (fn (x) (+ x 1))
                        (let double (fn (x) (* x 2))
                          (let add1-then-double (call compose double add1)
                            (call add1-then-double 5)))))";
        // (5 + 1) * 2 = 12
        match pe(code) {
            PValue::Static(Value::Int(12)) => {}
            other => panic!("Expected Static(Int(12)), got {:?}", other),
        }
    }

    #[test]
    fn test_partial_application_simulation() {
        // Simulate partial application: add(a)(b) = a + b
        let code = "(let add (fn (a) (fn (b) (+ a b)))
                      (let add10 (call add 10)
                        (+ (call add10 5) (call add10 7))))";
        // 15 + 17 = 32
        match pe(code) {
            PValue::Static(Value::Int(32)) => {}
            other => panic!("Expected Static(Int(32)), got {:?}", other),
        }
    }

    #[test]
    fn test_dynamic_var_in_deeply_nested_expression() {
        let code = "(+ (+ (+ (+ (+ x 1) 2) 3) 4) 5)";
        check_semantic_equivalence(code, &["x"], &[("x", 100)]);
    }

    #[test]
    fn test_multiple_dynamic_vars() {
        let code = "(+ (+ (+ a b) c) d)";
        check_semantic_equivalence(code, &["a", "b", "c", "d"], &[("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    }

    #[test]
    fn test_mixed_static_dynamic_complex() {
        let code = "(let static1 10
                      (let static2 20
                        (+ (+ static1 dyn1) (+ static2 dyn2))))";
        check_semantic_equivalence(code, &["dyn1", "dyn2"], &[("dyn1", 5), ("dyn2", 7)]);
    }

    #[test]
    fn test_if_with_function_in_branches() {
        let code = "(let f (fn (x) (+ x 1))
                      (let g (fn (x) (+ x 2))
                        (if true (call f 5) (call g 5))))";
        match pe(code) {
            PValue::Static(Value::Int(6)) => {}
            other => panic!("Expected Static(Int(6)), got {:?}", other),
        }
    }

    #[test]
    fn test_array_of_function_results() {
        let code = "(let f (fn (x) (* x 2))
                      (array (call f 1) (call f 2) (call f 3)))";
        match pe(code) {
            PValue::Static(Value::Array(elements)) => {
                assert_eq!(*elements.borrow(), vec![Value::Int(2), Value::Int(4), Value::Int(6)]);
            }
            other => panic!("Expected Static Array, got {:?}", other),
        }
    }

    #[test]
    fn test_power_function() {
        // power(base, exp) - iterative
        let code = "(let base 2
                      (let exp 10
                        (let result 1
                          (let i 0
                            (begin
                              (while (< i exp)
                                (begin
                                  (set! result (* result base))
                                  (set! i (+ i 1))))
                              result)))))";
        match pe(code) {
            PValue::Static(Value::Int(1024)) => {}
            other => panic!("Expected Static(Int(1024)), got {:?}", other),
        }
    }

    #[test]
    fn test_gcd_iterative() {
        // gcd(48, 18) = 6 using subtraction method
        let code = "(let a 48
                      (let b 18
                        (begin
                          (while (== (== a b) false)
                            (if (< a b)
                              (set! b (- b a))
                              (set! a (- a b))))
                          a)))";
        match pe(code) {
            PValue::Static(Value::Int(6)) => {}
            other => panic!("Expected Static(Int(6)), got {:?}", other),
        }
    }

    #[test]
    fn test_residual_preserves_variable_names() {
        let code = "(let meaningful_name x (+ meaningful_name 1))";
        let pv = pe_with_dynamic(code, &["x"]);
        let residual = residualize(&pv).to_string();
        assert!(
            residual.contains("meaningful_name"),
            "Should preserve variable name: {}",
            residual
        );
    }

    #[test]
    fn test_double_negation_static() {
        let code = "(- 0 (- 0 42))";
        match pe(code) {
            PValue::Static(Value::Int(42)) => {}
            other => panic!("Expected Static(Int(42)), got {:?}", other),
        }
    }

    #[test]
    fn test_boolean_equality() {
        let code = "(== (== true true) true)";
        match pe(code) {
            PValue::Static(Value::Bool(true)) => {}
            other => panic!("Expected Static(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_complex_boolean_expression() {
        // (a && b) || (c && d) where all are static
        let code = "(|| (&& true false) (&& true true))";
        match pe(code) {
            PValue::Static(Value::Bool(true)) => {}
            other => panic!("Expected Static(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_let_in_array() {
        let code = "(array (let x 1 (+ x 1)) (let y 2 (* y 2)) (let z 3 (+ z z)))";
        match pe(code) {
            PValue::Static(Value::Array(elements)) => {
                assert_eq!(*elements.borrow(), vec![Value::Int(2), Value::Int(4), Value::Int(6)]);
            }
            other => panic!("Expected Static Array, got {:?}", other),
        }
    }

    #[test]
    fn test_function_taking_function() {
        let code = "(let apply (fn (f x) (call f x))
                      (let inc (fn (n) (+ n 1))
                        (call apply inc 10)))";
        match pe(code) {
            PValue::Static(Value::Int(11)) => {}
            other => panic!("Expected Static(Int(11)), got {:?}", other),
        }
    }

    #[test]
    fn test_twice_function() {
        // Apply a function twice
        let code = "(let twice (fn (f) (fn (x) (call f (call f x))))
                      (let add3 (fn (n) (+ n 3))
                        (let add6 (call twice add3)
                          (call add6 10))))";
        match pe(code) {
            PValue::Static(Value::Int(16)) => {}
            other => panic!("Expected Static(Int(16)), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // Semantic equivalence verification tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_semantic_equiv_simple() {
        check_semantic_equivalence("(+ x 1)", &["x"], &[("x", 5)]);
    }

    #[test]
    fn test_semantic_equiv_multiple_uses() {
        check_semantic_equivalence("(+ (+ x x) (+ x x))", &["x"], &[("x", 3)]);
    }

    #[test]
    fn test_semantic_equiv_with_static() {
        check_semantic_equivalence("(let a 10 (+ a x))", &["x"], &[("x", 5)]);
    }

    #[test]
    fn test_semantic_equiv_if() {
        check_semantic_equivalence("(if (< x 5) (+ x 1) (- x 1))", &["x"], &[("x", 3)]);
        check_semantic_equivalence("(if (< x 5) (+ x 1) (- x 1))", &["x"], &[("x", 7)]);
    }

    #[test]
    fn test_semantic_equiv_function() {
        check_semantic_equivalence(
            "(let f (fn (a) (+ a 10)) (call f x))",
            &["x"],
            &[("x", 5)],
        );
    }

    #[test]
    fn test_semantic_equiv_closure() {
        check_semantic_equivalence(
            "(let n 5 (let f (fn (x) (+ x n)) (call f y)))",
            &["y"],
            &[("y", 10)],
        );
    }

    #[test]
    fn test_semantic_equiv_nested_functions() {
        check_semantic_equivalence(
            "(let make-adder (fn (n) (fn (x) (+ x n)))
               (let add5 (call make-adder 5)
                 (call add5 y)))",
            &["y"],
            &[("y", 20)],
        );
    }
}

// ============================================================================
// RESIDUAL BODY PARTIAL EVALUATION TESTS
// ============================================================================

mod residual_pe {
    use super::*;

    #[test]
    fn test_while_body_optimized() {
        // When while condition is dynamic, the body should still be PE'd
        // x is unknown but initialized, so while loop can't unroll
        let code = "(let x (call getSomeValue) (while (>= x 0) (begin (set! x (+ x 0)) x)))";
        let pv = pe_with_dynamic(code, &["x"]);
        let residual = residualize(&pv).to_string();
        // (+ x 0) should fold to x
        assert!(!residual.contains("+ x 0"), "Should fold (+ x 0) to x: {}", residual);
    }

    #[test]
    fn test_function_body_optimized() {
        // Function bodies should be PE'd even when emitted as residual
        let code = "(fn (x) (+ x 0))";
        let pv = pe(code);
        let residual = residualize(&pv).to_string();
        // The function body (+ x 0) should be optimized to just x
        assert!(!residual.contains("+ x 0"), "Should fold (+ x 0) to x in fn body: {}", residual);
    }

    #[test]
    fn test_function_with_mutation_body_optimized() {
        // Even functions with captured mutations should have their bodies PE'd
        let code = "(let count 0 (fn () (begin (set! count (+ count 1)) (* count 1))))";
        let pv = pe(code);
        let residual = residualize(&pv).to_string();
        // (* count 1) should fold to count
        assert!(!residual.contains("* count 1"), "Should fold (* count 1) to count: {}", residual);
    }

    #[test]
    fn test_nested_optimization_in_residual() {
        // Complex nested expressions in residual should all be optimized
        let code = "(fn (a b c) (begin (+ a 0) (- b 0) (* c 1) (| a 0) (^ b 0)))";
        let pv = pe(code);
        let residual = residualize(&pv).to_string();
        // All identity operations should be folded away
        assert!(!residual.contains("+ a 0"), "Should fold (+ a 0): {}", residual);
        assert!(!residual.contains("- b 0"), "Should fold (- b 0): {}", residual);
        assert!(!residual.contains("* c 1"), "Should fold (* c 1): {}", residual);
        assert!(!residual.contains("| a 0"), "Should fold (| a 0): {}", residual);
        assert!(!residual.contains("^ b 0"), "Should fold (^ b 0): {}", residual);
    }

    #[test]
    fn test_bitwise_and_zero_in_function() {
        // A function that uses bitwise AND with 0 should fold to 0
        let code = "(fn (x) (& x 0))";
        let pv = pe(code);
        let residual = residualize(&pv).to_string();
        // The function body (& x 0) should fold to 0
        assert!(residual.contains("(fn (x) 0)"), "Should fold (& x 0) to 0 in fn: {}", residual);
    }

    #[test]
    fn test_multiply_zero_in_function() {
        // A function that multiplies by 0 should fold to 0
        let code = "(fn (x) (* x 0))";
        let pv = pe(code);
        let residual = residualize(&pv).to_string();
        // The function body (* x 0) should fold to 0
        assert!(residual.contains("(fn (x) 0)"), "Should fold (* x 0) to 0 in fn: {}", residual);
    }

    #[test]
    fn test_dead_code_elimination_in_begin() {
        // Pure expressions before the last one should be eliminated
        let code = "(begin 1 2 3 (call f))";
        let pv = pe_with_dynamic(code, &["f"]);
        let residual = residualize(&pv).to_string();
        // 1, 2, 3 should be eliminated - only the call remains
        assert!(!residual.contains("1"), "Should eliminate 1: {}", residual);
        assert!(!residual.contains("2"), "Should eliminate 2: {}", residual);
        assert!(!residual.contains("3"), "Should eliminate 3: {}", residual);
        assert!(residual.contains("call f"), "Should keep the call: {}", residual);
    }

    #[test]
    fn test_dead_code_keeps_last_expression() {
        // The last expression should be kept even if pure
        let code = "(begin (call f) 42)";
        let pv = pe_with_dynamic(code, &["f"]);
        let residual = residualize(&pv).to_string();
        // Both should be present - call has side effects, 42 is return value
        assert!(residual.contains("call f"), "Should keep the call: {}", residual);
        assert!(residual.contains("42"), "Should keep return value 42: {}", residual);
    }

    #[test]
    fn test_array_index_out_of_bounds_folds_to_undefined() {
        // Indexing an empty array with any index should return undefined
        let code = "(index (array) 0)";
        let pv = pe(code);
        assert!(matches!(pv, PValue::Static(Value::Undefined)), "Should fold to undefined: {:?}", pv);

        // Negative index should also return undefined
        let code = "(index (array 1 2 3) -1)";
        let pv = pe(code);
        assert!(matches!(pv, PValue::Static(Value::Undefined)), "Negative index should fold to undefined: {:?}", pv);
    }

    #[test]
    fn test_array_len_folds() {
        // Length of static array should fold
        let code = "(len (array 1 2 3))";
        let pv = pe(code);
        assert!(matches!(pv, PValue::Static(Value::Int(3))), "Should fold to 3: {:?}", pv);

        // Empty array
        let code = "(len (array))";
        let pv = pe(code);
        assert!(matches!(pv, PValue::Static(Value::Int(0))), "Should fold to 0: {:?}", pv);
    }

    // -------------------------------------------------------------------------
    // New optimization tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_switch_cases_eliminated() {
        // Switch cases with only break should be eliminated
        // Construct the AST directly since the parser doesn't support switch
        use partial3::ast::Expr;
        let switch_expr = Expr::Switch {
            discriminant: Box::new(Expr::Var("x".to_string())),
            cases: vec![
                (Expr::Int(1), vec![Expr::Break]),                          // empty - should be eliminated
                (Expr::Int(2), vec![Expr::Call(Box::new(Expr::Var("f".to_string())), vec![])]), // has code
                (Expr::Int(3), vec![Expr::Break]),                          // empty - should be eliminated
            ],
            default: None,
        };
        let env = new_penv();
        env.borrow_mut().insert("x".to_string(), PValue::Dynamic(Expr::Var("x".to_string())));
        env.borrow_mut().insert("f".to_string(), PValue::Dynamic(Expr::Var("f".to_string())));
        let pv = partial_eval(&switch_expr, &env);
        let residual = residualize(&pv).to_string();
        // Only case 2 should remain (has actual code)
        assert!(!residual.contains("case 1"), "Empty case 1 should be eliminated: {}", residual);
        assert!(residual.contains("case 2"), "Case 2 with code should remain: {}", residual);
        assert!(!residual.contains("case 3"), "Empty case 3 should be eliminated: {}", residual);
    }

    #[test]
    fn test_logical_and_short_circuit_true() {
        // (&& true x) should fold to x
        let code = "(&& true x)";
        let pv = pe_with_dynamic(code, &["x"]);
        let residual = residualize(&pv).to_string();
        assert_eq!(residual, "x", "(&& true x) should fold to x: {}", residual);
    }

    #[test]
    fn test_logical_and_short_circuit_false() {
        // (&& false x) should fold to false
        let code = "(&& false x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(false))), "(&& false x) should fold to false: {:?}", pv);
    }

    #[test]
    fn test_logical_or_short_circuit_true() {
        // (|| true x) should fold to true
        let code = "(|| true x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(true))), "(|| true x) should fold to true: {:?}", pv);
    }

    #[test]
    fn test_logical_or_short_circuit_false() {
        // (|| false x) should fold to x
        let code = "(|| false x)";
        let pv = pe_with_dynamic(code, &["x"]);
        let residual = residualize(&pv).to_string();
        assert_eq!(residual, "x", "(|| false x) should fold to x: {}", residual);
    }

    #[test]
    fn test_self_comparison_equal() {
        // (== x x) should fold to true
        let code = "(== x x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(true))), "(== x x) should fold to true: {:?}", pv);
    }

    #[test]
    fn test_self_comparison_not_equal() {
        // (!= x x) should fold to false
        let code = "(!= x x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(false))), "(!= x x) should fold to false: {:?}", pv);
    }

    #[test]
    fn test_self_comparison_lte() {
        // (<= x x) should fold to true
        let code = "(<= x x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(true))), "(<= x x) should fold to true: {:?}", pv);
    }

    #[test]
    fn test_self_comparison_gte() {
        // (>= x x) should fold to true
        let code = "(>= x x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(true))), "(>= x x) should fold to true: {:?}", pv);
    }

    #[test]
    fn test_self_comparison_lt() {
        // (< x x) should fold to false
        let code = "(< x x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(false))), "(< x x) should fold to false: {:?}", pv);
    }

    #[test]
    fn test_self_comparison_gt() {
        // (> x x) should fold to false
        let code = "(> x x)";
        let pv = pe_with_dynamic(code, &["x"]);
        assert!(matches!(pv, PValue::Static(Value::Bool(false))), "(> x x) should fold to false: {:?}", pv);
    }

    #[test]
    fn test_if_identical_branches_pure_condition() {
        // (if condition x x) with pure condition should fold to x
        let code = "(if flag 42 42)";
        let pv = pe_with_dynamic(code, &["flag"]);
        assert!(matches!(pv, PValue::Static(Value::Int(42))), "(if flag 42 42) should fold to 42: {:?}", pv);
    }

    #[test]
    fn test_if_identical_branches_pure_condition_with_var() {
        // (if condition x x) with pure condition and same variable should fold to x
        let code = "(if flag y y)";
        let pv = pe_with_dynamic(code, &["flag", "y"]);
        let residual = residualize(&pv).to_string();
        assert_eq!(residual, "y", "(if flag y y) should fold to y: {}", residual);
    }

    #[test]
    fn test_if_identical_branches_impure_condition() {
        // (if (call f) x x) with impure condition should keep condition for side effects
        let code = "(if (call f) 42 42)";
        let pv = pe_with_dynamic(code, &["f"]);
        let residual = residualize(&pv).to_string();
        // Should produce (begin (call f) 42)
        assert!(residual.contains("call f"), "Should keep impure condition: {}", residual);
        assert!(residual.contains("42"), "Should keep the value: {}", residual);
    }

    #[test]
    fn test_double_negation_dynamic() {
        // (! (! x)) should fold to x
        // Construct the AST directly since the parser doesn't support !
        use partial3::ast::Expr;
        let double_neg = Expr::LogNot(Box::new(Expr::LogNot(Box::new(Expr::Var("x".to_string())))));
        let env = new_penv();
        env.borrow_mut().insert("x".to_string(), PValue::Dynamic(Expr::Var("x".to_string())));
        let pv = partial_eval(&double_neg, &env);
        let residual = residualize(&pv).to_string();
        assert_eq!(residual, "x", "(! (! x)) should fold to x: {}", residual);
    }

    #[test]
    fn test_double_negation_static_bool() {
        // (! (! true)) should fold to true
        // Construct the AST directly since the parser doesn't support !
        use partial3::ast::Expr;
        let double_neg = Expr::LogNot(Box::new(Expr::LogNot(Box::new(Expr::Bool(true)))));
        let pv = partial_eval(&double_neg, &new_penv());
        assert!(matches!(pv, PValue::Static(Value::Bool(true))), "(! (! true)) should fold to true: {:?}", pv);
    }

    #[test]
    fn test_complex_expression_with_new_optimizations() {
        // Complex expression combining multiple optimizations
        let code = "(let result (&& true (|| false (== x x)))
                      (if result y y))";
        let pv = pe_with_dynamic(code, &["x", "y"]);
        let residual = residualize(&pv).to_string();
        // (== x x) = true, (|| false true) = true, (&& true true) = true
        // (if true y y) = y
        assert_eq!(residual, "y", "Complex expression should fold to y: {}", residual);
    }

    #[test]
    fn test_self_comparison_not_applied_to_impure() {
        // Self-comparison should NOT be applied to impure expressions like function calls
        let code = "(== (call f) (call f))";
        let pv = pe_with_dynamic(code, &["f"]);
        let residual = residualize(&pv).to_string();
        // Should NOT fold because function calls may have side effects / different results
        assert!(residual.contains("call f"), "Impure self-comparison should not fold: {}", residual);
    }
}

// ============================================================================
// TYPED ARRAY TESTS
// ============================================================================

mod typed_array_tests {
    use partial3::ast::Expr;
    use partial3::opaque::OpaqueRegistry;
    use partial3::parse::parse;
    use partial3::partial::{new_penv, partial_eval, with_opaque_registry, PValue};
    use partial3::value::Value;

    /// Parse and partially evaluate with opaque handlers
    fn pe_with_opaque(s: &str) -> PValue {
        let expr = parse(s).unwrap();
        with_opaque_registry(OpaqueRegistry::with_builtins(), || {
            partial_eval(&expr, &new_penv())
        })
    }

    /// Parse and partially evaluate with specified dynamic variables and opaque handlers
    fn pe_with_dynamic_and_opaque(s: &str, dynamic_vars: &[&str]) -> PValue {
        let expr = parse(s).unwrap();
        let env = new_penv();
        for var in dynamic_vars {
            env.borrow_mut()
                .insert(var.to_string(), PValue::Dynamic(Expr::Var(var.to_string())));
        }
        with_opaque_registry(OpaqueRegistry::with_builtins(), || {
            partial_eval(&expr, &env)
        })
    }

    #[test]
    fn test_arraybuffer_creation() {
        let pv = pe_with_opaque("(new ArrayBuffer 8)");
        match pv {
            PValue::Static(Value::Opaque { label, state, .. }) => {
                assert!(label.contains("ArrayBuffer"), "Should create ArrayBuffer: {}", label);
                assert!(state.is_some(), "Should have state");
            }
            _ => panic!("Expected static opaque value"),
        }
    }

    #[test]
    fn test_arraybuffer_bytelength() {
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 16)
                                   (prop buf "byteLength"))"#);
        match pv {
            PValue::Static(Value::Int(16)) => {}
            _ => panic!("Expected byteLength=16, got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_creation() {
        // Test that DataView creation works by checking we can access byteLength
        // (This proves the DataView state is properly created and accessible)
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 8)
                                   (let view (new DataView buf)
                                     (prop view "byteLength")))"#);
        match pv {
            PValue::Static(Value::Int(8)) => {}
            _ => panic!("Expected DataView with byteLength=8, got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_bytelength() {
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 32)
                                   (let view (new DataView buf)
                                     (prop view "byteLength")))"#);
        match pv {
            PValue::Static(Value::Int(32)) => {}
            _ => panic!("Expected byteLength=32, got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_setint8_getint8() {
        // Set a value and get it back
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setInt8") 0 42)
                                       (call (prop view "getInt8") 0))))"#);
        match pv {
            PValue::Static(Value::Int(42)) => {}
            _ => panic!("Expected 42, got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_getuint8() {
        // Test unsigned byte reading
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setUint8") 0 255)
                                       (call (prop view "getUint8") 0))))"#);
        match pv {
            PValue::Static(Value::Int(255)) => {}
            _ => panic!("Expected 255, got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_signed_byte() {
        // Test signed byte conversion (255 as unsigned = -1 as signed)
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setUint8") 0 255)
                                       (call (prop view "getInt8") 0))))"#);
        match pv {
            PValue::Static(Value::Int(-1)) => {}
            _ => panic!("Expected -1 (signed byte from 255), got {:?}", pv),
        }
    }

    #[test]
    fn test_buffer_sharing_dataview() {
        // Two DataViews on the same buffer should share memory
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view1 (new DataView buf)
                                     (let view2 (new DataView buf)
                                       (begin
                                         (call (prop view1 "setInt8") 0 99)
                                         (call (prop view2 "getInt8") 0)))))"#);
        match pv {
            PValue::Static(Value::Int(99)) => {}
            _ => panic!("Expected 99 (shared buffer), got {:?}", pv),
        }
    }

    #[test]
    fn test_uint8array_creation() {
        // Test that Uint8Array creation from ArrayBuffer works by checking we can access length
        // (This proves the Uint8Array state is properly created and accessible)
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 8)
                                   (let arr (new Uint8Array buf)
                                     (prop arr "length")))"#);
        match pv {
            PValue::Static(Value::Int(8)) => {}
            _ => panic!("Expected Uint8Array with length=8, got {:?}", pv),
        }
    }

    #[test]
    fn test_uint8array_from_array() {
        let pv = pe_with_opaque("(new Uint8Array (array 1 2 3 4 5))");
        match pv {
            PValue::Static(Value::Opaque { label, state, .. }) => {
                assert!(label.contains("Uint8Array"), "Should create Uint8Array: {}", label);
                assert!(state.is_some(), "Should have state");
            }
            _ => panic!("Expected static opaque Uint8Array value"),
        }
    }

    #[test]
    fn test_uint8array_index_access() {
        // Access elements by index
        let pv = pe_with_opaque("(let arr (new Uint8Array (array 10 20 30 40))
                                   (index arr 2))");
        match pv {
            PValue::Static(Value::Int(30)) => {}
            _ => panic!("Expected 30, got {:?}", pv),
        }
    }

    #[test]
    fn test_uint8array_length() {
        let pv = pe_with_opaque(r#"(let arr (new Uint8Array (array 1 2 3 4 5))
                                   (prop arr "length"))"#);
        match pv {
            PValue::Static(Value::Int(5)) => {}
            _ => panic!("Expected length=5, got {:?}", pv),
        }
    }

    #[test]
    fn test_buffer_sharing_uint8array_dataview() {
        // Uint8Array and DataView on same buffer should share memory
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let arr (new Uint8Array buf)
                                     (let view (new DataView buf)
                                       (begin
                                         (call (prop view "setInt8") 0 77)
                                         (index arr 0)))))"#);
        match pv {
            PValue::Static(Value::Int(77)) => {}
            _ => panic!("Expected 77 (shared buffer), got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_arithmetic() {
        // Compute 10 + 5 after storing values
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setInt8") 0 10)
                                       (+ (call (prop view "getInt8") 0) 5))))"#);
        match pv {
            PValue::Static(Value::Int(15)) => {}
            _ => panic!("Expected 15, got {:?}", pv),
        }
    }

    #[test]
    fn test_uint8array_out_of_bounds() {
        let pv = pe_with_opaque("(let arr (new Uint8Array (array 1 2 3))
                                   (index arr 10))");
        match pv {
            PValue::Static(Value::Undefined) => {}
            _ => panic!("Expected undefined for out of bounds, got {:?}", pv),
        }
    }

    #[test]
    fn test_uint8array_negative_index() {
        let pv = pe_with_opaque("(let arr (new Uint8Array (array 1 2 3))
                                   (index arr -1))");
        match pv {
            PValue::Static(Value::Undefined) => {}
            _ => panic!("Expected undefined for negative index, got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_int16_big_endian() {
        // Big-endian (default): 0x0102 = 258
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setInt8") 0 1)
                                       (call (prop view "setInt8") 1 2)
                                       (call (prop view "getInt16") 0))))"#);
        match pv {
            PValue::Static(Value::Int(258)) => {}
            _ => panic!("Expected 258 (big-endian int16), got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_int16_little_endian() {
        // Little-endian: 0x0201 = 513
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setInt8") 0 1)
                                       (call (prop view "setInt8") 1 2)
                                       (call (prop view "getInt16") 0 true))))"#);
        match pv {
            PValue::Static(Value::Int(513)) => {}
            _ => panic!("Expected 513 (little-endian int16), got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_setint16() {
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 4)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setInt16") 0 4660)
                                       (call (prop view "getInt16") 0))))"#);
        match pv {
            PValue::Static(Value::Int(4660)) => {}  // 0x1234 = 4660
            _ => panic!("Expected 4660 (0x1234), got {:?}", pv),
        }
    }

    #[test]
    fn test_dataview_int32() {
        let pv = pe_with_opaque(r#"(let buf (new ArrayBuffer 8)
                                   (let view (new DataView buf)
                                     (begin
                                       (call (prop view "setInt32") 0 305419896)
                                       (call (prop view "getInt32") 0))))"#);
        match pv {
            PValue::Static(Value::Int(305419896)) => {}  // 0x12345678 = 305419896
            _ => panic!("Expected 305419896 (0x12345678), got {:?}", pv),
        }
    }

    #[test]
    fn test_dynamic_index_residualizes() {
        // When index is dynamic, should produce residual
        let pv = pe_with_dynamic_and_opaque(
            "(let arr (new Uint8Array (array 1 2 3))
               (index arr x))",
            &["x"]
        );
        match pv {
            PValue::Dynamic(expr) => {
                let residual = expr.to_string();
                assert!(residual.contains("index"), "Should have residual index: {}", residual);
            }
            _ => panic!("Expected dynamic with residual, got {:?}", pv),
        }
    }

    #[test]
    fn test_dynamic_offset_residualizes() {
        // When offset is dynamic, should produce residual method call
        let pv = pe_with_dynamic_and_opaque(
            r#"(let buf (new ArrayBuffer 4)
               (let view (new DataView buf)
                 (call (prop view "getInt8") x)))"#,
            &["x"]
        );
        match pv {
            PValue::Dynamic(expr) => {
                let residual = expr.to_string();
                assert!(residual.contains("call"), "Should have residual call: {}", residual);
            }
            _ => panic!("Expected dynamic with residual, got {:?}", pv),
        }
    }

    // ========================================================================
    // TextDecoder Tests
    // ========================================================================

    #[test]
    fn test_textdecoder_decode_hello() {
        // "Hello" in ASCII/UTF-8: [72, 101, 108, 108, 111]
        let pv = pe_with_opaque(r#"
            (let decoder (new TextDecoder)
              (let arr (new Uint8Array (array 72 101 108 108 111))
                (call (prop decoder "decode") arr)))
        "#);
        match pv {
            PValue::Static(Value::String(s)) => {
                assert_eq!(s, "Hello", "Should decode to 'Hello'");
            }
            _ => panic!("Expected Static String, got {:?}", pv),
        }
    }

    #[test]
    fn test_textdecoder_decode_unicode() {
        // "Héllo" in UTF-8: [72, 195, 169, 108, 108, 111] (é is 0xC3 0xA9)
        let pv = pe_with_opaque(r#"
            (let decoder (new TextDecoder)
              (let arr (new Uint8Array (array 72 195 169 108 108 111))
                (call (prop decoder "decode") arr)))
        "#);
        match pv {
            PValue::Static(Value::String(s)) => {
                assert_eq!(s, "Héllo", "Should decode UTF-8 with accented char");
            }
            _ => panic!("Expected Static String, got {:?}", pv),
        }
    }

    #[test]
    fn test_textdecoder_decode_empty() {
        let pv = pe_with_opaque(r#"
            (let decoder (new TextDecoder)
              (let arr (new Uint8Array (array))
                (call (prop decoder "decode") arr)))
        "#);
        match pv {
            PValue::Static(Value::String(s)) => {
                assert_eq!(s, "", "Should decode empty array to empty string");
            }
            _ => panic!("Expected Static String, got {:?}", pv),
        }
    }

    #[test]
    fn test_textdecoder_with_buffer_from_dataview() {
        // Write bytes via DataView, then decode via TextDecoder
        let pv = pe_with_opaque(r#"
            (let buf (new ArrayBuffer 5)
              (let view (new DataView buf)
                (let decoder (new TextDecoder)
                  (let arr (new Uint8Array buf)
                    (begin
                      (call (prop view "setUint8") 0 72)
                      (call (prop view "setUint8") 1 105)
                      (call (prop decoder "decode") arr))))))
        "#);
        match pv {
            PValue::Static(Value::String(s)) => {
                // Only first 2 bytes are set, rest are 0
                assert!(s.starts_with("Hi"), "Should decode bytes written via DataView, got: {:?}", s);
            }
            _ => panic!("Expected Static String, got {:?}", pv),
        }
    }

    #[test]
    fn test_textdecoder_decode_with_arithmetic() {
        // Decode and get length
        let pv = pe_with_opaque(r#"
            (let decoder (new TextDecoder)
              (let arr (new Uint8Array (array 72 105))
                (prop (call (prop decoder "decode") arr) "length")))
        "#);
        match pv {
            PValue::Static(Value::Int(2)) => {}
            _ => panic!("Expected length 2, got {:?}", pv),
        }
    }
}
