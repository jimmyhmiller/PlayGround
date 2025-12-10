// Test that builtins work as first-class values
use quick_clojure_poc::*;
use std::sync::Arc;
use std::cell::UnsafeCell;

fn run_test(code: &str, expected: i64) {
    let val = reader::read(code).expect(&format!("Failed to read: {}", code));
    let ast = clojure_ast::analyze(&val).expect(&format!("Failed to analyze: {}", code));

    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    let mut compiler = compiler::Compiler::new(runtime);
    let result_reg = compiler.compile(&ast).expect(&format!("Compiler failed for: {}", code));
    let instructions = compiler.take_instructions();

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    codegen.compile(&instructions, &result_reg, 0).expect(&format!("Codegen failed for: {}", code));
    let tagged_result = codegen.execute()
        .expect(&format!("Execute failed for: {}", code));

    // Untag the result (integers are tagged by shifting left 3 bits)
    let result = tagged_result >> 3;

    assert_eq!(result, expected, "Failed for: {}", code);
}

fn main() {
    // Test 1: Direct call still works (inline compiled)
    println!("Test 1: (+ 3 4) direct call");
    run_test("(+ 3 4)", 7);
    println!("  PASS: 7");

    // Test 2: def my-add + and call it
    println!("\nTest 2: (do (def my-add +) (my-add 3 4))");
    run_test("(do (def my-add +) (my-add 3 4))", 7);
    println!("  PASS: 7");

    // Test 3: def my-mul * and call it
    println!("\nTest 3: (do (def my-mul *) (my-mul 6 7))");
    run_test("(do (def my-mul *) (my-mul 6 7))", 42);
    println!("  PASS: 42");

    // Test 4: def my-sub - and call it
    println!("\nTest 4: (do (def my-sub -) (my-sub 10 3))");
    run_test("(do (def my-sub -) (my-sub 10 3))", 7);
    println!("  PASS: 7");

    // Test 5: Use builtin directly in a function
    println!("\nTest 5: ((fn [f] (f 3 4)) +)");
    run_test("((fn [f] (f 3 4)) +)", 7);
    println!("  PASS: 7");

    println!("\n=== ALL TESTS PASSED ===");
}
