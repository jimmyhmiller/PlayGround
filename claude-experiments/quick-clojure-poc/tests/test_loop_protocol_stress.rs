// Stress tests for loop/recur with protocol calls
//
// These tests verify that the register allocation and spill/restore logic
// works correctly when protocol calls (like -conj) are used inside loops.
// This was a significant bug that caused crashes due to temp register clobbering.

use quick_clojure_poc::*;
use std::sync::Arc;
use std::cell::UnsafeCell;

/// Helper to load clojure.core into the compiler/runtime
fn load_core(compiler: &mut compiler::Compiler, runtime: &Arc<UnsafeCell<gc_runtime::GCRuntime>>) {
    use std::fs;
    use std::io::BufRead;

    let file = fs::File::open("src/clojure/core.clj").expect("Failed to open core.clj");
    let reader = std::io::BufReader::new(file);
    let mut accumulated = String::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }

        accumulated.push_str(&line);
        accumulated.push('\n');

        if let Ok(val) = reader::read(&accumulated) {
            if let Ok(ast) = clojure_ast::analyze(&val) {
                if let Ok(result_val) = compiler.compile(&ast) {
                    let result_reg = compiler.ensure_register(result_val);
                    let instructions = compiler.take_instructions();
                    let mut codegen = arm_codegen::Arm64CodeGen::new();

                    if codegen.compile(&instructions, &result_reg, 0).is_ok() {
                        let _ = codegen.execute();
                    }
                }
            }
            accumulated.clear();
        }
    }
}

/// Helper function to run code and get the raw tagged result
fn run_and_get_tagged(code: &str) -> i64 {
    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());
    let mut compiler = compiler::Compiler::new(runtime.clone());

    // Load clojure.core to get EMPTY-LIST, -conj, first, etc.
    load_core(&mut compiler, &runtime);

    let val = reader::read(code).expect(&format!("Failed to read: {}", code));
    let ast = clojure_ast::analyze(&val).expect(&format!("Failed to analyze: {}", code));

    let result_val = compiler.compile(&ast).expect(&format!("Compiler failed for: {}", code));
    let result_reg = compiler.ensure_register(result_val);
    let instructions = compiler.take_instructions();

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    codegen.compile(&instructions, &result_reg, 0).expect(&format!("Codegen failed for: {}", code));
    codegen.execute().expect(&format!("Execute failed for: {}", code))
}

/// Helper to check that code runs without crashing and returns a heap object (list)
fn run_expect_list(code: &str) {
    let result = run_and_get_tagged(code);
    // Heap objects have tags 5 (closure/list) or 6 (other heap objects)
    let tag = result & 0b111;
    assert!(tag == 5 || tag == 6, "Expected heap object (tag 5 or 6), got tag {} for: {}", tag, code);
}

/// Helper to check that code returns a specific integer
fn run_expect_int(code: &str, expected: i64) {
    let result = run_and_get_tagged(code);
    // Integers have tag 0b000, value is shifted left by 3
    let tag = result & 0b111;
    assert_eq!(tag, 0, "Expected integer (tag 0), got tag {} for: {}", tag, code);
    let value = result >> 3;
    assert_eq!(value, expected, "Expected {}, got {} for: {}", expected, value, code);
}

// =============================================================================
// Basic Protocol Call Tests
// =============================================================================

#[test]
fn test_simple_conj() {
    run_expect_list("(-conj EMPTY-LIST 42)");
}

#[test]
fn test_nested_conj_5() {
    run_expect_list("(-conj (-conj (-conj (-conj (-conj EMPTY-LIST 0) 1) 2) 3) 4)");
}

#[test]
fn test_first_of_conj() {
    run_expect_int("(first (-conj EMPTY-LIST 42))", 42);
}

#[test]
fn test_first_of_nested_conj() {
    // conj prepends, so first should be 4
    run_expect_int("(first (-conj (-conj (-conj (-conj (-conj EMPTY-LIST 0) 1) 2) 3) 4))", 4);
}

// =============================================================================
// Loop with Protocol Calls
// =============================================================================

#[test]
fn test_loop_conj_2_iterations() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 2)
            (recur (+ i 1) (-conj acc i))
            acc))
    "#;
    run_expect_list(code);
}

#[test]
fn test_loop_conj_10_iterations() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 10)
            (recur (+ i 1) (-conj acc i))
            acc))
    "#;
    run_expect_list(code);
}

#[test]
fn test_loop_conj_50_iterations() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 50)
            (recur (+ i 1) (-conj acc i))
            (first acc)))
    "#;
    run_expect_int(code, 49);
}

#[test]
fn test_loop_conj_100_iterations() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 100)
            (recur (+ i 1) (-conj acc i))
            (first acc)))
    "#;
    run_expect_int(code, 99);
}

#[test]
fn test_loop_conj_1000_iterations() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 1000)
            (recur (+ i 1) (-conj acc i))
            (first acc)))
    "#;
    run_expect_int(code, 999);
}

// =============================================================================
// Nested Loops
// =============================================================================

#[test]
fn test_nested_loops_5x3() {
    // Outer loop 5 times, inner loop 3 times each
    let code = r#"
        (loop [i 0 result EMPTY-LIST]
          (if (< i 5)
            (let [inner-result (loop [j 0 acc result]
                                 (if (< j 3)
                                   (recur (+ j 1) (-conj acc (+ (* i 10) j)))
                                   acc))]
              (recur (+ i 1) inner-result))
            (first result)))
    "#;
    // Last value added: i=4, j=2 -> 4*10+2 = 42
    run_expect_int(code, 42);
}

#[test]
fn test_nested_loops_with_multiple_lists() {
    // Build 3 lists simultaneously with nested inner loops
    let code = r#"
        (loop [outer 0 list-a EMPTY-LIST list-b EMPTY-LIST]
          (if (< outer 4)
            (let [new-a (loop [inner 0 acc list-a]
                          (if (< inner 2)
                            (recur (+ inner 1) (-conj acc (* outer 10)))
                            acc))
                  new-b (loop [inner 0 acc list-b]
                          (if (< inner 2)
                            (recur (+ inner 1) (-conj acc (* outer 100)))
                            acc))]
              (recur (+ outer 1) new-a new-b))
            (+ (first list-a) (first list-b))))
    "#;
    // list-a first: 30 (outer=3), list-b first: 300 (outer=3)
    run_expect_int(code, 330);
}

// =============================================================================
// Multiple Lists Built Simultaneously
// =============================================================================

#[test]
fn test_three_lists_simultaneous() {
    let code = r#"
        (loop [i 0 list1 EMPTY-LIST list2 EMPTY-LIST list3 EMPTY-LIST]
          (if (< i 15)
            (recur (+ i 1)
                   (-conj list1 i)
                   (-conj list2 (* i 2))
                   (-conj list3 (* i i)))
            (+ (+ (first list1) (first list2)) (first list3))))
    "#;
    // i=14: list1 first=14, list2 first=28, list3 first=196
    run_expect_int(code, 238);
}

#[test]
fn test_conditional_list_building() {
    // Build evens and odds lists separately
    let code = r#"
        (loop [i 0 evens EMPTY-LIST odds EMPTY-LIST]
          (if (< i 20)
            (if (= 0 (- i (* 2 (/ i 2))))
              (recur (+ i 1) (-conj evens i) odds)
              (recur (+ i 1) evens (-conj odds i)))
            (+ (first evens) (first odds))))
    "#;
    // evens first: 18, odds first: 19
    run_expect_int(code, 37);
}

// =============================================================================
// Closures with Protocol Calls
// =============================================================================

#[test]
fn test_closure_calling_conj_in_loop() {
    let code = r#"
        (let [add-to-list (fn [lst x] (-conj lst x))]
          (loop [i 0 acc EMPTY-LIST]
            (if (< i 10)
              (recur (+ i 1) (add-to-list acc i))
              (first acc))))
    "#;
    run_expect_int(code, 9);
}

#[test]
fn test_closure_capturing_list() {
    let code = r#"
        (let [base-list (-conj (-conj (-conj EMPTY-LIST 1) 2) 3)
              add-to-base (fn [x] (-conj base-list x))]
          (loop [i 0 results EMPTY-LIST]
            (if (< i 10)
              (let [new-list (add-to-base (* i 10))
                    first-val (first new-list)]
                (recur (+ i 1) (-conj results first-val)))
              (first results))))
    "#;
    // Last iteration i=9: add-to-base(90) -> first is 90
    run_expect_int(code, 90);
}

#[test]
fn test_nested_closures_with_lists() {
    let code = r#"
        (let [outer-list (-conj EMPTY-LIST 100)
              make-inner (fn [x]
                           (let [inner-list (-conj outer-list x)]
                             (fn [y]
                               (let [deep-list (-conj inner-list y)]
                                 (first deep-list)))))]
          (let [inner1 (make-inner 200)]
            (loop [i 0 results EMPTY-LIST]
              (if (< i 5)
                (let [r1 (inner1 i)]
                  (recur (+ i 1) (-conj results r1)))
                (first results)))))
    "#;
    // Last iteration i=4: inner1(4) -> first of (-conj (-conj outer-list 200) 4) = 4
    run_expect_int(code, 4);
}

#[test]
fn test_higher_order_functions_with_conj() {
    let code = r#"
        (let [make-multiplier (fn [factor]
                                (fn [x] (* factor x)))]
          (let [mult2 (make-multiplier 2)
                mult3 (make-multiplier 3)]
            (loop [i 0 results EMPTY-LIST]
              (if (< i 5)
                (let [val (+ (mult2 i) (mult3 i))]
                  (recur (+ i 1) (-conj results val)))
                (first results)))))
    "#;
    // i=4: mult2(4)=8, mult3(4)=12, sum=20
    run_expect_int(code, 20);
}

// =============================================================================
// Deep Let Bindings
// =============================================================================

#[test]
fn test_deep_let_bindings_with_conj() {
    let code = r#"
        (let [a (-conj EMPTY-LIST 1)
              b (-conj a 2)
              c (-conj b 3)
              d (-conj c 4)
              e (-conj d 5)
              f (-conj e 6)
              g (-conj f 7)
              h (-conj g 8)
              i (-conj h 9)
              j (-conj i 10)]
          (+ (first j) (first a)))
    "#;
    // first of j = 10, first of a = 1
    run_expect_int(code, 11);
}

#[test]
fn test_multiple_conjs_per_iteration() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 10)
            (let [step1 (-conj acc i)
                  step2 (-conj step1 (+ i 100))
                  step3 (-conj step2 (+ i 200))]
              (recur (+ i 1) step3))
            (first acc)))
    "#;
    // Last iteration i=9: step3 first is 9+200=209
    run_expect_int(code, 209);
}

// =============================================================================
// Register Pressure Tests
// =============================================================================

#[test]
fn test_many_live_variables_8() {
    let code = r#"
        (let [a 1 b 2 c 3 d 4 e 5 f 6 g 7 h 8]
          (let [sum1 (+ a b)
                sum2 (+ c d)
                sum3 (+ e f)
                sum4 (+ g h)]
            (let [total (+ (+ sum1 sum2) (+ sum3 sum4))]
              (loop [i 0 acc EMPTY-LIST]
                (if (< i 5)
                  (recur (+ i 1) (-conj acc (+ total i)))
                  (first acc))))))
    "#;
    // sum1=3, sum2=7, sum3=11, sum4=15, total=36
    // i=4: total+4 = 40
    run_expect_int(code, 40);
}

#[test]
fn test_many_live_variables_16() {
    let code = r#"
        (let [v1 1 v2 2 v3 3 v4 4 v5 5 v6 6 v7 7 v8 8
              v9 9 v10 10 v11 11 v12 12 v13 13 v14 14 v15 15 v16 16]
          (let [s1 (+ v1 v2)
                s2 (+ v3 v4)
                s3 (+ v5 v6)
                s4 (+ v7 v8)]
            (let [t1 (+ s1 s2)
                  t2 (+ s3 s4)]
              (loop [i 0 acc EMPTY-LIST]
                (if (< i 3)
                  (recur (+ i 1) (-conj acc (+ (+ t1 t2) i)))
                  (first acc))))))
    "#;
    // s1=3, s2=7, s3=11, s4=15, t1=10, t2=26, sum=36
    // i=2: 36+2=38
    run_expect_int(code, 38);
}

// =============================================================================
// Deeply Nested Conditionals
// =============================================================================

#[test]
fn test_deeply_nested_ifs_with_conj() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 20)
            (let [val (if (< i 5)
                        (if (< i 2)
                          (* i 1000)
                          (* i 100))
                        (if (< i 10)
                          (* i 10)
                          i))]
              (recur (+ i 1) (-conj acc val)))
            (first acc)))
    "#;
    // i=19: val = 19 (since 19 >= 10)
    run_expect_int(code, 19);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_loop_zero_iterations() {
    let code = r#"
        (loop [i 0 acc EMPTY-LIST]
          (if (< i 0)
            (recur (+ i 1) (-conj acc i))
            acc))
    "#;
    run_expect_list(code);
}

#[test]
fn test_nested_20_conjs() {
    let code = r#"
        (first
          (-conj (-conj (-conj (-conj (-conj
          (-conj (-conj (-conj (-conj (-conj
          (-conj (-conj (-conj (-conj (-conj
          (-conj (-conj (-conj (-conj (-conj
            EMPTY-LIST 0) 1) 2) 3) 4) 5) 6) 7) 8) 9)
            10) 11) 12) 13) 14) 15) 16) 17) 18) 19))
    "#;
    run_expect_int(code, 19);
}

// =============================================================================
// Ultimate Stress Test
// =============================================================================

#[test]
fn test_ultimate_stress() {
    // Nested loops + closures + multiple lists
    let code = r#"
        (let [make-multiplier (fn [factor] (fn [x] (* factor x)))]
          (let [mult2 (make-multiplier 2)
                mult3 (make-multiplier 3)]
            (loop [outer 0 list-a EMPTY-LIST list-b EMPTY-LIST]
              (if (< outer 5)
                (let [new-a (loop [inner 0 acc list-a]
                              (if (< inner 3)
                                (recur (+ inner 1) (-conj acc (mult2 (+ (* outer 10) inner))))
                                acc))
                      new-b (loop [inner 0 acc list-b]
                              (if (< inner 3)
                                (recur (+ inner 1) (-conj acc (mult3 (+ (* outer 10) inner))))
                                acc))]
                  (recur (+ outer 1) new-a new-b))
                (+ (first list-a) (first list-b))))))
    "#;
    // outer=4, inner=2: value = 4*10+2 = 42
    // mult2(42) = 84, mult3(42) = 126, sum = 210
    run_expect_int(code, 210);
}
