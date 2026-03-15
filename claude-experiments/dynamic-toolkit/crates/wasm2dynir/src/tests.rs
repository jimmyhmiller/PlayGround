use crate::{translate_wasm, translate_wasm_module};
use dynir::interp::*;
use dynir::ir::Module;
use dynir::verify::verify;
use dynir::NoGcRoots;
use dynvalue::NanBox;

fn run_wasm_i32(wat: &str, args: &[u64]) -> i32 {
    let wasm = wat::parse_str(wat).expect("failed to parse WAT");
    let (func, _imports) = translate_wasm(&wasm).expect("failed to translate");
    verify(&func).expect("IR verification failed");
    let (module, entry) = Module::from_function(func.clone());
    let roots = NoGcRoots;
    let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    match interp.run(entry, args).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("expected Value, got {:?}", other),
    }
}

fn run_wasm_i64(wat: &str, args: &[u64]) -> i64 {
    let wasm = wat::parse_str(wat).expect("failed to parse WAT");
    let (func, _imports) = translate_wasm(&wasm).expect("failed to translate");
    verify(&func).expect("IR verification failed");
    let (module, entry) = Module::from_function(func.clone());
    let roots = NoGcRoots;
    let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    match interp.run(entry, args).unwrap() {
        InterpResult::Value(v) => v as i64,
        other => panic!("expected Value, got {:?}", other),
    }
}

#[test]
fn return_const() {
    let result = run_wasm_i32(
        r#"(module
            (func (export "main") (result i32)
                i32.const 42))"#,
        &[],
    );
    assert_eq!(result, 42);
}

#[test]
fn add_two() {
    let result = run_wasm_i32(
        r#"(module
            (func (export "add") (param i32) (param i32) (result i32)
                local.get 0
                local.get 1
                i32.add))"#,
        &[10, 32],
    );
    assert_eq!(result, 42);
}

#[test]
fn arithmetic() {
    let result = run_wasm_i32(
        r#"(module
            (func (export "calc") (param i32) (param i32) (result i32)
                local.get 0
                local.get 1
                i32.mul
                local.get 0
                i32.add))"#,
        &[5, 7],
    );
    // 5 * 7 + 5 = 40
    assert_eq!(result, 40);
}

#[test]
fn if_else() {
    let wat = r#"(module
        (func (export "max") (param i32) (param i32) (result i32)
            local.get 0
            local.get 1
            i32.gt_s
            if (result i32)
                local.get 0
            else
                local.get 1
            end))"#;
    assert_eq!(run_wasm_i32(wat, &[10, 20]), 20);
    assert_eq!(run_wasm_i32(wat, &[30, 20]), 30);
}

#[test]
fn simple_loop() {
    // sum 0..n
    let wat = r#"(module
        (func (export "sum") (param $n i32) (result i32)
            (local $i i32)
            (local $acc i32)
            i32.const 0
            local.set $i
            i32.const 0
            local.set $acc
            block $exit
                loop $loop
                    local.get $i
                    local.get $n
                    i32.ge_s
                    br_if $exit
                    local.get $acc
                    local.get $i
                    i32.add
                    local.set $acc
                    local.get $i
                    i32.const 1
                    i32.add
                    local.set $i
                    br $loop
                end
            end
            local.get $acc))"#;
    assert_eq!(run_wasm_i32(wat, &[10]), 45);
    assert_eq!(run_wasm_i32(wat, &[0]), 0);
    assert_eq!(run_wasm_i32(wat, &[100]), 4950);
}

#[test]
fn fibonacci() {
    let wat = r#"(module
        (func (export "fib") (param $n i32) (result i32)
            (local $a i32)
            (local $b i32)
            (local $i i32)
            (local $tmp i32)
            i32.const 0
            local.set $a
            i32.const 1
            local.set $b
            i32.const 0
            local.set $i
            block $exit
                loop $loop
                    local.get $i
                    local.get $n
                    i32.ge_s
                    br_if $exit
                    local.get $a
                    local.get $b
                    i32.add
                    local.set $tmp
                    local.get $b
                    local.set $a
                    local.get $tmp
                    local.set $b
                    local.get $i
                    i32.const 1
                    i32.add
                    local.set $i
                    br $loop
                end
            end
            local.get $a))"#;
    assert_eq!(run_wasm_i32(wat, &[0]), 0);
    assert_eq!(run_wasm_i32(wat, &[1]), 1);
    assert_eq!(run_wasm_i32(wat, &[10]), 55);
    assert_eq!(run_wasm_i32(wat, &[20]), 6765);
}

#[test]
fn factorial() {
    let wat = r#"(module
        (func (export "fact") (param $n i32) (result i32)
            (local $result i32)
            i32.const 1
            local.set $result
            block $exit
                loop $loop
                    local.get $n
                    i32.const 1
                    i32.le_s
                    br_if $exit
                    local.get $result
                    local.get $n
                    i32.mul
                    local.set $result
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.set $n
                    br $loop
                end
            end
            local.get $result))"#;
    assert_eq!(run_wasm_i32(wat, &[1]), 1);
    assert_eq!(run_wasm_i32(wat, &[5]), 120);
    assert_eq!(run_wasm_i32(wat, &[10]), 3628800);
}

#[test]
fn nested_if() {
    let wat = r#"(module
        (func (export "clamp") (param $x i32) (param $lo i32) (param $hi i32) (result i32)
            local.get $x
            local.get $lo
            i32.lt_s
            if (result i32)
                local.get $lo
            else
                local.get $x
                local.get $hi
                i32.gt_s
                if (result i32)
                    local.get $hi
                else
                    local.get $x
                end
            end))"#;
    assert_eq!(run_wasm_i32(wat, &[5, 0, 10]), 5);
    assert_eq!(run_wasm_i32(wat, &[(-5i32) as u64, 0, 10]), 0);
    assert_eq!(run_wasm_i32(wat, &[15, 0, 10]), 10);
}

#[test]
fn void_if() {
    // if without result, used for side effects (setting a local)
    let wat = r#"(module
        (func (export "abs") (param $x i32) (result i32)
            local.get $x
            i32.const 0
            i32.lt_s
            if
                i32.const 0
                local.get $x
                i32.sub
                local.set $x
            end
            local.get $x))"#;
    assert_eq!(run_wasm_i32(wat, &[5]), 5);
    assert_eq!(run_wasm_i32(wat, &[(-5i32) as u64 & 0xFFFFFFFF]), 5);
}

#[test]
fn local_tee() {
    let wat = r#"(module
        (func (export "test") (param $x i32) (result i32)
            local.get $x
            i32.const 10
            i32.add
            local.tee $x
            local.get $x
            i32.add))"#;
    // x = x + 10, then x + x
    assert_eq!(run_wasm_i32(wat, &[5]), 30); // (5+10) + (5+10) = 30
}

#[test]
fn i64_arithmetic() {
    let wat = r#"(module
        (func (export "add64") (param i64) (param i64) (result i64)
            local.get 0
            local.get 1
            i64.add))"#;
    assert_eq!(
        run_wasm_i64(wat, &[1_000_000_000_000, 2_000_000_000_000]),
        3_000_000_000_000
    );
}

#[test]
fn print_ir() {
    // Not a real test, just prints the IR for debugging.
    let wasm = wat::parse_str(
        r#"(module
            (func (export "fib") (param $n i32) (result i32)
                (local $a i32)
                (local $b i32)
                (local $i i32)
                (local $tmp i32)
                i32.const 0
                local.set $a
                i32.const 1
                local.set $b
                i32.const 0
                local.set $i
                block $exit
                    loop $loop
                        local.get $i
                        local.get $n
                        i32.ge_s
                        br_if $exit
                        local.get $a
                        local.get $b
                        i32.add
                        local.set $tmp
                        local.get $b
                        local.set $a
                        local.get $tmp
                        local.set $b
                        local.get $i
                        i32.const 1
                        i32.add
                        local.set $i
                        br $loop
                    end
                end
                local.get $a))"#,
    )
    .unwrap();
    let (func, _) = translate_wasm(&wasm).unwrap();
    verify(&func).expect("IR verification failed");
    println!("=== fib IR ===\n{}", func);
}

// ─── Multi-function module tests ────────────────────────────────────

#[test]
fn two_functions_call() {
    let wat = r#"(module
        (func $double (param i32) (result i32)
            local.get 0
            i32.const 2
            i32.mul)
        (func (export "main") (param i32) (result i32)
            local.get 0
            call $double))"#;
    let wasm = wat::parse_str(wat).expect("failed to parse WAT");
    let (module, entry) = translate_wasm_module(&wasm).expect("failed to translate");
    let roots = NoGcRoots;
    let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    let result = match interp.run(entry, &[21]).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("expected Value, got {:?}", other),
    };
    assert_eq!(result, 42);
}

#[test]
fn recursive_factorial_module() {
    let wat = r#"(module
        (func $fact (param $n i32) (result i32)
            local.get $n
            i32.const 1
            i32.le_s
            if (result i32)
                i32.const 1
            else
                local.get $n
                local.get $n
                i32.const 1
                i32.sub
                call $fact
                i32.mul
            end)
        (func (export "main") (param i32) (result i32)
            local.get 0
            call $fact))"#;
    let wasm = wat::parse_str(wat).expect("failed to parse WAT");
    let (module, entry) = translate_wasm_module(&wasm).expect("failed to translate");
    let roots = NoGcRoots;
    let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    let result = match interp.run(entry, &[10]).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("expected Value, got {:?}", other),
    };
    assert_eq!(result, 3628800);
}

#[test]
fn fifty_nested_functions() {
    // Generate a WAT module with 50 functions chained:
    //   func_0(x) = x + 1
    //   func_1(x) = func_0(x) + 1
    //   func_2(x) = func_1(x) + 1
    //   ...
    //   func_49(x) = func_48(x) + 1   (exported as "main")
    //
    // main(0) should return 50.

    let n = 50;
    let mut wat = String::from("(module\n");

    // func_0: base case, just adds 1
    wat.push_str("  (func $func_0 (param i32) (result i32)\n");
    wat.push_str("    local.get 0\n");
    wat.push_str("    i32.const 1\n");
    wat.push_str("    i32.add)\n");

    // func_1 through func_49: each calls the previous and adds 1
    for i in 1..n {
        wat.push_str(&format!("  (func $func_{i} (param i32) (result i32)\n"));
        wat.push_str("    local.get 0\n");
        wat.push_str(&format!("    call $func_{}\n", i - 1));
        wat.push_str("    i32.const 1\n");
        wat.push_str("    i32.add)\n");
    }

    // Export the last function
    wat.push_str(&format!("  (export \"main\" (func $func_{})))\n", n - 1));

    let wasm = wat::parse_str(&wat).expect("failed to parse WAT");
    let (module, entry) = translate_wasm_module(&wasm).expect("failed to translate");

    let roots = NoGcRoots;
    let interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    let result = match interp.run(entry, &[0]).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("expected Value, got {:?}", other),
    };
    assert_eq!(result, n as i32, "50 nested calls should each add 1");

    // Also test with a non-zero input
    let result2 = match interp.run(entry, &[100]).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("expected Value, got {:?}", other),
    };
    assert_eq!(result2, 100 + n as i32);
}

// ─── JIT module tests ──────────────────────────────────────────────

#[test]
fn two_functions_call_jit() {
    let wat = r#"(module
        (func $double (param i32) (result i32)
            local.get 0
            i32.const 2
            i32.mul)
        (func (export "main") (param i32) (result i32)
            local.get 0
            call $double))"#;
    let wasm = wat::parse_str(wat).expect("failed to parse WAT");
    let (module, entry) = translate_wasm_module(&wasm).expect("failed to translate");
    let jit = dynlower::JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(entry, &[21]) as i32, 42);
}

#[test]
fn recursive_factorial_module_jit() {
    let wat = r#"(module
        (func $fact (param $n i32) (result i32)
            local.get $n
            i32.const 1
            i32.le_s
            if (result i32)
                i32.const 1
            else
                local.get $n
                local.get $n
                i32.const 1
                i32.sub
                call $fact
                i32.mul
            end)
        (func (export "main") (param i32) (result i32)
            local.get 0
            call $fact))"#;
    let wasm = wat::parse_str(wat).expect("failed to parse WAT");
    let (module, entry) = translate_wasm_module(&wasm).expect("failed to translate");
    let jit = dynlower::JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(entry, &[1]) as i32, 1);
    assert_eq!(jit.call(entry, &[5]) as i32, 120);
    assert_eq!(jit.call(entry, &[10]) as i32, 3628800);
}

#[test]
fn fifty_nested_functions_jit() {
    let n = 50;
    let mut wat = String::from("(module\n");
    wat.push_str("  (func $func_0 (param i32) (result i32)\n");
    wat.push_str("    local.get 0\n    i32.const 1\n    i32.add)\n");
    for i in 1..n {
        wat.push_str(&format!("  (func $func_{i} (param i32) (result i32)\n"));
        wat.push_str("    local.get 0\n");
        wat.push_str(&format!("    call $func_{}\n", i - 1));
        wat.push_str("    i32.const 1\n    i32.add)\n");
    }
    wat.push_str(&format!("  (export \"main\" (func $func_{})))\n", n - 1));

    let wasm = wat::parse_str(&wat).expect("failed to parse WAT");
    let (module, entry) = translate_wasm_module(&wasm).expect("failed to translate");
    let jit = dynlower::JitModule::compile::<NanBox>(&module, &[]);
    assert_eq!(jit.call(entry, &[0]) as i32, 50);
    assert_eq!(jit.call(entry, &[100]) as i32, 150);
}
