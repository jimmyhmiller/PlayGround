#!/bin/bash
# Simple test with fewer registers

cargo build --example debug_spill || exit 1

echo "Running with 10 registers (no spilling)..."
cat > /tmp/test_no_spill.rs << 'EOF'
use quick_clojure_poc::reader::read;
use quick_clojure_poc::clojure_ast::analyze;
use quick_clojure_poc::compiler::Compiler;
use quick_clojure_poc::gc_runtime::GCRuntime;
use quick_clojure_poc::arm_codegen::Arm64CodeGen;
use quick_clojure_poc::trampoline;
use std::sync::Arc;
use std::cell::UnsafeCell;

fn main() {
    let code = "(let [a 1 b 2 c 3 d 4 e 5] (+ a (+ b (+ c (+ d e)))))";

    let val = read(code).unwrap();
    let ast = analyze(&val).unwrap();

    let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());

    let mut compiler = Compiler::new(runtime);
    let result_reg = compiler.compile(&ast).unwrap();
    let instructions = compiler.finish();

    let mut codegen = Arm64CodeGen::new();
    // Use 0 (default) registers - should not spill
    let _machine_code = codegen.compile(&instructions, &result_reg, 0).unwrap();

    println!("Executing with NO spilling...");
    let result = codegen.execute().unwrap();
    println!("Result: {}", result);
    assert_eq!(result, 15);
    println!("✓ SUCCESS!");
}
EOF

rustc --edition 2021 -L target/debug/deps /tmp/test_no_spill.rs -l quick_clojure_poc -o /tmp/test_no_spill 2>&1 | head -10
if [ -f /tmp/test_no_spill ]; then
    /tmp/test_no_spill
else
    echo "Compilation failed"
fi
