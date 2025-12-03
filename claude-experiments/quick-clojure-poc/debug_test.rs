// Debug test program to step through JIT code instruction by instruction
// Compile with: rustc --edition 2021 -L target/debug/deps debug_test.rs -l quick_clojure_poc
// Or better: cargo build && cargo run --example debug_test

use quick_clojure_poc::reader::read;
use quick_clojure_poc::clojure_ast::analyze;
use quick_clojure_poc::compiler::Compiler;
use quick_clojure_poc::gc_runtime::GCRuntime;
use quick_clojure_poc::arm_codegen::Arm64CodeGen;
use quick_clojure_poc::register_allocation::linear_scan::LinearScan;
use std::sync::{Arc, Mutex};

fn main() {
    // Test case that forces spilling (4 registers available)
    let code = "(let [a 1 b 2 c 3 d 4 e 5] (+ a (+ b (+ c (+ d e)))))";

    println!("=== DEBUG TEST ===");
    println!("Code: {}", code);
    println!("\nThis test will trigger a breakpoint in the trampoline.");
    println!("Run with: lldb target/debug/debug_test");
    println!("Then in lldb:");
    println!("  (lldb) run");
    println!("  (lldb) register read    # to see all registers");
    println!("  (lldb) si               # to step instruction");
    println!("  (lldb) memory read -fx -c8 $sp    # to see stack");
    println!("  (lldb) memory read -fx -c8 $x29   # to see frame pointer area");
    println!("\n");

    let val = read(code).unwrap();
    let ast = analyze(&val).unwrap();

    let runtime = Arc::new(Mutex::new(GCRuntime::new()));
    let mut compiler = Compiler::new(runtime);
    let result_reg = compiler.compile(&ast).unwrap();
    let instructions = compiler.finish();

    // Use LinearScan allocator with 4 registers to force spilling
    let mut allocator = LinearScan::new(instructions.clone(), 4);
    allocator.mark_live_until_end(match result_reg {
        quick_clojure_poc::ir::IrValue::Register(r) => r,
        _ => panic!("Expected register"),
    });
    allocator.allocate();

    println!("=== Register Allocation ===");
    println!("Spills: {}", allocator.spill_locations.len());
    println!("Stack slots: {}", allocator.next_stack_slot);

    let mut codegen = Arm64CodeGen::new();
    let machine_code = codegen.compile(&instructions, &result_reg, 0).unwrap();

    println!("\n=== Generated ARM64 Instructions ===");
    for (i, inst) in machine_code.iter().enumerate() {
        println!("  {:04x}: {:08x}", i * 4, inst);
    }

    println!("\n=== Executing ===");
    println!("Expected result: 15");

    let result = codegen.execute().unwrap();
    println!("Actual result: {}", result);

    if result == 15 {
        println!("✓ Test passed!");
    } else {
        println!("✗ Test failed!");
    }
}
