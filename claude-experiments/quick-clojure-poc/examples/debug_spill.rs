// Debug example to step through spilling code with lldb
//
// Build and run:
//   cargo build --example debug_spill
//   lldb target/debug/examples/debug_spill
//
// In lldb:
//   (lldb) run
//   (lldb) register read       # See all registers
//   (lldb) register read x29   # See frame pointer
//   (lldb) register read sp    # See stack pointer
//   (lldb) si                  # Step one instruction
//   (lldb) memory read -fx -c16 $sp    # View stack memory
//   (lldb) memory read -fx -c16 $x29   # View frame pointer area

use quick_clojure_poc::reader::read;
use quick_clojure_poc::clojure_ast::analyze;
use quick_clojure_poc::compiler::Compiler;
use quick_clojure_poc::gc_runtime::GCRuntime;
use quick_clojure_poc::arm_codegen::Arm64CodeGen;
use quick_clojure_poc::register_allocation::linear_scan::LinearScan;
use quick_clojure_poc::trampoline;
use std::sync::Arc;
use std::cell::UnsafeCell;

fn main() {
    // Test case that forces spilling with 4 registers
    let code = "(let [a 1 b 2 c 3 d 4 e 5] (+ a (+ b (+ c (+ d e)))))";

    println!("=== LLDB DEBUG EXAMPLE ===");
    println!("Testing: {}", code);
    println!("Expected result: 15");
    println!("");
    println!("The program will hit a breakpoint in the trampoline.");
    println!("You can step through the trampoline and JIT code instruction by instruction.");
    println!("");

    let val = read(code).unwrap();
    let ast = analyze(&val).unwrap();

    // Create runtime with UnsafeCell (required by Compiler)
    let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());

    let mut compiler = Compiler::new(runtime);
    let result_reg = compiler.compile(&ast).unwrap();
    let instructions = compiler.finish();

    println!("=== IR Instructions ===");
    for (i, inst) in instructions.iter().enumerate() {
        println!("{:3}: {:?}", i, inst);
    }

    // Allocate with 4 registers to force spilling
    let mut allocator = LinearScan::new(instructions.clone(), 4);
    if let quick_clojure_poc::ir::IrValue::Register(r) = result_reg {
        allocator.mark_live_until_end(r);
    }
    allocator.allocate();

    println!("\n=== Allocation Results ===");
    println!("Spills: {}", allocator.spill_locations.len());
    println!("Stack slots: {}", allocator.next_stack_slot);
    for (vreg, loc) in &allocator.spill_locations {
        println!("  v{} spilled to slot {}", vreg.index, loc);
    }

    let mut codegen = Arm64CodeGen::new();
    let machine_code = codegen.compile(&instructions, &result_reg, 4).unwrap();

    println!("\n=== ARM64 Machine Code ===");
    for (i, inst) in machine_code.iter().enumerate() {
        println!("  0x{:04x}: 0x{:08x}", i * 4, inst);
    }

    println!("\n=== Executing (breakpoint will trigger) ===");

    match codegen.execute() {
        Ok(result) => {
            println!("\n=== Result ===");
            println!("Got: {}", result);
            if result == 15 {
                println!("✓ SUCCESS");
            } else {
                println!("✗ FAILED - expected 15");
            }
        }
        Err(e) => {
            println!("\n=== ERROR ===");
            println!("Execution failed: {}", e);
        }
    }
}
