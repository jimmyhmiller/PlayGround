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
use quick_clojure_poc::trampoline::{self, Trampoline};
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
    compiler.compile(&ast).unwrap();
    let instructions = compiler.take_instructions();
    let num_locals = compiler.builder.num_locals;

    println!("=== IR Instructions ===");
    for (i, inst) in instructions.iter().enumerate() {
        println!("{:3}: {:?}", i, inst);
    }

    // Allocate with 4 registers to force spilling (for debug info)
    let mut allocator = LinearScan::new(instructions.clone(), 4);
    allocator.allocate();

    println!("\n=== Allocation Results ===");
    println!("Spills: {}", allocator.spill_locations.len());
    println!("Stack slots: {}", allocator.next_stack_slot);
    for (vreg, loc) in &allocator.spill_locations {
        println!("  v{} spilled to slot {}", vreg.index(), loc);
    }

    let compiled = Arm64CodeGen::compile_function(&instructions, num_locals, 0).unwrap();

    println!("\n=== ARM64 Machine Code ===");
    println!("Compiled {} instructions at {:p}", compiled.code_len, compiled.code_ptr as *const u8);

    println!("\n=== Executing (breakpoint will trigger) ===");

    let tramp = Trampoline::new(64 * 1024);
    let result = unsafe { tramp.execute(compiled.code_ptr as *const u8) };
    // Result is tagged
    let untagged = result >> 3;

    println!("\n=== Result ===");
    println!("Got (tagged): {}", result);
    println!("Got (untagged): {}", untagged);
    if untagged == 15 {
        println!("✓ SUCCESS");
    } else {
        println!("✗ FAILED - expected 15");
    }
}
