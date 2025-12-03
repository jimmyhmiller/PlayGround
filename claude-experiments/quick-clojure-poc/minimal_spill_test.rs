// Minimal test case for debugging spill crash
// Compile: rustc --edition 2021 -L target/debug/deps minimal_spill_test.rs -l quick_clojure_poc
// Run: ./minimal_spill_test
// Debug: lldb ./minimal_spill_test

use quick_clojure_poc::arm_codegen::Arm64CodeGen;
use quick_clojure_poc::ir::{Instruction, IrValue, VirtualRegister};

fn main() {
    println!("=== Minimal Spill Test ===\n");

    // Create minimal IR that DEFINITELY forces spilling with 4 registers:
    // Load 5 constants simultaneously (all live at same time)
    // Then use all 5 in calculations - impossible with only 4 registers

    let instructions = vec![
        // Load 5 constants - all stay live
        Instruction::LoadConstant(reg(0), IrValue::TaggedConstant(8)),   // 1 << 3
        Instruction::LoadConstant(reg(1), IrValue::TaggedConstant(16)),  // 2 << 3
        Instruction::LoadConstant(reg(2), IrValue::TaggedConstant(24)),  // 3 << 3
        Instruction::LoadConstant(reg(3), IrValue::TaggedConstant(32)),  // 4 << 3
        Instruction::LoadConstant(reg(4), IrValue::TaggedConstant(40)),  // 5 << 3

        // Now use all 5 values - forces spilling!
        // Untag all 5 values
        Instruction::Untag(reg(5), reg(0)),
        Instruction::Untag(reg(6), reg(1)),
        Instruction::Untag(reg(7), reg(2)),
        Instruction::Untag(reg(8), reg(3)),
        Instruction::Untag(reg(9), reg(4)),

        // Add them: (1+2) + (3+4) + 5
        Instruction::AddInt(reg(10), reg(5), reg(6)),  // 1+2 = 3
        Instruction::AddInt(reg(11), reg(7), reg(8)),  // 3+4 = 7
        Instruction::AddInt(reg(12), reg(10), reg(11)), // 3+7 = 10
        Instruction::AddInt(reg(13), reg(12), reg(9)),  // 10+5 = 15

        // Tag result
        Instruction::Tag(reg(14), reg(13), IrValue::TaggedConstant(0)),
    ];

    let result_reg = reg(14);

    println!("IR Instructions:");
    for (i, inst) in instructions.iter().enumerate() {
        println!("  {}: {:?}", i, inst);
    }

    // Compile with 4 registers to force spilling
    let mut codegen = Arm64CodeGen::new();
    println!("\nCompiling with 4 registers (should force spilling)...");

    match codegen.compile(&instructions, &result_reg, 4) {
        Ok(code) => {
            println!("✓ Compilation successful, {} instructions generated", code.len());

            println!("\nGenerated machine code:");
            for (i, inst) in code.iter().enumerate() {
                println!("  0x{:04x}: 0x{:08x}", i * 4, inst);
            }

            println!("\nExecuting...");
            match codegen.execute() {
                Ok(result) => {
                    println!("✓ Result: {}", result);
                    if result == 15 {
                        println!("✓✓✓ TEST PASSED! ✓✓✓");
                    } else {
                        println!("✗ Expected 15, got {}", result);
                    }
                }
                Err(e) => {
                    println!("✗ Execution failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            println!("✗ Compilation failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn reg(index: usize) -> IrValue {
    IrValue::Register(VirtualRegister {
        index,
        is_argument: false,
    })
}
