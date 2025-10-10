use arm_codegen_generic::{ArmCodeGen, rust_function_generator::RustFunctionGenerator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Generating ARM64 Instructions for CPython JIT");
    println!("==================================================");

    let arm = ArmCodeGen::new()?;

    // Instructions needed for CPython JIT:
    let jit_instructions = vec![
        // 1. ADD (register) - Add two 64-bit registers
        "AddAddsubShift",

        // 2. SUB (register) - Subtract two 64-bit registers
        "SubAddsubShift",

        // 3. CMP (register) - Compare two 64-bit registers
        "CmpSubsAddsubShift",

        // 4. CBNZ - Compare and Branch on Non-Zero (64-bit)
        "Cbnz",

        // 5-10. Conditional branches (using BCond with different condition codes)
        // B.LT, B.GE, B.LE, B.GT, B.EQ, B.NE
        "BCond",
    ];

    // Generate Rust function-based code
    let rust_code = arm.generate(RustFunctionGenerator, jit_instructions.clone());

    // Count generated functions
    let func_count = rust_code.lines()
        .filter(|line| line.trim().starts_with("pub fn "))
        .count();

    println!("📊 Statistics:");
    println!("  Instructions requested: {}", jit_instructions.len());
    println!("  Rust functions generated: {}", func_count);
    println!("  Total code size: {} KB", rust_code.len() / 1024);

    println!("\n🎯 Generated Rust function signatures:");
    for line in rust_code.lines() {
        if line.trim().starts_with("pub fn ") {
            println!("  {}", line.trim().trim_end_matches(" {"));
        }
    }

    println!("\n💾 Writing Rust code to cpython_jit_arm64.rs...");
    std::fs::write("cpython_jit_arm64.rs", &rust_code)?;

    println!("\n✅ ARM64 instruction encoders generated for CPython JIT!");
    println!("📁 File: cpython_jit_arm64.rs");
    println!("🔧 Copy the generated functions to your CPython JIT project");

    // Print usage examples
    println!("\n📖 Usage Examples:");
    println!("```rust");
    println!("// Add two registers: X0 = X1 + X2");
    println!("let encoding = add_addsub_shift(1, 0, X2, 0, X1, X0);");
    println!();
    println!("// Subtract registers: X0 = X1 - X2");
    println!("let encoding = sub_addsub_shift(1, 0, X2, 0, X1, X0);");
    println!();
    println!("// Compare registers: CMP X1, X2");
    println!("let encoding = cmp_subs_addsub_shift(1, 0, X2, 0, X1);");
    println!();
    println!("// Branch if not zero: CBNZ X0, offset");
    println!("let encoding = cbnz(offset, X0);");
    println!();
    println!("// Conditional branches (condition codes from ARM64 spec):");
    println!("let b_eq = b_cond(offset, 0);  // B.EQ (equal)");
    println!("let b_ne = b_cond(offset, 1);  // B.NE (not equal)");
    println!("let b_ge = b_cond(offset, 10); // B.GE (signed >=)");
    println!("let b_lt = b_cond(offset, 11); // B.LT (signed <)");
    println!("let b_gt = b_cond(offset, 12); // B.GT (signed >)");
    println!("let b_le = b_cond(offset, 13); // B.LE (signed <=)");
    println!("```");

    Ok(())
}
