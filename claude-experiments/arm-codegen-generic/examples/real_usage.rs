use arm_codegen_generic::{
    ArmCodeGen,
    rust_function_generator::RustFunctionGenerator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Real-World Usage Example");
    println!("==========================");
    
    let arm = ArmCodeGen::new()?;
    
    // Scenario: Building a JIT compiler that needs specific ARM instructions
    println!("🎯 JIT Compiler Scenario - Need arithmetic and control flow:");
    
    let jit_instructions = vec![
        "AddAddsubImm",     // ADD immediate 
        "AddAddsubShift",   // ADD shifted register
        "SubAddsubImm",     // SUB immediate
        "MovMovzImm",       // MOV immediate
        "CmpAddsubImm",     // Compare immediate
        "BCond",            // Conditional branch
        "Ret",              // Return
    ];
    
    let jit_code = arm.generate(RustFunctionGenerator, jit_instructions);
    
    // Count generated functions
    let func_count = jit_code.lines()
        .filter(|line| line.trim().starts_with("pub fn ") && 
                      !line.contains("sf(") && !line.contains("encode(") && !line.contains("truncate_imm"))
        .count();
    
    println!("Generated {} functions for JIT compiler", func_count);
    println!("Code size: {} KB", jit_code.len() / 1024);
    
    // Show how the generated functions would be used
    println!("\n📝 Generated function signatures:");
    for line in jit_code.lines() {
        if line.trim().starts_with("pub fn ") && 
           !line.contains("sf(") && !line.contains("encode(") && !line.contains("truncate_imm") {
            println!("  {}", line.trim());
        }
    }
    
    println!("\n🔧 Usage in your code:");
    println!("```rust");
    println!("// Include the generated functions");
    println!("include!(concat!(env!(\"OUT_DIR\"), \"/arm_instructions.rs\"));");
    println!("");
    println!("// Use them to encode instructions");
    println!("let add_inst = add_addsub_imm(1, 0, 42, X2, X1);  // ADD X1, X2, #42");
    println!("let mov_inst = mov_movz_imm(1, 0, 100, X3);       // MOV X3, #100");
    println!("let cmp_inst = cmp_addsub_imm(1, 0, 0, X1);       // CMP X1, #0");
    println!("```");
    
    println!("\n✅ Perfect for:");
    println!("  • JIT compilers");
    println!("  • Assemblers"); 
    println!("  • Code generators");
    println!("  • Embedded systems");
    println!("  • Anything needing ARM machine code");
    
    Ok(())
}