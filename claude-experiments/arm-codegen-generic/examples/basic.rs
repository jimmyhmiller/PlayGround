use arm_codegen_generic::{
    ArmCodeGen,
    rust_function_generator::RustFunctionGenerator,
    cpp_function_generator::CppFunctionGenerator
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the ARM code generator (loads all 799 ARM instructions)
    let arm = ArmCodeGen::new()?;
    
    // Generate Rust functions for specific ARM instructions
    let rust_code = arm.generate(
        RustFunctionGenerator, 
        vec!["AddAddsubImm", "MovMovzImm"]
    );
    
    println!("Generated Rust functions:");
    for line in rust_code.lines() {
        if line.trim().starts_with("pub fn ") && 
           !line.contains("sf(") && !line.contains("encode(") && !line.contains("truncate_imm") {
            println!("  {}", line.trim());
        }
    }
    
    // Generate C++ functions for the same instructions
    let cpp_code = arm.generate(
        CppFunctionGenerator,
        vec!["AddAddsubImm", "MovMovzImm"]
    );
    
    println!("\nGenerated C++ functions:");
    for line in cpp_code.lines() {
        if line.trim().starts_with("constexpr uint32_t ") && !line.contains("truncate_imm") {
            println!("  {}", line.trim());
        }
    }
    
    println!("\n✅ Generated ARM instruction encoders in both languages!");
    
    Ok(())
}