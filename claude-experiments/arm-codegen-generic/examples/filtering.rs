use arm_codegen_generic::{
    ArmCodeGen, InstructionFilter,
    rust_function_generator::RustFunctionGenerator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    
    // Example 1: Allow only specific instruction categories
    println!("🔍 Finding ADD-related instructions:");
    let add_instructions = arm.find_instructions("add");
    println!("Found {} ADD instructions", add_instructions.len());
    
    // Generate code for first 5 ADD instructions
    let first_five_adds: Vec<&str> = add_instructions.into_iter().take(5).collect();
    let add_code = arm.generate(RustFunctionGenerator, first_five_adds);
    
    let func_count = add_code.lines()
        .filter(|line| line.trim().starts_with("pub fn ") && 
                      !line.contains("sf(") && !line.contains("encode(") && !line.contains("truncate_imm"))
        .count();
    println!("Generated {} ADD functions", func_count);
    
    // Example 2: Use filtering to exclude unwanted instructions
    println!("\n⚡ Using instruction filters:");
    let filter = InstructionFilter::new()
        .allow(arm.find_instructions("mov").into_iter().map(|s| s.to_string()).collect())
        .block(vec!["Deprecated".to_string(), "Reserved".to_string()]);
    
    let filtered_code = arm.generate_filtered(RustFunctionGenerator, filter);
    let filtered_count = filtered_code.lines()
        .filter(|line| line.trim().starts_with("pub fn ") && 
                      !line.contains("sf(") && !line.contains("encode(") && !line.contains("truncate_imm"))
        .count();
    println!("Generated {} MOV functions with filtering", filtered_count);
    
    // Example 3: Generate everything (huge!)
    println!("\n🌍 Generate all ARM instructions:");
    let all_code = arm.generate_all(RustFunctionGenerator);
    let total_funcs = all_code.lines()
        .filter(|line| line.trim().starts_with("pub fn ") && 
                      !line.contains("sf(") && !line.contains("encode(") && !line.contains("truncate_imm"))
        .count();
    println!("Generated {} total functions ({} KB)", total_funcs, all_code.len() / 1024);
    
    Ok(())
}