use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    let all = arm.available_instructions();
    
    // Look for "unscaled" which is what LDUR/STUR are called in the ARM spec
    for instr in &all {
        let lower = instr.to_lowercase();
        if lower.contains("unscal") || lower.contains("ldur") || lower.contains("stur") {
            println!("{}", instr);
        }
    }
    
    // Also search via find_instructions
    println!("\n--- Find unscaled ---");
    for instr in arm.find_instructions("unscaled") {
        println!("{}", instr);
    }
    
    Ok(())
}
