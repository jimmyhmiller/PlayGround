use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    
    println!("🔍 ARM Instruction Discovery");
    println!("============================");
    
    // Get basic statistics
    let all_instructions = arm.available_instructions();
    println!("Total instructions available: {}", all_instructions.len());
    
    // Explore instruction categories
    let categories = [
        ("ADD", "add"),
        ("SUB", "sub"), 
        ("MOV", "mov"),
        ("Load", "ldr"),
        ("Store", "str"),
        ("Branch", "branch"),
        ("Compare", "cmp"),
        ("Multiply", "mul"),
    ];
    
    println!("\n📊 Instruction categories:");
    for (name, pattern) in categories {
        let found = arm.find_instructions(pattern);
        println!("  {}: {} instructions", name, found.len());
        
        // Show first few examples
        if !found.is_empty() {
            for instr in found.iter().take(3) {
                if let Some((_, title)) = arm.instruction_info(instr) {
                    println!("    {} - {}", instr, title);
                }
            }
            if found.len() > 3 {
                println!("    ... and {} more", found.len() - 3);
            }
        }
        println!();
    }
    
    // Search for specific patterns
    println!("🔎 Search examples:");
    
    let immediate_instrs = arm.find_instructions("immediate");
    println!("Instructions with 'immediate': {}", immediate_instrs.len());
    
    let vector_instrs = arm.find_instructions("vector");
    println!("Vector instructions: {}", vector_instrs.len());
    
    let crypto_instrs = arm.find_instructions("aes");
    println!("Cryptographic instructions: {}", crypto_instrs.len());
    
    // Show instruction details
    println!("\n📋 Instruction details:");
    let sample_instructions = ["AddAddsubImm", "MovMovzImm", "LdrLitLit"];
    
    for instr_name in sample_instructions {
        if let Some((name, title)) = arm.instruction_info(instr_name) {
            println!("  {} - {}", name, title);
        }
    }
    
    println!("\n✨ Use these names with arm.generate() to create encoder functions!");
    
    Ok(())
}