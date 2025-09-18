use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let codegen = ArmCodeGen::new()?;

    let available = codegen.available_instructions();
    println!("Available instructions (first 20):");
    for (i, instr) in available.iter().take(20).enumerate() {
        println!("{}: {}", i + 1, instr);
    }

    println!("\nTotal instructions available: {}", available.len());

    // Look for ADD instructions
    let add_instructions = codegen.find_instructions("add");
    println!("\nADD-related instructions:");
    for instr in &add_instructions {
        if let Some((name, title)) = codegen.instruction_info(instr) {
            println!("  {} - {}", name, title);
        }
    }

    Ok(())
}