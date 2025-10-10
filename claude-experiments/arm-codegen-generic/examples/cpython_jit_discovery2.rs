use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;

    println!("🔍 CPython JIT - Conditional Branch Discovery");
    println!("==============================================\n");

    // Search for branch instructions
    let patterns = vec!["branch", "b.", "cond", "bcond"];

    for pattern in patterns {
        let found = arm.find_instructions(pattern);
        if !found.is_empty() {
            println!("Pattern '{}' found {} matches:", pattern, found.len());
            for instr in found.iter() {
                if let Some((_, title)) = arm.instruction_info(instr) {
                    println!("    {} - {}", instr, title);
                }
            }
            println!();
        }
    }

    Ok(())
}
