use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let codegen = ArmCodeGen::new()?;

    let available = codegen.available_instructions();

    // Look for ADD instructions to debug
    for name in available.iter().take(5) {
        println!("\n=== {} ===", name);

        // This is a bit of a hack since we can't access instructions directly
        // Let's just look at one we know has issues
        if name == "AddAddsubExt" {
            println!("Found AddAddsubExt - this likely has the u0 issue");
        }
    }

    Ok(())
}