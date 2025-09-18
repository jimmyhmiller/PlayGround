use arm_codegen_generic::{ArmCodeGen, InstructionFilter, zig_function_generator::ZigFunctionGenerator};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let codegen = ArmCodeGen::new()?;

    // Generate a subset of instructions for testing
    let test_instructions = vec![
        "AddAddsubImm",
        "AddAddsubShift",
        "SubAddsubImm",
        "MovAddAddsubImm",
        "LdrRegGen",
        "StrImmGen",
        "Bl",
        "Ret"
    ];

    println!("Generating Zig code for {} instructions...", test_instructions.len());

    let generator = ZigFunctionGenerator;
    let zig_code = codegen.generate(generator, test_instructions);

    // Write to file
    fs::write("arm_jit_instructions_generated.zig", &zig_code)?;

    println!("Generated Zig code saved to 'arm_jit_instructions_generated.zig'");
    println!("\nFirst 1000 characters of generated code:");
    println!("{}", &zig_code[..std::cmp::min(1000, zig_code.len())]);

    Ok(())
}