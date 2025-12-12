use arm_codegen_generic::{
    ArmCodeGen,
    simple_rust_generator::SimpleRustGenerator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    
    for instr in arm.find_instructions("csinc") {
        println!("Found: {}", instr);
    }
    
    let instructions = vec![
        "CsetCsinc",
    ];
    
    let code = arm.generate(SimpleRustGenerator, instructions);
    println!("{}", code);
    
    Ok(())
}
