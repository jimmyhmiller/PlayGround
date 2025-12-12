use arm_codegen_generic::{
    ArmCodeGen,
    simple_rust_generator::SimpleRustGenerator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    
    let instructions = vec![
        "FcmpFloat",
    ];
    
    let code = arm.generate(SimpleRustGenerator, instructions);
    println!("{}", code);
    
    Ok(())
}
