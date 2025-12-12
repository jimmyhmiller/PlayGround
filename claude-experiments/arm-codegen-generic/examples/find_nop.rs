use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    
    for instr in arm.find_instructions("nop") {
        println!("{}", instr);
    }
    
    Ok(())
}
