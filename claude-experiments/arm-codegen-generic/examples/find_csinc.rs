use arm_codegen_generic::ArmCodeGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    
    // Look for CSINC, CSEL, CSET
    for instr in arm.find_instructions("csinc") {
        println!("{}", instr);
    }
    for instr in arm.find_instructions("csel") {
        println!("{}", instr);
    }
    for instr in arm.find_instructions("cset") {
        println!("{}", instr);
    }
    
    Ok(())
}
