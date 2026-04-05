use arm_codegen_generic::ArmCodeGen;
fn main() {
    let arm = ArmCodeGen::new().unwrap();
    for name in arm.available_instructions() {
        println!("{}", name);
    }
}
