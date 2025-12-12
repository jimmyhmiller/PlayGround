use arm_codegen_generic::{ArmCodeGen, rust_function_generator::RustFunctionGenerator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    let code = arm.generate(RustFunctionGenerator, vec!["Csinc"]);
    println!("{}", code);
    Ok(())
}
