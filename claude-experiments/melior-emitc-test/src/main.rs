use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        Block, BlockLike, Location, Module, Region, RegionLike,
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, IntegerType},
        Identifier,
    },
    pass::PassManager,
    utility::register_all_dialects,
};
use std::fs;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let location = Location::unknown(&context);

    // Create a module with emitc operations
    let mut module = Module::new(location);
    let i32_type = IntegerType::new(&context, 32);
    let function_type = FunctionType::new(&context, &[], &[i32_type.into()]);

    let region = Region::new();
    let entry_block = Block::new(&[]);
    region.append_block(entry_block);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(&context, "sym_name"),
                StringAttribute::new(&context, "main").into(),
            ),
            (
                Identifier::new(&context, "function_type"),
                TypeAttribute::new(function_type.into()).into(),
            ),
        ])
        .add_regions([region])
        .build()?;

    let function_region = function.region(0)?;
    let entry_block = function_region.first_block().unwrap();

    // Use arith operations that will be converted to emitc
    let const_10 = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(
            Identifier::new(&context, "value"),
            IntegerAttribute::new(i32_type.into(), 10).into(),
        )])
        .add_results(&[i32_type.into()])
        .build()?;

    let const_32 = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(
            Identifier::new(&context, "value"),
            IntegerAttribute::new(i32_type.into(), 32).into(),
        )])
        .add_results(&[i32_type.into()])
        .build()?;

    let add_op = OperationBuilder::new("arith.addi", location)
        .add_operands(&[const_10.result(0)?.into(), const_32.result(0)?.into()])
        .add_results(&[i32_type.into()])
        .build()?;

    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[add_op.result(0)?.into()])
        .build()?;

    entry_block.append_operation(const_10);
    entry_block.append_operation(const_32);
    entry_block.append_operation(add_op);
    entry_block.append_operation(return_op);
    module.body().append_operation(function);

    println!("BEFORE conversion:");
    println!("{}", module.as_operation());

    // Convert arith and func to emitc dialect
    let pm = PassManager::new(&context);
    pm.add_pass(melior::pass::conversion::create_arith_to_emit_c());
    pm.add_pass(melior::pass::conversion::create_func_to_emit_c());

    pm.run(&mut module)?;

    println!("\nAFTER conversion to EmitC:");
    let emitc_ir = format!("{}", module.as_operation());
    println!("{}", emitc_ir);

    // Write the EmitC IR to a file
    fs::write("output.mlir", &emitc_ir)?;
    println!("\nEmitC IR written to output.mlir");

    // Translate EmitC to C++ using mlir-translate
    println!("\nTranslating to C++...");
    let output = Command::new("mlir-translate")
        .arg("--mlir-to-cpp")
        .arg("output.mlir")
        .output()?;

    if !output.status.success() {
        eprintln!("mlir-translate failed: {}", String::from_utf8_lossy(&output.stderr));
        return Err("Failed to translate to C++".into());
    }

    let c_code = String::from_utf8(output.stdout)?;
    fs::write("output.c", &c_code)?;

    println!("\nGenerated C code:");
    println!("{}", c_code);
    println!("C code written to output.c");

    Ok(())
}
