use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        Block, Identifier, Location, Module, Region,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType},
    },
};
use mlir_sys::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Minimal MLIR Context Test");

    // Setup MLIR context and dialects
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
        mlirRegisterAllPasses();
    }

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let location = Location::unknown(&context);
    let module = Module::new(location);

    println!("✅ Created empty module without crash");

    // Try creating a simple function
    println!("🔧 Creating simple function...");
    create_minimal_function(&context, &module)?;

    println!("✅ Created function without crash");
    println!("✅ MLIR context setup successful");

    Ok(())
}

fn create_minimal_function(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);

    // Create function: i32 test(i32 a, i32 b) { return; }
    let i32_type = IntegerType::new(context, 32).into();
    let function_type = FunctionType::new(context, &[i32_type, i32_type], &[i32_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "test").into(),
            ),
            (
                Identifier::new(context, "function_type"),
                TypeAttribute::new(function_type.into()).into(),
            ),
        ])
        .add_regions([Region::new()])
        .build()?;

    module.body().append_operation(function.clone());

    // Create function body with arguments
    let block = Block::new(&[(i32_type, location), (i32_type, location)]);
    let region = function.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();
    let arg_a = block_ref.argument(0)?;

    // Create add operation
    let add_op = OperationBuilder::new("arith.addi", location)
        .add_operands(&[arg_a.into(), arg_a.into()])
        .add_results(&[i32_type])
        .build()?;
    block_ref.append_operation(add_op.clone());

    // Return the result
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[add_op.result(0)?.into()])
        .build()?;
    block_ref.append_operation(return_op);

    Ok(())
}
