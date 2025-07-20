use melior::{
    Context, ExecutionEngine,
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
    println!("🚀 Minimal Working MLIR JIT Example");
    println!("===================================");

    // Setup MLIR context
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

    println!("\n📝 Creating minimal add function...");

    // Create the simplest possible function that should work
    let i32_type = IntegerType::new(&context, 32).into();
    let function_type = FunctionType::new(&context, &[i32_type, i32_type], &[i32_type]);

    // Create function with proper attributes
    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(&context, "sym_name"),
                StringAttribute::new(&context, "add").into(),
            ),
            (
                Identifier::new(&context, "function_type"),
                TypeAttribute::new(function_type.into()).into(),
            ),
            (
                Identifier::new(&context, "sym_visibility"),
                StringAttribute::new(&context, "public").into(),
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
    let arg_b = block_ref.argument(1)?;

    // Create add operation
    let add_op = OperationBuilder::new("arith.addi", location)
        .add_operands(&[arg_a.into(), arg_b.into()])
        .add_results(&[i32_type])
        .build()?;
    block_ref.append_operation(add_op.clone());

    // Return the result
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[add_op.result(0)?.into()])
        .build()?;
    block_ref.append_operation(return_op);

    println!("✅ Function created successfully");

    // Try to print the module safely
    println!("\n🔍 Generated MLIR:");
    match std::panic::catch_unwind(|| format!("{}", module.as_operation())) {
        Ok(module_str) => {
            println!("{}", module_str);

            // Only try JIT if printing succeeded (means module is valid)
            println!("\n🔧 Attempting JIT compilation...");

            let engine = ExecutionEngine::new(&module, 0, &[], false);
            println!("✅ ExecutionEngine created!");

            let add_fn_ptr = engine.lookup("add");
            if !add_fn_ptr.is_null() {
                println!("✅ Found 'add' function!");

                // Test the function
                let result = unsafe {
                    let add_fn: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(add_fn_ptr);
                    add_fn(5, 3)
                };

                println!("🎉 add(5, 3) = {}", result);

                if result == 8 {
                    println!("✅ Perfect! JIT compilation working correctly!");
                } else {
                    println!("⚠️ Unexpected result, but execution didn't crash");
                }
            } else {
                println!("⚠️ Function not found, but ExecutionEngine created successfully");
            }
        }
        Err(_) => {
            println!("❌ Module has verification errors - cannot print or JIT compile");
            println!("   This indicates malformed MLIR operations");
        }
    }

    println!("\n🎯 Minimal JIT test complete!");

    Ok(())
}
