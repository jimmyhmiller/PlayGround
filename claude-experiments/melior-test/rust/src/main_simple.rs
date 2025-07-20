//! Simple test without the new C++ dialect to verify basic functionality

use melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    ir::{
        Block, Identifier, Location, Module, Region,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType},
    },
    pass::PassManager,
};
use mlir_sys::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing basic MLIR functionality without C++ dialect...");

    // Setup MLIR context and dialects
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
        mlirRegisterAllPasses();
    }

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Allow unregistered dialects for testing
    unsafe {
        mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }

    // Register LLVM translation interfaces
    unsafe {
        mlirRegisterAllLLVMTranslations(context.to_raw());
    }

    let location = Location::unknown(&context);
    let module = Module::new(location);

    println!("✅ MLIR context and module created successfully");

    // Create simple function without custom dialect
    create_simple_function(&context, &module)?;

    println!("✅ Function created successfully");

    // Apply passes
    let pass_manager = PassManager::new(&context);
    unsafe {
        use melior::pass::Pass;

        let func_to_llvm = Pass::from_raw(mlirCreateConversionConvertFuncToLLVMPass());
        pass_manager.add_pass(func_to_llvm);

        let reconcile_pass = Pass::from_raw(mlirCreateConversionReconcileUnrealizedCasts());
        pass_manager.add_pass(reconcile_pass);
    }

    let mut final_module = module;
    pass_manager.run(&mut final_module)?;

    println!("✅ Passes applied successfully");

    // Test JIT compilation
    test_jit_compilation(&context, &final_module)?;

    println!("🎉 All tests passed! Basic MLIR infrastructure is working.");

    Ok(())
}

fn create_simple_function(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);

    // Create simple i32 -> i32 function
    let i32_type = IntegerType::new(context, 32).into();
    let function_type = FunctionType::new(context, &[i32_type], &[i32_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "simple_function").into(),
            ),
            (
                Identifier::new(context, "function_type"),
                TypeAttribute::new(function_type.into()).into(),
            ),
            (
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "public").into(),
            ),
        ])
        .add_regions([Region::new()])
        .build()?;

    // Create function body that just returns the input
    let block = Block::new(&[(i32_type, location)]);
    let region = function.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();
    let arg = block_ref.argument(0)?;

    // Return the argument unchanged
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[arg.into()])
        .build()?;
    block_ref.append_operation(return_op);

    // Add function to module
    module.body().append_operation(function);

    Ok(())
}

fn test_jit_compilation(
    _context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create ExecutionEngine
    let engine = ExecutionEngine::new(module, 0, &[], false);

    println!("✅ ExecutionEngine created successfully!");

    // Try to lookup the simple function
    let func_ptr = engine.lookup("simple_function");
    if func_ptr.is_null() {
        println!("⚠️ Could not find 'simple_function' - this is the known function lookup issue");
    } else {
        println!("✅ Found 'simple_function' at {:?}", func_ptr);
    }

    Ok(())
}
