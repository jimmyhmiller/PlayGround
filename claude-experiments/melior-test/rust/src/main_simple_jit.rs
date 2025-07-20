//! Simple JIT test that avoids problematic arithmetic operations
//!
//! This test creates the simplest possible functions to verify JIT lookup works

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
    println!("🔬 Testing simple MLIR JIT function lookup...");

    // Setup MLIR context and dialects
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
        mlirRegisterAllPasses();
    }

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Register LLVM translation interfaces
    unsafe {
        mlirRegisterAllLLVMTranslations(context.to_raw());
    }

    let location = Location::unknown(&context);
    let module = Module::new(location);

    // Create simple identity functions
    create_identity_functions(&context, &module)?;

    // Convert to LLVM IR
    let mut final_module = module;
    apply_lowering_passes(&context, &mut final_module)?;

    // JIT compile and test function lookup
    test_function_lookup(&context, &final_module)?;

    Ok(())
}

fn create_identity_functions(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);
    let i32_type = IntegerType::new(context, 32).into();

    // Create identity function: i32 identity(i32 x) { return x; }
    create_identity_function(context, module, location, i32_type, "simple_identity")?;

    // Create another identity function with different name
    create_identity_function(context, module, location, i32_type, "another_identity")?;

    Ok(())
}

fn create_identity_function(
    context: &Context,
    module: &Module,
    location: Location,
    i32_type: melior::ir::Type,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let function_type = FunctionType::new(context, &[i32_type], &[i32_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, name).into(),
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

    let block = Block::new(&[(i32_type, location)]);
    let function_clone = function.clone();
    let region = function_clone.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();
    let arg0 = block_ref.argument(0)?.into();

    // Simply return the input (no arithmetic operations)
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[arg0])
        .build()?;
    block_ref.append_operation(return_op);

    module.body().append_operation(function);
    Ok(())
}

fn apply_lowering_passes(
    context: &Context,
    module: &mut Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let pass_manager = PassManager::new(context);

    unsafe {
        use melior::pass::Pass;

        // Convert func dialect to LLVM dialect
        let func_to_llvm = Pass::from_raw(mlirCreateConversionConvertFuncToLLVMPass());
        pass_manager.add_pass(func_to_llvm);

        // Finalize conversion to LLVM (skip math passes since we don't use arithmetic)
        let reconcile_pass = Pass::from_raw(mlirCreateConversionReconcileUnrealizedCasts());
        pass_manager.add_pass(reconcile_pass);
    }

    println!("🔧 Applying LLVM lowering passes...");
    pass_manager.run(module)?;
    println!("✅ Lowering passes completed successfully");

    Ok(())
}

fn test_function_lookup(
    _context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Creating ExecutionEngine...");

    // Test different optimization levels
    for opt_level in [0, 1, 2] {
        println!("\n📊 Testing optimization level {}", opt_level);

        let engine = ExecutionEngine::new(module, opt_level, &[], false);
        println!("✅ ExecutionEngine created with opt level {}", opt_level);

        // Try to lookup functions
        let function_names = ["simple_identity", "another_identity"];
        let mut found_functions = 0;

        for name in &function_names {
            let func_ptr = engine.lookup(name);
            if func_ptr.is_null() {
                println!("❌ Could not find '{}' function", name);
            } else {
                println!("✅ Found '{}' function at {:p}", name, func_ptr);
                found_functions += 1;
            }
        }

        if found_functions == function_names.len() {
            println!(
                "🎉 All functions found at optimization level {}!",
                opt_level
            );

            // Test calling the identity function
            test_function_execution(&engine)?;
            return Ok(());
        } else {
            println!(
                "⚠️ Only {}/{} functions found at opt level {}",
                found_functions,
                function_names.len(),
                opt_level
            );
        }
    }

    println!("❌ Functions not found at any optimization level");
    println!("💡 This suggests functions are being optimized away or not exported properly");

    Ok(())
}

fn test_function_execution(engine: &ExecutionEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing function execution...");

    // Try to get and call the identity function
    let func_ptr = engine.lookup("simple_identity");
    if !func_ptr.is_null() {
        // Cast function pointer and call it
        unsafe {
            let func: extern "C" fn(i32) -> i32 = std::mem::transmute(func_ptr);
            let test_value = 123;
            let result = func(test_value);
            println!("✅ simple_identity({}) returned: {}", test_value, result);

            if result == test_value {
                println!("🎯 Function executed correctly!");
            } else {
                println!(
                    "⚠️ Function returned unexpected value: {} (expected {})",
                    result, test_value
                );
            }
        }
    } else {
        println!("❌ Cannot test execution - simple_identity function not found");
    }

    Ok(())
}
