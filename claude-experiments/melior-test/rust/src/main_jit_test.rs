//! Test JIT compilation and function lookup without tensor_ops dialect
//!
//! This test creates simple identity functions using only standard MLIR operations
//! to verify that the JIT compilation and function symbol lookup works correctly.

use melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    ir::{
        Block, Identifier, Location, Module, Region,
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType},
    },
    pass::PassManager,
};
use mlir_sys::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing MLIR JIT function lookup...");

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

    // Create simple functions with side effects to prevent optimization
    create_simple_functions(&context, &module)?;

    // Convert to LLVM IR
    let mut final_module = module;
    apply_lowering_passes(&context, &mut final_module)?;

    // JIT compile and test function lookup
    test_function_lookup(&context, &final_module)?;

    Ok(())
}

fn create_simple_functions(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);
    let i32_type = IntegerType::new(context, 32).into();

    // Create add function: i32 add(i32 a, i32 b) { return a + b; }
    create_add_function(context, module, location, i32_type)?;

    // Create identity function: i32 identity(i32 x) { return x; }
    create_identity_function(context, module, location, i32_type)?;

    // Create constant function: i32 get_constant() { return 42; }
    create_constant_function(context, module, location, i32_type)?;

    Ok(())
}

fn create_add_function(
    context: &Context,
    module: &Module,
    location: Location,
    i32_type: melior::ir::Type,
) -> Result<(), Box<dyn std::error::Error>> {
    let function_type = FunctionType::new(context, &[i32_type, i32_type], &[i32_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "simple_add").into(),
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

    let block = Block::new(&[(i32_type, location), (i32_type, location)]);
    let function_clone = function.clone();
    let region = function_clone.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();
    let arg0 = block_ref.argument(0)?.into();
    let arg1 = block_ref.argument(1)?.into();

    // Add the two arguments using arith.addi
    let add_op = OperationBuilder::new("arith.addi", location)
        .add_operands(&[arg0, arg1])
        .add_results(&[i32_type])
        .build()?;
    block_ref.append_operation(add_op.clone());

    // Return the result
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[add_op.result(0)?.into()])
        .build()?;
    block_ref.append_operation(return_op);

    module.body().append_operation(function);
    Ok(())
}

fn create_identity_function(
    context: &Context,
    module: &Module,
    location: Location,
    i32_type: melior::ir::Type,
) -> Result<(), Box<dyn std::error::Error>> {
    let function_type = FunctionType::new(context, &[i32_type], &[i32_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "simple_identity").into(),
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

    // Simply return the input
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[arg0])
        .build()?;
    block_ref.append_operation(return_op);

    module.body().append_operation(function);
    Ok(())
}

fn create_constant_function(
    context: &Context,
    module: &Module,
    location: Location,
    i32_type: melior::ir::Type,
) -> Result<(), Box<dyn std::error::Error>> {
    let function_type = FunctionType::new(context, &[], &[i32_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "get_constant").into(),
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

    let block = Block::new(&[]);
    let function_clone = function.clone();
    let region = function_clone.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();

    // Create constant 42
    let const_op = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(
            Identifier::new(context, "value"),
            IntegerAttribute::new(i32_type, 42).into(),
        )])
        .add_results(&[i32_type])
        .build()?;
    block_ref.append_operation(const_op.clone());

    // Return the constant
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[const_op.result(0)?.into()])
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

        // Convert arith dialect to LLVM dialect
        let arith_to_llvm = Pass::from_raw(mlirCreateConversionConvertMathToLLVMPass());
        pass_manager.add_pass(arith_to_llvm);

        // Finalize conversion to LLVM
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
        let function_names = ["simple_add", "simple_identity", "get_constant"];
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

            // Test calling the constant function
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

    // Try to get and call the constant function
    let func_ptr = engine.lookup("get_constant");
    if !func_ptr.is_null() {
        // Cast function pointer and call it
        unsafe {
            let func: extern "C" fn() -> i32 = std::mem::transmute(func_ptr);
            let result = func();
            println!("✅ get_constant() returned: {}", result);

            if result == 42 {
                println!("🎯 Function executed correctly!");
            } else {
                println!(
                    "⚠️ Function returned unexpected value: {} (expected 42)",
                    result
                );
            }
        }
    } else {
        println!("❌ Cannot test execution - get_constant function not found");
    }

    Ok(())
}
