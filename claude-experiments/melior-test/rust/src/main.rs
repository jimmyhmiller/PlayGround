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

mod tensor_ops_dialect;
// mod tensor_ops_lowering; // Temporarily disabled

use tensor_ops_dialect::TensorOpsDialect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup MLIR context and dialects
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
        mlirRegisterAllPasses();
    }

    // Register custom TensorOps dialect
    TensorOpsDialect::register(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Allow unregistered dialects for tensor_ops
    unsafe {
        mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }

    // Register LLVM translation interfaces
    unsafe {
        mlirRegisterAllLLVMTranslations(context.to_raw());
    }

    let location = Location::unknown(&context);
    let module = Module::new(location);

    // Create function using tensor_ops dialect
    create_tensor_computation(&context, &module)?;

    // Apply lowering: tensor_ops -> standard dialects -> LLVM
    // let lowered_module = tensor_ops_lowering::TensorOpsLowering::apply_lowering(&context, &module)?; // Temporarily disabled
    let mut final_module = module; // Use original module for now

    // Apply standard to LLVM lowering passes
    let pass_manager = PassManager::new(&context);
    unsafe {
        use melior::pass::Pass;

        let func_to_llvm = Pass::from_raw(mlirCreateConversionConvertFuncToLLVMPass());
        pass_manager.add_pass(func_to_llvm);

        let math_to_llvm = Pass::from_raw(mlirCreateConversionConvertMathToLLVMPass());
        pass_manager.add_pass(math_to_llvm);

        let reconcile_pass = Pass::from_raw(mlirCreateConversionReconcileUnrealizedCasts());
        pass_manager.add_pass(reconcile_pass);
    }

    pass_manager.run(&mut final_module)?;

    // JIT compile and execute
    jit_compile_and_execute(&context, &final_module)?;

    Ok(())
}

fn create_tensor_computation(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);

    // Create function that uses tensor_ops dialect
    let i32_type = IntegerType::new(context, 32).into();
    let tensor_type = melior::ir::r#type::RankedTensorType::new(&[2, 2], i32_type, None).into();
    let function_type = FunctionType::new(context, &[], &[tensor_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "tensor_computation").into(),
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

    // Create function body
    let block = Block::new(&[]);
    let region = function.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();

    // Create tensor_ops.constant operations
    let const1_op = OperationBuilder::new("tensor_ops.constant", location)
        .add_attributes(&[(
            Identifier::new(context, "value"),
            StringAttribute::new(context, "dense<[[1, 2], [3, 4]]>").into(),
        )])
        .add_results(&[tensor_type])
        .build()?;
    block_ref.append_operation(const1_op.clone());

    let const2_op = OperationBuilder::new("tensor_ops.constant", location)
        .add_attributes(&[(
            Identifier::new(context, "value"),
            StringAttribute::new(context, "dense<[[5, 6], [7, 8]]>").into(),
        )])
        .add_results(&[tensor_type])
        .build()?;
    block_ref.append_operation(const2_op.clone());

    // Create tensor_ops.add operation
    let add_op = OperationBuilder::new("tensor_ops.add", location)
        .add_operands(&[const1_op.result(0)?.into(), const2_op.result(0)?.into()])
        .add_results(&[tensor_type])
        .build()?;
    block_ref.append_operation(add_op.clone());

    // Return result
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[add_op.result(0)?.into()])
        .build()?;
    block_ref.append_operation(return_op);

    // Add function to module
    module.body().append_operation(function);

    Ok(())
}

fn jit_compile_and_execute(
    _context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create ExecutionEngine with optimizations disabled
    let engine = ExecutionEngine::new(module, 0, &[], false);

    println!("✅ ExecutionEngine created successfully!");

    // Try to lookup functions
    let function_names = ["tensor_computation", "tensor_add", "tensor_mul"];
    for name in &function_names {
        let func_ptr = engine.lookup(name);
        if func_ptr.is_null() {
            println!("⚠️ Could not find '{}' function", name);
        } else {
            println!("✅ Found '{}' function at {:?}", name, func_ptr);
        }
    }

    Ok(())
}
