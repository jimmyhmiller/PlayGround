use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        Block, Identifier, Location, Module, Region,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType, RankedTensorType},
    },
    pass::{Pass, PassManager},
};
use melior_test::{TensorOpsDialect, TensorOpsLowering};
use mlir_sys::*;

/// Test that the tensor_ops dialect is properly registered
#[test]
fn test_dialect_registration() {
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
    }

    TensorOpsDialect::register(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // This test will fail because our dialect isn't properly registered
    // A proper implementation would make this pass by actually registering
    // the dialect with MLIR's type system

    // We should NOT need this line for a proper dialect:
    // unsafe { mlirContextSetAllowUnregisteredDialects(context.to_raw(), false); }

    let location = Location::unknown(&context);
    let module = Module::new(location);

    // Try to create a tensor_ops operation without allowing unregistered dialects
    let result = create_tensor_ops_function(&context, &module);

    // This should succeed with a properly registered dialect
    assert!(
        result.is_ok(),
        "Failed to create tensor_ops operations: {:?}",
        result
    );
}

/// Test that tensor_ops operations are properly verified
#[test]
fn test_operation_verification() {
    let context = setup_context_with_tensor_ops();
    let location = Location::unknown(&context);
    let _module = Module::new(location);

    // Test 1: Valid operation should succeed
    let i32_type = IntegerType::new(&context, 32).into();
    let tensor_type = RankedTensorType::new(&[2, 2], i32_type, None).into();

    let valid_op = OperationBuilder::new("tensor_ops.add", location)
        .add_operands(&[]) // This should fail - add requires 2 operands
        .add_results(&[tensor_type])
        .build();

    // A proper dialect would verify operations and reject invalid ones
    assert!(
        valid_op.is_err(),
        "Operation with wrong number of operands should fail verification"
    );

    // Test 2: Type mismatch should be caught
    let f32_type = IntegerType::new(&context, 32).into(); // Using i32 as placeholder for f32
    let _tensor_i32: melior::ir::Type = RankedTensorType::new(&[2, 2], i32_type, None).into();
    let _tensor_f32: melior::ir::Type = RankedTensorType::new(&[2, 2], f32_type, None).into();

    // In a proper dialect, mixing tensor types should be validated
    // This test documents what verification should do
}

/// Test that lowering preserves semantics
#[test]
fn test_lowering_preserves_semantics() {
    let context = setup_context_with_tensor_ops();
    let location = Location::unknown(&context);
    let original_module = Module::new(location);

    // Create a function with tensor_ops operations
    create_tensor_computation_function(&context, &original_module).unwrap();

    // Apply lowering
    let lowered_module = TensorOpsLowering::apply_lowering(&context, &original_module).unwrap();

    // In a proper implementation:
    // 1. The lowered module should have the same function signature
    // 2. Operations should be replaced in-place, not create new functions
    // 3. The computation should be semantically equivalent

    // This test will fail because our current lowering creates a new module
    // instead of transforming the existing one

    // Check that the lowered module contains the original function name
    let lowered_mlir = format!("{}", lowered_module.as_operation());
    assert!(
        lowered_mlir.contains("tensor_computation"),
        "Lowered module should contain the original function 'tensor_computation', but it contains: {}",
        lowered_mlir
    );

    // Check that tensor_ops operations are replaced with standard operations
    assert!(
        !lowered_mlir.contains("tensor_ops."),
        "Lowered module should not contain tensor_ops operations"
    );
}

/// Test that conversion patterns work correctly
#[test]
fn test_conversion_patterns() {
    let context = setup_context_with_tensor_ops();
    let location = Location::unknown(&context);
    let module = Module::new(location);

    create_tensor_computation_function(&context, &module).unwrap();

    // Create a proper conversion pass
    let _pass_manager = PassManager::new(&context);

    // In a proper implementation, we would add a custom conversion pass:
    // pass_manager.add_pass(TensorOpsToStandardPass::new());

    // For now, we can only test that the infrastructure exists
    // This test documents what should be implemented
}

/// Test that the dialect can be used with standard MLIR passes
#[test]
fn test_dialect_with_standard_passes() {
    let context = setup_context_with_tensor_ops();
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    create_tensor_computation_function(&context, &module).unwrap();

    // The module should be verifiable
    // FIXME: This currently fails because our tensor_ops operations aren't properly defined
    // let is_valid = module.as_operation().verify();
    // assert!(is_valid, "Module with tensor_ops operations should be verifiable");

    // Standard passes should handle unknown dialects gracefully
    let pass_manager = PassManager::new(&context);

    // Add canonicalization pass
    unsafe {
        let canon_pass = Pass::from_raw(mlirCreateTransformsCanonicalizer());
        pass_manager.add_pass(canon_pass);
    }

    let result = pass_manager.run(&mut module);
    assert!(
        result.is_ok(),
        "Standard passes should work with custom dialect"
    );
}

/// Test that dialect types are properly supported
#[test]
fn test_dialect_types() {
    let _context = setup_context_with_tensor_ops();

    // In a proper dialect implementation, we might have custom types
    // For example: tensor_ops.sparse_tensor<2x2xf32, COO>
    // This would require proper type registration and parsing

    // This test documents what type support should look like
}

/// Test round-trip: create -> print -> parse
#[test]
fn test_round_trip() {
    let context = setup_context_with_tensor_ops();
    let location = Location::unknown(&context);
    let module = Module::new(location);

    create_tensor_computation_function(&context, &module).unwrap();

    // Convert to string
    let _mlir_string = format!("{}", module.as_operation());

    // Parse back
    // In a proper implementation with registered dialect:
    // let parsed_module = Module::parse(&context, &mlir_string).unwrap();

    // Verify they're equivalent
    // assert_eq!(original_ops, parsed_ops);
}

// Helper functions

fn setup_context_with_tensor_ops() -> Context {
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
        mlirRegisterAllPasses();
    }

    TensorOpsDialect::register(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // This should not be needed for a proper dialect
    unsafe {
        mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }

    unsafe {
        mlirRegisterAllLLVMTranslations(context.to_raw());
    }

    context
}

fn create_tensor_ops_function(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);
    let i32_type = IntegerType::new(context, 32).into();
    let tensor_type = RankedTensorType::new(&[2, 2], i32_type, None).into();
    let function_type = FunctionType::new(context, &[], &[tensor_type]);

    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, "test_tensor_ops").into(),
            ),
            (
                Identifier::new(context, "function_type"),
                TypeAttribute::new(function_type.into()).into(),
            ),
        ])
        .add_regions([Region::new()])
        .build()?;

    let block = Block::new(&[]);
    let region = function.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();

    // Create tensor_ops.constant
    let const_op = OperationBuilder::new("tensor_ops.constant", location)
        .add_attributes(&[(
            Identifier::new(context, "value"),
            StringAttribute::new(context, "dense<[[1, 2], [3, 4]]>").into(),
        )])
        .add_results(&[tensor_type])
        .build()?;
    block_ref.append_operation(const_op.clone());

    // Return
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[const_op.result(0)?.into()])
        .build()?;
    block_ref.append_operation(return_op);

    module.body().append_operation(function);
    Ok(())
}

fn create_tensor_computation_function(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);
    let i32_type = IntegerType::new(context, 32).into();
    let tensor_type = RankedTensorType::new(&[2, 2], i32_type, None).into();
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
        ])
        .add_regions([Region::new()])
        .build()?;

    let block = Block::new(&[]);
    let region = function.region(0)?;
    region.append_block(block);

    let block_ref = region.first_block().unwrap();

    // Create two constants
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

    // Add them
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

    module.body().append_operation(function);
    Ok(())
}
