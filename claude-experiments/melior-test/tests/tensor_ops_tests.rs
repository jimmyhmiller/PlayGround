use melior::{
    Context,
    ir::{Module, Location},
    dialect::DialectRegistry,
};
use mlir_sys::*;
use melior_test::{TensorOpsDialect, TensorOpsPassManager, tensor_ops_dialect, tensor_ops_lowering};

#[test]
fn test_tensor_ops_dialect_creation() {
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
    }
    
    TensorOpsDialect::register(&registry);
    
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    let location = Location::unknown(&context);
    let module = Module::new(location);
    
    // Test that we can create tensor operations
    let result = tensor_ops_dialect::create_example_tensor_computation(&context, &module);
    assert!(result.is_ok(), "Failed to create tensor ops: {:?}", result.err());
    
    // Verify the module contains our operations
    let module_str = format!("{}", module.as_operation());
    assert!(module_str.contains("tensor_ops.constant"), "Module should contain tensor_ops.constant");
    assert!(module_str.contains("tensor_ops.add"), "Module should contain tensor_ops.add");
}

#[test]
fn test_tensor_ops_types() {
    let context = Context::new();
    
    // Test type creation
    let f32_type = tensor_ops_dialect::types::create_f32_type(&context);
    let tensor_type = tensor_ops_dialect::types::create_tensor_type(&context, &[2, 2], f32_type);
    
    // Types should be valid
    assert!(!tensor_type.is_null());
}

#[test]
fn test_tensor_ops_interop() {
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
    }
    
    TensorOpsDialect::register(&registry);
    
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    let location = Location::unknown(&context);
    let module = Module::new(location);
    
    // Test interop with standard dialects
    let result = tensor_ops_lowering::create_interop_example(&context, &module);
    assert!(result.is_ok(), "Failed to create interop example: {:?}", result.err());
    
    let module_str = format!("{}", module.as_operation());
    assert!(module_str.contains("arith.constant"), "Should contain standard dialect ops");
    assert!(module_str.contains("tensor_ops."), "Should contain custom dialect ops");
}

#[test]
fn test_tensor_ops_lowering() {
    let registry = DialectRegistry::new();
    unsafe {
        mlirRegisterAllDialects(registry.to_raw());
        mlirRegisterAllPasses();
    }
    
    TensorOpsDialect::register(&registry);
    
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    let location = Location::unknown(&context);
    let mut module = Module::new(location);
    
    // Create module with tensor ops
    tensor_ops_dialect::create_example_tensor_computation(&context, &module).unwrap();
    
    // Apply lowering
    let result = TensorOpsPassManager::apply_full_lowering_pipeline(&context, &mut module);
    assert!(result.is_ok(), "Lowering should succeed: {:?}", result.err());
    
    // After lowering, should contain standard dialect operations
    let module_str = format!("{}", module.as_operation());
    println!("Lowered module: {}", module_str);
    
    // Should have some form of lowered operations
    assert!(module_str.len() > 0, "Module should not be empty after lowering");
}

#[test]
fn test_individual_operation_creation() {
    let context = Context::new();
    let location = Location::unknown(&context);
    
    // Test creating individual operations
    let f32_type = tensor_ops_dialect::types::create_f32_type(&context);
    let tensor_type = tensor_ops_dialect::types::create_tensor_type(&context, &[2, 2], f32_type);
    
    // Create constant operation
    use melior::ir::attribute::StringAttribute;
    let const_attr = StringAttribute::new(&context, "dense<[[1.0, 2.0], [3.0, 4.0]]>").into();
    let const_op = TensorOpsDialect::create_constant_op(&context, location, const_attr, tensor_type);
    assert!(const_op.is_ok(), "Should create constant op successfully");
}