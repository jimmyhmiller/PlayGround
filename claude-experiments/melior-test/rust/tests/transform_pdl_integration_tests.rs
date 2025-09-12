use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        attribute::IntegerAttribute,
        Location, Module, 
        operation::OperationBuilder,
        r#type::{IntegerType, RankedTensorType},
    },
    utility::register_all_dialects,
};

use melior_test::{
    TensorOpsDialect, TransformDialect, PdlDialect, TransformPdlBuilder
};

#[test]
fn test_transform_sequence_creation() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    // Allow unregistered dialects for Transform and PDL
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test creating a Transform sequence
    let sequence = TransformDialect::create_sequence(&context, location)?;
    assert!(sequence.to_string().contains("transform.sequence"));
    
    println!("✅ Transform sequence created: {}", sequence);
    Ok(())
}

#[test]
fn test_pdl_pattern_creation() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test creating a PDL pattern
    let pattern = PdlDialect::create_pattern(&context, location, 1)?;
    assert!(pattern.to_string().contains("pdl.pattern"));
    assert!(pattern.to_string().contains("benefit = 1"));
    
    println!("✅ PDL pattern created: {}", pattern);
    Ok(())
}

#[test]
fn test_transform_named_sequence() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test creating a named sequence
    let named_seq = TransformDialect::create_named_sequence(
        &context, 
        location, 
        "tensor_optimization"
    )?;
    
    assert!(named_seq.to_string().contains("transform.named_sequence"));
    assert!(named_seq.to_string().contains("tensor_optimization"));
    
    println!("✅ Named sequence created: {}", named_seq);
    Ok(())
}

#[test]
fn test_pdl_operation_creation() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test creating PDL type and operand operations
    let pdl_type = PdlDialect::create_type(&context, location)?;
    assert!(pdl_type.to_string().contains("pdl.type"));
    
    let pdl_operand = PdlDialect::create_operand(&context, location, None)?;
    assert!(pdl_operand.to_string().contains("pdl.operand"));
    
    println!("✅ PDL type: {}", pdl_type);
    println!("✅ PDL operand: {}", pdl_operand);
    Ok(())
}

#[test]
fn test_tensor_ops_with_transform_integration() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Create tensor operations
    let i32_type = IntegerType::new(&context, 32);
    let tensor_type = RankedTensorType::new(&[2, 2], i32_type.into(), None);
    
    let constant_value = IntegerAttribute::new(i32_type.into(), 42);
    let constant_op = TensorOpsDialect::create_constant_op(
        &context,
        location,
        constant_value.into(),
        tensor_type.into(),
    )?;
    
    assert!(constant_op.to_string().contains("tensor_ops.constant"));
    
    // Create transform operations that could work with tensor ops
    let transform_seq = TransformDialect::create_sequence(&context, location)?;
    assert!(transform_seq.to_string().contains("transform.sequence"));
    
    println!("✅ Tensor constant: {}", constant_op);
    println!("✅ Transform sequence: {}", transform_seq);
    Ok(())
}

#[test]
fn test_transform_pdl_builder() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test the high-level builder
    let builder = TransformPdlBuilder::new(&context, location);
    
    let tensor_pattern = builder.create_tensor_optimization_pattern("tensor_ops.add")?;
    assert!(tensor_pattern.to_string().contains("pdl.pattern"));
    
    let transform_sequence = builder.create_tensor_transform_sequence()?;
    assert!(transform_sequence.to_string().contains("transform.sequence"));
    
    println!("✅ Builder tensor pattern: {}", tensor_pattern);
    println!("✅ Builder transform sequence: {}", transform_sequence);
    Ok(())
}

#[test]
fn test_complete_transform_pdl_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    let _module = Module::new(location);
    
    println!("🚀 Complete Transform + PDL workflow test");
    
    // Step 1: Create tensor operations
    let i32_type = IntegerType::new(&context, 32);
    let tensor_type = RankedTensorType::new(&[4, 4], i32_type.into(), None);
    
    let const1 = TensorOpsDialect::create_constant_op(
        &context,
        location,
        IntegerAttribute::new(i32_type.into(), 10).into(),
        tensor_type.into(),
    )?;
    
    let const2 = TensorOpsDialect::create_constant_op(
        &context,
        location,
        IntegerAttribute::new(i32_type.into(), 20).into(),
        tensor_type.into(),
    )?;
    
    let add_op = TensorOpsDialect::create_add_op(
        &context,
        location,
        const1.result(0)?,
        const2.result(0)?,
        tensor_type.into(),
    )?;
    
    println!("✅ Created tensor operations");
    
    // Step 2: Create PDL patterns for optimization
    let optimization_pattern = PdlDialect::create_pattern(&context, location, 2)?;
    println!("✅ Created optimization pattern: {}", optimization_pattern);
    
    // Step 3: Create transform sequence for applying patterns
    let main_sequence = TransformDialect::create_sequence(&context, location)?;
    println!("✅ Created main transform sequence: {}", main_sequence);
    
    // Step 4: Create named sequences for reusable optimizations
    let tensor_opt_sequence = TransformDialect::create_named_sequence(
        &context,
        location,
        "tensor_optimization_pipeline",
    )?;
    println!("✅ Created named optimization sequence: {}", tensor_opt_sequence);
    
    println!("✅ Complete workflow test passed!");
    println!("🎯 Tensor ops: {}", add_op);
    println!("🎯 PDL pattern: {}", optimization_pattern);  
    println!("🎯 Transform sequence: {}", main_sequence);
    
    Ok(())
}

#[test]
fn test_error_handling_and_robustness() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test that we can create operations without crashing
    let sequence_result = TransformDialect::create_sequence(&context, location);
    assert!(sequence_result.is_ok());
    
    let pattern_result = PdlDialect::create_pattern(&context, location, 1);
    assert!(pattern_result.is_ok());
    
    let named_seq_result = TransformDialect::create_named_sequence(
        &context,
        location,
        "test_sequence",
    );
    assert!(named_seq_result.is_ok());
    
    println!("✅ Error handling test passed - all operations created successfully");
    Ok(())
}