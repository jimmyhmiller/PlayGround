use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        attribute::IntegerAttribute,
        Location, Module, 
        r#type::{IntegerType, RankedTensorType},
    },
    utility::register_all_dialects,
};

use melior_test::{
    TensorOpsDialect, TransformDialect, PdlDialect, TransformPdlBuilder
};

#[test]
fn test_basic_transform_operations() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    // Allow unregistered dialects for Transform and PDL
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test Transform dialect operations
    let sequence = TransformDialect::create_sequence(&context, location)?;
    assert!(sequence.to_string().contains("transform.sequence"));
    println!("✅ Transform sequence: {}", sequence);
    
    let named_seq = TransformDialect::create_named_sequence(&context, location, "test")?;
    assert!(named_seq.to_string().contains("transform.named_sequence"));
    assert!(named_seq.to_string().contains("test"));
    println!("✅ Named sequence: {}", named_seq);
    
    Ok(())
}

#[test]
fn test_basic_pdl_operations() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Test PDL dialect operations
    let pattern = PdlDialect::create_pattern(&context, location, 5)?;
    assert!(pattern.to_string().contains("pdl.pattern"));
    assert!(pattern.to_string().contains("benefit = 5"));
    println!("✅ PDL pattern: {}", pattern);
    
    let pdl_type = PdlDialect::create_type(&context, location)?;
    assert!(pdl_type.to_string().contains("pdl.type"));
    println!("✅ PDL type: {}", pdl_type);
    
    let pdl_operand = PdlDialect::create_operand(&context, location, None)?;
    assert!(pdl_operand.to_string().contains("pdl.operand"));
    println!("✅ PDL operand: {}", pdl_operand);
    
    Ok(())
}

#[test]
fn test_transform_pdl_integration() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    // Create a tensor operation
    let i32_type = IntegerType::new(&context, 32);
    let tensor_type = RankedTensorType::new(&[2, 2], i32_type.into(), None);
    let constant_value = IntegerAttribute::new(i32_type.into(), 42);
    let tensor_op = TensorOpsDialect::create_constant_op(
        &context,
        location,
        constant_value.into(),
        tensor_type.into(),
    )?;
    
    println!("✅ Created tensor operation: {}", tensor_op);
    
    // Create transform and PDL operations that could work together
    let transform_seq = TransformDialect::create_sequence(&context, location)?;
    let pdl_pattern = PdlDialect::create_pattern(&context, location, 1)?;
    
    println!("✅ Transform sequence: {}", transform_seq);
    println!("✅ PDL pattern: {}", pdl_pattern);
    
    // Test the high-level builder
    let builder = TransformPdlBuilder::new(&context, location);
    let optimization_pattern = builder.create_tensor_optimization_pattern("tensor_ops.add")?;
    let transform_sequence = builder.create_tensor_transform_sequence()?;
    
    println!("✅ Builder optimization pattern: {}", optimization_pattern);
    println!("✅ Builder transform sequence: {}", transform_sequence);
    
    Ok(())
}

#[test] 
fn test_complete_workflow_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Transform + PDL Integration Workflow Test");
    
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    let _module = Module::new(location);
    
    // Step 1: Create tensor operations to optimize
    println!("📋 Step 1: Creating tensor operations");
    
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
    
    // Create an addition operation
    let add_op = TensorOpsDialect::create_add_op(
        &context,
        location,
        const1.result(0)?.into(),
        const2.result(0)?.into(),
        tensor_type.into(),
    )?;
    
    println!("✅ Tensor constant 1: {}", const1);
    println!("✅ Tensor constant 2: {}", const2); 
    println!("✅ Tensor addition: {}", add_op);
    
    // Step 2: Create PDL patterns for matching
    println!("\n🎯 Step 2: Creating PDL patterns");
    
    let pattern1 = PdlDialect::create_pattern(&context, location, 1)?;
    let pattern2 = PdlDialect::create_pattern(&context, location, 2)?;
    
    println!("✅ PDL pattern 1 (benefit=1): {}", pattern1);
    println!("✅ PDL pattern 2 (benefit=2): {}", pattern2);
    
    // Step 3: Create Transform sequences
    println!("\n🔄 Step 3: Creating Transform sequences");
    
    let main_sequence = TransformDialect::create_sequence(&context, location)?;
    let opt_sequence = TransformDialect::create_named_sequence(
        &context,
        location,
        "tensor_optimization_pipeline",
    )?;
    
    println!("✅ Main transform sequence: {}", main_sequence);
    println!("✅ Named optimization sequence: {}", opt_sequence);
    
    // Step 4: Integration via high-level builder
    println!("\n🏗️  Step 4: High-level integration");
    
    let builder = TransformPdlBuilder::new(&context, location);
    
    let tensor_pattern = builder.create_tensor_optimization_pattern("tensor_ops.add")?;
    let tensor_transform = builder.create_tensor_transform_sequence()?;
    
    println!("✅ Integrated tensor pattern: {}", tensor_pattern);
    println!("✅ Integrated tensor transform: {}", tensor_transform);
    
    println!("\n✨ Integration Summary:");
    println!("🎯 Created {} tensor operations", 3);
    println!("🔍 Created {} PDL patterns for matching", 2);
    println!("🔄 Created {} Transform sequences", 2);
    println!("🏗️  Integrated via high-level builder");
    
    println!("\n🚀 Complete Transform + PDL workflow test passed!");
    
    Ok(())
}