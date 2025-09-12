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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MLIR Transform + PDL Integration Demo");
    println!("========================================\n");
    
    // Initialize MLIR context
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    // Allow unregistered dialects for Transform and PDL
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    let _module = Module::new(location);
    
    println!("📋 Step 1: Creating sample tensor operations");
    println!("---------------------------------------------");
    
    // Create tensor operations
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
        const1.result(0)?.into(),
        const2.result(0)?.into(),
        tensor_type.into(),
    )?;
    
    println!("✅ Tensor constant 1: {}", const1);
    println!("✅ Tensor constant 2: {}", const2);
    println!("✅ Tensor addition: {}", add_op);
    
    println!("\n🎯 Step 2: Creating PDL patterns for pattern matching");
    println!("----------------------------------------------------");
    
    // Create PDL patterns with different benefits
    let low_benefit_pattern = PdlDialect::create_pattern(&context, location, 1)?;
    let high_benefit_pattern = PdlDialect::create_pattern(&context, location, 10)?;
    
    println!("✅ Low benefit pattern: {}", low_benefit_pattern);
    println!("✅ High benefit pattern: {}", high_benefit_pattern);
    
    // Create PDL matching components
    let pdl_type = PdlDialect::create_type(&context, location)?;
    let pdl_operand = PdlDialect::create_operand(&context, location, None)?;
    
    println!("✅ PDL type matcher: {}", pdl_type);
    println!("✅ PDL operand matcher: {}", pdl_operand);
    
    println!("\n🔄 Step 3: Creating Transform dialect sequences");
    println!("-----------------------------------------------");
    
    // Create transform sequences
    let main_sequence = TransformDialect::create_sequence(&context, location)?;
    let named_sequence = TransformDialect::create_named_sequence(
        &context,
        location,
        "tensor_optimization_pipeline",
    )?;
    
    println!("✅ Main transform sequence: {}", main_sequence);
    println!("✅ Named optimization pipeline: {}", named_sequence);
    
    // Create transform operations for control flow
    let yield_op = TransformDialect::create_yield(&context, location, &[])?;
    
    println!("✅ Transform yield: {}", yield_op);
    
    println!("\n🏗️  Step 4: High-level integration via builder");
    println!("----------------------------------------------");
    
    // Use the high-level builder
    let builder = TransformPdlBuilder::new(&context, location);
    
    let tensor_pattern = builder.create_tensor_optimization_pattern("tensor_ops.add")?;
    let transform_sequence = builder.create_tensor_transform_sequence()?;
    
    println!("✅ Builder-created tensor pattern: {}", tensor_pattern);
    println!("✅ Builder-created transform sequence: {}", transform_sequence);
    
    println!("\n✨ Integration Summary");
    println!("=====================");
    println!("🎯 Tensor Operations: Successfully created tensor constants and arithmetic");
    println!("🔍 PDL Patterns: Created pattern matching with benefit-based prioritization");
    println!("🔄 Transform Sequences: Built transformation pipelines with named sequences");
    println!("🏗️  High-level Builder: Integrated functionality via builder pattern");
    
    println!("\n🎪 Key Features Demonstrated:");
    println!("• PDL pattern creation with benefit levels");
    println!("• Transform sequence orchestration");  
    println!("• Named sequences for reusable transformations");
    println!("• Integration with existing tensor operations");
    println!("• High-level builder API for ease of use");
    
    println!("\n🔧 Technical Details:");
    println!("• Using unregistered dialects (transform, pdl)");
    println!("• Working with MLIR 0.19 via melior crate");
    println!("• Type-safe Rust wrappers around MLIR C API");
    println!("• Compatible with existing tensor operation infrastructure");
    
    println!("\n🚀 Transform + PDL Integration Demo Complete!");
    println!("This demonstrates a working foundation for:");
    println!("• Pattern-based tensor optimizations");
    println!("• Declarative transformation sequences");
    println!("• Extensible optimization pipelines");
    
    Ok(())
}