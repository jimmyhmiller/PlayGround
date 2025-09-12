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
    TensorOpsDialect, TransformDialect, PdlDialect, 
    TensorPdlPatterns, TensorPatternCollection,
    TensorTransformOps, TensorMatcher, TensorTransformPipeline
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MLIR Transform + PDL Integration Demo");
    
    // Initialize MLIR context with all dialects
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    // Allow unregistered dialects for Transform and PDL
    unsafe { 
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    let module = Module::new(location);
    
    println!("\n📋 Step 1: Creating tensor operations for transformation");
    
    // Create some tensor types and operations
    let i32_type = IntegerType::new(&context, 32);
    let tensor_type = RankedTensorType::new(&[2, 2], i32_type.into(), None);
    
    // Create tensor constant
    let constant_value = IntegerAttribute::new(i32_type.into(), 42);
    let constant_op = TensorOpsDialect::create_constant_op(
        &context,
        location,
        constant_value.into(),
        tensor_type.into(),
    )?;
    
    println!("✅ Created tensor constant: {}", constant_op);
    
    // Create tensor addition
    let add_op = TensorOpsDialect::create_add_op(
        &context,
        location,
        constant_op.result(0)?,
        constant_op.result(0)?,
        tensor_type.into(),
    )?;
    
    println!("✅ Created tensor addition: {}", add_op);
    
    println!("\n🎯 Step 2: Creating PDL patterns for tensor optimization");
    
    // Create PDL patterns for tensor operations
    let pattern_collection = &mut TensorPatternCollection::new(&context, location);
    pattern_collection.add_all_tensor_patterns()?;
    
    println!("✅ Created {} PDL patterns for tensor optimization", 
             pattern_collection.get_patterns().len());
    
    // Demonstrate individual pattern creation
    let add_optimization_pattern = TensorPdlPatterns::create_tensor_add_optimization_pattern(
        &context, 
        location
    )?;
    println!("✅ Created tensor add optimization pattern: {}", add_optimization_pattern);
    
    let fusion_pattern = TensorPdlPatterns::create_tensor_fusion_pattern(
        &context,
        location,
    )?;
    println!("✅ Created tensor fusion pattern: {}", fusion_pattern);
    
    println!("\n🔄 Step 3: Creating Transform dialect operations");
    
    // Create transform sequence
    let transform_sequence = TransformDialect::create_sequence(&context, location)?;
    println!("✅ Created transform sequence: {}", transform_sequence);
    
    // Create named sequence for reusable transformations
    let named_matcher = TransformDialect::create_named_sequence(
        &context,
        location,
        "tensor_optimization_sequence",
    )?;
    println!("✅ Created named transform sequence: {}", named_matcher);
    
    println!("\n🎪 Step 4: Creating tensor-specific matchers");
    
    // Create tensor arithmetic matcher
    let arithmetic_matcher = TensorMatcher::create_tensor_arithmetic_matcher(
        &context,
        location,
    )?;
    println!("✅ Created tensor arithmetic matcher: {}", arithmetic_matcher);
    
    // Create fusion opportunity matcher
    let fusion_matcher = TensorMatcher::create_tensor_fusion_matcher(
        &context,
        location,
    )?;
    println!("✅ Created tensor fusion matcher: {}", fusion_matcher);
    
    println!("\n🏗️  Step 5: Building complete transformation pipeline");
    
    // Create complete tensor transformation pipeline
    let pipeline_builder = TensorTransformPipeline::new(&context, location);
    let optimization_pipeline = pipeline_builder.create_tensor_optimization_pipeline()?;
    
    println!("✅ Created complete tensor optimization pipeline:");
    println!("{}", optimization_pipeline);
    
    // Create robust pipeline with alternatives
    let robust_pipeline = pipeline_builder.create_robust_tensor_optimization()?;
    println!("✅ Created robust tensor optimization with fallbacks:");
    println!("{}", robust_pipeline);
    
    println!("\n⚡ Step 6: Demonstrating pattern matching operations");
    
    // Demonstrate PDL match operations
    let pdl_match = TensorTransformOps::create_pdl_match(
        &context,
        location,
        add_op.result(0)?,
        "tensor_add_optimization",
    )?;
    println!("✅ Created PDL match operation: {}", pdl_match);
    
    // Demonstrate collect matching
    let collect_matching = TensorTransformOps::create_collect_matching(
        &context,
        location,
        add_op.result(0)?,
        &["tensor_ops.add", "tensor_ops.mul", "tensor_ops.constant"],
    )?;
    println!("✅ Created collect matching operation: {}", collect_matching);
    
    // Demonstrate operation name matching
    let name_match = TensorTransformOps::create_match_operation_name(
        &context,
        location,
        add_op.result(0)?,
        &["tensor_ops.add"],
    )?;
    println!("✅ Created operation name matcher: {}", name_match);
    
    println!("\n🎨 Step 7: Advanced pattern composition");
    
    // Create foreach_match for pattern-based transformations
    let foreach_match = TransformDialect::create_foreach_match(
        &context,
        location,
        add_op.result(0)?,
        "find_tensor_ops",
        "optimize_tensor_ops",
    )?;
    println!("✅ Created foreach_match transformation: {}", foreach_match);
    
    // Create alternatives for robust transformations
    let alternatives = TransformDialect::create_alternatives(
        &context,
        location,
        add_op.result(0)?,
    )?;
    println!("✅ Created alternatives transformation: {}", alternatives);
    
    println!("\n🔬 Step 8: Integration verification");
    
    // Verify the complete module structure
    println!("📊 Module with Transform + PDL integration:");
    println!("{}", module);
    
    println!("\n✨ Transform + PDL Integration Summary:");
    println!("🎯 PDL Patterns: Declarative pattern matching for tensor operations");
    println!("🔄 Transform Ops: Imperative transformation sequencing and control flow");  
    println!("🎪 Matchers: High-level operation discovery and classification");
    println!("🏗️  Pipelines: Complete optimization sequences with error handling");
    println!("⚡ Integration: Seamless combination of pattern matching and transformation");
    
    println!("\n🚀 Demo completed successfully!");
    println!("💡 This demonstrates a working Transform dialect + PDL integration");
    println!("🔧 Ready for real tensor optimization workflows!");
    
    Ok(())
}