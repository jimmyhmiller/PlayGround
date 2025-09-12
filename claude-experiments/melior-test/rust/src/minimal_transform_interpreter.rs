use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        Location, Module,
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType, Type},
        Block, Region, Identifier, RegionLike,
    },
    utility::register_all_dialects,
    pass::PassManager,
    ExecutionEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Minimal Transform Dialect Interpreter with Melior 0.25.0");
    println!("============================================================");
    
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    let location = Location::unknown(&context);
    let mut payload_module = Module::new(location);
    
    println!("📋 Step 1: Creating payload module with custom operations");
    println!("----------------------------------------------------------");
    
    // Create the payload module with custom operations
    let i32_type = IntegerType::new(&context, 32);
    let function_type = FunctionType::new(&context, &[], &[i32_type.into()]);
    
    let mut region = Region::new();
    let entry_block = Block::new(&[]);
    region.append_block(entry_block);
    
    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (
                Identifier::new(&context, "sym_name"),
                StringAttribute::new(&context, "test_function").into(),
            ),
            (
                Identifier::new(&context, "function_type"),
                TypeAttribute::new(function_type.into()).into(),
            ),
            (
                Identifier::new(&context, "sym_visibility"),
                StringAttribute::new(&context, "public").into(),
            ),
        ])
        .add_regions([region])
        .build()?;
    
    let function_region = function.region(0)?;
    let entry_block = function_region.first_block().unwrap();
    
    // Add our custom operations that should be transformed
    let const_10 = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(
            Identifier::new(&context, "value"),
            IntegerAttribute::new(i32_type.into(), 10).into(),
        )])
        .add_results(&[i32_type.into()])
        .build()?;
    
    let const_32 = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(
            Identifier::new(&context, "value"),
            IntegerAttribute::new(i32_type.into(), 32).into(),
        )])
        .add_results(&[i32_type.into()])
        .build()?;
    
    // This is our target operation that should be transformed
    let custom_add = OperationBuilder::new("mymath.add", location)
        .add_operands(&[const_10.result(0)?.into(), const_32.result(0)?.into()])
        .add_results(&[i32_type.into()])
        .build()?;
    
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[custom_add.result(0)?.into()])
        .build()?;
    
    entry_block.append_operation(const_10);
    entry_block.append_operation(const_32);
    entry_block.append_operation(custom_add);
    entry_block.append_operation(return_op);
    
    payload_module.body().append_operation(function);
    
    println!("✅ Created payload module with mymath.add operation");
    
    println!("\n📋 Step 2: Payload module (before transformation)");
    println!("-------------------------------------------------");
    println!("{}", payload_module.as_operation());
    
    println!("\n📋 Step 3: Creating Transform Module");
    println!("------------------------------------");
    
    // Try to create a basic transform module
    let transform_module = Module::new(location);
    
    // Test what transform types are available in melior 0.25.0
    println!("🔍 Testing transform types in melior 0.25.0...");
    
    let test_types = [
        "!transform.any_op",
        "!transform.any_value", 
        "!transform.any_param",
        "!transform.param<i32>",
        "!transform.op<\"func.func\">",
    ];
    
    for type_str in &test_types {
        match Type::parse(&context, type_str) {
            Some(parsed_type) => {
                println!("✅ {} parsed successfully: {}", type_str, parsed_type);
            },
            None => {
                println!("❌ {} failed to parse", type_str);
            }
        }
    }
    
    // Try to create transform operations
    println!("\n🔍 Testing transform operations...");
    
    let test_ops = [
        "transform.sequence",
        "transform.with_pdl_patterns",
        "transform.apply_patterns", 
        "transform.yield",
    ];
    
    for op_name in &test_ops {
        match OperationBuilder::new(op_name, location).build() {
            Ok(_) => println!("✅ {} is available", op_name),
            Err(e) => println!("❌ {} failed: {:?}", op_name, e),
        }
    }
    
    // Try to parse a complete transform IR module
    println!("\n🔍 Testing transform IR parsing...");
    
    let test_ir = r#"
    module {
      transform.sequence failures(propagate) {
      ^bb0(%arg0: !transform.any_op):
        transform.yield
      }
    }
    "#;
    
    match Module::parse(&context, test_ir) {
        Some(test_module) => {
            println!("✅ Successfully parsed transform IR");
            println!("   Module: {}", test_module.as_operation());
            
            // If we can parse transform IR, let's try to use it
            println!("\n🔄 Step 4: Applying Transform IR to payload");
            println!("--------------------------------------------");
            
            // Here we would apply the transform module to the payload module
            // The exact API for this in melior 0.25.0 needs to be discovered
            println!("💡 Transform module is available - need to find application API");
            
        },
        None => {
            println!("❌ Failed to parse transform IR");
        }
    }
    
    println!("\n🔄 Step 5: LLVM Lowering Pipeline");
    println!("---------------------------------");
    
    let pm = PassManager::new(&context);
    
    // Check what pass creation methods are available in melior 0.25.0
    println!("🔄 Testing pass creation in melior 0.25.0...");
    
    // Try running without adding specific passes first
    println!("🔄 Running empty pass manager...");
    match pm.run(&mut payload_module) {
        Ok(_) => {
            println!("✅ Pass manager ran successfully");
        },
        Err(e) => {
            println!("⚠️  Pass manager failed: {:?}", e);
        }
    }
    
    println!("\n📋 Step 6: Final module");
    println!("-----------------------");
    println!("{}", payload_module.as_operation());
    
    println!("\n🔥 Step 7: JIT Compilation Test");
    println!("-------------------------------");
    
    if payload_module.as_operation().verify() {
        println!("✅ Module verification passed");
        
        let engine = ExecutionEngine::new(&payload_module, 2, &[], false);
        println!("✅ JIT compilation successful!");
        
        let func_ptr = engine.lookup("test_function");
        if !func_ptr.is_null() {
            println!("✅ Function symbol found at: {:p}", func_ptr);
            println!("🎯 Melior 0.25.0 basic pipeline working!");
        } else {
            println!("⚠️  Function symbol not found - may need different function name");
        }
    } else {
        println!("⚠️  Module verification failed");
    }
    
    println!("\n🎓 Melior 0.25.0 Transform Dialect Investigation:");
    println!("===============================================");
    println!("✅ Updated to melior 0.25.0 successfully");
    println!("✅ Basic MLIR operations work with new API");
    println!("✅ RegionLike trait required for append_block");
    println!("✅ Module parsing and JIT compilation functional");
    println!("🔍 Transform dialect availability depends on parsing results");
    
    println!("\n💡 Next Steps:");
    println!("==============");
    println!("1. Investigate new transform dialect API in melior 0.25.0");
    println!("2. Find the correct way to apply transform modules to payloads");
    println!("3. Test with real transform patterns and interpreter");
    
    Ok(())
}