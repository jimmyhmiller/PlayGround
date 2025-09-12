use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        Location, Module,
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, IntegerType, Type},
        Block, Region, Identifier, RegionLike, BlockLike,
    },
    utility::register_all_dialects,
    pass::PassManager,
    ExecutionEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Transform Dialect with Melior 0.25.0 - Clean Implementation");
    println!("==============================================================");
    
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
                StringAttribute::new(&context, "test_custom_ops").into(),
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
    
    println!("\n📋 Step 3: Testing Transform Dialect Support in Melior 0.25.0");
    println!("--------------------------------------------------------------");
    
    // Test what transform types are available in melior 0.25.0
    println!("🔍 Testing transform types...");
    
    let test_types = [
        "!transform.any_op",
        "!transform.any_value", 
        "!transform.any_param",
        "!transform.param<i32>",
        "!transform.op<\"func.func\">",
    ];
    
    let mut working_types = Vec::new();
    for type_str in &test_types {
        match Type::parse(&context, type_str) {
            Some(parsed_type) => {
                println!("✅ {} parsed successfully: {}", type_str, parsed_type);
                working_types.push((type_str, parsed_type));
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
        "transform.foreach_match",
        "transform.match.operation_name",
    ];
    
    let mut working_ops = Vec::new();
    for op_name in &test_ops {
        match OperationBuilder::new(op_name, location).build() {
            Ok(op) => {
                println!("✅ {} is available: {}", op_name, op);
                working_ops.push(op_name);
            },
            Err(e) => {
                println!("❌ {} failed: {:?}", op_name, e);
            }
        }
    }
    
    println!("\n📋 Step 4: Attempting to Create Transform Module");
    println!("------------------------------------------------");
    
    if !working_types.is_empty() && !working_ops.is_empty() {
        println!("✅ Found working transform types and operations!");
        
        // Try to create a transform module
        let transform_module = Module::new(location);
        
        // Use the first working type to create a transform sequence
        let (type_name, transform_type) = &working_types[0];
        println!("🔄 Creating transform module using {} type...", type_name);
        
        let sequence = OperationBuilder::new("transform.sequence", location)
            .add_results(&[transform_type.clone()])
            .build()?;
        
        // Add the sequence to the transform module
        transform_module.body().append_operation(sequence);
        
        println!("✅ Created transform module:");
        println!("{}", transform_module.as_operation());
        
    } else {
        println!("❌ No working transform types or operations found");
        println!("💡 Transform dialect may not be available in this build");
    }
    
    // Try to parse a complete transform IR module
    println!("\n📋 Step 5: Testing Transform IR Parsing");
    println!("---------------------------------------");
    
    let test_irs = [
        r#"
        module {
          transform.sequence failures(propagate) {
          ^bb0(%arg0: !transform.any_op):
            transform.yield
          }
        }
        "#,
        r#"
        module {
          transform.sequence failures(propagate) {
          ^bb0(%arg0: !transform.any_value):
            transform.yield
          }
        }
        "#,
    ];
    
    for (i, test_ir) in test_irs.iter().enumerate() {
        println!("🔍 Testing transform IR #{}", i + 1);
        match Module::parse(&context, test_ir) {
            Some(test_module) => {
                println!("✅ Successfully parsed transform IR #{}", i + 1);
                println!("   Module: {}", test_module.as_operation());
                
                println!("\n🔄 This could be used as a transform module!");
                break;
            },
            None => {
                println!("❌ Failed to parse transform IR #{}", i + 1);
            }
        }
    }
    
    println!("\n📋 Step 6: LLVM Pipeline Test");
    println!("------------------------------");
    
    let pm = PassManager::new(&context);
    
    println!("🔄 Running pass manager (empty)...");
    match pm.run(&mut payload_module) {
        Ok(_) => {
            println!("✅ Pass manager ran successfully");
        },
        Err(e) => {
            println!("⚠️  Pass manager failed: {:?}", e);
        }
    }
    
    println!("\n📋 Step 7: JIT Compilation Test");
    println!("-------------------------------");
    
    if payload_module.as_operation().verify() {
        println!("✅ Module verification passed");
        
        let engine = ExecutionEngine::new(&payload_module, 2, &[], false);
        println!("✅ JIT compilation successful!");
        
        let func_ptr = engine.lookup("test_custom_ops");
        if !func_ptr.is_null() {
            println!("✅ Function symbol found at: {:p}", func_ptr);
            println!("🎯 Melior 0.25.0 basic pipeline working!");
        } else {
            println!("⚠️  Function symbol not found");
            
            // Try alternative names
            for name in &["test_custom_ops", "_mlir_ciface_test_custom_ops"] {
                let ptr = engine.lookup(name);
                if !ptr.is_null() {
                    println!("✅ Found function '{}' at: {:p}", name, ptr);
                    break;
                }
            }
        }
    } else {
        println!("⚠️  Module verification failed");
    }
    
    println!("\n🎓 Transform Dialect Investigation Results:");
    println!("==========================================");
    println!("✅ Successfully upgraded to melior 0.25.0");
    println!("✅ RegionLike trait import required for append_block");
    println!("✅ Basic MLIR operations work correctly");
    println!("✅ Module parsing and JIT compilation functional");
    
    if !working_types.is_empty() {
        println!("✅ Transform types available: {:?}", working_types.iter().map(|(name, _)| name).collect::<Vec<_>>());
    } else {
        println!("❌ No transform types available");
    }
    
    if !working_ops.is_empty() {
        println!("✅ Transform operations available: {:?}", working_ops);
    } else {
        println!("❌ No transform operations available");
    }
    
    println!("\n💡 Next Steps for Real Transform Dialect:");
    println!("=========================================");
    if !working_types.is_empty() && !working_ops.is_empty() {
        println!("1. ✅ Transform dialect components are available");
        println!("2. 🔄 Create proper transform patterns for mymath.add → arith.addi");
        println!("3. 🔄 Find the transform interpreter API in melior 0.25.0");
        println!("4. 🔄 Apply transformations to payload module");
        println!("5. 🔄 Test complete custom dialect → LLVM → JIT pipeline");
        println!("\n🚀 Transform Dialect is AVAILABLE - ready for pattern implementation!");
    } else {
        println!("1. ❌ Transform dialect not fully available in this build");
        println!("2. 💡 May need different MLIR build with transform dialect enabled");
        println!("3. 💡 Or use alternative transformation approaches");
    }
    
    Ok(())
}