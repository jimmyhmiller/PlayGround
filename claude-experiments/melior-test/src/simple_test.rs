use melior::{
    Context, ExecutionEngine,
    ir::{Module, Location, Block, Region, operation::OperationBuilder, r#type::FunctionType, attribute::{StringAttribute, TypeAttribute}, Identifier},
    dialect::DialectRegistry,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Simple MLIR Test - JIT Execution");
    println!("==================================");
    
    // Create context and load dialects
    let registry = DialectRegistry::new();
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    let location = Location::unknown(&context);
    let module = Module::new(location);
    
    // Create a simple function that just returns
    println!("\n📝 Creating simple function...");
    create_simple_llvm_function(&context, &module)?;
    
    println!("Generated MLIR:");
    println!("{}", module.as_operation());
    
    // Try to create ExecutionEngine
    println!("\n⚡ Testing ExecutionEngine...");
    
    let result = std::panic::catch_unwind(|| {
        println!("Creating ExecutionEngine...");
        ExecutionEngine::new(&module, 2, &[], false)
    });
    
    match result {
        Ok(engine) => {
            println!("✅ ExecutionEngine created successfully!");
            
            // Try to lookup the function
            let func_ptr = engine.lookup("simple");
            if func_ptr.is_null() {
                println!("❌ Function lookup failed: returned null pointer");
            } else {
                println!("✅ Function lookup successful!");
                unsafe {
                    let func: extern "C" fn() = std::mem::transmute(func_ptr);
                    println!("Calling function...");
                    func();
                    println!("✅ Function executed successfully!");
                }
            }
        }
        Err(_) => {
            println!("❌ ExecutionEngine creation failed");
            return Err("ExecutionEngine creation failed".into());
        }
    }
    
    println!("\n🎉 Test completed!");
    Ok(())
}

fn create_simple_llvm_function(context: &Context, module: &Module) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);
    
    // Create function type: () -> ()
    let function_type = FunctionType::new(context, &[], &[]).into();
    
    // Create llvm.func directly (skip func dialect entirely)
    // Use the same approach as our working manual conversion
    let function = OperationBuilder::new("llvm.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), StringAttribute::new(context, "simple").into()),
            (Identifier::new(context, "function_type"), TypeAttribute::new(function_type).into()),
            (Identifier::new(context, "linkage"), StringAttribute::new(context, "external").into()),
        ])
        .add_regions([Region::new()])
        .build()?;
    
    // Add function to module
    module.body().append_operation(function.clone());
    
    // Create function body
    let block = Block::new(&[]);
    let region = function.region(0)?;
    region.append_block(block);
    
    // Add llvm.return
    let return_op = OperationBuilder::new("llvm.return", location)
        .build()?;
    region.first_block().unwrap().append_operation(return_op);
    
    Ok(())
}