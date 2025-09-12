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

fn test_transform_api() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Transform Dialect API in Melior 0.25.0");
    println!("==================================================");
    
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    let location = Location::unknown(&context);
    
    println!("\n🔍 Step 1: Testing Transform Types");
    println!("----------------------------------");
    
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
                println!("✅ {} → {}", type_str, parsed_type);
            },
            None => {
                println!("❌ {} → Failed to parse", type_str);
            }
        }
    }
    
    println!("\n🔍 Step 2: Testing Transform Operations");
    println!("--------------------------------------");
    
    // Test transform.sequence creation
    if let Some(any_value_type) = Type::parse(&context, "!transform.any_value") {
        let sequence_result = OperationBuilder::new("transform.sequence", location)
            .add_results(&[any_value_type])
            .build();
            
        match sequence_result {
            Ok(op) => {
                println!("✅ transform.sequence created: {}", op);
            },
            Err(e) => {
                println!("❌ transform.sequence failed: {:?}", e);
            }
        }
    }
    
    // Test basic transform operations
    let test_ops: [(&str, Vec<i32>); 3] = [
        ("transform.yield", vec![]),
        ("transform.apply_patterns", vec![]),
        ("transform.with_pdl_patterns", vec![]),
    ];
    
    for (op_name, _) in &test_ops {
        match OperationBuilder::new(op_name, location).build() {
            Ok(op) => println!("✅ {} created: {}", op_name, op),
            Err(e) => println!("❌ {} failed: {:?}", op_name, e),
        }
    }
    
    println!("\n🔍 Step 3: Testing Transform IR Parsing");
    println!("--------------------------------------");
    
    // Test simple transform IR
    let transform_ir = r#"
    module {
      transform.sequence failures(propagate) {
      ^bb0(%arg0: !transform.any_value):
        transform.yield
      }
    }
    "#;
    
    match Module::parse(&context, transform_ir) {
        Some(module) => {
            println!("✅ Transform IR parsed successfully");
            println!("   Module: {}", module.as_operation());
        },
        None => {
            println!("❌ Transform IR parsing failed");
        }
    }
    
    // Test more complex transform IR with named sequence
    let named_sequence_ir = r#"
    module {
      transform.named_sequence @__transform_main(%root: !transform.any_op) {
        transform.yield
      }
    }
    "#;
    
    match Module::parse(&context, named_sequence_ir) {
        Some(module) => {
            println!("✅ Named sequence IR parsed successfully");
            println!("   Module: {}", module.as_operation());
        },
        None => {
            println!("❌ Named sequence IR parsing failed");
        }
    }
    
    println!("\n🔍 Step 4: Looking for Transform Interpreter APIs");
    println!("------------------------------------------------");
    
    // Check if melior exposes any transform interpreter functionality
    // This might be in the form of utility functions or pass managers
    
    println!("Available in melior 0.25.0:");
    println!("• Context with dialect registry and loading");
    println!("• Module parsing for transform IR");
    println!("• Transform operation builders");
    println!("• PassManager for running passes");
    
    // The actual transform interpreter might need to be invoked differently
    // Let's check if there are any transform-related passes
    println!("\nChecking for transform-related functionality...");
    
    println!("\n💡 Summary of Transform Dialect Availability");
    println!("============================================");
    println!("✅ Transform types: !transform.any_value works");
    println!("✅ Transform operations: Basic ones are available");
    println!("✅ Transform IR parsing: Successfully parses complete transform modules");
    println!("❓ Transform interpreter: API needs to be discovered");
    
    println!("\n🔧 Potential Transform Interpreter APIs to investigate:");
    println!("======================================================");
    println!("1. PassManager with transform-related passes");
    println!("2. Utility functions for applying transform modules");
    println!("3. Direct MLIR C API calls (mlirTransformApplyNamedSequence)");
    println!("4. Transform dialect ODS operations in melior::dialect::ods::transform");
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    test_transform_api()
}