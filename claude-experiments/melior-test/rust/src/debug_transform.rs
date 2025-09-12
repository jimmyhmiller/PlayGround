use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        Location, Module,
        r#type::Type,
        operation::OperationBuilder,
    },
    utility::register_all_dialects,
};

use std::sync::Once;

static INIT: Once = Once::new();

fn init_mlir_once() {
    INIT.call_once(|| {
        unsafe {
            let registry = DialectRegistry::new();
            mlir_sys::mlirRegisterAllDialects(registry.to_raw());
            mlir_sys::mlirRegisterAllPasses();
        }
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debugging Transform Dialect Issues");
    println!("====================================");
    
    init_mlir_once();
    
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    
    unsafe { 
        mlir_sys::mlirRegisterAllLLVMTranslations(context.to_raw());
        mlir_sys::mlirContextSetAllowUnregisteredDialects(context.to_raw(), true);
    }
    
    let location = Location::unknown(&context);
    
    println!("\n📋 Step 1: Testing Transform Dialect Type Creation");
    println!("-------------------------------------------------");
    
    // Test 1: Can we create transform types using the C API directly?
    println!("🔍 Testing mlirTransformAnyOpTypeGet...");
    let transform_any_op_type = unsafe {
        mlir_sys::mlirTransformAnyOpTypeGet(context.to_raw())
    };
    
    // Check if the type is valid (there's no mlirTypeIsNull, so we'll check differently)
    let null_type = unsafe { mlir_sys::MlirType { ptr: std::ptr::null_mut() } };
    if unsafe { mlir_sys::mlirTypeEqual(transform_any_op_type, null_type) } {
        println!("❌ Failed to create !transform.any_op type via C API");
    } else {
        println!("✅ Successfully created !transform.any_op type via C API");
    }
    
    // Test 2: Can we parse transform types?
    println!("\n🔍 Testing Type::parse for !transform.any_op...");
    match Type::parse(&context, "!transform.any_op") {
        Some(parsed_type) => {
            println!("✅ Successfully parsed !transform.any_op type");
            println!("   Type: {}", parsed_type);
        },
        None => {
            println!("❌ Failed to parse !transform.any_op type");
            println!("   This suggests transform dialect may not be properly loaded");
        }
    }
    
    // Test 3: Can we parse other transform types?
    let test_types = [
        "!transform.any_value",
        "!transform.any_param", 
        "!transform.param<i32>",
        "!transform.op<\"func.func\">",
    ];
    
    println!("\n🔍 Testing other transform types...");
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
    
    println!("\n📋 Step 2: Testing Simple Transform Operations");
    println!("---------------------------------------------");
    
    // Let's try creating a very simple transform operation that's more likely to work
    let module = Module::new(location);
    
    // Test creating a transform.sequence operation (simpler than named_sequence)
    println!("🔍 Testing basic transform.sequence creation...");
    
    // Instead of using named_sequence, let's try a basic sequence
    if let Some(any_op_type) = Type::parse(&context, "!transform.any_op") {
        use melior::ir::operation::OperationBuilder;
        
        println!("🔍 Attempting to create transform.sequence...");
        match OperationBuilder::new("transform.sequence", location)
            .add_results(&[any_op_type])
            .build() {
                Ok(seq_op) => {
                    println!("✅ Successfully created transform.sequence operation");
                    println!("   Operation: {}", seq_op);
                },
                Err(e) => {
                    println!("❌ Failed to create transform.sequence: {:?}", e);
                }
            }
    } else {
        println!("❌ Cannot create transform operations - type parsing failed");
    }
    
    println!("\n📋 Step 3: Checking Available Operations");
    println!("---------------------------------------");
    
    // Let's check what operations are actually available by trying to create
    // some basic ones that should exist
    let test_ops = [
        "transform.sequence",
        "transform.with_pdl_patterns", 
        "transform.apply_patterns",
        "transform.yield",
    ];
    
    for op_name in &test_ops {
        println!("🔍 Testing operation: {}", op_name);
        match OperationBuilder::new(op_name, location).build() {
            Ok(_) => println!("✅ {} is available", op_name),
            Err(e) => println!("❌ {} failed: {:?}", op_name, e),
        }
    }
    
    println!("\n📋 Step 4: Transform Dialect Registration Check");
    println!("----------------------------------------------");
    
    // Check if we can load a module with transform operations
    println!("🔍 Testing module with simple transform IR...");
    
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
        },
        None => {
            println!("❌ Failed to parse transform IR");
            println!("   This indicates transform dialect operations aren't available");
        }
    }
    
    println!("\n🎓 Debug Summary");
    println!("================");
    println!("The segfault is likely caused by one of these issues:");
    println!("1. Transform dialect operations not properly registered");
    println!("2. Incorrect operation creation (missing required attributes/regions)");
    println!("3. Type parsing failures leading to invalid operation construction");
    println!("4. MLIR build doesn't include full transform dialect support");
    
    println!("\n💡 Next Steps:");
    println!("==============");
    println!("- Check which transform operations are actually available");
    println!("- Use simpler transform operations that are guaranteed to exist");
    println!("- Focus on the transform interpreter pass rather than manual construction");
    
    Ok(())
}