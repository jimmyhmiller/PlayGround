use melior::{
    Context,
    ir::{Module, Location},
    dialect::DialectRegistry,
};
use mlir_sys::*;

mod tensor_ops_dialect;
mod tensor_ops_lowering;

use tensor_ops_dialect::TensorOpsDialect;
use tensor_ops_lowering::TensorOpsPassManager;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Custom MLIR Dialect Demo: TensorOps");
    println!("=====================================");
    
    // Setup MLIR context
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
    
    // Demonstrate our custom dialect
    println!("\n📋 Step 1: Creating operations with TensorOps dialect");
    tensor_ops_dialect::create_example_tensor_computation(&context, &module)?;
    
    println!("🔍 TensorOps dialect MLIR:");
    println!("{}", module.as_operation());
    
    // Demonstrate interop with standard dialects  
    println!("\n📋 Step 2: Demonstrating dialect interoperability");
    tensor_ops_lowering::create_interop_example(&context, &module)?;
    
    println!("🔍 Mixed dialect MLIR (TensorOps + Standard):");
    println!("{}", module.as_operation());
    
    // Demonstrate lowering pipeline
    println!("\n📋 Step 3: Applying lowering transformations");
    TensorOpsPassManager::apply_full_lowering_pipeline(&context, &mut module)?;
    
    println!("🔍 After lowering pipeline:");
    println!("{}", module.as_operation());
    
    // Summary
    println!("\n✅ Custom Dialect Implementation Summary");
    println!("========================================");
    println!("🎯 Successfully implemented:");
    println!("   • Custom TensorOps dialect with operations:");
    println!("     - tensor_ops.add");
    println!("     - tensor_ops.mul"); 
    println!("     - tensor_ops.constant");
    println!("     - tensor_ops.reshape");
    println!("   • Lowering patterns to standard dialects:");
    println!("     - tensor_ops.add    → arith.addf");
    println!("     - tensor_ops.mul    → arith.mulf");
    println!("     - tensor_ops.constant → arith.constant");
    println!("     - tensor_ops.reshape → tensor.reshape");
    println!("   • Multi-stage lowering pipeline:");
    println!("     - TensorOps → Standard dialects → LLVM");
    println!("   • Dialect interoperability:");
    println!("     - Mixed TensorOps and standard operations");
    
    println!("\n🚀 This demonstrates the core MLIR concepts:");
    println!("   • Custom dialect definition");
    println!("   • Operation creation and manipulation"); 
    println!("   • Lowering and transformation passes");
    println!("   • Dialect interoperability");
    println!("   • Progressive lowering to target dialects");
    
    Ok(())
}