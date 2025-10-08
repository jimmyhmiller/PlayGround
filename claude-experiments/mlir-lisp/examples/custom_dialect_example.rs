/// Example demonstrating custom "lisp" dialect operations
///
/// This shows:
/// 1. How to emit custom operations (lisp.constant, lisp.add, etc.)
/// 2. How the custom IR looks
/// 3. How to write a lowering pass to convert to standard MLIR
///
/// Run with: cargo run --example custom_dialect_example

use mlir_lisp::{
    mlir_context::MlirContext,
    emitter::Emitter,
    lisp_ops::LispOps,
};
use melior::ir::{
    Module, Region, Block, BlockLike, RegionLike, Location,
    operation::OperationBuilder,
    r#type::FunctionType,
    attribute::{TypeAttribute, StringAttribute},
    Identifier,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create MLIR context
    let ctx = MlirContext::new();
    let module = ctx.create_module();
    let emitter = Emitter::new(&ctx);

    // Create a function that uses our custom dialect
    println!("=== Building function with custom 'lisp' operations ===\n");

    let i32_type = emitter.parse_type("i32")?;
    let func_type = FunctionType::new(ctx.context(), &[], &[i32_type]);

    let region = Region::new();
    let block = Block::new(&[]);
    region.append_block(block);
    let entry_block = region.first_block().unwrap();

    // Emit: lisp.constant 40
    let val1 = LispOps::emit_constant(&emitter, &entry_block, 40)?;

    // Emit: lisp.constant 2
    let val2 = LispOps::emit_constant(&emitter, &entry_block, 2)?;

    // Emit: lisp.add %val1, %val2
    let result = LispOps::emit_add(&emitter, &entry_block, val1, val2)?;

    // Emit: return
    let return_op = OperationBuilder::new("func.return", Location::unknown(ctx.context()))
        .add_operands(&[result])
        .build()?;
    unsafe { entry_block.append_operation(return_op); }

    // Create the function
    let function = OperationBuilder::new("func.func", Location::unknown(ctx.context()))
        .add_attributes(&[
            (
                Identifier::new(ctx.context(), "sym_name"),
                StringAttribute::new(ctx.context(), "custom_add").into(),
            ),
            (
                Identifier::new(ctx.context(), "function_type"),
                TypeAttribute::new(func_type.into()).into(),
            ),
        ])
        .add_regions([region])
        .build()?;

    module.body().append_operation(function);

    println!("Generated MLIR with custom operations:");
    println!("{}\n", module.as_operation());

    println!("=== Custom Dialect Operations ===");
    println!("✓ lisp.constant - our own constant operation");
    println!("✓ lisp.add - our own addition operation");
    println!("\nThese operations:");
    println!("- Are in our custom 'lisp' namespace");
    println!("- Can have custom semantics");
    println!("- Can be transformed/lowered to standard MLIR");
    println!("- Can be optimized with pattern matching");

    Ok(())
}
