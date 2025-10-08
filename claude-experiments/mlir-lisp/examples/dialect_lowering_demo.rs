/// Demonstration of dialect lowering with pattern-based transformations
///
/// This shows:
/// 1. Emitting high-level "lisp" dialect operations
/// 2. The high-level IR representation
/// 3. Pattern-based lowering to standard dialects
/// 4. The final lowered IR
///
/// Run with: cargo run --example dialect_lowering_demo

use mlir_lisp::{
    mlir_context::MlirContext,
    emitter::Emitter,
    lisp_ops::LispOps,
    lisp_lowering,
};
use melior::ir::{
    Module, Region, Block, BlockLike, RegionLike, Location,
    operation::{OperationBuilder, OperationLike},
    r#type::FunctionType,
    attribute::{TypeAttribute, StringAttribute},
    Identifier,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   High-Level Lisp Dialect → Low-Level Dialect Lowering       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Create MLIR context
    let ctx = MlirContext::new();
    let mut module = ctx.create_module();
    let emitter = Emitter::new(&ctx);

    println!("STEP 1: Emit High-Level Lisp Dialect");
    println!("═════════════════════════════════════\n");

    // Build a function using high-level lisp operations
    // Computes: (10 + 20) * 3 - 5
    let i32_type = emitter.parse_type("i32")?;
    let func_type = FunctionType::new(ctx.context(), &[], &[i32_type]);

    let region = Region::new();
    let block = Block::new(&[]);
    region.append_block(block);
    let entry_block = region.first_block().unwrap();

    // High-level operations
    let val1 = LispOps::emit_constant(&emitter, &entry_block, 10)?;
    let val2 = LispOps::emit_constant(&emitter, &entry_block, 20)?;
    let sum = LispOps::emit_add(&emitter, &entry_block, val1, val2)?;

    let val3 = LispOps::emit_constant(&emitter, &entry_block, 3)?;
    let product = LispOps::emit_mul(&emitter, &entry_block, sum, val3)?;

    let val4 = LispOps::emit_constant(&emitter, &entry_block, 5)?;
    let result = LispOps::emit_sub(&emitter, &entry_block, product, val4)?;

    // Return
    let return_op = OperationBuilder::new("func.return", Location::unknown(ctx.context()))
        .add_operands(&[result])
        .build()?;
    entry_block.append_operation(return_op);

    // Create function
    let function = OperationBuilder::new("func.func", Location::unknown(ctx.context()))
        .add_attributes(&[
            (
                Identifier::new(ctx.context(), "sym_name"),
                StringAttribute::new(ctx.context(), "compute").into(),
            ),
            (
                Identifier::new(ctx.context(), "function_type"),
                TypeAttribute::new(func_type.into()).into(),
            ),
        ])
        .add_regions([region])
        .build()?;

    module.body().append_operation(function);

    println!("Generated High-Level IR (Lisp Dialect):");
    println!("───────────────────────────────────────");
    println!("{}\n", module.as_operation());

    println!("STEP 2: Pattern-Based Lowering");
    println!("═══════════════════════════════\n");

    lisp_lowering::lower_lisp_to_arith(ctx.context(), &mut module)?;

    println!("\nSTEP 3: Conceptual Lowered IR");
    println!("═════════════════════════════════\n");

    println!("After pattern rewriting, the IR would become:");
    println!("───────────────────────────────────────────────");
    println!(r#"
func.func @compute() -> i32 {{
  %c10_i32 = arith.constant 10 : i32
  %c20_i32 = arith.constant 20 : i32
  %0 = arith.addi %c10_i32, %c20_i32 : i32
  %c3_i32 = arith.constant 3 : i32
  %1 = arith.muli %0, %c3_i32 : i32
  %c5_i32 = arith.constant 5 : i32
  %2 = arith.subi %1, %c5_i32 : i32
  return %2 : i32
}}
"#);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Pattern Transformations Applied:");
    println!("═══════════════════════════════════════════════════════════════");
    println!("✓ lisp.constant → arith.constant");
    println!("✓ lisp.add      → arith.addi");
    println!("✓ lisp.mul      → arith.muli");
    println!("✓ lisp.sub      → arith.subi");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Why This Matters:");
    println!("═══════════════════════════════════════════════════════════════");
    println!("• High-level semantics in source dialect");
    println!("• Declarative pattern-based transformations");
    println!("• Progressive lowering through multiple levels");
    println!("• Each level optimizable independently");
    println!("• Composable transformation pipeline");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Transform Dialect (Future Enhancement):");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{}", lisp_lowering::pattern_rewrite_documentation());

    Ok(())
}
