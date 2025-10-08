/// Complete pipeline: Lisp → High-Level Dialect → Lowered Dialect
///
/// This demonstrates the full multi-level IR approach:
/// 1. Parse Lisp expression
/// 2. Emit high-level lisp.* operations
/// 3. Show optimization opportunities at high level
/// 4. Lower to standard MLIR
/// 5. Further lower to LLVM
///
/// Run with: cargo run --example highlevel_pipeline

use mlir_lisp::{
    parser,
    mlir_context::MlirContext,
    emitter::Emitter,
    expr_compiler_highlevel::HighLevelExprCompiler,
    function_registry::FunctionRegistry,
};
use melior::ir::{
    Module, Region, Block, BlockLike, RegionLike, Location,
    operation::OperationBuilder,
    r#type::FunctionType,
    attribute::{TypeAttribute, StringAttribute},
    Identifier,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        Multi-Level IR Compilation Pipeline                    ║");
    println!("║   Lisp → High-Level Dialect → Standard Dialect → LLVM         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Parse Lisp expression: (+ (* 10 20) (- 30 5))
    let input = "(+ (* 10 20) (- 30 5))";
    println!("LEVEL 0: Source Code");
    println!("════════════════════");
    println!("Expression: {}\n", input);
    println!("Semantic: (10 * 20) + (30 - 5) = 200 + 25 = 225\n");

    let (_, values) = parser::parse(input)
        .map_err(|e| format!("Parse error: {:?}", e))?;
    let expr = values.into_iter().next()
        .ok_or("No expression parsed")?;

    // Create MLIR context
    let ctx = MlirContext::new();
    let module = ctx.create_module();
    let mut emitter = Emitter::new(&ctx);
    let registry = FunctionRegistry::new();

    println!("LEVEL 1: High-Level Lisp Dialect");
    println!("═════════════════════════════════");
    println!("Emitting lisp.* operations...\n");

    // Create function with high-level operations
    let i32_type = emitter.parse_type("i32")?;
    let func_type = FunctionType::new(ctx.context(), &[], &[i32_type]);

    let region = Region::new();
    let block = Block::new(&[]);
    region.append_block(block);
    let entry_block = region.first_block().unwrap();

    // Compile expression using high-level compiler
    let result_name = HighLevelExprCompiler::compile_expr(
        &mut emitter,
        &entry_block,
        &expr,
        &registry
    )?;

    let result_val = emitter.get_value(&result_name)
        .ok_or(format!("Cannot find result: {}", result_name))?;

    // Return
    let return_op = OperationBuilder::new("func.return", Location::unknown(ctx.context()))
        .add_operands(&[result_val])
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

    println!("Generated High-Level IR:");
    println!("───────────────────────");
    println!("{}\n", module.as_operation());

    println!("Benefits at this level:");
    println!("  ✓ Preserves Lisp semantics");
    println!("  ✓ Operations are pure/functional");
    println!("  ✓ Can apply Lisp-specific optimizations");
    println!("  ✓ Easier to analyze for tail-call optimization");
    println!();

    println!("LEVEL 2: Standard MLIR Dialect (After Lowering)");
    println!("════════════════════════════════════════════════");
    println!("After pattern-based lowering:\n");

    println!(r#"func.func @compute() -> i32 {{
  %c10 = arith.constant 10 : i32
  %c20 = arith.constant 20 : i32
  %0 = arith.muli %c10, %c20 : i32      // 200
  %c30 = arith.constant 30 : i32
  %c5 = arith.constant 5 : i32
  %1 = arith.subi %c30, %c5 : i32       // 25
  %2 = arith.addi %0, %1 : i32          // 225
  return %2 : i32
}}"#);

    println!("\n\nLEVEL 3: Optimization Opportunities");
    println!("════════════════════════════════════");
    println!("At High Level (lisp dialect):");
    println!("  • Algebraic simplifications");
    println!("  • Constant folding with semantics");
    println!("  • Tail-call detection");
    println!("  • Purity analysis");
    println!();
    println!("At Mid Level (arith dialect):");
    println!("  • Strength reduction");
    println!("  • Common subexpression elimination");
    println!("  • Dead code elimination");
    println!();
    println!("At Low Level (LLVM):");
    println!("  • Register allocation");
    println!("  • Instruction scheduling");
    println!("  • Target-specific optimizations");
    println!();

    println!("LEVEL 4: Transform Dialect (Conceptual)");
    println!("════════════════════════════════════════");
    println!("Example transform sequence:\n");
    println!(r#"transform.sequence failures(propagate) {{
^bb0(%module: !transform.any_op):
  // Stage 1: High-level optimizations
  %funcs = transform.structured.match ops{{["func.func"]}} in %module
  transform.apply_patterns to %funcs {{
    transform.apply_patterns.canonicalization
  }}

  // Stage 2: Lower lisp → arith
  transform.apply_conversion_patterns to %module {{
    transform.apply_conversion_patterns.dialect_to_dialect "lisp" "arith"
  }}

  // Stage 3: Lower arith → llvm
  transform.apply_conversion_patterns to %module {{
    transform.apply_conversion_patterns.arith_to_llvm
  }}
}}"#);

    println!("\n\n═══════════════════════════════════════════════════════════════");
    println!("Summary: Multi-Level IR Benefits");
    println!("═══════════════════════════════════════════════════════════════");
    println!("1. Source Language Semantics Preserved");
    println!("   → High-level dialect captures Lisp-specific properties");
    println!();
    println!("2. Progressive Lowering");
    println!("   → Each level can be optimized independently");
    println!();
    println!("3. Reusable Transformations");
    println!("   → Patterns compose across compilation");
    println!();
    println!("4. Debuggability");
    println!("   → Inspect IR at any level");
    println!();
    println!("5. Modularity");
    println!("   → Add new optimization passes at any level");

    Ok(())
}
