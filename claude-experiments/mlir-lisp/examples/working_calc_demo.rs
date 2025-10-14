/// Working Calc Dialect Demo
///
/// This demonstrates:
/// 1. Defining a calc dialect in Lisp
/// 2. Defining transform patterns in Lisp
/// 3. Writing a program using calc.* operations
/// 4. Compiling with those operations actually emitted
///
/// Run: cargo run --example working_calc_demo

use mlir_lisp::{
    self_contained::SelfContainedCompiler,
    parser,
    emitter::Emitter,
    function_registry::FunctionRegistry,
    expr_compiler::ExprCompiler,
    pattern_executor::PatternExecutor,
    mlir_context::MlirContext,
};
use melior::{
    Context,
    ir::{Module, Location, Block, Region, BlockLike, RegionLike, operation::{OperationBuilder, OperationLike}},
    dialect::DialectRegistry,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║            Working Calc Dialect Demo                        ║");
    println!("║  Dialect definitions → Transform patterns → Compilation!    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let registry = DialectRegistry::new();
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let mut compiler = SelfContainedCompiler::new(&context);

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 1: Define Calc Dialect");
    println!("═══════════════════════════════════════════════════════════════\n");

    let dialect_def = r#"
(defirdl-dialect calc
  :namespace "calc"
  :description "Simple calculator dialect"

  (defirdl-op constant
    :summary "Integer constant"
    :attributes [(value IntegerAttr)]
    :results [(result I32)]
    :traits [Pure NoMemoryEffect])

  (defirdl-op add
    :summary "Addition"
    :operands [(lhs I32) (rhs I32)]
    :results [(result I32)]
    :traits [Pure Commutative NoMemoryEffect])

  (defirdl-op mul
    :summary "Multiplication"
    :operands [(lhs I32) (rhs I32)]
    :results [(result I32)]
    :traits [Pure Commutative NoMemoryEffect]))
"#;

    println!("{}", dialect_def);

    match compiler.eval_string(dialect_def) {
        Ok(_) => println!("✅ Calc dialect registered!"),
        Err(e) => {
            println!("❌ Error: {}", e);
            return;
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("STEP 2: Define Transform Patterns");
    println!("═══════════════════════════════════════════════════════════════\n");

    let patterns = r#"
(defpdl-pattern lower-calc-constant
  :benefit 1
  :description "Lower calc.constant to arith.constant"
  :match (pdl.operation "calc.constant" :attributes {:value val})
  :rewrite (pdl.operation "arith.constant" :attributes {:value val}))

(defpdl-pattern lower-calc-add
  :benefit 1
  :description "Lower calc.add to arith.addi"
  :match (pdl.operation "calc.add" :operands [lhs rhs])
  :rewrite (pdl.operation "arith.addi" :operands [lhs rhs]))

(defpdl-pattern lower-calc-mul
  :benefit 1
  :description "Lower calc.mul to arith.muli"
  :match (pdl.operation "calc.mul" :operands [lhs rhs])
  :rewrite (pdl.operation "arith.muli" :operands [lhs rhs]))
"#;

    println!("{}", patterns);

    match compiler.eval_string(patterns) {
        Ok(_) => println!("✅ Patterns registered!"),
        Err(e) => {
            println!("❌ Error: {}", e);
            return;
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("STEP 3: Compile a Program Using calc.* Operations");
    println!("═══════════════════════════════════════════════════════════════\n");

    let program = "(calc.add (calc.mul (calc.constant 10) (calc.constant 20)) (calc.constant 30))";
    println!("Program: {}\n", program);
    println!("Expected: (10 * 20) + 30 = 230\n");

    match parser::parse(program) {
        Ok((_, values)) => {
            if values.is_empty() {
                println!("❌ No values parsed");
                return;
            }

            // Create module and compile
            let mlir_ctx = MlirContext::new();

            // Allow unregistered dialects (for calc.* operations)
            mlir_ctx.context().set_allow_unregistered_dialects(true);

            let module = Module::new(Location::unknown(mlir_ctx.context()));
            let mut emitter = Emitter::new(&mlir_ctx);
            let func_registry = FunctionRegistry::new();

            // Create a simple main function
            println!("Creating function to hold the expression...\n");

            // Build function manually
            let func_type = melior::ir::r#type::FunctionType::new(
                mlir_ctx.context(),
                &[],
                &[melior::ir::r#type::IntegerType::new(mlir_ctx.context(), 32).into()],
            );

            let region = Region::new();
            let entry_block = Block::new(&[]);
            region.append_block(entry_block);
            let block_ref = region.first_block().unwrap();

            // Compile the expression with dialect registry
            println!("Compiling with calc.* operations...\n");
            match ExprCompiler::compile_expr(
                &mut emitter,
                &block_ref,
                &values[0],
                &func_registry,
                Some(compiler.registry()),
            ) {
                Ok(result_name) => {
                    println!("✅ Compiled successfully!");
                    println!("   Result SSA name: {}\n", result_name);

                    // Emit return
                    if let Some(result_val) = emitter.get_value(&result_name) {
                        let return_op = OperationBuilder::new("func.return", Location::unknown(mlir_ctx.context()))
                            .add_operands(&[result_val])
                            .build()
                            .unwrap();
                        unsafe { block_ref.append_operation(return_op); }
                    }

                    // Create function operation
                    let func_op = OperationBuilder::new("func.func", Location::unknown(mlir_ctx.context()))
                        .add_attributes(&[
                            (
                                melior::ir::Identifier::new(mlir_ctx.context(), "sym_name"),
                                melior::ir::attribute::StringAttribute::new(mlir_ctx.context(), "compute").into(),
                            ),
                            (
                                melior::ir::Identifier::new(mlir_ctx.context(), "function_type"),
                                melior::ir::attribute::TypeAttribute::new(func_type.into()).into(),
                            ),
                        ])
                        .add_regions([region])
                        .build()
                        .unwrap();

                    module.body().append_operation(func_op);

                    println!("═══════════════════════════════════════════════════════════════");
                    println!("Generated MLIR (using calc.* operations!):");
                    println!("═══════════════════════════════════════════════════════════════\n");

                    // Print without verification first to see what we have
                    println!("{}", module.as_operation());

                    if !module.as_operation().verify() {
                        println!("\n⚠️  Note: Module did not pass verification (unregistered dialect)");
                    }

                    println!("\n═══════════════════════════════════════════════════════════════");
                    println!("STEP 4: Apply Transform Patterns");
                    println!("═══════════════════════════════════════════════════════════════\n");

                    let pattern_executor = PatternExecutor::new(mlir_ctx.context());
                    match pattern_executor.apply_patterns(&module, compiler.registry()) {
                        Ok(count) => {
                            println!("Pattern matching complete");
                            println!("(Full pattern execution would transform calc.* → arith.*)");
                        }
                        Err(e) => println!("❌ Pattern application error: {}", e),
                    }

                    println!("\n═══════════════════════════════════════════════════════════════");
                    println!("Summary");
                    println!("═══════════════════════════════════════════════════════════════\n");

                    println!("✅ WORKING:");
                    println!("   • Dialect defined in Lisp and registered");
                    println!("   • Patterns defined in Lisp and registered");
                    println!("   • Program compiled using calc.* operations");
                    println!("   • IR generated with custom dialect operations");
                    println!();

                    println!("🔄 NEXT STEPS:");
                    println!("   • Implement full pattern matcher");
                    println!("   • Apply patterns to transform IR");
                    println!("   • Lower to LLVM and execute");
                    println!();

                    println!("🎯 KEY ACHIEVEMENT:");
                    println!("   The dialect definitions are NOW USED, not just stored!");
                    println!("   Operations are emitted from the Lisp-defined dialect!");
                }
                Err(e) => {
                    println!("❌ Compilation error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Parse error: {:?}", e);
        }
    }
}
