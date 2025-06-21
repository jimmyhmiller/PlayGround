use melior::{
    dialect::{arith, func, DialectRegistry},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, BlockLike, Location, Module, Region, RegionLike, Value,
    },
    pass::{self, PassManager},
    utility::register_all_dialects,
    Context as MeliorContext, ExecutionEngine,
};

#[derive(Debug, Clone)]
pub enum Expr {
    Number(i64),
    Add(Box<Expr>, Box<Expr>),
}

fn parse_simple() -> Expr {
    // Simple hardcoded 2+2 expression
    Expr::Add(Box::new(Expr::Number(2)), Box::new(Expr::Number(2)))
}

fn compile_to_mlir(context: &MeliorContext) -> Result<Module, Box<dyn std::error::Error>> {
    let location = Location::unknown(context);

    // Create MLIR module
    let module = Module::new(location);

    // Create i64 type
    let i64_type = IntegerType::new(context, 64).into();

    // Create main function type: () -> i64
    let function_type = FunctionType::new(context, &[], &[i64_type]).into();

    // Create main function
    let main_func = func::func(
        context,
        StringAttribute::new(context, "main"),
        TypeAttribute::new(function_type),
        Region::new(),
        &[],
        location,
    );

    // Create function body
    let block = Block::new(&[]);
    let region = main_func.region(0).unwrap();
    region.append_block(block);

    // Get the first block from the region
    let entry_block = region.first_block().unwrap();

    // Compile expression: 2 + 2
    let expr = parse_simple();
    let result = compile_expr(context, &entry_block, &expr, location)?;

    // Return the result
    let return_op = func::r#return(&[result], location);
    entry_block.append_operation(return_op);

    // Add function to module
    module.body().append_operation(main_func);

    Ok(module)
}

fn compile_expr<'c>(
    context: &'c MeliorContext,
    block: &'c Block<'c>,
    expr: &Expr,
    location: Location<'c>,
) -> Result<Value<'c, 'c>, Box<dyn std::error::Error>> {
    match expr {
        Expr::Number(n) => {
            let i64_type = IntegerType::new(context, 64).into();
            let value_attr = IntegerAttribute::new(i64_type, *n).into();
            let result = block
                .append_operation(arith::constant(context, value_attr, location))
                .result(0)?
                .into();
            Ok(result)
        }
        Expr::Add(left, right) => {
            let left_val = compile_expr(context, block, left, location)?;
            let right_val = compile_expr(context, block, right, location)?;
            let result = block
                .append_operation(arith::addi(left_val, right_val, location))
                .result(0)?
                .into();
            Ok(result)
        }
    }
}

fn setup_mlir_context() -> MeliorContext {
    let context = MeliorContext::new();

    // Register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    context
}

fn run_passes(
    context: &MeliorContext,
    module: &mut Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let pass_manager = PassManager::new(context);

    // Add conversion passes to lower to LLVM dialect
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());

    // Run passes
    pass_manager.run(module)?;

    Ok(())
}

fn jit_compile_and_run(
    _context: &MeliorContext,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create execution engine for JIT compilation
    let engine = ExecutionEngine::new(module, 2, &[], false);

    // Get function pointer to main
    let main_fn = engine.lookup("main");

    if main_fn.is_null() {
        return Err("Failed to find main function".into());
    }

    // Cast to function type and call
    let main: extern "C" fn() -> i64 = unsafe { std::mem::transmute(main_fn) };
    let result = main();

    println!("Result: {}", result);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize MLIR
    let context = setup_mlir_context();

    // Compile expression to MLIR
    let mut module = compile_to_mlir(&context)?;

    println!("Generated MLIR:");
    println!("{}", module.as_operation());

    // Lower to LLVM dialect
    run_passes(&context, &mut module)?;

    println!("\nAfter lowering to LLVM dialect:");
    println!("{}", module.as_operation());

    // JIT compile and run
    println!("\nJIT compiling and running:");
    jit_compile_and_run(&context, &module)?;

    Ok(())
}
