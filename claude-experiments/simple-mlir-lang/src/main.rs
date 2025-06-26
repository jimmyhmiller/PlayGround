use melior::{
    dialect::{arith, func, DialectRegistry},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute, DenseElementsAttribute, Attribute},
        r#type::{FunctionType, IntegerType},
        operation::OperationBuilder,
        Block, BlockLike, Location, Module, Region, RegionLike, Value, Type,
    },
    pass::{self, PassManager},
    utility::register_all_dialects,
    Context as MeliorContext, ExecutionEngine,
};

#[derive(Debug, Clone)]
pub enum Expr {
    Number(i64),
    Add(Box<Expr>, Box<Expr>),
    VectorAdd(Vec<i64>, Vec<i64>),
}

fn parse_simple() -> Vec<Expr> {
    vec![
        // Scalar addition: 2+2
        Expr::Add(Box::new(Expr::Number(2)), Box::new(Expr::Number(2))),
        // Vector addition: [1,2,3,4] + [5,6,7,8]
        Expr::VectorAdd(vec![1, 2, 3, 4], vec![5, 6, 7, 8]),
    ]
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

    // Compile expressions
    let exprs = parse_simple();
    let mut last_result = None;
    
    for expr in &exprs {
        let result = compile_expr(context, &entry_block, expr, location)?;
        last_result = Some(result);
        
        // Print each result for demonstration
        match expr {
            Expr::Add(_, _) => {
                // For scalar, just keep the result
            }
            Expr::VectorAdd(_, _) => {
                // For vector, extract the first element to return as scalar using raw operation
                let zero = entry_block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                    
                // Create vector.extractelement operation manually
                let extract_op = OperationBuilder::new("vector.extractelement", location)
                    .add_operands(&[result, zero])
                    .add_results(&[IntegerType::new(context, 64).into()])
                    .build()?;
                    
                last_result = Some(
                    entry_block
                        .append_operation(extract_op)
                        .result(0)?
                        .into()
                );
            }
            _ => {}
        }
    }

    // Return the last result (or 0 if no expressions)
    let return_value = last_result.unwrap_or_else(|| {
        entry_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(i64_type, 0).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into()
    });

    let return_op = func::r#return(&[return_value], location);
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
        Expr::VectorAdd(left_vec, right_vec) => {
            if left_vec.len() != right_vec.len() {
                return Err("Vector length mismatch".into());
            }
            
            let vector_len = left_vec.len();
            let i64_type = IntegerType::new(context, 64).into();
            
            // Create vector type using raw MLIR type parsing
            let vector_type_str = format!("vector<{}xi64>", vector_len);
            let vector_type = Type::parse(context, &vector_type_str)
                .ok_or("Failed to parse vector type")?;
            
            // Convert integer attributes to generic attributes
            let left_elements: Vec<Attribute> = left_vec.iter()
                .map(|&n| IntegerAttribute::new(i64_type, n).into())
                .collect();
            let right_elements: Vec<Attribute> = right_vec.iter()
                .map(|&n| IntegerAttribute::new(i64_type, n).into())
                .collect();
            
            let left_attr = DenseElementsAttribute::new(vector_type, &left_elements)?.into();
            let right_attr = DenseElementsAttribute::new(vector_type, &right_elements)?.into();
            
            // Create vector constants
            let left_vector = block
                .append_operation(arith::constant(context, left_attr, location))
                .result(0)?
                .into();
                
            let right_vector = block
                .append_operation(arith::constant(context, right_attr, location))
                .result(0)?
                .into();
            
            // Perform vector addition
            let result = block
                .append_operation(arith::addi(left_vector, right_vector, location))
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
    pass_manager.add_pass(pass::conversion::create_vector_to_llvm());
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
