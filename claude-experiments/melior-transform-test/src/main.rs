use melior::{
    Context,
    dialect::{DialectRegistry, transform::{TransformOptions, apply_named_sequence}},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        Location, Module,
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, IntegerType},
        Block, Region, Identifier, RegionLike, BlockLike,
    },
    utility::register_all_dialects,
    pass::PassManager,
    ExecutionEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);
    
    let location = Location::unknown(&context);
    
    // Create payload module with mymath.add
    let payload_module = Module::new(location);
    let i32_type = IntegerType::new(&context, 32);
    let function_type = FunctionType::new(&context, &[], &[i32_type.into()]);
    
    let region = Region::new();
    let entry_block = Block::new(&[]);
    region.append_block(entry_block);
    
    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (Identifier::new(&context, "sym_name"), StringAttribute::new(&context, "main").into()),
            (Identifier::new(&context, "function_type"), TypeAttribute::new(function_type.into()).into()),
        ])
        .add_regions([region])
        .build()?;
    
    let function_region = function.region(0)?;
    let entry_block = function_region.first_block().unwrap();
    
    let const_10 = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(Identifier::new(&context, "value"), IntegerAttribute::new(i32_type.into(), 10).into())])
        .add_results(&[i32_type.into()])
        .build()?;
    
    let const_32 = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(Identifier::new(&context, "value"), IntegerAttribute::new(i32_type.into(), 32).into())])
        .add_results(&[i32_type.into()])
        .build()?;
    
    let custom_add = OperationBuilder::new("mymath.add", location)
        .add_operands(&[const_10.result(0)?.into(), const_32.result(0)?.into()])
        .add_results(&[i32_type.into()])
        .build()?;
    
    let return_op = OperationBuilder::new("func.return", location)
        .add_operands(&[custom_add.result(0)?.into()])
        .build()?;
    
    entry_block.append_operation(const_10);
    entry_block.append_operation(const_32);
    entry_block.append_operation(custom_add);
    entry_block.append_operation(return_op);
    payload_module.body().append_operation(function);
    
    println!("BEFORE:");
    println!("{}", payload_module.as_operation());
    
    // Transform module using PDL to replace mymath.add with arith.addi
    let transform_ir = r#"
    module {
      transform.with_pdl_patterns {
      ^bb0(%root: !transform.any_op):
        pdl.pattern @mymath_to_arith : benefit(1) {
          %lhs = pdl.operand
          %rhs = pdl.operand
          %result_type = pdl.type
          %mymath_op = pdl.operation "mymath.add"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
          pdl.rewrite %mymath_op {
            %arith_op = pdl.operation "arith.addi"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
            pdl.replace %mymath_op with %arith_op
          }
        }
        
        transform.sequence %root : !transform.any_op failures(propagate) {
        ^bb1(%arg1: !transform.any_op):
          %matched = pdl_match @mymath_to_arith in %arg1 : (!transform.any_op) -> !transform.any_op
          transform.yield
        }
      }
    }
    "#;
    
    let transform_module = Module::parse(&context, transform_ir).unwrap();
    let transform_options = TransformOptions::new();
    let transform_op = transform_module.body().first_operation().unwrap();
    
    // Apply transform
    apply_named_sequence(
        &payload_module.as_operation(),
        &transform_op,
        &transform_module.as_operation(),
        &transform_options,
    )?;
    
    println!("\nAFTER:");
    println!("{}", payload_module.as_operation());
    
    // Lower to LLVM and JIT compile
    let mut module = payload_module;
    let pm = PassManager::new(&context);
    pm.add_pass(melior::pass::conversion::create_arith_to_llvm());
    pm.add_pass(melior::pass::conversion::create_func_to_llvm());
    
    pm.run(&mut module)?;
    
    let engine = ExecutionEngine::new(&module, 2, &[], false);
    let func_ptr = engine.lookup("main");
    
    if !func_ptr.is_null() {
        type MainFn = unsafe extern "C" fn() -> i32;
        let main_fn: MainFn = unsafe { std::mem::transmute(func_ptr) };
        let result = unsafe { main_fn() };
        println!("\nRESULT: {}", result);
    }
    
    Ok(())
}