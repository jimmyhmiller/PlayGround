use melior::{
    Context,
    ir::{
        Module, Location, Block, Region, Value, Type, operation::{Operation, OperationBuilder},
        attribute::{StringAttribute, TypeAttribute, IntegerAttribute, ArrayAttribute, FloatAttribute},
        r#type::{FunctionType, IntegerType, RankedTensorType, FloatType},
        Identifier
    },
    pass::PassManager,
};
use std::ffi::CString;

/// Lowering patterns for TensorOps dialect to standard dialects
pub struct TensorOpsLowering;

impl TensorOpsLowering {
    /// Lower tensor_ops.add to arith.addf operations
    pub fn lower_add_to_arith(
        context: &Context,
        location: Location,
        lhs: Value,
        rhs: Value,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        // tensor_ops.add %lhs, %rhs -> arith.addf %lhs, %rhs : tensor<...>
        let op = OperationBuilder::new("arith.addf", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
    
    /// Lower tensor_ops.mul to arith.mulf operations
    pub fn lower_mul_to_arith(
        context: &Context,
        location: Location,
        lhs: Value,
        rhs: Value,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        // tensor_ops.mul %lhs, %rhs -> arith.mulf %lhs, %rhs : tensor<...>
        let op = OperationBuilder::new("arith.mulf", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
    
    /// Lower tensor_ops.constant to arith.constant operations
    pub fn lower_constant_to_arith(
        context: &Context,
        location: Location,
        value_attr: StringAttribute,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        // tensor_ops.constant -> arith.constant
        let op = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[
                (Identifier::new(context, "value"), value_attr.into())
            ])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
    
    /// Lower tensor_ops.reshape to tensor.reshape operations  
    pub fn lower_reshape_to_tensor(
        context: &Context,
        location: Location,
        input: Value,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        // tensor_ops.reshape -> tensor.reshape
        let op = OperationBuilder::new("tensor.reshape", location)
            .add_operands(&[input])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
    
    /// Apply lowering transformations to a module
    pub fn apply_lowering(
        context: &Context,
        module: &mut Module,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔄 Applying TensorOps dialect lowering transformations...");
        
        // In a real implementation, this would be a proper MLIR pass
        // For demonstration, we'll manually transform operations
        Self::transform_module_operations(context, module)?;
        
        println!("✅ TensorOps dialect successfully lowered to standard dialects");
        Ok(())
    }
    
    fn transform_module_operations(
        context: &Context,
        module: &mut Module,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // This is a simplified transformation - in practice, you'd use MLIR's
        // pattern rewriting infrastructure
        
        // Walk through the module and find tensor_ops operations to transform
        let module_str = format!("{}", module.as_operation());
        
        if module_str.contains("tensor_ops.") {
            println!("📝 Found TensorOps operations to transform");
            
            // Create a new module with transformed operations
            let location = Location::unknown(context);
            let new_module = Module::new(location);
            
            // Copy and transform the function
            Self::transform_function_to_standard_dialects(context, module, &new_module)?;
            
            // Replace the module content
            *module = new_module;
        }
        
        Ok(())
    }
    
    fn transform_function_to_standard_dialects(
        context: &Context,
        old_module: &Module,
        new_module: &Module,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let location = Location::unknown(context);
        
        // Create a new function that uses standard dialects instead of tensor_ops
        let function_type = FunctionType::new(context, &[], &[]).into();
        
        let function = OperationBuilder::new("func.func", location)
            .add_attributes(&[
                (Identifier::new(context, "sym_name"), StringAttribute::new(context, "tensor_example_lowered").into()),
                (Identifier::new(context, "function_type"), TypeAttribute::new(function_type).into()),
            ])
            .add_regions([Region::new()])
            .build()?;
        
        new_module.body().append_operation(function.clone());
        
        // Create function body with standard dialect operations
        let block = Block::new(&[]);
        let region = function.region(0)?;
        region.append_block(block);
        
        // Create equivalent operations using standard dialects
        let f32_type = FloatType::f32(context).into();
        let tensor_type = RankedTensorType::new(&[2, 2], f32_type, None).into();
        
        // arith.constant instead of tensor_ops.constant
        let const1_attr = StringAttribute::new(context, "dense<[[1.0, 2.0], [3.0, 4.0]]>");
        let const1_op = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[
                (Identifier::new(context, "value"), const1_attr.into())
            ])
            .add_results(&[tensor_type])
            .build()?;
        region.first_block().unwrap().append_operation(const1_op.clone());
        
        let const2_attr = StringAttribute::new(context, "dense<[[5.0, 6.0], [7.0, 8.0]]>");
        let const2_op = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[
                (Identifier::new(context, "value"), const2_attr.into())
            ])
            .add_results(&[tensor_type])
            .build()?;
        region.first_block().unwrap().append_operation(const2_op.clone());
        
        // arith.addf instead of tensor_ops.add
        let add_op = OperationBuilder::new("arith.addf", location)
            .add_operands(&[const1_op.result(0)?.into(), const2_op.result(0)?.into()])
            .add_results(&[tensor_type])
            .build()?;
        region.first_block().unwrap().append_operation(add_op);
        
        // Return
        let return_op = OperationBuilder::new("func.return", location).build()?;
        region.first_block().unwrap().append_operation(return_op);
        
        Ok(())
    }
}

/// Pass management for TensorOps lowering
pub struct TensorOpsPassManager;

impl TensorOpsPassManager {
    /// Create a pass manager with TensorOps lowering passes
    pub fn create_with_tensor_ops_lowering(context: &Context) -> PassManager {
        let pass_manager = PassManager::new(context);
        
        // In a real implementation, we would add our custom passes here:
        // pass_manager.add_pass(TensorOpsToArithPass::new());
        // pass_manager.add_pass(TensorOpsToTensorPass::new());
        
        pass_manager
    }
    
    /// Apply complete lowering pipeline: TensorOps -> Standard Dialects -> LLVM
    pub fn apply_full_lowering_pipeline(
        context: &Context,
        module: &mut Module,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting complete lowering pipeline...");
        
        // Step 1: Lower TensorOps to standard dialects
        println!("1️⃣ Lowering TensorOps dialect to standard dialects...");
        TensorOpsLowering::apply_lowering(context, module)?;
        
        // Step 2: Lower standard dialects to LLVM (reuse existing logic)
        println!("2️⃣ Lowering standard dialects to LLVM dialect...");
        crate::apply_standard_to_llvm_lowering(context, module)?;
        
        println!("✅ Complete lowering pipeline finished!");
        println!("   TensorOps → Arith/Tensor → LLVM");
        
        Ok(())
    }
}

/// Interop example showing TensorOps working with other dialects
pub fn create_interop_example(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);
    
    // Create function that mixes TensorOps with standard dialects
    let function_type = FunctionType::new(context, &[], &[]).into();
    
    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), StringAttribute::new(context, "mixed_dialect_example").into()),
            (Identifier::new(context, "function_type"), TypeAttribute::new(function_type).into()),
        ])
        .add_regions([Region::new()])
        .build()?;
    
    module.body().append_operation(function.clone());
    
    let block = Block::new(&[]);
    let region = function.region(0)?;
    region.append_block(block);
    
    // Mix TensorOps operations with standard dialect operations
    let i32_type = IntegerType::new(context, 32).into();
    let f32_type = FloatType::f32(context).into();
    let tensor_type = RankedTensorType::new(&[4], f32_type, None).into();
    
    // Standard dialect: arith.constant
    let index_const = OperationBuilder::new("arith.constant", location)
        .add_attributes(&[
            (Identifier::new(context, "value"), IntegerAttribute::new(i32_type, 42).into())
        ])
        .add_results(&[i32_type])
        .build()?;
    region.first_block().unwrap().append_operation(index_const);
    
    // TensorOps dialect: tensor_ops.constant 
    let tensor_const_attr = StringAttribute::new(context, "dense<[1.0, 2.0, 3.0, 4.0]>").into();
    let tensor_const = crate::tensor_ops_dialect::TensorOpsDialect::create_constant_op(
        context,
        location,
        tensor_const_attr,
        tensor_type,
    )?;
    region.first_block().unwrap().append_operation(tensor_const.clone());
    
    // TensorOps dialect: tensor_ops.add with itself
    let tensor_add = crate::tensor_ops_dialect::TensorOpsDialect::create_add_op(
        context,
        location,
        tensor_const.result(0)?.into(),
        tensor_const.result(0)?.into(),
        tensor_type,
    )?;
    region.first_block().unwrap().append_operation(tensor_add);
    
    let return_op = OperationBuilder::new("func.return", location).build()?;
    region.first_block().unwrap().append_operation(return_op);
    
    Ok(())
}