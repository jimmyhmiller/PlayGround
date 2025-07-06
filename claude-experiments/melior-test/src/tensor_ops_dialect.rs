use melior::{
    Context, 
    ir::{
        Module, Location, Block, Region, Value, Type, Attribute,
        operation::{Operation, OperationBuilder},
        attribute::{StringAttribute, TypeAttribute, IntegerAttribute, ArrayAttribute},
        r#type::{FunctionType, IntegerType, RankedTensorType},
        Identifier
    },
    dialect::DialectRegistry,
};
use std::ffi::CString;

/// Custom TensorOps dialect for high-level tensor operations
pub struct TensorOpsDialect;

impl TensorOpsDialect {
    pub const NAMESPACE: &'static str = "tensor_ops";
    
    /// Register the dialect (in a real implementation, this would use TableGen)
    pub fn register(registry: &DialectRegistry) {
        // In a real implementation, this would register the dialect with MLIR
        // For now, we'll create operations manually
    }
    
    /// Create a tensor.add operation: %result = tensor_ops.add %lhs, %rhs : tensor<4x4xf32>
    pub fn create_add_op(
        context: &Context,
        location: Location,
        lhs: Value,
        rhs: Value,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        let op = OperationBuilder::new("tensor_ops.add", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
    
    /// Create a tensor.mul operation: %result = tensor_ops.mul %lhs, %rhs : tensor<4x4xf32>
    pub fn create_mul_op(
        context: &Context,
        location: Location,
        lhs: Value,
        rhs: Value,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        let op = OperationBuilder::new("tensor_ops.mul", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
    
    /// Create a tensor.constant operation: %result = tensor_ops.constant dense<[[1,2],[3,4]]> : tensor<2x2xi32>
    pub fn create_constant_op(
        context: &Context,
        location: Location,
        value: Attribute,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        let op = OperationBuilder::new("tensor_ops.constant", location)
            .add_attributes(&[
                (Identifier::new(context, "value"), value)
            ])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
    
    /// Create a tensor.reshape operation: %result = tensor_ops.reshape %input : tensor<4x4xf32> to tensor<16xf32>
    pub fn create_reshape_op(
        context: &Context,
        location: Location,
        input: Value,
        result_type: Type,
    ) -> Result<Operation, Box<dyn std::error::Error>> {
        let op = OperationBuilder::new("tensor_ops.reshape", location)
            .add_operands(&[input])
            .add_results(&[result_type])
            .build()?;
        
        Ok(op)
    }
}

/// Helper functions for creating tensor types
pub mod types {
    use super::*;
    
    /// Create a ranked tensor type: tensor<4x4xf32>
    pub fn create_tensor_type(context: &Context, shape: &[i64], element_type: Type) -> Type {
        RankedTensorType::new(shape, element_type, None).into()
    }
    
    /// Create f32 type
    pub fn create_f32_type(context: &Context) -> Type {
        use melior::ir::r#type::FloatType;
        FloatType::f32(context).into()
    }
    
    /// Create i32 type  
    pub fn create_i32_type(context: &Context) -> Type {
        IntegerType::new(context, 32).into()
    }
}

/// Example usage of the tensor_ops dialect
pub fn create_example_tensor_computation(
    context: &Context,
    module: &Module,
) -> Result<(), Box<dyn std::error::Error>> {
    let location = Location::unknown(context);
    
    // Create function that uses our tensor operations
    let function_type = FunctionType::new(context, &[], &[]).into();
    
    let function = OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), StringAttribute::new(context, "tensor_example").into()),
            (Identifier::new(context, "function_type"), TypeAttribute::new(function_type).into()),
        ])
        .add_regions([Region::new()])
        .build()?;
    
    module.body().append_operation(function.clone());
    
    // Create function body
    let block = Block::new(&[]);
    let region = function.region(0)?;
    region.append_block(block);
    
    // Create tensor types
    let f32_type = types::create_f32_type(context);
    let tensor_type = types::create_tensor_type(context, &[2, 2], f32_type);
    
    // Create constants using our dialect
    let const1_attr = StringAttribute::new(context, "dense<[[1.0, 2.0], [3.0, 4.0]]>").into();
    let const1_op = TensorOpsDialect::create_constant_op(context, location, const1_attr, tensor_type)?;
    region.first_block().unwrap().append_operation(const1_op.clone());
    
    let const2_attr = StringAttribute::new(context, "dense<[[5.0, 6.0], [7.0, 8.0]]>").into();
    let const2_op = TensorOpsDialect::create_constant_op(context, location, const2_attr, tensor_type)?;
    region.first_block().unwrap().append_operation(const2_op.clone());
    
    // Create add operation using our dialect
    let add_op = TensorOpsDialect::create_add_op(
        context,
        location,
        const1_op.result(0)?.into(),
        const2_op.result(0)?.into(),
        tensor_type,
    )?;
    region.first_block().unwrap().append_operation(add_op);
    
    // Return
    let return_op = OperationBuilder::new("func.return", location).build()?;
    region.first_block().unwrap().append_operation(return_op);
    
    Ok(())
}