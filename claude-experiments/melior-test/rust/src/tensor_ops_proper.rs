//! Proper TensorOps dialect implementation using C++ dialect through FFI

use melior::{
    Context, 
    ir::{
        Location, Value, Type, Attribute,
        operation::Operation,
        r#type::{IntegerType, RankedTensorType},
        ValueLike, TypeLike, AttributeLike,
    },
    dialect::DialectRegistry,
};
use crate::tensor_ops_ffi::*;

/// Proper TensorOps dialect using C++ implementation
pub struct TensorOpsDialect;

impl TensorOpsDialect {
    pub const NAMESPACE: &'static str = "tensor_ops";
    
    /// Register the dialect with MLIR
    pub fn register(registry: &DialectRegistry) {
        unsafe {
            let handle = mlirGetDialectHandle__tensor_ops__();
            mlir_sys::mlirDialectHandleInsertDialect(handle, registry.to_raw());
        }
    }
    
    /// Create a tensor_ops.add operation using the C API
    pub fn create_add_op<'c>(
        context: &'c Context,
        location: Location<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        result_type: Type<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        unsafe {
            let op = mlirTensorOpsCreateAddOp(
                context.to_raw(),
                lhs.to_raw(),
                rhs.to_raw(),
                result_type.to_raw(),
                location.to_raw(),
            );
            
            let null_op = mlir_sys::MlirOperation { ptr: std::ptr::null_mut() };
            if mlir_sys::mlirOperationEqual(op, null_op) {
                return Err("Failed to create tensor_ops.add operation".into());
            }
            
            Ok(Operation::from_raw(op))
        }
    }
    
    /// Create a tensor_ops.mul operation using the C API
    pub fn create_mul_op<'c>(
        context: &'c Context,
        location: Location<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        result_type: Type<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        unsafe {
            let op = mlirTensorOpsCreateMulOp(
                context.to_raw(),
                lhs.to_raw(),
                rhs.to_raw(),
                result_type.to_raw(),
                location.to_raw(),
            );
            
            let null_op = mlir_sys::MlirOperation { ptr: std::ptr::null_mut() };
            if mlir_sys::mlirOperationEqual(op, null_op) {
                return Err("Failed to create tensor_ops.mul operation".into());
            }
            
            Ok(Operation::from_raw(op))
        }
    }
    
    /// Create a tensor_ops.constant operation using the C API
    pub fn create_constant_op<'c>(
        context: &'c Context,
        location: Location<'c>,
        value: Attribute<'c>,
        result_type: Type<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        unsafe {
            let op = mlirTensorOpsCreateConstantOp(
                context.to_raw(),
                value.to_raw(),
                result_type.to_raw(),
                location.to_raw(),
            );
            
            let null_op = mlir_sys::MlirOperation { ptr: std::ptr::null_mut() };
            if mlir_sys::mlirOperationEqual(op, null_op) {
                return Err("Failed to create tensor_ops.constant operation".into());
            }
            
            Ok(Operation::from_raw(op))
        }
    }
    
    /// Create a tensor_ops.reshape operation using the C API
    pub fn create_reshape_op<'c>(
        context: &'c Context,
        location: Location<'c>,
        input: Value<'c, '_>,
        shape: Attribute<'c>,
        result_type: Type<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        unsafe {
            let op = mlirTensorOpsCreateReshapeOp(
                context.to_raw(),
                input.to_raw(),
                shape.to_raw(),
                result_type.to_raw(),
                location.to_raw(),
            );
            
            let null_op = mlir_sys::MlirOperation { ptr: std::ptr::null_mut() };
            if mlir_sys::mlirOperationEqual(op, null_op) {
                return Err("Failed to create tensor_ops.reshape operation".into());
            }
            
            Ok(Operation::from_raw(op))
        }
    }
}

/// Helper functions for creating tensor types
pub mod types {
    use super::*;
    
    /// Create a ranked tensor type
    pub fn create_tensor_type<'c>(_context: &'c Context, shape: &[u64], element_type: Type<'c>) -> Type<'c> {
        RankedTensorType::new(shape, element_type, None).into()
    }
    
    /// Create i32 type  
    pub fn create_i32_type<'c>(context: &'c Context) -> Type<'c> {
        IntegerType::new(context, 32).into()
    }
}