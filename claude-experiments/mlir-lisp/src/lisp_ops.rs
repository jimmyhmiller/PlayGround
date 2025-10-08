/// Custom MLIR operations in the "lisp" namespace
///
/// This demonstrates how to create custom operations without needing
/// a full dialect definition in C++. We use MLIR's generic operation
/// builder to create ops with custom names.

use crate::emitter::Emitter;
use melior::ir::{
    Block, BlockLike, Location,
    operation::{OperationBuilder, OperationLike},
};

/// High-level "lisp" dialect operations
///
/// These operations represent Lisp semantics at a high level:
/// - lisp.constant - immutable constant values
/// - lisp.add - pure functional addition
/// - lisp.sub - pure functional subtraction
/// - lisp.mul - pure functional multiplication
/// - lisp.call - tail-call optimizable function calls
///
/// These will be lowered to standard MLIR dialects via pattern rewrites.
pub struct LispOps;

impl LispOps {
    /// Emit a "lisp.constant" operation
    /// This is our own custom constant operation that we can later lower
    pub fn emit_constant<'c>(
        emitter: &Emitter<'c>,
        block: &Block<'c>,
        value: i64,
    ) -> Result<melior::ir::Value<'c, 'c>, String> {
        let i32_type = emitter.parse_type("i32")?;

        // Create a custom "lisp.constant" operation
        let op = OperationBuilder::new("lisp.constant", Location::unknown(emitter.context()))
            .add_attributes(&[(
                melior::ir::Identifier::new(emitter.context(), "value"),
                melior::ir::attribute::IntegerAttribute::new(i32_type, value).into(),
            )])
            .add_results(&[i32_type])
            .build()
            .map_err(|e| format!("Failed to build lisp.constant: {:?}", e))?;

        let result = op.result(0)
            .map_err(|e| format!("Failed to get result: {:?}", e))?;

        unsafe { block.append_operation(op); }

        Ok(result.into())
    }

    /// Emit a "lisp.add" operation
    /// Custom addition that we can transform/optimize
    pub fn emit_add<'c>(
        emitter: &Emitter<'c>,
        block: &Block<'c>,
        lhs: melior::ir::Value<'c, 'c>,
        rhs: melior::ir::Value<'c, 'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, String> {
        let i32_type = emitter.parse_type("i32")?;

        let op = OperationBuilder::new("lisp.add", Location::unknown(emitter.context()))
            .add_operands(&[lhs, rhs])
            .add_results(&[i32_type])
            .build()
            .map_err(|e| format!("Failed to build lisp.add: {:?}", e))?;

        let result = op.result(0)
            .map_err(|e| format!("Failed to get result: {:?}", e))?;

        unsafe { block.append_operation(op); }

        Ok(result.into())
    }

    /// Emit a "lisp.call" operation
    /// Custom function call that could have special semantics
    pub fn emit_call<'c>(
        emitter: &Emitter<'c>,
        block: &Block<'c>,
        callee: &str,
        args: &[melior::ir::Value<'c, 'c>],
        result_type: melior::ir::r#type::Type<'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, String> {
        let op = OperationBuilder::new("lisp.call", Location::unknown(emitter.context()))
            .add_attributes(&[(
                melior::ir::Identifier::new(emitter.context(), "callee"),
                melior::ir::attribute::FlatSymbolRefAttribute::new(emitter.context(), callee).into(),
            )])
            .add_operands(args)
            .add_results(&[result_type])
            .build()
            .map_err(|e| format!("Failed to build lisp.call: {:?}", e))?;

        let result = op.result(0)
            .map_err(|e| format!("Failed to get result: {:?}", e))?;

        unsafe { block.append_operation(op); }

        Ok(result.into())
    }

    /// Emit a "lisp.sub" operation
    pub fn emit_sub<'c>(
        emitter: &Emitter<'c>,
        block: &Block<'c>,
        lhs: melior::ir::Value<'c, 'c>,
        rhs: melior::ir::Value<'c, 'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, String> {
        let i32_type = emitter.parse_type("i32")?;

        let op = OperationBuilder::new("lisp.sub", Location::unknown(emitter.context()))
            .add_operands(&[lhs, rhs])
            .add_results(&[i32_type])
            .build()
            .map_err(|e| format!("Failed to build lisp.sub: {:?}", e))?;

        let result = op.result(0)
            .map_err(|e| format!("Failed to get result: {:?}", e))?;

        block.append_operation(op);

        Ok(result.into())
    }

    /// Emit a "lisp.mul" operation
    pub fn emit_mul<'c>(
        emitter: &Emitter<'c>,
        block: &Block<'c>,
        lhs: melior::ir::Value<'c, 'c>,
        rhs: melior::ir::Value<'c, 'c>,
    ) -> Result<melior::ir::Value<'c, 'c>, String> {
        let i32_type = emitter.parse_type("i32")?;

        let op = OperationBuilder::new("lisp.mul", Location::unknown(emitter.context()))
            .add_operands(&[lhs, rhs])
            .add_results(&[i32_type])
            .build()
            .map_err(|e| format!("Failed to build lisp.mul: {:?}", e))?;

        let result = op.result(0)
            .map_err(|e| format!("Failed to get result: {:?}", e))?;

        block.append_operation(op);

        Ok(result.into())
    }
}
