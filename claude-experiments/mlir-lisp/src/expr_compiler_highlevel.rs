/// High-level expression compiler that emits lisp.* dialect operations
///
/// This is an alternative to expr_compiler.rs that emits our custom
/// high-level dialect instead of going directly to arith/scf.
///
/// Benefits:
/// - More semantic information preserved
/// - Better optimization opportunities at high level
/// - Can analyze Lisp-specific patterns
/// - Progressive lowering gives us multiple optimization stages

use crate::parser::Value;
use crate::emitter::Emitter;
use crate::function_registry::FunctionRegistry;
use crate::lisp_ops::LispOps;
use melior::ir::Block;

/// High-level expression compiler
pub struct HighLevelExprCompiler;

impl HighLevelExprCompiler {
    /// Compile an expression to high-level lisp dialect operations
    pub fn compile_expr<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        expr: &Value,
        registry: &FunctionRegistry<'c>,
    ) -> Result<String, String> {
        match expr {
            // Symbol - reference to existing value
            Value::Symbol(name) => Ok(name.clone()),

            // Integer literal - emit lisp.constant
            Value::Integer(n) => {
                let const_val = LispOps::emit_constant(emitter, block, *n)?;
                let name = emitter.generate_name("const");
                emitter.register_value(name.clone(), const_val);
                Ok(name)
            }

            // List - function call or special form
            Value::List(elements) if !elements.is_empty() => {
                if let Value::Symbol(op) = &elements[0] {
                    match op.as_str() {
                        "+" => Self::compile_add(emitter, block, &elements[1..], registry),
                        "-" => Self::compile_sub(emitter, block, &elements[1..], registry),
                        "*" => Self::compile_mul(emitter, block, &elements[1..], registry),
                        // For now, forward other ops to standard compiler
                        _ => Err(format!("Operation {} not yet supported in high-level mode", op)),
                    }
                } else {
                    Err("Expression must start with a symbol".to_string())
                }
            }

            _ => Err(format!("Cannot compile expression: {:?}", expr)),
        }
    }

    /// Compile addition: (+ a b)
    fn compile_add<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
    ) -> Result<String, String> {
        if args.len() != 2 {
            return Err("+ requires exactly 2 arguments".to_string());
        }

        // Recursively compile operands
        let left_name = Self::compile_expr(emitter, block, &args[0], registry)?;
        let right_name = Self::compile_expr(emitter, block, &args[1], registry)?;

        let left_val = emitter.get_value(&left_name)
            .ok_or(format!("Cannot find value: {}", left_name))?;
        let right_val = emitter.get_value(&right_name)
            .ok_or(format!("Cannot find value: {}", right_name))?;

        // Emit lisp.add
        let result = LispOps::emit_add(emitter, block, left_val, right_val)?;
        let name = emitter.generate_name("add");
        emitter.register_value(name.clone(), result);

        Ok(name)
    }

    /// Compile subtraction: (- a b)
    fn compile_sub<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
    ) -> Result<String, String> {
        if args.len() != 2 {
            return Err("- requires exactly 2 arguments".to_string());
        }

        let left_name = Self::compile_expr(emitter, block, &args[0], registry)?;
        let right_name = Self::compile_expr(emitter, block, &args[1], registry)?;

        let left_val = emitter.get_value(&left_name)
            .ok_or(format!("Cannot find value: {}", left_name))?;
        let right_val = emitter.get_value(&right_name)
            .ok_or(format!("Cannot find value: {}", right_name))?;

        let result = LispOps::emit_sub(emitter, block, left_val, right_val)?;
        let name = emitter.generate_name("sub");
        emitter.register_value(name.clone(), result);

        Ok(name)
    }

    /// Compile multiplication: (* a b)
    fn compile_mul<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
    ) -> Result<String, String> {
        if args.len() != 2 {
            return Err("* requires exactly 2 arguments".to_string());
        }

        let left_name = Self::compile_expr(emitter, block, &args[0], registry)?;
        let right_name = Self::compile_expr(emitter, block, &args[1], registry)?;

        let left_val = emitter.get_value(&left_name)
            .ok_or(format!("Cannot find value: {}", left_name))?;
        let right_val = emitter.get_value(&right_name)
            .ok_or(format!("Cannot find value: {}", right_name))?;

        let result = LispOps::emit_mul(emitter, block, left_val, right_val)?;
        let name = emitter.generate_name("mul");
        emitter.register_value(name.clone(), result);

        Ok(name)
    }
}
