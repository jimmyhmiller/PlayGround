use crate::parser::Value;
use crate::emitter::Emitter;
use crate::function_registry::FunctionRegistry;
use crate::dialect_registry::DialectRegistry;
use melior::ir::Block;
use melior::ir::operation::OperationLike;

/// Compile high-level expressions to MLIR operations
/// Handles nested expressions and generates appropriate SSA names
pub struct ExprCompiler;

impl ExprCompiler {
    /// Compile an expression and return the SSA name of its result
    /// This recursively compiles nested expressions first
    pub fn compile_expr<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        expr: &Value,
        registry: &FunctionRegistry<'c>,
        dialect_registry: Option<&DialectRegistry>,
    ) -> Result<String, String> {
        match expr {
            // Symbol - reference to existing value (function argument, etc.)
            Value::Symbol(name) => Ok(name.clone()),

            // Integer literal - emit constant
            Value::Integer(n) => {
                let const_op = Value::List(vec![
                    Value::Symbol("op".to_string()),
                    Value::Symbol("arith.constant".to_string()),
                    Value::Keyword("attrs".to_string()),
                    Value::Map(vec![(
                        Value::Keyword("value".to_string()),
                        Value::Integer(*n),
                    )]),
                    Value::Keyword("results".to_string()),
                    Value::Vector(vec![Value::Symbol("i32".to_string())]),
                ]);

                // Emit the operation and get auto-generated name
                if let Value::List(elements) = const_op {
                    if let Value::Symbol(s) = &elements[0] {
                        if s == "op" {
                            if let Some(name) = emitter.emit_op_form_public(block, &elements[1..])? {
                                return Ok(name);
                            }
                        }
                    }
                }
                Err("Failed to emit constant".to_string())
            }

            // List - function call or special form
            Value::List(elements) if !elements.is_empty() => {
                if let Value::Symbol(op) = &elements[0] {
                    // Check if this is an explicit dialect operation (contains '.')
                    if op.contains('.') {
                        return Self::compile_dialect_op(emitter, block, op, &elements[1..], registry, dialect_registry);
                    }

                    match op.as_str() {
                        "if" => Self::compile_if(emitter, block, &elements[1..], registry, dialect_registry),
                        "+" | "-" | "*" | "/" => Self::compile_binop(emitter, block, op, &elements[1..], registry, dialect_registry),
                        "<" | "<=" | ">" | ">=" | "=" => Self::compile_comparison(emitter, block, op, &elements[1..], registry, dialect_registry),
                        _ => Self::compile_call(emitter, block, op, &elements[1..], registry, dialect_registry),
                    }
                } else {
                    Err("Expression must start with a symbol".to_string())
                }
            }

            _ => Err(format!("Cannot compile expression: {:?}", expr)),
        }
    }

    /// Compile binary operation: (+ a b)
    fn compile_binop<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        op: &str,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
        dialect_registry: Option<&DialectRegistry>,
    ) -> Result<String, String> {
        if args.len() != 2 {
            return Err(format!("{} requires exactly 2 arguments", op));
        }

        // Recursively compile operands
        let left = Self::compile_expr(emitter, block, &args[0], registry, dialect_registry)?;
        let right = Self::compile_expr(emitter, block, &args[1], registry, dialect_registry)?;

        // Check if there's a registered dialect operation for this
        let mlir_op = if let Some(dreg) = dialect_registry {
            // Look for a dialect that has an operation for this operator
            // For now, use arith as default, but this could be configurable
            match op {
                "+" => Self::find_dialect_op(dreg, "add").unwrap_or("arith.addi"),
                "-" => Self::find_dialect_op(dreg, "sub").unwrap_or("arith.subi"),
                "*" => Self::find_dialect_op(dreg, "mul").unwrap_or("arith.muli"),
                "/" => Self::find_dialect_op(dreg, "div").unwrap_or("arith.divsi"),
                _ => return Err(format!("Unknown binary op: {}", op)),
            }
        } else {
            match op {
                "+" => "arith.addi",
                "-" => "arith.subi",
                "*" => "arith.muli",
                "/" => "arith.divsi",
                _ => return Err(format!("Unknown binary op: {}", op)),
            }
        };

        let op_form = vec![
            Value::Symbol(mlir_op.to_string()),
            Value::Keyword("operands".to_string()),
            Value::Vector(vec![
                Value::Symbol(left),
                Value::Symbol(right),
            ]),
            Value::Keyword("results".to_string()),
            Value::Vector(vec![Value::Symbol("i32".to_string())]),
        ];

        if let Some(name) = emitter.emit_op_form_public(block, &op_form)? {
            Ok(name)
        } else {
            Err(format!("Failed to emit {}", op))
        }
    }

    /// Compile comparison: (< a b)
    fn compile_comparison<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        op: &str,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
        dialect_registry: Option<&DialectRegistry>,
    ) -> Result<String, String> {
        if args.len() != 2 {
            return Err(format!("{} requires exactly 2 arguments", op));
        }

        // Recursively compile operands
        let left = Self::compile_expr(emitter, block, &args[0], registry, dialect_registry)?;
        let right = Self::compile_expr(emitter, block, &args[1], registry, dialect_registry)?;

        // Map to MLIR predicate
        let predicate = match op {
            "<" => "slt",
            "<=" => "sle",
            ">" => "sgt",
            ">=" => "sge",
            "=" => "eq",
            _ => return Err(format!("Unknown comparison: {}", op)),
        };

        let op_form = vec![
            Value::Symbol("arith.cmpi".to_string()),
            Value::Keyword("attrs".to_string()),
            Value::Map(vec![(
                Value::Keyword("predicate".to_string()),
                Value::String(predicate.to_string()),  // Changed from Symbol to String
            )]),
            Value::Keyword("operands".to_string()),
            Value::Vector(vec![
                Value::Symbol(left),
                Value::Symbol(right),
            ]),
            Value::Keyword("results".to_string()),
            Value::Vector(vec![Value::Symbol("i1".to_string())]),
        ];

        if let Some(name) = emitter.emit_op_form_public(block, &op_form)? {
            Ok(name)
        } else {
            Err("Failed to emit comparison".to_string())
        }
    }

    /// Compile function call: (fib n)
    fn compile_call<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        func_name: &str,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
        dialect_registry: Option<&DialectRegistry>,
    ) -> Result<String, String> {
        // Check if function is declared
        if !registry.is_declared(func_name) {
            return Err(format!("Undeclared function: {}", func_name));
        }

        // Get return type from registry
        let return_type = registry.get_return_type(func_name)
            .ok_or(format!("Cannot get return type for: {}", func_name))?;

        let type_string = Self::type_to_string(return_type);

        // Recursively compile arguments
        let mut arg_names = Vec::new();
        for arg in args {
            arg_names.push(Value::Symbol(Self::compile_expr(emitter, block, arg, registry, dialect_registry)?));
        }

        let op_form = vec![
            Value::Symbol("func.call".to_string()),
            Value::Keyword("attrs".to_string()),
            Value::Map(vec![(
                Value::Keyword("callee".to_string()),
                Value::String(func_name.to_string()),
            )]),
            Value::Keyword("operands".to_string()),
            Value::Vector(arg_names),
            Value::Keyword("results".to_string()),
            Value::Vector(vec![Value::Symbol(type_string)]),
        ];

        if let Some(name) = emitter.emit_op_form_public(block, &op_form)? {
            Ok(name)
        } else {
            Err("Failed to emit call".to_string())
        }
    }

    /// Compile if expression: (if cond then else)
    /// Uses scf.if from the SCF dialect which yields a value
    fn compile_if<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
        dialect_registry: Option<&DialectRegistry>,
    ) -> Result<String, String> {
        use melior::ir::{Region, Block as MeliorBlock, Location, operation::OperationBuilder, RegionLike, BlockLike};

        if args.len() != 3 {
            return Err("if requires exactly 3 arguments: condition, then-expr, else-expr".to_string());
        }

        // Compile condition (should yield i1)
        let cond_name = Self::compile_expr(emitter, block, &args[0], registry, dialect_registry)?;
        let cond_val = emitter.get_value(&cond_name)
            .ok_or(format!("Cannot find condition value: {}", cond_name))?;

        // Create then region
        let then_region = Region::new();
        let then_block = MeliorBlock::new(&[]);
        then_region.append_block(then_block);
        let then_block_ref = then_region.first_block()
            .ok_or("Failed to get then block")?;

        // Compile then expression in its block
        let then_result_name = Self::compile_expr(emitter, &then_block_ref, &args[1], registry, dialect_registry)?;
        let then_result = emitter.get_value(&then_result_name)
            .ok_or(format!("Cannot find then result: {}", then_result_name))?;

        // Emit scf.yield for then block
        let then_yield = OperationBuilder::new("scf.yield", Location::unknown(emitter.context()))
            .add_operands(&[then_result])
            .build()
            .map_err(|e| format!("Failed to build then yield: {:?}", e))?;
        unsafe { then_block_ref.append_operation(then_yield); }

        // Create else region
        let else_region = Region::new();
        let else_block = MeliorBlock::new(&[]);
        else_region.append_block(else_block);
        let else_block_ref = else_region.first_block()
            .ok_or("Failed to get else block")?;

        // Compile else expression in its block
        let else_result_name = Self::compile_expr(emitter, &else_block_ref, &args[2], registry, dialect_registry)?;
        let else_result = emitter.get_value(&else_result_name)
            .ok_or(format!("Cannot find else result: {}", else_result_name))?;

        // Emit scf.yield for else block
        let else_yield = OperationBuilder::new("scf.yield", Location::unknown(emitter.context()))
            .add_operands(&[else_result])
            .build()
            .map_err(|e| format!("Failed to build else yield: {:?}", e))?;
        unsafe { else_block_ref.append_operation(else_yield); }

        // Get result type from then branch
        // For melior Value, we need to get the type differently
        // Let's just use i32 for now - we can improve type inference later
        let result_type = emitter.parse_type("i32")
            .map_err(|e| format!("Failed to parse result type: {}", e))?;

        // Build scf.if operation
        let if_op = OperationBuilder::new("scf.if", Location::unknown(emitter.context()))
            .add_operands(&[cond_val])
            .add_regions([then_region, else_region])
            .add_results(&[result_type])
            .build()
            .map_err(|e| format!("Failed to build scf.if: {:?}", e))?;

        // Get the result before appending (same pattern as emit_op_form_impl)
        let result_name = emitter.generate_name("if_result");
        if let Ok(result_val) = if_op.result(0) {
            emitter.register_value(result_name.clone(), result_val.into());
        }

        unsafe { block.append_operation(if_op); }

        Ok(result_name)
    }

    /// Convert MLIR type to string for emission
    /// For now, just format it - this is a simple heuristic
    fn type_to_string<'c>(ty: melior::ir::Type<'c>) -> String {
        let type_str = format!("{}", ty);
        // Extract just the type name (e.g., "i32" from more complex format)
        if type_str.contains("i32") {
            "i32".to_string()
        } else if type_str.contains("i64") {
            "i64".to_string()
        } else if type_str.contains("f32") {
            "f32".to_string()
        } else if type_str.contains("f64") {
            "f64".to_string()
        } else if type_str.contains("i1") {
            "i1".to_string()
        } else {
            // Default to i32
            "i32".to_string()
        }
    }

    /// Find a dialect operation by short name (e.g., "add" -> "calc.add")
    /// Returns the full operation name if found in any registered dialect
    fn find_dialect_op(dialect_registry: &DialectRegistry, op_name: &str) -> Option<&'static str> {
        // For now, check for the "calc" dialect specifically
        // In the future, this could be configurable
        if let Some(dialect) = dialect_registry.get_dialect("calc") {
            for op in &dialect.operations {
                if op.name == op_name {
                    // Return a statically allocated string
                    // This is a bit hacky, but works for our purposes
                    return match op_name {
                        "add" => Some("calc.add"),
                        "sub" => Some("calc.sub"),
                        "mul" => Some("calc.mul"),
                        "div" => Some("calc.div"),
                        "constant" => Some("calc.constant"),
                        _ => None,
                    };
                }
            }
        }
        None
    }

    /// Compile an explicit dialect operation like (calc.add x y)
    fn compile_dialect_op<'c>(
        emitter: &mut Emitter<'c>,
        block: &Block<'c>,
        op_name: &str,
        args: &[Value],
        registry: &FunctionRegistry<'c>,
        dialect_registry: Option<&DialectRegistry>,
    ) -> Result<String, String> {
        // Split operation name to get dialect and op
        let parts: Vec<&str> = op_name.split('.').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid dialect operation: {}", op_name));
        }

        let dialect_name = parts[0];
        let short_op_name = parts[1];

        // Look up the operation definition in the dialect registry
        if let Some(dreg) = dialect_registry {
            if let Some(dialect) = dreg.get_dialect(dialect_name) {
                for op_def in &dialect.operations {
                    if op_def.name == short_op_name {
                        // Handle special case for constant operations
                        if short_op_name == "constant" {
                            if args.len() != 1 {
                                return Err(format!("{} requires exactly 1 argument", op_name));
                            }

                            // Extract the constant value
                            let val = match &args[0] {
                                Value::Integer(n) => *n,
                                _ => return Err("Constant requires integer argument".to_string()),
                            };

                            let op_form = vec![
                                Value::Symbol(op_name.to_string()),
                                Value::Keyword("attrs".to_string()),
                                Value::Map(vec![(
                                    Value::Keyword("value".to_string()),
                                    Value::Integer(val),
                                )]),
                                Value::Keyword("results".to_string()),
                                Value::Vector(vec![Value::Symbol("i32".to_string())]),
                            ];

                            if let Some(name) = emitter.emit_op_form_public(block, &op_form)? {
                                return Ok(name);
                            } else {
                                return Err(format!("Failed to emit {}", op_name));
                            }
                        }

                        // Handle binary operations (add, mul, etc.)
                        if args.len() != 2 {
                            return Err(format!("{} requires exactly 2 arguments", op_name));
                        }

                        // Recursively compile operands
                        let left = Self::compile_expr(emitter, block, &args[0], registry, dialect_registry)?;
                        let right = Self::compile_expr(emitter, block, &args[1], registry, dialect_registry)?;

                        let op_form = vec![
                            Value::Symbol(op_name.to_string()),
                            Value::Keyword("operands".to_string()),
                            Value::Vector(vec![
                                Value::Symbol(left),
                                Value::Symbol(right),
                            ]),
                            Value::Keyword("results".to_string()),
                            Value::Vector(vec![Value::Symbol("i32".to_string())]),
                        ];

                        if let Some(name) = emitter.emit_op_form_public(block, &op_form)? {
                            return Ok(name);
                        } else {
                            return Err(format!("Failed to emit {}", op_name));
                        }
                    }
                }
            }
        }

        Err(format!("Unknown dialect operation: {}", op_name))
    }
}
