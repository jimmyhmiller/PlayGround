use melior::{
    Context,
    ir::{
        Block, Location, Module, Operation, Region, Value, ValueLike,
        operation::OperationBuilder,
    },
    pass::{Pass, PassManager},
};

/// Transforms our custom "calc" dialect to standard MLIR dialects
/// calc.const -> arith.constant
/// calc.add -> arith.addi
/// calc.mul -> arith.muli  
/// calc.return -> func.return
pub struct CalcToStandardTransform;

impl CalcToStandardTransform {
    /// Transform a calc.const operation to arith.constant
    pub fn transform_const<'c>(
        context: &'c Context,
        calc_const: &Operation<'c>,
        location: Location<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        // Extract the value attribute from calc.const
        let value_attr = calc_const
            .attribute("value")
            .unwrap();
        
        // Get the result type
        let result_type = calc_const.result(0)?.into().r#type();

        // Create arith.constant with the same value and type
        let arith_const = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[(
                melior::ir::Identifier::new(context, "value"),
                value_attr,
            )])
            .add_results(&[result_type])
            .build()?;

        Ok(arith_const)
    }

    /// Transform a calc.add operation to arith.addi
    pub fn transform_add<'c>(
        context: &'c Context,
        calc_add: &Operation<'c>,
        location: Location<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let result_type = calc_add.result(0)?.into().r#type();

        let arith_add = OperationBuilder::new("arith.addi", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;

        Ok(arith_add)
    }

    /// Transform a calc.mul operation to arith.muli
    pub fn transform_mul<'c>(
        context: &'c Context,
        calc_mul: &Operation<'c>,
        location: Location<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let result_type = calc_mul.result(0)?.into().r#type();

        let arith_mul = OperationBuilder::new("arith.muli", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;

        Ok(arith_mul)
    }

    /// Transform a calc.return operation to func.return
    pub fn transform_return<'c>(
        _context: &'c Context,
        _calc_return: &Operation<'c>,
        location: Location<'c>,
        operand: Option<Value<'c, '_>>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let mut builder = OperationBuilder::new("func.return", location);
        
        if let Some(val) = operand {
            builder = builder.add_operands(&[val]);
        }
        
        let func_return = builder.build()?;
        Ok(func_return)
    }

    /// Transform an entire module from calc dialect to standard dialects
    pub fn transform_module<'c>(
        context: &'c Context,
        calc_module: &Module<'c>,
    ) -> Result<Module<'c>, Box<dyn std::error::Error>> {
        let location = Location::unknown(context);
        let standard_module = Module::new(location);

        // Process each operation in the module
        for operation in calc_module.body().operations() {
            if operation.name() == "func.func" {
                // Transform function body
                let transformed_func = Self::transform_function(context, &operation)?;
                standard_module.body().append_operation(transformed_func);
            } else {
                // Copy other operations as-is (like module attributes)
                // In a real implementation, you'd clone the operation
                continue;
            }
        }

        Ok(standard_module)
    }

    /// Transform a function from calc dialect to standard dialects
    fn transform_function<'c>(
        context: &'c Context,
        calc_func: &Operation<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let location = Location::unknown(context);
        
        // Create new function with same signature
        let mut region = Region::new();
        let mut new_block = Block::new(&[]);

        // Copy function arguments
        let original_region = calc_func.region(0)?;
        let original_block = original_region.first_block()
            .ok_or("Function has no entry block")?;
        
        // Add arguments to new block
        for arg in original_block.arguments() {
            new_block.add_argument(arg.r#type(), location);
        }

        // Transform operations in the function body
        let mut value_mapping = std::collections::HashMap::new();
        
        // Map original arguments to new arguments
        for (old_arg, new_arg) in original_block.arguments().zip(new_block.arguments()) {
            value_mapping.insert(old_arg.into(), new_arg);
        }

        // Transform each operation
        for operation in original_block.operations() {
            let transformed_op = match operation.name() {
                "calc.const" => {
                    Self::transform_const(context, &operation, location)?
                }
                "calc.add" => {
                    let lhs = Self::map_value(&value_mapping, operation.operand(0)?)?;
                    let rhs = Self::map_value(&value_mapping, operation.operand(1)?)?;
                    Self::transform_add(context, &operation, location, lhs, rhs)?
                }
                "calc.mul" => {
                    let lhs = Self::map_value(&value_mapping, operation.operand(0)?)?;
                    let rhs = Self::map_value(&value_mapping, operation.operand(1)?)?;
                    Self::transform_mul(context, &operation, location, lhs, rhs)?
                }
                "calc.return" => {
                    let operand = if operation.num_operands() > 0 {
                        Some(Self::map_value(&value_mapping, operation.operand(0)?)?)
                    } else {
                        None
                    };
                    Self::transform_return(context, &operation, location, operand)?
                }
                _ => {
                    return Err(format!("Unknown calc dialect operation: {}", operation.name()).into());
                }
            };

            // Update value mapping for results
            for (old_result, new_result) in operation.results().zip(transformed_op.results()) {
                value_mapping.insert(old_result.into(), new_result.into());
            }

            new_block.append_operation(transformed_op);
        }

        region.append_block(new_block);

        // Create new function operation
        let func_op = OperationBuilder::new("func.func", location)
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(context, "sym_name"),
                    calc_func.attribute("sym_name")
                        .ok_or("Function missing sym_name")?,
                ),
                (
                    melior::ir::Identifier::new(context, "function_type"),
                    calc_func.attribute("function_type")
                        .ok_or("Function missing function_type")?,
                ),
                (
                    melior::ir::Identifier::new(context, "sym_visibility"),
                    calc_func.attribute("sym_visibility")
                        .unwrap_or_else(|| {
                            melior::ir::attribute::StringAttribute::new(context, "private").into()
                        }),
                ),
            ])
            .add_regions([region])
            .build()?;

        Ok(func_op)
    }

    /// Helper to map values from old to new operations
    fn map_value<'c>(
        mapping: &std::collections::HashMap<Value<'c, 'c>, Value<'c, 'c>>,
        old_value: Value<'c, '_>,
    ) -> Result<Value<'c, 'c>, Box<dyn std::error::Error>> {
        mapping.get(&old_value.into())
            .copied()
            .ok_or_else(|| format!("Value not found in mapping").into())
    }
}

/// LLVM lowering utilities
pub struct LLVMLowering;

impl LLVMLowering {
    /// Create and configure a pass manager for lowering to LLVM
    pub fn create_lowering_pipeline<'c>(
        context: &'c Context,
    ) -> Result<PassManager<'c>, Box<dyn std::error::Error>> {
        let pm = PassManager::new(context);

        // Convert standard dialects to LLVM dialect
        unsafe {
            // Add conversion passes
            let func_to_llvm = Pass::from_raw(mlir_sys::mlirCreateConversionConvertFuncToLLVMPass());
            pm.add_pass(func_to_llvm);

            let math_to_llvm = Pass::from_raw(mlir_sys::mlirCreateConversionConvertMathToLLVMPass());
            pm.add_pass(math_to_llvm);

            // Reconcile unrealized casts
            let reconcile = Pass::from_raw(mlir_sys::mlirCreateConversionReconcileUnrealizedCasts());
            pm.add_pass(reconcile);
        }

        Ok(pm)
    }

    /// Apply LLVM lowering passes to a module
    pub fn lower_to_llvm<'c>(
        context: &'c Context,
        module: &Module<'c>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pm = Self::create_lowering_pipeline(context)?;
        
        match pm.run(module) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("LLVM lowering failed: {:?}", e).into()),
        }
    }
}