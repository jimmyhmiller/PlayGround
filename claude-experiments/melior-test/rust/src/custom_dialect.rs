use melior::{
    Context,
    ir::{
        Attribute, Block, Identifier, Location, Module, Region, Type, Value,
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        operation::{Operation, OperationBuilder},
        r#type::{FunctionType, IntegerType},
    },
};

/// A simple arithmetic dialect for demonstration
/// Operations: calc.const, calc.add, calc.mul, calc.return
pub struct CalcDialect;

impl CalcDialect {
    pub const NAMESPACE: &'static str = "calc";

    /// Create a calc.const operation - loads a constant integer
    pub fn create_const<'c>(
        context: &'c Context,
        location: Location<'c>,
        value: i64,
        result_type: Type<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let op = OperationBuilder::new("calc.const", location)
            .add_attributes(&[(
                Identifier::new(context, "value"),
                IntegerAttribute::new(result_type, value).into(),
            )])
            .add_results(&[result_type])
            .build()?;

        Ok(op)
    }

    /// Create a calc.add operation - adds two values
    pub fn create_add<'c>(
        context: &'c Context,
        location: Location<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        result_type: Type<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let op = OperationBuilder::new("calc.add", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;

        Ok(op)
    }

    /// Create a calc.mul operation - multiplies two values
    pub fn create_mul<'c>(
        context: &'c Context,
        location: Location<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        result_type: Type<'c>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let op = OperationBuilder::new("calc.mul", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?;

        Ok(op)
    }

    /// Create a calc.return operation - returns a value
    pub fn create_return<'c>(
        context: &'c Context,
        location: Location<'c>,
        operand: Option<Value<'c, '_>>,
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        let mut builder = OperationBuilder::new("calc.return", location);
        
        if let Some(val) = operand {
            builder = builder.add_operands(&[val]);
        }
        
        let op = builder.build()?;
        Ok(op)
    }

    /// Create a complete calc function with body
    pub fn create_function<'c>(
        context: &'c Context,
        location: Location<'c>,
        name: &str,
        input_types: &[Type<'c>],
        output_types: &[Type<'c>],
    ) -> Result<Operation<'c>, Box<dyn std::error::Error>> {
        // Create function type
        let function_type = FunctionType::new(context, input_types, output_types);

        // Create function region with entry block
        let mut region = Region::new();
        let mut entry_block = Block::new(&[]);
        
        // Add arguments to the entry block
        for input_type in input_types {
            entry_block.add_argument(*input_type, location);
        }
        
        region.append_block(entry_block);

        // Create func.func operation (using standard func dialect)
        let op = OperationBuilder::new("func.func", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "sym_name"),
                    StringAttribute::new(context, name).into(),
                ),
                (
                    Identifier::new(context, "function_type"),
                    TypeAttribute::new(function_type.into()).into(),
                ),
                (
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "public").into(),
                ),
            ])
            .add_regions([region])
            .build()?;

        Ok(op)
    }

    /// Build a complete arithmetic program in our dialect
    /// Computes: (a + b) * c where a=10, b=20, c=3
    /// Expected result: (10 + 20) * 3 = 90
    pub fn build_arithmetic_program<'c>(
        context: &'c Context,
        location: Location<'c>,
    ) -> Result<Module<'c>, Box<dyn std::error::Error>> {
        let module = Module::new(location);
        
        // Create i32 type for our calculations
        let i32_type = IntegerType::new(context, 32);

        // Create function: calc_example() -> i32
        let function = Self::create_function(
            context,
            location,
            "calc_example",
            &[],  // No inputs
            &[i32_type.into()],  // Returns i32
        )?;

        // Get the function's entry block to add operations
        let function_region = function.region(0)?;
        let entry_block = function_region.first_block().unwrap();

        // Build the arithmetic expression: (10 + 20) * 3

        // Constants
        let const_10 = Self::create_const(context, location, 10, i32_type.into())?;
        let const_20 = Self::create_const(context, location, 20, i32_type.into())?;
        let const_3 = Self::create_const(context, location, 3, i32_type.into())?;

        // Add operation: 10 + 20
        let add_result = Self::create_add(
            context,
            location,
            const_10.result(0)?.into(),
            const_20.result(0)?.into(),
            i32_type.into(),
        )?;

        // Multiply operation: (10 + 20) * 3
        let mul_result = Self::create_mul(
            context,
            location,
            add_result.result(0)?.into(),
            const_3.result(0)?.into(),
            i32_type.into(),
        )?;

        // Return the result
        let return_op = Self::create_return(
            context,
            location,
            Some(mul_result.result(0)?.into()),
        )?;

        // Add all operations to the function body
        entry_block.append_operation(const_10);
        entry_block.append_operation(const_20);
        entry_block.append_operation(const_3);
        entry_block.append_operation(add_result);
        entry_block.append_operation(mul_result);
        entry_block.append_operation(return_op);

        // Add function to module
        module.body().append_operation(function);

        Ok(module)
    }
}