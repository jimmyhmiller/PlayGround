use crate::mlir_context::MlirContext;
use crate::parser::Value;
use melior::{
    ir::{
        Block, BlockLike, Module, Region, RegionLike,
        attribute::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, IntegerType, Type},
        Identifier,
    },
};
use std::collections::HashMap;

use melior::ir::block::BlockRef;

/// Emitter converts parsed Lisp AST into MLIR operations
pub struct Emitter<'c> {
    ctx: &'c MlirContext,
    /// Symbol table mapping SSA value names (%foo) to their MLIR values
    values: HashMap<String, melior::ir::Value<'c, 'c>>,
    /// Block table mapping block names to their MLIR blocks
    blocks: HashMap<String, BlockRef<'c, 'c>>,
}

impl<'c> Emitter<'c> {
    pub fn new(ctx: &'c MlirContext) -> Self {
        Self {
            ctx,
            values: HashMap::new(),
            blocks: HashMap::new(),
        }
    }

    /// Parse a type from a symbol (e.g., "i32" -> IntegerType(32))
    pub fn parse_type(&self, symbol: &str) -> Result<Type<'c>, String> {
        match symbol {
            "i1" => Ok(IntegerType::new(self.ctx.context(), 1).into()),
            "i8" => Ok(IntegerType::new(self.ctx.context(), 8).into()),
            "i16" => Ok(IntegerType::new(self.ctx.context(), 16).into()),
            "i32" => Ok(IntegerType::new(self.ctx.context(), 32).into()),
            "i64" => Ok(IntegerType::new(self.ctx.context(), 64).into()),
            "f16" | "f32" | "f64" | "bf16" => {
                Type::parse(self.ctx.context(), symbol)
                    .ok_or_else(|| format!("Failed to parse type: {}", symbol))
            }
            _ => Err(format!("Unknown type: {}", symbol)),
        }
    }

    /// Emit a simple function with a single block
    /// Example: (defn main [] i32 ...)
    pub fn emit_function(
        &mut self,
        module: &Module<'c>,
        name: &str,
        _args: &[Value],
        ret_type: Type<'c>,
        body: &[Value],
    ) -> Result<(), String> {
        self.emit_function_with_args(module, name, &[], &[], ret_type, body)
    }

    /// Emit a function with arguments
    pub fn emit_function_with_args(
        &mut self,
        module: &Module<'c>,
        name: &str,
        arg_names: &[String],
        arg_types: &[Type<'c>],
        ret_type: Type<'c>,
        body: &[Value],
    ) -> Result<(), String> {
        let function_type = FunctionType::new(self.ctx.context(), arg_types, &[ret_type]);

        let region = Region::new();

        // Create entry block with arguments
        let block_args: Vec<(Type<'c>, melior::ir::Location<'c>)> = arg_types
            .iter()
            .map(|ty| (*ty, self.ctx.location()))
            .collect();
        let entry_block = Block::new(&block_args);
        region.append_block(entry_block);

        let function = OperationBuilder::new("func.func", self.ctx.location())
            .add_attributes(&[
                (
                    Identifier::new(self.ctx.context(), "sym_name"),
                    StringAttribute::new(self.ctx.context(), name).into(),
                ),
                (
                    Identifier::new(self.ctx.context(), "function_type"),
                    TypeAttribute::new(function_type.into()).into(),
                ),
            ])
            .add_regions([region])
            .build()
            .map_err(|e| format!("Failed to build function: {:?}", e))?;

        // Emit body operations into the entry block
        let function_region = function.region(0)
            .map_err(|_| "Function has no region")?;
        let entry_block = function_region.first_block()
            .ok_or("Function region has no block")?;

        // Register function arguments as values
        for (idx, arg_name) in arg_names.iter().enumerate() {
            if let Ok(arg) = entry_block.argument(idx) {
                self.values.insert(arg_name.clone(), arg.into());
            }
        }

        for expr in body {
            self.emit_operation(&entry_block, expr)?;
        }

        module.body().append_operation(function);
        Ok(())
    }

    /// Emit a function with explicit region/block structure for control flow
    /// Example: (region (block entry [] ...) (block then [] ...) ...)
    pub fn emit_function_with_blocks(
        &mut self,
        module: &Module<'c>,
        name: &str,
        _args: &[Value],
        ret_type: Type<'c>,
        blocks_spec: &[Value],
    ) -> Result<(), String> {
        self.emit_function_with_blocks_and_args(module, name, &[], &[], ret_type, blocks_spec)
    }

    /// Emit a function with arguments and explicit blocks
    pub fn emit_function_with_blocks_and_args(
        &mut self,
        module: &Module<'c>,
        name: &str,
        arg_names: &[String],
        arg_types: &[Type<'c>],
        ret_type: Type<'c>,
        blocks_spec: &[Value],
    ) -> Result<(), String> {
        let function_type = FunctionType::new(self.ctx.context(), arg_types, &[ret_type]);

        let region = Region::new();

        // First pass: create all blocks, add them to region, and register references
        let mut block_indices: HashMap<String, usize> = HashMap::new();
        for (idx, block_spec) in blocks_spec.iter().enumerate() {
            if let Value::List(elements) = block_spec {
                if let Some(Value::Symbol(s)) = elements.get(0) {
                    if s == "block" {
                        if let Some(Value::Symbol(block_name)) = elements.get(1) {
                            // Parse block arguments if present
                            let mut block_args = if let Some(Value::Vector(arg_type_vals)) = elements.get(2) {
                                let mut args = vec![];
                                for arg_type in arg_type_vals {
                                    if let Value::Symbol(type_name) = arg_type {
                                        args.push((self.parse_type(type_name)?, self.ctx.location()));
                                    }
                                }
                                args
                            } else {
                                vec![]
                            };

                            // For the first block, prepend function arguments
                            if idx == 0 {
                                let mut func_args: Vec<(Type<'c>, melior::ir::Location<'c>)> = arg_types
                                    .iter()
                                    .map(|ty| (*ty, self.ctx.location()))
                                    .collect();
                                func_args.extend(block_args);
                                block_args = func_args;
                            }

                            let block = Block::new(&block_args);
                            region.append_block(block);
                            block_indices.insert(block_name.clone(), idx);
                        }
                    }
                }
            }
        }

        // Store block references in our block table for successor lookup
        // Iterate through blocks using first_block() and next_in_region()
        let mut blocks_vec: Vec<BlockRef<'c, 'c>> = vec![];
        let mut current_block = region.first_block();
        while let Some(block) = current_block {
            blocks_vec.push(block);
            current_block = block.next_in_region();
        }

        for (block_name, idx) in &block_indices {
            if let Some(block_ref) = blocks_vec.get(*idx) {
                self.blocks.insert(block_name.clone(), *block_ref);
            }
        }

        // Second pass: emit operations into each block
        for (idx, block_spec) in blocks_spec.iter().enumerate() {
            if let Value::List(elements) = block_spec {
                if let Some(Value::Symbol(s)) = elements.get(0) {
                    if s == "block" {
                        if let Some(block_ref) = blocks_vec.get(idx) {
                            // For the first block (entry), register function arguments
                            if idx == 0 {
                                for (func_arg_idx, arg_name) in arg_names.iter().enumerate() {
                                    if let Ok(arg) = block_ref.argument(func_arg_idx) {
                                        self.values.insert(arg_name.clone(), arg.into());
                                    }
                                }
                            }

                            // Register block arguments as values (^0, ^1, etc.)
                            let mut arg_idx = 0;
                            loop {
                                match block_ref.argument(arg_idx) {
                                    Ok(arg) => {
                                        self.values.insert(format!("^{}", arg_idx), arg.into());
                                        arg_idx += 1;
                                    }
                                    Err(_) => break,
                                }
                            }

                            // Find where operations start (after block name and optional args vector)
                            let ops_start = if elements.get(2).map(|v| matches!(v, Value::Vector(_))).unwrap_or(false) {
                                3
                            } else {
                                2
                            };

                            // Emit operations
                            for op_expr in &elements[ops_start..] {
                                self.emit_operation(&block_ref, op_expr)?;
                            }
                        }
                    }
                }
            }
        }

        let function = OperationBuilder::new("func.func", self.ctx.location())
            .add_attributes(&[
                (
                    Identifier::new(self.ctx.context(), "sym_name"),
                    StringAttribute::new(self.ctx.context(), name).into(),
                ),
                (
                    Identifier::new(self.ctx.context(), "function_type"),
                    TypeAttribute::new(function_type.into()).into(),
                ),
            ])
            .add_regions([region])
            .build()
            .map_err(|e| format!("Failed to build function: {:?}", e))?;

        module.body().append_operation(function);
        Ok(())
    }

    /// Emit a single operation into a block
    /// This is where we'll handle (op arith.constant ...) style syntax
    fn emit_operation(&mut self, block: &Block<'c>, expr: &Value) -> Result<(), String> {
        match expr {
            Value::List(elements) if !elements.is_empty() => {
                match &elements[0] {
                    Value::Symbol(s) if s == "op" => {
                        self.emit_op_form(block, &elements[1..])
                    }
                    Value::Symbol(s) if s == "block" => {
                        Err("block special form not supported in this context".to_string())
                    }
                    Value::Symbol(s) if s == "region" => {
                        Err("region special form not supported in this context".to_string())
                    }
                    Value::Symbol(op_name) if op_name.contains('.') => {
                        // Direct operation form: (arith.constant :value 10 ...)
                        self.emit_direct_op(block, op_name, &elements[1..])
                    }
                    _ => Err(format!("Unknown operation form: {:?}", elements[0])),
                }
            }
            _ => Err(format!("Expected operation list, got: {:?}", expr)),
        }
    }

    /// Emit operation using (op name :attrs {...} :results [...] :as %name) syntax
    fn emit_op_form(&mut self, block: &Block<'c>, args: &[Value]) -> Result<(), String> {
        if args.is_empty() {
            return Err("op form requires operation name".to_string());
        }

        let op_name = match &args[0] {
            Value::Symbol(s) => s,
            _ => return Err("Operation name must be a symbol".to_string()),
        };

        // Parse keyword arguments
        let mut attrs: Vec<(Identifier, melior::ir::Attribute)> = vec![];
        let mut results: Vec<Type> = vec![];
        let mut operands: Vec<melior::ir::Value> = vec![];
        let mut result_name: Option<String> = None;
        let mut successors: Vec<&Block<'c>> = vec![];

        let mut i = 1;
        while i < args.len() {
            match &args[i] {
                Value::Keyword(kw) if kw == "attrs" && i + 1 < args.len() => {
                    // Parse attributes from map
                    if let Value::Map(pairs) = &args[i + 1] {
                        for (k, v) in pairs {
                            if let Value::Keyword(key) = k {
                                attrs.push(self.parse_attribute(key, v)?);
                            }
                        }
                    }
                    i += 2;
                }
                Value::Keyword(kw) if kw == "results" && i + 1 < args.len() => {
                    // Parse result types from vector
                    if let Value::Vector(types) = &args[i + 1] {
                        for t in types {
                            if let Value::Symbol(type_name) = t {
                                results.push(self.parse_type(type_name)?);
                            }
                        }
                    }
                    i += 2;
                }
                Value::Keyword(kw) if kw == "operands" && i + 1 < args.len() => {
                    // Parse operands from vector
                    if let Value::Vector(operand_names) = &args[i + 1] {
                        for op_name in operand_names {
                            if let Value::Symbol(name) = op_name {
                                let val = self.values.get(name)
                                    .ok_or(format!("Undefined value: {}", name))?;
                                operands.push(*val);
                            }
                        }
                    }
                    i += 2;
                }
                Value::Keyword(kw) if kw == "dest" && i + 1 < args.len() => {
                    // Branch destination block
                    if let Value::Symbol(block_name) = &args[i + 1] {
                        let dest_block = self.blocks.get(block_name)
                            .ok_or(format!("Undefined block: {}", block_name))?;
                        successors.push(dest_block);
                    }
                    i += 2;
                }
                Value::Keyword(kw) if kw == "args" && i + 1 < args.len() => {
                    // Branch arguments (operands passed to successor block)
                    // These are handled as regular operands for branch ops
                    if let Value::Vector(arg_names) = &args[i + 1] {
                        for arg_name in arg_names {
                            if let Value::Symbol(name) = arg_name {
                                let val = self.values.get(name)
                                    .ok_or(format!("Undefined value: {}", name))?;
                                operands.push(*val);
                            }
                        }
                    }
                    i += 2;
                }
                Value::Keyword(kw) if kw == "true" && i + 1 < args.len() => {
                    // True branch for cf.cond_br
                    if let Value::Symbol(block_name) = &args[i + 1] {
                        let dest_block = self.blocks.get(block_name)
                            .ok_or(format!("Undefined block: {}", block_name))?;
                        successors.push(dest_block);
                    }
                    i += 2;
                }
                Value::Keyword(kw) if kw == "false" && i + 1 < args.len() => {
                    // False branch for cf.cond_br
                    if let Value::Symbol(block_name) = &args[i + 1] {
                        let dest_block = self.blocks.get(block_name)
                            .ok_or(format!("Undefined block: {}", block_name))?;
                        successors.push(dest_block);
                    }
                    i += 2;
                }
                Value::Keyword(kw) if kw == "as" && i + 1 < args.len() => {
                    if let Value::Symbol(name) = &args[i + 1] {
                        result_name = Some(name.clone());
                    }
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }

        let mut builder = OperationBuilder::new(op_name, self.ctx.location())
            .add_attributes(&attrs)
            .add_operands(&operands)
            .add_results(&results);

        // Add successors for branch operations
        if !successors.is_empty() {
            builder = builder.add_successors(&successors);
        }

        // Special handling for cf.cond_br - needs operandSegmentSizes attribute
        if op_name == "cf.cond_br" {
            use melior::ir::attribute::DenseI32ArrayAttribute;
            let segment_sizes = DenseI32ArrayAttribute::new(
                self.ctx.context(),
                &[operands.len() as i32, 0, 0]
            );
            builder = builder.add_attributes(&[(
                Identifier::new(self.ctx.context(), "operandSegmentSizes"),
                segment_sizes.into(),
            )]);
        }

        let op = builder.build()
            .map_err(|e| format!("Failed to build operation: {:?}", e))?;

        // Store result value if named
        if let Some(name) = result_name {
            if let Ok(result_val) = op.result(0) {
                self.values.insert(name, result_val.into());
            }
        }

        block.append_operation(op);
        Ok(())
    }

    /// Emit operation using direct syntax: (arith.constant :value 10)
    fn emit_direct_op(&mut self, block: &Block<'c>, op_name: &str, args: &[Value]) -> Result<(), String> {
        // For now, just forward to emit_op_form style
        let mut new_args = vec![Value::Symbol(op_name.to_string())];
        new_args.extend_from_slice(args);
        self.emit_op_form(block, &new_args)
    }

    /// Parse an attribute from a keyword-value pair
    fn parse_attribute(&self, key: &str, value: &Value) -> Result<(Identifier<'c>, melior::ir::Attribute<'c>), String> {
        let identifier = Identifier::new(self.ctx.context(), key);

        let attr = match value {
            Value::Integer(n) => {
                // Check for special enum attributes like predicate
                if key == "predicate" {
                    // For arith.cmpi, predicate is an IntegerAttribute with i64 type
                    IntegerAttribute::new(
                        IntegerType::new(self.ctx.context(), 64).into(),
                        *n,
                    ).into()
                } else {
                    // Default to i32 for integer attributes
                    IntegerAttribute::new(
                        IntegerType::new(self.ctx.context(), 32).into(),
                        *n,
                    ).into()
                }
            }
            Value::Float(f) => {
                // Default to f64 for float attributes
                let f64_type = Type::parse(self.ctx.context(), "f64")
                    .ok_or("Failed to parse f64 type")?;
                FloatAttribute::new(
                    self.ctx.context(),
                    f64_type,
                    *f,
                ).into()
            }
            Value::String(s) => {
                // Check for special enum attributes like predicate
                if key == "predicate" {
                    // Map predicate string to integer value
                    let predicate_value = match s.as_str() {
                        "eq" => 0,   // equal
                        "ne" => 1,   // not equal
                        "slt" => 2,  // signed less than
                        "sle" => 3,  // signed less than or equal
                        "sgt" => 4,  // signed greater than
                        "sge" => 5,  // signed greater than or equal
                        "ult" => 6,  // unsigned less than
                        "ule" => 7,  // unsigned less than or equal
                        "ugt" => 8,  // unsigned greater than
                        "uge" => 9,  // unsigned greater than or equal
                        _ => return Err(format!("Unknown predicate: {}", s)),
                    };
                    IntegerAttribute::new(
                        IntegerType::new(self.ctx.context(), 64).into(),
                        predicate_value,
                    ).into()
                } else if key == "callee" {
                    // Function call callee needs FlatSymbolRefAttribute
                    use melior::ir::attribute::FlatSymbolRefAttribute;
                    FlatSymbolRefAttribute::new(self.ctx.context(), s).into()
                } else {
                    StringAttribute::new(self.ctx.context(), s).into()
                }
            }
            Value::Symbol(type_name) => {
                let ty = self.parse_type(type_name)?;
                TypeAttribute::new(ty).into()
            }
            _ => return Err(format!("Unsupported attribute value: {:?}", value)),
        };

        Ok((identifier, attr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_type() {
        let ctx = MlirContext::new();
        let emitter = Emitter::new(&ctx);

        assert!(emitter.parse_type("i32").is_ok());
        assert!(emitter.parse_type("i64").is_ok());
        assert!(emitter.parse_type("foo").is_err());
    }

    #[test]
    fn test_emit_simple_function() {
        let ctx = MlirContext::new();
        let mut emitter = Emitter::new(&ctx);
        let module = ctx.create_module();

        let i32_type = emitter.parse_type("i32").unwrap();

        // Empty body for now
        let result = emitter.emit_function(&module, "test", &[], i32_type, &[]);
        assert!(result.is_ok());
    }
}
