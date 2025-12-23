use std::collections::HashMap;

use melior::{
    dialect::arith,
    ir::{
        attribute::{
            FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, IntegerType},
        Attribute, Block, BlockLike, BlockRef, Identifier, Location, Module, Region, RegionLike,
        Type, TypeLike, Value, ValueLike,
    },
    Context,
};
use thiserror::Error;

use crate::ast::{self, AttributeValue, Node};
use crate::dialect::DialectRegistry;
use crate::value;

#[derive(Debug, Error)]
pub enum GeneratorError {
    #[error("undefined symbol: {0}")]
    UndefinedSymbol(String),

    #[error("type parse error: {0}")]
    TypeParseError(String),

    #[error("invalid operand type")]
    InvalidOperandType,

    #[error("unsupported node type")]
    UnsupportedNodeType,

    #[error("missing result type")]
    MissingResultType,

    #[error("missing type for block argument: {0}")]
    MissingType(String),

    #[error("operation creation failed: {0}")]
    OperationCreationFailed(String),
}

/// Symbol table for tracking SSA values by name
pub struct SymbolTable<'c, 'a> {
    scopes: Vec<HashMap<String, Value<'c, 'a>>>,
}

impl<'c, 'a> SymbolTable<'c, 'a> {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    pub fn define(&mut self, name: impl Into<String>, value: Value<'c, 'a>) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), value);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<Value<'c, 'a>> {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(*value);
            }
        }
        None
    }

    /// Iterate over all symbols in all scopes
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value<'c, 'a>)> {
        self.scopes.iter().flat_map(|scope| scope.iter())
    }
}

impl<'c, 'a> Default for SymbolTable<'c, 'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// IR Generator - converts AST nodes to MLIR IR
pub struct IRGenerator<'c> {
    registry: &'c DialectRegistry,
    location: Location<'c>,
}

impl<'c> IRGenerator<'c> {
    pub fn new(registry: &'c DialectRegistry) -> Self {
        let location = Location::unknown(registry.context());
        Self { registry, location }
    }

    /// Generate MLIR module from AST nodes
    pub fn generate(&self, nodes: &[Node]) -> Result<Module<'c>, GeneratorError> {
        let module = self.registry.create_module();

        {
            let module_body = module.body();
            let mut symbol_table = SymbolTable::new();

            for node in nodes {
                self.generate_node(node, &module_body, &mut symbol_table)?;
            }
        }

        Ok(module)
    }

    /// Verify the generated module against MLIR's verifier
    pub fn verify(&self, module: &Module<'c>) -> bool {
        module.as_operation().verify()
    }

    /// Print the module to a string
    pub fn print_module_to_string(&self, module: &Module<'c>) -> String {
        module.as_operation().to_string()
    }

    /// Generate IR for a single node
    fn generate_node<'a>(
        &self,
        node: &Node,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        match node {
            Node::Module(module) => self.generate_module(module, block, symbol_table),
            Node::Operation(op) => self.generate_operation(op, block, symbol_table),
            Node::Region(_) => Err(GeneratorError::UnsupportedNodeType),
            Node::Block(_) => Err(GeneratorError::UnsupportedNodeType),
            Node::Def(binding) => self.generate_def(binding, block, symbol_table),
            Node::Let(let_expr) => self.generate_let(let_expr, block, symbol_table),
            Node::TypeAnnotation(ta) => {
                self.generate_type_annotation(&ta.value, &ta.typ, block, symbol_table)
            }
            Node::FunctionType(_) => Err(GeneratorError::UnsupportedNodeType),
            Node::Literal(value) => self.generate_literal(value, block, symbol_table),
            Node::Require(_) => {
                // Require nodes are handled at the module loading level, not during IR generation
                Ok(None)
            }
            Node::Compilation(_) => {
                // Compilation nodes are metadata for the JIT runner, not IR
                Ok(None)
            }
        }
    }

    /// Generate module contents
    fn generate_module<'a>(
        &self,
        module: &ast::Module,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        for child in &module.body {
            self.generate_node(child, block, symbol_table)?;
        }
        Ok(None)
    }

    /// Generate an MLIR operation
    fn generate_operation<'a>(
        &self,
        op: &ast::Operation,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        let context = self.registry.context();
        let qualified_name = op.qualified_name();

        // Clone attributes so we can modify them
        let mut attributes = op.attributes.clone();

        // Convert predicate string to integer for comparison operations
        if qualified_name == "arith.cmpi" {
            if let Some(AttributeValue::String(pred_str)) = attributes.get("predicate") {
                let pred_val = match pred_str.as_str() {
                    "eq" => 0,
                    "ne" => 1,
                    "slt" => 2,
                    "sle" => 3,
                    "sgt" => 4,
                    "sge" => 5,
                    "ult" => 6,
                    "ule" => 7,
                    "ugt" => 8,
                    "uge" => 9,
                    _ => 0,
                };
                attributes.insert("predicate".to_string(), AttributeValue::Number(pred_val as f64));
            }
        }
        if qualified_name == "arith.cmpf" {
            if let Some(AttributeValue::String(pred_str)) = attributes.get("predicate") {
                let pred_val = match pred_str.as_str() {
                    "false" => 0,
                    "oeq" => 1,
                    "ogt" => 2,
                    "oge" => 3,
                    "olt" => 4,
                    "ole" => 5,
                    "one" => 6,
                    "ord" => 7,
                    "ueq" => 8,
                    "ugt" => 9,
                    "uge" => 10,
                    "ult" => 11,
                    "ule" => 12,
                    "une" => 13,
                    "uno" => 14,
                    "true" => 15,
                    _ => 0,
                };
                attributes.insert("predicate".to_string(), AttributeValue::Number(pred_val as f64));
            }
        }

        // Remove successors from attributes - they're not MLIR attributes but block references
        // Control flow operations like cf.br, cf.cond_br use successors differently
        attributes.remove("successors");

        // Filter operands - some may be converted to attributes
        let mut filtered_operands: Vec<&Node> = Vec::new();

        for (i, operand_node) in op.operands.iter().enumerate() {
            // Special handling for func.call: first string literal becomes callee attribute
            if qualified_name == "func.call" && i == 0 {
                if let Node::Literal(value::Value::String(s)) = operand_node {
                    // Add @ prefix for symbol reference if not present
                    let callee = if s.starts_with('@') {
                        s.clone()
                    } else {
                        format!("@{}", s)
                    };
                    attributes.insert("callee".to_string(), AttributeValue::String(callee));
                    continue;
                }
            }
            filtered_operands.push(operand_node);
        }

        // First pass: infer type from typed operands
        let mut inferred_type: Option<Type<'c>> = None;
        for operand_node in &filtered_operands {
            if let Some(t) = self.get_operand_type(operand_node, symbol_table)? {
                inferred_type = Some(t);
                break;
            }
        }

        // Generate operands
        let mut operand_values = Vec::new();
        for operand_node in &filtered_operands {
            let value = self.resolve_operand(operand_node, block, symbol_table, inferred_type)?;
            operand_values.push(value);
        }

        // Generate result types
        let mut result_types: Vec<Type<'c>> = Vec::new();
        for result_type in &op.result_types {
            let mlir_type = self.parse_type(&result_type.name)?;
            result_types.push(mlir_type);
        }

        // Check for :result attribute which specifies result type(s)
        if let Some(result_attr) = attributes.remove("result") {
            match result_attr {
                AttributeValue::Type(t) => {
                    result_types.push(self.parse_type(&t.name)?);
                }
                AttributeValue::String(s) => {
                    // Support string as type (e.g., "memref<?xf32>")
                    result_types.push(self.parse_type(&s)?);
                }
                AttributeValue::MLIRLiteral(s) => {
                    // Support MLIR literal types (e.g., memref<10xi64>)
                    result_types.push(self.parse_type(&s)?);
                }
                AttributeValue::Array(arr) => {
                    for item in arr {
                        match item {
                            AttributeValue::Type(t) => {
                                result_types.push(self.parse_type(&t.name)?);
                            }
                            AttributeValue::String(s) => {
                                result_types.push(self.parse_type(&s)?);
                            }
                            AttributeValue::MLIRLiteral(s) => {
                                result_types.push(self.parse_type(&s)?);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // Generate regions for this operation (with parent scope access)
        let mut regions: Vec<Region<'c>> = self.generate_regions_with_scope(&op.regions, symbol_table)?;

        // func.func requires at least one region (even for external declarations)
        if qualified_name == "func.func" && regions.is_empty() {
            regions.push(Region::new());
        }

        // Build the operation
        let operation = self.build_operation(
            context,
            &qualified_name,
            &operand_values,
            &result_types,
            &attributes,
            regions,
            inferred_type,
        )?;

        // Append to block
        let op_ref = block.append_operation(operation);

        // Return first result if any
        if op_ref.result_count() > 0 {
            return Ok(Some(op_ref.result(0).unwrap().into()));
        }

        Ok(None)
    }

    /// Generate regions with access to parent scope
    fn generate_regions_with_scope<'a>(
        &self,
        regions: &[ast::Region],
        parent_scope: &SymbolTable<'c, 'a>,
    ) -> Result<Vec<Region<'c>>, GeneratorError> {
        let mut result = Vec::new();

        for region in regions {
            let mlir_region = self.generate_region(region, Some(parent_scope))?;
            result.push(mlir_region);
        }

        Ok(result)
    }

    /// Generate a region with its blocks (two-pass for successor support)
    fn generate_region<'a>(
        &self,
        region: &ast::Region,
        parent_scope: Option<&SymbolTable<'c, 'a>>,
    ) -> Result<Region<'c>, GeneratorError> {
        use std::collections::HashMap;

        let mlir_region = Region::new();

        // First pass: create all blocks and track labels
        let mut block_map: HashMap<String, usize> = HashMap::new();
        let mut block_refs: Vec<BlockRef<'c, '_>> = Vec::new();
        let mut block_args: Vec<Vec<(String, Type<'c>)>> = Vec::new();

        for (idx, block) in region.blocks.iter().enumerate() {
            // Prepare block argument types and locations
            let mut arg_types_locs: Vec<(Type<'c>, Location<'c>)> = Vec::new();
            let mut arg_names: Vec<(String, Type<'c>)> = Vec::new();

            for arg in &block.arguments {
                if let Some(ref t) = arg.typ {
                    let mlir_type = self.parse_type(&t.name)?;
                    arg_types_locs.push((mlir_type, self.location));
                    arg_names.push((arg.name.clone(), mlir_type));
                } else {
                    return Err(GeneratorError::MissingType(arg.name.clone()));
                }
            }

            // Create block with arguments
            let mlir_block = Block::new(&arg_types_locs);
            let block_ref = mlir_region.append_block(mlir_block);

            // Track label if present
            if let Some(ref label) = block.label {
                block_map.insert(label.clone(), idx);
            }

            block_refs.push(block_ref);
            block_args.push(arg_names);
        }

        // Second pass: generate operations with access to all blocks
        for (idx, block) in region.blocks.iter().enumerate() {
            let block_ref = &block_refs[idx];

            // Create symbol table for this block, inheriting from parent if present
            let mut symbol_table: SymbolTable<'c, '_> = SymbolTable::new();

            // Copy symbols from parent scope if available
            if let Some(parent) = parent_scope {
                for (name, value) in parent.iter() {
                    symbol_table.define(name, *value);
                }
            }

            // Register block arguments (may shadow parent symbols)
            for (i, (name, _)) in block_args[idx].iter().enumerate() {
                if let Ok(block_arg) = block_ref.argument(i) {
                    symbol_table.define(name, block_arg.into());
                }
            }

            // Generate operations with block map for successors
            for op_node in &block.operations {
                self.generate_node_with_blocks(
                    op_node,
                    block_ref,
                    &mut symbol_table,
                    &block_map,
                    &block_refs,
                )?;
            }
        }

        Ok(mlir_region)
    }

    /// Generate a node with access to block map for successor references
    fn generate_node_with_blocks<'a>(
        &self,
        node: &Node,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
        block_map: &std::collections::HashMap<String, usize>,
        block_refs: &[BlockRef<'c, '_>],
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        match node {
            Node::Operation(op) => {
                self.generate_operation_with_blocks(op, block, symbol_table, block_map, block_refs)
            }
            _ => self.generate_node(node, block, symbol_table),
        }
    }

    /// Generate an operation with block successor support
    fn generate_operation_with_blocks<'a>(
        &self,
        op: &ast::Operation,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
        block_map: &std::collections::HashMap<String, usize>,
        block_refs: &[BlockRef<'c, '_>],
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        let context = self.registry.context();
        let qualified_name = op.qualified_name();

        // Check if this is a control flow operation with block successors
        let is_cf_op = qualified_name == "cf.br"
            || qualified_name == "cf.cond_br"
            || qualified_name == "scf.yield";

        if !is_cf_op {
            // Regular operation
            return self.generate_operation(op, block, symbol_table);
        }

        // Clone attributes
        let mut attributes = op.attributes.clone();
        attributes.remove("successors");

        // Parse operands, separating block labels from values
        let mut operand_values: Vec<Value<'c, 'a>> = Vec::new();
        let mut successor_blocks: Vec<(BlockRef<'c, '_>, Vec<Value<'c, 'a>>)> = Vec::new();
        let mut current_successor_args: Vec<Value<'c, 'a>> = Vec::new();
        let mut in_successor = false;
        let mut current_successor_label: Option<String> = None;

        for operand_node in &op.operands {
            // Check if it's a block label
            if let Node::Literal(value::Value::Symbol(sym)) = operand_node {
                if sym.name.starts_with('^') {
                    // Flush previous successor if any
                    if let Some(label) = current_successor_label.take() {
                        if let Some(&idx) = block_map.get(&label) {
                            let br = block_refs[idx];
                            successor_blocks.push((br, std::mem::take(&mut current_successor_args)));
                        }
                    }
                    // Start new successor
                    current_successor_label = Some(sym.name.clone());
                    in_successor = true;
                    continue;
                }
            }

            // Resolve the operand value
            let value = self.resolve_operand(operand_node, block, symbol_table, None)?;

            if in_successor {
                current_successor_args.push(value);
            } else {
                operand_values.push(value);
            }
        }

        // Flush last successor
        if let Some(label) = current_successor_label.take() {
            if let Some(&idx) = block_map.get(&label) {
                let br = block_refs[idx];
                successor_blocks.push((br, std::mem::take(&mut current_successor_args)));
            }
        }

        // Build the operation
        let mut op_builder = OperationBuilder::new(&qualified_name, self.location);

        if !operand_values.is_empty() {
            op_builder = op_builder.add_operands(&operand_values);
        }

        // Add successor blocks with their arguments
        for (succ_block, succ_args) in &successor_blocks {
            op_builder = op_builder.add_successors(&[succ_block]);
            if !succ_args.is_empty() {
                op_builder = op_builder.add_operands(succ_args);
            }
        }

        // Add attributes
        for (key, value) in &attributes {
            let attr = self.convert_attribute_value(value)?;
            op_builder =
                op_builder.add_attributes(&[(Identifier::new(context, key), attr)]);
        }

        let operation = op_builder.build().map_err(|e| {
            GeneratorError::OperationCreationFailed(format!("{}: {:?}", qualified_name, e))
        })?;

        let op_ref = block.append_operation(operation);

        if op_ref.result_count() > 0 {
            return Ok(Some(op_ref.result(0).unwrap().into()));
        }

        Ok(None)
    }

    /// Build an MLIR operation
    fn build_operation(
        &self,
        context: &'c Context,
        name: &str,
        operands: &[Value<'c, '_>],
        result_types: &[Type<'c>],
        attributes: &HashMap<String, AttributeValue>,
        regions: Vec<Region<'c>>,
        inferred_type: Option<Type<'c>>,
    ) -> Result<melior::ir::Operation<'c>, GeneratorError> {
        let location = self.location;

        // Convert attributes
        let mut named_attrs: Vec<(Identifier<'c>, Attribute<'c>)> = Vec::new();
        for (key, value) in attributes {
            let attr = self.convert_attribute_value(value)?;
            named_attrs.push((Identifier::new(context, key), attr));
        }

        // Check if operation supports type inference using MLIR's InferTypeOpInterface
        let supports_inference = context.operation_supports_type_inference(name);

        // Determine result types - use inference if no explicit types, no regions, and operation supports it
        let final_result_types: Vec<Type<'c>> = if result_types.is_empty() && regions.is_empty() && supports_inference {
            // Try to use inferred type for operations that produce results
            if let Some(t) = inferred_type {
                vec![t]
            } else {
                vec![]
            }
        } else {
            result_types.to_vec()
        };

        // Check if we should enable type inference
        let enable_inference =
            result_types.is_empty() && regions.is_empty() && supports_inference;

        // Build operation state
        let mut op_builder = OperationBuilder::new(name, location);

        if !operands.is_empty() {
            op_builder = op_builder.add_operands(operands);
        }

        if !final_result_types.is_empty() {
            op_builder = op_builder.add_results(&final_result_types);
        }

        for (id, attr) in named_attrs {
            op_builder = op_builder.add_attributes(&[(id, attr)]);
        }

        // Add regions
        if !regions.is_empty() {
            op_builder = op_builder.add_regions_vec(regions);
        }

        // Enable result type inference for certain operations
        if enable_inference {
            op_builder = op_builder.enable_result_type_inference();
        }

        op_builder
            .build()
            .map_err(|e| GeneratorError::OperationCreationFailed(format!("{}: {:?}", name, e)))
    }

    /// Generate def (variable binding)
    fn generate_def<'a>(
        &self,
        binding: &ast::Binding,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        let value = self.generate_node(&binding.value, block, symbol_table)?;

        if let Some(v) = value {
            if binding.names.len() == 1 {
                symbol_table.define(&binding.names[0], v);
                return Ok(Some(v));
            } else {
                // Multi-result destructuring
                // Get the last operation and its results
                if let Some(terminator) = block.terminator() {
                    // Use the terminator to find last op
                    let _ = terminator;
                }

                // Actually, we need to get the defining operation of v
                // Since v is a Value, we can iterate back through the block
                let mut last_op = None;
                let mut iter = block.first_operation();
                while let Some(op) = iter {
                    last_op = Some(op);
                    iter = op.next_in_block();
                }

                if let Some(op) = last_op {
                    if op.result_count() as usize != binding.names.len() {
                        return Err(GeneratorError::InvalidOperandType);
                    }

                    for (i, name) in binding.names.iter().enumerate() {
                        if let Ok(result) = op.result(i) {
                            symbol_table.define(name, result.into());
                        }
                    }
                }
                return Ok(Some(v));
            }
        }

        Ok(None)
    }

    /// Generate let expression
    fn generate_let<'a>(
        &self,
        let_expr: &ast::LetExpr,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        symbol_table.push_scope();

        // Generate bindings
        for binding in &let_expr.bindings {
            self.generate_def(binding, block, symbol_table)?;
        }

        // Generate body, return last value
        let mut last_value = None;
        for body_node in &let_expr.body {
            last_value = self.generate_node(body_node, block, symbol_table)?;
        }

        symbol_table.pop_scope();
        Ok(last_value)
    }

    /// Generate type annotation
    fn generate_type_annotation<'a>(
        &self,
        value_node: &Node,
        typ: &ast::Type,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        if let Node::Literal(value::Value::Number(n)) = value_node {
            return self.generate_constant(*n, &typ.name, block);
        }

        self.generate_node(value_node, block, symbol_table)
    }

    /// Generate a constant operation
    fn generate_constant<'a>(
        &self,
        value: f64,
        type_name: &str,
        block: &'a BlockRef<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        let context = self.registry.context();
        let mlir_type = self.parse_type(type_name)?;

        let attr = self.create_number_attribute(value, mlir_type);

        let operation = arith::constant(context, attr, self.location);
        let op_ref = block.append_operation(operation);

        if op_ref.result_count() > 0 {
            return Ok(Some(op_ref.result(0).unwrap().into()));
        }

        Ok(None)
    }

    /// Generate literal (symbol reference)
    fn generate_literal<'a>(
        &self,
        value: &value::Value,
        _block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
    ) -> Result<Option<Value<'c, 'a>>, GeneratorError> {
        match value {
            value::Value::Symbol(sym) => symbol_table
                .lookup(&sym.name)
                .map(Some)
                .ok_or_else(|| GeneratorError::UndefinedSymbol(sym.name.clone())),
            value::Value::List(_) => {
                // Lists that made it to IR generation are compiler directives
                Ok(None)
            }
            _ => Err(GeneratorError::UnsupportedNodeType),
        }
    }

    /// Get the type of an operand without generating code
    fn get_operand_type<'a>(
        &self,
        node: &Node,
        symbol_table: &SymbolTable<'c, 'a>,
    ) -> Result<Option<Type<'c>>, GeneratorError> {
        match node {
            Node::Literal(value::Value::Symbol(sym)) => {
                if let Some(value) = symbol_table.lookup(&sym.name) {
                    Ok(Some(value.r#type()))
                } else {
                    Ok(None)
                }
            }
            Node::TypeAnnotation(ta) => {
                let t = self.parse_type(&ta.typ.name)?;
                Ok(Some(t))
            }
            _ => Ok(None),
        }
    }

    /// Resolve an operand node to an MLIR value
    fn resolve_operand<'a>(
        &self,
        node: &Node,
        block: &'a BlockRef<'c, 'a>,
        symbol_table: &mut SymbolTable<'c, 'a>,
        inferred_type: Option<Type<'c>>,
    ) -> Result<Value<'c, 'a>, GeneratorError> {
        match node {
            Node::Literal(value::Value::Symbol(sym)) => symbol_table
                .lookup(&sym.name)
                .ok_or_else(|| GeneratorError::UndefinedSymbol(sym.name.clone())),
            Node::Literal(value::Value::Number(n)) => {
                // Bare number - use inferred type or default to i64
                let type_name = if let Some(t) = inferred_type {
                    type_to_string(&t)
                } else {
                    "i64".to_string()
                };
                self.generate_constant(*n, &type_name, block)?
                    .ok_or(GeneratorError::InvalidOperandType)
            }
            Node::TypeAnnotation(ta) => {
                if let Node::Literal(value::Value::Number(n)) = &*ta.value {
                    self.generate_constant(*n, &ta.typ.name, block)?
                        .ok_or(GeneratorError::InvalidOperandType)
                } else {
                    self.resolve_operand(&ta.value, block, symbol_table, inferred_type)
                }
            }
            Node::Operation(op) => {
                let result = self.generate_operation(op, block, symbol_table)?;
                result.ok_or(GeneratorError::InvalidOperandType)
            }
            _ => Err(GeneratorError::InvalidOperandType),
        }
    }

    /// Parse a type string into MlirType
    fn parse_type(&self, type_name: &str) -> Result<Type<'c>, GeneratorError> {
        let context = self.registry.context();

        // Handle common built-in types
        match type_name {
            "i1" => Ok(IntegerType::new(context, 1).into()),
            "i8" => Ok(IntegerType::new(context, 8).into()),
            "i16" => Ok(IntegerType::new(context, 16).into()),
            "i32" => Ok(IntegerType::new(context, 32).into()),
            "i64" => Ok(IntegerType::new(context, 64).into()),
            "f16" => Ok(Type::float16(context)),
            "f32" => Ok(Type::float32(context)),
            "f64" => Ok(Type::float64(context)),
            "index" => Ok(Type::index(context)),
            _ => {
                // For complex types, use parser
                Type::parse(context, type_name)
                    .ok_or_else(|| GeneratorError::TypeParseError(type_name.to_string()))
            }
        }
    }

    /// Create a number attribute for the given value and type
    fn create_number_attribute(&self, value: f64, mlir_type: Type<'c>) -> Attribute<'c> {
        let context = self.registry.context();

        if mlir_type.is_integer() {
            IntegerAttribute::new(mlir_type, value as i64).into()
        } else if mlir_type.is_f16() || mlir_type.is_f32() || mlir_type.is_f64() {
            FloatAttribute::new(context, mlir_type, value).into()
        } else if mlir_type.is_index() {
            IntegerAttribute::new(mlir_type, value as i64).into()
        } else {
            // Default to integer
            IntegerAttribute::new(mlir_type, value as i64).into()
        }
    }

    /// Convert AST AttributeValue to MlirAttribute
    fn convert_attribute_value(
        &self,
        value: &AttributeValue,
    ) -> Result<Attribute<'c>, GeneratorError> {
        let context = self.registry.context();

        match value {
            AttributeValue::String(s) => {
                // Strings starting with @ are symbol references
                if s.starts_with('@') {
                    Ok(FlatSymbolRefAttribute::new(context, &s[1..]).into())
                } else {
                    // Plain strings become StringAttr
                    // For MLIR attribute literals, use (: literal type) syntax
                    Ok(StringAttribute::new(context, s).into())
                }
            }
            AttributeValue::Number(n) => {
                // Check if this is an integer value
                if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                    let i64_type = IntegerType::new(context, 64);
                    Ok(IntegerAttribute::new(i64_type.into(), *n as i64).into())
                } else {
                    let f64_type = Type::float64(context);
                    Ok(FloatAttribute::new(context, f64_type, *n).into())
                }
            }
            AttributeValue::Boolean(b) => {
                let i1_type = IntegerType::new(context, 1);
                Ok(IntegerAttribute::new(i1_type.into(), *b as i64).into())
            }
            AttributeValue::Type(t) => {
                let mlir_type = self.parse_type(&t.name)?;
                Ok(TypeAttribute::new(mlir_type).into())
            }
            AttributeValue::FunctionType(ft) => {
                let mut input_types = Vec::new();
                let mut result_types = Vec::new();

                for t in &ft.arg_types {
                    input_types.push(self.parse_type(&t.name)?);
                }
                for t in &ft.return_types {
                    result_types.push(self.parse_type(&t.name)?);
                }

                let func_type = FunctionType::new(context, &input_types, &result_types);
                Ok(TypeAttribute::new(func_type.into()).into())
            }
            AttributeValue::Array(arr) => {
                let mut attrs: Vec<Attribute<'c>> = Vec::new();
                for item in arr {
                    attrs.push(self.convert_attribute_value(item)?);
                }
                Ok(melior::ir::attribute::ArrayAttribute::new(context, &attrs).into())
            }
            AttributeValue::TypedNumber(tn) => {
                let mlir_type = self.parse_type(&tn.typ.name)?;
                Ok(self.create_number_attribute(tn.value, mlir_type))
            }
            AttributeValue::TypedMLIRLiteral(tl) => {
                // Construct "literal : type" and parse as MLIR attribute
                let attr_str = format!("{} : {}", tl.literal, tl.typ.name);
                Attribute::parse(context, &attr_str)
                    .ok_or_else(|| GeneratorError::TypeParseError(format!("Invalid typed MLIR attribute: {}", attr_str)))
            }
            AttributeValue::MLIRLiteral(s) => {
                // Try to parse as MLIR attribute literal (array<i32: ...>, dense<...>, etc.)
                if let Some(attr) = Attribute::parse(context, s) {
                    Ok(attr)
                } else {
                    // Fall back to TypeAttribute for things like memref<128xf32>
                    // (MLIR types that should be wrapped as attributes)
                    let mlir_type = self.parse_type(s)?;
                    Ok(TypeAttribute::new(mlir_type).into())
                }
            }
        }
    }
}


/// Convert a Type to a string representation
fn type_to_string(t: &Type) -> String {
    format!("{}", t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Parser, Reader, Tokenizer};

    #[test]
    fn test_ir_generator_empty() {
        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);

        let nodes: Vec<Node> = vec![];
        let module = generator.generate(&nodes).unwrap();

        // Module should be valid
        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_dialect_registry_creation() {
        let registry = DialectRegistry::new();
        let _context = registry.context();
    }

    #[test]
    fn test_ir_generator_creation() {
        let registry = DialectRegistry::new();
        let _generator = IRGenerator::new(&registry);
    }

    #[test]
    fn test_symbol_table_basic_operations() {
        // Test that symbol table can define and lookup values
        // We test this by compiling code that uses variables
        let source = "(require-dialect arith) (module (let [x (: 42 i32)] (arith.addi x x)))";

        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let nodes = parser.parse(&values).unwrap();

        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let module = generator.generate(&nodes).unwrap();

        // Module should be valid - this means symbol lookup worked
        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_symbol_table_scoping() {
        // Test that inner scopes can shadow outer scope variables
        let source = r#"
            (require-dialect arith)
            (module
              (let [x (: 10 i32)]
                (let [x (: 20 i32)]
                  (arith.addi x x))))
        "#;

        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let nodes = parser.parse(&values).unwrap();

        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let module = generator.generate(&nodes).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_symbol_table_nested_scope_lookup() {
        // Test that inner scopes can see outer scope variables
        let source = r#"
            (require-dialect arith)
            (module
              (let [outer (: 10 i32)]
                (let [inner (: 20 i32)]
                  (arith.addi outer inner))))
        "#;

        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let nodes = parser.parse(&values).unwrap();

        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let module = generator.generate(&nodes).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_compile_simple_expression() {
        // Integration test: compile a simple arith expression
        let source = "(require-dialect arith) (module (arith.addi (: 1 i32) (: 2 i32)))";

        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let nodes = parser.parse(&values).unwrap();

        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let module = generator.generate(&nodes).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_undefined_symbol_error() {
        // Test that using an undefined symbol produces an error
        let source = "(require-dialect arith) (module (arith.addi undefined_var (: 1 i32)))";

        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let nodes = parser.parse(&values).unwrap();

        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let result = generator.generate(&nodes);

        assert!(result.is_err());
        match result {
            Err(GeneratorError::UndefinedSymbol(name)) => {
                assert_eq!(name, "undefined_var");
            }
            _ => panic!("Expected UndefinedSymbol error"),
        }
    }

    #[test]
    fn test_operation_with_region() {
        // Test an operation with a region (func.func)
        let source = r#"
            (require-dialect func)
            (module
              (func.func {:sym_name "test"
                          :function_type (-> [] [i32])}
                (region
                  (block []
                    (func.return (: 42 i32))))))
        "#;

        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let nodes = parser.parse(&values).unwrap();

        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let module = generator.generate(&nodes).unwrap();

        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_def_in_function_block() {
        // Test def inside a function block - variable should be available for later use
        // This matches the pattern in memory.lsp where def is used to bind alloc result
        let source = r#"
            (require-dialect [memref :as m])
            (require-dialect [func :as f])
            (module
              (f/func {:sym_name "test"
                       :function_type (-> [] [memref<10xi64>])}
                (region
                  (block []
                    (def buffer (m/alloc {:result memref<10xi64>}))
                    (f/return buffer)))))
        "#;

        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let nodes = parser.parse(&values).unwrap();

        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let module = generator.generate(&nodes).unwrap();

        assert!(module.as_operation().verify());
    }
}
