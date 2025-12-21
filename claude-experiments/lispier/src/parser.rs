use thiserror::Error;

use crate::ast::{
    AttributeValue, Binding, Block, BlockArgument, Compilation, FunctionType, LetExpr, Module,
    Node, Operation, Pass, Region, Require, Target, Type, TypeAnnotation, TypedNumber,
};
use crate::value::Value;

#[derive(Debug, Error)]
pub enum ParserError {
    #[error("expected symbol")]
    ExpectedSymbol,

    #[error("invalid def form")]
    InvalidDefForm,

    #[error("invalid def name")]
    InvalidDefName,

    #[error("invalid let form")]
    InvalidLetForm,

    #[error("let bindings must be a vector")]
    LetBindingsMustBeVector,

    #[error("invalid type annotation")]
    InvalidTypeAnnotation,

    #[error("invalid type")]
    InvalidType,

    #[error("invalid function type")]
    InvalidFunctionType,

    #[error("invalid require form - expected (require [\"path\" :as alias])")]
    InvalidRequireForm,

    #[error("invalid compilation form - expected (compilation (target ...) ...)")]
    InvalidCompilationForm,

    #[error("invalid target form - expected (target backend (pass ...) ...)")]
    InvalidTargetForm,

    #[error("invalid pass form - expected (pass name) or (pass name {{:attr val}})")]
    InvalidPassForm,
}

pub struct Parser;

impl Parser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a list of reader values into AST nodes
    pub fn parse(&mut self, values: &[Value]) -> Result<Vec<Node>, ParserError> {
        let mut nodes = Vec::new();

        for value in values {
            let node = self.parse_value(value)?;
            nodes.push(node);
        }

        Ok(nodes)
    }

    fn parse_value(&mut self, value: &Value) -> Result<Node, ParserError> {
        match value {
            Value::List(items) => self.parse_list(items, value),
            Value::Symbol(_) => Ok(Node::literal(value.clone())),
            Value::Number(_)
            | Value::String(_)
            | Value::Boolean(_)
            | Value::Nil
            | Value::Keyword(_) => Ok(Node::literal(value.clone())),
            Value::Vector(_) => Ok(Node::literal(value.clone())),
            Value::Map(_) => Ok(Node::literal(value.clone())),
        }
    }

    fn parse_list(&mut self, items: &[Value], original: &Value) -> Result<Node, ParserError> {
        if items.is_empty() {
            return Ok(Node::literal(original.clone()));
        }

        let first = &items[0];
        let sym = match first {
            Value::Symbol(s) => s,
            _ => return Err(ParserError::ExpectedSymbol),
        };

        // Check for special forms
        match sym.name.as_str() {
            "module" => self.parse_module(items),
            "do" | "region" => self.parse_region(items),
            "block" => self.parse_block(items),
            "def" => self.parse_def(items),
            "let" => self.parse_let(items),
            ":" => self.parse_type_annotation(items),
            "->" => self.parse_function_type(items),
            "require-dialect" | "use-dialect" => {
                // These are processed during reading, just return a literal
                Ok(Node::literal(original.clone()))
            }
            "require" => self.parse_require(items),
            "compilation" => self.parse_compilation(items),
            _ => self.parse_operation(items),
        }
    }

    fn parse_module(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        // (module (require-dialect ...) (do ...))
        let mut module = Module::new();

        // Parse the module body
        for item in items.iter().skip(1) {
            if let Value::List(list_items) = item {
                if !list_items.is_empty() {
                    if let Value::Symbol(first_sym) = &list_items[0] {
                        // Skip require-dialect and use-dialect
                        if first_sym.name == "require-dialect" || first_sym.name == "use-dialect" {
                            continue;
                        }
                        // Flatten do blocks at module level (they're grouping, not regions)
                        if first_sym.name == "do" {
                            for inner_item in list_items.iter().skip(1) {
                                // Skip nested require-dialect
                                if let Value::List(inner_list) = inner_item {
                                    if !inner_list.is_empty() {
                                        if let Value::Symbol(inner_sym) = &inner_list[0] {
                                            if inner_sym.name == "require-dialect"
                                                || inner_sym.name == "use-dialect"
                                            {
                                                continue;
                                            }
                                        }
                                    }
                                }
                                let node = self.parse_value(inner_item)?;
                                module.body.push(node);
                            }
                            continue;
                        }
                    }
                }
            }

            let node = self.parse_value(item)?;
            module.body.push(node);
        }

        Ok(Node::module(module))
    }

    fn parse_require(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        // (require ["./file.lisp" :as m])
        if items.len() < 2 {
            return Err(ParserError::InvalidRequireForm);
        }

        if let Value::Vector(spec) = &items[1] {
            if spec.len() >= 3 {
                if let (Value::String(path), Value::Keyword(kw), Value::Symbol(alias)) =
                    (&spec[0], &spec[1], &spec[2])
                {
                    if kw == ":as" {
                        let is_project_relative = path.starts_with('@');
                        return Ok(Node::require(Require {
                            path: path.clone(),
                            alias: alias.name.clone(),
                            is_project_relative,
                        }));
                    }
                }
            }
        }

        Err(ParserError::InvalidRequireForm)
    }

    fn parse_compilation(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        // (compilation (target rocm ...) (target cuda ...) ...)
        let mut compilation = Compilation::new();

        for item in items.iter().skip(1) {
            if let Value::List(list_items) = item {
                if !list_items.is_empty() {
                    if let Value::Symbol(first) = &list_items[0] {
                        if first.name == "target" {
                            let target = self.parse_target(list_items)?;
                            compilation.targets.push(target);
                            continue;
                        }
                    }
                }
            }
            return Err(ParserError::InvalidCompilationForm);
        }

        Ok(Node::compilation(compilation))
    }

    fn parse_target(&mut self, items: &[Value]) -> Result<Target, ParserError> {
        // (target rocm (pass gpu-kernel-outlining) (pass rocdl-attach-target) ...)
        if items.len() < 2 {
            return Err(ParserError::InvalidTargetForm);
        }

        let backend = match &items[1] {
            Value::Symbol(sym) => sym.name.clone(),
            _ => return Err(ParserError::InvalidTargetForm),
        };

        let mut target = Target::new(backend);

        for item in items.iter().skip(2) {
            if let Value::List(list_items) = item {
                if !list_items.is_empty() {
                    if let Value::Symbol(first) = &list_items[0] {
                        if first.name == "pass" {
                            let pass = self.parse_pass(list_items)?;
                            target.passes.push(pass);
                            continue;
                        }
                    }
                }
            }
            return Err(ParserError::InvalidTargetForm);
        }

        Ok(target)
    }

    fn parse_pass(&mut self, items: &[Value]) -> Result<Pass, ParserError> {
        // (pass gpu-kernel-outlining) or (pass rocdl-attach-target {:chip "gfx1151"})
        if items.len() < 2 {
            return Err(ParserError::InvalidPassForm);
        }

        let name = match &items[1] {
            Value::Symbol(sym) => sym.name.clone(),
            _ => return Err(ParserError::InvalidPassForm),
        };

        let mut pass = Pass::new(name);

        // Check for optional attributes map
        if items.len() >= 3 {
            if let Value::Map(map) = &items[2] {
                for (key, val) in map {
                    let val_str = match val {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        Value::Symbol(sym) => sym.name.clone(),
                        _ => continue,
                    };
                    pass.attributes.insert(key.clone(), val_str);
                }
            }
        }

        Ok(pass)
    }

    fn parse_operation(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        let first = &items[0];
        let sym = match first {
            Value::Symbol(s) => s,
            _ => return Err(ParserError::ExpectedSymbol),
        };

        // Get namespace if present
        let mut op = if let Some(ref ns) = sym.namespace {
            Operation::with_namespace(&sym.name, &ns.name)
        } else {
            Operation::new(&sym.name)
        };

        let mut i = 1;

        // Check for attributes map (second element if it's a map)
        if i < items.len() {
            if let Value::Map(_) = &items[i] {
                self.parse_attributes(&items[i], &mut op.attributes)?;
                i += 1;
            }
        }

        // Parse operands and regions
        while i < items.len() {
            let item = &items[i];

            // Check if it's a region (do or region block)
            if let Value::List(list_items) = item {
                if !list_items.is_empty() {
                    if let Value::Symbol(region_first) = &list_items[0] {
                        if region_first.name == "do" || region_first.name == "region" {
                            let region_node = self.parse_region(list_items)?;
                            if let Node::Region(region) = region_node {
                                op.regions.push(region);
                                i += 1;
                                continue;
                            }
                        }
                    }
                }
            }

            // Otherwise, it's an operand
            let node = self.parse_value(item)?;
            op.operands.push(node);
            i += 1;
        }

        Ok(Node::operation(op))
    }

    fn parse_attributes(
        &mut self,
        value: &Value,
        attributes: &mut std::collections::HashMap<String, AttributeValue>,
    ) -> Result<(), ParserError> {
        if let Value::Map(map) = value {
            for (key, val) in map {
                let attr_value = self.parse_attribute_value(val)?;
                attributes.insert(key.clone(), attr_value);
            }
        }
        Ok(())
    }

    fn parse_attribute_value(&mut self, value: &Value) -> Result<AttributeValue, ParserError> {
        match value {
            Value::String(s) => Ok(AttributeValue::String(s.clone())),
            Value::Number(n) => Ok(AttributeValue::Number(*n)),
            Value::Boolean(b) => Ok(AttributeValue::Boolean(*b)),
            Value::Vector(items) => {
                let mut arr = Vec::new();
                for item in items {
                    let attr_val = self.parse_attribute_value(item)?;
                    arr.push(attr_val);
                }
                Ok(AttributeValue::Array(arr))
            }
            Value::Symbol(sym) => {
                // Treat as a type
                Ok(AttributeValue::Type(Type::new(&sym.name)))
            }
            Value::List(items) => {
                // Check if it's a function type or type annotation
                if !items.is_empty() {
                    if let Value::Symbol(first_sym) = &items[0] {
                        if first_sym.name == "->" {
                            let ft = self.parse_function_type_value(items)?;
                            return Ok(AttributeValue::FunctionType(ft));
                        }
                        // Type annotation: (: value type)
                        if first_sym.name == ":" && items.len() == 3 {
                            if let (Value::Number(n), Value::Symbol(typ_sym)) =
                                (&items[1], &items[2])
                            {
                                return Ok(AttributeValue::TypedNumber(TypedNumber {
                                    value: *n,
                                    typ: Type::new(&typ_sym.name),
                                }));
                            }
                        }
                    }
                }
                Ok(AttributeValue::String("<list>".to_string()))
            }
            _ => Ok(AttributeValue::String("<unknown>".to_string())),
        }
    }

    fn parse_region(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        let mut region = Region::new();

        // Parse blocks in the region
        for item in items.iter().skip(1) {
            if let Value::List(list_items) = item {
                if !list_items.is_empty() {
                    if let Value::Symbol(first) = &list_items[0] {
                        if first.name == "block" {
                            let block_node = self.parse_block(list_items)?;
                            if let Node::Block(block) = block_node {
                                region.blocks.push(block);
                                continue;
                            }
                        }
                    }
                }
            }

            // If not a block, treat as an operation in an implicit block
            if region.blocks.is_empty() {
                let implicit_block = Block::new();
                region.blocks.push(implicit_block);
            }

            let node = self.parse_value(item)?;
            let last_block = region.blocks.last_mut().unwrap();
            last_block.operations.push(node);
        }

        Ok(Node::region(region))
    }

    fn parse_block(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        let mut label: Option<String> = None;
        let mut arg_start = 1;
        let mut has_label = false;

        // Check for label
        if items.len() > 1 {
            if let Value::Symbol(label_sym) = &items[1] {
                if label_sym.name.starts_with('^') {
                    label = Some(label_sym.name.clone());
                    arg_start = 2;
                    has_label = true;
                }
            }
        }

        let mut block = match label {
            Some(l) => Block::with_label(l),
            None => Block::new(),
        };

        // Parse arguments (if present as a vector)
        let args_index = arg_start;
        if args_index < items.len() {
            if let Value::Vector(args_vec) = &items[args_index] {
                for arg_item in args_vec {
                    // Arguments can be symbols or type annotations
                    match arg_item {
                        Value::Symbol(sym) => {
                            block.arguments.push(BlockArgument::new(&sym.name));
                        }
                        Value::List(list_items) => {
                            // (: name type)
                            if list_items.len() >= 3 {
                                if let Value::Symbol(first) = &list_items[0] {
                                    if first.name == ":" {
                                        if let Value::Symbol(name_sym) = &list_items[1] {
                                            if let Value::Symbol(type_sym) = &list_items[2] {
                                                block.arguments.push(BlockArgument::with_type(
                                                    &name_sym.name,
                                                    Type::new(&type_sym.name),
                                                ));
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => continue,
                    }
                }
                arg_start = args_index + 1;
            } else {
                arg_start = if has_label { args_index } else { 1 };
            }
        }

        // Parse operations in the block
        for item in items.iter().skip(arg_start) {
            let node = self.parse_value(item)?;
            block.operations.push(node);
        }

        Ok(Node::block(block))
    }

    fn parse_def(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        // (def name value)
        if items.len() < 3 {
            return Err(ParserError::InvalidDefForm);
        }

        let mut names = Vec::new();

        // Get name
        match &items[1] {
            Value::Symbol(sym) => {
                names.push(sym.name.clone());
            }
            Value::Vector(vec) => {
                // Destructuring: (def [a b] ...)
                for item in vec {
                    if let Value::Symbol(sym) = item {
                        names.push(sym.name.clone());
                    }
                }
            }
            _ => return Err(ParserError::InvalidDefName),
        }

        // Parse value
        let value = self.parse_value(&items[2])?;

        Ok(Node::def(Binding::new(names, value)))
    }

    fn parse_let(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        // (let [bindings...] body...)
        if items.len() < 3 {
            return Err(ParserError::InvalidLetForm);
        }

        let mut let_expr = LetExpr::new();

        // Parse bindings
        let bindings_vec = match &items[1] {
            Value::Vector(vec) => vec,
            _ => return Err(ParserError::LetBindingsMustBeVector),
        };

        if bindings_vec.len() % 2 != 0 {
            return Err(ParserError::InvalidLetForm);
        }

        let mut i = 0;
        while i < bindings_vec.len() {
            let name_val = &bindings_vec[i];
            let value_val = &bindings_vec[i + 1];

            let mut names = Vec::new();

            match name_val {
                Value::Symbol(sym) => {
                    names.push(sym.name.clone());
                }
                Value::Vector(vec) => {
                    for item in vec {
                        if let Value::Symbol(sym) = item {
                            names.push(sym.name.clone());
                        }
                    }
                }
                _ => return Err(ParserError::InvalidDefName),
            }

            let value = self.parse_value(value_val)?;
            let_expr.bindings.push(Binding::new(names, value));
            i += 2;
        }

        // Parse body
        for item in items.iter().skip(2) {
            let node = self.parse_value(item)?;
            let_expr.body.push(node);
        }

        Ok(Node::let_expr(let_expr))
    }

    fn parse_type_annotation(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        // (: value type)
        if items.len() < 3 {
            return Err(ParserError::InvalidTypeAnnotation);
        }

        let val_node = self.parse_value(&items[1])?;

        let typ = match &items[2] {
            Value::Symbol(sym) => Type::new(&sym.name),
            Value::List(_) => {
                // Could be a function type or complex type
                Type::new("<complex>")
            }
            _ => return Err(ParserError::InvalidType),
        };

        Ok(Node::type_annotation(TypeAnnotation::new(val_node, typ)))
    }

    fn parse_function_type_value(&mut self, items: &[Value]) -> Result<FunctionType, ParserError> {
        // (-> [args...] [returns...])
        if items.len() < 3 {
            return Err(ParserError::InvalidFunctionType);
        }

        let mut ft = FunctionType::new();

        // Parse argument types
        if let Value::Vector(args_vec) = &items[1] {
            for type_val in args_vec {
                match type_val {
                    Value::Symbol(sym) => {
                        ft.arg_types.push(Type::new(&sym.name));
                    }
                    Value::String(s) => {
                        // Support string-quoted types (e.g., "memref<?xf32>")
                        ft.arg_types.push(Type::new(s));
                    }
                    _ => {}
                }
            }
        }

        // Parse return types
        if let Value::Vector(returns_vec) = &items[2] {
            for type_val in returns_vec {
                match type_val {
                    Value::Symbol(sym) => {
                        ft.return_types.push(Type::new(&sym.name));
                    }
                    Value::String(s) => {
                        // Support string-quoted types (e.g., "memref<?xf32>")
                        ft.return_types.push(Type::new(s));
                    }
                    _ => {}
                }
            }
        }

        Ok(ft)
    }

    fn parse_function_type(&mut self, items: &[Value]) -> Result<Node, ParserError> {
        let ft = self.parse_function_type_value(items)?;
        Ok(Node::function_type(ft))
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Reader;
    use crate::tokenizer::Tokenizer;
    use pretty_assertions::assert_eq;

    fn parse_str(input: &str) -> Vec<Node> {
        let mut tokenizer = Tokenizer::new(input);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        parser.parse(&values).unwrap()
    }

    #[test]
    fn test_simple_operation() {
        let nodes = parse_str("(arith.addi x y)");

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_operation());

        let op = nodes[0].as_operation();
        assert_eq!(op.name, "addi");
        assert_eq!(op.namespace.as_ref().unwrap(), "arith");
    }

    #[test]
    fn test_with_attributes() {
        let nodes = parse_str("(arith.constant {:value 42})");

        assert_eq!(nodes.len(), 1);

        let op = nodes[0].as_operation();
        assert!(op.attributes.contains_key("value"));
    }

    #[test]
    fn test_def_binding() {
        let nodes = parse_str("(def x 42)");

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_def());

        let binding = nodes[0].as_def();
        assert_eq!(binding.names.len(), 1);
        assert_eq!(binding.names[0], "x");
    }

    #[test]
    fn test_let_form() {
        let nodes = parse_str("(let [x 1 y 2] (+ x y))");

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_let());
    }

    #[test]
    fn test_function_type() {
        let nodes = parse_str("(-> [i32 f64] [i1])");

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_function_type());
    }

    #[test]
    fn test_region() {
        let nodes = parse_str(
            r#"(region
              (block ^entry
                (arith.addi 1 2)))"#,
        );

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_region());
    }

    #[test]
    fn test_module() {
        let nodes = parse_str(
            r#"(module
              (do
                (block
                  (arith.addi 1 2))))"#,
        );

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_module());
    }

    #[test]
    fn test_block_label() {
        let nodes = parse_str(
            r#"(block ^entry
              (arith.addi 1 2))"#,
        );

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_block());
        assert_eq!(nodes[0].as_block().operations.len(), 1);
    }

    #[test]
    fn test_type_annotation() {
        let nodes = parse_str("(: 42 i32)");

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_type_annotation());
        assert_eq!(nodes[0].as_type_annotation().typ.name, "i32");
        assert!(nodes[0].as_type_annotation().value.is_literal());
    }

    #[test]
    fn test_function_type_preserves_types() {
        let nodes = parse_str("(-> [i32 f64] [i1 i64])");

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_function_type());

        let ft = nodes[0].as_function_type();
        assert_eq!(ft.arg_types.len(), 2);
        assert_eq!(ft.arg_types[0].name, "i32");
        assert_eq!(ft.arg_types[1].name, "f64");
        assert_eq!(ft.return_types.len(), 2);
        assert_eq!(ft.return_types[0].name, "i1");
        assert_eq!(ft.return_types[1].name, "i64");
    }

    #[test]
    fn test_block_arguments_capture_types() {
        let nodes = parse_str(
            r#"(block ^entry [(: x i32) (: y f32)]
              (arith.addi x y))"#,
        );

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_block());

        let block = nodes[0].as_block();
        assert_eq!(block.arguments.len(), 2);
        assert_eq!(block.arguments[0].name, "x");
        assert!(block.arguments[0].typ.is_some());
        assert_eq!(block.arguments[0].typ.as_ref().unwrap().name, "i32");
        assert_eq!(block.arguments[1].name, "y");
        assert!(block.arguments[1].typ.is_some());
        assert_eq!(block.arguments[1].typ.as_ref().unwrap().name, "f32");
    }

    #[test]
    fn test_def_destructuring() {
        let nodes = parse_str("(def [a b] (arith.multi_result))");

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_def());

        let binding = nodes[0].as_def();
        assert_eq!(binding.names.len(), 2);
        assert_eq!(binding.names[0], "a");
        assert_eq!(binding.names[1], "b");
    }

    #[test]
    fn test_let_requires_vector_bindings() {
        let mut tokenizer = Tokenizer::new("(let bad 1)");
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let values = reader.read().unwrap();
        let mut parser = Parser::new();
        let result = parser.parse(&values);

        assert!(matches!(result, Err(ParserError::LetBindingsMustBeVector)));
    }
}
