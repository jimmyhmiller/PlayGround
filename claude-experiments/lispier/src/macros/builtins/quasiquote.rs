//! The `quasiquote` macro for code generation in JIT macros
//!
//! This macro generates MLIR code that constructs Value objects at runtime.
//! Unlike traditional Lisp quasiquote (which is a tree transformation),
//! this version generates code that builds the tree using FFI calls.
//!
//! Syntax:
//!   `(foo ~x bar) -> (quasiquote (foo (unquote x) bar))
//!
//! Expands to MLIR code like:
//!   (do
//!     (def __qq_0 (func.call {:callee @value_list_new :result !llvm.ptr}))
//!     (def __qq_1 (func.call {:callee @value_symbol_foo :result !llvm.ptr}))
//!     (func.call {:callee @value_list_push} __qq_0 __qq_1)
//!     (func.call {:callee @value_list_push} __qq_0 x)
//!     ...
//!     __qq_0)

use crate::macros::{Macro, MacroError};
use crate::value::{Symbol, Value};
use std::collections::HashMap;

/// The `quasiquote` macro for generating code that builds Values
pub struct QuasiquoteMacro;

impl Macro for QuasiquoteMacro {
    fn name(&self) -> &str {
        "quasiquote"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 1 {
            return Err(MacroError::WrongArity {
                macro_name: "quasiquote".into(),
                expected: "1".into(),
                got: args.len(),
            });
        }

        let mut ctx = QuasiquoteContext::new();
        ctx.expand_quasiquote(&args[0])
    }

    fn doc(&self) -> Option<&str> {
        Some("Generate code that constructs a Value. Use ~ to unquote expressions.")
    }
}

/// Context for quasiquote expansion
struct QuasiquoteContext {
    /// Counter for generating unique variable names
    gensym_counter: usize,
    /// Map from symbol names to their FFI function names (if they exist)
    known_symbols: HashMap<&'static str, &'static str>,
}

impl QuasiquoteContext {
    fn new() -> Self {
        let mut known_symbols = HashMap::new();
        // Map symbol names to their FFI function names
        known_symbols.insert("arith.addi", "value_symbol_arith_addi");
        known_symbols.insert("arith.subi", "value_symbol_arith_subi");
        known_symbols.insert("arith.muli", "value_symbol_arith_muli");
        known_symbols.insert("arith.divsi", "value_symbol_arith_divsi");
        known_symbols.insert("func.return", "value_symbol_func_return");
        known_symbols.insert("def", "value_symbol_def");
        known_symbols.insert("do", "value_symbol_do");
        known_symbols.insert(":", "value_symbol_colon");

        Self {
            gensym_counter: 0,
            known_symbols,
        }
    }

    /// Generate a unique variable name
    fn gensym(&mut self, prefix: &str) -> String {
        let name = format!("__{}__{}", prefix, self.gensym_counter);
        self.gensym_counter += 1;
        name
    }

    /// Main entry point: expand a quasiquoted form into code that builds it
    fn expand_quasiquote(&mut self, form: &Value) -> Result<Value, MacroError> {
        match form {
            // Check for unquote: (unquote x) -> just return x
            Value::List(items) if self.is_unquote(items) => {
                if items.len() != 2 {
                    return Err(MacroError::ExpansionFailed(
                        "unquote requires exactly one argument".into(),
                    ));
                }
                // Return the unquoted expression as-is
                Ok(items[1].clone())
            }

            // Check for unquote-splice - error for now (V1 doesn't support it)
            Value::List(items) if self.is_unquote_splice(items) => {
                Err(MacroError::ExpansionFailed(
                    "unquote-splice (~@) is not yet supported".into(),
                ))
            }

            // Check for nested quasiquote - error for now
            Value::List(items) if self.is_quasiquote(items) => {
                Err(MacroError::ExpansionFailed(
                    "nested quasiquote is not yet supported".into(),
                ))
            }

            // List: generate code to create a list and push each element
            Value::List(items) => self.expand_list(items),

            // Vector: generate code to create a vector and push each element
            Value::Vector(items) => self.expand_vector(items),

            // Symbol: generate code to create the symbol
            Value::Symbol(sym) => self.expand_symbol(sym),

            // Number: generate code to create a number
            Value::Number(n) => self.expand_number(*n),

            // String: generate code to create a string (not yet supported)
            Value::String(_) => Err(MacroError::ExpansionFailed(
                "strings in quasiquote not yet supported".into(),
            )),

            // Keyword: generate code to create a keyword (not yet supported)
            Value::Keyword(_) => Err(MacroError::ExpansionFailed(
                "keywords in quasiquote not yet supported".into(),
            )),

            // Map: not yet supported
            Value::Map(_) => Err(MacroError::ExpansionFailed(
                "maps in quasiquote not yet supported".into(),
            )),

            // Boolean
            Value::Boolean(b) => self.expand_boolean(*b),

            // Nil
            Value::Nil => self.expand_nil(),
        }
    }

    fn is_unquote(&self, items: &[Value]) -> bool {
        if let Some(Value::Symbol(sym)) = items.first() {
            sym.name == "unquote"
        } else {
            false
        }
    }

    fn is_unquote_splice(&self, items: &[Value]) -> bool {
        if let Some(Value::Symbol(sym)) = items.first() {
            sym.name == "unquote-splice"
        } else {
            false
        }
    }

    fn is_quasiquote(&self, items: &[Value]) -> bool {
        if let Some(Value::Symbol(sym)) = items.first() {
            sym.name == "quasiquote"
        } else {
            false
        }
    }

    /// Expand a list: (a b c) -> code that creates [a, b, c]
    fn expand_list(&mut self, items: &[Value]) -> Result<Value, MacroError> {
        let list_var = self.gensym("qq");

        let mut bindings: Vec<Value> = Vec::new();
        let mut body: Vec<Value> = Vec::new();

        // First binding: __qq_0 = value_list_new()
        bindings.push(Value::symbol(&list_var));
        bindings.push(self.make_ffi_call("value_list_new", vec![]));

        // For each item, generate code to create it and push onto the list
        for item in items {
            // Check for unquote-splice
            if let Value::List(inner) = item {
                if self.is_unquote_splice(inner) {
                    return Err(MacroError::ExpansionFailed(
                        "unquote-splice (~@) is not yet supported".into(),
                    ));
                }
            }

            // Expand the item (might be simple or might be a nested structure)
            let item_expr = self.expand_quasiquote(item)?;

            // If the item_expr is simple (a symbol reference), we can use it directly
            // Otherwise, we need to bind it to a temp var first
            let item_var = if self.is_simple_expr(&item_expr) {
                item_expr
            } else {
                let temp = self.gensym("item");
                bindings.push(Value::symbol(&temp));
                bindings.push(item_expr);
                Value::symbol(&temp)
            };

            // (func.call {:callee @value_list_push} list_var item_var)
            body.push(self.make_ffi_call_stmt(
                "value_list_push",
                vec![Value::symbol(&list_var), item_var],
            ));
        }

        // Return the list variable as the final expression
        body.push(Value::symbol(&list_var));

        // Build (let [bindings...] body...)
        let mut let_form = vec![
            Value::symbol("let"),
            Value::Vector(bindings),
        ];
        let_form.extend(body);
        Ok(Value::List(let_form))
    }

    /// Expand a vector: [a b c] -> code that creates vector
    fn expand_vector(&mut self, items: &[Value]) -> Result<Value, MacroError> {
        let vec_var = self.gensym("vec");

        let mut bindings: Vec<Value> = Vec::new();
        let mut body: Vec<Value> = Vec::new();

        // First binding: __vec_0 = value_vector_new()
        bindings.push(Value::symbol(&vec_var));
        bindings.push(self.make_ffi_call("value_vector_new", vec![]));

        // For each item, generate code to create it and push onto the vector
        for item in items {
            let item_expr = self.expand_quasiquote(item)?;

            let item_var = if self.is_simple_expr(&item_expr) {
                item_expr
            } else {
                let temp = self.gensym("item");
                bindings.push(Value::symbol(&temp));
                bindings.push(item_expr);
                Value::symbol(&temp)
            };

            // Use value_list_push for vectors too (they use the same push mechanism)
            body.push(self.make_ffi_call_stmt(
                "value_list_push",
                vec![Value::symbol(&vec_var), item_var],
            ));
        }

        body.push(Value::symbol(&vec_var));

        // Build (let [bindings...] body...)
        let mut let_form = vec![
            Value::symbol("let"),
            Value::Vector(bindings),
        ];
        let_form.extend(body);
        Ok(Value::List(let_form))
    }

    /// Expand a symbol: foo -> (func.call {:callee @value_symbol_foo ...})
    fn expand_symbol(&mut self, sym: &Symbol) -> Result<Value, MacroError> {
        // Use the fully qualified name (includes namespace if present)
        let qualified = sym.qualified_name();

        // Check if we have a known FFI function for this symbol
        if let Some(ffi_name) = self.known_symbols.get(qualified.as_str()) {
            Ok(self.make_ffi_call(ffi_name, vec![]))
        } else {
            // For unknown symbols, we need to create them dynamically
            // This requires passing a string to value_symbol_new, which needs
            // string handling we don't have yet
            Err(MacroError::ExpansionFailed(format!(
                "unknown symbol '{}' in quasiquote. Add a value_symbol_{} FFI function, \
                 or use an existing symbol: {:?}",
                qualified,
                qualified.replace('.', "_").replace('-', "_"),
                self.known_symbols.keys().collect::<Vec<_>>()
            )))
        }
    }

    /// Expand a number: 42 -> (func.call {:callee @value_number_new ...} 42.0)
    fn expand_number(&mut self, n: f64) -> Result<Value, MacroError> {
        // (func.call {:callee @value_number_new :result !llvm.ptr} (: n f64))
        Ok(self.make_ffi_call(
            "value_number_new",
            vec![Value::List(vec![
                Value::symbol(":"),
                Value::Number(n),
                Value::symbol("f64"),
            ])],
        ))
    }

    /// Expand a boolean
    fn expand_boolean(&mut self, b: bool) -> Result<Value, MacroError> {
        // value_bool_new takes a bool (i1 in LLVM)
        // For simplicity, use 1 or 0
        let val = if b { 1 } else { 0 };
        Ok(self.make_ffi_call(
            "value_bool_new",
            vec![Value::List(vec![
                Value::symbol(":"),
                Value::Number(val as f64),
                Value::symbol("i1"),
            ])],
        ))
    }

    /// Expand nil
    fn expand_nil(&mut self) -> Result<Value, MacroError> {
        Ok(self.make_ffi_call("value_nil_new", vec![]))
    }

    /// Check if an expression is "simple" (just a symbol reference)
    fn is_simple_expr(&self, expr: &Value) -> bool {
        matches!(expr, Value::Symbol(_))
    }

    /// Make an FFI call expression with result
    /// (func.call {:callee @name :result !llvm.ptr} args...)
    fn make_ffi_call(&self, name: &str, args: Vec<Value>) -> Value {
        let mut map = HashMap::new();
        map.insert("callee".to_string(), Value::symbol(&format!("@{}", name)));
        map.insert("result".to_string(), Value::symbol("!llvm.ptr"));

        let mut call = vec![Value::symbol("func.call"), Value::Map(map)];
        call.extend(args);
        Value::List(call)
    }

    /// Make an FFI call statement without result binding
    /// (func.call {:callee @name} args...)
    fn make_ffi_call_stmt(&self, name: &str, args: Vec<Value>) -> Value {
        let mut map = HashMap::new();
        map.insert("callee".to_string(), Value::symbol(&format!("@{}", name)));

        let mut call = vec![Value::symbol("func.call"), Value::Map(map)];
        call.extend(args);
        Value::List(call)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_unquote() {
        let macro_impl = QuasiquoteMacro;

        // (quasiquote (unquote x)) -> x
        let args = vec![Value::List(vec![
            Value::symbol("unquote"),
            Value::symbol("x"),
        ])];

        let result = macro_impl.expand(&args).unwrap();

        // Should just return x
        if let Value::Symbol(sym) = &result {
            assert_eq!(sym.name, "x");
        } else {
            panic!("Expected symbol x, got {:?}", result);
        }
    }

    #[test]
    fn test_unquote_splice_error() {
        let macro_impl = QuasiquoteMacro;

        // (quasiquote (unquote-splice x)) -> error
        let args = vec![Value::List(vec![
            Value::symbol("unquote-splice"),
            Value::symbol("x"),
        ])];

        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_symbol_error() {
        let macro_impl = QuasiquoteMacro;

        // (quasiquote unknown-symbol) -> error (no FFI function)
        let args = vec![Value::symbol("unknown-symbol")];

        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_known_symbol() {
        let macro_impl = QuasiquoteMacro;

        // (quasiquote arith.addi) -> FFI call
        let args = vec![Value::symbol("arith.addi")];

        let result = macro_impl.expand(&args).unwrap();

        // Should be a func.call
        if let Value::List(items) = &result {
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "func.call");
            } else {
                panic!("Expected func.call symbol");
            }
        } else {
            panic!("Expected list");
        }
    }
}
