//! External function declaration macro
//!
//! extern-fn - Declare external function: (extern-fn name (-> [args...] [rets...]))
//!             -> (func.func {:sym_name "name" :function_type (-> ...) :sym_visibility "private"})
//!
//! Note: Named extern-fn (not extern) to avoid conflict with the built-in (extern :value-ffi) form.
//!
//! Example:
//!   (extern-fn malloc (-> [i64] [!llvm.ptr]))
//!   -> (func.func {:sym_name "malloc" :function_type (-> [i64] [!llvm.ptr]) :sym_visibility "private"})

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// External function declaration: (extern-fn name fn-type)
pub struct ExternMacro;

impl Macro for ExternMacro {
    fn name(&self) -> &str {
        "extern-fn"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() != 2 {
            return Err(MacroError::WrongArity {
                macro_name: "extern-fn".into(),
                expected: "2 (name, function-type)".into(),
                got: args.len(),
            });
        }

        let func_name = match &args[0] {
            Value::Symbol(sym) => sym.name.clone(),
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "extern-fn".into(),
                    expected: "symbol for function name",
                    got: format!("{:?}", args[0]),
                })
            }
        };

        let func_type = args[1].clone();

        // Build {:sym_name "name" :function_type FN_TYPE :sym_visibility "private"}
        let mut attrs = HashMap::new();
        attrs.insert("sym_name".to_string(), Value::String(func_name));
        attrs.insert("function_type".to_string(), func_type);
        attrs.insert(
            "sym_visibility".to_string(),
            Value::String("private".to_string()),
        );

        Ok(Value::List(vec![
            Value::symbol("func.func"),
            Value::Map(attrs),
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Declare external function: (extern-fn name (-> [args] [rets])) -> (func.func {:sym_name ... :sym_visibility \"private\"})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extern_fn() {
        let macro_impl = ExternMacro;
        // (extern-fn malloc (-> [i64] [!llvm.ptr]))
        let fn_type = Value::List(vec![
            Value::symbol("->"),
            Value::Vector(vec![Value::symbol("i64")]),
            Value::Vector(vec![Value::symbol("!llvm.ptr")]),
        ]);
        let args = vec![Value::symbol("malloc"), fn_type];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].as_symbol().name, "func.func");

            if let Value::Map(m) = &items[1] {
                if let Some(Value::String(name)) = m.get("sym_name") {
                    assert_eq!(name, "malloc");
                } else {
                    panic!("Expected sym_name string");
                }
                if let Some(Value::String(vis)) = m.get("sym_visibility") {
                    assert_eq!(vis, "private");
                } else {
                    panic!("Expected sym_visibility string");
                }
                assert!(m.contains_key("function_type"));
            } else {
                panic!("Expected map");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_extern_fn_wrong_arity() {
        let macro_impl = ExternMacro;
        let args = vec![Value::symbol("malloc")];
        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }
}
