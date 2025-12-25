//! Function call macros for high-level syntax
//!
//! call  - Call with typed return: (call TYPE func args...) -> (func.call {:callee @func :result TYPE} args...)
//! call! - Void call: (call! func args...) -> (func.call {:callee @func} args...)
//!
//! Example:
//!   (call i64 add x y)          -> (func.call {:callee @add :result i64} x y)
//!   (call! free ptr)            -> (func.call {:callee @free} ptr)

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Typed function call: (call TYPE func args...) -> (func.call {:callee @func :result TYPE} args...)
pub struct CallMacro;

impl Macro for CallMacro {
    fn name(&self) -> &str {
        "call"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.len() < 2 {
            return Err(MacroError::WrongArity {
                macro_name: "call".into(),
                expected: "at least 2 (TYPE, func-name, args...)".into(),
                got: args.len(),
            });
        }

        let result_type = args[0].clone();
        let func_name = match &args[1] {
            Value::Symbol(sym) => format!("@{}", sym.name),
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "call".into(),
                    expected: "symbol for function name",
                    got: format!("{:?}", args[1]),
                })
            }
        };

        let mut attrs = HashMap::new();
        attrs.insert("callee".to_string(), Value::symbol(&func_name));
        attrs.insert("result".to_string(), result_type);

        let mut call = vec![Value::symbol("func.call"), Value::Map(attrs)];
        call.extend(args[2..].iter().cloned());

        Ok(Value::List(call))
    }

    fn doc(&self) -> Option<&str> {
        Some("Typed function call: (call TYPE func args...) -> (func.call {:callee @func :result TYPE} args...)")
    }
}

/// Void function call: (call! func args...) -> (func.call {:callee @func} args...)
pub struct CallBangMacro;

impl Macro for CallBangMacro {
    fn name(&self) -> &str {
        "call!"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        if args.is_empty() {
            return Err(MacroError::WrongArity {
                macro_name: "call!".into(),
                expected: "at least 1 (func-name, args...)".into(),
                got: 0,
            });
        }

        let func_name = match &args[0] {
            Value::Symbol(sym) => format!("@{}", sym.name),
            _ => {
                return Err(MacroError::TypeError {
                    macro_name: "call!".into(),
                    expected: "symbol for function name",
                    got: format!("{:?}", args[0]),
                })
            }
        };

        let mut attrs = HashMap::new();
        attrs.insert("callee".to_string(), Value::symbol(&func_name));

        let mut call = vec![Value::symbol("func.call"), Value::Map(attrs)];
        call.extend(args[1..].iter().cloned());

        Ok(Value::List(call))
    }

    fn doc(&self) -> Option<&str> {
        Some("Void function call: (call! func args...) -> (func.call {:callee @func} args...)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_typed() {
        let macro_impl = CallMacro;
        let args = vec![
            Value::symbol("i64"),
            Value::symbol("add"),
            Value::symbol("x"),
            Value::symbol("y"),
        ];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 4); // func.call, attrs, x, y
            assert_eq!(items[0].as_symbol().name, "func.call");

            if let Value::Map(m) = &items[1] {
                if let Some(Value::Symbol(callee)) = m.get("callee") {
                    assert_eq!(callee.name, "@add");
                } else {
                    panic!("Expected callee symbol");
                }
                assert!(m.contains_key("result"));
            } else {
                panic!("Expected map");
            }

            assert_eq!(items[2].as_symbol().name, "x");
            assert_eq!(items[3].as_symbol().name, "y");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_call_void() {
        let macro_impl = CallBangMacro;
        let args = vec![Value::symbol("free"), Value::symbol("ptr")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 3); // func.call, attrs, ptr
            assert_eq!(items[0].as_symbol().name, "func.call");

            if let Value::Map(m) = &items[1] {
                if let Some(Value::Symbol(callee)) = m.get("callee") {
                    assert_eq!(callee.name, "@free");
                } else {
                    panic!("Expected callee symbol");
                }
                // No result key for void calls
                assert!(!m.contains_key("result"));
            } else {
                panic!("Expected map");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_call_no_args() {
        let macro_impl = CallMacro;
        let args = vec![Value::symbol("i64"), Value::symbol("get_value")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 2); // func.call, attrs
        } else {
            panic!("Expected list");
        }
    }
}
