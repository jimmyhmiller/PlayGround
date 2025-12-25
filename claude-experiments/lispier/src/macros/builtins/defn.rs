//! The `defn` macro for defining functions
//!
//! Syntax:
//!   (defn name [params...] body...)
//!   (defn name [params...] -> return-type body...)
//!
//! Expands to:
//!   (func.func {:sym_name "name"
//!               :function_type (-> [param-types...] [return-types...])
//!               :llvm.emit_c_interface true}
//!     (do
//!       (block [params...]
//!         body...)))

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// The `defn` macro for defining functions
pub struct DefnMacro;

impl Macro for DefnMacro {
    fn name(&self) -> &str {
        "defn"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // (defn name [params...] body...)
        // (defn name [params...] -> return-type body...)

        if args.len() < 3 {
            return Err(MacroError::WrongArity {
                macro_name: "defn".into(),
                expected: "at least 3 (name, params, body)".into(),
                got: args.len(),
            });
        }

        // Extract function name
        let name = match &args[0] {
            Value::Symbol(sym) => sym.name.clone(),
            other => {
                return Err(MacroError::TypeError {
                    macro_name: "defn".into(),
                    expected: "symbol for function name",
                    got: format!("{:?}", other),
                })
            }
        };

        // Extract parameters
        let params = match &args[1] {
            Value::Vector(v) => v.clone(),
            other => {
                return Err(MacroError::TypeError {
                    macro_name: "defn".into(),
                    expected: "vector of parameters",
                    got: format!("{:?}", other),
                })
            }
        };

        // Check for explicit return type: [params] -> type body...
        let (return_types, body_start) = if args.len() > 3 {
            if let Value::Symbol(sym) = &args[2] {
                if sym.name == "->" {
                    // Next arg is the return type
                    let ret_type = args[3].clone();
                    (vec![ret_type], 4)
                } else {
                    (vec![], 2)
                }
            } else {
                (vec![], 2)
            }
        } else {
            (vec![], 2)
        };

        let body: Vec<Value> = args[body_start..].to_vec();

        if body.is_empty() {
            return Err(MacroError::WrongArity {
                macro_name: "defn".into(),
                expected: "at least one body expression".into(),
                got: 0,
            });
        }

        // Extract parameter types for function_type
        let (param_types, block_args) = extract_params(&params)?;

        // Build the expansion
        Ok(build_func(name, param_types, return_types, block_args, body))
    }

    fn doc(&self) -> Option<&str> {
        Some("Define a function. (defn name [params...] -> return-type body...)")
    }
}

/// Extract parameter types from parameter list
///
/// Parameters can be:
/// - (: name type) - typed parameter
fn extract_params(params: &[Value]) -> Result<(Vec<Value>, Vec<Value>), MacroError> {
    let mut types = Vec::new();
    let mut block_args = Vec::new();

    for param in params {
        match param {
            Value::List(items) if items.len() >= 3 => {
                // (: name type) form
                if let Value::Symbol(colon) = &items[0] {
                    if colon.name == ":" {
                        types.push(items[2].clone());
                        block_args.push(param.clone());
                        continue;
                    }
                }
                return Err(MacroError::TypeError {
                    macro_name: "defn".into(),
                    expected: "type annotation (: name type)",
                    got: format!("{:?}", param),
                });
            }
            Value::Symbol(_) => {
                return Err(MacroError::ExpansionFailed(
                    "defn parameters must have type annotations: (: name type)".into(),
                ));
            }
            other => {
                return Err(MacroError::TypeError {
                    macro_name: "defn".into(),
                    expected: "parameter (: name type)",
                    got: format!("{:?}", other),
                })
            }
        }
    }

    Ok((types, block_args))
}

/// Build the func.func expansion
fn build_func(
    name: String,
    param_types: Vec<Value>,
    return_types: Vec<Value>,
    block_args: Vec<Value>,
    body: Vec<Value>,
) -> Value {
    // Build: (func.func {:sym_name "name" ...} (do (block [...] body...)))

    // Function type: (-> [param-types] [return-types])
    let func_type = Value::List(vec![
        Value::symbol("->"),
        Value::Vector(param_types),
        Value::Vector(return_types),
    ]);

    // Attributes map
    let mut attrs = HashMap::new();
    attrs.insert("sym_name".to_string(), Value::String(name));
    attrs.insert("function_type".to_string(), func_type);
    attrs.insert("llvm.emit_c_interface".to_string(), Value::Boolean(true));

    // Block with args and body
    let mut block_items = vec![Value::symbol("block"), Value::Vector(block_args)];
    block_items.extend(body);
    let block = Value::List(block_items);

    // Region (do block)
    let region = Value::List(vec![Value::symbol("do"), block]);

    // Final func.func form
    Value::List(vec![Value::symbol("func.func"), Value::Map(attrs), region])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defn_basic() {
        let macro_impl = DefnMacro;

        // (defn main [] -> i32 (func.return (: 42 i32)))
        let args = vec![
            Value::symbol("main"),
            Value::Vector(vec![]),
            Value::symbol("->"),
            Value::symbol("i32"),
            Value::List(vec![
                Value::symbol("func.return"),
                Value::List(vec![
                    Value::symbol(":"),
                    Value::Number(42.0),
                    Value::symbol("i32"),
                ]),
            ]),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should be a list starting with func.func
        if let Value::List(items) = &result {
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "func.func");
            } else {
                panic!("Expected func.func symbol");
            }

            // Second item should be the attributes map
            if let Value::Map(attrs) = &items[1] {
                assert!(attrs.contains_key("sym_name"));
                assert!(attrs.contains_key("function_type"));
            } else {
                panic!("Expected attributes map");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_defn_with_params() {
        let macro_impl = DefnMacro;

        // (defn add [(: x i32) (: y i32)] -> i32 (func.return (arith.addi x y)))
        let args = vec![
            Value::symbol("add"),
            Value::Vector(vec![
                Value::List(vec![
                    Value::symbol(":"),
                    Value::symbol("x"),
                    Value::symbol("i32"),
                ]),
                Value::List(vec![
                    Value::symbol(":"),
                    Value::symbol("y"),
                    Value::symbol("i32"),
                ]),
            ]),
            Value::symbol("->"),
            Value::symbol("i32"),
            Value::List(vec![
                Value::symbol("func.return"),
                Value::List(vec![
                    Value::symbol("arith.addi"),
                    Value::symbol("x"),
                    Value::symbol("y"),
                ]),
            ]),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should expand successfully
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 3); // func.func, attrs, region
        } else {
            panic!("Expected list");
        }
    }
}
