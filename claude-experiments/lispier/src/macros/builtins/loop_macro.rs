//! The `loop` macro for iteration using scf.while
//!
//! Syntax:
//!   (loop {:result type} [var init]
//!     :while cond-expr
//!     :do body-expr)
//!
//! Expands to:
//!   (scf.while {:result type} init
//!     (region (block [(: var type)]
//!       (scf.condition cond-expr var)))
//!     (region (block [(: var type)]
//!       (scf.yield body-expr))))

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// The `loop` macro for iteration
pub struct LoopMacro;

impl Macro for LoopMacro {
    fn name(&self) -> &str {
        "loop"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // (loop {:result type} [var init] :while cond :do body)
        // args: [{:result type}, [var init], :while, cond, :do, body]
        if args.len() != 6 {
            return Err(MacroError::WrongArity {
                macro_name: "loop".into(),
                expected: "6 ({:result type} [var init] :while cond :do body)".into(),
                got: args.len(),
            });
        }

        // Extract type map
        let type_map = match &args[0] {
            Value::Map(m) => {
                let result_type = m.get("result").ok_or_else(|| MacroError::InvalidSyntax {
                    macro_name: "loop".into(),
                    message: "type map must have :result key".into(),
                })?;
                result_type.clone()
            }
            _ => {
                return Err(MacroError::InvalidSyntax {
                    macro_name: "loop".into(),
                    message: "first argument must be {:result type}".into(),
                });
            }
        };

        // Extract [var init] binding
        let (var, init) = match &args[1] {
            Value::Vector(items) if items.len() == 2 => {
                let var = match &items[0] {
                    Value::Symbol(sym) => sym.name.clone(),
                    _ => {
                        return Err(MacroError::InvalidSyntax {
                            macro_name: "loop".into(),
                            message: "binding variable must be a symbol".into(),
                        });
                    }
                };
                (var, items[1].clone())
            }
            _ => {
                return Err(MacroError::InvalidSyntax {
                    macro_name: "loop".into(),
                    message: "second argument must be [var init]".into(),
                });
            }
        };

        // Check :while keyword (keyword includes the colon)
        match &args[2] {
            Value::Keyword(kw) if kw == ":while" => {}
            _ => {
                return Err(MacroError::InvalidSyntax {
                    macro_name: "loop".into(),
                    message: "expected :while keyword".into(),
                });
            }
        }

        let cond_expr = args[3].clone();

        // Check :do keyword (keyword includes the colon)
        match &args[4] {
            Value::Keyword(kw) if kw == ":do" => {}
            _ => {
                return Err(MacroError::InvalidSyntax {
                    macro_name: "loop".into(),
                    message: "expected :do keyword".into(),
                });
            }
        }

        let body_expr = args[5].clone();

        // Build typed block argument: (: var type)
        let typed_arg = Value::List(vec![
            Value::symbol(":"),
            Value::symbol(&var),
            type_map.clone(),
        ]);

        // Build condition region:
        // (region (block [(: var type)] (scf.condition cond-expr var)))
        let cond_block = Value::List(vec![
            Value::symbol("block"),
            Value::Vector(vec![typed_arg.clone()]),
            Value::List(vec![
                Value::symbol("scf.condition"),
                cond_expr,
                Value::symbol(&var),
            ]),
        ]);
        let cond_region = Value::List(vec![Value::symbol("region"), cond_block]);

        // Build body region:
        // (region (block [(: var type)] (scf.yield body-expr)))
        let body_block = Value::List(vec![
            Value::symbol("block"),
            Value::Vector(vec![typed_arg]),
            Value::List(vec![Value::symbol("scf.yield"), body_expr]),
        ]);
        let body_region = Value::List(vec![Value::symbol("region"), body_block]);

        // Build scf.while: (scf.while {:result type} init cond-region body-region)
        Ok(Value::List(vec![
            Value::symbol("scf.while"),
            args[0].clone(), // {:result type}
            init,
            cond_region,
            body_region,
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Loop with while condition: (loop {:result type} [var init] :while cond :do body)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_expansion() {
        let macro_impl = LoopMacro;
        
        // Build {:result i64}
        let mut type_map = std::collections::HashMap::new();
        type_map.insert("result".to_string(), Value::symbol("i64"));
        
        let args = vec![
            Value::Map(type_map),
            Value::Vector(vec![Value::symbol("x"), Value::Number(0.0)]),
            Value::Keyword(":while".into()),
            Value::symbol("cond"),
            Value::Keyword(":do".into()),
            Value::symbol("body"),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should be (scf.while {:result i64} 0 (region ...) (region ...))
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 5);
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "scf.while");
            } else {
                panic!("Expected scf.while symbol");
            }
        } else {
            panic!("Expected list");
        }
    }
}
