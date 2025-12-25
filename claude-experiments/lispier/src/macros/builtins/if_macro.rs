//! The `if` macro for conditional execution with else branch
//!
//! Syntax:
//!   (if condition then-expr else-expr)              ; void return
//!   (if {:result type} condition then-expr else-expr) ; typed return
//!
//! Expands to:
//!   (scf.if condition
//!     (region (block [] (scf.yield then-expr)))
//!     (region (block [] (scf.yield else-expr))))
//!
//! Or with type:
//!   (scf.if {:result type} condition
//!     (region (block [] (scf.yield then-expr)))
//!     (region (block [] (scf.yield else-expr))))
//!
//! When a branch is a `do` block:
//!   (if cond (do stmt1 stmt2 last-expr) else)
//! Expands to:
//!   (scf.if cond
//!     (region (block [] stmt1 stmt2 (scf.yield last-expr)))
//!     ...)

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// The `if` macro for two-branch conditionals
pub struct IfMacro;

/// Check if a Value is a (do ...) block and extract its contents
fn extract_do_block(expr: &Value) -> Option<Vec<Value>> {
    if let Value::List(items) = expr {
        if items.len() >= 2 {
            if let Value::Symbol(sym) = &items[0] {
                if sym.name == "do" {
                    return Some(items[1..].to_vec());
                }
            }
        }
    }
    None
}

/// Check if a Value is a (let [bindings...] body...) and extract bindings and body
/// Returns (bindings as def statements, body expressions)
fn extract_let_block(expr: &Value) -> Option<(Vec<Value>, Vec<Value>)> {
    if let Value::List(items) = expr {
        if items.len() >= 3 {
            if let Value::Symbol(sym) = &items[0] {
                if sym.name == "let" {
                    if let Value::Vector(bindings_vec) = &items[1] {
                        // Convert bindings to def statements
                        let mut defs = Vec::new();
                        let mut i = 0;
                        while i + 1 < bindings_vec.len() {
                            let name = bindings_vec[i].clone();
                            let value = bindings_vec[i + 1].clone();
                            defs.push(Value::List(vec![
                                Value::symbol("def"),
                                name,
                                value,
                            ]));
                            i += 2;
                        }
                        // Body is everything after the bindings vector
                        let body: Vec<Value> = items[2..].to_vec();
                        return Some((defs, body));
                    }
                }
            }
        }
    }
    None
}

/// Build a region block from an expression.
/// If the expression is a `do` block, emit all statements and yield the last.
/// If the expression is a `let` block, emit bindings as defs and yield the last body expr.
/// Otherwise, just yield the expression.
fn build_region_block(expr: Value) -> Value {
    let block_contents = if let Some(mut do_contents) = extract_do_block(&expr) {
        if do_contents.is_empty() {
            // Empty do block - yield nil
            vec![Value::List(vec![Value::symbol("scf.yield")])]
        } else {
            // Pop the last expression, yield it
            let last = do_contents.pop().unwrap();
            let mut contents = do_contents;
            contents.push(Value::List(vec![Value::symbol("scf.yield"), last]));
            contents
        }
    } else if let Some((defs, mut body)) = extract_let_block(&expr) {
        // Let block: emit defs, then yield the last body expression
        let mut contents = defs;
        if body.is_empty() {
            contents.push(Value::List(vec![Value::symbol("scf.yield")]));
        } else {
            let last = body.pop().unwrap();
            contents.extend(body);
            contents.push(Value::List(vec![Value::symbol("scf.yield"), last]));
        }
        contents
    } else {
        // Single expression - just yield it
        vec![Value::List(vec![Value::symbol("scf.yield"), expr])]
    };

    // Build (block [] contents...)
    let mut block = vec![Value::symbol("block"), Value::Vector(vec![])];
    block.extend(block_contents);
    let block_value = Value::List(block);

    // Build (region block)
    Value::List(vec![Value::symbol("region"), block_value])
}

impl Macro for IfMacro {
    fn name(&self) -> &str {
        "if"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // Check for optional type map: (if {:result type} condition then-expr else-expr)
        let (type_map, condition, then_expr, else_expr) = match args.len() {
            3 => {
                // (if condition then-expr else-expr) - void return
                (None, args[0].clone(), args[1].clone(), args[2].clone())
            }
            4 => {
                // (if {:result type} condition then-expr else-expr) - typed return
                if let Value::Map(_) = &args[0] {
                    (Some(args[0].clone()), args[1].clone(), args[2].clone(), args[3].clone())
                } else {
                    return Err(MacroError::InvalidSyntax {
                        macro_name: "if".into(),
                        message: "first argument must be a map with :result type".into(),
                    });
                }
            }
            _ => {
                return Err(MacroError::WrongArity {
                    macro_name: "if".into(),
                    expected: "3 (cond, then, else) or 4 ({:result type}, cond, then, else)".into(),
                    got: args.len(),
                });
            }
        };

        // Build regions (handles do blocks specially)
        let then_region = build_region_block(then_expr);
        let else_region = build_region_block(else_expr);

        // Build scf.if with both regions, optionally with type map
        match type_map {
            Some(tmap) => Ok(Value::List(vec![
                Value::symbol("scf.if"),
                tmap,
                condition,
                then_region,
                else_region,
            ])),
            None => Ok(Value::List(vec![
                Value::symbol("scf.if"),
                condition,
                then_region,
                else_region,
            ])),
        }
    }

    fn doc(&self) -> Option<&str> {
        Some("Conditional with else branch: (if condition then-expr else-expr)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_if_basic() {
        let macro_impl = IfMacro;

        // (if cond then-val else-val)
        let args = vec![
            Value::symbol("cond"),
            Value::symbol("then-val"),
            Value::symbol("else-val"),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should be (scf.if cond (region ...) (region ...))
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 4);

            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "scf.if");
            } else {
                panic!("Expected scf.if symbol");
            }

            // Second item is the condition
            if let Value::Symbol(sym) = &items[1] {
                assert_eq!(sym.name, "cond");
            } else {
                panic!("Expected condition symbol");
            }

            // Third item is the then-region
            if let Value::List(region_items) = &items[2] {
                if let Value::Symbol(sym) = &region_items[0] {
                    assert_eq!(sym.name, "region");
                } else {
                    panic!("Expected region symbol");
                }
            } else {
                panic!("Expected then-region list");
            }

            // Fourth item is the else-region
            if let Value::List(region_items) = &items[3] {
                if let Value::Symbol(sym) = &region_items[0] {
                    assert_eq!(sym.name, "region");
                } else {
                    panic!("Expected region symbol");
                }
            } else {
                panic!("Expected else-region list");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_if_wrong_arity() {
        let macro_impl = IfMacro;

        // (if cond then-val) - missing else
        let args = vec![Value::symbol("cond"), Value::symbol("then-val")];

        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }
}
