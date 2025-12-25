//! The `and` and `or` macros for short-circuiting boolean operations
//!
//! Syntax:
//!   (and expr1 expr2 ...)
//!   (or expr1 expr2 ...)
//!
//! These use scf.if for short-circuit evaluation.

use crate::macros::{Macro, MacroError};
use crate::value::Value;
use std::collections::HashMap;

/// Helper to build {:result i1} map for typed if
fn i1_result_map() -> Value {
    let mut map = HashMap::new();
    map.insert("result".to_string(), Value::symbol("i1"));
    Value::Map(map)
}

/// The `and` macro for short-circuiting logical and
///
/// (and) -> true (i1 1)
/// (and a) -> a
/// (and a b) -> (if a b false)
/// (and a b c ...) -> (if a (and b c ...) false)
pub struct AndMacro;

impl Macro for AndMacro {
    fn name(&self) -> &str {
        "and"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        match args.len() {
            0 => {
                // (and) -> true
                Ok(Value::List(vec![
                    Value::symbol(":"),
                    Value::Number(1.0),
                    Value::symbol("i1"),
                ]))
            }
            1 => {
                // (and a) -> a
                Ok(args[0].clone())
            }
            _ => {
                // (and a b ...) -> (if {:result i1} a (and b ...) false)
                let first = args[0].clone();
                let rest: Vec<Value> = args[1..].to_vec();

                // Build (and rest...)
                let mut and_rest = vec![Value::symbol("and")];
                and_rest.extend(rest);

                // Build false literal
                let false_val = Value::List(vec![
                    Value::symbol(":"),
                    Value::Number(0.0),
                    Value::symbol("i1"),
                ]);

                // Build (if {:result i1} first (and rest...) false)
                Ok(Value::List(vec![
                    Value::symbol("if"),
                    i1_result_map(),
                    first,
                    Value::List(and_rest),
                    false_val,
                ]))
            }
        }
    }

    fn doc(&self) -> Option<&str> {
        Some("Short-circuiting logical and: (and expr1 expr2 ...)")
    }
}

/// The `or` macro for short-circuiting logical or
///
/// (or) -> false (i1 0)
/// (or a) -> a
/// (or a b) -> (if a true b)
/// (or a b c ...) -> (if a true (or b c ...))
pub struct OrMacro;

impl Macro for OrMacro {
    fn name(&self) -> &str {
        "or"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        match args.len() {
            0 => {
                // (or) -> false
                Ok(Value::List(vec![
                    Value::symbol(":"),
                    Value::Number(0.0),
                    Value::symbol("i1"),
                ]))
            }
            1 => {
                // (or a) -> a
                Ok(args[0].clone())
            }
            _ => {
                // (or a b ...) -> (if {:result i1} a true (or b ...))
                let first = args[0].clone();
                let rest: Vec<Value> = args[1..].to_vec();

                // Build (or rest...)
                let mut or_rest = vec![Value::symbol("or")];
                or_rest.extend(rest);

                // Build true literal
                let true_val = Value::List(vec![
                    Value::symbol(":"),
                    Value::Number(1.0),
                    Value::symbol("i1"),
                ]);

                // Build (if {:result i1} first true (or rest...))
                Ok(Value::List(vec![
                    Value::symbol("if"),
                    i1_result_map(),
                    first,
                    true_val,
                    Value::List(or_rest),
                ]))
            }
        }
    }

    fn doc(&self) -> Option<&str> {
        Some("Short-circuiting logical or: (or expr1 expr2 ...)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_empty() {
        let macro_impl = AndMacro;
        let result = macro_impl.expand(&[]).unwrap();
        // Should be (: 1 i1)
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_and_single() {
        let macro_impl = AndMacro;
        let args = vec![Value::symbol("a")];
        let result = macro_impl.expand(&args).unwrap();
        // Should be just a
        if let Value::Symbol(sym) = &result {
            assert_eq!(sym.name, "a");
        } else {
            panic!("Expected symbol");
        }
    }

    #[test]
    fn test_and_two() {
        let macro_impl = AndMacro;
        let args = vec![Value::symbol("a"), Value::symbol("b")];
        let result = macro_impl.expand(&args).unwrap();
        // Should be (if a (and b) false)
        if let Value::List(items) = &result {
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "if");
            } else {
                panic!("Expected if symbol");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_or_empty() {
        let macro_impl = OrMacro;
        let result = macro_impl.expand(&[]).unwrap();
        // Should be (: 0 i1)
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_or_two() {
        let macro_impl = OrMacro;
        let args = vec![Value::symbol("a"), Value::symbol("b")];
        let result = macro_impl.expand(&args).unwrap();
        // Should be (if a true (or b))
        if let Value::List(items) = &result {
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "if");
            } else {
                panic!("Expected if symbol");
            }
        } else {
            panic!("Expected list");
        }
    }
}
