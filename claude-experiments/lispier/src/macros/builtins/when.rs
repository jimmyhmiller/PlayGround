//! The `when` macro for conditional execution (no else branch)
//!
//! Syntax:
//!   (when condition body...)
//!
//! Expands to:
//!   (scf.if condition
//!     (region
//!       (block []
//!         body...
//!         (scf.yield))))

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// The `when` macro for single-branch conditionals
pub struct WhenMacro;

impl Macro for WhenMacro {
    fn name(&self) -> &str {
        "when"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // (when condition body...)

        if args.is_empty() {
            return Err(MacroError::WrongArity {
                macro_name: "when".into(),
                expected: "at least 1 (condition)".into(),
                got: 0,
            });
        }

        let condition = args[0].clone();
        let body: Vec<Value> = args[1..].to_vec();

        // Build the then-region:
        // (region (block [] body... (scf.yield)))
        let mut block_body = vec![Value::symbol("block"), Value::Vector(vec![])];
        block_body.extend(body);
        block_body.push(Value::List(vec![Value::symbol("scf.yield")]));

        let then_region = Value::List(vec![Value::symbol("region"), Value::List(block_body)]);

        // Build scf.if with single then-region (no else)
        Ok(Value::List(vec![
            Value::symbol("scf.if"),
            condition,
            then_region,
        ]))
    }

    fn doc(&self) -> Option<&str> {
        Some("Execute body if condition is true. (when condition body...)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_when_basic() {
        let macro_impl = WhenMacro;

        // (when cond (do-something))
        let args = vec![
            Value::symbol("cond"),
            Value::List(vec![Value::symbol("do-something")]),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should be (scf.if cond (region ...))
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 3);

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

            // Third item is the region
            if let Value::List(region_items) = &items[2] {
                if let Value::Symbol(sym) = &region_items[0] {
                    assert_eq!(sym.name, "region");
                } else {
                    panic!("Expected region symbol");
                }
            } else {
                panic!("Expected region list");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_when_empty_body() {
        let macro_impl = WhenMacro;

        // (when cond) - just condition, no body
        let args = vec![Value::symbol("cond")];

        let result = macro_impl.expand(&args).unwrap();

        // Should still work, just with empty body before yield
        if let Value::List(items) = &result {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_when_no_args() {
        let macro_impl = WhenMacro;

        // (when) - no args should error
        let args: Vec<Value> = vec![];

        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }
}
