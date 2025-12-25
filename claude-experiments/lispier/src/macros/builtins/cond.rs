//! The `cond` macro for multi-branch conditionals
//!
//! Syntax:
//!   (cond
//!     condition1 result1
//!     condition2 result2
//!     :else default)
//!
//! Or with result type:
//!   (cond {:result type}
//!     condition1 result1
//!     :else default)
//!
//! Expands to nested scf.if forms

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// The `cond` macro for multi-branch conditionals
pub struct CondMacro;

impl Macro for CondMacro {
    fn name(&self) -> &str {
        "cond"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        // Check for optional result type attribute: (cond {:result type} ...)
        let (result_attr, pairs_start) = if let Some(Value::Map(map)) = args.first() {
            if map.contains_key("result") {
                (Some(map.clone()), 1)
            } else {
                (None, 0)
            }
        } else {
            (None, 0)
        };

        let pair_args = &args[pairs_start..];

        if pair_args.len() % 2 != 0 {
            return Err(MacroError::ExpansionFailed(
                "cond requires pairs of (condition result)".into(),
            ));
        }

        if pair_args.is_empty() {
            return Err(MacroError::WrongArity {
                macro_name: "cond".into(),
                expected: "at least 2 (condition, result)".into(),
                got: 0,
            });
        }

        // Collect pairs
        let pairs: Vec<(&Value, &Value)> = pair_args
            .chunks(2)
            .map(|chunk| (&chunk[0], &chunk[1]))
            .collect();

        // Build nested if structure
        build_nested_if(&pairs, 0, &result_attr)
    }

    fn doc(&self) -> Option<&str> {
        Some("Multi-branch conditional. (cond cond1 result1 cond2 result2 :else default)")
    }
}

/// Build nested scf.if forms from condition/result pairs
fn build_nested_if(
    pairs: &[(&Value, &Value)],
    idx: usize,
    result_attr: &Option<HashMap<String, Value>>,
) -> Result<Value, MacroError> {
    if idx >= pairs.len() {
        return Err(MacroError::ExpansionFailed(
            "cond requires :else clause".into(),
        ));
    }

    let (condition, result) = pairs[idx];

    // Check for :else (keyword starting with colon or the keyword type)
    let is_else = match condition {
        Value::Keyword(kw) => kw == "else" || kw == ":else",
        Value::Symbol(sym) => sym.name == ":else",
        _ => false,
    };

    if is_else {
        // Base case: just yield the result
        return Ok(wrap_yield(result.clone()));
    }

    // Build then-region
    let then_region = Value::List(vec![
        Value::symbol("region"),
        Value::List(vec![
            Value::symbol("block"),
            Value::Vector(vec![]),
            wrap_yield(result.clone()),
        ]),
    ]);

    // Build else-region (recursively)
    let else_body = build_nested_if(pairs, idx + 1, result_attr)?;
    let else_region = Value::List(vec![
        Value::symbol("region"),
        Value::List(vec![
            Value::symbol("block"),
            Value::Vector(vec![]),
            else_body,
        ]),
    ]);

    // Build scf.if
    let mut if_items = vec![Value::symbol("scf.if")];

    // Add result attribute if present
    if let Some(attrs) = result_attr {
        if_items.push(Value::Map(attrs.clone()));
    }

    if_items.push(condition.clone());
    if_items.push(then_region);
    if_items.push(else_region);

    Ok(Value::List(if_items))
}

/// Wrap a value in an scf.yield
fn wrap_yield(value: Value) -> Value {
    Value::List(vec![Value::symbol("scf.yield"), value])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cond_basic() {
        let macro_impl = CondMacro;

        // (cond cond1 result1 :else default)
        let args = vec![
            Value::symbol("cond1"),
            Value::symbol("result1"),
            Value::Keyword("else".into()),
            Value::symbol("default"),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should be (scf.if cond1 (region ...) (region ...))
        if let Value::List(items) = &result {
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "scf.if");
            } else {
                panic!("Expected scf.if symbol");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_cond_with_result_type() {
        let macro_impl = CondMacro;

        // (cond {:result i32} cond1 result1 :else default)
        let mut attrs = HashMap::new();
        attrs.insert("result".to_string(), Value::symbol("i32"));

        let args = vec![
            Value::Map(attrs),
            Value::symbol("cond1"),
            Value::symbol("result1"),
            Value::Keyword("else".into()),
            Value::symbol("default"),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should have attributes map after scf.if
        if let Value::List(items) = &result {
            assert!(items.len() >= 4); // scf.if, attrs, condition, then, else
            if let Value::Map(map) = &items[1] {
                assert!(map.contains_key("result"));
            } else {
                panic!("Expected attributes map");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_cond_multiple_branches() {
        let macro_impl = CondMacro;

        // (cond cond1 result1 cond2 result2 :else default)
        let args = vec![
            Value::symbol("cond1"),
            Value::symbol("result1"),
            Value::symbol("cond2"),
            Value::symbol("result2"),
            Value::Keyword("else".into()),
            Value::symbol("default"),
        ];

        let result = macro_impl.expand(&args).unwrap();

        // Should create nested structure
        if let Value::List(items) = &result {
            // First level: scf.if cond1 then else
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "scf.if");
            }

            // else branch should contain another scf.if
            if let Value::List(else_region) = &items[3] {
                // region -> block -> ... scf.if inside
                if let Value::List(block) = &else_region[1] {
                    // The block body should contain another scf.if or yield
                    // block[2] is the body expression
                    if let Value::List(nested) = &block[2] {
                        if let Value::Symbol(sym) = &nested[0] {
                            assert_eq!(sym.name, "scf.if");
                        }
                    }
                }
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_cond_no_else() {
        let macro_impl = CondMacro;

        // (cond cond1 result1) - no :else
        let args = vec![Value::symbol("cond1"), Value::symbol("result1")];

        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }
}
