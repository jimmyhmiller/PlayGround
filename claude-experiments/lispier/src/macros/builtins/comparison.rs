//! Comparison operator macros for high-level syntax
//!
//! Integer comparisons: <=i, <i, >=i, >i, =i, !=i
//!
//! Example:
//!   (<=i x y)  -> (arith.cmpi {:predicate "sle"} x y)
//!   (=i a b)   -> (arith.cmpi {:predicate "eq"} a b)

use std::collections::HashMap;

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Helper to create a comparison macro expansion
fn comparison_expand(
    macro_name: &str,
    predicate: &str,
    args: &[Value],
) -> Result<Value, MacroError> {
    if args.len() != 2 {
        return Err(MacroError::WrongArity {
            macro_name: macro_name.into(),
            expected: "2".into(),
            got: args.len(),
        });
    }

    let mut attrs = HashMap::new();
    attrs.insert("predicate".to_string(), Value::String(predicate.to_string()));

    Ok(Value::List(vec![
        Value::symbol("arith.cmpi"),
        Value::Map(attrs),
        args[0].clone(),
        args[1].clone(),
    ]))
}

// =============================================================================
// Integer Comparison Macros
// =============================================================================

/// Less than or equal (signed): (<=i a b) -> (arith.cmpi {:predicate "sle"} a b)
pub struct LeIMacro;

impl Macro for LeIMacro {
    fn name(&self) -> &str {
        "<=i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        comparison_expand("<=i", "sle", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Signed less-or-equal: (<=i a b) -> (arith.cmpi {:predicate \"sle\"} a b)")
    }
}

/// Less than (signed): (<i a b) -> (arith.cmpi {:predicate "slt"} a b)
pub struct LtIMacro;

impl Macro for LtIMacro {
    fn name(&self) -> &str {
        "<i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        comparison_expand("<i", "slt", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Signed less-than: (<i a b) -> (arith.cmpi {:predicate \"slt\"} a b)")
    }
}

/// Greater than or equal (signed): (>=i a b) -> (arith.cmpi {:predicate "sge"} a b)
pub struct GeIMacro;

impl Macro for GeIMacro {
    fn name(&self) -> &str {
        ">=i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        comparison_expand(">=i", "sge", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Signed greater-or-equal: (>=i a b) -> (arith.cmpi {:predicate \"sge\"} a b)")
    }
}

/// Greater than (signed): (>i a b) -> (arith.cmpi {:predicate "sgt"} a b)
pub struct GtIMacro;

impl Macro for GtIMacro {
    fn name(&self) -> &str {
        ">i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        comparison_expand(">i", "sgt", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Signed greater-than: (>i a b) -> (arith.cmpi {:predicate \"sgt\"} a b)")
    }
}

/// Equal: (=i a b) -> (arith.cmpi {:predicate "eq"} a b)
pub struct EqIMacro;

impl Macro for EqIMacro {
    fn name(&self) -> &str {
        "=i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        comparison_expand("=i", "eq", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Integer equality: (=i a b) -> (arith.cmpi {:predicate \"eq\"} a b)")
    }
}

/// Not equal: (!=i a b) -> (arith.cmpi {:predicate "ne"} a b)
pub struct NeIMacro;

impl Macro for NeIMacro {
    fn name(&self) -> &str {
        "!=i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        comparison_expand("!=i", "ne", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Integer not-equal: (!=i a b) -> (arith.cmpi {:predicate \"ne\"} a b)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_le_i() {
        let macro_impl = LeIMacro;
        let args = vec![Value::symbol("x"), Value::symbol("y")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 4);
            assert_eq!(items[0].as_symbol().name, "arith.cmpi");
            // items[1] is the map with predicate
            if let Value::Map(m) = &items[1] {
                if let Some(Value::String(pred)) = m.get("predicate") {
                    assert_eq!(pred, "sle");
                } else {
                    panic!("Expected predicate string");
                }
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
    fn test_wrong_arity() {
        let macro_impl = EqIMacro;
        let args = vec![Value::symbol("x")];
        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }
}
