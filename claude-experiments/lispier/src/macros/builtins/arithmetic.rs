//! Arithmetic operator macros for high-level syntax
//!
//! Integer operators: +i, -i, *i, /i, /ui
//! Float operators: +f, -f, *f, /f
//!
//! Example:
//!   (+i x y)  -> (arith.addi x y)
//!   (*f a b)  -> (arith.mulf a b)

use crate::macros::{Macro, MacroError};
use crate::value::Value;

/// Helper to create a binary arithmetic macro
fn binary_arith_expand(
    macro_name: &str,
    mlir_op: &str,
    args: &[Value],
) -> Result<Value, MacroError> {
    if args.len() != 2 {
        return Err(MacroError::WrongArity {
            macro_name: macro_name.into(),
            expected: "2".into(),
            got: args.len(),
        });
    }
    Ok(Value::List(vec![
        Value::symbol(mlir_op),
        args[0].clone(),
        args[1].clone(),
    ]))
}

// =============================================================================
// Integer Arithmetic Macros
// =============================================================================

/// Integer addition: (+i a b) -> (arith.addi a b)
pub struct AddIMacro;

impl Macro for AddIMacro {
    fn name(&self) -> &str {
        "+i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("+i", "arith.addi", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Integer addition: (+i a b) -> (arith.addi a b)")
    }
}

/// Integer subtraction: (-i a b) -> (arith.subi a b)
pub struct SubIMacro;

impl Macro for SubIMacro {
    fn name(&self) -> &str {
        "-i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("-i", "arith.subi", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Integer subtraction: (-i a b) -> (arith.subi a b)")
    }
}

/// Integer multiplication: (*i a b) -> (arith.muli a b)
pub struct MulIMacro;

impl Macro for MulIMacro {
    fn name(&self) -> &str {
        "*i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("*i", "arith.muli", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Integer multiplication: (*i a b) -> (arith.muli a b)")
    }
}

/// Signed integer division: (/i a b) -> (arith.divsi a b)
pub struct DivSIMacro;

impl Macro for DivSIMacro {
    fn name(&self) -> &str {
        "/i"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("/i", "arith.divsi", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Signed integer division: (/i a b) -> (arith.divsi a b)")
    }
}

/// Unsigned integer division: (/ui a b) -> (arith.divui a b)
pub struct DivUIMacro;

impl Macro for DivUIMacro {
    fn name(&self) -> &str {
        "/ui"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("/ui", "arith.divui", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Unsigned integer division: (/ui a b) -> (arith.divui a b)")
    }
}

// =============================================================================
// Float Arithmetic Macros
// =============================================================================

/// Float addition: (+f a b) -> (arith.addf a b)
pub struct AddFMacro;

impl Macro for AddFMacro {
    fn name(&self) -> &str {
        "+f"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("+f", "arith.addf", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Float addition: (+f a b) -> (arith.addf a b)")
    }
}

/// Float subtraction: (-f a b) -> (arith.subf a b)
pub struct SubFMacro;

impl Macro for SubFMacro {
    fn name(&self) -> &str {
        "-f"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("-f", "arith.subf", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Float subtraction: (-f a b) -> (arith.subf a b)")
    }
}

/// Float multiplication: (*f a b) -> (arith.mulf a b)
pub struct MulFMacro;

impl Macro for MulFMacro {
    fn name(&self) -> &str {
        "*f"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("*f", "arith.mulf", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Float multiplication: (*f a b) -> (arith.mulf a b)")
    }
}

/// Float division: (/f a b) -> (arith.divf a b)
pub struct DivFMacro;

impl Macro for DivFMacro {
    fn name(&self) -> &str {
        "/f"
    }

    fn expand(&self, args: &[Value]) -> Result<Value, MacroError> {
        binary_arith_expand("/f", "arith.divf", args)
    }

    fn doc(&self) -> Option<&str> {
        Some("Float division: (/f a b) -> (arith.divf a b)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_i() {
        let macro_impl = AddIMacro;
        let args = vec![Value::symbol("x"), Value::symbol("y")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_symbol().name, "arith.addi");
            assert_eq!(items[1].as_symbol().name, "x");
            assert_eq!(items[2].as_symbol().name, "y");
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_wrong_arity() {
        let macro_impl = AddIMacro;
        let args = vec![Value::symbol("x")];
        let result = macro_impl.expand(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_f() {
        let macro_impl = DivFMacro;
        let args = vec![Value::symbol("a"), Value::symbol("b")];
        let result = macro_impl.expand(&args).unwrap();

        if let Value::List(items) = result {
            assert_eq!(items[0].as_symbol().name, "arith.divf");
        } else {
            panic!("Expected list");
        }
    }
}
