/// IRDL Emitter - Generate IRDL dialect definition IR from Lisp definitions
///
/// Takes parsed IRDL dialect definitions from the DialectRegistry and generates
/// textual MLIR IR that can be parsed and loaded with load_irdl_dialects().

use crate::dialect_registry::{IrdlDialect, IrdlOperation};
use crate::parser::Value;

pub struct IrdlEmitter;

impl IrdlEmitter {
    /// Generate IRDL module IR from a dialect definition
    ///
    /// Example output:
    /// ```mlir
    /// irdl.dialect @calc {
    ///   irdl.operation @add {
    ///     %i32_type = irdl.is i32
    ///     irdl.operands(lhs: %i32_type, rhs: %i32_type)
    ///     irdl.results(result: %i32_type)
    ///   }
    /// }
    /// ```
    pub fn emit_dialect(dialect: &IrdlDialect) -> String {
        let mut ir = format!("irdl.dialect @{} {{\n", dialect.namespace);

        for op in &dialect.operations {
            ir.push_str(&Self::emit_operation(op));
        }

        ir.push_str("}\n");
        ir
    }

    /// Emit a single IRDL operation definition
    fn emit_operation(op: &IrdlOperation) -> String {
        let mut ir = format!("  irdl.operation @{} {{\n", op.name);

        // Emit type constraints - use a unified type for simplicity
        ir.push_str("    %i32_type = irdl.is i32\n");

        // Emit operands
        if !op.operands.is_empty() {
            let operand_specs = Self::format_parameters_simple(&op.operands);
            ir.push_str(&format!("    irdl.operands({})\n", operand_specs));
        }

        // Emit results
        if !op.results.is_empty() {
            let result_specs = Self::format_parameters_simple(&op.results);
            ir.push_str(&format!("    irdl.results({})\n", result_specs));
        }

        // Emit attributes if any
        if !op.attributes.is_empty() {
            // For now, skip attributes - can be added later
        }

        ir.push_str("  }\n");
        ir
    }

    /// Collect unique type names from parameter list
    fn collect_type_constraints(params: &[Value], types: &mut Vec<String>) {
        for param in params {
            if let Some(type_name) = Self::extract_type_name(param) {
                if !types.contains(&type_name) {
                    types.push(type_name);
                }
            }
        }
    }

    /// Extract type name from a parameter definition
    /// Handles: (lhs i32), i32, or just "i32"
    fn extract_type_name(param: &Value) -> Option<String> {
        match param {
            // (lhs i32) or (lhs: i32)
            Value::List(elements) if elements.len() >= 2 => {
                if let Value::Symbol(type_name) = &elements[1] {
                    Some(type_name.clone())
                } else {
                    None
                }
            }
            // Just i32
            Value::Symbol(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Format parameters for irdl.operands or irdl.results
    /// Example: "lhs: %i32_type, rhs: %i32_type"
    fn format_parameters_simple(params: &[Value]) -> String {
        params.iter().enumerate().map(|(idx, param)| {
            let param_name = Self::extract_param_name(param)
                .unwrap_or_else(|| format!("param{}", idx));

            format!("{}: %i32_type", param_name)
        }).collect::<Vec<_>>().join(", ")
    }

    /// Extract parameter name from definition
    /// (lhs i32) -> "lhs", i32 -> "operand"
    fn extract_param_name(param: &Value) -> Option<String> {
        match param {
            Value::List(elements) if !elements.is_empty() => {
                if let Value::Symbol(name) = &elements[0] {
                    Some(name.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect_registry::IrdlDialect;

    #[test]
    fn test_emit_simple_dialect() {
        let dialect = IrdlDialect {
            name: "calc".to_string(),
            namespace: "calc".to_string(),
            description: "Calculator dialect".to_string(),
            operations: vec![
                IrdlOperation {
                    name: "add".to_string(),
                    summary: "Addition".to_string(),
                    description: "".to_string(),
                    operands: vec![
                        Value::List(vec![Value::Symbol("lhs".to_string()), Value::Symbol("i32".to_string())]),
                        Value::List(vec![Value::Symbol("rhs".to_string()), Value::Symbol("i32".to_string())]),
                    ],
                    results: vec![
                        Value::List(vec![Value::Symbol("result".to_string()), Value::Symbol("i32".to_string())]),
                    ],
                    attributes: vec![],
                    traits: vec![],
                    constraints: vec![],
                },
            ],
        };

        let ir = IrdlEmitter::emit_dialect(&dialect);
        println!("{}", ir);

        assert!(ir.contains("irdl.dialect @calc"));
        assert!(ir.contains("irdl.operation @add"));
        assert!(ir.contains("irdl.is i32"));
        assert!(ir.contains("irdl.operands"));
        assert!(ir.contains("irdl.results"));
    }
}
