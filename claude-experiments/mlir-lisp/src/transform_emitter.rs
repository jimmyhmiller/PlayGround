/// Transform Emitter - Generate Transform dialect IR from Lisp pattern definitions
///
/// Takes PDL pattern definitions and generates Transform dialect IR that can be
/// applied using apply_named_sequence().

use crate::dialect_registry::PdlPattern;
use crate::parser::Value;

pub struct TransformEmitter;

impl TransformEmitter {
    /// Generate a complete transform module from PDL patterns
    ///
    /// Example output:
    /// ```mlir
    /// module {
    ///   transform.with_pdl_patterns {
    ///   ^bb0(%root: !transform.any_op):
    ///     pdl.pattern @calc_to_arith : benefit(1) {
    ///       %lhs = pdl.operand
    ///       %rhs = pdl.operand
    ///       %result_type = pdl.type
    ///       %op = pdl.operation "calc.add"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
    ///       pdl.rewrite %op {
    ///         %new_op = pdl.operation "arith.addi"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
    ///         pdl.replace %op with %new_op
    ///       }
    ///     }
    ///
    ///     transform.sequence %root : !transform.any_op failures(propagate) {
    ///     ^bb1(%arg1: !transform.any_op):
    ///       %matched = pdl_match @calc_to_arith in %arg1 : (!transform.any_op) -> !transform.any_op
    ///       transform.yield
    ///     }
    ///   }
    /// }
    /// ```
    pub fn emit_transform_module(patterns: &[PdlPattern]) -> Result<String, String> {
        if patterns.is_empty() {
            return Err("No patterns to emit".into());
        }

        let mut ir = String::from("module {\n");
        ir.push_str("  transform.with_pdl_patterns {\n");
        ir.push_str("  ^bb0(%root: !transform.any_op):\n");

        // Emit all PDL patterns
        for pattern in patterns {
            ir.push_str(&Self::emit_pdl_pattern(pattern)?);
        }

        // Emit transform sequence that applies all patterns
        ir.push_str(&Self::emit_transform_sequence(patterns));

        ir.push_str("  }\n");
        ir.push_str("}\n");

        Ok(ir)
    }

    /// Emit a single PDL pattern
    fn emit_pdl_pattern(pattern: &PdlPattern) -> Result<String, String> {
        let mut ir = format!("    pdl.pattern @{} : benefit({}) {{\n",
            pattern.name, pattern.benefit);

        // Extract match and rewrite operations from the pattern bodies
        let (match_op, match_operands) = Self::extract_operation_info(&pattern.match_body)?;
        let (rewrite_op, _) = Self::extract_operation_info(&pattern.rewrite_body)?;

        // Determine operand count
        let operand_count = match_operands.len();

        // Emit operand declarations
        for i in 0..operand_count {
            ir.push_str(&format!("      %operand{} = pdl.operand\n", i));
        }

        // Emit type declaration
        ir.push_str("      %result_type = pdl.type\n");

        // Emit match operation
        let operand_refs: Vec<String> = (0..operand_count)
            .map(|i| format!("%operand{}", i))
            .collect();

        let operand_str = if operand_refs.is_empty() {
            String::new()
        } else {
            format!("({} : {})",
                operand_refs.join(", "),
                vec!["!pdl.value"; operand_count].join(", "))
        };

        ir.push_str(&format!("      %matched_op = pdl.operation \"{}\"{}-> (%result_type : !pdl.type)\n",
            match_op, operand_str));

        // Emit rewrite block
        ir.push_str("      pdl.rewrite %matched_op {\n");

        ir.push_str(&format!("        %new_op = pdl.operation \"{}\"{}-> (%result_type : !pdl.type)\n",
            rewrite_op, operand_str));

        ir.push_str("        pdl.replace %matched_op with %new_op\n");
        ir.push_str("      }\n");
        ir.push_str("    }\n\n");

        Ok(ir)
    }

    /// Emit the transform.sequence that applies all patterns
    fn emit_transform_sequence(patterns: &[PdlPattern]) -> String {
        let mut ir = String::from("    transform.sequence %root : !transform.any_op failures(propagate) {\n");
        ir.push_str("    ^bb1(%arg1: !transform.any_op):\n");

        for pattern in patterns {
            ir.push_str(&format!("      %{}_matched = pdl_match @{} in %arg1 : (!transform.any_op) -> !transform.any_op\n",
                pattern.name, pattern.name));
        }

        ir.push_str("      transform.yield\n");
        ir.push_str("    }\n");

        ir
    }

    /// Extract operation name and operands from a pattern body
    ///
    /// Handles forms like:
    /// - (calc.add $x $y)
    /// - List with operation at index 0
    fn extract_operation_info(body: &Value) -> Result<(String, Vec<String>), String> {
        match body {
            Value::List(elements) if !elements.is_empty() => {
                let op_name = match &elements[0] {
                    Value::Symbol(s) => s.clone(),
                    _ => return Err("Pattern operation must be a symbol".into()),
                };

                let mut operands = Vec::new();
                for elem in &elements[1..] {
                    if let Value::Symbol(s) = elem {
                        operands.push(s.clone());
                    }
                }

                Ok((op_name, operands))
            }
            _ => Err("Pattern body must be a list".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect_registry::PdlPattern;
    use crate::parser::Value;

    #[test]
    fn test_emit_transform_module() {
        let patterns = vec![
            PdlPattern {
                name: "calc_add_to_arith".to_string(),
                benefit: 1,
                description: "Lower calc.add to arith.addi".to_string(),
                match_body: Value::List(vec![
                    Value::Symbol("calc.add".to_string()),
                    Value::Symbol("$x".to_string()),
                    Value::Symbol("$y".to_string()),
                ]),
                rewrite_body: Value::List(vec![
                    Value::Symbol("arith.addi".to_string()),
                    Value::Symbol("$x".to_string()),
                    Value::Symbol("$y".to_string()),
                ]),
                constraints: vec![],
            },
        ];

        let ir = TransformEmitter::emit_transform_module(&patterns).unwrap();
        println!("{}", ir);

        assert!(ir.contains("module {"));
        assert!(ir.contains("transform.with_pdl_patterns"));
        assert!(ir.contains("pdl.pattern @calc_add_to_arith"));
        assert!(ir.contains("pdl.operation \"calc.add\""));
        assert!(ir.contains("pdl.operation \"arith.addi\""));
        assert!(ir.contains("pdl.replace"));
        assert!(ir.contains("transform.sequence"));
        assert!(ir.contains("pdl_match @calc_add_to_arith"));
    }
}
