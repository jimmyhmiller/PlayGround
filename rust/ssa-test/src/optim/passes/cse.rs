//! Common Subexpression Elimination (CSE).
//!
//! Finds redundant computations of the same expression and reuses the first result.
//!
//! If we have:
//!   v1 := a + b
//!   ...
//!   v2 := a + b  (same expression)
//!
//! We can replace uses of v2 with v1 (and DCE will remove the second instruction).

use std::collections::HashMap;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::SsaVariable;

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction, ExpressionKey};

/// Common Subexpression Elimination pass.
///
/// # Algorithm
/// 1. For each instruction, compute its `expression_key()`
/// 2. If we've seen this key before, replace the destination with the earlier result
/// 3. If not, record this key -> destination mapping
///
/// # Limitations
/// - Simple local CSE within a block (not global CSE)
/// - Does not account for dominance (could eliminate expressions that aren't guaranteed to execute)
/// - For full effectiveness, run with copy propagation and DCE
///
/// # Note
/// This is a simplified CSE. A more sophisticated implementation would:
/// - Consider dominance (only reuse if earlier computation dominates)
/// - Handle global CSE across blocks
/// - Use value numbering for more opportunities
#[derive(Debug, Clone, Default)]
pub struct CommonSubexpressionElimination {
    /// Whether to do global CSE (across blocks) or just local
    global: bool,
}

impl CommonSubexpressionElimination {
    /// Create a new local CSE pass.
    pub fn new() -> Self {
        CommonSubexpressionElimination { global: false }
    }

    /// Create a global CSE pass (less conservative, may not be correct without dominance info).
    pub fn global() -> Self {
        CommonSubexpressionElimination { global: true }
    }
}

impl<V, I, F> OptimizationPass<V, I, F> for CommonSubexpressionElimination
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();

        // Map from expression key to the variable that holds its result
        let mut global_available: HashMap<ExpressionKey<V>, SsaVariable> = HashMap::new();

        // Map from original variable to replacement variable
        let mut replacements: HashMap<SsaVariable, SsaVariable> = HashMap::new();

        // First pass: find common subexpressions
        for block in &translator.blocks {
            // Local available expressions (reset per block unless global)
            let mut local_available: HashMap<ExpressionKey<V>, SsaVariable> = if self.global {
                global_available.clone()
            } else {
                HashMap::new()
            };

            for instr in &block.instructions {
                // Get the expression key for this instruction
                let key = match instr.expression_key() {
                    Some(k) => k,
                    None => continue,
                };

                // Get the destination
                let dest = match instr.destination() {
                    Some(d) => d.clone(),
                    None => continue,
                };

                // Check if we've computed this expression before
                if let Some(existing) = local_available.get(&key) {
                    // Found a common subexpression!
                    // Record that we should replace uses of dest with existing
                    replacements.insert(dest, existing.clone());
                    stats.cse_eliminations += 1;
                } else {
                    // First time seeing this expression
                    local_available.insert(key, dest);
                }
            }

            if self.global {
                global_available = local_available;
            }
        }

        if replacements.is_empty() {
            return PassResult::unchanged();
        }

        // Second pass: replace uses of eliminated expressions
        for block in &mut translator.blocks {
            for instr in &mut block.instructions {
                instr.visit_values_mut(|value| {
                    if let Some(var) = value.as_var() {
                        if let Some(replacement) = replacements.get(var) {
                            *value = V::from_var(replacement.clone());
                        }
                    }
                });
            }
        }

        // Replace in phi operands
        for phi in translator.phis.values_mut() {
            for operand in &mut phi.operands {
                if let Some(var) = operand.as_var() {
                    if let Some(replacement) = replacements.get(var) {
                        *operand = V::from_var(replacement.clone());
                    }
                }
            }
        }

        PassResult::changed(stats)
    }

    fn invalidates(&self) -> Invalidations {
        Invalidations {
            liveness: true,
            use_def: true,
            dominators: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cse_new() {
        let cse = CommonSubexpressionElimination::new();
        assert!(!cse.global);
    }

    #[test]
    fn test_cse_global() {
        let cse = CommonSubexpressionElimination::global();
        assert!(cse.global);
    }
}
