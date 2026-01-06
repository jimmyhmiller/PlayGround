//! Dead Code Elimination (DCE).
//!
//! Removes instructions whose results are never used.
//! An instruction can be eliminated if:
//! 1. It has no side effects
//! 2. It is not a terminator
//! 3. Its destination variable is never used
//!
//! This is an aggressive DCE that also removes dead phi nodes.

use std::collections::HashSet;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::SsaVariable;

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction};

/// Dead Code Elimination pass.
///
/// Removes instructions and phi nodes that define variables which are never used.
///
/// # Algorithm
/// 1. Build the set of all used variables by scanning all instructions and phi operands
/// 2. Mark all instructions with side effects as live
/// 3. Mark all terminators as live
/// 4. For each instruction that defines a variable:
///    - If the variable is used, mark instruction as live
///    - Otherwise, remove the instruction
/// 5. Remove unused phi nodes
#[derive(Debug, Clone, Default)]
pub struct DeadCodeElimination {
    /// Track statistics
    _stats: PassStats,
}

impl DeadCodeElimination {
    pub fn new() -> Self {
        DeadCodeElimination::default()
    }

    /// Iteratively compute the set of live variables.
    /// A variable is live if:
    /// 1. It's used by a side-effecting instruction or terminator
    /// 2. It's used by another live instruction
    fn compute_live_variables<V, I, F>(
        &self,
        translator: &SSATranslator<V, I, F>,
    ) -> HashSet<SsaVariable>
    where
        V: OptimizableValue,
        I: OptimizableInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        let mut live = HashSet::new();
        let mut worklist: Vec<SsaVariable> = Vec::new();

        // Initial pass: mark variables used by critical instructions
        for block in &translator.blocks {
            for instr in &block.instructions {
                if instr.has_side_effects() || instr.is_terminator() {
                    instr.visit_values(|value| {
                        if let Some(var) = value.as_var() {
                            if !live.contains(var) {
                                live.insert(var.clone());
                                worklist.push(var.clone());
                            }
                        }
                    });
                }
            }
        }

        // Build def-to-uses map for efficient lookup
        use std::collections::HashMap;
        let mut def_to_operands: HashMap<SsaVariable, Vec<SsaVariable>> = HashMap::new();

        for block in &translator.blocks {
            for instr in &block.instructions {
                if let Some(dest) = instr.destination() {
                    let mut operands = Vec::new();
                    instr.visit_values(|value| {
                        if let Some(var) = value.as_var() {
                            operands.push(var.clone());
                        }
                    });
                    def_to_operands.insert(dest.clone(), operands);
                }
            }
        }

        // Add phi definitions
        for phi in translator.phis.values() {
            if let Some(dest) = &phi.dest {
                let operands: Vec<SsaVariable> = phi
                    .operands
                    .iter()
                    .filter_map(|v| v.as_var().cloned())
                    .collect();
                def_to_operands.insert(dest.clone(), operands);
            }
        }

        // Build reverse map: variable -> instructions that use it
        let mut var_to_defs: HashMap<SsaVariable, Vec<SsaVariable>> = HashMap::new();
        for (def, operands) in &def_to_operands {
            for operand in operands {
                var_to_defs
                    .entry(operand.clone())
                    .or_default()
                    .push(def.clone());
            }
        }

        // Propagate liveness backwards
        while let Some(var) = worklist.pop() {
            // If this variable is defined by an instruction, make the operands live
            if let Some(operands) = def_to_operands.get(&var) {
                for operand in operands {
                    if !live.contains(operand) {
                        live.insert(operand.clone());
                        worklist.push(operand.clone());
                    }
                }
            }
        }

        live
    }
}

impl<V, I, F> OptimizationPass<V, I, F> for DeadCodeElimination
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();

        // Compute live variables
        let live_vars = self.compute_live_variables(translator);

        // Remove dead instructions
        for block in &mut translator.blocks {
            let original_count = block.instructions.len();

            block.instructions.retain(|instr| {
                // Keep instructions with side effects
                if instr.has_side_effects() {
                    return true;
                }

                // Keep terminators
                if instr.is_terminator() {
                    return true;
                }

                // Keep instructions whose destination is live
                if let Some(dest) = instr.destination() {
                    if live_vars.contains(dest) {
                        return true;
                    }
                    // Dead instruction - remove it
                    return false;
                }

                // No destination - keep (might be important)
                true
            });

            stats.instructions_removed += original_count - block.instructions.len();
        }

        // Remove dead phi nodes
        let phi_ids_to_remove: Vec<_> = translator
            .phis
            .iter()
            .filter_map(|(phi_id, phi)| {
                if let Some(dest) = &phi.dest {
                    if !live_vars.contains(dest) {
                        return Some(*phi_id);
                    }
                }
                None
            })
            .collect();

        for phi_id in phi_ids_to_remove {
            translator.phis.remove(&phi_id);
            stats.phis_simplified += 1;
        }

        if stats.instructions_removed > 0 || stats.phis_simplified > 0 {
            PassResult::changed(stats)
        } else {
            PassResult::unchanged()
        }
    }

    fn invalidates(&self) -> Invalidations {
        Invalidations::instructions_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dce_new() {
        let _dce = DeadCodeElimination::new();
        // Basic instantiation test - full tests in integration tests
    }
}
