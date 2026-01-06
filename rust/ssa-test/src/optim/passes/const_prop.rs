//! Constant Propagation.
//!
//! Replaces uses of variables with their constant values when known.
//! Works in tandem with constant folding - propagate constants to enable more folding.

use std::collections::HashMap;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::SsaVariable;

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction};

/// Constant Propagation pass.
///
/// For each variable known to be constant, replaces uses of that variable
/// with the constant value.
///
/// # Algorithm
/// 1. Scan all instructions to find constant assignments (dest := const)
/// 2. Build a map of variable -> constant value
/// 3. Replace uses of those variables with the constant values
///
/// Note: In SSA form, each variable has one definition, so if a variable
/// is defined as a constant, all uses can be replaced.
#[derive(Debug, Clone, Default)]
pub struct ConstantPropagation;

impl ConstantPropagation {
    pub fn new() -> Self {
        ConstantPropagation
    }

    /// Find all variables that are defined as constants.
    fn find_constants<V, I, F>(
        &self,
        translator: &SSATranslator<V, I, F>,
    ) -> HashMap<SsaVariable, V>
    where
        V: OptimizableValue,
        I: OptimizableInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        let mut constants = HashMap::new();

        for block in &translator.blocks {
            for instr in &block.instructions {
                // Check if this is a copy of a constant
                if let Some((dest, value)) = instr.as_copy() {
                    if value.is_constant() {
                        constants.insert(dest.clone(), value.clone());
                    }
                }
            }
        }

        constants
    }
}

impl<V, I, F> OptimizationPass<V, I, F> for ConstantPropagation
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    fn name(&self) -> &'static str {
        "const-prop"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();

        // Find all constant definitions
        let constants = self.find_constants(translator);

        if constants.is_empty() {
            return PassResult::unchanged();
        }

        // Replace uses in instructions
        for block in &mut translator.blocks {
            for instr in &mut block.instructions {
                instr.visit_values_mut(|value| {
                    if let Some(var) = value.as_var() {
                        if let Some(constant) = constants.get(var) {
                            *value = constant.clone();
                            stats.values_propagated += 1;
                        }
                    }
                });
            }
        }

        // Replace uses in phi operands
        for phi in translator.phis.values_mut() {
            for operand in &mut phi.operands {
                if let Some(var) = operand.as_var() {
                    if let Some(constant) = constants.get(var) {
                        *operand = constant.clone();
                        stats.values_propagated += 1;
                    }
                }
            }
        }

        if stats.values_propagated > 0 {
            PassResult::changed(stats)
        } else {
            PassResult::unchanged()
        }
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
    fn test_const_prop_new() {
        let _cp = ConstantPropagation::new();
        // Basic instantiation test - full tests in integration tests
    }
}
