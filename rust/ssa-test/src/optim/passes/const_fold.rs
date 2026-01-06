//! Constant Folding.
//!
//! Evaluates constant expressions at compile time.
//! For an instruction where all operands are constants, compute the result
//! and replace the instruction with a constant assignment.

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction, InstructionMutator};

/// Constant Folding pass.
///
/// For each instruction that operates only on constants, evaluate the result
/// at compile time and replace the instruction with a constant assignment.
///
/// # Algorithm
/// 1. For each instruction with a destination:
/// 2. Try to fold using `try_fold()` (user-implemented)
/// 3. If successful, replace instruction with constant assignment
///
/// Requires the instruction factory to implement `InstructionMutator` to create
/// constant assignments.
#[derive(Debug, Clone, Default)]
pub struct ConstantFolding;

impl ConstantFolding {
    pub fn new() -> Self {
        ConstantFolding
    }
}

impl<V, I, F> OptimizationPass<V, I, F> for ConstantFolding
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I> + InstructionMutator,
{
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();

        // Fold constants in instructions
        for block in &mut translator.blocks {
            for instr in &mut block.instructions {
                // Skip instructions without destinations
                let dest = match instr.destination() {
                    Some(d) => d.clone(),
                    None => continue,
                };

                // Try to fold
                if let Some(constant) = instr.try_fold() {
                    // Replace with constant assignment
                    *instr = F::create_constant_assign(dest, constant);
                    stats.expressions_folded += 1;
                }
            }
        }

        if stats.expressions_folded > 0 {
            PassResult::changed(stats)
        } else {
            PassResult::unchanged()
        }
    }

    fn invalidates(&self) -> Invalidations {
        // Folding changes instruction contents
        Invalidations::instructions_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_fold_new() {
        let _cf = ConstantFolding::new();
        // Basic instantiation test - full tests in integration tests
    }
}
