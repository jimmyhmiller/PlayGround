//! Copy Propagation.
//!
//! Replaces uses of copied variables with the original source.
//!
//! For a copy `y := x`, replace all uses of `y` with `x`.
//! In SSA form, this is safe because each variable has exactly one definition.

use std::collections::HashMap;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::SsaVariable;

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction};

/// Copy Propagation pass.
///
/// For each copy instruction `dest := src`, replaces all uses of `dest` with `src`.
///
/// # Algorithm
/// 1. Find all copy instructions (identified by `as_copy()`)
/// 2. Build a copy map: dest -> src (following chains: if y := x and z := y, then z -> x)
/// 3. Replace all variable uses according to the copy map
/// 4. Copy instructions become dead and can be removed by DCE
///
/// Note: This pass does NOT remove the copy instructions themselves.
/// Run DCE after copy propagation to clean them up.
#[derive(Debug, Clone, Default)]
pub struct CopyPropagation;

impl CopyPropagation {
    pub fn new() -> Self {
        CopyPropagation
    }

    /// Build the copy map, following chains.
    fn build_copy_map<V, I, F>(
        &self,
        translator: &SSATranslator<V, I, F>,
    ) -> HashMap<SsaVariable, V>
    where
        V: OptimizableValue,
        I: OptimizableInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        // First pass: collect all direct copies
        let mut direct_copies: HashMap<SsaVariable, V> = HashMap::new();

        for block in &translator.blocks {
            for instr in &block.instructions {
                if let Some((dest, src)) = instr.as_copy() {
                    // Only propagate variable copies, not constants or phis
                    if src.as_var().is_some() {
                        direct_copies.insert(dest.clone(), src.clone());
                    }
                }
            }
        }

        // Second pass: follow chains
        // If we have y := x and z := y, we want z -> x
        let mut resolved: HashMap<SsaVariable, V> = HashMap::new();

        fn resolve<V: OptimizableValue>(
            var: &SsaVariable,
            direct: &HashMap<SsaVariable, V>,
            resolved: &mut HashMap<SsaVariable, V>,
            visited: &mut Vec<SsaVariable>,
        ) -> V {
            // Check if already resolved
            if let Some(result) = resolved.get(var) {
                return result.clone();
            }

            // Check for cycles (shouldn't happen in valid SSA, but be safe)
            if visited.contains(var) {
                return V::from_var(var.clone());
            }

            // Check if there's a copy for this variable
            if let Some(src) = direct.get(var) {
                if let Some(src_var) = src.as_var() {
                    visited.push(var.clone());
                    let result = resolve(src_var, direct, resolved, visited);
                    visited.pop();
                    resolved.insert(var.clone(), result.clone());
                    return result;
                } else {
                    // Source is not a variable (e.g., constant)
                    resolved.insert(var.clone(), src.clone());
                    return src.clone();
                }
            }

            // No copy - use the variable itself
            V::from_var(var.clone())
        }

        // Resolve all copies
        let keys: Vec<_> = direct_copies.keys().cloned().collect();
        for var in keys {
            let mut visited = Vec::new();
            resolve(&var, &direct_copies, &mut resolved, &mut visited);
        }

        resolved
    }
}

impl<V, I, F> OptimizationPass<V, I, F> for CopyPropagation
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    fn name(&self) -> &'static str {
        "copy-prop"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();

        // Build copy map
        let copy_map = self.build_copy_map(translator);

        if copy_map.is_empty() {
            return PassResult::unchanged();
        }

        // Replace uses in instructions
        for block in &mut translator.blocks {
            for instr in &mut block.instructions {
                instr.visit_values_mut(|value| {
                    if let Some(var) = value.as_var() {
                        if let Some(replacement) = copy_map.get(var) {
                            *value = replacement.clone();
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
                    if let Some(replacement) = copy_map.get(var) {
                        *operand = replacement.clone();
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
        // Copy prop changes uses but not defs
        Invalidations {
            liveness: true, // Live ranges change
            use_def: true,  // Use-def chains change
            dominators: false, // Control flow unchanged
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_prop_new() {
        let _cp = CopyPropagation::new();
        // Basic instantiation test - full tests in integration tests
    }
}
