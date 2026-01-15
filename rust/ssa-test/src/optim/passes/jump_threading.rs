//! Jump Threading Pass.
//!
//! Eliminates blocks that only contain an unconditional jump by rewriting
//! predecessors to jump directly to the target. This simplifies the CFG
//! and removes unnecessary intermediate blocks.
//!
//! # Example
//! Before:
//! ```text
//! Block A: ... br B
//! Block B: br C      <- trivial jump block
//! Block C: ...
//! ```
//!
//! After:
//! ```text
//! Block A: ... br C  <- jumps directly to C
//! Block B: (empty)   <- unreachable, will be cleaned up by CfgCleanup
//! Block C: ...
//! ```

use std::collections::HashMap;

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::BlockId;

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction, InstructionMutator};

/// Jump Threading pass.
///
/// Identifies blocks that only contain a single unconditional jump and rewrites
/// predecessors to bypass these trivial blocks.
///
/// # Algorithm
/// 1. Find all trivial jump blocks (single unconditional jump, no phis)
/// 2. Build a map from trivial block -> ultimate target (following chains)
/// 3. Rewrite all jump targets to use ultimate targets
/// 4. Mark trivial blocks as unreachable (CfgCleanup will remove them)
#[derive(Debug, Clone, Default)]
pub struct JumpThreading;

impl JumpThreading {
    pub fn new() -> Self {
        JumpThreading
    }
}

impl<V, I, F> OptimizationPass<V, I, F> for JumpThreading
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I> + InstructionMutator,
{
    fn name(&self) -> &'static str {
        "jump-threading"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();

        // Step 1: Find trivial jump blocks
        // A trivial jump block:
        // - Has exactly one instruction (an unconditional jump)
        // - Has no phis defined in it
        // - Is not the entry block
        let trivial_blocks: HashMap<BlockId, BlockId> = translator.blocks
            .iter()
            .filter(|block| {
                // Not entry block
                block.id.0 != 0
                // Has exactly one instruction
                && block.instructions.len() == 1
                // That instruction is an unconditional jump (has exactly one target)
                && block.instructions[0].jump_targets().len() == 1
                // No phis in this block
                && !translator.phis.values().any(|phi| phi.block_id == block.id)
            })
            .map(|block| {
                let target = block.instructions[0].jump_targets()[0];
                (block.id, target)
            })
            .collect();

        if trivial_blocks.is_empty() {
            return PassResult::unchanged();
        }

        // Step 2: Build ultimate target map (follow chains of trivial blocks)
        let ultimate_targets: HashMap<BlockId, BlockId> = trivial_blocks
            .keys()
            .map(|&block_id| {
                let mut current = block_id;
                // Follow the chain until we hit a non-trivial block
                while let Some(&next) = trivial_blocks.get(&current) {
                    if next == block_id {
                        // Cycle detected, stop here
                        break;
                    }
                    current = next;
                }
                // current is now either the target of a trivial block (non-trivial),
                // or we hit a cycle
                let ultimate = trivial_blocks.get(&current).copied().unwrap_or(current);
                (block_id, ultimate)
            })
            .collect();

        // Step 3: Rewrite jump targets and track edge changes
        // We need to track which edges are being added/removed for predecessor updates
        // Also track which trivial block was bypassed so we can copy its phi operands
        let mut changed = false;
        // (source, trivial_block) - edge being removed
        let mut edges_removed: Vec<(BlockId, BlockId)> = Vec::new();
        // (source, ultimate_target, trivial_block) - edge being added, with bypassed block
        let mut edges_added: Vec<(BlockId, BlockId, BlockId)> = Vec::new();

        for block in &mut translator.blocks {
            let source_id = block.id;
            for instr in &mut block.instructions {
                // Try to rewrite each trivial block target to its ultimate target
                for (&trivial_block, &ultimate_target) in &ultimate_targets {
                    if instr.rewrite_jump_target(trivial_block, ultimate_target) {
                        changed = true;
                        stats.instructions_removed += 1;
                        // Track the edge change with the trivial block that was bypassed
                        edges_removed.push((source_id, trivial_block));
                        edges_added.push((source_id, ultimate_target, trivial_block));
                    }
                }
            }
        }

        // Step 4: Update predecessor lists based on edge changes
        // For each removed edge: remove source from target's predecessors (and update phis)
        // For each added edge: add source to target's predecessors (and copy phi operands
        // from the bypassed trivial block)

        // First, collect phi operands to copy BEFORE modifying anything
        // Map: (ultimate_target, trivial_block) -> Vec<(PhiId, operand_value)>
        let mut phi_operands_to_copy: HashMap<(BlockId, BlockId), Vec<(crate::types::PhiId, V)>> = HashMap::new();
        for &(_, ultimate_target, trivial_block) in &edges_added {
            let key = (ultimate_target, trivial_block);
            if phi_operands_to_copy.contains_key(&key) {
                continue; // Already collected for this pair
            }

            // Find the phi operand index for trivial_block in ultimate_target's predecessors
            let preds = &translator.blocks[ultimate_target.0].predecessors;
            if let Some(trivial_idx) = preds.iter().position(|&p| p == trivial_block) {
                // Collect operands for all phis in ultimate_target
                let operands: Vec<_> = translator.phis
                    .iter()
                    .filter(|(_, phi)| phi.block_id == ultimate_target)
                    .filter_map(|(&phi_id, phi)| {
                        phi.operands.get(trivial_idx).cloned().map(|op| (phi_id, op))
                    })
                    .collect();
                phi_operands_to_copy.insert(key, operands);
            }
        }

        // Handle removed edges and their phi operands
        for (source, target) in &edges_removed {
            let block = &mut translator.blocks[target.0];
            if let Some(idx) = block.predecessors.iter().position(|&p| p == *source) {
                block.predecessors.remove(idx);

                // Remove corresponding phi operands
                for phi in translator.phis.values_mut() {
                    if phi.block_id == *target && idx < phi.operands.len() {
                        phi.operands.remove(idx);
                    }
                }
                changed = true;
            }
        }

        // Handle added edges
        // For trivial blocks being bypassed, copy the phi operand from the trivial block
        for (source, ultimate_target, trivial_block) in &edges_added {
            let block = &mut translator.blocks[ultimate_target.0];

            // Only add if not already a predecessor
            if !block.predecessors.contains(source) {
                block.predecessors.push(*source);

                // For phis: copy the operand from the trivial block being bypassed
                // Since trivial blocks have no phis, the value flowing through is unchanged
                let key = (*ultimate_target, *trivial_block);
                if let Some(operands) = phi_operands_to_copy.get(&key) {
                    // Use the pre-collected operands
                    for (phi_id, operand) in operands {
                        if let Some(phi) = translator.phis.get_mut(phi_id) {
                            phi.operands.push(operand.clone());
                        }
                    }
                } else {
                    // Fallback: if we couldn't find the trivial block's operand,
                    // try to find it now (though predecessors may have changed)
                    let preds = &translator.blocks[ultimate_target.0].predecessors;
                    if let Some(trivial_idx) = preds.iter().position(|&p| p == *trivial_block) {
                        for phi in translator.phis.values_mut() {
                            if phi.block_id == *ultimate_target {
                                if let Some(operand) = phi.operands.get(trivial_idx).cloned() {
                                    phi.operands.push(operand);
                                } else {
                                    // Last resort: use undefined if we can't find the operand
                                    phi.operands.push(V::undefined());
                                }
                            }
                        }
                    } else {
                        // Trivial block is not a predecessor - use undefined
                        for phi in translator.phis.values_mut() {
                            if phi.block_id == *ultimate_target {
                                phi.operands.push(V::undefined());
                            }
                        }
                    }
                }
                changed = true;
            }
        }

        // Note: CfgCleanup should still run after this to:
        // - Clear unreachable trivial blocks
        // - Remove any remaining stale edges

        if changed {
            PassResult::changed(stats)
        } else {
            PassResult::unchanged()
        }
    }

    fn invalidates(&self) -> Invalidations {
        // Modifies CFG structure
        Invalidations::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_threading_new() {
        let _pass = JumpThreading::new();
        // Basic instantiation test
    }
}
