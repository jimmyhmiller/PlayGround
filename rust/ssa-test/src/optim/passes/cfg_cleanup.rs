//! CFG Cleanup Pass.
//!
//! Cleans up the control-flow graph after other optimizations:
//! - Rebuilds predecessor lists from actual control flow (using jump_targets())
//! - Identifies and removes unreachable blocks
//! - Cleans up phi nodes that reference removed blocks

use std::collections::{HashMap, HashSet, VecDeque};

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::BlockId;

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction};

/// CFG Cleanup pass.
///
/// Cleans up the CFG after control-flow simplification and other passes:
/// 1. Rebuilds predecessor lists by scanning terminators (using jump_targets())
/// 2. Identifies unreachable blocks (not reachable from entry)
/// 3. Marks unreachable blocks as empty (actual removal would require renumbering)
/// 4. Removes phi operands for dead predecessor edges
///
/// # Note
/// This pass doesn't actually remove blocks from the blocks vector (which would
/// require renumbering all BlockIds). Instead, it clears unreachable blocks'
/// instructions and predecessors.
#[derive(Debug, Clone, Default)]
pub struct CfgCleanup;

impl CfgCleanup {
    pub fn new() -> Self {
        CfgCleanup
    }
}

impl<V, I, F> OptimizationPass<V, I, F> for CfgCleanup
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    fn name(&self) -> &'static str {
        "cfg-cleanup"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();
        let mut changed = false;

        // Step 1: Rebuild predecessors from actual control flow
        let predecessors_changed = rebuild_predecessors_from_terminators(translator);
        if predecessors_changed {
            changed = true;
        }

        // Step 2: Find all reachable blocks via BFS from entry
        let reachable = find_reachable_blocks(translator);

        // Step 3: Clear unreachable blocks
        for block in &mut translator.blocks {
            if !reachable.contains(&block.id) {
                if !block.instructions.is_empty() {
                    stats.instructions_removed += block.instructions.len();
                    block.instructions.clear();
                    block.predecessors.clear();
                    changed = true;
                }
            }
        }

        // Step 4: Remove phis in unreachable blocks
        let phi_ids_to_remove: Vec<_> = translator
            .phis
            .iter()
            .filter(|(_, phi)| !reachable.contains(&phi.block_id))
            .map(|(id, _)| *id)
            .collect();

        for phi_id in phi_ids_to_remove {
            translator.phis.remove(&phi_id);
            stats.phis_simplified += 1;
            changed = true;
        }

        // Step 5: Remove stale predecessor edges pointing to unreachable blocks
        // AND update phi operands accordingly

        // First pass: collect phi updates for unreachable predecessor removal
        let mut unreachable_pred_updates: Vec<(BlockId, Vec<BlockId>, Vec<usize>)> = Vec::new();

        for block in &translator.blocks {
            if !reachable.contains(&block.id) {
                continue; // Skip unreachable blocks entirely
            }

            // Find indices of unreachable predecessors
            let indices_to_remove: Vec<usize> = block.predecessors
                .iter()
                .enumerate()
                .filter(|(_, pred)| !reachable.contains(pred))
                .map(|(idx, _)| idx)
                .collect();

            if !indices_to_remove.is_empty() {
                unreachable_pred_updates.push((
                    block.id,
                    block.predecessors.clone(),
                    indices_to_remove,
                ));
            }
        }

        // Second pass: update phis for removed unreachable predecessors
        for (block_id, _old_preds, indices_to_remove) in &unreachable_pred_updates {
            // Remove in reverse order to preserve indices
            let mut sorted_indices = indices_to_remove.clone();
            sorted_indices.sort_by(|a, b| b.cmp(a));

            for phi in translator.phis.values_mut() {
                if phi.block_id == *block_id {
                    for &idx in &sorted_indices {
                        if idx < phi.operands.len() {
                            phi.operands.remove(idx);
                        }
                    }
                }
            }
            changed = true;
        }

        // Third pass: actually remove the predecessor edges
        for block in &mut translator.blocks {
            block.predecessors.retain(|pred| reachable.contains(pred));
        }

        // Step 6: Simplify single-operand phis
        // A phi with one operand is equivalent to a copy - we can replace the
        // phi assignment with a direct copy from the single operand
        let single_operand_phis: Vec<_> = translator.phis
            .iter()
            .filter(|(_, phi)| phi.operands.len() == 1)
            .map(|(id, phi)| (*id, phi.operands[0].clone()))
            .collect();

        for (phi_id, single_value) in single_operand_phis {
            // Replace all uses of Value::Phi(phi_id) with single_value
            for block in &mut translator.blocks {
                for instr in &mut block.instructions {
                    instr.visit_values_mut(|v| {
                        if v.as_phi() == Some(phi_id) {
                            *v = single_value.clone();
                        }
                    });
                }
            }

            // Also update other phis that might reference this phi
            for phi in translator.phis.values_mut() {
                for operand in &mut phi.operands {
                    if operand.as_phi() == Some(phi_id) {
                        *operand = single_value.clone();
                    }
                }
            }

            // Remove the simplified phi
            translator.phis.remove(&phi_id);
            stats.phis_simplified += 1;
            changed = true;
        }

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

/// Rebuild predecessor lists by scanning all terminators using jump_targets().
///
/// This is the authoritative way to determine control flow edges.
/// Returns true if any predecessors were changed.
fn rebuild_predecessors_from_terminators<V, I, F>(
    translator: &mut SSATranslator<V, I, F>,
) -> bool
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    // Build the new predecessor map from scratch
    let mut new_predecessors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();

    // Initialize all blocks with empty predecessor lists
    for block in &translator.blocks {
        new_predecessors.insert(block.id, Vec::new());
    }

    // Scan all terminators to find edges
    for block in &translator.blocks {
        let source_id = block.id;

        for instr in &block.instructions {
            // Get jump targets from this instruction
            let targets = instr.jump_targets();
            for target in targets {
                if let Some(preds) = new_predecessors.get_mut(&target) {
                    if !preds.contains(&source_id) {
                        preds.push(source_id);
                    }
                }
            }
        }
    }

    // First pass: collect all phi update information (immutable borrow)
    // We need: (block_id, old_predecessors, removed_predecessors)
    let mut phi_updates: Vec<(BlockId, Vec<BlockId>, Vec<BlockId>)> = Vec::new();
    let mut changed = false;

    for block in &translator.blocks {
        let new_preds = new_predecessors.get(&block.id).cloned().unwrap_or_default();
        let old_preds = &block.predecessors;

        if *old_preds != new_preds {
            // Predecessors changed - need to update phi operands
            let new_set: HashSet<_> = new_preds.iter().copied().collect();

            let removed: Vec<_> = old_preds.iter()
                .filter(|p| !new_set.contains(p))
                .copied()
                .collect();

            // Collect the update info for later
            if !removed.is_empty() {
                phi_updates.push((block.id, old_preds.clone(), removed));
            }

            // Note: We don't handle added predecessors here because that would require
            // knowing what value to use for the new phi operand. That's the responsibility
            // of the pass that added the edge.

            changed = true;
        }
    }

    // Second pass: apply phi updates (mutable borrow of phis only)
    for (block_id, old_preds, removed) in phi_updates {
        // Find indices of removed predecessors (in reverse order for safe removal)
        let mut indices_to_remove: Vec<usize> = removed
            .iter()
            .filter_map(|pred| old_preds.iter().position(|p| p == pred))
            .collect();
        indices_to_remove.sort_by(|a, b| b.cmp(a)); // Reverse order

        // Remove corresponding operands from all phis in this block
        for phi in translator.phis.values_mut() {
            if phi.block_id == block_id {
                for &idx in &indices_to_remove {
                    if idx < phi.operands.len() {
                        phi.operands.remove(idx);
                    }
                }
            }
        }
    }

    // Third pass: update the predecessor lists
    for block in &mut translator.blocks {
        let new_preds = new_predecessors.remove(&block.id).unwrap_or_default();
        block.predecessors = new_preds;
    }

    changed
}

/// Find all blocks reachable from the entry block via BFS.
/// Uses jump_targets() to find successors.
fn find_reachable_blocks<V, I, F>(translator: &SSATranslator<V, I, F>) -> HashSet<BlockId>
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut reachable = HashSet::new();
    let mut worklist = VecDeque::new();

    let entry = BlockId(0);
    worklist.push_back(entry);
    reachable.insert(entry);

    while let Some(block_id) = worklist.pop_front() {
        let block = &translator.blocks[block_id.0];

        // Find successors by examining all instructions' jump_targets
        for instr in &block.instructions {
            for target in instr.jump_targets() {
                if !reachable.contains(&target) {
                    reachable.insert(target);
                    worklist.push_back(target);
                }
            }
        }
    }

    reachable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_cleanup_new() {
        let _pass = CfgCleanup::new();
        // Basic instantiation test
    }
}
