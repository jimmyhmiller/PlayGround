//! Control Flow Simplification.
//!
//! Simplifies control-flow instructions when their operands are known constants.
//! For example, a conditional branch with a constant condition becomes an
//! unconditional jump, and guards that always pass can be removed.

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::{BlockId, SsaVariable};

use crate::optim::analysis::AnalysisCache;
use crate::optim::pass::{OptimizationPass, PassResult, PassStats, Invalidations};
use crate::optim::traits::{OptimizableValue, OptimizableInstruction, InstructionMutator, ControlFlowSimplification};

/// Control Flow Simplification pass.
///
/// For each instruction, calls `try_simplify_control_flow()` to determine
/// if the instruction can be simplified based on constant operands.
///
/// # Algorithm
/// 1. For each instruction:
/// 2. Call `try_simplify_control_flow()` (user-implemented)
/// 3. Handle the result:
///    - `NoChange`: skip
///    - `Remove`: delete the instruction
///    - `Jump { target, dead_targets }`: replace terminator with unconditional jump
///    - `PassThrough { dest, source, dead_target }`: replace guard with copy, remove dead edge
///    - `FailJump { target, fall_through_target }`: mid-block guard becomes jump, truncate rest
///
/// When modifying control flow, this pass also updates the CFG by removing
/// dead edges and cleaning up phi nodes in no-longer-targeted blocks.
#[derive(Debug, Clone, Default)]
pub struct ControlFlowSimplificationPass;

impl ControlFlowSimplificationPass {
    pub fn new() -> Self {
        ControlFlowSimplificationPass
    }
}

/// Represents a simplification action to be applied.
/// Generic over V to store the source value for PassThrough.
enum SimplificationAction<V> {
    Remove {
        block_id: BlockId,
        instr_idx: usize,
    },
    ReplaceWithJump {
        block_id: BlockId,
        instr_idx: usize,
        target: BlockId,
        dead_targets: Vec<BlockId>,
    },
    /// Guard passes - replace with copy instruction + jump, remove dead edge
    ReplaceWithCopy {
        block_id: BlockId,
        instr_idx: usize,
        dest: SsaVariable,
        source: V,
        dead_target: BlockId,
        fall_through_target: BlockId,
    },
    /// Guard fails - becomes jump, truncate block, update CFG
    FailToJump {
        block_id: BlockId,
        instr_idx: usize,
        target: BlockId,
        fall_through_target: BlockId,
    },
}

impl<V, I, F> OptimizationPass<V, I, F> for ControlFlowSimplificationPass
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I> + InstructionMutator,
{
    fn name(&self) -> &'static str {
        "control-flow-simplify"
    }

    fn run(
        &mut self,
        translator: &mut SSATranslator<V, I, F>,
        _cache: &mut AnalysisCache<V, I>,
    ) -> PassResult {
        let mut stats = PassStats::new();

        // Collect simplifications to apply
        // We need to collect first because we'll modify the CFG
        let mut actions: Vec<SimplificationAction<V>> = Vec::new();

        for block in &translator.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                match instr.try_simplify_control_flow() {
                    ControlFlowSimplification::NoChange => {}
                    ControlFlowSimplification::Remove => {
                        actions.push(SimplificationAction::Remove {
                            block_id: block.id,
                            instr_idx: idx,
                        });
                    }
                    ControlFlowSimplification::Jump { target, dead_targets } => {
                        actions.push(SimplificationAction::ReplaceWithJump {
                            block_id: block.id,
                            instr_idx: idx,
                            target,
                            dead_targets,
                        });
                    }
                    ControlFlowSimplification::PassThrough { dest, source, dead_target, fall_through_target } => {
                        actions.push(SimplificationAction::ReplaceWithCopy {
                            block_id: block.id,
                            instr_idx: idx,
                            dest,
                            source,
                            dead_target,
                            fall_through_target,
                        });
                    }
                    ControlFlowSimplification::FailJump { target, fall_through_target } => {
                        actions.push(SimplificationAction::FailToJump {
                            block_id: block.id,
                            instr_idx: idx,
                            target,
                            fall_through_target,
                        });
                    }
                }
            }
        }

        // Filter out redundant FailToJump actions in the same block.
        // If we have multiple FailToJump in the same block, only keep the one with
        // the lowest index - the others will be truncated anyway, and keeping them
        // would cause us to not properly remove their target edges.
        let mut seen_fail_blocks: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
        let actions: Vec<_> = actions
            .into_iter()
            .filter(|action| {
                if let SimplificationAction::FailToJump { block_id, .. } = action {
                    // Keep only the first (lowest index) FailToJump per block
                    seen_fail_blocks.insert(*block_id)
                } else {
                    true
                }
            })
            .collect();

        // Apply simplifications in reverse order to preserve indices
        for action in actions.into_iter().rev() {
            match action {
                SimplificationAction::Remove { block_id, instr_idx } => {
                    translator.blocks[block_id.0].instructions.remove(instr_idx);
                    stats.instructions_removed += 1;
                }

                SimplificationAction::ReplaceWithJump {
                    block_id,
                    instr_idx,
                    target,
                    dead_targets,
                } => {
                    // Replace with unconditional jump
                    translator.blocks[block_id.0].instructions[instr_idx] =
                        F::create_jump(target);

                    // Update CFG: remove dead edges and phi operands
                    for dead_target in dead_targets {
                        remove_predecessor_edge(translator, dead_target, block_id);
                    }

                    stats.instructions_removed += 1;
                }

                SimplificationAction::ReplaceWithCopy {
                    block_id,
                    instr_idx,
                    dest,
                    source,
                    dead_target,
                    fall_through_target,
                } => {
                    // Replace guard with copy: dest := source
                    translator.blocks[block_id.0].instructions[instr_idx] =
                        F::create_copy(dest, source);

                    // Add explicit jump to fall-through target after the copy.
                    // This is needed because the guard was a terminator with explicit targets,
                    // but the copy is not. Without the explicit jump, find_reachable_blocks
                    // won't find the fall-through successor.
                    // Insert right after the copy (at instr_idx + 1)
                    translator.blocks[block_id.0].instructions.insert(
                        instr_idx + 1,
                        F::create_jump(fall_through_target),
                    );

                    // Update CFG: remove dead edge to fail target
                    remove_predecessor_edge(translator, dead_target, block_id);

                    stats.instructions_removed += 1;
                }

                SimplificationAction::FailToJump {
                    block_id,
                    instr_idx,
                    target,
                    fall_through_target,
                } => {
                    // Collect all jump targets from instructions that will be truncated
                    // (including the current instruction's fall_through, and any subsequent
                    // instructions' targets). We need to remove these edges from the CFG.
                    let mut edges_to_remove: Vec<BlockId> = vec![fall_through_target];

                    // Check instructions after instr_idx for any jump targets
                    let block_instrs = &translator.blocks[block_id.0].instructions;
                    for i in (instr_idx + 1)..block_instrs.len() {
                        for jump_target in block_instrs[i].jump_targets() {
                            edges_to_remove.push(jump_target);
                        }
                    }

                    // Replace guard with jump
                    translator.blocks[block_id.0].instructions[instr_idx] =
                        F::create_jump(target);

                    // Truncate: remove all instructions after the jump
                    translator.blocks[block_id.0].instructions.truncate(instr_idx + 1);

                    // Update CFG: remove all dead edges
                    for dead_target in edges_to_remove {
                        // Don't remove edge to our new target
                        if dead_target != target {
                            remove_predecessor_edge(translator, dead_target, block_id);
                        }
                    }

                    stats.instructions_removed += 1;
                }
            }
        }

        if stats.instructions_removed > 0 {
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

/// Remove a predecessor edge from a block and update phi nodes.
///
/// When block `source` no longer jumps to block `target`, we need to:
/// 1. Remove `source` from `target`'s predecessor list
/// 2. Remove the corresponding operand from all phi nodes in `target`
fn remove_predecessor_edge<V, I, F>(
    translator: &mut SSATranslator<V, I, F>,
    target: BlockId,
    source: BlockId,
)
where
    V: OptimizableValue,
    I: OptimizableInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let block = &mut translator.blocks[target.0];

    // Find index of source in predecessors
    if let Some(idx) = block.predecessors.iter().position(|&p| p == source) {
        // Remove from predecessors
        block.predecessors.remove(idx);

        // Remove corresponding operand from all phis in this block
        for phi in translator.phis.values_mut() {
            if phi.block_id == target && idx < phi.operands.len() {
                phi.operands.remove(idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_flow_simplification_new() {
        let _pass = ControlFlowSimplificationPass::new();
        // Basic instantiation test - full tests in integration tests
    }
}
