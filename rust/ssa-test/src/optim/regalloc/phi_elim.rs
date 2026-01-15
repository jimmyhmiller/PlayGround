//! Phi elimination: convert SSA phi nodes to parallel copies.
//!
//! Before register allocation, phi nodes must be eliminated because
//! they don't correspond to real machine instructions. Instead, we
//! insert copy instructions at the end of predecessor blocks.
//!
//! For example:
//! ```text
//! B1:                           B2:
//!   v1 = ...                      v2 = ...
//!   jmp B3                        jmp B3
//!
//! B3:
//!   v3 = phi(v1, v2)
//!   use v3
//! ```
//!
//! Becomes:
//! ```text
//! B1:                           B2:
//!   v1 = ...                      v2 = ...
//!   v3 = v1  <-- copy added       v3 = v2  <-- copy added
//!   jmp B3                        jmp B3
//!
//! B3:
//!   use v3
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::optim::traits::{OptimizableInstruction, OptimizableValue};
use crate::traits::{InstructionFactory, SsaInstruction, SsaValue};
use crate::translator::SSATranslator;
use crate::types::BlockId;

/// Violations found during phi elimination validation.
#[derive(Debug, Clone, PartialEq)]
pub enum PhiEliminationViolation {
    /// Phi nodes remain after elimination (should be empty)
    PhisRemainAfterElimination { count: usize },
    /// Block has no instructions (empty block)
    EmptyBlock { block_id: BlockId },
    /// Block is unreachable (no predecessors and not entry)
    UnreachableBlock { block_id: BlockId },
}

impl fmt::Display for PhiEliminationViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhiEliminationViolation::PhisRemainAfterElimination { count } => {
                write!(f, "{} phi nodes remain after elimination (should be 0)", count)
            }
            PhiEliminationViolation::EmptyBlock { block_id } => {
                write!(f, "Block {:?} has no instructions after phi elimination", block_id)
            }
            PhiEliminationViolation::UnreachableBlock { block_id } => {
                write!(f, "Block {:?} is unreachable (no predecessors)", block_id)
            }
        }
    }
}

/// Phi elimination pass.
///
/// Converts phi nodes to copy instructions at the end of predecessor blocks.
pub struct PhiElimination;

impl PhiElimination {
    /// Eliminate all phi nodes in the translator.
    ///
    /// This inserts copy instructions at the end of predecessor blocks
    /// and removes the phi nodes from the translator's phi map.
    ///
    /// # Critical Edge Splitting
    ///
    /// If there are critical edges (edge from a block with multiple successors
    /// to a block with multiple predecessors), they must be split first to
    /// avoid incorrect copy placement. Call `split_critical_edges` first if
    /// your CFG may have critical edges.
    pub fn eliminate<V, I, F>(translator: &mut SSATranslator<V, I, F>)
    where
        V: SsaValue + OptimizableValue,
        I: SsaInstruction<Value = V> + OptimizableInstruction,
        F: InstructionFactory<Instr = I>,
    {
        // Collect all phi information first to avoid borrow issues
        // IMPORTANT: Sort by phi ID to ensure deterministic ordering.
        // HashMap iteration order is non-deterministic, which can cause
        // different copy insertion orders and incorrect results.
        let mut phi_ids: Vec<_> = translator.phis.keys().cloned().collect();
        phi_ids.sort_by_key(|id| id.0);

        let phi_info: Vec<_> = phi_ids.iter()
            .filter_map(|phi_id| {
                let phi = translator.phis.get(phi_id)?;
                let dest = phi.dest.clone()?;
                let block_id = phi.block_id;
                let operands = phi.operands.clone();
                Some((block_id, dest, operands))
            })
            .collect();

        // For each phi, insert copies in predecessor blocks
        for (block_id, dest, operands) in phi_info {
            let block = &translator.blocks[block_id.0];
            let predecessors = block.predecessors.clone();

            // Each operand corresponds to a predecessor in order
            for (pred_idx, pred_id) in predecessors.iter().enumerate() {
                if pred_idx >= operands.len() {
                    continue;
                }

                let operand = &operands[pred_idx];

                // Create copy: dest := operand
                let copy = F::create_copy(dest.clone(), operand.clone());

                // Insert before terminator in predecessor
                Self::insert_before_terminator(translator, *pred_id, copy);
            }
        }

        // Clear all phis
        translator.phis.clear();
    }

    /// Insert an instruction before the terminator in a block.
    ///
    /// If the block has no terminator, the instruction is appended.
    fn insert_before_terminator<V, I, F>(
        translator: &mut SSATranslator<V, I, F>,
        block_id: BlockId,
        instruction: I,
    )
    where
        V: SsaValue + OptimizableValue,
        I: SsaInstruction<Value = V> + OptimizableInstruction,
        F: InstructionFactory<Instr = I>,
    {
        let block = &mut translator.blocks[block_id.0];
        let len = block.instructions.len();

        if len == 0 {
            block.instructions.push(instruction);
        } else {
            // Check if the last instruction is actually a terminator
            let last_is_terminator = block.instructions.last()
                .map(|i| i.is_terminator())
                .unwrap_or(false);

            if last_is_terminator {
                // Insert before the terminator
                block.instructions.insert(len - 1, instruction);
            } else {
                // No terminator - append to the end
                block.instructions.push(instruction);
            }
        }
    }

    /// Split critical edges in the CFG.
    ///
    /// A critical edge is an edge from a block with multiple successors
    /// to a block with multiple predecessors. These must be split before
    /// phi elimination to ensure copies are placed correctly.
    ///
    /// Returns the number of edges split.
    pub fn split_critical_edges<V, I, F>(translator: &mut SSATranslator<V, I, F>) -> usize
    where
        V: SsaValue,
        I: SsaInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        // Build successor map
        let mut successors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for block in &translator.blocks {
            for pred_id in &block.predecessors {
                successors.entry(*pred_id).or_default().push(block.id);
            }
        }

        // Find critical edges: (from_block, to_block)
        let mut critical_edges: Vec<(BlockId, BlockId)> = Vec::new();
        for (from_id, succs) in &successors {
            if succs.len() <= 1 {
                continue;  // Not a multi-successor block
            }
            for to_id in succs {
                let to_block = &translator.blocks[to_id.0];
                if to_block.predecessors.len() > 1 {
                    // This is a critical edge
                    critical_edges.push((*from_id, *to_id));
                }
            }
        }

        let count = critical_edges.len();

        // Split each critical edge by inserting a new block
        for (from_id, to_id) in critical_edges {
            Self::split_edge(translator, from_id, to_id);
        }

        count
    }

    /// Split a single edge by inserting a new block.
    fn split_edge<V, I, F>(
        translator: &mut SSATranslator<V, I, F>,
        from_id: BlockId,
        to_id: BlockId,
    )
    where
        V: SsaValue,
        I: SsaInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        // Create a new block for the edge
        let new_block_id = translator.create_block();

        // Update predecessor list of target block
        let to_block = &mut translator.blocks[to_id.0];
        let pred_idx = to_block.predecessors.iter()
            .position(|&p| p == from_id);
        if let Some(idx) = pred_idx {
            to_block.predecessors[idx] = new_block_id;
        }

        // Set predecessor of new block
        translator.blocks[new_block_id.0].predecessors.push(from_id);

        // Update phi operands: operands from `from_id` now come from `new_block_id`
        // This is handled implicitly because phi operands correspond to predecessors
        // in order, and we've updated the predecessor list.

        // Note: We don't update the terminator of from_block here because
        // we don't know the instruction format. The caller needs to update
        // jump targets if necessary.
    }

    /// Compute which variables need parallel copies to avoid lost copies.
    ///
    /// Returns a set of variable pairs (src, dst) that conflict and need
    /// special handling.
    pub fn find_copy_conflicts<V, I, F>(
        translator: &SSATranslator<V, I, F>,
    ) -> HashSet<(String, String)>
    where
        V: SsaValue,
        I: SsaInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        let mut conflicts = HashSet::new();

        // For each phi, check if its destination is used by another phi
        // in the same block as a source
        for phi in translator.phis.values() {
            let dest = match &phi.dest {
                Some(d) => d.0.clone(),
                None => continue,
            };

            for other_phi in translator.phis.values() {
                if phi.id == other_phi.id || phi.block_id != other_phi.block_id {
                    continue;
                }

                for operand in &other_phi.operands {
                    if let Some(var) = operand.as_var() {
                        if var.0 == dest {
                            // other_phi uses the variable that phi defines
                            if let Some(other_dest) = &other_phi.dest {
                                conflicts.insert((dest.clone(), other_dest.0.clone()));
                            }
                        }
                    }
                }
            }
        }

        conflicts
    }

    /// Validate the IR state after phi elimination.
    ///
    /// Returns a list of violations found. An empty list indicates valid state.
    pub fn validate<V, I, F>(
        translator: &SSATranslator<V, I, F>,
    ) -> Vec<PhiEliminationViolation>
    where
        V: SsaValue,
        I: SsaInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        let mut violations = Vec::new();

        // Check that all phis have been eliminated
        if !translator.phis.is_empty() {
            violations.push(PhiEliminationViolation::PhisRemainAfterElimination {
                count: translator.phis.len(),
            });
        }

        // Check for empty blocks
        let entry_block = BlockId(0);
        for block in &translator.blocks {
            if block.instructions.is_empty() {
                violations.push(PhiEliminationViolation::EmptyBlock {
                    block_id: block.id,
                });
            }

            // Check for unreachable blocks (except entry)
            if block.id != entry_block && block.predecessors.is_empty() {
                violations.push(PhiEliminationViolation::UnreachableBlock {
                    block_id: block.id,
                });
            }
        }

        violations
    }

    /// Eliminate phi nodes and validate the result.
    ///
    /// This is equivalent to calling `eliminate` followed by `validate`,
    /// but panics if validation fails.
    pub fn eliminate_and_verify<V, I, F>(translator: &mut SSATranslator<V, I, F>)
    where
        V: SsaValue + OptimizableValue,
        I: SsaInstruction<Value = V> + OptimizableInstruction,
        F: InstructionFactory<Instr = I>,
    {
        Self::eliminate(translator);

        let violations = Self::validate(translator);
        if !violations.is_empty() {
            let mut msg = String::from("Phi elimination validation failed:\n");
            for v in &violations {
                msg.push_str(&format!("  - {}\n", v));
            }
            panic!("{}", msg);
        }
    }

    /// Eliminate phi nodes and clean up any resulting trampolines.
    ///
    /// This is the recommended method for phi elimination as it:
    /// 1. Converts phi nodes to copy instructions
    /// 2. Removes any trampoline blocks created by critical edge splitting
    ///
    /// Returns the number of trampolines eliminated.
    ///
    /// # Example
    /// ```ignore
    /// // Before register allocation:
    /// PhiElimination::eliminate_with_cleanup(&mut translator);
    /// // Now the IR has no phis and no trampolines
    /// ```
    pub fn eliminate_with_cleanup<V, I, F>(translator: &mut SSATranslator<V, I, F>) -> usize
    where
        V: SsaValue + OptimizableValue,
        I: SsaInstruction<Value = V> + OptimizableInstruction,
        F: InstructionFactory<Instr = I>,
    {
        Self::eliminate(translator);
        eliminate_trampolines(translator)
    }
}

/// Assert phi elimination validation passes.
pub fn assert_valid_after_phi_elimination<V, I, F>(translator: &SSATranslator<V, I, F>)
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let violations = PhiElimination::validate(translator);
    if !violations.is_empty() {
        let mut msg = String::from("Post-phi-elimination validation failed:\n");
        for v in &violations {
            msg.push_str(&format!("  - {}\n", v));
        }
        panic!("{}", msg);
    }
}

/// Trampoline elimination: remove blocks that only contain an unconditional jump.
///
/// After phi elimination and critical edge splitting, some blocks may end up
/// containing only a jump instruction (trampolines). These waste code space
/// and cause unnecessary branches.
///
/// This function:
/// 1. Identifies trampoline blocks (single unconditional jump, no phis)
/// 2. Rewrites all predecessors to jump directly to the ultimate target
/// 3. Clears the trampoline blocks
///
/// # Returns
/// The number of trampolines eliminated.
///
/// # Example
/// Before:
/// ```text
/// B0: if cond goto B1 else goto B2
/// B1: ...code... goto B3
/// B2: goto B3      <- trampoline (created by critical edge split)
/// B3: ...
/// ```
///
/// After:
/// ```text
/// B0: if cond goto B1 else goto B3   <- B2 bypassed
/// B1: ...code... goto B3
/// B2: (empty)                         <- cleared
/// B3: ...
/// ```
pub fn eliminate_trampolines<V, I, F>(translator: &mut SSATranslator<V, I, F>) -> usize
where
    V: SsaValue + OptimizableValue,
    I: SsaInstruction<Value = V> + OptimizableInstruction,
    F: InstructionFactory<Instr = I>,
{
    // Step 1: Identify trampoline blocks
    // A trampoline block:
    // - Is not the entry block (BlockId(0))
    // - Has exactly one instruction
    // - That instruction is an unconditional jump (exactly one target)
    // - Has no phi nodes (should be empty after phi elimination anyway)
    let trampolines: HashMap<BlockId, BlockId> = translator.blocks
        .iter()
        .filter(|block| {
            block.id.0 != 0
                && block.instructions.len() == 1
                && block.instructions[0].jump_targets().len() == 1
                && !translator.phis.values().any(|phi| phi.block_id == block.id)
        })
        .map(|block| {
            let target = block.instructions[0].jump_targets()[0];
            (block.id, target)
        })
        .collect();

    if trampolines.is_empty() {
        return 0;
    }

    // Step 2: Build ultimate target map (follow chains of trampolines)
    // A -> B -> C becomes A -> C
    let ultimate_targets: HashMap<BlockId, BlockId> = trampolines
        .keys()
        .map(|&block_id| {
            let mut current = block_id;
            let mut visited = HashSet::new();
            visited.insert(current);

            // Follow the chain until we hit a non-trampoline or a cycle
            while let Some(&next) = trampolines.get(&current) {
                if visited.contains(&next) {
                    // Cycle detected, stop here
                    break;
                }
                visited.insert(next);
                current = next;
            }

            // current is either in trampolines (chain end) or not (final target)
            let ultimate = trampolines.get(&current).copied().unwrap_or(current);
            (block_id, ultimate)
        })
        .collect();

    // Step 3: Rewrite jump targets in all blocks
    for block in &mut translator.blocks {
        for instr in &mut block.instructions {
            for (&trampoline, &ultimate) in &ultimate_targets {
                instr.rewrite_jump_target(trampoline, ultimate);
            }
        }
    }

    // Step 4: Collect trampoline predecessor information
    // We need to know who jumps to each trampoline so we can update successor predecessors
    let trampoline_preds: HashMap<BlockId, Vec<BlockId>> = trampolines
        .keys()
        .map(|&t| (t, translator.blocks[t.0].predecessors.clone()))
        .collect();

    // Step 5: Update predecessor lists
    // For each non-trampoline block, update predecessors that were trampolines
    for block in &mut translator.blocks {
        if trampolines.contains_key(&block.id) {
            continue; // Skip trampolines, we'll clear them
        }

        let mut new_preds = Vec::new();
        for &pred in &block.predecessors {
            if ultimate_targets.contains_key(&pred) {
                // This predecessor was a trampoline - find who jumped to it
                // and add them as the new predecessors
                if let Some(tpreds) = trampoline_preds.get(&pred) {
                    for &tp in tpreds {
                        if !new_preds.contains(&tp) {
                            new_preds.push(tp);
                        }
                    }
                }
            } else {
                if !new_preds.contains(&pred) {
                    new_preds.push(pred);
                }
            }
        }
        block.predecessors = new_preds;
    }

    // Step 5: Clear trampoline blocks (make them empty/unreachable)
    let count = trampolines.len();
    for &trampoline_id in trampolines.keys() {
        let block = &mut translator.blocks[trampoline_id.0];
        block.instructions.clear();
        block.predecessors.clear();
    }

    count
}

#[cfg(test)]
mod tests {
    // Tests require concrete implementations - see integration tests
}
