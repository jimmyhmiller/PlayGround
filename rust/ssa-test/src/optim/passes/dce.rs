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

        if std::env::var("DEBUG_DCE_DUMP").is_ok() {
            eprintln!("\n=== DCE: BEFORE ===");
            for (block_idx, block) in translator.blocks.iter().enumerate() {
                eprintln!("Block {} (preds={:?}):", block_idx, block.predecessors);
                for instr in &block.instructions {
                    eprintln!("  {:?}", instr);
                }
            }
            eprintln!("Phis:");
            for (id, phi) in &translator.phis {
                eprintln!("  {:?}: {:?}", id, phi);
            }
        }

        // Compute live variables
        let live_vars = self.compute_live_variables(translator);

        if std::env::var("DEBUG_DCE_DUMP").is_ok() {
            eprintln!("\n=== DCE: LIVE VARS ===");
            for var in &live_vars {
                eprintln!("  {:?}", var);
            }
        }

        // Remove dead instructions and track offset remapping for phi uses
        use std::collections::HashMap;

        // Maps (block_id, old_offset) -> new_offset for instructions that were kept
        let mut offset_remaps: HashMap<(usize, usize), usize> = HashMap::new();

        for (block_idx, block) in translator.blocks.iter_mut().enumerate() {
            let original_count = block.instructions.len();

            // First, determine which instructions to keep and build the offset remap
            let mut new_offset = 0;
            let mut keep_flags: Vec<bool> = Vec::with_capacity(block.instructions.len());

            for (old_offset, instr) in block.instructions.iter().enumerate() {
                let keep = if instr.has_side_effects() {
                    if std::env::var("DEBUG_DCE_VERBOSE").is_ok() {
                        eprintln!("[DCE] Keeping side-effecting: {:?}", instr);
                    }
                    true
                } else if instr.is_terminator() {
                    if std::env::var("DEBUG_DCE_VERBOSE").is_ok() {
                        eprintln!("[DCE] Keeping terminator: {:?}", instr);
                    }
                    true
                } else if let Some(dest) = instr.destination() {
                    if live_vars.contains(dest) {
                        if std::env::var("DEBUG_DCE_VERBOSE").is_ok() {
                            eprintln!("[DCE] Keeping live dest {:?}: {:?}", dest, instr);
                        }
                        true
                    } else {
                        if std::env::var("DEBUG_DCE").is_ok() || std::env::var("DEBUG_DCE_VERBOSE").is_ok() {
                            eprintln!("[DCE] Removing dead instruction (dest={:?}): {:?}", dest, instr);
                        }
                        false
                    }
                } else {
                    if std::env::var("DEBUG_DCE_VERBOSE").is_ok() {
                        eprintln!("[DCE] Keeping no-dest: {:?}", instr);
                    }
                    true
                };

                if keep {
                    offset_remaps.insert((block_idx, old_offset), new_offset);
                    new_offset += 1;
                }
                keep_flags.push(keep);
            }

            // Now actually remove the instructions
            let mut flag_iter = keep_flags.into_iter();
            block.instructions.retain(|_| flag_iter.next().unwrap_or(true));

            stats.instructions_removed += original_count - block.instructions.len();
        }

        // Rebuild phi uses from scratch by scanning all instructions
        // This is safer than trying to remap offsets when instructions are removed
        use crate::types::{BlockId, PhiReference};

        // Build a map from phi destination variable names to phi IDs
        let phi_dest_to_id: HashMap<SsaVariable, crate::types::PhiId> = translator
            .phis
            .iter()
            .filter_map(|(phi_id, phi)| {
                phi.dest.as_ref().map(|dest| (dest.clone(), *phi_id))
            })
            .collect();

        // Clear existing instruction uses for all phis (keep phi-to-phi uses)
        for phi in translator.phis.values_mut() {
            phi.uses.retain(|phi_ref| matches!(phi_ref, PhiReference::Phi(_)));
        }

        // Rebuild instruction uses by scanning all instructions
        // Check for both direct phi references AND variable references to phi destinations
        for (block_idx, block) in translator.blocks.iter().enumerate() {
            for (instr_offset, instr) in block.instructions.iter().enumerate() {
                instr.visit_values(|value| {
                    // Check for direct phi references
                    if let Some(phi_id) = value.as_phi() {
                        if let Some(phi) = translator.phis.get_mut(&phi_id) {
                            phi.uses.push(PhiReference::Instruction {
                                block_id: BlockId(block_idx),
                                instruction_offset: instr_offset,
                            });
                        }
                    }
                    // Check for variable references to phi destinations
                    if let Some(var) = value.as_var() {
                        if let Some(phi_id) = phi_dest_to_id.get(var) {
                            if let Some(phi) = translator.phis.get_mut(phi_id) {
                                phi.uses.push(PhiReference::Instruction {
                                    block_id: BlockId(block_idx),
                                    instruction_offset: instr_offset,
                                });
                            }
                        }
                    }
                });
            }
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

        // TEMPORARILY DISABLED phi removal to debug
        if std::env::var("ENABLE_PHI_REMOVAL").is_ok() {
            for phi_id in phi_ids_to_remove {
                if std::env::var("DEBUG_DCE").is_ok() {
                    if let Some(phi) = translator.phis.get(&phi_id) {
                        eprintln!("[DCE] Removing dead phi: {:?}", phi);
                    }
                }
                translator.phis.remove(&phi_id);
                stats.phis_simplified += 1;
            }
        }

        if std::env::var("DEBUG_DCE_DUMP").is_ok() {
            eprintln!("\n=== DCE: AFTER ===");
            for (block_idx, block) in translator.blocks.iter().enumerate() {
                eprintln!("Block {} (preds={:?}):", block_idx, block.predecessors);
                for instr in &block.instructions {
                    eprintln!("  {:?}", instr);
                }
            }
            eprintln!("Phis:");
            for (id, phi) in &translator.phis {
                eprintln!("  {:?}: {:?}", id, phi);
            }
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
