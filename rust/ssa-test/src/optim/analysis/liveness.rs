//! Live variable analysis.
//!
//! Computes which variables are live at each program point.
//! A variable is live if its value may be used before being redefined.

use std::collections::{HashMap, HashSet};

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::{BlockId, SsaVariable};

use super::super::traits::{OptimizableValue, OptimizableInstruction};

/// Location within a block (instruction index)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramPoint {
    pub block: BlockId,
    pub index: usize,
}

/// Result of liveness analysis.
///
/// Provides live-in and live-out sets for each basic block.
#[derive(Debug, Clone)]
pub struct LivenessAnalysis {
    /// Variables live at the entry of each block
    pub live_in: HashMap<BlockId, HashSet<SsaVariable>>,
    /// Variables live at the exit of each block
    pub live_out: HashMap<BlockId, HashSet<SsaVariable>>,
    /// Variables defined in each block
    pub defs: HashMap<BlockId, HashSet<SsaVariable>>,
    /// Variables used in each block (before any local def)
    pub uses: HashMap<BlockId, HashSet<SsaVariable>>,
}

impl LivenessAnalysis {
    /// Compute liveness analysis for the given SSA program.
    ///
    /// Uses the standard dataflow equations:
    /// - USE[B] = variables used in B before any definition
    /// - DEF[B] = variables defined in B
    /// - IN[B] = USE[B] ∪ (OUT[B] - DEF[B])
    /// - OUT[B] = ∪ IN[S] for all successors S of B
    pub fn compute<V, I, F>(translator: &SSATranslator<V, I, F>) -> Self
    where
        V: OptimizableValue,
        I: OptimizableInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        let mut live_in: HashMap<BlockId, HashSet<SsaVariable>> = HashMap::new();
        let mut live_out: HashMap<BlockId, HashSet<SsaVariable>> = HashMap::new();
        let mut defs: HashMap<BlockId, HashSet<SsaVariable>> = HashMap::new();
        let mut uses: HashMap<BlockId, HashSet<SsaVariable>> = HashMap::new();

        // Initialize maps and compute local USE/DEF sets
        for block in &translator.blocks {
            let block_id = block.id;
            live_in.insert(block_id, HashSet::new());
            live_out.insert(block_id, HashSet::new());

            let mut block_defs = HashSet::new();
            let mut block_uses = HashSet::new();

            // Process instructions in order
            for instr in &block.instructions {
                // Uses: variables used that weren't defined locally yet
                instr.visit_values(|value| {
                    if let Some(var) = value.as_var() {
                        if !block_defs.contains(var) {
                            block_uses.insert(var.clone());
                        }
                    }
                });

                // Defs: variables defined by this instruction
                if let Some(dest) = instr.destination() {
                    block_defs.insert(dest.clone());
                }
            }

            // Also include phi node definitions and uses
            for phi in translator.phis.values() {
                if phi.block_id == block_id {
                    // Phi defines its destination variable
                    if let Some(dest) = &phi.dest {
                        block_defs.insert(dest.clone());
                    }
                    // Phi operands come from predecessors, not this block
                    // So they contribute to the predecessor's live-out, not this block's use
                }
            }

            defs.insert(block_id, block_defs);
            uses.insert(block_id, block_uses);
        }

        // Add phi operand uses to predecessor blocks
        for phi in translator.phis.values() {
            let phi_block = &translator.blocks[phi.block_id.0];
            for (i, pred_id) in phi_block.predecessors.iter().enumerate() {
                if i < phi.operands.len() {
                    let operand = &phi.operands[i];
                    if let Some(var) = operand.as_var() {
                        // This variable is used at the end of the predecessor
                        // It should be in live_out of pred
                        uses.entry(*pred_id).or_default().insert(var.clone());
                    }
                }
            }
        }

        // Build successor map for dataflow
        let mut successors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for block in &translator.blocks {
            successors.insert(block.id, Vec::new());
        }
        for block in &translator.blocks {
            for pred_id in &block.predecessors {
                successors.entry(*pred_id).or_default().push(block.id);
            }
        }

        // Iterate until fixed point
        let mut changed = true;
        while changed {
            changed = false;

            // Process blocks in reverse order (more efficient for backward analysis)
            for block in translator.blocks.iter().rev() {
                let block_id = block.id;

                // OUT[B] = ∪ IN[S] for all successors S
                let mut new_out = HashSet::new();
                if let Some(succs) = successors.get(&block_id) {
                    for succ_id in succs {
                        if let Some(succ_in) = live_in.get(succ_id) {
                            new_out.extend(succ_in.iter().cloned());
                        }
                    }
                }

                // IN[B] = USE[B] ∪ (OUT[B] - DEF[B])
                let block_defs = defs.get(&block_id).unwrap();
                let block_uses = uses.get(&block_id).unwrap();

                let mut new_in = block_uses.clone();
                for var in &new_out {
                    if !block_defs.contains(var) {
                        new_in.insert(var.clone());
                    }
                }

                // Check for changes
                let old_in = live_in.get(&block_id).unwrap();
                let old_out = live_out.get(&block_id).unwrap();

                if &new_in != old_in || &new_out != old_out {
                    changed = true;
                    live_in.insert(block_id, new_in);
                    live_out.insert(block_id, new_out);
                }
            }
        }

        LivenessAnalysis {
            live_in,
            live_out,
            defs,
            uses,
        }
    }

    /// Check if a variable is live at the exit of a block.
    pub fn is_live_out(&self, block_id: BlockId, var: &SsaVariable) -> bool {
        self.live_out
            .get(&block_id)
            .map(|set| set.contains(var))
            .unwrap_or(false)
    }

    /// Check if a variable is live at the entry of a block.
    pub fn is_live_in(&self, block_id: BlockId, var: &SsaVariable) -> bool {
        self.live_in
            .get(&block_id)
            .map(|set| set.contains(var))
            .unwrap_or(false)
    }

    /// Get all variables that are live at some point in the program.
    pub fn all_live_variables(&self) -> HashSet<SsaVariable> {
        let mut all = HashSet::new();
        for vars in self.live_in.values() {
            all.extend(vars.iter().cloned());
        }
        for vars in self.live_out.values() {
            all.extend(vars.iter().cloned());
        }
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests require concrete implementations - see integration tests
    // Basic API test
    #[test]
    fn test_liveness_api() {
        let analysis = LivenessAnalysis {
            live_in: HashMap::new(),
            live_out: HashMap::new(),
            defs: HashMap::new(),
            uses: HashMap::new(),
        };

        let var = SsaVariable::new("x");
        assert!(!analysis.is_live_in(BlockId(0), &var));
        assert!(!analysis.is_live_out(BlockId(0), &var));
        assert!(analysis.all_live_variables().is_empty());
    }
}
