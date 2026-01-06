//! Spill code insertion for register allocation.
//!
//! When a variable is spilled to the stack, we need to insert:
//! - Store instructions after each definition
//! - Load instructions before each use
//!
//! This module provides traits and utilities for inserting spill code.

use std::collections::HashMap;

use crate::traits::{InstructionFactory, SsaInstruction, SsaValue};
use crate::translator::SSATranslator;
use crate::types::{BlockId, SsaVariable};

use super::interval::Location;
use super::linear_scan::AllocationResult;

/// Trait for instruction factories that can create spill/reload instructions.
///
/// Implement this trait on your instruction factory to enable spill code
/// insertion after register allocation.
///
/// # Example
/// ```ignore
/// impl SpillCodeFactory for MyInstructionFactory {
///     fn create_spill(src: SsaVariable, slot: usize) -> MyInstruction {
///         MyInstruction::Spill { src, slot }
///     }
///
///     fn create_reload(dest: SsaVariable, slot: usize) -> MyInstruction {
///         MyInstruction::Reload { dest, slot }
///     }
///
///     fn create_reg_move(dest: SsaVariable, src: SsaVariable) -> MyInstruction {
///         MyInstruction::Move { dest, src }
///     }
/// }
/// ```
pub trait SpillCodeFactory: InstructionFactory {
    /// Create a spill instruction (store variable to stack slot).
    ///
    /// The generated instruction should store the value of `src` to
    /// the stack slot at offset `slot * stack_slot_size`.
    fn create_spill(src: SsaVariable, slot: usize) -> Self::Instr;

    /// Create a reload instruction (load from stack slot to variable).
    ///
    /// The generated instruction should load from the stack slot at
    /// offset `slot * stack_slot_size` into `dest`.
    fn create_reload(dest: SsaVariable, slot: usize) -> Self::Instr;

    /// Create a register-to-register move instruction.
    ///
    /// Used when coalescing fails and an explicit copy is needed.
    fn create_reg_move(dest: SsaVariable, src: SsaVariable) -> Self::Instr;
}

/// Statistics from spill code insertion.
#[derive(Debug, Clone, Default)]
pub struct SpillStats {
    /// Number of spill (store) instructions inserted.
    pub spills_inserted: usize,

    /// Number of reload (load) instructions inserted.
    pub reloads_inserted: usize,

    /// Number of register moves inserted.
    pub moves_inserted: usize,
}

/// Insert spill and reload code based on allocation results.
///
/// This function modifies the translator in place, inserting:
/// - Store instructions after each definition of a spilled variable
/// - Load instructions before each use of a spilled variable
///
/// Returns statistics about the inserted code.
pub fn insert_spill_code<V, I, F>(
    translator: &mut SSATranslator<V, I, F>,
    allocation: &AllocationResult,
) -> SpillStats
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: SpillCodeFactory<Instr = I>,
{
    let mut stats = SpillStats::default();

    // Find all spilled variables and their slots
    let spilled: HashMap<SsaVariable, usize> = allocation.assignments
        .iter()
        .filter_map(|(var, loc)| {
            if let Location::StackSlot(slot) = loc {
                Some((var.clone(), *slot))
            } else {
                None
            }
        })
        .collect();

    if spilled.is_empty() {
        return stats;
    }

    // Process each block
    for block_idx in 0..translator.blocks.len() {
        let _block_id = BlockId(block_idx);

        // We need to insert instructions, which changes indices,
        // so we work in reverse order within each block
        let mut insertions: Vec<(usize, I, InsertPosition)> = Vec::new();

        let block = &translator.blocks[block_idx];

        for (instr_idx, instr) in block.instructions.iter().enumerate() {
            // Check for definitions of spilled variables
            if let Some(dest) = instr.destination() {
                if let Some(&slot) = spilled.get(dest) {
                    // Insert spill after this instruction
                    let spill = F::create_spill(dest.clone(), slot);
                    insertions.push((instr_idx, spill, InsertPosition::After));
                    stats.spills_inserted += 1;
                }
            }

            // Check for uses of spilled variables
            let mut used_spilled: Vec<(SsaVariable, usize)> = Vec::new();
            instr.visit_values(|value| {
                if let Some(var) = value.as_var() {
                    if let Some(&slot) = spilled.get(var) {
                        used_spilled.push((var.clone(), slot));
                    }
                }
            });

            // Insert reloads before this instruction
            for (var, slot) in used_spilled {
                // Create a temporary variable for the reload
                // Note: In a real implementation, we'd want to reuse the same
                // reload for multiple uses in the same instruction
                let reload = F::create_reload(var, slot);
                insertions.push((instr_idx, reload, InsertPosition::Before));
                stats.reloads_inserted += 1;
            }
        }

        // Apply insertions in reverse order to maintain correct indices
        apply_insertions(&mut translator.blocks[block_idx].instructions, insertions);
    }

    stats
}

/// Position for inserting an instruction relative to an index.
#[derive(Debug, Clone, Copy)]
enum InsertPosition {
    Before,
    After,
}

/// Apply insertions to an instruction list.
fn apply_insertions<I>(instructions: &mut Vec<I>, mut insertions: Vec<(usize, I, InsertPosition)>) {
    // Sort by index (descending) so we can insert without invalidating indices
    insertions.sort_by(|a, b| b.0.cmp(&a.0));

    for (idx, instr, position) in insertions {
        let insert_idx = match position {
            InsertPosition::Before => idx,
            InsertPosition::After => idx + 1,
        };
        instructions.insert(insert_idx, instr);
    }
}

/// Information about spill slots for a function.
#[derive(Debug, Clone)]
pub struct SpillSlotInfo {
    /// Number of spill slots used.
    pub num_slots: usize,

    /// Size of each slot in bytes.
    pub slot_size: usize,

    /// Map from variable to its spill slot.
    pub variable_slots: HashMap<SsaVariable, usize>,
}

impl SpillSlotInfo {
    /// Create spill slot info from allocation result.
    pub fn from_allocation(allocation: &AllocationResult, slot_size: usize) -> Self {
        let variable_slots: HashMap<SsaVariable, usize> = allocation.assignments
            .iter()
            .filter_map(|(var, loc)| {
                if let Location::StackSlot(slot) = loc {
                    Some((var.clone(), *slot))
                } else {
                    None
                }
            })
            .collect();

        SpillSlotInfo {
            num_slots: allocation.stack_slots_used,
            slot_size,
            variable_slots,
        }
    }

    /// Calculate the total stack space needed for spills.
    pub fn total_spill_size(&self) -> usize {
        self.num_slots * self.slot_size
    }

    /// Get the offset for a spill slot.
    pub fn slot_offset(&self, slot: usize) -> usize {
        slot * self.slot_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spill_slot_info() {
        let mut assignments = HashMap::new();
        assignments.insert(SsaVariable::new("a"), Location::Register(0));
        assignments.insert(SsaVariable::new("b"), Location::StackSlot(0));
        assignments.insert(SsaVariable::new("c"), Location::StackSlot(1));

        let allocation = AllocationResult {
            assignments,
            stack_slots_used: 2,
            stats: Default::default(),
        };

        let info = SpillSlotInfo::from_allocation(&allocation, 8);

        assert_eq!(info.num_slots, 2);
        assert_eq!(info.slot_size, 8);
        assert_eq!(info.total_spill_size(), 16);
        assert_eq!(info.slot_offset(0), 0);
        assert_eq!(info.slot_offset(1), 8);
        assert_eq!(info.variable_slots.len(), 2);
        assert!(info.variable_slots.contains_key(&SsaVariable::new("b")));
    }
}
