//! Generic SSA translator implementing the Braun et al. algorithm.
//!
//! This module provides the core SSA construction algorithm, generic over
//! user-defined value and instruction types.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::traits::{InstructionFactory, SsaInstruction, SsaValue};
use crate::types::{Block, BlockId, Phi, PhiId, PhiReference, SsaVariable};

/// Generic SSA translator.
///
/// Implements the Braun et al. algorithm for SSA construction.
/// Generic over:
/// - `V`: The value type (must implement `SsaValue`)
/// - `I`: The instruction type (must implement `SsaInstruction<Value = V>`)
/// - `F`: The instruction factory (must implement `InstructionFactory<Instr = I>`)
pub struct SSATranslator<V, I, F>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    /// Variable name -> Block -> Value mapping (reaching definitions)
    pub definition: HashMap<String, HashMap<BlockId, V>>,
    /// Blocks that have been sealed (all predecessors known)
    pub sealed_blocks: HashSet<BlockId>,
    /// Incomplete phis waiting for block sealing
    pub incomplete_phis: HashMap<BlockId, HashMap<String, PhiId>>,
    /// All phi nodes
    pub phis: HashMap<PhiId, Phi<V>>,
    /// All basic blocks
    pub blocks: Vec<Block<I>>,
    /// Next SSA variable ID
    pub next_variable_id: usize,
    /// Next phi ID
    pub next_phi_id: usize,
    /// Current block being translated
    pub current_block: BlockId,
    /// Phantom data for the factory type
    _factory: PhantomData<F>,
}

impl<V, I, F> Default for SSATranslator<V, I, F>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V, I, F> SSATranslator<V, I, F>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    /// Create a new SSA translator with an initial block.
    pub fn new() -> Self {
        let mut translator = SSATranslator {
            definition: HashMap::new(),
            sealed_blocks: HashSet::new(),
            incomplete_phis: HashMap::new(),
            phis: HashMap::new(),
            blocks: Vec::new(),
            next_variable_id: 0,
            next_phi_id: 0,
            current_block: BlockId(0),
            _factory: PhantomData,
        };

        let initial_block = translator.create_block();
        translator.current_block = initial_block;
        translator
    }

    /// Record a variable definition in the given block.
    pub fn write_variable(&mut self, variable: String, block_id: BlockId, value: V) {
        self.definition
            .entry(variable)
            .or_default()
            .insert(block_id, value);
    }

    /// Read a variable, potentially creating phi nodes.
    pub fn read_variable(&mut self, variable: String, block_id: BlockId) -> V {
        if self.definition.contains_key(&variable) {
            if let Some(value) = self
                .definition
                .get(&variable)
                .and_then(|v| v.get(&block_id))
            {
                return value.clone();
            }
        }
        self.read_variable_recursively(variable, block_id)
    }

    fn read_variable_recursively(&mut self, variable: String, block_id: BlockId) -> V {
        let value: V;
        if !self.sealed_blocks.contains(&block_id) {
            // Block not sealed - create incomplete phi and materialize to variable
            let phi_id = self.create_phi(block_id);
            let temp_var = self.materialize_phi(phi_id, block_id);
            value = V::from_var(temp_var);
            self.incomplete_phis
                .entry(block_id)
                .or_default()
                .insert(variable.clone(), phi_id);
        } else {
            let block = self
                .blocks
                .get(block_id.0)
                .expect("Block not found");
            if block.predecessors.len() == 1 {
                value = self.read_variable(variable.clone(), block.predecessors[0]);
            } else {
                // Multiple predecessors - create phi and materialize to variable
                let phi_id = self.create_phi(block_id);
                let temp_var = self.materialize_phi(phi_id, block_id);
                let var_value = V::from_var(temp_var.clone());
                self.write_variable(variable.clone(), block_id, var_value.clone());
                let result = self.add_phi_operands(&variable, phi_id);

                // If the phi was removed as trivial, use the replacement value directly
                // and clean up the unnecessary copy instruction
                if !self.phis.contains_key(&phi_id) {
                    // Phi was removed - update definition and remove the copy instruction
                    self.write_variable(variable.clone(), block_id, result.clone());
                    // Remove the materialized instruction (vN := replacement)
                    self.remove_copy_instruction(block_id, &temp_var);
                    value = result;
                } else {
                    value = var_value;
                }
            }
        }
        self.write_variable(variable.clone(), block_id, value.clone());
        value
    }

    /// Materialize a phi to a variable by creating an Assign instruction at block start
    fn materialize_phi(&mut self, phi_id: PhiId, block_id: BlockId) -> SsaVariable {
        let temp_var = self.get_temp_variable("phi");
        // Insert phi assignment at the beginning of the block, after any existing phi assignments
        let insert_pos = self.find_phi_insert_position(block_id);
        let phi_assign = F::create_phi_assign(temp_var.clone(), phi_id);
        self.blocks[block_id.0].instructions.insert(insert_pos, phi_assign);
        temp_var
    }

    /// Find the position to insert a new phi assignment (after existing phi assigns)
    fn find_phi_insert_position(&self, block_id: BlockId) -> usize {
        let block = &self.blocks[block_id.0];
        for (i, instr) in block.instructions.iter().enumerate() {
            if !instr.is_phi_assignment() {
                return i;
            }
        }
        block.instructions.len()
    }

    /// Remove a copy instruction (dest := something) from a block
    fn remove_copy_instruction(&mut self, block_id: BlockId, dest: &SsaVariable) {
        let block = &mut self.blocks[block_id.0];
        block.instructions.retain(|instr| {
            instr.destination() != Some(dest)
        });
    }

    fn add_phi_operands(&mut self, variable: &String, phi_id: PhiId) -> V {
        // Check if phi still exists (might have been removed as trivial)
        let block_id = match self.phis.get(&phi_id) {
            Some(phi) => phi.block_id,
            None => {
                // Phi was already removed, return current definition
                if let Some(block_defs) = self.definition.get(variable) {
                    // Find a valid definition - this is a fallback
                    for (_, value) in block_defs {
                        return value.clone();
                    }
                }
                return V::undefined();
            }
        };

        // Check if this phi already has operands
        if let Some(phi) = self.phis.get(&phi_id) {
            if !phi.operands.is_empty() {
                return V::from_phi(phi_id);
            }
        }

        let block = self
            .blocks
            .get(block_id.0)
            .expect("Block not found");

        for predecessor in block.predecessors.clone() {
            let value = self.read_variable(variable.clone(), predecessor);
            if let Some(phi) = self.phis.get_mut(&phi_id) {
                phi.operands.push(value.clone());
            }
            if let Some(id) = value.as_phi() {
                self.add_phi_phi_use(phi_id, id);
            }
        }
        // Don't try to remove trivial phi immediately - wait until all phis are resolved
        self.try_remove_trivial_phi(phi_id)
    }

    fn try_remove_trivial_phi(&mut self, phi_id: PhiId) -> V {
        let phi = match self.phis.get(&phi_id) {
            Some(p) => p.clone(),
            None => return V::undefined(), // Phi was already removed
        };

        // Don't try to remove incomplete phis - they haven't had operands filled in yet
        let is_incomplete = self.incomplete_phis.values()
            .any(|block_phis| block_phis.values().any(|&id| id == phi_id));
        if is_incomplete {
            return V::from_phi(phi_id);
        }

        // Don't try to remove phis with no operands - they're not yet resolved
        if phi.operands.is_empty() {
            return V::from_phi(phi_id);
        }

        let mut same: Option<V> = None;

        for op in phi.operands.iter() {
            // Skip self-references and duplicates
            if Some(op) == same.as_ref() || op.is_same_phi(phi_id) {
                continue;
            }
            // If we already found one operand and this is different, keep the phi
            if same.is_some() {
                return V::from_phi(phi_id);
            }
            same = Some(op.clone());
        }

        // Determine replacement value
        let replacement = if same.is_none() {
            V::undefined()
        } else {
            same.unwrap()
        };

        // Remove the trivial phi from the map BEFORE replacing uses
        // This prevents infinite recursion
        self.phis.remove(&phi_id);

        // Replace all uses of this phi with the replacement
        // Use the cloned phi's uses since we already removed the phi from the map
        self.replace_phi_uses_with_list(&phi.uses, phi_id, replacement.clone());

        // Update the definition map - any variable defined as this phi should now use replacement
        let block_id = phi.block_id;
        for (_var_name, block_defs) in self.definition.iter_mut() {
            if let Some(value) = block_defs.get_mut(&block_id) {
                if value.is_same_phi(phi_id) {
                    *value = replacement.clone();
                }
            }
        }

        // Collect phi users to process (avoiding borrow issues)
        let phi_users: Vec<PhiId> = phi.uses.iter()
            .filter_map(|user| {
                if let PhiReference::Phi(user_phi_id) = user {
                    // Don't process self or already-removed phis
                    if *user_phi_id != phi_id && self.phis.contains_key(user_phi_id) {
                        return Some(*user_phi_id);
                    }
                }
                None
            })
            .collect();

        // Try to recursively remove all phi users, which might have become trivial
        for user_phi_id in phi_users {
            self.try_remove_trivial_phi(user_phi_id);
        }

        replacement
    }

    /// Seal a block, indicating all predecessors are known.
    pub fn seal_block(&mut self, block_id: BlockId) {
        self.sealed_blocks.insert(block_id);
        self.blocks[block_id.0].seal();

        let predecessor_count = self.blocks[block_id.0].predecessors.len();

        if let Some(phis) = self.incomplete_phis.remove(&block_id) {
            for (variable, phi_id) in phis {
                if predecessor_count == 1 {
                    // Single predecessor: no phi needed, just read from predecessor
                    let pred = self.blocks[block_id.0].predecessors[0];
                    let value = self.read_variable(variable.clone(), pred);

                    // Only update definition if there isn't already a local definition
                    // (the incomplete phi was for a READ, not a WRITE - don't overwrite writes)
                    let has_local_def = self.definition
                        .get(&variable)
                        .and_then(|m| m.get(&block_id))
                        .map(|v| !v.is_same_phi(phi_id))
                        .unwrap_or(false);

                    if !has_local_def {
                        self.write_variable(variable.clone(), block_id, value.clone());
                    }

                    // Replace uses of this phi with the value (in instructions)
                    if let Some(phi) = self.phis.get(&phi_id).cloned() {
                        self.replace_phi_uses_with_list(&phi.uses, phi_id, value.clone());
                    }

                    // Remove the unnecessary phi
                    self.phis.remove(&phi_id);
                } else {
                    // Multiple predecessors: need to add operands and potentially simplify
                    self.add_phi_operands(&variable, phi_id);
                }
            }
        }
    }

    /// Get a fresh temporary variable.
    pub fn get_temp_variable(&mut self, _prefix: &str) -> SsaVariable {
        let variable = SsaVariable(format!("v{}", self.next_variable_id));
        self.next_variable_id += 1;
        variable
    }

    /// Create a new phi node.
    pub fn create_phi(&mut self, block_id: BlockId) -> PhiId {
        let phi_id = PhiId(self.next_phi_id);
        self.next_phi_id += 1;

        let phi = Phi::new(phi_id, block_id);

        self.phis.insert(phi_id, phi);
        let instruction_offset = self.blocks[block_id.0].instructions.len();
        self.add_phi_use(phi_id, block_id, instruction_offset);
        phi_id
    }

    fn add_phi_use(&mut self, phi_id: PhiId, block_id: BlockId, instruction_offset: usize) {
        if let Some(phi) = self.phis.get_mut(&phi_id) {
            phi.uses.push(PhiReference::Instruction {
                block_id,
                instruction_offset,
            });
        }
    }

    fn add_phi_phi_use(&mut self, phi_id: PhiId, user_phi_id: PhiId) {
        if let Some(phi) = self.phis.get_mut(&phi_id) {
            phi.uses.push(PhiReference::Phi(user_phi_id));
        }
    }

    fn replace_phi_uses_with_list(&mut self, uses: &[PhiReference], phi_id: PhiId, replacement: V) {
        // Replace in tracked phi-to-phi uses
        for phi_ref in uses {
            match phi_ref {
                PhiReference::Instruction { block_id, instruction_offset } => {
                    self.replace_value_at_location(*block_id, *instruction_offset, phi_id, replacement.clone());
                }
                PhiReference::Phi(ph_id) => {
                    if let Some(p) = self.phis.get_mut(ph_id) {
                        for operand in &mut p.operands {
                            if operand.is_same_phi(phi_id) {
                                *operand = replacement.clone();
                            }
                        }
                    }
                }
            }
        }

        // Also scan ALL instructions for this phi reference
        // This is needed because use tracking for incomplete phis may be incorrect
        self.replace_phi_in_all_instructions(phi_id, replacement);
    }

    fn replace_phi_in_all_instructions(&mut self, phi_id: PhiId, replacement: V) {
        for block in &mut self.blocks {
            for instruction in &mut block.instructions {
                instruction.visit_values_mut(|value| {
                    if value.is_same_phi(phi_id) {
                        *value = replacement.clone();
                    }
                });
            }
        }
    }

    fn replace_value_at_location(&mut self, block_id: BlockId, instruction_offset: usize, old_phi_id: PhiId, new_value: V) {
        // First, replace in instructions
        if let Some(block) = self.blocks.get_mut(block_id.0) {
            if let Some(instruction) = block.instructions.get_mut(instruction_offset) {
                instruction.visit_values_mut(|value| {
                    if value.is_same_phi(old_phi_id) {
                        *value = new_value.clone();
                    }
                });
            }
        }

        // Then, replace in other phis' operands
        for other_phi in self.phis.values_mut() {
            for operand in &mut other_phi.operands {
                if operand.is_same_phi(old_phi_id) {
                    *operand = new_value.clone();
                }
            }
        }
    }

    /// Create a new basic block.
    pub fn create_block(&mut self) -> BlockId {
        let block_id = BlockId(self.blocks.len());
        self.blocks.push(Block::new(block_id));
        block_id
    }

    /// Add an instruction to the current block.
    pub fn emit(&mut self, instruction: I) {
        self.blocks[self.current_block.0].add_instruction(instruction);
    }

    /// Add a predecessor to a block.
    pub fn add_predecessor(&mut self, block_id: BlockId, predecessor: BlockId) {
        self.blocks[block_id.0].add_predecessor(predecessor);
    }
}
