//! SSA Property Validation
//!
//! Generic validation for SSA properties, based on "Simple and Efficient
//! Construction of Static Single Assignment Form" by Braun et al. (CC 2013)

use std::collections::{HashMap, HashSet};

use crate::traits::{InstructionFactory, SsaInstruction, SsaValue};
use crate::translator::SSATranslator;
use crate::types::{BlockId, PhiId, SsaVariable};

/// SSA validation violation types
#[derive(Debug, Clone, PartialEq)]
pub enum SSAViolation<V> {
    /// Phi has all same operands (should have been removed as trivial)
    TrivialPhi {
        phi_id: PhiId,
        block_id: BlockId,
        unique_operand: V,
    },
    /// Phi exists in entry block (entry has no predecessors, so phi is invalid)
    PhiInEntryBlock {
        phi_id: PhiId,
    },
    /// Phi exists in unreachable block
    PhiInUnreachableBlock {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Block has no predecessors (and is not the entry block) - unreachable code
    UnreachableBlock {
        block_id: BlockId,
    },
    /// Phi operand index doesn't align with predecessor
    PhiOperandPredecessorMismatch {
        phi_id: PhiId,
        operand_idx: usize,
        operand_from_block: BlockId,
        expected_predecessor: BlockId,
    },
    /// Phi operand count doesn't match block's predecessor count
    OperandCountMismatch {
        phi_id: PhiId,
        block_id: BlockId,
        operand_count: usize,
        predecessor_count: usize,
    },
    /// Phi references another phi that doesn't exist
    DanglingPhiReference {
        phi_id: PhiId,
        referenced_phi: PhiId,
    },
    /// Phi exists in a block with 0 or 1 predecessors (not a join point)
    PhiInNonJoinBlock {
        phi_id: PhiId,
        block_id: BlockId,
        predecessor_count: usize,
    },
    /// Block is not sealed after construction
    UnsealedBlock {
        block_id: BlockId,
    },
    /// Phi has no operands (incomplete)
    EmptyPhi {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Self-referential phi with no other operands
    OnlySelfReferencePhi {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Instruction references a phi that doesn't exist
    DanglingPhiInInstruction {
        block_id: BlockId,
        instruction_index: usize,
        referenced_phi: PhiId,
    },
    /// Phi used directly as operand (should be assigned to variable first)
    PhiUsedDirectlyAsOperand {
        block_id: BlockId,
        instruction_index: usize,
        phi_id: PhiId,
    },
    /// Phi is missing its destination variable (not materialized)
    PhiMissingDestination {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Variable is defined more than once
    MultipleDefinitions {
        variable: SsaVariable,
        definition_sites: Vec<(BlockId, usize)>,
    },
    /// Use of undefined variable
    UndefinedVariableUse {
        block_id: BlockId,
        instruction_index: usize,
        variable: SsaVariable,
    },
    /// Dead phi - phi has no non-phi users
    DeadPhi {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Dominance violation
    DominanceViolation {
        block_id: BlockId,
        instruction_index: usize,
        variable: SsaVariable,
        def_block: BlockId,
    },
    /// Phi operand dominance violation - operand doesn't dominate predecessor
    PhiOperandDominanceViolation {
        phi_id: PhiId,
        block_id: BlockId,
        operand_index: usize,
        predecessor: BlockId,
        variable: SsaVariable,
        def_block: BlockId,
    },
    /// Phi referenced as value in instruction (should have been replaced with variable)
    PhiNotInlined {
        phi_id: PhiId,
        block_id: BlockId,
        instruction_index: usize,
    },
}

impl<V: std::fmt::Debug> std::fmt::Display for SSAViolation<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SSAViolation::TrivialPhi { phi_id, block_id, unique_operand } => {
                write!(f, "Trivial phi {:?} in block {:?}: all operands are {:?}",
                    phi_id, block_id, unique_operand)
            }
            SSAViolation::PhiInEntryBlock { phi_id } => {
                write!(f, "Phi {:?} exists in entry block (entry has no predecessors)", phi_id)
            }
            SSAViolation::PhiInUnreachableBlock { phi_id, block_id } => {
                write!(f, "Phi {:?} exists in unreachable block {:?}", phi_id, block_id)
            }
            SSAViolation::UnreachableBlock { block_id } => {
                write!(f, "Block {:?} has no predecessors (unreachable code)", block_id)
            }
            SSAViolation::PhiOperandPredecessorMismatch { phi_id, operand_idx, operand_from_block, expected_predecessor } => {
                write!(f, "Phi {:?} operand {} is from {:?} but expected from predecessor {:?}",
                    phi_id, operand_idx, operand_from_block, expected_predecessor)
            }
            SSAViolation::OperandCountMismatch { phi_id, block_id, operand_count, predecessor_count } => {
                write!(f, "Phi {:?} in block {:?} has {} operands but block has {} predecessors",
                    phi_id, block_id, operand_count, predecessor_count)
            }
            SSAViolation::DanglingPhiReference { phi_id, referenced_phi } => {
                write!(f, "Phi {:?} references non-existent phi {:?}", phi_id, referenced_phi)
            }
            SSAViolation::PhiInNonJoinBlock { phi_id, block_id, predecessor_count } => {
                write!(f, "Phi {:?} exists in block {:?} which has only {} predecessors",
                    phi_id, block_id, predecessor_count)
            }
            SSAViolation::UnsealedBlock { block_id } => {
                write!(f, "Block {:?} is not sealed after construction", block_id)
            }
            SSAViolation::EmptyPhi { phi_id, block_id } => {
                write!(f, "Phi {:?} in block {:?} has no operands", phi_id, block_id)
            }
            SSAViolation::OnlySelfReferencePhi { phi_id, block_id } => {
                write!(f, "Phi {:?} in block {:?} only references itself", phi_id, block_id)
            }
            SSAViolation::DanglingPhiInInstruction { block_id, instruction_index, referenced_phi } => {
                write!(f, "Instruction {} in block {:?} references non-existent phi {:?}",
                    instruction_index, block_id, referenced_phi)
            }
            SSAViolation::PhiUsedDirectlyAsOperand { block_id, instruction_index, phi_id } => {
                write!(f, "Instruction {} in block {:?} uses phi {:?} directly",
                    instruction_index, block_id, phi_id)
            }
            SSAViolation::PhiMissingDestination { phi_id, block_id } => {
                write!(f, "Phi {:?} in block {:?} has no destination variable (not materialized)",
                    phi_id, block_id)
            }
            SSAViolation::MultipleDefinitions { variable, definition_sites } => {
                write!(f, "Variable {:?} has multiple definitions at {:?}",
                    variable, definition_sites)
            }
            SSAViolation::UndefinedVariableUse { block_id, instruction_index, variable } => {
                write!(f, "Use of undefined variable {:?} at instruction {} in block {:?}",
                    variable, instruction_index, block_id)
            }
            SSAViolation::DeadPhi { phi_id, block_id } => {
                write!(f, "Dead phi {:?} in block {:?}", phi_id, block_id)
            }
            SSAViolation::DominanceViolation { block_id, instruction_index, variable, def_block } => {
                write!(f, "Dominance violation: {:?} at {} in {:?} not dominated by def in {:?}",
                    variable, instruction_index, block_id, def_block)
            }
            SSAViolation::PhiOperandDominanceViolation { phi_id, block_id, operand_index, predecessor, variable, def_block } => {
                write!(f, "Phi operand dominance violation: {:?} operand {} in {:?} uses {:?} (def in {:?}) which doesn't dominate predecessor {:?}",
                    phi_id, operand_index, block_id, variable, def_block, predecessor)
            }
            SSAViolation::PhiNotInlined { phi_id, block_id, instruction_index } => {
                write!(f, "Phi {:?} referenced as value at instruction {} in block {:?} (should have been replaced with variable)",
                    phi_id, instruction_index, block_id)
            }
        }
    }
}

/// Validate all SSA properties
pub fn validate_ssa<V, I, F>(translator: &SSATranslator<V, I, F>) -> Vec<SSAViolation<V>>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut violations = Vec::new();
    let valid_phi_ids: HashSet<PhiId> = translator.phis.keys().copied().collect();

    // Check each phi
    for (phi_id, phi) in &translator.phis {
        let block = &translator.blocks[phi.block_id.0];

        // Property 1: No trivial phis
        if let Some(violation) = check_trivial_phi(*phi_id, phi, &valid_phi_ids) {
            violations.push(violation);
        }

        // Property 2: Operand count matches predecessor count
        if !phi.operands.is_empty() && block.predecessors.len() > 1 {
            if phi.operands.len() != block.predecessors.len() {
                violations.push(SSAViolation::OperandCountMismatch {
                    phi_id: *phi_id,
                    block_id: phi.block_id,
                    operand_count: phi.operands.len(),
                    predecessor_count: block.predecessors.len(),
                });
            }
        }

        // Property 3: All phi references are valid
        for operand in &phi.operands {
            if let Some(ref_phi_id) = operand.as_phi() {
                if !valid_phi_ids.contains(&ref_phi_id) {
                    violations.push(SSAViolation::DanglingPhiReference {
                        phi_id: *phi_id,
                        referenced_phi: ref_phi_id,
                    });
                }
            }
        }

        // Property 4: Phis only at join points
        if block.predecessors.len() < 2 && !phi.operands.is_empty() {
            violations.push(SSAViolation::PhiInNonJoinBlock {
                phi_id: *phi_id,
                block_id: phi.block_id,
                predecessor_count: block.predecessors.len(),
            });
        }

        // Check for empty phis
        if phi.operands.is_empty() && translator.sealed_blocks.contains(&phi.block_id) {
            violations.push(SSAViolation::EmptyPhi {
                phi_id: *phi_id,
                block_id: phi.block_id,
            });
        }

        // Check that phi has destination after materialization
        // (Only check if block is sealed - unmaterialized phis in unsealed blocks are OK)
        if phi.dest.is_none() && translator.sealed_blocks.contains(&phi.block_id) && !phi.operands.is_empty() {
            violations.push(SSAViolation::PhiMissingDestination {
                phi_id: *phi_id,
                block_id: phi.block_id,
            });
        }
    }

    // Property 5: All blocks should be sealed
    for block in &translator.blocks {
        if !block.sealed {
            violations.push(SSAViolation::UnsealedBlock {
                block_id: block.id,
            });
        }
    }

    // Property 6: All phi references in instructions must be valid
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            instruction.visit_values(|value| {
                if let Some(phi_id) = value.as_phi() {
                    if !valid_phi_ids.contains(&phi_id) {
                        violations.push(SSAViolation::DanglingPhiInInstruction {
                            block_id: block.id,
                            instruction_index: instr_idx,
                            referenced_phi: phi_id,
                        });
                    }
                }
            });
        }
    }

    // Property 7: Phis should only be used in Assign instructions
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            if !instruction.is_phi_assignment() {
                instruction.visit_values(|value| {
                    if let Some(phi_id) = value.as_phi() {
                        violations.push(SSAViolation::PhiUsedDirectlyAsOperand {
                            block_id: block.id,
                            instruction_index: instr_idx,
                            phi_id,
                        });
                    }
                });
            }
        }
    }

    // Property 8: No phi references should remain in instructions
    // After materialization, all Value::Phi(id) should be replaced with Value::Var
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            instruction.visit_values(|value| {
                if let Some(phi_id) = value.as_phi() {
                    violations.push(SSAViolation::PhiNotInlined {
                        phi_id,
                        block_id: block.id,
                        instruction_index: instr_idx,
                    });
                }
            });
        }
    }

    // Collect definition info
    let def_info = collect_definitions(translator);

    // Property 9: Single assignment
    violations.extend(check_single_assignment(&def_info));

    // Property 10: All uses valid
    violations.extend(check_all_uses_valid(translator, &def_info));

    // Property 11: Dominance
    violations.extend(check_dominance(translator, &def_info));

    // Property 12: Dead phis
    violations.extend(check_dead_phis(translator));

    // Property 13: Phi in entry block
    violations.extend(check_phi_entry_block(translator));

    // Property 14: Unreachable blocks
    violations.extend(check_unreachable_blocks(translator));

    violations
}

/// Check that no phi nodes exist in the entry block
/// Entry block has no predecessors, so phi nodes are invalid there
fn check_phi_entry_block<V, I, F>(translator: &SSATranslator<V, I, F>) -> Vec<SSAViolation<V>>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut violations = Vec::new();
    let entry_block = BlockId(0);

    for (phi_id, phi) in &translator.phis {
        if phi.block_id == entry_block && !phi.operands.is_empty() {
            violations.push(SSAViolation::PhiInEntryBlock { phi_id: *phi_id });
        }
    }

    violations
}

/// Check for unreachable blocks (blocks with no predecessors that aren't the entry block)
fn check_unreachable_blocks<V, I, F>(translator: &SSATranslator<V, I, F>) -> Vec<SSAViolation<V>>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut violations = Vec::new();
    let entry_block = BlockId(0);

    for block in &translator.blocks {
        // Skip entry block - it's expected to have no predecessors
        if block.id == entry_block {
            continue;
        }

        // Block has no predecessors - it's unreachable
        if block.predecessors.is_empty() {
            violations.push(SSAViolation::UnreachableBlock { block_id: block.id });
        }
    }

    violations
}

/// Check if a phi is trivial
fn check_trivial_phi<V: SsaValue>(
    phi_id: PhiId,
    phi: &crate::types::Phi<V>,
    _valid_phi_ids: &HashSet<PhiId>,
) -> Option<SSAViolation<V>> {
    if phi.operands.is_empty() {
        return None;
    }

    let mut unique_value: Option<&V> = None;
    let mut has_non_self_operand = false;

    for operand in &phi.operands {
        if operand.is_same_phi(phi_id) {
            continue;
        }

        has_non_self_operand = true;

        match unique_value {
            None => unique_value = Some(operand),
            Some(existing) => {
                if existing != operand {
                    return None;
                }
            }
        }
    }

    if !has_non_self_operand {
        return Some(SSAViolation::OnlySelfReferencePhi {
            phi_id,
            block_id: phi.block_id,
        });
    }

    if let Some(value) = unique_value {
        return Some(SSAViolation::TrivialPhi {
            phi_id,
            block_id: phi.block_id,
            unique_operand: value.clone(),
        });
    }

    None
}

/// Pre-computed definition information
struct DefinitionInfo {
    definitions: HashMap<SsaVariable, Vec<(BlockId, usize)>>,
    def_blocks: HashMap<SsaVariable, BlockId>,
    defined_vars: HashSet<SsaVariable>,
}

/// Collect all definition information
fn collect_definitions<V, I, F>(translator: &SSATranslator<V, I, F>) -> DefinitionInfo
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut definitions: HashMap<SsaVariable, Vec<(BlockId, usize)>> = HashMap::new();
    let mut def_blocks: HashMap<SsaVariable, BlockId> = HashMap::new();
    let mut defined_vars: HashSet<SsaVariable> = HashSet::new();

    // Collect definitions from instructions
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            if let Some(var) = instruction.destination() {
                definitions
                    .entry(var.clone())
                    .or_default()
                    .push((block.id, instr_idx));
                def_blocks.insert(var.clone(), block.id);
                defined_vars.insert(var.clone());
            }
        }
    }

    // Collect definitions from phi destinations
    // Phi destinations are defined at the beginning of the block (instruction index 0)
    for phi in translator.phis.values() {
        if let Some(ref dest) = phi.dest {
            definitions
                .entry(dest.clone())
                .or_default()
                .push((phi.block_id, 0));
            def_blocks.insert(dest.clone(), phi.block_id);
            defined_vars.insert(dest.clone());
        }
    }

    DefinitionInfo {
        definitions,
        def_blocks,
        defined_vars,
    }
}

/// Check single assignment property
fn check_single_assignment<V: SsaValue>(def_info: &DefinitionInfo) -> Vec<SSAViolation<V>> {
    let mut violations = Vec::new();

    for (variable, sites) in &def_info.definitions {
        if sites.len() > 1 {
            violations.push(SSAViolation::MultipleDefinitions {
                variable: variable.clone(),
                definition_sites: sites.clone(),
            });
        }
    }

    violations
}

/// Check all uses are valid
fn check_all_uses_valid<V, I, F>(
    translator: &SSATranslator<V, I, F>,
    def_info: &DefinitionInfo,
) -> Vec<SSAViolation<V>>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut violations = Vec::new();

    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            instruction.visit_values(|value| {
                if let Some(var) = value.as_var() {
                    if !def_info.defined_vars.contains(var) {
                        violations.push(SSAViolation::UndefinedVariableUse {
                            block_id: block.id,
                            instruction_index: instr_idx,
                            variable: var.clone(),
                        });
                    }
                }
            });
        }
    }

    violations
}

/// Compute dominators for all blocks
fn compute_dominators<V, I, F>(translator: &SSATranslator<V, I, F>) -> HashMap<BlockId, HashSet<BlockId>>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut dominators: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();

    if translator.blocks.is_empty() {
        return dominators;
    }

    let entry = BlockId(0);
    let all_blocks: HashSet<BlockId> = translator.blocks.iter().map(|b| b.id).collect();

    for block in &translator.blocks {
        if block.id == entry {
            let mut entry_doms = HashSet::new();
            entry_doms.insert(entry);
            dominators.insert(entry, entry_doms);
        } else {
            dominators.insert(block.id, all_blocks.clone());
        }
    }

    let mut changed = true;
    while changed {
        changed = false;

        for block in &translator.blocks {
            if block.id == entry {
                continue;
            }

            let mut new_doms: Option<HashSet<BlockId>> = None;

            for pred in &block.predecessors {
                let pred_doms = dominators.get(pred).cloned().unwrap_or_default();
                new_doms = match new_doms {
                    None => Some(pred_doms),
                    Some(current) => Some(current.intersection(&pred_doms).cloned().collect()),
                };
            }

            let mut new_doms = new_doms.unwrap_or_default();
            new_doms.insert(block.id);

            if dominators.get(&block.id) != Some(&new_doms) {
                dominators.insert(block.id, new_doms);
                changed = true;
            }
        }
    }

    dominators
}

/// Check dominance property
fn check_dominance<V, I, F>(
    translator: &SSATranslator<V, I, F>,
    def_info: &DefinitionInfo,
) -> Vec<SSAViolation<V>>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut violations = Vec::new();
    let dominators = compute_dominators(translator);

    // Check instruction operand dominance
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            instruction.visit_values(|value| {
                if let Some(var) = value.as_var() {
                    if let Some(&def_block) = def_info.def_blocks.get(var) {
                        if def_block == block.id {
                            // Same block - check instruction order
                            let def_idx = block.instructions.iter().position(|instr| {
                                instr.destination() == Some(var)
                            });

                            if let Some(def_idx) = def_idx {
                                if def_idx > instr_idx {
                                    violations.push(SSAViolation::DominanceViolation {
                                        block_id: block.id,
                                        instruction_index: instr_idx,
                                        variable: var.clone(),
                                        def_block,
                                    });
                                }
                            }
                        } else {
                            // Different blocks - check dominance
                            let block_doms = dominators.get(&block.id).cloned().unwrap_or_default();
                            if !block_doms.contains(&def_block) {
                                violations.push(SSAViolation::DominanceViolation {
                                    block_id: block.id,
                                    instruction_index: instr_idx,
                                    variable: var.clone(),
                                    def_block,
                                });
                            }
                        }
                    }
                }
            });
        }
    }

    // Check phi operand dominance
    // Each phi operand must dominate its corresponding predecessor block
    for (phi_id, phi) in &translator.phis {
        let block = &translator.blocks[phi.block_id.0];

        // Skip incomplete phis (no operands yet) or unsealed blocks
        if phi.operands.is_empty() || !block.sealed {
            continue;
        }

        // Operands and predecessors are parallel arrays
        for (operand_idx, operand) in phi.operands.iter().enumerate() {
            // Get the corresponding predecessor
            let predecessor = if operand_idx < block.predecessors.len() {
                block.predecessors[operand_idx]
            } else {
                continue; // Operand count mismatch is reported elsewhere
            };

            // Check if operand is a variable
            if let Some(var) = operand.as_var() {
                if let Some(&def_block) = def_info.def_blocks.get(var) {
                    // The definition must dominate the predecessor block
                    let pred_doms = dominators.get(&predecessor).cloned().unwrap_or_default();
                    if !pred_doms.contains(&def_block) {
                        violations.push(SSAViolation::PhiOperandDominanceViolation {
                            phi_id: *phi_id,
                            block_id: phi.block_id,
                            operand_index: operand_idx,
                            predecessor,
                            variable: var.clone(),
                            def_block,
                        });
                    }
                }
            }
        }
    }

    violations
}

/// Check for dead phis (phis with no non-phi users)
fn check_dead_phis<V, I, F>(translator: &SSATranslator<V, I, F>) -> Vec<SSAViolation<V>>
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let mut violations = Vec::new();

    // Build a map of phi_id -> assigned variable using phi.dest
    let phi_to_var: HashMap<PhiId, SsaVariable> = translator.phis
        .iter()
        .filter_map(|(phi_id, phi)| {
            phi.dest.as_ref().map(|dest| (*phi_id, dest.clone()))
        })
        .collect();

    // Collect all phis that are "used" by non-phi instructions
    let mut used_phis: HashSet<PhiId> = HashSet::new();

    // First pass: find phis whose destination variable is used in instructions
    for block in &translator.blocks {
        for instruction in &block.instructions {
            instruction.visit_values(|value| {
                // Direct phi use (shouldn't happen in well-formed SSA, but check anyway)
                if let Some(phi_id) = value.as_phi() {
                    used_phis.insert(phi_id);
                }
                // Variable use - check if it's a phi result
                if let Some(var) = value.as_var() {
                    for (phi_id, phi_var) in &phi_to_var {
                        if phi_var == var {
                            used_phis.insert(*phi_id);
                        }
                    }
                }
            });
        }
    }

    // Second pass: propagate "used" status through phi references
    // (if phi A uses phi B's result, and A is used, then B is also used)
    let mut changed = true;
    while changed {
        changed = false;
        for (phi_id, phi) in &translator.phis {
            if used_phis.contains(phi_id) {
                // This phi is used, mark all phis it references as used
                for operand in &phi.operands {
                    if let Some(ref_phi_id) = operand.as_phi() {
                        if !used_phis.contains(&ref_phi_id) {
                            used_phis.insert(ref_phi_id);
                            changed = true;
                        }
                    }
                    // Also check if operand is a variable that's a phi result
                    if let Some(var) = operand.as_var() {
                        for (other_phi_id, phi_var) in &phi_to_var {
                            if phi_var == var && !used_phis.contains(other_phi_id) {
                                used_phis.insert(*other_phi_id);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Report dead phis (those not in used_phis)
    for (phi_id, phi) in &translator.phis {
        // Only check phis that have operands (non-empty) and are in sealed blocks
        if !phi.operands.is_empty() && translator.sealed_blocks.contains(&phi.block_id) {
            if !used_phis.contains(phi_id) {
                violations.push(SSAViolation::DeadPhi {
                    phi_id: *phi_id,
                    block_id: phi.block_id,
                });
            }
        }
    }

    violations
}

/// Assert no violations occurred
pub fn assert_valid_ssa<V, I, F>(translator: &SSATranslator<V, I, F>)
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    let violations = validate_ssa(translator);
    if !violations.is_empty() {
        let mut msg = String::from("SSA validation failed:\n");
        for v in &violations {
            msg.push_str(&format!("  - {}\n", v));
        }
        panic!("{}", msg);
    }
}

/// Print debug information about SSA state
pub fn debug_ssa_state<V, I, F>(translator: &SSATranslator<V, I, F>)
where
    V: SsaValue,
    I: SsaInstruction<Value = V>,
    F: InstructionFactory<Instr = I>,
{
    println!("=== SSA State Debug ===");
    println!("\nBlocks ({}):", translator.blocks.len());
    for block in &translator.blocks {
        println!("  Block {:?} (sealed: {}, predecessors: {:?})",
            block.id, block.sealed, block.predecessors);
        for (i, instr) in block.instructions.iter().enumerate() {
            println!("    [{}] {:?}", i, instr);
        }
    }

    println!("\nPhis ({}):", translator.phis.len());
    for (phi_id, phi) in &translator.phis {
        let dest_str = phi.dest.as_ref()
            .map(|v| v.name().to_string())
            .unwrap_or_else(|| "?".to_string());
        println!("  {} = φ{:?} in block {:?}: operands={:?}",
            dest_str, phi_id, phi.block_id, phi.operands);
    }

    println!("\nIncomplete Phis:");
    for (block_id, phis) in &translator.incomplete_phis {
        println!("  Block {:?}: {:?}", block_id, phis);
    }

    println!("\nSealed Blocks: {:?}", translator.sealed_blocks);

    let violations = validate_ssa(translator);
    println!("\nViolations ({}):", violations.len());
    for v in &violations {
        println!("  - {}", v);
    }
    println!("=== End Debug ===\n");
}
