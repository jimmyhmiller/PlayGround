/// SSA Property Tests
///
/// Based on "Simple and Efficient Construction of Static Single Assignment Form"
/// by Braun et al. (CC 2013)
///
/// Key properties that valid SSA must satisfy:
/// 1. No trivial phis - a phi is trivial if all operands are the same (ignoring self-references)
/// 2. Phi operand count matches predecessor count
/// 3. All phi references are valid (no dangling references)
/// 4. Phis only exist at join points (blocks with multiple predecessors)
/// 5. All blocks are sealed after construction

use crate::instruction::{BlockId, PhiId, Value};
use crate::SSATranslator;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub enum SSAViolation {
    /// Phi has all same operands (should have been removed as trivial)
    TrivialPhi {
        phi_id: PhiId,
        block_id: BlockId,
        unique_operand: Value,
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
    /// Self-referential phi with no other operands (undefined behavior)
    OnlySelfReferencePhi {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Phi operand references undefined value where it shouldn't
    UndefinedOperand {
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
    /// Phi assignment not at the beginning of block
    PhiNotAtBlockStart {
        block_id: BlockId,
        instruction_index: usize,
        phi_id: PhiId,
    },
    /// Phi operands all resolve to the same value (should be trivial after copy propagation)
    PhiOperandsFromSameSource {
        phi_id: PhiId,
        block_id: BlockId,
        resolved_value: Value,
        operands: Vec<Value>,
    },
    /// Variable is defined more than once (violates single assignment)
    MultipleDefinitions {
        variable: crate::instruction::Variable,
        definition_sites: Vec<(BlockId, usize)>, // (block, instruction index)
    },
    /// Use of undefined variable
    UndefinedVariableUse {
        block_id: BlockId,
        instruction_index: usize,
        variable: crate::instruction::Variable,
    },
    /// Dead phi - phi has no non-phi users (not pruned)
    DeadPhi {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Redundant phi SCC - mutually recursive phis that all resolve to same value
    RedundantPhiSCC {
        phi_ids: Vec<PhiId>,
    },
    /// Use does not have definition that dominates it
    DominanceViolation {
        block_id: BlockId,
        instruction_index: usize,
        variable: crate::instruction::Variable,
        def_block: BlockId,
    },
}

impl std::fmt::Display for SSAViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SSAViolation::TrivialPhi { phi_id, block_id, unique_operand } => {
                write!(f, "Trivial phi {:?} in block {:?}: all operands are {:?}",
                    phi_id, block_id, unique_operand)
            }
            SSAViolation::OperandCountMismatch { phi_id, block_id, operand_count, predecessor_count } => {
                write!(f, "Phi {:?} in block {:?} has {} operands but block has {} predecessors",
                    phi_id, block_id, operand_count, predecessor_count)
            }
            SSAViolation::DanglingPhiReference { phi_id, referenced_phi } => {
                write!(f, "Phi {:?} references non-existent phi {:?}", phi_id, referenced_phi)
            }
            SSAViolation::PhiInNonJoinBlock { phi_id, block_id, predecessor_count } => {
                write!(f, "Phi {:?} exists in block {:?} which has only {} predecessors (not a join point)",
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
            SSAViolation::UndefinedOperand { phi_id, block_id } => {
                write!(f, "Phi {:?} in block {:?} has undefined operand (may indicate bug)",
                    phi_id, block_id)
            }
            SSAViolation::DanglingPhiInInstruction { block_id, instruction_index, referenced_phi } => {
                write!(f, "Instruction {} in block {:?} references non-existent phi {:?}",
                    instruction_index, block_id, referenced_phi)
            }
            SSAViolation::PhiUsedDirectlyAsOperand { block_id, instruction_index, phi_id } => {
                write!(f, "Instruction {} in block {:?} uses phi {:?} directly as operand (should assign to var first)",
                    instruction_index, block_id, phi_id)
            }
            SSAViolation::PhiNotAtBlockStart { block_id, instruction_index, phi_id } => {
                write!(f, "Phi {:?} assignment at instruction {} in block {:?} (phis must be at block start)",
                    phi_id, instruction_index, block_id)
            }
            SSAViolation::PhiOperandsFromSameSource { phi_id, block_id, resolved_value, operands } => {
                write!(f, "Phi {:?} in block {:?} has operands {:?} that all resolve to {:?} (trivial after copy propagation)",
                    phi_id, block_id, operands, resolved_value)
            }
            SSAViolation::MultipleDefinitions { variable, definition_sites } => {
                write!(f, "Variable {:?} has multiple definitions at {:?} (violates single assignment)",
                    variable, definition_sites)
            }
            SSAViolation::UndefinedVariableUse { block_id, instruction_index, variable } => {
                write!(f, "Use of undefined variable {:?} at instruction {} in block {:?}",
                    variable, instruction_index, block_id)
            }
            SSAViolation::DeadPhi { phi_id, block_id } => {
                write!(f, "Dead phi {:?} in block {:?} has no non-phi users (not pruned)",
                    phi_id, block_id)
            }
            SSAViolation::RedundantPhiSCC { phi_ids } => {
                write!(f, "Redundant phi SCC: {:?} are mutually recursive and resolve to same value",
                    phi_ids)
            }
            SSAViolation::DominanceViolation { block_id, instruction_index, variable, def_block } => {
                write!(f, "Dominance violation: use of {:?} at instruction {} in block {:?} is not dominated by definition in block {:?}",
                    variable, instruction_index, block_id, def_block)
            }
        }
    }
}

/// Validates all SSA properties and returns a list of violations
pub fn validate_ssa(translator: &SSATranslator) -> Vec<SSAViolation> {
    let mut violations = Vec::new();

    // Collect all valid phi IDs
    let valid_phi_ids: HashSet<PhiId> = translator.phis.keys().copied().collect();

    // Check each phi
    for (phi_id, phi) in &translator.phis {
        let block = &translator.blocks[phi.block_id.0];

        // Property 1: No trivial phis
        if let Some(violation) = check_trivial_phi(*phi_id, phi, &valid_phi_ids) {
            violations.push(violation);
        }

        // Property 1b: Phi operands shouldn't all resolve to the same source (trivial after copy propagation)
        if let Some(violation) = check_phi_operands_same_source(*phi_id, phi, translator) {
            violations.push(violation);
        }

        // Property 2: Operand count matches predecessor count
        // (Only check for sealed blocks with operands)
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
            if let Value::Phi(ref_phi_id) = operand {
                if !valid_phi_ids.contains(ref_phi_id) {
                    violations.push(SSAViolation::DanglingPhiReference {
                        phi_id: *phi_id,
                        referenced_phi: *ref_phi_id,
                    });
                }
            }
        }

        // Property 4: Phis only at join points (2+ predecessors)
        // Exception: entry block can have phis for parameters, but this simple language doesn't have params
        if block.predecessors.len() < 2 && !phi.operands.is_empty() {
            violations.push(SSAViolation::PhiInNonJoinBlock {
                phi_id: *phi_id,
                block_id: phi.block_id,
                predecessor_count: block.predecessors.len(),
            });
        }

        // Check for empty phis (incomplete construction)
        if phi.operands.is_empty() && translator.sealed_blocks.contains(&phi.block_id) {
            violations.push(SSAViolation::EmptyPhi {
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
            for phi_id in get_phi_references_in_instruction(instruction) {
                if !valid_phi_ids.contains(&phi_id) {
                    violations.push(SSAViolation::DanglingPhiInInstruction {
                        block_id: block.id,
                        instruction_index: instr_idx,
                        referenced_phi: phi_id,
                    });
                }
            }
        }
    }

    // Property 7: Phis should only be used in Assign instructions (materialized to a variable)
    // Not directly as operands in BinaryOp, UnaryOp, ConditionalJump, Print
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            for phi_id in get_invalid_phi_uses(instruction) {
                violations.push(SSAViolation::PhiUsedDirectlyAsOperand {
                    block_id: block.id,
                    instruction_index: instr_idx,
                    phi_id,
                });
            }
        }
    }

    // Property 8: All phi assignments must be at the beginning of blocks
    // Once we see a non-phi instruction, no more phi assignments are allowed
    for block in &translator.blocks {
        let mut seen_non_phi = false;
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            if let Some(phi_id) = get_phi_assignment(instruction) {
                if seen_non_phi {
                    violations.push(SSAViolation::PhiNotAtBlockStart {
                        block_id: block.id,
                        instruction_index: instr_idx,
                        phi_id,
                    });
                }
            } else {
                seen_non_phi = true;
            }
        }
    }

    // Collect definition info once for efficiency
    let def_info = collect_definitions(translator);

    // Property 9: Single assignment - each SSA variable defined exactly once
    violations.extend(check_single_assignment(&def_info));

    // Property 10: All uses valid - every variable use refers to a defined variable
    violations.extend(check_all_uses_valid(translator, &def_info));

    // Property 11: No dead phis - every phi has at least one non-phi user (pruned form)
    violations.extend(check_no_dead_phis(translator));

    // Property 12: No redundant phi SCCs - mutually recursive phis resolving to same value
    violations.extend(check_redundant_phi_sccs(translator));

    // Property 13: Dominance - definitions must dominate their uses
    violations.extend(check_dominance(translator, &def_info));

    violations
}

/// Get the destination variable of an instruction, if it defines one
fn get_instruction_destination(instruction: &crate::instruction::Instruction) -> Option<&crate::instruction::Variable> {
    use crate::instruction::Instruction;
    match instruction {
        Instruction::Assign { dest, .. } => Some(dest),
        Instruction::BinaryOp { dest, .. } => Some(dest),
        Instruction::UnaryOp { dest, .. } => Some(dest),
        _ => None,
    }
}

/// Visit all values used in an instruction (generic visitor pattern)
fn visit_instruction_values<F>(instruction: &crate::instruction::Instruction, visitor: &mut F)
where
    F: FnMut(&Value),
{
    use crate::instruction::Instruction;
    match instruction {
        Instruction::Assign { value, .. } => visitor(value),
        Instruction::BinaryOp { left, right, .. } => {
            visitor(left);
            visitor(right);
        }
        Instruction::UnaryOp { operand, .. } => visitor(operand),
        Instruction::ConditionalJump { condition, .. } => visitor(condition),
        Instruction::Print { value } => visitor(value),
        Instruction::Jump { .. } => {}
    }
}

/// Pre-computed definition information for efficiency
struct DefinitionInfo {
    /// Variable -> list of (block, instruction_index) where defined
    definitions: HashMap<crate::instruction::Variable, Vec<(BlockId, usize)>>,
    /// Variable -> block where defined (for dominance check)
    def_blocks: HashMap<crate::instruction::Variable, BlockId>,
    /// Set of all defined variables (for use validation)
    defined_vars: HashSet<crate::instruction::Variable>,
}

/// Collect all definition information in a single pass
fn collect_definitions(translator: &SSATranslator) -> DefinitionInfo {
    let mut definitions: HashMap<crate::instruction::Variable, Vec<(BlockId, usize)>> = HashMap::new();
    let mut def_blocks: HashMap<crate::instruction::Variable, BlockId> = HashMap::new();
    let mut defined_vars: HashSet<crate::instruction::Variable> = HashSet::new();

    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            if let Some(var) = get_instruction_destination(instruction) {
                definitions
                    .entry(var.clone())
                    .or_default()
                    .push((block.id, instr_idx));
                def_blocks.insert(var.clone(), block.id);
                defined_vars.insert(var.clone());
            }
        }
    }

    DefinitionInfo {
        definitions,
        def_blocks,
        defined_vars,
    }
}

/// Check if instruction is a phi assignment, return the phi id if so
fn get_phi_assignment(instruction: &crate::instruction::Instruction) -> Option<PhiId> {
    use crate::instruction::Instruction;

    if let Instruction::Assign { value: Value::Phi(phi_id), .. } = instruction {
        Some(*phi_id)
    } else {
        None
    }
}

/// Get phi references that are invalidly used (not in an Assign instruction)
fn get_invalid_phi_uses(instruction: &crate::instruction::Instruction) -> Vec<PhiId> {
    use crate::instruction::Instruction;

    // Phi in Assign is VALID (that's how phis get materialized)
    // Phi anywhere else is INVALID
    if matches!(instruction, Instruction::Assign { .. }) {
        return Vec::new();
    }

    let mut invalid_phis = Vec::new();
    visit_instruction_values(instruction, &mut |value| {
        if let Value::Phi(phi_id) = value {
            invalid_phis.push(*phi_id);
        }
    });
    invalid_phis
}

/// Extract all phi references from an instruction
fn get_phi_references_in_instruction(instruction: &crate::instruction::Instruction) -> Vec<PhiId> {
    let mut phi_refs = Vec::new();
    visit_instruction_values(instruction, &mut |value| {
        if let Value::Phi(phi_id) = value {
            phi_refs.push(*phi_id);
        }
    });
    phi_refs
}

/// Trace a value back through copy assignments to find the original source
/// Returns the original value that this traces back to
fn trace_value_to_source(value: &Value, translator: &SSATranslator) -> Value {
    match value {
        Value::Var(var) => {
            // Look for an assignment `var := something` in any block
            for block in &translator.blocks {
                for instruction in &block.instructions {
                    if let crate::instruction::Instruction::Assign { dest, value: assigned_value } = instruction {
                        if dest == var {
                            // Found the assignment - if it's a simple copy, recurse
                            match assigned_value {
                                Value::Var(_) => return trace_value_to_source(assigned_value, translator),
                                Value::Phi(_) => return value.clone(), // Stop at phis
                                _ => return assigned_value.clone(), // Literals, etc.
                            }
                        }
                    }
                }
            }
            // No assignment found, return as-is
            value.clone()
        }
        _ => value.clone(),
    }
}

/// Check if all phi operands resolve to the same source value
fn check_phi_operands_same_source(
    phi_id: PhiId,
    phi: &crate::instruction::Phi,
    translator: &SSATranslator,
) -> Option<SSAViolation> {
    if phi.operands.len() < 2 {
        return None;
    }

    let mut resolved_values: Vec<Value> = Vec::new();
    for operand in &phi.operands {
        // Skip self-references
        if let Value::Phi(op_phi_id) = operand {
            if *op_phi_id == phi_id {
                continue;
            }
        }
        resolved_values.push(trace_value_to_source(operand, translator));
    }

    if resolved_values.is_empty() {
        return None;
    }

    // Check if all resolved values are the same
    let first = &resolved_values[0];
    if resolved_values.iter().all(|v| v == first) {
        return Some(SSAViolation::PhiOperandsFromSameSource {
            phi_id,
            block_id: phi.block_id,
            resolved_value: first.clone(),
            operands: phi.operands.clone(),
        });
    }

    None
}

/// Check if a phi is trivial (all operands are the same, excluding self-references)
fn check_trivial_phi(
    phi_id: PhiId,
    phi: &crate::instruction::Phi,
    _valid_phi_ids: &HashSet<PhiId>,
) -> Option<SSAViolation> {
    if phi.operands.is_empty() {
        return None; // Empty phis are caught elsewhere
    }

    let mut unique_value: Option<&Value> = None;
    let mut has_non_self_operand = false;

    for operand in &phi.operands {
        // Skip self-references
        if let Value::Phi(op_phi_id) = operand {
            if *op_phi_id == phi_id {
                continue;
            }
        }

        has_non_self_operand = true;

        match unique_value {
            None => unique_value = Some(operand),
            Some(existing) => {
                if existing != operand {
                    // Found two different operands, not trivial
                    return None;
                }
            }
        }
    }

    // If all operands were self-references
    if !has_non_self_operand {
        return Some(SSAViolation::OnlySelfReferencePhi {
            phi_id,
            block_id: phi.block_id,
        });
    }

    // If we found a unique value (all non-self operands are the same), it's trivial
    if let Some(value) = unique_value {
        return Some(SSAViolation::TrivialPhi {
            phi_id,
            block_id: phi.block_id,
            unique_operand: value.clone(),
        });
    }

    None
}

/// Check that each SSA variable has exactly one definition (single assignment property)
fn check_single_assignment(def_info: &DefinitionInfo) -> Vec<SSAViolation> {
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

/// Check that all variable uses reference defined variables
fn check_all_uses_valid(translator: &SSATranslator, def_info: &DefinitionInfo) -> Vec<SSAViolation> {
    let mut violations = Vec::new();

    // Check all uses against pre-computed defined variables
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            let used_vars = get_used_variables(instruction);
            for var in used_vars {
                if !def_info.defined_vars.contains(&var) {
                    violations.push(SSAViolation::UndefinedVariableUse {
                        block_id: block.id,
                        instruction_index: instr_idx,
                        variable: var,
                    });
                }
            }
        }
    }

    violations
}

/// Get all variables used in an instruction
fn get_used_variables(instruction: &crate::instruction::Instruction) -> Vec<crate::instruction::Variable> {
    let mut vars = Vec::new();
    visit_instruction_values(instruction, &mut |value| {
        if let Value::Var(var) = value {
            vars.push(var.clone());
        }
    });
    vars
}

/// Check that every phi has at least one non-phi user (pruned SSA form)
fn check_no_dead_phis(translator: &SSATranslator) -> Vec<SSAViolation> {
    use crate::instruction::Instruction;

    let mut violations = Vec::new();

    // For each phi, check if it has any non-phi users
    for (phi_id, phi) in &translator.phis {
        let mut has_non_phi_user = false;

        // Find the variable that this phi is assigned to
        let phi_var = find_phi_variable(*phi_id, translator);

        if let Some(var) = phi_var {
            // Check if this variable is used anywhere (not just in phi operands)
            for block in &translator.blocks {
                for instruction in &block.instructions {
                    // Skip the phi assignment itself
                    if let Instruction::Assign { dest, value: Value::Phi(id) } = instruction {
                        if id == phi_id && dest == &var {
                            continue;
                        }
                    }

                    let used_vars = get_used_variables(instruction);
                    if used_vars.contains(&var) {
                        // Check if this use is in a phi operand
                        let is_phi_use = matches!(instruction, Instruction::Assign { value: Value::Phi(_), .. });
                        if !is_phi_use {
                            has_non_phi_user = true;
                            break;
                        }
                    }
                }
                if has_non_phi_user {
                    break;
                }
            }

            // Also check if it's used as a phi operand that eventually has a non-phi user
            // (transitive check - for now we do a simple version)
            if !has_non_phi_user {
                has_non_phi_user = phi_has_transitive_non_phi_user(*phi_id, translator, &mut HashSet::new());
            }
        }

        if !has_non_phi_user {
            violations.push(SSAViolation::DeadPhi {
                phi_id: *phi_id,
                block_id: phi.block_id,
            });
        }
    }

    violations
}

/// Find the variable that a phi is assigned to
fn find_phi_variable(phi_id: PhiId, translator: &SSATranslator) -> Option<crate::instruction::Variable> {
    use crate::instruction::Instruction;

    for block in &translator.blocks {
        for instruction in &block.instructions {
            if let Instruction::Assign { dest, value: Value::Phi(id) } = instruction {
                if *id == phi_id {
                    return Some(dest.clone());
                }
            }
        }
    }
    None
}

/// Check if a phi has any non-phi user transitively
fn phi_has_transitive_non_phi_user(
    phi_id: PhiId,
    translator: &SSATranslator,
    visited: &mut HashSet<PhiId>,
) -> bool {
    use crate::instruction::Instruction;

    if visited.contains(&phi_id) {
        return false; // Cycle detected
    }
    visited.insert(phi_id);

    let phi_var = match find_phi_variable(phi_id, translator) {
        Some(v) => v,
        None => return false,
    };

    // Check all uses of this phi's variable in instructions
    for block in &translator.blocks {
        for instruction in &block.instructions {
            let used_vars = get_used_variables(instruction);
            if used_vars.contains(&phi_var) {
                match instruction {
                    Instruction::Assign { dest, value: Value::Phi(id) } => {
                        // Skip if this is the phi assignment itself
                        if *id == phi_id && dest == &phi_var {
                            continue;
                        }
                        // This use is in another phi - check transitively
                        if phi_has_transitive_non_phi_user(*id, translator, visited) {
                            return true;
                        }
                    }
                    _ => {
                        // Non-phi use found
                        return true;
                    }
                }
            }
        }
    }

    // Also check if this phi variable is used as an operand in other phis
    for (other_phi_id, other_phi) in &translator.phis {
        if *other_phi_id == phi_id {
            continue;
        }
        for operand in &other_phi.operands {
            if let Value::Var(var) = operand {
                if var == &phi_var {
                    // This phi's value is used as an operand in another phi
                    if phi_has_transitive_non_phi_user(*other_phi_id, translator, visited) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Detect redundant phi SCCs (mutually recursive phis that resolve to the same value)
fn check_redundant_phi_sccs(translator: &SSATranslator) -> Vec<SSAViolation> {
    let mut violations = Vec::new();
    let mut visited_global: HashSet<PhiId> = HashSet::new();

    for phi_id in translator.phis.keys() {
        if visited_global.contains(phi_id) {
            continue;
        }

        // Find the SCC containing this phi
        let scc = find_phi_scc(*phi_id, translator);

        if scc.len() > 1 {
            // Check if all phis in this SCC resolve to the same non-phi value
            let mut external_values: HashSet<Value> = HashSet::new();

            for scc_phi_id in &scc {
                if let Some(phi) = translator.phis.get(scc_phi_id) {
                    for operand in &phi.operands {
                        match operand {
                            Value::Phi(op_id) if scc.contains(op_id) => {
                                // Internal to SCC, skip
                            }
                            _ => {
                                external_values.insert(operand.clone());
                            }
                        }
                    }
                }
            }

            // If all external values are the same, this SCC is redundant
            if external_values.len() == 1 {
                violations.push(SSAViolation::RedundantPhiSCC {
                    phi_ids: scc.clone(),
                });
            }
        }

        visited_global.extend(scc);
    }

    violations
}

/// Find the strongly connected component containing a phi
fn find_phi_scc(start_phi: PhiId, translator: &SSATranslator) -> Vec<PhiId> {
    // Simple DFS-based SCC detection for phi nodes
    let mut scc = Vec::new();
    let mut stack = vec![start_phi];
    let mut visited: HashSet<PhiId> = HashSet::new();

    while let Some(phi_id) = stack.pop() {
        if visited.contains(&phi_id) {
            continue;
        }
        visited.insert(phi_id);
        scc.push(phi_id);

        if let Some(phi) = translator.phis.get(&phi_id) {
            for operand in &phi.operands {
                if let Value::Phi(op_phi_id) = operand {
                    if !visited.contains(op_phi_id) {
                        stack.push(*op_phi_id);
                    }
                }
            }
        }
    }

    // Verify it's actually a cycle (all members reference each other)
    // For simplicity, we consider any connected component
    scc
}

/// Compute dominators for all blocks
fn compute_dominators(translator: &SSATranslator) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut dominators: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();

    if translator.blocks.is_empty() {
        return dominators;
    }

    let entry = BlockId(0);

    // Initialize: entry dominates only itself, others dominated by all
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

    // Iterate until fixed point
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
            new_doms.insert(block.id); // A block dominates itself

            if dominators.get(&block.id) != Some(&new_doms) {
                dominators.insert(block.id, new_doms);
                changed = true;
            }
        }
    }

    dominators
}

/// Check the dominance property: definitions must dominate their uses
fn check_dominance(translator: &SSATranslator, def_info: &DefinitionInfo) -> Vec<SSAViolation> {
    let mut violations = Vec::new();
    let dominators = compute_dominators(translator);

    // Check that each use is dominated by its definition
    for block in &translator.blocks {
        for (instr_idx, instruction) in block.instructions.iter().enumerate() {
            let used_vars = get_used_variables(instruction);

            for var in used_vars {
                if let Some(&def_block) = def_info.def_blocks.get(&var) {
                    // Special case: if use and def are in the same block,
                    // we need to check instruction order
                    if def_block == block.id {
                        // Find definition index
                        let def_idx = block.instructions.iter().position(|instr| {
                            get_instruction_destination(instr) == Some(&var)
                        });

                        if let Some(def_idx) = def_idx {
                            if def_idx > instr_idx {
                                // Use before definition in same block
                                violations.push(SSAViolation::DominanceViolation {
                                    block_id: block.id,
                                    instruction_index: instr_idx,
                                    variable: var,
                                    def_block,
                                });
                            }
                        }
                    } else {
                        // Different blocks: check dominance
                        let block_doms = dominators.get(&block.id).cloned().unwrap_or_default();
                        if !block_doms.contains(&def_block) {
                            violations.push(SSAViolation::DominanceViolation {
                                block_id: block.id,
                                instruction_index: instr_idx,
                                variable: var,
                                def_block,
                            });
                        }
                    }
                }
            }
        }
    }

    violations
}

/// Helper to assert no violations occurred
pub fn assert_valid_ssa(translator: &SSATranslator) {
    let violations = validate_ssa(translator);
    if !violations.is_empty() {
        let mut msg = String::from("SSA validation failed:\n");
        for v in &violations {
            msg.push_str(&format!("  - {}\n", v));
        }
        panic!("{}", msg);
    }
}

/// Print a detailed report of SSA state (useful for debugging)
pub fn debug_ssa_state(translator: &SSATranslator) {
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
        println!("  {:?} in block {:?}: operands={:?}", phi_id, phi.block_id, phi.operands);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{program, ast};

    /// Test 1: Simple straight-line code (no phis needed)
    #[test]
    fn test_straight_line_no_phis() {
        let program = program! {
            (set x 1)
            (set y 2)
            (set z (+ (var x) (var y)))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);

        // Should have no phis in straight-line code
        let non_empty_phis: Vec<_> = translator.phis.values()
            .filter(|p| !p.operands.is_empty())
            .collect();
        assert!(non_empty_phis.is_empty(),
            "Straight-line code should have no phis, found: {:?}", non_empty_phis);

        assert_valid_ssa(&translator);
    }

    /// Test 2: If-then-else with different assignments (phi needed)
    #[test]
    fn test_if_else_needs_phi() {
        let program = program! {
            (set x 0)
            (if (> 5 3)
                (set x 1)
                (set x 2))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);

        // Should have exactly one non-trivial phi for x at the merge point
        let non_trivial_phis: Vec<_> = translator.phis.values()
            .filter(|p| p.operands.len() >= 2)
            .filter(|p| {
                let unique: HashSet<_> = p.operands.iter()
                    .filter(|op| !matches!(op, Value::Phi(id) if *id == p.id))
                    .collect();
                unique.len() > 1
            })
            .collect();

        assert!(!non_trivial_phis.is_empty(),
            "If-else with different assignments should have a phi");
    }

    /// Test 3: If-then-else with same assignment (phi should be trivial and removed)
    #[test]
    fn test_if_else_same_value_no_phi() {
        let program = program! {
            (set x 0)
            (if (> 5 3)
                (set x 1)
                (set x 1))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);

        // The phi for x should be trivial (both branches assign 1) and removed
        let violations = validate_ssa(&translator);
        let trivial_violations: Vec<_> = violations.iter()
            .filter(|v| matches!(v, SSAViolation::TrivialPhi { .. }))
            .collect();

        assert!(trivial_violations.is_empty(),
            "Same value in both branches should result in trivial phi removal: {:?}",
            trivial_violations);
    }

    /// Test 4: While loop (phi needed at loop header)
    #[test]
    fn test_while_loop_needs_phi() {
        let program = program! {
            (set x 10)
            (while (> (var x) 0)
                (set x (- (var x) 1)))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 5: Variable used only in condition, not modified (no phi)
    #[test]
    fn test_unmodified_variable_no_phi() {
        let program = program! {
            (set x 5)
            (set y 0)
            (if (> (var x) 3)
                (set y 1)
                (set y 2))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 6: Nested if statements
    #[test]
    fn test_nested_if() {
        let program = program! {
            (set x 0)
            (if (> 5 3)
                (if (> 2 1)
                    (set x 1)
                    (set x 2))
                (set x 3))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 7: Multiple variables with different phi requirements
    #[test]
    fn test_multiple_variables() {
        let program = program! {
            (set x 1)
            (set y 2)
            (if (> (var x) 0)
                (begin
                    (set x 10)
                    (set y 20))
                (begin
                    (set x 100)))
            (print (var x))
            (print (var y))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 8: While loop with multiple variables
    #[test]
    fn test_while_multiple_variables() {
        let program = program! {
            (set i 0)
            (set sum 0)
            (while (< (var i) 10)
                (begin
                    (set sum (+ (var sum) (var i)))
                    (set i (+ (var i) 1))))
            (print (var sum))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 9: If without else (one branch doesn't modify)
    #[test]
    fn test_if_without_else() {
        let program = program! {
            (set x 1)
            (if (> 5 3)
                (set x 2))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 10: Complex control flow - if inside while
    #[test]
    fn test_if_inside_while() {
        let program = program! {
            (set x 10)
            (set y 0)
            (while (> (var x) 0)
                (begin
                    (if (> (var x) 5)
                        (set y (+ (var y) 2))
                        (set y (+ (var y) 1)))
                    (set x (- (var x) 1))))
            (print (var y))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }
}
