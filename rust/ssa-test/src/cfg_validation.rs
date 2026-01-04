//! CFG (Control Flow Graph) Validation
//!
//! This module provides validation for CFG properties including:
//! - Entry block invariants
//! - Edge symmetry (successor/predecessor consistency)
//! - Reachability from entry
//! - Proper termination
//! - Critical edge detection
//! - Reducibility analysis

use std::collections::HashSet;
use std::fmt::Debug;

use crate::cfg::{Cfg, CfgBlockId, CfgInstruction, ControlFlow};

/// CFG validation violation types
#[derive(Debug, Clone, PartialEq)]
pub enum CfgViolation<L: Clone + Debug> {
    // P0: Core invariants
    /// Entry block has predecessors (should have none)
    EntryBlockHasPredecessors {
        entry: CfgBlockId,
        predecessor_count: usize,
    },
    /// Block claims successor that doesn't list it as predecessor
    SuccessorPredecessorMismatch {
        block: CfgBlockId,
        claims_successor: CfgBlockId,
    },
    /// Block claims predecessor that doesn't list it as successor
    PredecessorSuccessorMismatch {
        block: CfgBlockId,
        claims_predecessor: CfgBlockId,
    },
    /// Block is not reachable from entry
    UnreachableBlock {
        block: CfgBlockId,
    },

    // P1: Termination & edges
    /// No exit block exists (potential infinite loop)
    NoExitBlock,
    /// Block has no successors but last instruction isn't a terminator
    MissingTerminator {
        block: CfgBlockId,
    },
    /// Block has successors but last instruction is a terminator
    TerminatorWithSuccessors {
        block: CfgBlockId,
        successor_count: usize,
    },
    /// Jump target label doesn't exist in CFG
    DanglingJumpTarget {
        block: CfgBlockId,
        target_label: L,
    },

    // P2: Consistency
    /// Block.id doesn't match its index in the blocks array
    BlockIndexMismatch {
        block_id: CfgBlockId,
        actual_index: usize,
    },
    /// Same successor appears multiple times
    DuplicateSuccessor {
        block: CfgBlockId,
        duplicate: CfgBlockId,
    },
    /// Same predecessor appears multiple times
    DuplicatePredecessor {
        block: CfgBlockId,
        duplicate: CfgBlockId,
    },
    /// Critical edge (multi-successor to multi-predecessor) - informational
    CriticalEdge {
        from: CfgBlockId,
        to: CfgBlockId,
    },
    /// Empty block (no instructions)
    EmptyBlock {
        block: CfgBlockId,
    },

    // P3: Advanced
    /// CFG contains irreducible control flow
    IrreducibleLoop {
        entry_blocks: Vec<CfgBlockId>,
    },
}

impl<L: Clone + Debug> std::fmt::Display for CfgViolation<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CfgViolation::EntryBlockHasPredecessors { entry, predecessor_count } => {
                write!(f, "Entry block {:?} has {} predecessors (should have none)",
                    entry, predecessor_count)
            }
            CfgViolation::SuccessorPredecessorMismatch { block, claims_successor } => {
                write!(f, "Block {:?} claims {:?} as successor, but {:?} doesn't list {:?} as predecessor",
                    block, claims_successor, claims_successor, block)
            }
            CfgViolation::PredecessorSuccessorMismatch { block, claims_predecessor } => {
                write!(f, "Block {:?} claims {:?} as predecessor, but {:?} doesn't list {:?} as successor",
                    block, claims_predecessor, claims_predecessor, block)
            }
            CfgViolation::UnreachableBlock { block } => {
                write!(f, "Block {:?} is not reachable from entry", block)
            }
            CfgViolation::NoExitBlock => {
                write!(f, "CFG has no exit block (potential infinite loop)")
            }
            CfgViolation::MissingTerminator { block } => {
                write!(f, "Block {:?} has no successors but doesn't end with a terminator", block)
            }
            CfgViolation::TerminatorWithSuccessors { block, successor_count } => {
                write!(f, "Block {:?} ends with terminator but has {} successors",
                    block, successor_count)
            }
            CfgViolation::DanglingJumpTarget { block, target_label } => {
                write!(f, "Block {:?} jumps to non-existent label {:?}", block, target_label)
            }
            CfgViolation::BlockIndexMismatch { block_id, actual_index } => {
                write!(f, "Block {:?} is at index {} in blocks array", block_id, actual_index)
            }
            CfgViolation::DuplicateSuccessor { block, duplicate } => {
                write!(f, "Block {:?} has duplicate successor {:?}", block, duplicate)
            }
            CfgViolation::DuplicatePredecessor { block, duplicate } => {
                write!(f, "Block {:?} has duplicate predecessor {:?}", block, duplicate)
            }
            CfgViolation::CriticalEdge { from, to } => {
                write!(f, "Critical edge from {:?} to {:?}", from, to)
            }
            CfgViolation::EmptyBlock { block } => {
                write!(f, "Block {:?} has no instructions", block)
            }
            CfgViolation::IrreducibleLoop { entry_blocks } => {
                write!(f, "Irreducible loop with multiple entry blocks: {:?}", entry_blocks)
            }
        }
    }
}

/// Validate all CFG properties
pub fn validate_cfg<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    // P0: Core invariants
    violations.extend(check_entry_block(cfg));
    violations.extend(check_edge_symmetry(cfg));
    violations.extend(check_reachability(cfg));

    // P1: Termination & edges
    violations.extend(check_termination(cfg));
    violations.extend(check_edge_validity(cfg));

    // P2: Consistency
    violations.extend(check_block_indices(cfg));
    violations.extend(check_duplicates(cfg));
    violations.extend(detect_critical_edges(cfg));
    violations.extend(check_empty_blocks(cfg));

    // P3: Advanced
    violations.extend(check_reducibility(cfg));

    violations
}

/// Check that entry block has no predecessors
fn check_entry_block<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    if let Some(entry_block) = cfg.get_block(cfg.entry) {
        if !entry_block.predecessors.is_empty() {
            violations.push(CfgViolation::EntryBlockHasPredecessors {
                entry: cfg.entry,
                predecessor_count: entry_block.predecessors.len(),
            });
        }
    }

    violations
}

/// Check that successor/predecessor relationships are symmetric
fn check_edge_symmetry<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    for block in cfg.iter_blocks() {
        // Check that each successor lists this block as predecessor
        for &succ in &block.successors {
            if let Some(succ_block) = cfg.get_block(succ) {
                if !succ_block.predecessors.contains(&block.id) {
                    violations.push(CfgViolation::SuccessorPredecessorMismatch {
                        block: block.id,
                        claims_successor: succ,
                    });
                }
            }
        }

        // Check that each predecessor lists this block as successor
        for &pred in &block.predecessors {
            if let Some(pred_block) = cfg.get_block(pred) {
                if !pred_block.successors.contains(&block.id) {
                    violations.push(CfgViolation::PredecessorSuccessorMismatch {
                        block: block.id,
                        claims_predecessor: pred,
                    });
                }
            }
        }
    }

    violations
}

/// Compute set of reachable blocks from entry via DFS
pub fn compute_reachable<I: CfgInstruction>(cfg: &Cfg<I>) -> HashSet<CfgBlockId> {
    let mut reachable = HashSet::new();
    let mut stack = vec![cfg.entry];

    while let Some(block_id) = stack.pop() {
        if reachable.contains(&block_id) {
            continue;
        }
        reachable.insert(block_id);

        if let Some(block) = cfg.get_block(block_id) {
            for &succ in &block.successors {
                if !reachable.contains(&succ) {
                    stack.push(succ);
                }
            }
        }
    }

    reachable
}

/// Check that all blocks are reachable from entry
fn check_reachability<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();
    let reachable = compute_reachable(cfg);

    for block in cfg.iter_blocks() {
        if !reachable.contains(&block.id) {
            violations.push(CfgViolation::UnreachableBlock { block: block.id });
        }
    }

    violations
}

/// Check termination properties
fn check_termination<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();
    let mut has_exit = false;

    for block in cfg.iter_blocks() {
        let is_exit = block.successors.is_empty();

        if is_exit {
            has_exit = true;

            // Check if last instruction is actually a terminator
            if let Some(last_instr) = block.instructions.last() {
                match last_instr.control_flow() {
                    ControlFlow::Terminate => {
                        // Good - proper terminator
                    }
                    _ => {
                        // Block has no successors but doesn't end with Terminate
                        violations.push(CfgViolation::MissingTerminator { block: block.id });
                    }
                }
            } else {
                // Empty block with no successors
                violations.push(CfgViolation::MissingTerminator { block: block.id });
            }
        } else {
            // Block has successors - check it doesn't end with Terminate
            if let Some(last_instr) = block.instructions.last() {
                if matches!(last_instr.control_flow(), ControlFlow::Terminate) {
                    violations.push(CfgViolation::TerminatorWithSuccessors {
                        block: block.id,
                        successor_count: block.successors.len(),
                    });
                }
            }
        }
    }

    if !has_exit && !cfg.is_empty() {
        violations.push(CfgViolation::NoExitBlock);
    }

    violations
}

/// Check that all jump targets are valid
fn check_edge_validity<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    for block in cfg.iter_blocks() {
        for instr in &block.instructions {
            let targets: Vec<I::Label> = match instr.control_flow() {
                ControlFlow::Jump(target) => vec![target],
                ControlFlow::ConditionalJump(target) => vec![target],
                ControlFlow::Branch(targets) => targets,
                _ => vec![],
            };

            for target in targets {
                if !cfg.label_to_block.contains_key(&target) {
                    violations.push(CfgViolation::DanglingJumpTarget {
                        block: block.id,
                        target_label: target,
                    });
                }
            }
        }
    }

    violations
}

/// Check that block IDs match their array indices
fn check_block_indices<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    for (index, block) in cfg.blocks.iter().enumerate() {
        if block.id.0 != index {
            violations.push(CfgViolation::BlockIndexMismatch {
                block_id: block.id,
                actual_index: index,
            });
        }
    }

    violations
}

/// Check for duplicate successors and predecessors
fn check_duplicates<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    for block in cfg.iter_blocks() {
        // Check duplicate successors
        let mut seen_succs = HashSet::new();
        for &succ in &block.successors {
            if !seen_succs.insert(succ) {
                violations.push(CfgViolation::DuplicateSuccessor {
                    block: block.id,
                    duplicate: succ,
                });
            }
        }

        // Check duplicate predecessors
        let mut seen_preds = HashSet::new();
        for &pred in &block.predecessors {
            if !seen_preds.insert(pred) {
                violations.push(CfgViolation::DuplicatePredecessor {
                    block: block.id,
                    duplicate: pred,
                });
            }
        }
    }

    violations
}

/// Detect critical edges (from multi-successor to multi-predecessor blocks)
fn detect_critical_edges<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    for block in cfg.iter_blocks() {
        if block.successors.len() > 1 {
            for &succ in &block.successors {
                if let Some(succ_block) = cfg.get_block(succ) {
                    if succ_block.predecessors.len() > 1 {
                        violations.push(CfgViolation::CriticalEdge {
                            from: block.id,
                            to: succ,
                        });
                    }
                }
            }
        }
    }

    violations
}

/// Check for empty blocks
fn check_empty_blocks<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    for block in cfg.iter_blocks() {
        if block.instructions.is_empty() {
            violations.push(CfgViolation::EmptyBlock { block: block.id });
        }
    }

    violations
}

/// Check for irreducible control flow
///
/// Uses a simplified approach: a CFG is reducible if every cycle has a single entry point.
/// We detect irreducible loops by finding strongly connected components (SCCs) with
/// multiple entry points.
fn check_reducibility<I: CfgInstruction>(cfg: &Cfg<I>) -> Vec<CfgViolation<I::Label>> {
    let mut violations = Vec::new();

    if cfg.is_empty() {
        return violations;
    }

    // Find all back edges (edges where target dominates source)
    let dominators = compute_dominators(cfg);
    let mut back_edges = Vec::new();

    for block in cfg.iter_blocks() {
        for &succ in &block.successors {
            if let Some(doms) = dominators.get(&block.id) {
                if doms.contains(&succ) {
                    back_edges.push((block.id, succ));
                }
            }
        }
    }

    // For each back edge target (loop header), check if there are multiple entries
    for &(_, header) in &back_edges {
        let loop_blocks = find_natural_loop(cfg, header, &back_edges);
        let mut entries = Vec::new();

        for &loop_block in &loop_blocks {
            if let Some(block) = cfg.get_block(loop_block) {
                for &pred in &block.predecessors {
                    if !loop_blocks.contains(&pred) && loop_block != header {
                        // Found an entry point that's not the header
                        entries.push(loop_block);
                        break;
                    }
                }
            }
        }

        // Header is always an entry
        entries.push(header);

        if entries.len() > 1 {
            violations.push(CfgViolation::IrreducibleLoop {
                entry_blocks: entries,
            });
        }
    }

    violations
}

/// Compute dominators using iterative dataflow
fn compute_dominators<I: CfgInstruction>(cfg: &Cfg<I>) -> std::collections::HashMap<CfgBlockId, HashSet<CfgBlockId>> {
    use std::collections::HashMap;

    let mut dominators: HashMap<CfgBlockId, HashSet<CfgBlockId>> = HashMap::new();
    let all_blocks: HashSet<CfgBlockId> = cfg.blocks.iter().map(|b| b.id).collect();

    // Initialize: entry dominated only by itself, others by all
    for block in cfg.iter_blocks() {
        if block.id == cfg.entry {
            let mut entry_doms = HashSet::new();
            entry_doms.insert(cfg.entry);
            dominators.insert(cfg.entry, entry_doms);
        } else {
            dominators.insert(block.id, all_blocks.clone());
        }
    }

    // Iterate until fixpoint
    let mut changed = true;
    while changed {
        changed = false;

        for block in cfg.iter_blocks() {
            if block.id == cfg.entry {
                continue;
            }

            let mut new_doms: Option<HashSet<CfgBlockId>> = None;

            for &pred in &block.predecessors {
                let pred_doms = dominators.get(&pred).cloned().unwrap_or_default();
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

/// Find natural loop given a header and back edges
fn find_natural_loop<I: CfgInstruction>(
    cfg: &Cfg<I>,
    header: CfgBlockId,
    back_edges: &[(CfgBlockId, CfgBlockId)],
) -> HashSet<CfgBlockId> {
    let mut loop_blocks = HashSet::new();
    loop_blocks.insert(header);

    // Find all back edge sources for this header
    let sources: Vec<CfgBlockId> = back_edges
        .iter()
        .filter(|(_, h)| *h == header)
        .map(|(s, _)| *s)
        .collect();

    // BFS backwards from sources to find all loop blocks
    let mut worklist: Vec<CfgBlockId> = sources.clone();
    for source in sources {
        loop_blocks.insert(source);
    }

    while let Some(block_id) = worklist.pop() {
        if let Some(block) = cfg.get_block(block_id) {
            for &pred in &block.predecessors {
                if !loop_blocks.contains(&pred) {
                    loop_blocks.insert(pred);
                    worklist.push(pred);
                }
            }
        }
    }

    loop_blocks
}

/// Assert no violations occurred
pub fn assert_valid_cfg<I: CfgInstruction>(cfg: &Cfg<I>) {
    let violations = validate_cfg(cfg);
    let errors: Vec<_> = violations.iter()
        .filter(|v| !matches!(v, CfgViolation::CriticalEdge { .. }))
        .collect();

    if !errors.is_empty() {
        let mut msg = String::from("CFG validation failed:\n");
        for v in &errors {
            msg.push_str(&format!("  - {}\n", v));
        }
        panic!("{}", msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::CfgBuilder;

    // Test instruction for testing
    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    enum TestInstr {
        Lbl(String),
        Nop,
        Jump(String),
        JumpIf(String),
        Return,
    }

    impl CfgInstruction for TestInstr {
        type Label = String;

        fn as_label(&self) -> Option<&String> {
            match self {
                TestInstr::Lbl(l) => Some(l),
                _ => None,
            }
        }

        fn control_flow(&self) -> ControlFlow<String> {
            match self {
                TestInstr::Jump(target) => ControlFlow::Jump(target.clone()),
                TestInstr::JumpIf(target) => ControlFlow::ConditionalJump(target.clone()),
                TestInstr::Return => ControlFlow::Terminate,
                _ => ControlFlow::FallThrough,
            }
        }
    }

    #[test]
    fn test_valid_cfg() {
        let instrs = vec![
            TestInstr::Nop,
            TestInstr::Return,
        ];
        let cfg = CfgBuilder::build(instrs);
        let violations = validate_cfg(&cfg);

        // Should have no errors (maybe critical edge info, but that's ok)
        let errors: Vec<_> = violations.iter()
            .filter(|v| !matches!(v, CfgViolation::CriticalEdge { .. }))
            .collect();
        assert!(errors.is_empty(), "Unexpected violations: {:?}", errors);
    }

    #[test]
    fn test_unreachable_block() {
        // Build a CFG with an unreachable block manually
        let instrs = vec![
            TestInstr::Nop,
            TestInstr::Return,
            TestInstr::Lbl("unreachable".into()),
            TestInstr::Return,
        ];
        let cfg = CfgBuilder::build(instrs);
        let violations = validate_cfg(&cfg);

        let unreachable: Vec<_> = violations.iter()
            .filter(|v| matches!(v, CfgViolation::UnreachableBlock { .. }))
            .collect();
        assert!(!unreachable.is_empty(), "Should detect unreachable block");
    }

    #[test]
    fn test_simple_loop() {
        let instrs = vec![
            TestInstr::Lbl("loop".into()),
            TestInstr::Nop,
            TestInstr::JumpIf("loop".into()),
            TestInstr::Return,
        ];
        let cfg = CfgBuilder::build(instrs);
        let violations = validate_cfg(&cfg);

        // Simple loops are reducible
        let irreducible: Vec<_> = violations.iter()
            .filter(|v| matches!(v, CfgViolation::IrreducibleLoop { .. }))
            .collect();
        assert!(irreducible.is_empty(), "Simple loop should be reducible");
    }

    #[test]
    fn test_critical_edge() {
        // Create a CFG with a critical edge:
        // B0 has conditional jump - 2 successors (to B1 and B2)
        // B1 and B2 both go to B3 (merge point) - B3 has 2 predecessors
        // But B0 doesn't go directly to B3, so no critical edge in simple if-else.
        //
        // To create a critical edge, we need a conditional that jumps directly
        // to a merge point that also has another predecessor:
        // B0 -> B1 (fallthrough) and B0 -> B2 (jump)
        // B1 -> B2 (fallthrough) - so B2 has 2 predecessors (B0, B1) and B0 has 2 successors
        let instrs = vec![
            TestInstr::Nop,
            TestInstr::JumpIf("merge".into()),  // B0: jump to B2 or fall to B1
            TestInstr::Nop,                      // B1: falls through to B2
            TestInstr::Lbl("merge".into()),      // B2: merge point (from B0 and B1)
            TestInstr::Return,
        ];
        let cfg = CfgBuilder::build(instrs);
        let violations = validate_cfg(&cfg);

        let critical: Vec<_> = violations.iter()
            .filter(|v| matches!(v, CfgViolation::CriticalEdge { .. }))
            .collect();
        // B0 has 2 successors, B2 has 2 predecessors, so B0->B2 is a critical edge
        assert!(!critical.is_empty(), "Should detect critical edge: {:?}", cfg);
    }
}
