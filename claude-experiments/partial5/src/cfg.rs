//! Control Flow Graph representation
//!
//! Represents the CFG extracted from state machines.

use std::collections::HashMap;
use swc_ecma_ast::*;

/// A control flow graph
#[derive(Debug)]
pub struct CFG {
    /// Entry block ID
    pub entry: BlockId,
    /// All blocks in the CFG
    pub blocks: HashMap<BlockId, BasicBlock>,
    /// Exit block ID (if any)
    pub exit: Option<BlockId>,
}

/// Block identifier
pub type BlockId = u32;

/// A basic block in the CFG
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Unique ID
    pub id: BlockId,
    /// Statements in this block
    pub stmts: Vec<Stmt>,
    /// How this block terminates
    pub terminator: Terminator,
}

/// How a basic block terminates
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional jump to another block
    Goto(BlockId),
    /// Conditional branch
    Branch {
        condition: Box<Expr>,
        if_true: BlockId,
        if_false: BlockId,
    },
    /// Return from the function/state machine
    Return(Option<Box<Expr>>),
    /// Exit the state machine (state = -1)
    Exit,
    /// Throw an exception (may be a computed goto!)
    Throw(Box<Expr>),
    /// Switch with multiple targets
    Switch {
        discriminant: Box<Expr>,
        cases: Vec<(i64, BlockId)>,
        default: Option<BlockId>,
    },
    /// Call a function and continue to next block
    /// Used for tail calls like `return v5()`
    TailCall {
        callee: Box<Expr>,
        args: Vec<ExprOrSpread>,
    },
}

impl CFG {
    /// Create a new empty CFG
    pub fn new(entry: BlockId) -> Self {
        CFG {
            entry,
            blocks: HashMap::new(),
            exit: None,
        }
    }

    /// Add a block to the CFG
    pub fn add_block(&mut self, block: BasicBlock) {
        self.blocks.insert(block.id, block);
    }

    /// Get a block by ID
    pub fn get_block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.get(&id)
    }

    /// Get successors of a block
    pub fn successors(&self, id: BlockId) -> Vec<BlockId> {
        match self.blocks.get(&id) {
            Some(block) => match &block.terminator {
                Terminator::Goto(target) => vec![*target],
                Terminator::Branch {
                    if_true, if_false, ..
                } => vec![*if_true, *if_false],
                Terminator::Switch { cases, default, .. } => {
                    let mut targets: Vec<BlockId> = cases.iter().map(|(_, b)| *b).collect();
                    if let Some(d) = default {
                        targets.push(*d);
                    }
                    targets
                }
                Terminator::Return(_) | Terminator::Exit | Terminator::TailCall { .. } => vec![],
                Terminator::Throw(_) => vec![], // Could have catch handler
            },
            None => vec![],
        }
    }
}

/// Convert a state machine to a CFG
pub fn state_machine_to_cfg(
    sm: &crate::statemachine::StateMachine,
) -> CFG {
    use crate::statemachine::StateTransition;

    let mut cfg = CFG::new(0);
    let mut next_block_id = 0u32;

    // Create a block for each case
    let mut state_to_block: HashMap<i64, BlockId> = HashMap::new();

    for (state, case) in &sm.cases {
        let block_id = next_block_id;
        next_block_id += 1;
        state_to_block.insert(*state, block_id);
    }

    // Create exit block
    let exit_block_id = next_block_id;
    next_block_id += 1;
    cfg.exit = Some(exit_block_id);
    cfg.add_block(BasicBlock {
        id: exit_block_id,
        stmts: vec![],
        terminator: Terminator::Exit,
    });

    // Create default block if needed
    let default_block_id = if sm.default.is_some() {
        let id = next_block_id;
        next_block_id += 1;
        Some(id)
    } else {
        None
    };

    // Populate blocks
    for (state, case) in &sm.cases {
        let block_id = state_to_block[state];

        let terminator = match &case.next_state {
            Some(StateTransition::Goto(target)) => {
                if let Some(&target_block) = state_to_block.get(target) {
                    Terminator::Goto(target_block)
                } else {
                    // Target not found - might be computed
                    Terminator::Exit
                }
            }
            Some(StateTransition::Exit) => Terminator::Goto(exit_block_id),
            Some(StateTransition::Branch {
                condition,
                if_true,
                if_false,
            }) => {
                let true_block = state_to_block.get(if_true).copied().unwrap_or(exit_block_id);
                let false_block = state_to_block.get(if_false).copied().unwrap_or(exit_block_id);
                Terminator::Branch {
                    condition: condition.clone(),
                    if_true: true_block,
                    if_false: false_block,
                }
            }
            Some(StateTransition::Dynamic(expr)) => {
                // Dynamic transition - we need to keep the switch
                // For now, model as a switch over all known targets
                let cases: Vec<(i64, BlockId)> = state_to_block
                    .iter()
                    .map(|(s, b)| (*s, *b))
                    .collect();
                Terminator::Switch {
                    discriminant: expr.clone(),
                    cases,
                    default: Some(exit_block_id),
                }
            }
            None => {
                // No explicit transition - fall through to default or exit
                if let Some(def) = default_block_id {
                    Terminator::Goto(def)
                } else {
                    Terminator::Exit
                }
            }
        };

        cfg.add_block(BasicBlock {
            id: block_id,
            stmts: case.stmts.clone(),
            terminator,
        });
    }

    // Add default block if present
    if let (Some(default_stmts), Some(block_id)) = (&sm.default, default_block_id) {
        cfg.add_block(BasicBlock {
            id: block_id,
            stmts: default_stmts.clone(),
            terminator: Terminator::Exit,
        });
    }

    // Set entry based on initial state or state 0
    let initial = sm.initial_state.unwrap_or(0);
    if let Some(&entry_block) = state_to_block.get(&initial) {
        cfg.entry = entry_block;
    } else if let Some(&block_0) = state_to_block.get(&0) {
        cfg.entry = block_0;
    }

    cfg
}

/// Pretty print a CFG for debugging
pub fn print_cfg(cfg: &CFG) -> String {
    let mut out = String::new();

    out.push_str(&format!("CFG (entry: {})\n", cfg.entry));
    out.push_str(&format!("Blocks: {:?}\n", cfg.blocks.keys().collect::<Vec<_>>()));

    for (id, block) in &cfg.blocks {
        out.push_str(&format!("\nBlock {}:\n", id));
        out.push_str(&format!("  stmts: {} statements\n", block.stmts.len()));
        out.push_str(&format!("  terminator: {:?}\n", block.terminator));
    }

    out
}
