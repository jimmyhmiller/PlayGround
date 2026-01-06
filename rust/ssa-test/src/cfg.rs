//! CFG (Control Flow Graph) construction from linear IR.
//!
//! This module provides generic CFG construction for any linear instruction stream
//! that uses labels and jumps for control flow.
//!
//! # Usage
//!
//! 1. Implement [`CfgInstruction`] for your instruction type
//! 2. Call [`CfgBuilder::build`] with your instruction stream
//! 3. Use the resulting [`Cfg`] for SSA translation or analysis
//!
//! # Example
//!
//! ```ignore
//! use ssa_lib::cfg::{CfgInstruction, ControlFlow, CfgBuilder};
//!
//! impl CfgInstruction for MyInstruction {
//!     type Label = String;
//!
//!     fn as_label(&self) -> Option<&Self::Label> {
//!         match self {
//!             MyInstruction::Label(l) => Some(l),
//!             _ => None,
//!         }
//!     }
//!
//!     fn control_flow(&self) -> ControlFlow<Self::Label> {
//!         match self {
//!             MyInstruction::Jump(target) => ControlFlow::Jump(target.clone()),
//!             MyInstruction::JumpIf { target, .. } => ControlFlow::ConditionalJump(target.clone()),
//!             MyInstruction::Return(_) => ControlFlow::Terminate,
//!             _ => ControlFlow::FallThrough,
//!         }
//!     }
//! }
//!
//! let cfg = CfgBuilder::build(instructions);
//! ```

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Control flow effect of an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlFlow<L> {
    /// Normal instruction - execution falls through to next instruction
    FallThrough,
    /// Unconditional jump to target label
    Jump(L),
    /// Conditional jump - goes to target label OR falls through
    ConditionalJump(L),
    /// Multiple possible targets (e.g., switch/match statement)
    Branch(Vec<L>),
    /// Terminates execution (return, unreachable, etc.)
    Terminate,
}

/// Trait for instructions that can be analyzed for control flow.
///
/// Implement this trait on your instruction type to enable CFG construction.
pub trait CfgInstruction: Clone + Debug {
    /// The label type used for jump targets.
    type Label: Clone + Eq + Hash + Debug;

    /// Returns the label if this instruction defines one.
    ///
    /// Label instructions mark the start of potential basic blocks.
    fn as_label(&self) -> Option<&Self::Label>;

    /// Returns the control flow effect of this instruction.
    fn control_flow(&self) -> ControlFlow<Self::Label>;
}

/// Unique identifier for a CFG basic block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CfgBlockId(pub usize);

impl std::fmt::Display for CfgBlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "B{}", self.0)
    }
}

/// A basic block in the CFG.
#[derive(Debug, Clone)]
pub struct CfgBlock<I> {
    /// Unique identifier for this block
    pub id: CfgBlockId,
    /// Instructions in this block (may include the leading label)
    pub instructions: Vec<I>,
    /// Predecessor blocks (incoming edges)
    pub predecessors: Vec<CfgBlockId>,
    /// Successor blocks (outgoing edges)
    pub successors: Vec<CfgBlockId>,
}

impl<I> CfgBlock<I> {
    /// Create a new empty basic block with the given ID.
    pub fn new(id: CfgBlockId) -> Self {
        CfgBlock {
            id,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }
}

/// A control flow graph.
#[derive(Debug, Clone)]
pub struct Cfg<I: CfgInstruction> {
    /// All basic blocks in the CFG
    pub blocks: Vec<CfgBlock<I>>,
    /// Entry block (always block 0)
    pub entry: CfgBlockId,
    /// Maps labels to the blocks that start with them
    pub label_to_block: HashMap<I::Label, CfgBlockId>,
}

impl<I: CfgInstruction> Cfg<I> {
    /// Get a block by ID
    pub fn get_block(&self, id: CfgBlockId) -> Option<&CfgBlock<I>> {
        self.blocks.get(id.0)
    }

    /// Get a mutable reference to a block by ID
    pub fn get_block_mut(&mut self, id: CfgBlockId) -> Option<&mut CfgBlock<I>> {
        self.blocks.get_mut(id.0)
    }

    /// Iterate over all blocks
    pub fn iter_blocks(&self) -> impl Iterator<Item = &CfgBlock<I>> {
        self.blocks.iter()
    }

    /// Number of blocks
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if CFG is empty
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Builder for constructing CFGs from linear instruction streams.
///
/// Uses the leader-based algorithm:
/// 1. Identify leaders (first instructions of basic blocks)
/// 2. Partition instructions into blocks
/// 3. Build edges based on control flow
pub struct CfgBuilder<I: CfgInstruction> {
    instructions: Vec<I>,
    /// Maps label -> instruction index
    label_to_index: HashMap<I::Label, usize>,
    /// Maps instruction index -> block ID (for leaders only)
    leader_to_block: HashMap<usize, CfgBlockId>,
    /// Set of leader indices
    leaders: Vec<usize>,
}

impl<I: CfgInstruction> CfgBuilder<I> {
    /// Build a CFG from a linear instruction stream.
    ///
    /// # Algorithm
    ///
    /// 1. **Identify leaders**: An instruction is a leader if:
    ///    - It's the first instruction
    ///    - It's a label (target of a jump)
    ///    - It immediately follows a jump or conditional jump
    ///
    /// 2. **Form basic blocks**: Each block starts at a leader and extends
    ///    until the next leader (exclusive).
    ///
    /// 3. **Build edges**: For each block ending with:
    ///    - `Jump(L)` → edge to block starting with L
    ///    - `ConditionalJump(L)` → edges to L and fall-through
    ///    - `Branch(targets)` → edges to all targets
    ///    - `FallThrough` → edge to next block
    ///    - `Terminate` → no outgoing edges
    pub fn build(instructions: Vec<I>) -> Cfg<I> {
        if instructions.is_empty() {
            return Cfg {
                blocks: vec![CfgBlock::new(CfgBlockId(0))],
                entry: CfgBlockId(0),
                label_to_block: HashMap::new(),
            };
        }

        let mut builder = CfgBuilder {
            instructions,
            label_to_index: HashMap::new(),
            leader_to_block: HashMap::new(),
            leaders: Vec::new(),
        };

        builder.build_internal()
    }

    fn build_internal(&mut self) -> Cfg<I> {
        // Pass 1: Build label -> index mapping and identify leaders
        self.identify_leaders();

        // Pass 2: Create blocks and partition instructions
        let mut cfg = self.create_blocks();

        // Pass 3: Build edges
        self.build_edges(&mut cfg);

        cfg
    }

    fn identify_leaders(&mut self) {
        // First instruction is always a leader
        self.leaders.push(0);

        for (i, instr) in self.instructions.iter().enumerate() {
            // If this is a label, record its position and mark as leader
            if let Some(label) = instr.as_label() {
                self.label_to_index.insert(label.clone(), i);
                if i > 0 && !self.leaders.contains(&i) {
                    self.leaders.push(i);
                }
            }

            // Instruction after a control flow change is a leader
            match instr.control_flow() {
                ControlFlow::Jump(_)
                | ControlFlow::ConditionalJump(_)
                | ControlFlow::Branch(_)
                | ControlFlow::Terminate => {
                    let next = i + 1;
                    if next < self.instructions.len() && !self.leaders.contains(&next) {
                        self.leaders.push(next);
                    }
                }
                ControlFlow::FallThrough => {}
            }
        }

        // Sort leaders for efficient block creation
        self.leaders.sort();

        // Map leaders to block IDs
        for (block_idx, &leader_idx) in self.leaders.iter().enumerate() {
            self.leader_to_block
                .insert(leader_idx, CfgBlockId(block_idx));
        }
    }

    fn create_blocks(&self) -> Cfg<I> {
        let mut blocks = Vec::with_capacity(self.leaders.len());

        for (block_idx, &start) in self.leaders.iter().enumerate() {
            let end = self
                .leaders
                .get(block_idx + 1)
                .copied()
                .unwrap_or(self.instructions.len());

            let mut block = CfgBlock::new(CfgBlockId(block_idx));
            block.instructions = self.instructions[start..end].to_vec();
            blocks.push(block);
        }

        // Build label -> block mapping
        let mut label_to_block = HashMap::new();
        for (label, &instr_idx) in &self.label_to_index {
            if let Some(&block_id) = self.leader_to_block.get(&instr_idx) {
                label_to_block.insert(label.clone(), block_id);
            }
        }

        Cfg {
            blocks,
            entry: CfgBlockId(0),
            label_to_block,
        }
    }

    fn build_edges(&self, cfg: &mut Cfg<I>) {
        for (block_idx, &start) in self.leaders.iter().enumerate() {
            let block_id = CfgBlockId(block_idx);
            let end = self
                .leaders
                .get(block_idx + 1)
                .copied()
                .unwrap_or(self.instructions.len());

            // Get the last instruction in this block
            if start >= end {
                continue;
            }
            let last_instr = &self.instructions[end - 1];

            // Determine successors based on control flow
            let mut successors = Vec::new();

            match last_instr.control_flow() {
                ControlFlow::FallThrough => {
                    // Edge to next block if exists
                    if block_idx + 1 < self.leaders.len() {
                        successors.push(CfgBlockId(block_idx + 1));
                    }
                }
                ControlFlow::Jump(target) => {
                    // Edge to target block only
                    if let Some(&target_idx) = self.label_to_index.get(&target) {
                        if let Some(&target_block) = self.leader_to_block.get(&target_idx) {
                            successors.push(target_block);
                        }
                    }
                }
                ControlFlow::ConditionalJump(target) => {
                    // Edge to target AND fall-through
                    if let Some(&target_idx) = self.label_to_index.get(&target) {
                        if let Some(&target_block) = self.leader_to_block.get(&target_idx) {
                            successors.push(target_block);
                        }
                    }
                    // Fall-through edge
                    if block_idx + 1 < self.leaders.len() {
                        successors.push(CfgBlockId(block_idx + 1));
                    }
                }
                ControlFlow::Branch(targets) => {
                    // Edges to all targets
                    for target in targets {
                        if let Some(&target_idx) = self.label_to_index.get(&target) {
                            if let Some(&target_block) = self.leader_to_block.get(&target_idx) {
                                if !successors.contains(&target_block) {
                                    successors.push(target_block);
                                }
                            }
                        }
                    }
                }
                ControlFlow::Terminate => {
                    // No successors
                }
            }

            // Set successors for this block
            cfg.blocks[block_idx].successors = successors.clone();

            // Add this block as predecessor to all successors
            for succ in successors {
                cfg.blocks[succ.0].predecessors.push(block_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test instruction for testing CFG construction
    #[derive(Clone, Debug)]
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
    fn test_empty_instructions() {
        let cfg: Cfg<TestInstr> = CfgBuilder::build(vec![]);
        assert_eq!(cfg.blocks.len(), 1);
        assert!(cfg.blocks[0].instructions.is_empty());
    }

    #[test]
    fn test_single_block() {
        let instrs = vec![TestInstr::Nop, TestInstr::Nop, TestInstr::Return];

        let cfg = CfgBuilder::build(instrs);
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.blocks[0].instructions.len(), 3);
        assert!(cfg.blocks[0].successors.is_empty());
    }

    #[test]
    fn test_simple_branch() {
        // if-else structure
        let instrs = vec![
            TestInstr::Nop,
            TestInstr::JumpIf("else".into()),
            TestInstr::Nop,
            TestInstr::Jump("end".into()),
            TestInstr::Lbl("else".into()),
            TestInstr::Nop,
            TestInstr::Lbl("end".into()),
            TestInstr::Return,
        ];

        let cfg = CfgBuilder::build(instrs);

        // Should have 4 blocks:
        // B0: nop, jumpif else
        // B1: nop, jump end
        // B2: label else, nop
        // B3: label end, return
        assert_eq!(cfg.blocks.len(), 4);

        // B0 successors: B2 (else), B1 (fall-through)
        assert!(cfg.blocks[0].successors.contains(&CfgBlockId(2)));
        assert!(cfg.blocks[0].successors.contains(&CfgBlockId(1)));

        // B1 successor: B3 (end)
        assert_eq!(cfg.blocks[1].successors, vec![CfgBlockId(3)]);

        // B2 successor: B3 (fall-through)
        assert_eq!(cfg.blocks[2].successors, vec![CfgBlockId(3)]);

        // B3: no successors (return)
        assert!(cfg.blocks[3].successors.is_empty());

        // Check predecessors of B3 (end): B1 and B2
        assert!(cfg.blocks[3].predecessors.contains(&CfgBlockId(1)));
        assert!(cfg.blocks[3].predecessors.contains(&CfgBlockId(2)));
    }

    #[test]
    fn test_loop() {
        // while loop structure
        let instrs = vec![
            TestInstr::Nop,
            TestInstr::Lbl("loop".into()),
            TestInstr::Nop,
            TestInstr::JumpIf("loop".into()),
            TestInstr::Return,
        ];

        let cfg = CfgBuilder::build(instrs);

        // Should have 3 blocks:
        // B0: nop
        // B1: label loop, nop, jumpif loop
        // B2: return
        assert_eq!(cfg.blocks.len(), 3);

        // B0 successor: B1 (fall-through)
        assert_eq!(cfg.blocks[0].successors, vec![CfgBlockId(1)]);

        // B1 successors: B1 (loop back), B2 (fall-through)
        assert!(cfg.blocks[1].successors.contains(&CfgBlockId(1)));
        assert!(cfg.blocks[1].successors.contains(&CfgBlockId(2)));

        // B1 predecessors: B0 and B1 (self-loop)
        assert!(cfg.blocks[1].predecessors.contains(&CfgBlockId(0)));
        assert!(cfg.blocks[1].predecessors.contains(&CfgBlockId(1)));
    }
}
