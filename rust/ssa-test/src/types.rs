//! Core types for the SSA library.
//!
//! These types are library-provided and used by generic SSA construction.

use std::fmt::Debug;
use std::hash::Hash;

/// SSA variable identifier (v0, v1, v2, ...)
///
/// This is a library-provided type for SSA-renamed variables.
/// Users' original variable names are tracked separately.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SsaVariable(pub String);

impl SsaVariable {
    pub fn new(name: &str) -> Self {
        SsaVariable(name.to_string())
    }

    pub fn temp(id: usize) -> Self {
        SsaVariable(format!("v{}", id))
    }

    pub fn name(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SsaVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a phi node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhiId(pub usize);

impl std::fmt::Display for PhiId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Phi{}", self.0)
    }
}

/// Unique identifier for a basic block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block{}", self.0)
    }
}

/// Reference to where a phi node is used
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhiReference {
    /// Used in an instruction at a specific location
    Instruction {
        block_id: BlockId,
        instruction_offset: usize,
    },
    /// Used as an operand in another phi
    Phi(PhiId),
}

/// A phi node - generic over the value type
///
/// In proper SSA, a phi directly defines a variable:
///   v7 = φ(v2, v5)
/// The `dest` field holds this destination variable once materialized.
#[derive(Debug, Clone, PartialEq)]
pub struct Phi<V> {
    pub id: PhiId,
    pub block_id: BlockId,
    pub operands: Vec<V>,
    pub uses: Vec<PhiReference>,
    /// The variable this phi defines (set during materialization)
    pub dest: Option<SsaVariable>,
}

impl<V> Phi<V> {
    pub fn new(id: PhiId, block_id: BlockId) -> Self {
        Phi {
            id,
            block_id,
            operands: Vec::new(),
            uses: Vec::new(),
            dest: None,
        }
    }
}

/// A basic block - generic over the instruction type
#[derive(Debug, Clone, PartialEq)]
pub struct Block<I> {
    pub id: BlockId,
    pub instructions: Vec<I>,
    pub predecessors: Vec<BlockId>,
    pub sealed: bool,
}

impl<I> Block<I> {
    pub fn new(id: BlockId) -> Self {
        Block {
            id,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            sealed: false,
        }
    }

    pub fn add_instruction(&mut self, instr: I) {
        self.instructions.push(instr);
    }

    pub fn add_predecessor(&mut self, predecessor: BlockId) {
        assert!(
            !self.sealed,
            "Cannot add predecessor to sealed block {:?}",
            self.id
        );
        self.predecessors.push(predecessor);
    }

    pub fn seal(&mut self) {
        self.sealed = true;
    }
}

/// A function containing multiple blocks
#[derive(Debug, Clone, PartialEq)]
pub struct Function<I> {
    pub blocks: Vec<Block<I>>,
    pub entry: BlockId,
}

impl<I> Function<I> {
    pub fn new(entry: BlockId) -> Self {
        Function {
            blocks: Vec::new(),
            entry,
        }
    }

    pub fn add_block(&mut self, block: Block<I>) {
        self.blocks.push(block);
    }
}
