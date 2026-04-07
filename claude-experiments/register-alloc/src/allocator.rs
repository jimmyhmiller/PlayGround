//! The allocator trait and allocation output types.

use std::collections::HashMap;
use std::fmt;

use crate::ir::{Function, SafepointAction};
use crate::target::Target;
use crate::types::*;

/// The output of register allocation: a mapping from virtual registers
/// to physical registers (or spill slots), plus any moves/spills the
/// allocator needs inserted.
#[derive(Debug, Clone)]
pub struct Allocation {
    /// For each instruction, the physical register assigned to each operand,
    /// in the same order as `Function::inst_operands`.
    /// Key: (InstId, operand_index), Value: the assigned physical register.
    pub inst_allocs: HashMap<(InstId, usize), PReg>,

    /// Moves that must be inserted. Each move is inserted at the given
    /// program point.
    pub moves: Vec<InsertedMove>,

    /// Spill slots allocated. Maps virtual registers that were spilled
    /// to their stack slot.
    pub spill_slots: HashMap<VReg, SpillSlot>,

    /// Total number of spill slots used.
    pub num_spill_slots: u32,

    /// Stackmaps: at each safepoint instruction, the location of every live
    /// value that requested `Record` or `SpillAndRecord`. Empty if the
    /// function has no safepoints or all actions are `CallingConvention`/`Ignore`.
    pub stackmaps: HashMap<InstId, Vec<StackmapEntry>>,
}

/// One entry in a safepoint's stackmap: where a live value can be found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackmapEntry {
    /// The virtual register this entry describes.
    pub vreg: VReg,
    /// Where the value lives at this safepoint.
    pub location: MoveOperand,
    /// The action that was requested (Record or SpillAndRecord).
    pub action: SafepointAction,
}

/// A move that the allocator needs the user to insert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertedMove {
    /// Where to insert this move in the program.
    pub at: MovePosition,
    /// Source: a physical register or spill slot.
    pub from: MoveOperand,
    /// Destination: a physical register or spill slot.
    pub to: MoveOperand,
    /// The register class (determines move instruction + spill size).
    pub class: RegClass,
}

/// Where in the program to insert a move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MovePosition {
    /// Before the given instruction.
    Before(InstId),
    /// After the given instruction.
    After(InstId),
    /// On the edge between two blocks (for phi resolution).
    /// Conceptually at the end of `from` block, targeting `to` block.
    BlockEdge { from: BlockId, to: BlockId },
}

/// Source or destination of an inserted move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveOperand {
    Reg(PReg),
    SpillSlot(SpillSlot),
}

/// Errors that can occur during allocation.
#[derive(Debug, Clone)]
pub enum AllocError {
    /// Ran out of registers and couldn't spill (shouldn't happen with
    /// a correct spilling strategy, but possible with constraints).
    OutOfRegisters {
        inst: InstId,
        class: RegClass,
    },
    /// Unsatisfiable constraint (e.g., two operands need the same fixed
    /// register simultaneously).
    UnsatisfiableConstraint {
        inst: InstId,
        detail: String,
    },
    /// The input IR is malformed.
    InvalidInput(String),
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::OutOfRegisters { inst, class } => {
                write!(f, "out of registers at {:?} for class {:?}", inst, class)
            }
            AllocError::UnsatisfiableConstraint { inst, detail } => {
                write!(
                    f,
                    "unsatisfiable constraint at {:?}: {}",
                    inst, detail
                )
            }
            AllocError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
        }
    }
}

impl std::error::Error for AllocError {}

impl Allocation {
    pub fn new() -> Self {
        Allocation {
            inst_allocs: HashMap::new(),
            moves: Vec::new(),
            spill_slots: HashMap::new(),
            num_spill_slots: 0,
            stackmaps: HashMap::new(),
        }
    }

    /// Look up the physical register assigned to operand `op_idx` of `inst`.
    pub fn get(&self, inst: InstId, op_idx: usize) -> Option<PReg> {
        self.inst_allocs.get(&(inst, op_idx)).copied()
    }

    /// Record an allocation.
    pub fn set(&mut self, inst: InstId, op_idx: usize, reg: PReg) {
        self.inst_allocs.insert((inst, op_idx), reg);
    }

    /// Allocate a new spill slot for a vreg and return it.
    pub fn add_spill(&mut self, vreg: VReg) -> SpillSlot {
        let slot = SpillSlot(self.num_spill_slots);
        self.num_spill_slots += 1;
        self.spill_slots.insert(vreg, slot);
        slot
    }
}

/// The main trait that register allocation algorithms implement.
pub trait RegisterAllocator {
    /// Allocate registers for the given function on the given target.
    ///
    /// Returns an `Allocation` mapping vregs to pregs, plus any
    /// moves/spills that need to be inserted.
    fn allocate<F: Function, T: Target>(
        &mut self,
        func: &F,
        target: &T,
    ) -> Result<Allocation, AllocError>;

    /// Human-readable name for this allocator (e.g., "linear-scan", "graph-coloring").
    fn name(&self) -> &str;
}
