//! Traits for interfacing with an arbitrary IR.
//!
//! Users implement these traits to let the register allocator read their IR.
//! The allocator never mutates the IR directly — it produces an [`Allocation`]
//! that the user applies however they like.

use crate::types::*;

/// A function (or compilation unit) to allocate registers for.
///
/// This is the main entry point trait. A `Function` is a collection of
/// basic blocks containing instructions.
pub trait Function {
    /// Iterator over all block IDs in layout order.
    type BlockIter<'a>: Iterator<Item = BlockId>
    where
        Self: 'a;

    /// Iterator over instruction IDs within a block, in order.
    type InstIter<'a>: Iterator<Item = InstId>
    where
        Self: 'a;

    /// Iterator over operands of an instruction.
    type OperandIter<'a>: Iterator<Item = Operand>
    where
        Self: 'a;

    /// Iterator over successor block IDs.
    type SuccIter<'a>: Iterator<Item = BlockId>
    where
        Self: 'a;

    /// Iterator over predecessor block IDs.
    type PredIter<'a>: Iterator<Item = BlockId>
    where
        Self: 'a;

    /// The total number of virtual registers used.
    fn num_vregs(&self) -> usize;

    /// The register class for a virtual register.
    fn vreg_class(&self, vreg: VReg) -> RegClass;

    /// All blocks in layout order.
    fn blocks(&self) -> Self::BlockIter<'_>;

    /// The number of blocks.
    fn num_blocks(&self) -> usize;

    /// The entry block.
    fn entry_block(&self) -> BlockId;

    /// Instructions in the given block, in order.
    fn block_insts(&self, block: BlockId) -> Self::InstIter<'_>;

    /// Successor blocks (branch targets) of a block.
    fn block_succs(&self, block: BlockId) -> Self::SuccIter<'_>;

    /// Predecessor blocks of a block.
    fn block_preds(&self, block: BlockId) -> Self::PredIter<'_>;

    /// The operands of an instruction.
    fn inst_operands(&self, inst: InstId) -> Self::OperandIter<'_>;

    /// Is this instruction a branch (transfers control to another block)?
    fn is_branch(&self, inst: InstId) -> bool;

    /// Is this instruction a return?
    fn is_return(&self, inst: InstId) -> bool;

    /// Is this instruction a call? Calls clobber caller-saved registers.
    fn is_call(&self, inst: InstId) -> bool;

    /// Physical registers clobbered by this instruction beyond what
    /// the operands describe. For calls, this is typically all caller-saved
    /// registers. For instructions like x86 `idiv`, this includes
    /// implicit register uses.
    fn inst_clobbers(&self, inst: InstId) -> &[PReg];

    /// Block parameters (phi inputs) for a block. In SSA form, these
    /// are the virtual registers defined at block entry by phi nodes.
    /// For non-SSA IRs, return empty.
    fn block_params(&self, block: BlockId) -> &[VReg];

    /// For a branch instruction, the arguments passed to each successor's
    /// block params. `succ_idx` is the index into the successor list.
    /// Returns the virtual registers passed as arguments.
    /// For non-SSA IRs, return empty.
    fn branch_args(&self, inst: InstId, succ_idx: usize) -> &[VReg];

    /// Returns the total number of instructions (used for sizing data structures).
    fn num_insts(&self) -> usize;
}

/// Optional trait for IRs that provide liveness info directly.
/// If not implemented, the allocator computes liveness itself.
pub trait ProvidesLiveness: Function {
    /// Live-in virtual registers for a block.
    fn live_in(&self, block: BlockId) -> &[VReg];

    /// Live-out virtual registers for a block.
    fn live_out(&self, block: BlockId) -> &[VReg];
}

/// Optional trait for SSA-form IRs. Enables SSA-aware allocation
/// strategies (e.g., SSA-based coloring, SSA destruction).
pub trait SSAFunction: Function {
    /// The unique instruction that defines this virtual register.
    /// In SSA form, every vreg has exactly one def.
    fn vreg_def(&self, vreg: VReg) -> InstId;

    /// The block containing the definition of this vreg.
    fn vreg_def_block(&self, vreg: VReg) -> BlockId;
}
