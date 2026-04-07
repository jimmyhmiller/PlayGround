//! Flat instruction stream support for register allocation.
//!
//! Many backends (JIT compilers, simple code generators) produce a flat
//! sequence of instructions with labels and branches rather than an
//! explicit CFG with basic blocks. This module provides:
//!
//! - [`FlatFunction`]: a simple trait for flat instruction streams
//! - [`FlatOperand`]: a simplified operand description
//! - [`allocate_flat`]: a one-call entry point that builds a CFG
//!   internally and runs the allocator
//!
//! # Example
//!
//! ```ignore
//! use regalloc::flat::*;
//! use regalloc::types::*;
//!
//! struct MyIR { /* ... */ }
//!
//! impl FlatFunction for MyIR {
//!     fn num_vregs(&self) -> usize { /* ... */ }
//!     fn vreg_class(&self, vreg: VReg) -> RegClass { /* ... */ }
//!     fn num_insts(&self) -> usize { /* ... */ }
//!     fn inst_operands(&self, index: usize) -> Vec<FlatOperand> { /* ... */ }
//!     // ...
//! }
//!
//! let alloc = allocate_flat(&my_ir, &my_target)?;
//! // alloc.get(InstId(5), 0) => physical register for operand 0 of instruction 5
//! ```

use std::collections::HashMap;

use crate::allocator::{AllocError, Allocation, RegisterAllocator};
use crate::ir::Function;
use crate::linear_scan::LinearScanAllocator;
use crate::target::Target;
use crate::types::*;

// ---------------------------------------------------------------------------
// FlatOperand — simplified operand descriptor
// ---------------------------------------------------------------------------

/// A simplified operand for flat instruction streams.
///
/// Users don't need to deal with the full [`Operand`] type — they just
/// describe what each instruction reads and writes.
#[derive(Clone, Debug)]
pub enum FlatOperand {
    /// The instruction defines (writes) this vreg. Any register in its class.
    Def(VReg),
    /// The instruction uses (reads) this vreg. Any register in its class.
    Use(VReg),
    /// The instruction reads and writes this vreg (tied operand).
    /// Common for accumulator-style instructions like ARM FMLA.
    UseDef(VReg),
    /// The instruction defines this vreg in a specific physical register.
    /// Used for call results (e.g., return value in X0/S0).
    DefFixed(VReg, PReg),
    /// The instruction uses this vreg and requires a specific physical register.
    /// Used for call arguments (e.g., first arg in X0/S0).
    UseFixed(VReg, PReg),
    /// Early def: defined before inputs are read, so it can't share a
    /// register with any input.
    EarlyDef(VReg),
}

impl FlatOperand {
    /// Extract the virtual register from this operand.
    pub fn vreg(&self) -> VReg {
        match self {
            FlatOperand::Def(v)
            | FlatOperand::Use(v)
            | FlatOperand::UseDef(v)
            | FlatOperand::DefFixed(v, _)
            | FlatOperand::UseFixed(v, _)
            | FlatOperand::EarlyDef(v) => *v,
        }
    }
}

// ---------------------------------------------------------------------------
// FlatFunction trait
// ---------------------------------------------------------------------------

/// A flat instruction stream that the register allocator can process.
///
/// This is the simplified alternative to [`Function`] for backends that
/// produce a linear sequence of instructions with labels and branches
/// rather than an explicit CFG.
///
/// The allocator will internally split the stream into basic blocks,
/// compute liveness, and run register allocation. The resulting
/// [`Allocation`] maps `(InstId(flat_index), operand_index)` to physical
/// registers, so you can look up results using original instruction indices.
pub trait FlatFunction {
    /// Total number of virtual registers used.
    fn num_vregs(&self) -> usize;

    /// The register class for a virtual register.
    fn vreg_class(&self, vreg: VReg) -> RegClass;

    /// Total number of instructions in the stream.
    fn num_insts(&self) -> usize;

    /// The operands of instruction at `index`.
    ///
    /// Return the defs, uses, and tied operands for this instruction.
    /// Label and branch pseudo-instructions should return empty operands
    /// (they don't read/write registers).
    fn inst_operands(&self, index: usize) -> Vec<FlatOperand>;

    /// If this instruction defines a label, return the label ID.
    ///
    /// Labels are branch targets. The label ID is an opaque `usize` that
    /// branches refer to. Multiple labels can share the same position.
    fn inst_label(&self, index: usize) -> Option<usize>;

    /// If this instruction is a branch, return its target label IDs.
    ///
    /// - Unconditional branch: return `Some(vec![target])`
    /// - Conditional branch: return `Some(vec![target])` (fall-through is implicit)
    /// - Not a branch: return `None`
    ///
    /// For conditional branches, the allocator assumes the next instruction
    /// is the fall-through target.
    fn inst_branch_targets(&self, index: usize) -> Option<Vec<usize>>;

    /// Is this instruction a return?
    fn is_return(&self, index: usize) -> bool;

    /// Is this instruction a call?
    fn is_call(&self, index: usize) -> bool;

    /// Physical registers clobbered by this instruction beyond what the
    /// operands describe. Typically all caller-saved registers for calls.
    fn inst_clobbers(&self, index: usize) -> Vec<PReg>;

    /// Is this a conditional branch? Only called when `inst_branch_targets`
    /// returns `Some`. Conditional branches have an implicit fall-through
    /// to the next instruction.
    ///
    /// Default: `false` (unconditional branch).
    fn is_conditional_branch(&self, index: usize) -> bool {
        let _ = index;
        false
    }
}

// ---------------------------------------------------------------------------
// Internal: flat → CFG adapter
// ---------------------------------------------------------------------------

/// A basic block built from a flat instruction stream.
struct FlatBlock {
    /// Block ID (index into the block list).
    id: BlockId,
    /// Flat instruction indices in this block.
    inst_indices: Vec<usize>,
    /// Successor block IDs.
    succs: Vec<BlockId>,
    /// Predecessor block IDs (filled in after all blocks are built).
    preds: Vec<BlockId>,
}

/// Adapter that wraps a `FlatFunction` and presents it as a `Function`.
///
/// Internally splits the flat stream into basic blocks at label and
/// branch boundaries, computes edges, and maps instructions.
struct FlatAdapter<'a, F: FlatFunction> {
    flat: &'a F,
    blocks: Vec<FlatBlock>,
    /// For each flat instruction index, its (BlockId, position-within-block).
    #[allow(dead_code)]
    inst_block: Vec<(BlockId, usize)>,
    /// All instructions in linearized order (same as flat order, but
    /// re-indexed as InstId). InstId(i) corresponds to flat index i.
    /// We store the operands converted to our Operand type.
    operands: Vec<Vec<Operand>>,
    /// Clobbers per instruction.
    clobbers: Vec<Vec<PReg>>,
    /// Which instructions are branches, returns, calls.
    is_branch: Vec<bool>,
    is_return: Vec<bool>,
    is_call: Vec<bool>,
}

impl<'a, F: FlatFunction> FlatAdapter<'a, F> {
    fn build(flat: &'a F) -> Self {
        let n = flat.num_insts();

        // Step 1: Identify block boundaries.
        // A new block starts:
        //   - At instruction 0
        //   - At any instruction that is a label target
        //   - At the instruction after any branch or return
        let mut block_starts: Vec<bool> = vec![false; n];
        if n > 0 {
            block_starts[0] = true;
        }

        // Also collect label → flat index mapping.
        let mut label_to_index: HashMap<usize, usize> = HashMap::new();

        for i in 0..n {
            if let Some(label_id) = flat.inst_label(i) {
                label_to_index.insert(label_id, i);
                // The label itself starts a new block.
                block_starts[i] = true;
            }
        }

        for i in 0..n {
            if flat.inst_branch_targets(i).is_some() || flat.is_return(i) {
                // The instruction after a branch/return starts a new block.
                if i + 1 < n {
                    block_starts[i + 1] = true;
                }
            }
        }

        // Also: branch targets start new blocks (they might not be labels
        // if the label is at the same position as another instruction).
        for i in 0..n {
            if let Some(targets) = flat.inst_branch_targets(i) {
                for label_id in targets {
                    if let Some(&target_idx) = label_to_index.get(&label_id) {
                        block_starts[target_idx] = true;
                    }
                }
            }
        }

        // Step 2: Build blocks.
        let mut blocks: Vec<FlatBlock> = Vec::new();
        let mut inst_to_block: Vec<BlockId> = vec![BlockId(0); n];
        let mut flat_idx_to_block: HashMap<usize, BlockId> = HashMap::new();

        let mut current_insts: Vec<usize> = Vec::new();

        for i in 0..n {
            if block_starts[i] && !current_insts.is_empty() {
                // Finish the previous block.
                let bid = BlockId(blocks.len() as u32);
                for &idx in &current_insts {
                    inst_to_block[idx] = bid;
                }
                flat_idx_to_block.insert(current_insts[0], bid);
                blocks.push(FlatBlock {
                    id: bid,
                    inst_indices: current_insts.clone(),
                    succs: Vec::new(),
                    preds: Vec::new(),
                });
                current_insts.clear();
            }
            current_insts.push(i);
        }
        // Finish the last block.
        if !current_insts.is_empty() {
            let bid = BlockId(blocks.len() as u32);
            for &idx in &current_insts {
                inst_to_block[idx] = bid;
            }
            flat_idx_to_block.insert(current_insts[0], bid);
            blocks.push(FlatBlock {
                id: bid,
                inst_indices: current_insts,
                succs: Vec::new(),
                preds: Vec::new(),
            });
        }

        // Build label_id → BlockId mapping.
        let mut label_to_block: HashMap<usize, BlockId> = HashMap::new();
        for (&label_id, &flat_idx) in &label_to_index {
            // The label's flat index might be inside a block (if the label
            // instruction is the first instruction of that block).
            label_to_block.insert(label_id, inst_to_block[flat_idx]);
        }

        // Step 3: Compute successor edges.
        for bi in 0..blocks.len() {
            let last_idx = *blocks[bi].inst_indices.last().unwrap();

            if flat.is_return(last_idx) {
                // No successors.
                continue;
            }

            if let Some(targets) = flat.inst_branch_targets(last_idx) {
                for label_id in &targets {
                    if let Some(&target_block) = label_to_block.get(label_id) {
                        if !blocks[bi].succs.contains(&target_block) {
                            blocks[bi].succs.push(target_block);
                        }
                    }
                }
                // Conditional branches also fall through.
                if flat.is_conditional_branch(last_idx) {
                    let next_block = BlockId(bi as u32 + 1);
                    if (next_block.0 as usize) < blocks.len()
                        && !blocks[bi].succs.contains(&next_block)
                    {
                        blocks[bi].succs.push(next_block);
                    }
                }
            } else {
                // Not a branch, not a return: fall through to next block.
                let next_block = BlockId(bi as u32 + 1);
                if (next_block.0 as usize) < blocks.len() {
                    blocks[bi].succs.push(next_block);
                }
            }
        }

        // Step 4: Compute predecessor edges.
        let num_blocks = blocks.len();
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); num_blocks];
        for block in &blocks {
            for &succ in &block.succs {
                preds[succ.0 as usize].push(block.id);
            }
        }
        for (i, block) in blocks.iter_mut().enumerate() {
            block.preds = preds[i].clone();
        }

        // Step 5: Build inst_block mapping.
        let mut inst_block_map: Vec<(BlockId, usize)> = vec![(BlockId(0), 0); n];
        for block in &blocks {
            for (pos, &flat_idx) in block.inst_indices.iter().enumerate() {
                inst_block_map[flat_idx] = (block.id, pos);
            }
        }

        // Step 6: Convert FlatOperands to Operands.
        let mut operands: Vec<Vec<Operand>> = Vec::with_capacity(n);
        let mut clobbers: Vec<Vec<PReg>> = Vec::with_capacity(n);
        let mut is_branch_vec: Vec<bool> = Vec::with_capacity(n);
        let mut is_return_vec: Vec<bool> = Vec::with_capacity(n);
        let mut is_call_vec: Vec<bool> = Vec::with_capacity(n);

        for i in 0..n {
            let flat_ops = flat.inst_operands(i);
            let ops: Vec<Operand> = flat_ops
                .into_iter()
                .map(|fo| convert_flat_operand(fo, flat))
                .collect();
            operands.push(ops);
            clobbers.push(flat.inst_clobbers(i));
            is_branch_vec.push(flat.inst_branch_targets(i).is_some());
            is_return_vec.push(flat.is_return(i));
            is_call_vec.push(flat.is_call(i));
        }

        FlatAdapter {
            flat,
            blocks,
            inst_block: inst_block_map,
            operands,
            clobbers,
            is_branch: is_branch_vec,
            is_return: is_return_vec,
            is_call: is_call_vec,
        }
    }
}

/// Convert a FlatOperand to our internal Operand type.
fn convert_flat_operand<F: FlatFunction>(fo: FlatOperand, flat: &F) -> Operand {
    match fo {
        FlatOperand::Def(vreg) => Operand {
            reg: Reg::Virtual(vreg),
            kind: OperandKind::Def,
            constraint: OperandConstraint::RegClass(flat.vreg_class(vreg)),
        },
        FlatOperand::Use(vreg) => Operand {
            reg: Reg::Virtual(vreg),
            kind: OperandKind::Use,
            constraint: OperandConstraint::RegClass(flat.vreg_class(vreg)),
        },
        FlatOperand::UseDef(vreg) => Operand {
            reg: Reg::Virtual(vreg),
            kind: OperandKind::UseDef,
            constraint: OperandConstraint::RegClass(flat.vreg_class(vreg)),
        },
        FlatOperand::DefFixed(vreg, preg) => Operand {
            reg: Reg::Virtual(vreg),
            kind: OperandKind::Def,
            constraint: OperandConstraint::FixedReg(preg),
        },
        FlatOperand::UseFixed(vreg, preg) => Operand {
            reg: Reg::Virtual(vreg),
            kind: OperandKind::Use,
            constraint: OperandConstraint::FixedReg(preg),
        },
        FlatOperand::EarlyDef(vreg) => Operand {
            reg: Reg::Virtual(vreg),
            kind: OperandKind::EarlyDef,
            constraint: OperandConstraint::RegClass(flat.vreg_class(vreg)),
        },
    }
}

// ---------------------------------------------------------------------------
// Function impl for FlatAdapter
// ---------------------------------------------------------------------------

impl<'a, F: FlatFunction> Function for FlatAdapter<'a, F> {
    type BlockIter<'b>
        = std::vec::IntoIter<BlockId>
    where
        Self: 'b;
    type InstIter<'b>
        = std::vec::IntoIter<InstId>
    where
        Self: 'b;
    type OperandIter<'b>
        = std::vec::IntoIter<Operand>
    where
        Self: 'b;
    type SuccIter<'b>
        = std::iter::Copied<std::slice::Iter<'b, BlockId>>
    where
        Self: 'b;
    type PredIter<'b>
        = std::iter::Copied<std::slice::Iter<'b, BlockId>>
    where
        Self: 'b;

    fn num_vregs(&self) -> usize {
        self.flat.num_vregs()
    }

    fn vreg_class(&self, vreg: VReg) -> RegClass {
        self.flat.vreg_class(vreg)
    }

    fn blocks(&self) -> Self::BlockIter<'_> {
        self.blocks.iter().map(|b| b.id).collect::<Vec<_>>().into_iter()
    }

    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry_block(&self) -> BlockId {
        BlockId(0)
    }

    fn block_insts(&self, block: BlockId) -> Self::InstIter<'_> {
        self.blocks[block.0 as usize]
            .inst_indices
            .iter()
            .map(|&i| InstId(i as u32))
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn block_succs(&self, block: BlockId) -> Self::SuccIter<'_> {
        self.blocks[block.0 as usize].succs.iter().copied()
    }

    fn block_preds(&self, block: BlockId) -> Self::PredIter<'_> {
        self.blocks[block.0 as usize].preds.iter().copied()
    }

    fn inst_operands(&self, inst: InstId) -> Self::OperandIter<'_> {
        self.operands[inst.0 as usize].clone().into_iter()
    }

    fn is_branch(&self, inst: InstId) -> bool {
        self.is_branch[inst.0 as usize]
    }

    fn is_return(&self, inst: InstId) -> bool {
        self.is_return[inst.0 as usize]
    }

    fn is_call(&self, inst: InstId) -> bool {
        self.is_call[inst.0 as usize]
    }

    fn inst_clobbers(&self, inst: InstId) -> &[PReg] {
        &self.clobbers[inst.0 as usize]
    }

    fn block_params(&self, _block: BlockId) -> &[VReg] {
        // Flat IRs don't have block parameters (no SSA phi nodes).
        &[]
    }

    fn branch_args(&self, _inst: InstId, _succ_idx: usize) -> &[VReg] {
        // Flat IRs don't have branch arguments.
        &[]
    }

    fn num_insts(&self) -> usize {
        self.flat.num_insts()
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Allocate registers for a flat instruction stream.
///
/// This is the main entry point for backends with flat IR. It:
/// 1. Splits the instruction stream into basic blocks at label/branch boundaries
/// 2. Computes liveness using standard backward dataflow
/// 3. Runs linear-scan register allocation
/// 4. Returns an [`Allocation`] where `InstId` values correspond to flat
///    instruction indices
///
/// # Errors
///
/// Returns [`AllocError`] if allocation fails (e.g., unsatisfiable constraints).
pub fn allocate_flat<F: FlatFunction, T: Target>(
    flat: &F,
    target: &T,
) -> Result<Allocation, AllocError> {
    let adapter = FlatAdapter::build(flat);
    let mut allocator = LinearScanAllocator;
    allocator.allocate(&adapter, target)
}

// ---------------------------------------------------------------------------
// SimpleFlat — a concrete FlatFunction for quick use
// ---------------------------------------------------------------------------

/// A concrete flat function that stores all data inline.
///
/// Useful for testing or for backends that want to build up the flat IR
/// incrementally and then allocate.
pub struct SimpleFlat {
    pub num_vregs: usize,
    pub vreg_classes: Vec<RegClass>,
    pub insts: Vec<SimpleFlatInst>,
}

/// An instruction in a [`SimpleFlat`] function.
pub struct SimpleFlatInst {
    pub operands: Vec<FlatOperand>,
    pub label: Option<usize>,
    pub branch_targets: Option<Vec<usize>>,
    pub is_conditional: bool,
    pub is_return: bool,
    pub is_call: bool,
    pub clobbers: Vec<PReg>,
}

impl SimpleFlatInst {
    /// Create a normal instruction with the given operands.
    pub fn op(operands: Vec<FlatOperand>) -> Self {
        SimpleFlatInst {
            operands,
            label: None,
            branch_targets: None,
            is_conditional: false,
            is_return: false,
            is_call: false,
            clobbers: Vec::new(),
        }
    }

    /// Create a label pseudo-instruction.
    pub fn label(id: usize) -> Self {
        SimpleFlatInst {
            operands: Vec::new(),
            label: Some(id),
            branch_targets: None,
            is_conditional: false,
            is_return: false,
            is_call: false,
            clobbers: Vec::new(),
        }
    }

    /// Create an unconditional branch.
    pub fn branch(target: usize) -> Self {
        SimpleFlatInst {
            operands: Vec::new(),
            label: None,
            branch_targets: Some(vec![target]),
            is_conditional: false,
            is_return: false,
            is_call: false,
            clobbers: Vec::new(),
        }
    }

    /// Create a conditional branch (with operands for the condition).
    pub fn cond_branch(target: usize, operands: Vec<FlatOperand>) -> Self {
        SimpleFlatInst {
            operands,
            label: None,
            branch_targets: Some(vec![target]),
            is_conditional: true,
            is_return: false,
            is_call: false,
            clobbers: Vec::new(),
        }
    }

    /// Create a return instruction.
    pub fn ret(operands: Vec<FlatOperand>) -> Self {
        SimpleFlatInst {
            operands,
            label: None,
            branch_targets: None,
            is_conditional: false,
            is_return: true,
            is_call: false,
            clobbers: Vec::new(),
        }
    }

    /// Create a call instruction.
    pub fn call(operands: Vec<FlatOperand>, clobbers: Vec<PReg>) -> Self {
        SimpleFlatInst {
            operands,
            label: None,
            branch_targets: None,
            is_conditional: false,
            is_return: false,
            is_call: true,
            clobbers,
        }
    }
}

impl FlatFunction for SimpleFlat {
    fn num_vregs(&self) -> usize {
        self.num_vregs
    }

    fn vreg_class(&self, vreg: VReg) -> RegClass {
        self.vreg_classes[vreg.0 as usize]
    }

    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    fn inst_operands(&self, index: usize) -> Vec<FlatOperand> {
        self.insts[index].operands.clone()
    }

    fn inst_label(&self, index: usize) -> Option<usize> {
        self.insts[index].label
    }

    fn inst_branch_targets(&self, index: usize) -> Option<Vec<usize>> {
        self.insts[index].branch_targets.clone()
    }

    fn is_return(&self, index: usize) -> bool {
        self.insts[index].is_return
    }

    fn is_call(&self, index: usize) -> bool {
        self.insts[index].is_call
    }

    fn inst_clobbers(&self, index: usize) -> Vec<PReg> {
        self.insts[index].clobbers.clone()
    }

    fn is_conditional_branch(&self, index: usize) -> bool {
        self.insts[index].is_conditional
    }
}

// ---------------------------------------------------------------------------
// SimpleFlatBuilder — ergonomic builder for SimpleFlat
// ---------------------------------------------------------------------------

/// Builder for constructing [`SimpleFlat`] functions incrementally.
///
/// Modeled after how nano-gpt's `MachBuilder` works — allocate vregs,
/// push instructions, and build.
pub struct SimpleFlatBuilder {
    num_vregs: u32,
    vreg_classes: Vec<RegClass>,
    insts: Vec<SimpleFlatInst>,
}

impl SimpleFlatBuilder {
    pub fn new() -> Self {
        SimpleFlatBuilder {
            num_vregs: 0,
            vreg_classes: Vec::new(),
            insts: Vec::new(),
        }
    }

    /// Allocate a new virtual register in the given class.
    pub fn vreg(&mut self, class: RegClass) -> VReg {
        let id = self.num_vregs;
        self.num_vregs += 1;
        self.vreg_classes.push(class);
        VReg(id)
    }

    /// Push an instruction. Returns its flat index.
    pub fn push(&mut self, inst: SimpleFlatInst) -> usize {
        let idx = self.insts.len();
        self.insts.push(inst);
        idx
    }

    /// Build the final SimpleFlat function.
    pub fn build(self) -> SimpleFlat {
        SimpleFlat {
            num_vregs: self.num_vregs as usize,
            vreg_classes: self.vreg_classes,
            insts: self.insts,
        }
    }
}
