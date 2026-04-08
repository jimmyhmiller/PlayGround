//! Bridge between `dynir::ir::Function` and `register_alloc::ir::Function`.
//!
//! Implements the `register_alloc::ir::Function` trait directly over the
//! dynir SSA IR so the batch register allocator can process it without
//! an intermediate machine IR.
//!
//! ## Design
//!
//! dynir stores instructions per-block (`Block.insts: Vec<InstNode>`).
//! register-alloc wants global `InstId` indices. We linearize by walking
//! blocks in order: block 0's insts get IDs 0..n₀, block 1 gets n₀..n₀+n₁,
//! etc. Each block's terminator is represented as the last InstId in that
//! block's range (it may use/def values and is a branch/return).
//!
//! `dynir::Value` maps 1:1 to `register_alloc::VReg` (both u32 newtypes).

use dynir::ir::{self as dir, Inst, Terminator};
use dynir::types::Type;
use regalloc::ir::{Function, SafepointAction};
use regalloc::target::{CallingConvention, RegInfo};
use regalloc::types::*;

// ── RegClass constants ────────────────────────────────────────────

pub const GP: RegClass = RegClass(0);
pub const FP: RegClass = RegClass(1);

// ── PReg encoding ─────────────────────────────────────────────────
// GP registers: PReg(0..=27) → X0..X27
// FP registers: PReg(32..=63) → D0..D31

pub fn gp_preg(i: u8) -> PReg {
    PReg(i as u16)
}
pub fn fp_preg(i: u8) -> PReg {
    PReg(32 + i as u16)
}

// ── Adapted function ──────────────────────────────────────────────

/// Wraps a `dynir::ir::Function` to implement `register_alloc::ir::Function`.
///
/// Linearizes block-local instruction indices into global `InstId`s.
/// The terminator of each block occupies the last InstId in the block's range.
pub struct DynIRFunction<'a> {
    func: &'a dir::Function,
    /// block_inst_base[b] = first global InstId for block b
    block_inst_base: Vec<u32>,
    /// Total instruction count (including terminators)
    total_insts: u32,
    /// Precomputed predecessors
    preds: Vec<Vec<dir::BlockId>>,
    /// Precomputed block params as VRegs (cached to return slices)
    block_param_vregs: Vec<Vec<VReg>>,
    /// Precomputed branch args per (terminator_inst_id, succ_idx)
    /// Stored as: branch_args_cache[block_idx][succ_idx] = Vec<VReg>
    branch_args_cache: Vec<Vec<Vec<VReg>>>,
    /// Precomputed operands per global InstId
    operands_cache: Vec<Vec<Operand>>,
    /// Caller-saved clobber set (for calls)
    call_clobbers: Vec<PReg>,
}

impl<'a> DynIRFunction<'a> {
    pub fn new(func: &'a dir::Function) -> Self {
        // Compute block_inst_base: linearize instructions
        let mut block_inst_base = Vec::with_capacity(func.blocks.len());
        let mut offset = 0u32;
        for block in &func.blocks {
            block_inst_base.push(offset);
            // +1 for the terminator
            offset += block.insts.len() as u32 + 1;
        }
        let total_insts = offset;

        let preds = func.predecessors();

        // Cache block params as VRegs
        let block_param_vregs: Vec<Vec<VReg>> = func
            .blocks
            .iter()
            .map(|b| b.params.iter().map(|(v, _)| VReg(v.index() as u32)).collect())
            .collect();

        // Cache branch args
        let branch_args_cache: Vec<Vec<Vec<VReg>>> = func
            .blocks
            .iter()
            .map(|b| {
                let succs = b.terminator.successors();
                succs
                    .iter()
                    .enumerate()
                    .map(|(succ_idx, _)| {
                        let args = Self::terminator_branch_args(&b.terminator, succ_idx);
                        args.iter().map(|v| VReg(v.index() as u32)).collect()
                    })
                    .collect()
            })
            .collect();

        // Build call clobbers (all caller-saved GP + FP)
        // X0-X15: standard caller-saved temporaries
        // X16-X18: IP0/IP1 (linker scratch) + platform register — also clobbered by calls
        // X27: used as scratch by emit_call to load the call table base
        let mut call_clobbers = Vec::new();
        for i in 0..=18u8 {
            call_clobbers.push(gp_preg(i));
        }
        call_clobbers.push(gp_preg(27));
        for i in 0..=7u8 {
            call_clobbers.push(fp_preg(i));
        }
        for i in 16..=31u8 {
            call_clobbers.push(fp_preg(i));
        }

        // Precompute operands for all instructions
        let mut operands_cache = Vec::with_capacity(total_insts as usize);
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let base = block_inst_base[block_idx];
            for (local_idx, inst_node) in block.insts.iter().enumerate() {
                let global_id = base + local_idx as u32;
                let _ = global_id; // just for clarity
                operands_cache.push(Self::compute_inst_operands(func, inst_node));
            }
            // Terminator operands
            operands_cache.push(Self::compute_terminator_operands(&block.terminator));
        }

        DynIRFunction {
            func,
            block_inst_base,
            total_insts,
            preds,
            block_param_vregs,
            branch_args_cache,
            operands_cache,
            call_clobbers,
        }
    }

    /// Get the base InstId offset for a block.
    pub fn block_inst_base(&self, block: usize) -> u32 {
        self.block_inst_base[block]
    }

    /// Global InstId for block b's local instruction index i.
    fn global_inst_id(&self, block: usize, local: usize) -> InstId {
        InstId(self.block_inst_base[block] + local as u32)
    }

    /// Global InstId for block b's terminator.
    fn terminator_inst_id(&self, block: usize) -> InstId {
        let base = self.block_inst_base[block];
        let n = self.func.blocks[block].insts.len() as u32;
        InstId(base + n)
    }

    /// Which block does this global InstId belong to?
    fn inst_block(&self, inst: InstId) -> usize {
        let id = inst.0;
        // Binary search for the block
        match self.block_inst_base.binary_search(&id) {
            Ok(b) => b,
            Err(b) => b - 1,
        }
    }

    /// Is this InstId a terminator?
    fn is_terminator(&self, inst: InstId) -> bool {
        let block = self.inst_block(inst);
        inst == self.terminator_inst_id(block)
    }

    /// Get the dynir instruction for a global InstId.
    fn get_inst(&self, inst: InstId) -> Option<&Inst> {
        let block = self.inst_block(inst);
        let local = inst.0 - self.block_inst_base[block];
        let block_data = &self.func.blocks[block];
        if (local as usize) < block_data.insts.len() {
            Some(&block_data.insts[local as usize].inst)
        } else {
            None // terminator
        }
    }

    fn get_terminator(&self, inst: InstId) -> Option<&Terminator> {
        if self.is_terminator(inst) {
            let block = self.inst_block(inst);
            Some(&self.func.blocks[block].terminator)
        } else {
            None
        }
    }

    fn reg_class_for_type(ty: Type) -> RegClass {
        match ty {
            Type::F64 => FP,
            _ => GP,
        }
    }

    fn compute_inst_operands(func: &dir::Function, inst_node: &dir::InstNode) -> Vec<Operand> {
        let mut ops = Vec::new();

        // Def (result)
        if let Some(val) = inst_node.value {
            let ty = func.value_type(val);
            let class = Self::reg_class_for_type(ty);

            // For calls, result must be in X0 (or D0 for float)
            let constraint = match &inst_node.inst {
                Inst::Call(_, _) | Inst::CallIndirect(_, _, _) => {
                    if ty == Type::F64 {
                        OperandConstraint::FixedReg(fp_preg(0))
                    } else {
                        OperandConstraint::FixedReg(gp_preg(0))
                    }
                }
                _ => OperandConstraint::RegClass(class),
            };
            ops.push(Operand {
                reg: Reg::Virtual(VReg(val.index() as u32)),
                kind: OperandKind::Def,
                constraint,
            });
        }

        // Uses
        match &inst_node.inst {
            Inst::Iconst(_, _) | Inst::F64Const(_) | Inst::StackAddr(_) => {}

            Inst::Call(fref, args) => {
                // Arguments go in fixed registers (calling convention)
                for (i, &arg) in args.iter().enumerate() {
                    let ty = func.value_type(arg);
                    let class = Self::reg_class_for_type(ty);
                    let constraint = if i < 16 {
                        // Internal CC: X0-X15 for GP args
                        OperandConstraint::FixedReg(gp_preg(i as u8))
                    } else {
                        OperandConstraint::RegClass(class)
                    };
                    ops.push(Operand {
                        reg: Reg::Virtual(VReg(arg.index() as u32)),
                        kind: OperandKind::Use,
                        constraint,
                    });
                }
            }

            Inst::CallIndirect(callee, args, _) => {
                // Callee pointer — needs to be in a register but not a fixed one
                // (it'll be moved to X28 for BLR)
                let ty = func.value_type(*callee);
                ops.push(Operand {
                    reg: Reg::Virtual(VReg(callee.index() as u32)),
                    kind: OperandKind::Use,
                    constraint: OperandConstraint::RegClass(Self::reg_class_for_type(ty)),
                });
                // Arguments in fixed registers
                for (i, &arg) in args.iter().enumerate() {
                    let ty = func.value_type(arg);
                    let class = Self::reg_class_for_type(ty);
                    let constraint = if i < 16 {
                        OperandConstraint::FixedReg(gp_preg(i as u8))
                    } else {
                        OperandConstraint::RegClass(class)
                    };
                    ops.push(Operand {
                        reg: Reg::Virtual(VReg(arg.index() as u32)),
                        kind: OperandKind::Use,
                        constraint,
                    });
                }
            }

            // Generic: all value operands are unconstrained uses
            other => {
                other.for_each_value(|v| {
                    let ty = func.value_type(v);
                    let class = Self::reg_class_for_type(ty);
                    ops.push(Operand {
                        reg: Reg::Virtual(VReg(v.index() as u32)),
                        kind: OperandKind::Use,
                        constraint: OperandConstraint::RegClass(class),
                    });
                });
            }
        }

        ops
    }

    fn compute_terminator_operands(term: &Terminator) -> Vec<Operand> {
        let mut ops = Vec::new();
        match term {
            Terminator::Ret(v) => {
                // Return value must be in X0
                ops.push(Operand {
                    reg: Reg::Virtual(VReg(v.index() as u32)),
                    kind: OperandKind::Use,
                    constraint: OperandConstraint::FixedReg(gp_preg(0)),
                });
            }
            Terminator::RetVoid => {}
            Terminator::Jump(_, args) => {
                for v in args {
                    ops.push(Operand {
                        reg: Reg::Virtual(VReg(v.index() as u32)),
                        kind: OperandKind::Use,
                        constraint: OperandConstraint::RegClass(GP),
                    });
                }
            }
            Terminator::BrIf {
                cond,
                then_args,
                else_args,
                ..
            } => {
                ops.push(Operand {
                    reg: Reg::Virtual(VReg(cond.index() as u32)),
                    kind: OperandKind::Use,
                    constraint: OperandConstraint::RegClass(GP),
                });
                for v in then_args.iter().chain(else_args.iter()) {
                    ops.push(Operand {
                        reg: Reg::Virtual(VReg(v.index() as u32)),
                        kind: OperandKind::Use,
                        constraint: OperandConstraint::RegClass(GP),
                    });
                }
            }
            Terminator::Switch {
                val,
                cases,
                default_args,
                ..
            } => {
                ops.push(Operand {
                    reg: Reg::Virtual(VReg(val.index() as u32)),
                    kind: OperandKind::Use,
                    constraint: OperandConstraint::RegClass(GP),
                });
                for (_, _, args) in cases {
                    for v in args {
                        ops.push(Operand {
                            reg: Reg::Virtual(VReg(v.index() as u32)),
                            kind: OperandKind::Use,
                            constraint: OperandConstraint::RegClass(GP),
                        });
                    }
                }
                for v in default_args {
                    ops.push(Operand {
                        reg: Reg::Virtual(VReg(v.index() as u32)),
                        kind: OperandKind::Use,
                        constraint: OperandConstraint::RegClass(GP),
                    });
                }
            }
            _ => {
                // Invoke, ResumeSlice, AbortToPrompt — treat conservatively
                term.for_each_value(|v| {
                    ops.push(Operand {
                        reg: Reg::Virtual(VReg(v.index() as u32)),
                        kind: OperandKind::Use,
                        constraint: OperandConstraint::RegClass(GP),
                    });
                });
            }
        }
        ops
    }

    fn terminator_branch_args(term: &Terminator, succ_idx: usize) -> &[dir::Value] {
        match term {
            Terminator::Jump(_, args) => {
                assert_eq!(succ_idx, 0);
                args
            }
            Terminator::BrIf {
                then_args,
                else_args,
                ..
            } => {
                if succ_idx == 0 {
                    then_args
                } else {
                    else_args
                }
            }
            Terminator::Switch {
                cases,
                default_args,
                ..
            } => {
                if succ_idx < cases.len() {
                    &cases[succ_idx].2
                } else {
                    default_args
                }
            }
            Terminator::Invoke {
                normal_args,
                exception_args,
                ..
            }
            | Terminator::InvokeIndirect {
                normal_args,
                exception_args,
                ..
            } => {
                if succ_idx == 0 {
                    normal_args
                } else {
                    exception_args
                }
            }
            _ => &[],
        }
    }
}

// ── Function trait impl ───────────────────────────────────────────

impl<'a> Function for DynIRFunction<'a> {
    type BlockIter<'b> = std::iter::Map<std::ops::Range<u32>, fn(u32) -> BlockId> where Self: 'b;
    type InstIter<'b> = std::iter::Map<std::ops::Range<u32>, fn(u32) -> InstId> where Self: 'b;
    type OperandIter<'b> = std::vec::IntoIter<Operand> where Self: 'b;
    type SuccIter<'b> = std::vec::IntoIter<BlockId> where Self: 'b;
    type PredIter<'b> = std::vec::IntoIter<BlockId> where Self: 'b;

    fn num_vregs(&self) -> usize {
        self.func.value_types.len()
    }

    fn vreg_class(&self, vreg: VReg) -> RegClass {
        let ty = self.func.value_types[vreg.0 as usize];
        Self::reg_class_for_type(ty)
    }

    fn blocks(&self) -> Self::BlockIter<'_> {
        (0..self.func.blocks.len() as u32).map(BlockId)
    }

    fn num_blocks(&self) -> usize {
        self.func.blocks.len()
    }

    fn entry_block(&self) -> BlockId {
        BlockId(0)
    }

    fn block_insts(&self, block: BlockId) -> Self::InstIter<'_> {
        let b = block.0 as usize;
        let base = self.block_inst_base[b];
        let n = self.func.blocks[b].insts.len() as u32;
        // Include terminator (+1)
        (base..base + n + 1).map(InstId)
    }

    fn block_succs(&self, block: BlockId) -> Self::SuccIter<'_> {
        self.func.blocks[block.0 as usize]
            .terminator
            .successors()
            .into_iter()
            .map(|b| BlockId(b.index() as u32))
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn block_preds(&self, block: BlockId) -> Self::PredIter<'_> {
        self.preds[block.0 as usize]
            .iter()
            .map(|b| BlockId(b.index() as u32))
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn inst_operands(&self, inst: InstId) -> Self::OperandIter<'_> {
        self.operands_cache[inst.0 as usize].clone().into_iter()
    }

    fn is_branch(&self, inst: InstId) -> bool {
        self.is_terminator(inst)
            && !matches!(
                self.get_terminator(inst),
                Some(Terminator::Ret(_) | Terminator::RetVoid)
            )
    }

    fn is_return(&self, inst: InstId) -> bool {
        matches!(
            self.get_terminator(inst),
            Some(Terminator::Ret(_) | Terminator::RetVoid)
        )
    }

    fn is_call(&self, inst: InstId) -> bool {
        matches!(
            self.get_inst(inst),
            Some(Inst::Call(_, _) | Inst::CallIndirect(_, _, _))
        )
    }

    fn inst_clobbers(&self, inst: InstId) -> &[PReg] {
        if self.is_call(inst) {
            &self.call_clobbers
        } else {
            &[]
        }
    }

    fn block_params(&self, block: BlockId) -> &[VReg] {
        &self.block_param_vregs[block.0 as usize]
    }

    fn branch_args(&self, inst: InstId, succ_idx: usize) -> &[VReg] {
        let block = self.inst_block(inst);
        if succ_idx < self.branch_args_cache[block].len() {
            &self.branch_args_cache[block][succ_idx]
        } else {
            &[]
        }
    }

    fn num_insts(&self) -> usize {
        self.total_insts as usize
    }

    fn remat_value(&self, vreg: VReg) -> Option<u64> {
        // Find the instruction that defines this vreg.
        // If it's an Iconst or F64Const, return the immediate value.
        let val = dir::Value::from_index(vreg.0 as usize);
        for block in &self.func.blocks {
            for inst_node in &block.insts {
                if inst_node.value == Some(val) {
                    return match &inst_node.inst {
                        Inst::Iconst(_, imm) => Some(*imm as u64),
                        Inst::F64Const(f) => Some(f.to_bits()),
                        _ => None,
                    };
                }
            }
        }
        None
    }
}

// ── AArch64 target ────────────────────────────────────────────────

/// AArch64 target description for the register allocator.
///
/// Uses the internal calling convention (16 GP arg registers)
/// with callee-saved X19-X28 available for long-lived values.
pub struct AArch64Target;

static GP_REGS: &[PReg] = &[
    PReg(0), PReg(1), PReg(2), PReg(3), PReg(4), PReg(5), PReg(6), PReg(7),
    PReg(8), PReg(9), PReg(10), PReg(11), PReg(12), PReg(13), PReg(14), PReg(15),
    PReg(16), PReg(17), PReg(18), PReg(19), PReg(20), PReg(21), PReg(22), PReg(23),
    PReg(24), PReg(25), PReg(26), PReg(27),
];

static FP_REGS: &[PReg] = &[
    PReg(32), PReg(33), PReg(34), PReg(35), PReg(36), PReg(37), PReg(38), PReg(39),
    PReg(40), PReg(41), PReg(42), PReg(43), PReg(44), PReg(45), PReg(46), PReg(47),
    PReg(48), PReg(49), PReg(50), PReg(51), PReg(52), PReg(53), PReg(54), PReg(55),
    PReg(56), PReg(57), PReg(58), PReg(59), PReg(60), PReg(61), PReg(62), PReg(63),
];

// Caller-saved: X0-X18, D0-D7, D16-D31
static CALLER_SAVED: &[PReg] = &[
    // GP: X0-X18 (X16/X17 = IP0/IP1 linker scratch, X18 = platform register)
    PReg(0), PReg(1), PReg(2), PReg(3), PReg(4), PReg(5), PReg(6), PReg(7),
    PReg(8), PReg(9), PReg(10), PReg(11), PReg(12), PReg(13), PReg(14), PReg(15),
    PReg(16), PReg(17), PReg(18),
    // FP: D0-D7
    PReg(32), PReg(33), PReg(34), PReg(35), PReg(36), PReg(37), PReg(38), PReg(39),
    // FP: D16-D31
    PReg(48), PReg(49), PReg(50), PReg(51), PReg(52), PReg(53), PReg(54), PReg(55),
    PReg(56), PReg(57), PReg(58), PReg(59), PReg(60), PReg(61), PReg(62), PReg(63),
];

// Callee-saved: X19-X28, D8-D15
static CALLEE_SAVED: &[PReg] = &[
    PReg(19), PReg(20), PReg(21), PReg(22), PReg(23), PReg(24), PReg(25), PReg(26),
    PReg(27),
    // FP: D8-D15
    PReg(40), PReg(41), PReg(42), PReg(43), PReg(44), PReg(45), PReg(46), PReg(47),
];

// Reserved: X28 (scratch), X29 (FP), X30 (LR), SP
static RESERVED: &[PReg] = &[
    PReg(28), PReg(29), PReg(30),
];

static GP_ARG_REGS: &[PReg] = &[
    PReg(0), PReg(1), PReg(2), PReg(3), PReg(4), PReg(5), PReg(6), PReg(7),
    PReg(8), PReg(9), PReg(10), PReg(11), PReg(12), PReg(13), PReg(14), PReg(15),
];

static GP_RET_REGS: &[PReg] = &[PReg(0)];
static FP_ARG_REGS: &[PReg] = &[
    PReg(32), PReg(33), PReg(34), PReg(35), PReg(36), PReg(37), PReg(38), PReg(39),
];
static FP_RET_REGS: &[PReg] = &[PReg(32)]; // D0

static REG_CLASSES: &[RegClass] = &[GP, FP];

static GP_REG_NAMES: &[&str] = &[
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30",
];

impl RegInfo for AArch64Target {
    type RegIter<'a> = std::iter::Copied<std::slice::Iter<'a, PReg>>;

    fn reg_classes(&self) -> &[RegClass] {
        REG_CLASSES
    }

    fn class_regs(&self, class: RegClass) -> Self::RegIter<'_> {
        if class == GP {
            GP_REGS.iter().copied()
        } else {
            FP_REGS.iter().copied()
        }
    }

    fn class_size(&self, class: RegClass) -> usize {
        if class == GP { 28 } else { 32 }
    }

    fn reg_class_of(&self, reg: PReg) -> RegClass {
        if reg.0 < 32 { GP } else { FP }
    }

    fn reg_name(&self, reg: PReg) -> &str {
        if reg.0 < 31 {
            GP_REG_NAMES[reg.0 as usize]
        } else if reg.0 >= 32 && reg.0 < 64 {
            // Could have proper D-register names but this suffices
            "d?"
        } else {
            "??"
        }
    }

    fn class_name(&self, class: RegClass) -> &str {
        if class == GP { "GP" } else { "FP" }
    }

    fn spill_size(&self, _class: RegClass) -> u32 {
        8
    }

    fn spill_align(&self, _class: RegClass) -> u32 {
        8
    }
}

impl CallingConvention for AArch64Target {
    fn callee_saved(&self) -> &[PReg] {
        CALLEE_SAVED
    }

    fn caller_saved(&self) -> &[PReg] {
        CALLER_SAVED
    }

    fn arg_regs(&self, class: RegClass) -> &[PReg] {
        if class == GP { GP_ARG_REGS } else { FP_ARG_REGS }
    }

    fn ret_regs(&self, class: RegClass) -> &[PReg] {
        if class == GP { GP_RET_REGS } else { FP_RET_REGS }
    }

    fn stack_pointer(&self) -> Option<PReg> {
        None // SP is implicit, not a numbered PReg
    }

    fn frame_pointer(&self) -> Option<PReg> {
        Some(PReg(29)) // X29
    }

    fn reserved_regs(&self) -> &[PReg] {
        RESERVED
    }
}
