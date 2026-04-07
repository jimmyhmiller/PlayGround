//! Testing infrastructure: mock IR, mock targets, and test generators
//! for exhaustively testing register allocators.

use std::collections::HashMap;

use crate::allocator::{AllocError, Allocation, RegisterAllocator};
use crate::cost::{AllocationCost, CostModel};
use crate::ir::{Function, SafepointAction};
use crate::target::{CallingConvention, RegInfo};
use crate::types::*;
use crate::verify;
use crate::verify::VerifyError;

// ============================================================
// Mock IR: a simple IR for testing allocators
// ============================================================

/// A test instruction.
#[derive(Clone, Debug)]
pub struct TestInst {
    pub id: InstId,
    pub operands: Vec<Operand>,
    pub clobbers: Vec<PReg>,
    pub is_branch: bool,
    pub is_return: bool,
    pub is_call: bool,
    pub is_safepoint: bool,
}

/// A test basic block.
#[derive(Clone, Debug)]
pub struct TestBlock {
    pub id: BlockId,
    pub insts: Vec<InstId>,
    pub succs: Vec<BlockId>,
    pub preds: Vec<BlockId>,
    pub params: Vec<VReg>,
}

/// A test function: the concrete IR that tests build and pass to allocators.
#[derive(Clone, Debug)]
pub struct TestFunction {
    pub blocks: Vec<TestBlock>,
    pub insts: Vec<TestInst>,
    pub num_vregs: usize,
    pub vreg_classes: Vec<RegClass>,
    /// branch_args[inst_id][succ_idx] = vec of vreg args
    pub branch_args: HashMap<(InstId, usize), Vec<VReg>>,
    /// Per-instruction, per-vreg safepoint actions.
    /// Only consulted when `is_safepoint` is true for the instruction.
    pub safepoint_actions: HashMap<(InstId, VReg), SafepointAction>,
}

impl Function for TestFunction {
    type BlockIter<'a> = std::iter::Map<std::slice::Iter<'a, TestBlock>, fn(&TestBlock) -> BlockId>;
    type InstIter<'a> = std::iter::Copied<std::slice::Iter<'a, InstId>>;
    type OperandIter<'a> = std::vec::IntoIter<Operand>;
    type SuccIter<'a> = std::iter::Copied<std::slice::Iter<'a, BlockId>>;
    type PredIter<'a> = std::iter::Copied<std::slice::Iter<'a, BlockId>>;

    fn num_vregs(&self) -> usize {
        self.num_vregs
    }

    fn vreg_class(&self, vreg: VReg) -> RegClass {
        self.vreg_classes[vreg.0 as usize]
    }

    fn blocks(&self) -> Self::BlockIter<'_> {
        self.blocks.iter().map(|b| b.id)
    }

    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry_block(&self) -> BlockId {
        self.blocks[0].id
    }

    fn block_insts(&self, block: BlockId) -> Self::InstIter<'_> {
        self.blocks[block.0 as usize].insts.iter().copied()
    }

    fn block_succs(&self, block: BlockId) -> Self::SuccIter<'_> {
        self.blocks[block.0 as usize].succs.iter().copied()
    }

    fn block_preds(&self, block: BlockId) -> Self::PredIter<'_> {
        self.blocks[block.0 as usize].preds.iter().copied()
    }

    fn inst_operands(&self, inst: InstId) -> Self::OperandIter<'_> {
        self.insts[inst.0 as usize].operands.clone().into_iter()
    }

    fn is_branch(&self, inst: InstId) -> bool {
        self.insts[inst.0 as usize].is_branch
    }

    fn is_return(&self, inst: InstId) -> bool {
        self.insts[inst.0 as usize].is_return
    }

    fn is_call(&self, inst: InstId) -> bool {
        self.insts[inst.0 as usize].is_call
    }

    fn inst_clobbers(&self, inst: InstId) -> &[PReg] {
        &self.insts[inst.0 as usize].clobbers
    }

    fn block_params(&self, block: BlockId) -> &[VReg] {
        &self.blocks[block.0 as usize].params
    }

    fn branch_args(&self, inst: InstId, succ_idx: usize) -> &[VReg] {
        self.branch_args
            .get(&(inst, succ_idx))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    fn is_safepoint(&self, inst: InstId) -> bool {
        self.insts[inst.0 as usize].is_safepoint
    }

    fn safepoint_action(&self, inst: InstId, vreg: VReg) -> SafepointAction {
        self.safepoint_actions
            .get(&(inst, vreg))
            .copied()
            .unwrap_or(SafepointAction::CallingConvention)
    }
}

// ============================================================
// Builder for constructing test functions ergonomically
// ============================================================

/// Builder for constructing test functions.
pub struct TestFunctionBuilder {
    blocks: Vec<TestBlock>,
    insts: Vec<TestInst>,
    num_vregs: u32,
    vreg_classes: Vec<RegClass>,
    branch_args: HashMap<(InstId, usize), Vec<VReg>>,
    safepoint_actions: HashMap<(InstId, VReg), SafepointAction>,
    pub current_block: Option<usize>,
}

impl TestFunctionBuilder {
    pub fn new() -> Self {
        TestFunctionBuilder {
            blocks: Vec::new(),
            insts: Vec::new(),
            num_vregs: 0,
            vreg_classes: Vec::new(),
            branch_args: HashMap::new(),
            safepoint_actions: HashMap::new(),
            current_block: None,
        }
    }

    /// Create a new virtual register in the given class.
    pub fn vreg(&mut self, class: RegClass) -> VReg {
        let id = self.num_vregs;
        self.num_vregs += 1;
        self.vreg_classes.push(class);
        VReg(id)
    }

    /// Start a new basic block. Returns the block ID.
    pub fn block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(TestBlock {
            id,
            insts: Vec::new(),
            succs: Vec::new(),
            preds: Vec::new(),
            params: Vec::new(),
        });
        self.current_block = Some(self.blocks.len() - 1);
        id
    }

    /// Add block parameters (phi destinations).
    pub fn block_params(&mut self, params: &[VReg]) {
        let idx = self.current_block.expect("no current block");
        self.blocks[idx].params = params.to_vec();
    }

    /// Add a normal instruction with the given operands.
    pub fn inst(&mut self, operands: Vec<Operand>) -> InstId {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(TestInst {
            id,
            operands,
            clobbers: Vec::new(),
            is_branch: false,
            is_return: false,
            is_call: false,
            is_safepoint: false,
        });
        let block_idx = self.current_block.expect("no current block");
        self.blocks[block_idx].insts.push(id);
        id
    }

    /// Add a call instruction.
    pub fn call(&mut self, operands: Vec<Operand>, clobbers: Vec<PReg>) -> InstId {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(TestInst {
            id,
            operands,
            clobbers,
            is_branch: false,
            is_return: false,
            is_call: true,
            is_safepoint: false,
        });
        let block_idx = self.current_block.expect("no current block");
        self.blocks[block_idx].insts.push(id);
        id
    }

    /// Add a safepoint call instruction.
    pub fn safepoint_call(&mut self, operands: Vec<Operand>, clobbers: Vec<PReg>) -> InstId {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(TestInst {
            id,
            operands,
            clobbers,
            is_branch: false,
            is_return: false,
            is_call: true,
            is_safepoint: true,
        });
        let block_idx = self.current_block.expect("no current block");
        self.blocks[block_idx].insts.push(id);
        id
    }

    /// Add a return instruction.
    pub fn ret(&mut self, operands: Vec<Operand>) -> InstId {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(TestInst {
            id,
            operands,
            clobbers: Vec::new(),
            is_branch: false,
            is_return: true,
            is_call: false,
            is_safepoint: false,
        });
        let block_idx = self.current_block.expect("no current block");
        self.blocks[block_idx].insts.push(id);
        id
    }

    /// Mark a vreg as needing a specific safepoint action at an instruction.
    pub fn set_safepoint_action(&mut self, inst: InstId, vreg: VReg, action: SafepointAction) {
        self.safepoint_actions.insert((inst, vreg), action);
    }

    /// Add a branch instruction to the given successors.
    pub fn branch(
        &mut self,
        operands: Vec<Operand>,
        succs: Vec<BlockId>,
        args_per_succ: Vec<Vec<VReg>>,
    ) -> InstId {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(TestInst {
            id,
            operands,
            clobbers: Vec::new(),
            is_branch: true,
            is_return: false,
            is_call: false,
            is_safepoint: false,
        });
        let block_idx = self.current_block.expect("no current block");
        self.blocks[block_idx].insts.push(id);

        // Record successors.
        self.blocks[block_idx].succs = succs.clone();

        // Record branch args.
        for (succ_idx, args) in args_per_succ.into_iter().enumerate() {
            self.branch_args.insert((id, succ_idx), args);
        }

        id
    }

    /// Build the function, resolving predecessor edges.
    pub fn build(mut self) -> TestFunction {
        // Compute predecessors from successors.
        let num_blocks = self.blocks.len();
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); num_blocks];
        for block in &self.blocks {
            for &succ in &block.succs {
                preds[succ.0 as usize].push(block.id);
            }
        }
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.preds = preds[i].clone();
        }

        TestFunction {
            blocks: self.blocks,
            insts: self.insts,
            num_vregs: self.num_vregs as usize,
            vreg_classes: self.vreg_classes,
            branch_args: self.branch_args,
            safepoint_actions: self.safepoint_actions,
        }
    }
}

// ============================================================
// Mock target: configurable register file for testing
// ============================================================

/// A configurable test target.
#[derive(Clone, Debug)]
pub struct TestTarget {
    pub classes: Vec<TestRegClass>,
    pub class_ids: Vec<RegClass>,
    pub reserved: Vec<PReg>,
    pub callee_saved: Vec<PReg>,
    pub caller_saved: Vec<PReg>,
    pub arg_regs: HashMap<RegClass, Vec<PReg>>,
    pub ret_regs: HashMap<RegClass, Vec<PReg>>,
    /// Maps PReg -> (class_index, name)
    reg_info: HashMap<PReg, (RegClass, String)>,
}

#[derive(Clone, Debug)]
pub struct TestRegClass {
    pub id: RegClass,
    pub name: String,
    pub regs: Vec<PReg>,
    pub reg_names: Vec<String>,
    pub spill_size: u32,
    pub spill_align: u32,
}

impl TestTarget {
    /// Create a simple target with `n` GPRs named r0..r(n-1).
    pub fn with_gpr(num_regs: u16) -> Self {
        let class = RegClass(0);
        let regs: Vec<PReg> = (0..num_regs).map(PReg).collect();
        let reg_names: Vec<String> = (0..num_regs).map(|i| format!("r{}", i)).collect();

        let mut reg_info = HashMap::new();
        for (i, &r) in regs.iter().enumerate() {
            reg_info.insert(r, (class, reg_names[i].clone()));
        }

        TestTarget {
            classes: vec![TestRegClass {
                id: class,
                name: "GPR".to_string(),
                regs: regs.clone(),
                reg_names,
                spill_size: 8,
                spill_align: 8,
            }],
            class_ids: vec![class],
            reserved: Vec::new(),
            callee_saved: Vec::new(),
            caller_saved: regs,
            arg_regs: HashMap::new(),
            ret_regs: HashMap::new(),
            reg_info,
        }
    }

    /// Create a target with `n_gpr` GPRs and `n_fp` FP registers.
    pub fn with_gpr_and_fp(n_gpr: u16, n_fp: u16) -> Self {
        let gpr_class = RegClass(0);
        let fp_class = RegClass(1);

        let gpr_regs: Vec<PReg> = (0..n_gpr).map(PReg).collect();
        let fp_regs: Vec<PReg> = (n_gpr..n_gpr + n_fp).map(PReg).collect();

        let gpr_names: Vec<String> = (0..n_gpr).map(|i| format!("r{}", i)).collect();
        let fp_names: Vec<String> = (0..n_fp).map(|i| format!("f{}", i)).collect();

        let mut reg_info = HashMap::new();
        for (i, &r) in gpr_regs.iter().enumerate() {
            reg_info.insert(r, (gpr_class, gpr_names[i].clone()));
        }
        for (i, &r) in fp_regs.iter().enumerate() {
            reg_info.insert(r, (fp_class, fp_names[i].clone()));
        }

        let mut all_regs = gpr_regs.clone();
        all_regs.extend_from_slice(&fp_regs);

        TestTarget {
            classes: vec![
                TestRegClass {
                    id: gpr_class,
                    name: "GPR".to_string(),
                    regs: gpr_regs,
                    reg_names: gpr_names,
                    spill_size: 8,
                    spill_align: 8,
                },
                TestRegClass {
                    id: fp_class,
                    name: "FP".to_string(),
                    regs: fp_regs,
                    reg_names: fp_names,
                    spill_size: 8,
                    spill_align: 8,
                },
            ],
            class_ids: vec![gpr_class, fp_class],
            reserved: Vec::new(),
            callee_saved: Vec::new(),
            caller_saved: all_regs,
            arg_regs: HashMap::new(),
            ret_regs: HashMap::new(),
            reg_info,
        }
    }

    /// Set which registers are reserved (unavailable for allocation).
    pub fn reserved(mut self, regs: Vec<PReg>) -> Self {
        self.reserved = regs;
        self
    }

    /// Set callee-saved registers.
    pub fn callee_saved(mut self, regs: Vec<PReg>) -> Self {
        self.callee_saved = regs;
        self
    }

    /// Set caller-saved registers.
    pub fn caller_saved(mut self, regs: Vec<PReg>) -> Self {
        self.caller_saved = regs;
        self
    }

    /// Set argument registers for a class.
    pub fn arg_regs(mut self, class: RegClass, regs: Vec<PReg>) -> Self {
        self.arg_regs.insert(class, regs);
        self
    }

    /// Set return registers for a class.
    pub fn ret_regs(mut self, class: RegClass, regs: Vec<PReg>) -> Self {
        self.ret_regs.insert(class, regs);
        self
    }
}

impl RegInfo for TestTarget {
    type RegIter<'a> = std::iter::Copied<std::slice::Iter<'a, PReg>>;

    fn reg_classes(&self) -> &[RegClass] {
        &self.class_ids
    }

    fn class_regs(&self, class: RegClass) -> Self::RegIter<'_> {
        self.classes[class.0 as usize].regs.iter().copied()
    }

    fn class_size(&self, class: RegClass) -> usize {
        self.classes[class.0 as usize].regs.len()
    }

    fn reg_class_of(&self, reg: PReg) -> RegClass {
        self.reg_info[&reg].0
    }

    fn reg_name(&self, reg: PReg) -> &str {
        &self.reg_info[&reg].1
    }

    fn class_name(&self, class: RegClass) -> &str {
        &self.classes[class.0 as usize].name
    }

    fn spill_size(&self, class: RegClass) -> u32 {
        self.classes[class.0 as usize].spill_size
    }

    fn spill_align(&self, class: RegClass) -> u32 {
        self.classes[class.0 as usize].spill_align
    }
}

impl CallingConvention for TestTarget {
    fn callee_saved(&self) -> &[PReg] {
        &self.callee_saved
    }

    fn caller_saved(&self) -> &[PReg] {
        &self.caller_saved
    }

    fn arg_regs(&self, class: RegClass) -> &[PReg] {
        self.arg_regs.get(&class).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn ret_regs(&self, class: RegClass) -> &[PReg] {
        self.ret_regs.get(&class).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn reserved_regs(&self) -> &[PReg] {
        &self.reserved
    }
}

// ============================================================
// Helpers for building operands concisely
// ============================================================

/// Shorthand for a "use" operand (read a vreg, any register in its class).
pub fn use_reg(vreg: VReg, class: RegClass) -> Operand {
    Operand {
        reg: Reg::Virtual(vreg),
        kind: OperandKind::Use,
        constraint: OperandConstraint::RegClass(class),
    }
}

/// Shorthand for a "def" operand (write a vreg, any register in its class).
pub fn def_reg(vreg: VReg, class: RegClass) -> Operand {
    Operand {
        reg: Reg::Virtual(vreg),
        kind: OperandKind::Def,
        constraint: OperandConstraint::RegClass(class),
    }
}

/// Shorthand for a "use" pinned to a specific physical register.
pub fn use_fixed(vreg: VReg, preg: PReg) -> Operand {
    Operand {
        reg: Reg::Virtual(vreg),
        kind: OperandKind::Use,
        constraint: OperandConstraint::FixedReg(preg),
    }
}

/// Shorthand for a "def" pinned to a specific physical register.
pub fn def_fixed(vreg: VReg, preg: PReg) -> Operand {
    Operand {
        reg: Reg::Virtual(vreg),
        kind: OperandKind::Def,
        constraint: OperandConstraint::FixedReg(preg),
    }
}

/// Shorthand for a "use-def" tied to operand at given index.
pub fn usedef_tied(vreg: VReg, tied_to: usize) -> Operand {
    Operand {
        reg: Reg::Virtual(vreg),
        kind: OperandKind::UseDef,
        constraint: OperandConstraint::Tied(tied_to),
    }
}

/// Shorthand for a "def" that reuses the allocation of the input at the given index.
pub fn def_reuse(vreg: VReg, reuse_of: usize) -> Operand {
    Operand {
        reg: Reg::Virtual(vreg),
        kind: OperandKind::Def,
        constraint: OperandConstraint::Reuse(reuse_of),
    }
}

/// Shorthand for an early def.
pub fn early_def(vreg: VReg, class: RegClass) -> Operand {
    Operand {
        reg: Reg::Virtual(vreg),
        kind: OperandKind::EarlyDef,
        constraint: OperandConstraint::RegClass(class),
    }
}

// ============================================================
// Test harness: run allocator, verify, and optionally score
// ============================================================

/// Result of running a test case.
#[derive(Debug)]
pub struct TestResult {
    pub allocator_name: String,
    pub allocation: Option<Allocation>,
    pub error: Option<AllocError>,
    pub verification_errors: Vec<VerifyError>,
    pub cost: Option<AllocationCost>,
}

impl TestResult {
    pub fn is_ok(&self) -> bool {
        self.allocation.is_some() && self.verification_errors.is_empty()
    }
}

/// Run an allocator on a test case, verify the result, and optionally score it.
pub fn run_test<A: RegisterAllocator, C: CostModel>(
    allocator: &mut A,
    func: &TestFunction,
    target: &TestTarget,
    cost_model: Option<&C>,
) -> TestResult {
    let name = allocator.name().to_string();

    match allocator.allocate(func, target) {
        Ok(alloc) => {
            let verify_result = verify::verify(func, target, &alloc);
            let verification_errors = verify_result.err().unwrap_or_default();
            let cost = cost_model.map(|cm| cm.evaluate(func, target, &alloc));

            TestResult {
                allocator_name: name,
                allocation: Some(alloc),
                error: None,
                verification_errors,
                cost,
            }
        }
        Err(e) => TestResult {
            allocator_name: name,
            allocation: None,
            error: Some(e),
            verification_errors: Vec::new(),
            cost: None,
        },
    }
}

/// Compare two allocators on the same test case.
pub fn compare_allocators<A1, A2, C>(
    alloc1: &mut A1,
    alloc2: &mut A2,
    func: &TestFunction,
    target: &TestTarget,
    cost_model: &C,
) -> (TestResult, TestResult)
where
    A1: RegisterAllocator,
    A2: RegisterAllocator,
    C: CostModel,
{
    let r1 = run_test(alloc1, func, target, Some(cost_model));
    let r2 = run_test(alloc2, func, target, Some(cost_model));
    (r1, r2)
}

// ============================================================
// Pre-built test cases
// ============================================================

/// A collection of standard test programs that exercise various
/// register allocation challenges.
pub struct TestSuite;

impl TestSuite {
    /// Single block, no pressure: just a sequence of independent operations.
    /// Should never require spilling.
    pub fn straight_line_no_pressure() -> TestFunction {
        let gpr = RegClass(0);
        let mut b = TestFunctionBuilder::new();
        let v0 = b.vreg(gpr);
        let v1 = b.vreg(gpr);
        let v2 = b.vreg(gpr);

        let _bb0 = b.block();
        // v0 = def
        b.inst(vec![def_reg(v0, gpr)]);
        // v1 = def
        b.inst(vec![def_reg(v1, gpr)]);
        // v2 = op(v0, v1)
        b.inst(vec![def_reg(v2, gpr), use_reg(v0, gpr), use_reg(v1, gpr)]);
        // return v2
        b.ret(vec![use_reg(v2, gpr)]);

        b.build()
    }

    /// Register pressure: more live values than registers (requires spilling).
    ///
    /// Creates N values defined in sequence, then used in sequence.
    /// All N values are simultaneously live between the last def and
    /// the first use, creating maximum register pressure of N.
    /// Each instruction has at most 1 operand, so spill code always
    /// has temp registers available.
    pub fn high_pressure(num_values: u32) -> TestFunction {
        let gpr = RegClass(0);
        let mut b = TestFunctionBuilder::new();
        let mut vregs = Vec::new();
        for _ in 0..num_values {
            vregs.push(b.vreg(gpr));
        }

        let _bb0 = b.block();
        // Define all values in sequence.
        for &v in &vregs {
            b.inst(vec![def_reg(v, gpr)]);
        }
        // Use all values in separate instructions.
        // After `use vi`, vi is dead, reducing pressure by 1 each time.
        for &v in &vregs {
            b.inst(vec![use_reg(v, gpr)]);
        }
        b.ret(vec![]);

        b.build()
    }

    /// Diamond CFG: tests phi resolution / block-edge moves.
    ///
    /// ```text
    ///       bb0
    ///      /   \
    ///    bb1   bb2
    ///      \   /
    ///       bb3
    /// ```
    pub fn diamond() -> TestFunction {
        let gpr = RegClass(0);
        let mut b = TestFunctionBuilder::new();
        let v_cond = b.vreg(gpr);
        let v_left = b.vreg(gpr);
        let v_right = b.vreg(gpr);
        let v_result = b.vreg(gpr);

        let bb0 = b.block();
        let bb1 = b.block();
        let bb2 = b.block();
        let bb3 = b.block();

        // bb0: branch to bb1 or bb2
        b.current_block = Some(0);
        b.inst(vec![def_reg(v_cond, gpr)]);
        b.branch(
            vec![use_reg(v_cond, gpr)],
            vec![bb1, bb2],
            vec![vec![], vec![]],
        );

        // bb1: define left value, jump to bb3
        b.current_block = Some(1);
        b.inst(vec![def_reg(v_left, gpr)]);
        b.branch(vec![], vec![bb3], vec![vec![v_left]]);

        // bb2: define right value, jump to bb3
        b.current_block = Some(2);
        b.inst(vec![def_reg(v_right, gpr)]);
        b.branch(vec![], vec![bb3], vec![vec![v_right]]);

        // bb3: phi(v_left, v_right) -> v_result, then return
        b.current_block = Some(3);
        b.block_params(&[v_result]);
        b.ret(vec![use_reg(v_result, gpr)]);

        // Need to fix: block was already created, we set params on bb3.
        // Let's rebuild bb3 properly.
        let _ = (bb0, bb1, bb2, bb3);
        b.build()
    }

    /// Loop: tests live ranges that span back-edges.
    ///
    /// ```text
    ///   bb0 -> bb1 -> bb2
    ///           ^      |
    ///           +------+
    /// ```
    pub fn simple_loop() -> TestFunction {
        let gpr = RegClass(0);
        let mut b = TestFunctionBuilder::new();
        let v_init = b.vreg(gpr);
        let v_i = b.vreg(gpr);
        let v_next = b.vreg(gpr);
        let v_cond = b.vreg(gpr);

        let bb0 = b.block();
        let bb1 = b.block();
        let bb2 = b.block();
        let bb3 = b.block();

        // bb0: init, jump to loop header
        b.current_block = Some(0);
        b.inst(vec![def_reg(v_init, gpr)]);
        b.branch(vec![], vec![bb1], vec![vec![v_init]]);

        // bb1 (loop header): phi(v_init from bb0, v_next from bb2)
        b.current_block = Some(1);
        b.block_params(&[v_i]);
        b.inst(vec![def_reg(v_cond, gpr), use_reg(v_i, gpr)]);
        b.branch(vec![use_reg(v_cond, gpr)], vec![bb2, bb3], vec![vec![], vec![]]);

        // bb2 (loop body): compute next, branch back
        b.current_block = Some(2);
        b.inst(vec![def_reg(v_next, gpr), use_reg(v_i, gpr)]);
        b.branch(vec![], vec![bb1], vec![vec![v_next]]);

        // bb3 (exit): return
        b.current_block = Some(3);
        b.ret(vec![use_reg(v_i, gpr)]);

        let _ = (bb0, bb1, bb2, bb3);
        b.build()
    }

    /// Fixed register constraints: like x86 `idiv` which needs
    /// specific registers.
    pub fn fixed_constraints() -> TestFunction {
        let gpr = RegClass(0);
        let r0 = PReg(0); // like rax
        let r2 = PReg(2); // like rdx

        let mut b = TestFunctionBuilder::new();
        let v_dividend = b.vreg(gpr);
        let v_divisor = b.vreg(gpr);
        let v_quotient = b.vreg(gpr);
        let v_remainder = b.vreg(gpr);

        let _bb0 = b.block();
        b.inst(vec![def_reg(v_dividend, gpr)]);
        b.inst(vec![def_reg(v_divisor, gpr)]);
        // idiv: dividend must be in r0, quotient comes out in r0, remainder in r2
        b.inst(vec![
            def_fixed(v_quotient, r0),
            def_fixed(v_remainder, r2),
            use_fixed(v_dividend, r0),
            use_reg(v_divisor, gpr),
        ]);
        // Use both results to keep them live.
        b.inst(vec![use_reg(v_quotient, gpr), use_reg(v_remainder, gpr)]);
        b.ret(vec![use_reg(v_quotient, gpr)]);

        b.build()
    }

    /// Call site: tests caller-saved register handling.
    pub fn with_call() -> TestFunction {
        let gpr = RegClass(0);
        let mut b = TestFunctionBuilder::new();
        let v_before = b.vreg(gpr);
        let v_arg = b.vreg(gpr);
        let v_result = b.vreg(gpr);

        let _bb0 = b.block();
        // Define a value that must survive across the call.
        b.inst(vec![def_reg(v_before, gpr)]);
        // Prepare call argument.
        b.inst(vec![def_reg(v_arg, gpr), use_reg(v_before, gpr)]);
        // Call: clobbers r0, r1, r2 (all caller-saved).
        b.call(
            vec![def_reg(v_result, gpr), use_reg(v_arg, gpr)],
            vec![PReg(0), PReg(1), PReg(2)],
        );
        // Use both the call result and the value from before the call.
        b.inst(vec![use_reg(v_result, gpr), use_reg(v_before, gpr)]);
        b.ret(vec![use_reg(v_result, gpr)]);

        b.build()
    }

    /// Two-address instruction (like x86 `add dst, src` where dst is also read).
    pub fn two_address() -> TestFunction {
        let gpr = RegClass(0);
        let mut b = TestFunctionBuilder::new();
        let v0 = b.vreg(gpr);
        let v1 = b.vreg(gpr);
        let v2 = b.vreg(gpr);

        let _bb0 = b.block();
        b.inst(vec![def_reg(v0, gpr)]);
        b.inst(vec![def_reg(v1, gpr)]);
        // v2 = add v0, v1  -- but v2 must be in same reg as v0 (tied)
        b.inst(vec![
            def_reuse(v2, 1), // v2 reuses v0's register
            use_reg(v0, gpr),
            use_reg(v1, gpr),
        ]);
        b.ret(vec![use_reg(v2, gpr)]);

        b.build()
    }

    /// Early def: the output register must not overlap any input register.
    pub fn early_def_test() -> TestFunction {
        let gpr = RegClass(0);
        let mut b = TestFunctionBuilder::new();
        let v0 = b.vreg(gpr);
        let v1 = b.vreg(gpr);
        let v2 = b.vreg(gpr);

        let _bb0 = b.block();
        b.inst(vec![def_reg(v0, gpr)]);
        b.inst(vec![def_reg(v1, gpr)]);
        // v2 = early_def(v0, v1) -- v2 can't share a register with v0 or v1
        b.inst(vec![
            early_def(v2, gpr),
            use_reg(v0, gpr),
            use_reg(v1, gpr),
        ]);
        b.ret(vec![use_reg(v2, gpr)]);

        b.build()
    }
}
