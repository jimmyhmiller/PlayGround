//! Generative tests for register allocation using a simple VM.
//!
//! This module creates a simple stack-based VM with a few registers,
//! then generates random programs, compiles them through the full
//! SSA → optimization → register allocation pipeline, and verifies
//! that execution produces correct results.

use std::collections::HashMap;

use proptest::prelude::*;
use ssa_test::optim::regalloc::{
    LinearScanAllocator, LinearScanConfig, IntervalAnalysis, Location,
    PhysicalRegister, RegisterClass, TargetArchitecture, PhiElimination,
};
use ssa_test::optim::analysis::LivenessAnalysis;
use ssa_test::optim::traits::ExpressionKey;
use ssa_test::traits::{InstructionFactory, SsaInstruction, SsaValue};
use ssa_test::optim::traits::{OptimizableInstruction, OptimizableValue};
use ssa_test::translator::SSATranslator;
use ssa_test::types::{PhiId, SsaVariable};

// ============================================================================
// Simple VM Definition
// ============================================================================

/// Physical registers for our simple VM (6 general purpose registers).
/// R0-R3 are allocatable:
///   - R0, R1: caller-saved (clobbered by calls)
///   - R2, R3: callee-saved (preserved across calls)
/// R4, R5 are reserved as scratch registers for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    R0,
    R1,
    R2,
    R3,
    R4,  // Scratch register for codegen
    R5,  // Scratch register for codegen
}

impl PhysicalRegister for Reg {
    fn id(&self) -> usize {
        match self {
            Reg::R0 => 0,
            Reg::R1 => 1,
            Reg::R2 => 2,
            Reg::R3 => 3,
            Reg::R4 => 4,
            Reg::R5 => 5,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Reg::R0 => "r0",
            Reg::R1 => "r1",
            Reg::R2 => "r2",
            Reg::R3 => "r3",
            Reg::R4 => "r4",
            Reg::R5 => "r5",
        }
    }
}

impl Reg {
    fn from_id(id: usize) -> Option<Reg> {
        match id {
            0 => Some(Reg::R0),
            1 => Some(Reg::R1),
            2 => Some(Reg::R2),
            3 => Some(Reg::R3),
            4 => Some(Reg::R4),
            5 => Some(Reg::R5),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    GP,
}

impl RegisterClass for RegClass {
    type Register = Reg;

    fn name(&self) -> &'static str {
        "gp"
    }

    fn allocatable_registers(&self) -> &'static [Reg] {
        // All 4 registers are allocatable.
        // R0, R1 are caller-saved (clobbered by calls)
        // R2, R3 are callee-saved (preserved across calls)
        &[Reg::R0, Reg::R1, Reg::R2, Reg::R3]
    }
}

#[derive(Debug, Clone)]
pub struct SimpleArch;

impl TargetArchitecture for SimpleArch {
    type Register = Reg;
    type Class = RegClass;

    fn register_classes(&self) -> &'static [RegClass] {
        &[RegClass::GP]
    }

    fn default_class(&self) -> RegClass {
        RegClass::GP
    }

    fn stack_slot_size(&self) -> usize {
        8
    }

    fn caller_saved(&self) -> &'static [Reg] {
        // R0 and R1 are clobbered by calls
        &[Reg::R0, Reg::R1]
    }

    fn callee_saved(&self) -> &'static [Reg] {
        // R2 and R3 are preserved across calls
        &[Reg::R2, Reg::R3]
    }
}

/// VM instruction set
#[derive(Debug, Clone)]
pub enum VmInstr {
    /// Load immediate: reg := imm
    LoadImm { dest: Reg, value: i64 },
    /// Move: dest := src
    Move { dest: Reg, src: Reg },
    /// Add: dest := left + right
    Add { dest: Reg, left: Reg, right: Reg },
    /// Sub: dest := left - right
    Sub { dest: Reg, left: Reg, right: Reg },
    /// Mul: dest := left * right
    Mul { dest: Reg, left: Reg, right: Reg },
    /// Negate: dest := -src
    Neg { dest: Reg, src: Reg },
    /// Spill: store reg to stack slot
    Spill { src: Reg, slot: usize },
    /// Reload: load from stack slot to reg
    Reload { dest: Reg, slot: usize },
    /// Call: simulates a function call that clobbers caller-saved registers (R0, R1)
    /// and returns a value in dest.
    Call { dest: Reg, return_value: i64 },
    /// No-op (placeholder)
    Nop,
}

/// Simple VM state
#[derive(Debug, Clone)]
pub struct Vm {
    /// Register file (6 registers: R0-R5)
    pub regs: [i64; 6],
    /// Stack (for spills)
    pub stack: Vec<i64>,
    /// Program counter
    pub pc: usize,
}

impl Vm {
    pub fn new(stack_slots: usize) -> Self {
        Vm {
            regs: [0; 6],
            stack: vec![0; stack_slots],
            pc: 0,
        }
    }

    /// Execute a single instruction
    pub fn step(&mut self, instr: &VmInstr) {
        match instr {
            VmInstr::LoadImm { dest, value } => {
                self.regs[dest.id()] = *value;
            }
            VmInstr::Move { dest, src } => {
                self.regs[dest.id()] = self.regs[src.id()];
            }
            VmInstr::Add { dest, left, right } => {
                self.regs[dest.id()] = self.regs[left.id()].wrapping_add(self.regs[right.id()]);
            }
            VmInstr::Sub { dest, left, right } => {
                self.regs[dest.id()] = self.regs[left.id()].wrapping_sub(self.regs[right.id()]);
            }
            VmInstr::Mul { dest, left, right } => {
                self.regs[dest.id()] = self.regs[left.id()].wrapping_mul(self.regs[right.id()]);
            }
            VmInstr::Neg { dest, src } => {
                self.regs[dest.id()] = self.regs[src.id()].wrapping_neg();
            }
            VmInstr::Spill { src, slot } => {
                if *slot < self.stack.len() {
                    self.stack[*slot] = self.regs[src.id()];
                }
            }
            VmInstr::Reload { dest, slot } => {
                if *slot < self.stack.len() {
                    self.regs[dest.id()] = self.stack[*slot];
                }
            }
            VmInstr::Call { dest, return_value } => {
                // Clobber caller-saved registers (R0, R1) with garbage
                self.regs[Reg::R0.id()] = 0xDEAD_BEEF_i64;
                self.regs[Reg::R1.id()] = 0xCAFE_BABE_i64;
                // Set the return value in the destination register
                self.regs[dest.id()] = *return_value;
            }
            VmInstr::Nop => {}
        }
        self.pc += 1;
    }

    /// Execute a program
    pub fn run(&mut self, program: &[VmInstr]) {
        for instr in program {
            self.step(instr);
        }
    }

    /// Get register value
    pub fn get_reg(&self, reg: Reg) -> i64 {
        self.regs[reg.id()]
    }
}

// ============================================================================
// SSA IR for testing
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TestValue {
    Literal(i64),
    Var(SsaVariable),
    Phi(PhiId),
    Undefined,
}

impl SsaValue for TestValue {
    fn from_phi(id: PhiId) -> Self {
        TestValue::Phi(id)
    }

    fn from_var(var: SsaVariable) -> Self {
        TestValue::Var(var)
    }

    fn undefined() -> Self {
        TestValue::Undefined
    }

    fn as_phi(&self) -> Option<PhiId> {
        match self {
            TestValue::Phi(id) => Some(*id),
            _ => None,
        }
    }

    fn as_var(&self) -> Option<&SsaVariable> {
        match self {
            TestValue::Var(v) => Some(v),
            _ => None,
        }
    }

    fn is_undefined(&self) -> bool {
        matches!(self, TestValue::Undefined)
    }
}

impl OptimizableValue for TestValue {
    type Constant = i64;

    fn as_constant(&self) -> Option<&i64> {
        match self {
            TestValue::Literal(n) => Some(n),
            _ => None,
        }
    }

    fn from_constant(c: i64) -> Self {
        TestValue::Literal(c)
    }

    fn is_constant(&self) -> bool {
        matches!(self, TestValue::Literal(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone)]
pub enum TestInstr {
    /// dest := value
    Assign { dest: SsaVariable, value: TestValue },
    /// dest := left op right
    BinOp { dest: SsaVariable, left: TestValue, op: BinOp, right: TestValue },
    /// dest := -operand
    Neg { dest: SsaVariable, operand: TestValue },
    /// dest := phi
    PhiAssign { dest: SsaVariable, phi_id: PhiId },
    /// Copy: dest := src
    Copy { dest: SsaVariable, src: TestValue },
    /// dest := call(return_value) - simulates a function call that clobbers caller-saved regs
    Call { dest: SsaVariable, return_value: i64 },
}

impl SsaInstruction for TestInstr {
    type Value = TestValue;

    fn visit_values<F: FnMut(&Self::Value)>(&self, mut visitor: F) {
        match self {
            TestInstr::Assign { value, .. } => visitor(value),
            TestInstr::BinOp { left, right, .. } => {
                visitor(left);
                visitor(right);
            }
            TestInstr::Neg { operand, .. } => visitor(operand),
            TestInstr::PhiAssign { .. } => {}
            TestInstr::Copy { src, .. } => visitor(src),
            TestInstr::Call { .. } => {} // No value operands
        }
    }

    fn visit_values_mut<F: FnMut(&mut Self::Value)>(&mut self, mut visitor: F) {
        match self {
            TestInstr::Assign { value, .. } => visitor(value),
            TestInstr::BinOp { left, right, .. } => {
                visitor(left);
                visitor(right);
            }
            TestInstr::Neg { operand, .. } => visitor(operand),
            TestInstr::PhiAssign { .. } => {}
            TestInstr::Copy { src, .. } => visitor(src),
            TestInstr::Call { .. } => {} // No value operands
        }
    }

    fn destination(&self) -> Option<&SsaVariable> {
        match self {
            TestInstr::Assign { dest, .. }
            | TestInstr::BinOp { dest, .. }
            | TestInstr::Neg { dest, .. }
            | TestInstr::PhiAssign { dest, .. }
            | TestInstr::Copy { dest, .. }
            | TestInstr::Call { dest, .. } => Some(dest),
        }
    }

    fn is_phi_assignment(&self) -> bool {
        matches!(self, TestInstr::PhiAssign { .. })
    }

    fn get_phi_assignment(&self) -> Option<PhiId> {
        match self {
            TestInstr::PhiAssign { phi_id, .. } => Some(*phi_id),
            _ => None,
        }
    }
}

impl OptimizableInstruction for TestInstr {
    fn has_side_effects(&self) -> bool {
        // Calls have side effects (they clobber registers)
        matches!(self, TestInstr::Call { .. })
    }

    fn is_terminator(&self) -> bool {
        false
    }

    fn as_copy(&self) -> Option<(&SsaVariable, &TestValue)> {
        match self {
            TestInstr::Copy { dest, src } => Some((dest, src)),
            TestInstr::Assign { dest, value } if value.as_var().is_some() => Some((dest, value)),
            _ => None,
        }
    }

    fn try_fold(&self) -> Option<i64> {
        match self {
            TestInstr::Assign { value: TestValue::Literal(n), .. } => Some(*n),
            TestInstr::BinOp { left: TestValue::Literal(l), op, right: TestValue::Literal(r), .. } => {
                Some(match op {
                    BinOp::Add => l.wrapping_add(*r),
                    BinOp::Sub => l.wrapping_sub(*r),
                    BinOp::Mul => l.wrapping_mul(*r),
                })
            }
            TestInstr::Neg { operand: TestValue::Literal(n), .. } => Some(n.wrapping_neg()),
            TestInstr::Call { return_value, .. } => Some(*return_value),
            _ => None,
        }
    }

    fn expression_key(&self) -> Option<ExpressionKey<TestValue>> {
        None
    }
}

pub struct TestFactory;

impl InstructionFactory for TestFactory {
    type Instr = TestInstr;

    fn create_phi_assign(dest: SsaVariable, phi_id: PhiId) -> TestInstr {
        TestInstr::PhiAssign { dest, phi_id }
    }

    fn create_copy(dest: SsaVariable, value: TestValue) -> TestInstr {
        TestInstr::Copy { dest, src: value }
    }
}

// ============================================================================
// Code Generation
// ============================================================================

/// Generate VM instructions from SSA after register allocation (v2 - respects allocation).
///
/// This version properly uses the allocated registers and spill slots.
/// It uses R4 and R5 as scratch registers for binary operations to avoid
/// clobbering allocated values in R0-R3.
pub fn generate_vm_code_v2(
    translator: &SSATranslator<TestValue, TestInstr, TestFactory>,
    assignments: &HashMap<SsaVariable, Location>,
) -> (Vec<VmInstr>, HashMap<SsaVariable, Location>) {
    let mut code = Vec::new();

    for block in &translator.blocks {
        for instr in &block.instructions {
            match instr {
                TestInstr::Assign { dest, value } => {
                    let dest_loc = assignments.get(dest).cloned().unwrap_or(Location::Register(0));

                    // Load value into the destination location
                    match dest_loc {
                        Location::Register(id) => {
                            let dest_reg = Reg::from_id(id).unwrap_or(Reg::R0);
                            emit_load_value_v2(&mut code, value, assignments, dest_reg);
                        }
                        Location::StackSlot(slot) => {
                            // Load into scratch register, then spill
                            emit_load_value_v2(&mut code, value, assignments, Reg::R5);
                            code.push(VmInstr::Spill { src: Reg::R5, slot });
                        }
                    }
                }
                TestInstr::BinOp { dest, left, op, right } => {
                    let dest_loc = assignments.get(dest).cloned().unwrap_or(Location::Register(0));

                    // Use R4 and R5 as scratch registers for operands
                    emit_load_value_v2(&mut code, left, assignments, Reg::R4);
                    emit_load_value_v2(&mut code, right, assignments, Reg::R5);

                    // Compute result into R4
                    match op {
                        BinOp::Add => code.push(VmInstr::Add { dest: Reg::R4, left: Reg::R4, right: Reg::R5 }),
                        BinOp::Sub => code.push(VmInstr::Sub { dest: Reg::R4, left: Reg::R4, right: Reg::R5 }),
                        BinOp::Mul => code.push(VmInstr::Mul { dest: Reg::R4, left: Reg::R4, right: Reg::R5 }),
                    }

                    // Store result to destination
                    emit_store_to_loc_v2(&mut code, Reg::R4, dest_loc);
                }
                TestInstr::Neg { dest, operand } => {
                    let dest_loc = assignments.get(dest).cloned().unwrap_or(Location::Register(0));

                    emit_load_value_v2(&mut code, operand, assignments, Reg::R4);
                    code.push(VmInstr::Neg { dest: Reg::R4, src: Reg::R4 });
                    emit_store_to_loc_v2(&mut code, Reg::R4, dest_loc);
                }
                TestInstr::Copy { dest, src } => {
                    let dest_loc = assignments.get(dest).cloned().unwrap_or(Location::Register(0));

                    // Try to do a direct move if both are registers
                    let src_loc = src.as_var().and_then(|v| assignments.get(v).cloned());

                    match (src_loc, dest_loc) {
                        (Some(Location::Register(s)), Location::Register(d)) if s == d => {
                            // Same register, no-op
                        }
                        (Some(Location::Register(s)), Location::Register(d)) => {
                            let src_reg = Reg::from_id(s).unwrap_or(Reg::R0);
                            let dest_reg = Reg::from_id(d).unwrap_or(Reg::R0);
                            code.push(VmInstr::Move { dest: dest_reg, src: src_reg });
                        }
                        _ => {
                            // Use scratch register
                            emit_load_value_v2(&mut code, src, assignments, Reg::R4);
                            emit_store_to_loc_v2(&mut code, Reg::R4, dest_loc);
                        }
                    }
                }
                TestInstr::PhiAssign { .. } => {
                    // Phis should be eliminated before codegen
                }
                TestInstr::Call { dest, return_value } => {
                    let dest_loc = assignments.get(dest).cloned().unwrap_or(Location::Register(0));

                    // Emit a call that returns the specified value.
                    // The call will clobber R0 and R1 (caller-saved registers).
                    // The result goes into R0 initially (calling convention).
                    code.push(VmInstr::Call { dest: Reg::R0, return_value: *return_value });

                    // Store result to destination
                    emit_store_to_loc_v2(&mut code, Reg::R0, dest_loc);
                }
            }
        }
    }

    (code, assignments.clone())
}

fn emit_load_value_v2(
    code: &mut Vec<VmInstr>,
    value: &TestValue,
    assignments: &HashMap<SsaVariable, Location>,
    dest: Reg,
) {
    match value {
        TestValue::Literal(n) => {
            code.push(VmInstr::LoadImm { dest, value: *n });
        }
        TestValue::Var(var) => {
            match assignments.get(var) {
                Some(Location::Register(id)) => {
                    if let Some(src) = Reg::from_id(*id) {
                        if src != dest {
                            code.push(VmInstr::Move { dest, src });
                        }
                        // If src == dest, no instruction needed
                    }
                }
                Some(Location::StackSlot(slot)) => {
                    code.push(VmInstr::Reload { dest, slot: *slot });
                }
                None => {
                    code.push(VmInstr::LoadImm { dest, value: 0 });
                }
            }
        }
        TestValue::Phi(_) | TestValue::Undefined => {
            code.push(VmInstr::LoadImm { dest, value: 0 });
        }
    }
}

fn emit_store_to_loc_v2(code: &mut Vec<VmInstr>, src: Reg, loc: Location) {
    match loc {
        Location::Register(id) => {
            if let Some(dest) = Reg::from_id(id) {
                if dest != src {
                    code.push(VmInstr::Move { dest, src });
                }
            }
        }
        Location::StackSlot(slot) => {
            code.push(VmInstr::Spill { src, slot });
        }
    }
}

// ============================================================================
// Interpreter for reference results
// ============================================================================

/// Interpret SSA directly to get expected results (no register allocation).
pub fn interpret_ssa(
    translator: &SSATranslator<TestValue, TestInstr, TestFactory>,
) -> HashMap<SsaVariable, i64> {
    let mut env: HashMap<SsaVariable, i64> = HashMap::new();

    // Handle phi nodes
    for phi in translator.phis.values() {
        if let Some(dest) = &phi.dest {
            // For straight-line code, just take first operand
            let value = phi.operands.first()
                .map(|op| eval_value(op, &env))
                .unwrap_or(0);
            env.insert(dest.clone(), value);
        }
    }

    for block in &translator.blocks {
        for instr in &block.instructions {
            match instr {
                TestInstr::Assign { dest, value } => {
                    env.insert(dest.clone(), eval_value(value, &env));
                }
                TestInstr::BinOp { dest, left, op, right } => {
                    let l = eval_value(left, &env);
                    let r = eval_value(right, &env);
                    let result = match op {
                        BinOp::Add => l.wrapping_add(r),
                        BinOp::Sub => l.wrapping_sub(r),
                        BinOp::Mul => l.wrapping_mul(r),
                    };
                    env.insert(dest.clone(), result);
                }
                TestInstr::Neg { dest, operand } => {
                    env.insert(dest.clone(), eval_value(operand, &env).wrapping_neg());
                }
                TestInstr::Copy { dest, src } => {
                    env.insert(dest.clone(), eval_value(src, &env));
                }
                TestInstr::PhiAssign { dest, phi_id } => {
                    if let Some(phi) = translator.phis.get(phi_id) {
                        let value = phi.operands.first()
                            .map(|op| eval_value(op, &env))
                            .unwrap_or(0);
                        env.insert(dest.clone(), value);
                    }
                }
                TestInstr::Call { dest, return_value } => {
                    env.insert(dest.clone(), *return_value);
                }
            }
        }
    }

    env
}

fn eval_value(value: &TestValue, env: &HashMap<SsaVariable, i64>) -> i64 {
    match value {
        TestValue::Literal(n) => *n,
        TestValue::Var(var) => env.get(var).copied().unwrap_or(0),
        TestValue::Phi(_) | TestValue::Undefined => 0,
    }
}

// ============================================================================
// Test Program Generation
// ============================================================================

/// A simple straight-line program specification.
#[derive(Debug, Clone)]
pub struct TestProgram {
    pub instructions: Vec<TestProgramInstr>,
}

#[derive(Debug, Clone)]
pub enum TestProgramInstr {
    Const { var_id: usize, value: i64 },
    BinOp { var_id: usize, left_id: usize, op: BinOp, right_id: usize },
    Neg { var_id: usize, src_id: usize },
    Copy { var_id: usize, src_id: usize },
    /// Simulates a function call that returns a value and clobbers caller-saved registers
    Call { var_id: usize, return_value: i64 },
}

impl TestProgram {
    /// Build an SSA translator from this program.
    pub fn to_ssa(&self) -> SSATranslator<TestValue, TestInstr, TestFactory> {
        let mut translator = SSATranslator::<TestValue, TestInstr, TestFactory>::new();
        let block_id = translator.current_block;
        translator.seal_block(block_id);

        for instr in &self.instructions {
            match instr {
                TestProgramInstr::Const { var_id, value } => {
                    let dest = SsaVariable::new(&format!("v{}", var_id));
                    translator.emit(TestInstr::Assign {
                        dest: dest.clone(),
                        value: TestValue::Literal(*value),
                    });
                    translator.write_variable(format!("v{}", var_id), block_id, TestValue::Var(dest));
                }
                TestProgramInstr::BinOp { var_id, left_id, op, right_id } => {
                    let dest = SsaVariable::new(&format!("v{}", var_id));
                    let left_val = translator.read_variable(format!("v{}", left_id), block_id);
                    let right_val = translator.read_variable(format!("v{}", right_id), block_id);
                    translator.emit(TestInstr::BinOp {
                        dest: dest.clone(),
                        left: left_val,
                        op: *op,
                        right: right_val,
                    });
                    translator.write_variable(format!("v{}", var_id), block_id, TestValue::Var(dest));
                }
                TestProgramInstr::Neg { var_id, src_id } => {
                    let dest = SsaVariable::new(&format!("v{}", var_id));
                    let src_val = translator.read_variable(format!("v{}", src_id), block_id);
                    translator.emit(TestInstr::Neg {
                        dest: dest.clone(),
                        operand: src_val,
                    });
                    translator.write_variable(format!("v{}", var_id), block_id, TestValue::Var(dest));
                }
                TestProgramInstr::Copy { var_id, src_id } => {
                    let dest = SsaVariable::new(&format!("v{}", var_id));
                    let src_val = translator.read_variable(format!("v{}", src_id), block_id);
                    translator.emit(TestInstr::Copy {
                        dest: dest.clone(),
                        src: src_val,
                    });
                    translator.write_variable(format!("v{}", var_id), block_id, TestValue::Var(dest));
                }
                TestProgramInstr::Call { var_id, return_value } => {
                    let dest = SsaVariable::new(&format!("v{}", var_id));
                    translator.emit(TestInstr::Call {
                        dest: dest.clone(),
                        return_value: *return_value,
                    });
                    translator.write_variable(format!("v{}", var_id), block_id, TestValue::Var(dest));
                }
            }
        }

        translator.materialize_all_phis();
        translator
    }

    /// Interpret the program directly (reference implementation).
    pub fn interpret(&self) -> HashMap<usize, i64> {
        let mut env: HashMap<usize, i64> = HashMap::new();

        for instr in &self.instructions {
            match instr {
                TestProgramInstr::Const { var_id, value } => {
                    env.insert(*var_id, *value);
                }
                TestProgramInstr::BinOp { var_id, left_id, op, right_id } => {
                    let l = env.get(left_id).copied().unwrap_or(0);
                    let r = env.get(right_id).copied().unwrap_or(0);
                    let result = match op {
                        BinOp::Add => l.wrapping_add(r),
                        BinOp::Sub => l.wrapping_sub(r),
                        BinOp::Mul => l.wrapping_mul(r),
                    };
                    env.insert(*var_id, result);
                }
                TestProgramInstr::Neg { var_id, src_id } => {
                    let val = env.get(src_id).copied().unwrap_or(0);
                    env.insert(*var_id, val.wrapping_neg());
                }
                TestProgramInstr::Copy { var_id, src_id } => {
                    let val = env.get(src_id).copied().unwrap_or(0);
                    env.insert(*var_id, val);
                }
                TestProgramInstr::Call { var_id, return_value } => {
                    env.insert(*var_id, *return_value);
                }
            }
        }

        env
    }

    /// Get all variable IDs defined by this program.
    pub fn defined_vars(&self) -> Vec<usize> {
        self.instructions.iter().map(|instr| match instr {
            TestProgramInstr::Const { var_id, .. }
            | TestProgramInstr::BinOp { var_id, .. }
            | TestProgramInstr::Neg { var_id, .. }
            | TestProgramInstr::Copy { var_id, .. }
            | TestProgramInstr::Call { var_id, .. } => *var_id,
        }).collect()
    }
}

// ============================================================================
// Property-based test strategies
// ============================================================================

fn arb_binop() -> impl Strategy<Value = BinOp> {
    prop_oneof![
        Just(BinOp::Add),
        Just(BinOp::Sub),
        Just(BinOp::Mul),
    ]
}

/// Generate a single instruction given the number of already-defined variables.
fn arb_instruction(var_id: usize, num_defined: usize) -> impl Strategy<Value = TestProgramInstr> {
    if num_defined == 0 {
        // First instruction must be a constant
        (-100i64..100i64)
            .prop_map(move |value| TestProgramInstr::Const { var_id, value })
            .boxed()
    } else {
        // Can be const, binop, neg, or copy
        prop_oneof![
            // Constant
            (-100i64..100i64).prop_map(move |value| TestProgramInstr::Const { var_id, value }),
            // BinOp (reference previous vars)
            (0..num_defined, arb_binop(), 0..num_defined)
                .prop_map(move |(l, op, r)| TestProgramInstr::BinOp {
                    var_id,
                    left_id: l,
                    op,
                    right_id: r,
                }),
            // Neg
            (0..num_defined).prop_map(move |src_id| TestProgramInstr::Neg { var_id, src_id }),
            // Copy
            (0..num_defined).prop_map(move |src_id| TestProgramInstr::Copy { var_id, src_id }),
        ]
        .boxed()
    }
}

/// Generate a valid straight-line program with a given number of instructions.
fn arb_program_with_size(num_instrs: usize) -> impl Strategy<Value = TestProgram> {
    let mut strategies: Vec<BoxedStrategy<TestProgramInstr>> = Vec::new();

    for i in 0..num_instrs {
        strategies.push(arb_instruction(i, i).boxed());
    }

    strategies
        .into_iter()
        .collect::<Vec<_>>()
        .prop_map(|instrs| TestProgram { instructions: instrs })
}

/// Generate a valid straight-line program.
fn arb_program(max_vars: usize) -> impl Strategy<Value = TestProgram> {
    (1..=max_vars).prop_flat_map(arb_program_with_size)
}

// ============================================================================
// Tests
// ============================================================================

/// Check if a TestInstr is a call instruction (clobbers caller-saved registers)
fn is_call_instr(instr: &TestInstr) -> bool {
    matches!(instr, TestInstr::Call { .. })
}

/// Full pipeline: SSA → RegAlloc → CodeGen → VM execution
///
/// This test verifies that register allocation produces correct code by:
/// 1. Running the program through the full compilation pipeline
/// 2. Executing the generated VM code
/// 3. Comparing the final result (last variable) with the interpreter
fn run_full_pipeline(program: &TestProgram) -> Result<(), String> {
    if program.instructions.is_empty() {
        return Ok(());
    }

    // 1. Get expected results from direct interpretation
    let expected = program.interpret();

    // 2. Build SSA
    let mut translator = program.to_ssa();

    // 3. Eliminate phis (for straight-line code this is a no-op)
    PhiElimination::eliminate(&mut translator);

    // 4. Compute liveness and intervals WITH call site detection
    let liveness = LivenessAnalysis::compute(&translator);
    let mut interval_analysis = IntervalAnalysis::compute_with_call_sites(
        &translator,
        &liveness,
        is_call_instr,  // Closure to detect call instructions
    );

    // 5. Run register allocation
    let config = LinearScanConfig::default();
    let mut allocator = LinearScanAllocator::with_config(SimpleArch, config);
    let alloc_result = allocator.allocate(&mut interval_analysis.intervals);

    // 6. Generate VM code that properly handles register allocation
    let (vm_code, _) = generate_vm_code_v2(&translator, &alloc_result.assignments);

    // 7. Run VM
    let mut vm = Vm::new(alloc_result.stack_slots_used.max(16));
    vm.run(&vm_code);

    // 8. Verify the final result
    // For straight-line code, we verify the last variable defined
    let last_var_id = program.instructions.len() - 1;
    let var = SsaVariable::new(&format!("v{}", last_var_id));
    let expected_val = expected.get(&last_var_id).copied().unwrap_or(0);

    if let Some(loc) = alloc_result.assignments.get(&var) {
        let actual_val = match loc {
            Location::Register(id) => {
                Reg::from_id(*id).map(|r| vm.get_reg(r)).unwrap_or(0)
            }
            Location::StackSlot(slot) => {
                vm.stack.get(*slot).copied().unwrap_or(0)
            }
        };

        if actual_val != expected_val {
            return Err(format!(
                "Final variable v{}: expected {}, got {} (loc: {:?})\nProgram: {:?}",
                last_var_id, expected_val, actual_val, loc, program
            ));
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn test_regalloc_correctness(program in arb_program(10)) {
        prop_assert!(run_full_pipeline(&program).is_ok(), "Pipeline failed: {:?}", run_full_pipeline(&program));
    }

    #[test]
    fn test_regalloc_with_many_vars(program in arb_program(20)) {
        // More variables than registers - forces spilling
        prop_assert!(run_full_pipeline(&program).is_ok(), "Pipeline failed: {:?}", run_full_pipeline(&program));
    }
}

// ============================================================================
// Specific regression tests
// ============================================================================

#[test]
fn test_simple_const() {
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 42 },
        ],
    };
    run_full_pipeline(&program).unwrap();
}

#[test]
fn test_simple_add() {
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 10 },
            TestProgramInstr::Const { var_id: 1, value: 20 },
            TestProgramInstr::BinOp { var_id: 2, left_id: 0, op: BinOp::Add, right_id: 1 },
        ],
    };
    run_full_pipeline(&program).unwrap();
}

#[test]
fn test_chain_of_ops() {
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 1 },
            TestProgramInstr::Const { var_id: 1, value: 2 },
            TestProgramInstr::BinOp { var_id: 2, left_id: 0, op: BinOp::Add, right_id: 1 },
            TestProgramInstr::BinOp { var_id: 3, left_id: 2, op: BinOp::Mul, right_id: 1 },
            TestProgramInstr::Neg { var_id: 4, src_id: 3 },
        ],
    };
    run_full_pipeline(&program).unwrap();
}

#[test]
fn test_many_live_vars() {
    // Create many variables that are all live at the same time
    // This should force spilling (we have 4 registers)
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 1 },
            TestProgramInstr::Const { var_id: 1, value: 2 },
            TestProgramInstr::Const { var_id: 2, value: 3 },
            TestProgramInstr::Const { var_id: 3, value: 4 },
            TestProgramInstr::Const { var_id: 4, value: 5 },
            TestProgramInstr::Const { var_id: 5, value: 6 },
            // Now use all of them
            TestProgramInstr::BinOp { var_id: 6, left_id: 0, op: BinOp::Add, right_id: 1 },
            TestProgramInstr::BinOp { var_id: 7, left_id: 2, op: BinOp::Add, right_id: 3 },
            TestProgramInstr::BinOp { var_id: 8, left_id: 4, op: BinOp::Add, right_id: 5 },
            TestProgramInstr::BinOp { var_id: 9, left_id: 6, op: BinOp::Add, right_id: 7 },
            TestProgramInstr::BinOp { var_id: 10, left_id: 9, op: BinOp::Add, right_id: 8 },
        ],
    };
    let result = run_full_pipeline(&program);
    assert!(result.is_ok(), "Error: {:?}", result);
}

#[test]
fn test_copy_chain() {
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 42 },
            TestProgramInstr::Copy { var_id: 1, src_id: 0 },
            TestProgramInstr::Copy { var_id: 2, src_id: 1 },
            TestProgramInstr::Copy { var_id: 3, src_id: 2 },
        ],
    };
    run_full_pipeline(&program).unwrap();
}

#[test]
fn test_reuse_after_last_use() {
    // After a variable's last use, its register can be reused
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 1 },
            TestProgramInstr::Const { var_id: 1, value: 2 },
            TestProgramInstr::BinOp { var_id: 2, left_id: 0, op: BinOp::Add, right_id: 1 },
            // v0 and v1 are dead after this point
            TestProgramInstr::Const { var_id: 3, value: 3 },
            TestProgramInstr::Const { var_id: 4, value: 4 },
            TestProgramInstr::BinOp { var_id: 5, left_id: 2, op: BinOp::Add, right_id: 3 },
            TestProgramInstr::BinOp { var_id: 6, left_id: 5, op: BinOp::Add, right_id: 4 },
        ],
    };
    run_full_pipeline(&program).unwrap();
}

// ============================================================================
// Call clobbering tests
// ============================================================================

#[test]
fn test_single_call() {
    // Simple case: a single call with no values crossing it
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Call { var_id: 0, return_value: 42 },
        ],
    };
    run_full_pipeline(&program).unwrap();
}

#[test]
fn test_call_result_used() {
    // Call result is used in a subsequent operation
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Call { var_id: 0, return_value: 10 },
            TestProgramInstr::Const { var_id: 1, value: 5 },
            TestProgramInstr::BinOp { var_id: 2, left_id: 0, op: BinOp::Add, right_id: 1 },
        ],
    };
    run_full_pipeline(&program).unwrap();
}

#[test]
fn test_value_survives_call() {
    // A value is defined BEFORE a call and used AFTER - must survive the call.
    // The register allocator must put v0 in a callee-saved register (R2 or R3)
    // or spill it.
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 100 },  // Must survive the call
            TestProgramInstr::Call { var_id: 1, return_value: 42 },  // Clobbers R0, R1
            TestProgramInstr::BinOp { var_id: 2, left_id: 0, op: BinOp::Add, right_id: 1 },  // Uses v0 after call
        ],
    };
    let result = run_full_pipeline(&program);
    assert!(result.is_ok(), "Error: {:?}", result);
}

#[test]
fn test_fib_pattern_two_calls() {
    // This is the fib(n-1) + fib(n-2) pattern that caused the original bug.
    // v0 = call1() - returns 10
    // v1 = call2() - returns 20, CLOBBERS R0/R1
    // v2 = v0 + v1 - v0 must have survived call2!
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Call { var_id: 0, return_value: 10 },  // First call
            TestProgramInstr::Call { var_id: 1, return_value: 20 },  // Second call clobbers
            TestProgramInstr::BinOp { var_id: 2, left_id: 0, op: BinOp::Add, right_id: 1 },  // Should be 30
        ],
    };
    let result = run_full_pipeline(&program);
    assert!(result.is_ok(), "Error: {:?}", result);
}

#[test]
fn test_multiple_values_survive_call() {
    // Multiple values must survive a call
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 1 },
            TestProgramInstr::Const { var_id: 1, value: 2 },
            TestProgramInstr::Call { var_id: 2, return_value: 100 },
            // Both v0 and v1 must survive the call
            TestProgramInstr::BinOp { var_id: 3, left_id: 0, op: BinOp::Add, right_id: 1 },
            TestProgramInstr::BinOp { var_id: 4, left_id: 3, op: BinOp::Add, right_id: 2 },
        ],
    };
    let result = run_full_pipeline(&program);
    assert!(result.is_ok(), "Error: {:?}", result);
}

#[test]
fn test_three_calls_chain() {
    // Three calls in a row, each result must survive subsequent calls
    // v0 = call() returns 1
    // v1 = call() returns 2, v0 must survive
    // v2 = call() returns 3, v0 and v1 must survive
    // v3 = v0 + v1 + v2 = 6
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Call { var_id: 0, return_value: 1 },
            TestProgramInstr::Call { var_id: 1, return_value: 2 },
            TestProgramInstr::Call { var_id: 2, return_value: 3 },
            TestProgramInstr::BinOp { var_id: 3, left_id: 0, op: BinOp::Add, right_id: 1 },
            TestProgramInstr::BinOp { var_id: 4, left_id: 3, op: BinOp::Add, right_id: 2 },
        ],
    };
    let result = run_full_pipeline(&program);
    assert!(result.is_ok(), "Error: {:?}", result);
}

#[test]
fn test_call_with_spilling_pressure() {
    // Many values that must survive a call - forces spilling
    // We have 2 callee-saved regs (R2, R3), so 3+ values that survive calls must spill
    let program = TestProgram {
        instructions: vec![
            TestProgramInstr::Const { var_id: 0, value: 1 },
            TestProgramInstr::Const { var_id: 1, value: 2 },
            TestProgramInstr::Const { var_id: 2, value: 3 },
            TestProgramInstr::Const { var_id: 3, value: 4 },
            TestProgramInstr::Call { var_id: 4, return_value: 100 },
            // All of v0-v3 must survive the call - this requires spilling
            TestProgramInstr::BinOp { var_id: 5, left_id: 0, op: BinOp::Add, right_id: 1 },
            TestProgramInstr::BinOp { var_id: 6, left_id: 2, op: BinOp::Add, right_id: 3 },
            TestProgramInstr::BinOp { var_id: 7, left_id: 5, op: BinOp::Add, right_id: 6 },
            TestProgramInstr::BinOp { var_id: 8, left_id: 7, op: BinOp::Add, right_id: 4 },
        ],
    };
    let result = run_full_pipeline(&program);
    assert!(result.is_ok(), "Error: {:?}", result);
}
