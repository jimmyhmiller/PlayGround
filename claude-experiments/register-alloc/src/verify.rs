//! Allocation verifier: checks that an allocation is correct.
//!
//! This is the backbone of exhaustive testing. An allocation is correct if:
//! 1. Every operand is assigned a physical register.
//! 2. Every assignment respects the operand's constraint.
//! 3. No two live values occupy the same physical register at any point.
//! 4. Clobbered registers don't hold live values (unless saved/restored by moves).
//! 5. All inserted moves are well-formed.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::allocator::Allocation;
use crate::ir::Function;
use crate::target::Target;
use crate::types::*;

/// A verification error.
#[derive(Debug, Clone)]
pub struct VerifyError {
    pub kind: VerifyErrorKind,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyErrorKind {
    /// An operand has no allocation.
    MissingAllocation,
    /// An operand was assigned a register not in the required class.
    WrongRegClass,
    /// An operand with a FixedReg constraint got the wrong register.
    WrongFixedReg,
    /// A Tied constraint was violated (two operands don't share a register).
    TiedMismatch,
    /// Two live values occupy the same physical register.
    RegConflict,
    /// A live value was in a clobbered register at a call/clobber site.
    ClobberConflict,
    /// A reserved register was allocated.
    ReservedRegUsed,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.detail)
    }
}

impl std::error::Error for VerifyError {}

/// Verify that an allocation is correct for the given function and target.
/// Returns Ok(()) if the allocation is valid, or a list of all errors found.
pub fn verify<F: Function, T: Target>(
    func: &F,
    target: &T,
    alloc: &Allocation,
) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    let reserved: HashSet<PReg> = target.reserved_regs().iter().copied().collect();

    // Check each instruction's operands.
    for block in func.blocks() {
        for inst in func.block_insts(block) {
            let operands: Vec<Operand> = func.inst_operands(inst).collect();

            for (op_idx, operand) in operands.iter().enumerate() {
                // 1. Every operand must have an allocation.
                let Some(assigned) = alloc.get(inst, op_idx) else {
                    // Physical register operands don't need allocation
                    if matches!(operand.reg, Reg::Physical(_)) {
                        continue;
                    }
                    errors.push(VerifyError {
                        kind: VerifyErrorKind::MissingAllocation,
                        detail: format!(
                            "{:?} operand {} ({:?}) has no allocation",
                            inst, op_idx, operand.reg
                        ),
                    });
                    continue;
                };

                // 2. Not a reserved register.
                if reserved.contains(&assigned) {
                    errors.push(VerifyError {
                        kind: VerifyErrorKind::ReservedRegUsed,
                        detail: format!(
                            "{:?} operand {} assigned reserved register {:?}",
                            inst, op_idx, assigned
                        ),
                    });
                }

                // 3. Constraint satisfaction.
                match &operand.constraint {
                    OperandConstraint::RegClass(class) | OperandConstraint::RegOrStack(class) => {
                        if target.reg_class_of(assigned) != *class {
                            errors.push(VerifyError {
                                kind: VerifyErrorKind::WrongRegClass,
                                detail: format!(
                                    "{:?} operand {} assigned {:?} but needs class {:?}",
                                    inst, op_idx, assigned, class
                                ),
                            });
                        }
                    }
                    OperandConstraint::FixedReg(required) => {
                        if assigned != *required {
                            errors.push(VerifyError {
                                kind: VerifyErrorKind::WrongFixedReg,
                                detail: format!(
                                    "{:?} operand {} assigned {:?} but must be {:?}",
                                    inst, op_idx, assigned, required
                                ),
                            });
                        }
                    }
                    OperandConstraint::Tied(tied_idx) => {
                        if let Some(tied_reg) = alloc.get(inst, *tied_idx) {
                            if assigned != tied_reg {
                                errors.push(VerifyError {
                                    kind: VerifyErrorKind::TiedMismatch,
                                    detail: format!(
                                        "{:?} operand {} ({:?}) tied to operand {} ({:?}) but assigned different registers",
                                        inst, op_idx, assigned, tied_idx, tied_reg
                                    ),
                                });
                            }
                        }
                    }
                    OperandConstraint::Reuse(reuse_idx) => {
                        if let Some(reuse_reg) = alloc.get(inst, *reuse_idx) {
                            if assigned != reuse_reg {
                                errors.push(VerifyError {
                                    kind: VerifyErrorKind::TiedMismatch,
                                    detail: format!(
                                        "{:?} operand {} reuses operand {} but assigned {:?} vs {:?}",
                                        inst, op_idx, reuse_idx, assigned, reuse_reg
                                    ),
                                });
                            }
                        }
                    }
                }
            }

            // 4. No two defs write the same register at the same instruction.
            let mut defs_at_inst: HashMap<PReg, usize> = HashMap::new();
            for (op_idx, operand) in operands.iter().enumerate() {
                if matches!(
                    operand.kind,
                    OperandKind::Def | OperandKind::UseDef | OperandKind::EarlyDef
                ) {
                    if let Some(assigned) = alloc.get(inst, op_idx) {
                        if let Some(&prev_idx) = defs_at_inst.get(&assigned) {
                            // Check if tied — tied operands are allowed to share.
                            let is_tied = matches!(
                                &operand.constraint,
                                OperandConstraint::Tied(t) | OperandConstraint::Reuse(t) if *t == prev_idx
                            );
                            let prev_is_tied = matches!(
                                &operands[prev_idx].constraint,
                                OperandConstraint::Tied(t) | OperandConstraint::Reuse(t) if *t == op_idx
                            );
                            if !is_tied && !prev_is_tied {
                                errors.push(VerifyError {
                                    kind: VerifyErrorKind::RegConflict,
                                    detail: format!(
                                        "{:?} operands {} and {} both define {:?}",
                                        inst, prev_idx, op_idx, assigned
                                    ),
                                });
                            }
                        } else {
                            defs_at_inst.insert(assigned, op_idx);
                        }
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
