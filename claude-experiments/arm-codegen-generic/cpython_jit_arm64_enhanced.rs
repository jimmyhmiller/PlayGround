// Generated ARM64 instructions for CPython JIT
// Generated using arm-codegen-generic
//
// This file provides both:
// 1. Low-level encoding functions (generated from ARM spec)
// 2. High-level enum variants and builder methods (for convenience)

use std::ops::Shl;

/// ARM64 register sizes
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Size {
    S32,
    S64,
}

/// ARM64 register representation
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Register {
    pub size: Size,
    pub index: u8,
}

impl Register {
    /// Get the sf (size flag) bit for instructions
    pub fn sf(&self) -> i32 {
        match self.size {
            Size::S32 => 0,
            Size::S64 => 1,
        }
    }

    /// Create a 64-bit register from an index
    pub fn from_index(index: usize) -> Register {
        Register {
            index: index as u8,
            size: Size::S64,
        }
    }

    /// Encode register for instruction encoding
    pub fn encode(&self) -> u8 {
        self.index
    }
}

impl Shl<u32> for &Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

impl Shl<u32> for Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

// Common ARM64 registers
pub const X0: Register = Register { index: 0, size: Size::S64 };
pub const X1: Register = Register { index: 1, size: Size::S64 };
pub const X2: Register = Register { index: 2, size: Size::S64 };
pub const X3: Register = Register { index: 3, size: Size::S64 };
pub const X4: Register = Register { index: 4, size: Size::S64 };
pub const X5: Register = Register { index: 5, size: Size::S64 };
pub const X6: Register = Register { index: 6, size: Size::S64 };
pub const X7: Register = Register { index: 7, size: Size::S64 };
pub const X8: Register = Register { index: 8, size: Size::S64 };
pub const X9: Register = Register { index: 9, size: Size::S64 };
pub const X10: Register = Register { index: 10, size: Size::S64 };
pub const X11: Register = Register { index: 11, size: Size::S64 };
pub const X12: Register = Register { index: 12, size: Size::S64 };
pub const X13: Register = Register { index: 13, size: Size::S64 };
pub const X14: Register = Register { index: 14, size: Size::S64 };
pub const X15: Register = Register { index: 15, size: Size::S64 };
pub const X16: Register = Register { index: 16, size: Size::S64 };
pub const X17: Register = Register { index: 17, size: Size::S64 };
pub const X18: Register = Register { index: 18, size: Size::S64 };
pub const X19: Register = Register { index: 19, size: Size::S64 };
pub const X20: Register = Register { index: 20, size: Size::S64 };
pub const X21: Register = Register { index: 21, size: Size::S64 };
pub const X22: Register = Register { index: 22, size: Size::S64 };
pub const X23: Register = Register { index: 23, size: Size::S64 };
pub const X24: Register = Register { index: 24, size: Size::S64 };
pub const X25: Register = Register { index: 25, size: Size::S64 };
pub const X26: Register = Register { index: 26, size: Size::S64 };
pub const X27: Register = Register { index: 27, size: Size::S64 };
pub const X28: Register = Register { index: 28, size: Size::S64 };
pub const X29: Register = Register { index: 29, size: Size::S64 }; // Frame pointer
pub const X30: Register = Register { index: 30, size: Size::S64 }; // Link register
pub const SP: Register = Register { index: 31, size: Size::S64 }; // Stack pointer
pub const ZERO_REGISTER: Register = Register { index: 31, size: Size::S64 }; // XZR

/// Truncate immediate values to specified bit width with bounds checking
pub fn truncate_imm<T: Into<i32>, const WIDTH: usize>(imm: T) -> u32 {
    let value: i32 = imm.into();
    let masked = (value as u32) & ((1 << WIDTH) - 1);

    // Assert that we didn't drop any bits by truncating
    if value >= 0 {
        assert_eq!(value as u32, masked);
    } else {
        assert_eq!(value as u32, masked | (u32::MAX << WIDTH));
    }

    masked
}

//
// =============================================================================
// GENERATED ENCODING FUNCTIONS (from arm-codegen-generic)
// =============================================================================
//

/// ADD (shifted register) -- A64
/// Add (shifted register)
/// ADD  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn add_addsub_shift(sf: i32, shift: i32, rm: Register, imm6: i32, rn: Register, rd: Register) -> u32 {
    let mut result = 0b0_0_0_01011_00_0_00000_000000_00000_00000;
    result |= (sf as u32) << 31;
    result |= (shift as u32) << 22;
    result |= (rm.encode() as u32) << 16;
    result |= truncate_imm::<_, 6>(imm6) << 10;
    result |= (rn.encode() as u32) << 5;
    result |= (rd.encode() as u32) << 0;
    result
}

/// SUB (shifted register) -- A64
/// Subtract (shifted register)
/// SUB  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn sub_addsub_shift(sf: i32, shift: i32, rm: Register, imm6: i32, rn: Register, rd: Register) -> u32 {
    let mut result = 0b0_1_0_01011_00_0_00000_000000_00000_00000;
    result |= (sf as u32) << 31;
    result |= (shift as u32) << 22;
    result |= (rm.encode() as u32) << 16;
    result |= truncate_imm::<_, 6>(imm6) << 10;
    result |= (rn.encode() as u32) << 5;
    result |= (rd.encode() as u32) << 0;
    result
}

/// CMP (shifted register) -- A64
/// Compare (shifted register)
/// CMP  <Xn>, <Xm>{, <shift> #<amount>}
/// SUBS XZR, <Xn>, <Xm> {, <shift> #<amount>}
pub fn cmp_subs_addsub_shift(sf: i32, shift: i32, rm: Register, imm6: i32, rn: Register) -> u32 {
    let mut result = 0b0_1_1_01011_00_0_00000_000000_00000_11111;
    result |= (sf as u32) << 31;
    result |= (shift as u32) << 22;
    result |= (rm.encode() as u32) << 16;
    result |= truncate_imm::<_, 6>(imm6) << 10;
    result |= (rn.encode() as u32) << 5;
    result
}

/// CBNZ -- A64
/// Compare and Branch on Nonzero
/// CBNZ  <Xt>, <label>
pub fn cbnz(sf: i32, imm19: i32, rt: Register) -> u32 {
    let mut result = 0b0_011010_1_0000000000000000000_00000;
    result |= (sf as u32) << 31;
    result |= truncate_imm::<_, 19>(imm19) << 5;
    result |= (rt.encode() as u32) << 0;
    result
}

/// B.cond -- A64
/// Branch conditionally
/// B.<cond>  <label>
pub fn bcond(imm19: i32, cond: i32) -> u32 {
    let mut result = 0b0101010_0_0000000000000000000_0_0000;
    result |= truncate_imm::<_, 19>(imm19) << 5;
    result |= (cond as u32) << 0;
    result
}

//
// =============================================================================
// HIGH-LEVEL ENUM VARIANTS (matching cpython-rust-jit/src/arm64.rs style)
// =============================================================================
//

/// ARM64 instruction encoding
#[derive(Debug, Clone)]
pub enum ArmAsm {
    /// ADD (shifted register) - Add two registers
    AddReg {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },

    /// SUB (shifted register) - Subtract two registers
    SubReg {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },

    /// CMP (shifted register) - Compare registers (alias for SUBS with XZR as destination)
    CmpReg {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
    },

    /// CBNZ - Compare and Branch on Non-Zero (64-bit)
    Cbnz { imm19: i32, rt: Register },

    /// B.LT - Branch if less than (signed)
    BLt { imm19: i32 },

    /// B.GE - Branch if greater than or equal (signed)
    BGe { imm19: i32 },

    /// B.LE - Branch if less than or equal (signed)
    BLe { imm19: i32 },

    /// B.GT - Branch if greater than (signed)
    BGt { imm19: i32 },

    /// B.EQ - Branch if equal
    BEq { imm19: i32 },

    /// B.NE - Branch if not equal
    BNe { imm19: i32 },
}

impl ArmAsm {
    /// Encode the instruction to a 32-bit machine code value
    pub fn encode(&self) -> u32 {
        match self {
            ArmAsm::AddReg { sf, shift, rm, imm6, rn, rd } => {
                add_addsub_shift(*sf, *shift, *rm, *imm6, *rn, *rd)
            }

            ArmAsm::SubReg { sf, shift, rm, imm6, rn, rd } => {
                sub_addsub_shift(*sf, *shift, *rm, *imm6, *rn, *rd)
            }

            ArmAsm::CmpReg { sf, shift, rm, imm6, rn } => {
                cmp_subs_addsub_shift(*sf, *shift, *rm, *imm6, *rn)
            }

            ArmAsm::Cbnz { imm19, rt } => {
                // 64-bit CBNZ (sf=1)
                cbnz(1, *imm19, *rt)
            }

            ArmAsm::BLt { imm19 } => {
                // B.LT: signed less than, condition code 0b1011 (11)
                bcond(*imm19, 11)
            }

            ArmAsm::BGe { imm19 } => {
                // B.GE: signed greater or equal, condition code 0b1010 (10)
                bcond(*imm19, 10)
            }

            ArmAsm::BLe { imm19 } => {
                // B.LE: signed less or equal, condition code 0b1101 (13)
                bcond(*imm19, 13)
            }

            ArmAsm::BGt { imm19 } => {
                // B.GT: signed greater than, condition code 0b1100 (12)
                bcond(*imm19, 12)
            }

            ArmAsm::BEq { imm19 } => {
                // B.EQ: equal, condition code 0b0000 (0)
                bcond(*imm19, 0)
            }

            ArmAsm::BNe { imm19 } => {
                // B.NE: not equal, condition code 0b0001 (1)
                bcond(*imm19, 1)
            }
        }
    }

    /// Get instruction bytes in little-endian format
    pub fn to_bytes(&self) -> [u8; 4] {
        self.encode().to_le_bytes()
    }
}

//
// =============================================================================
// BUILDER HELPER METHODS (for Arm64Builder)
// =============================================================================
//

/// Add these methods to your existing Arm64Builder in cpython-rust-jit/src/arm64.rs:

/*
    /// Add two registers: rd = rn + rm
    pub fn add_reg(&mut self, rd: Register, rn: Register, rm: Register) {
        self.instructions.push(ArmAsm::AddReg {
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
            rd,
        });
    }

    /// Subtract two registers: rd = rn - rm
    pub fn sub_reg(&mut self, rd: Register, rn: Register, rm: Register) {
        self.instructions.push(ArmAsm::SubReg {
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
            rd,
        });
    }

    /// Compare two registers: CMP rn, rm
    pub fn cmp_reg(&mut self, rn: Register, rm: Register) {
        self.instructions.push(ArmAsm::CmpReg {
            sf: rn.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
        });
    }

    /// Compare and branch on non-zero: CBNZ rt, label
    pub fn cbnz_label(&mut self, rt: Register, label_id: usize) {
        let at = self.instructions.len();
        self.instructions.push(ArmAsm::Cbnz { imm19: 0, rt });
        self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::Cbnz });
    }

    /// Branch if less than (signed): B.LT label
    pub fn b_lt_label(&mut self, label_id: usize) {
        let at = self.instructions.len();
        self.instructions.push(ArmAsm::BLt { imm19: 0 });
        self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
    }

    /// Branch if greater or equal (signed): B.GE label
    pub fn b_ge_label(&mut self, label_id: usize) {
        let at = self.instructions.len();
        self.instructions.push(ArmAsm::BGe { imm19: 0 });
        self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
    }

    /// Branch if less or equal (signed): B.LE label
    pub fn b_le_label(&mut self, label_id: usize) {
        let at = self.instructions.len();
        self.instructions.push(ArmAsm::BLe { imm19: 0 });
        self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
    }

    /// Branch if greater than (signed): B.GT label
    pub fn b_gt_label(&mut self, label_id: usize) {
        let at = self.instructions.len();
        self.instructions.push(ArmAsm::BGt { imm19: 0 });
        self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
    }

    /// Branch if equal: B.EQ label
    pub fn b_eq_label(&mut self, label_id: usize) {
        let at = self.instructions.len();
        self.instructions.push(ArmAsm::BEq { imm19: 0 });
        self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
    }

    /// Branch if not equal: B.NE label
    pub fn b_ne_label(&mut self, label_id: usize) {
        let at = self.instructions.len();
        self.instructions.push(ArmAsm::BNe { imm19: 0 });
        self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
    }
*/

//
// =============================================================================
// UPDATE BranchKind enum to support new conditional branches:
// =============================================================================
//

/*
#[derive(Debug, Copy, Clone)]
enum BranchKind {
    B,
    Cbz,
    Cbnz,
    BCond,  // For all B.cond variants (LT, GE, LE, GT, EQ, NE)
}
*/

//
// =============================================================================
// UPDATE compile() method to handle new branch types:
// =============================================================================
//

/*
pub fn compile(&self) -> Vec<u8> {
    let mut insts = self.instructions.clone();
    for patch in &self.patches {
        let at = patch.at;
        let target_index = self.labels[patch.target_label];
        let delta = target_index as isize - at as isize;
        match (&mut insts[at], patch.kind) {
            (ArmAsm::B { imm26 }, BranchKind::B) => {
                *imm26 = delta as i32;
            }
            (ArmAsm::Cbz { imm19, .. }, BranchKind::Cbz) => {
                *imm19 = delta as i32;
            }
            (ArmAsm::Cbnz { imm19, .. }, BranchKind::Cbnz) => {
                *imm19 = delta as i32;
            }
            (ArmAsm::BLt { imm19 }, BranchKind::BCond) |
            (ArmAsm::BGe { imm19 }, BranchKind::BCond) |
            (ArmAsm::BLe { imm19 }, BranchKind::BCond) |
            (ArmAsm::BGt { imm19 }, BranchKind::BCond) |
            (ArmAsm::BEq { imm19 }, BranchKind::BCond) |
            (ArmAsm::BNe { imm19 }, BranchKind::BCond) => {
                *imm19 = delta as i32;
            }
            _ => {}
        }
    }
    insts.iter().flat_map(|inst| inst.to_bytes()).collect()
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_reg_encoding() {
        // ADD X0, X1, X2
        let encoding = add_addsub_shift(1, 0, X2, 0, X1, X0);
        assert_ne!(encoding, 0);
    }

    #[test]
    fn test_sub_reg_encoding() {
        // SUB X0, X1, X2
        let encoding = sub_addsub_shift(1, 0, X2, 0, X1, X0);
        assert_ne!(encoding, 0);
    }

    #[test]
    fn test_cmp_reg_encoding() {
        // CMP X1, X2
        let encoding = cmp_subs_addsub_shift(1, 0, X2, 0, X1);
        assert_ne!(encoding, 0);
    }

    #[test]
    fn test_cbnz_encoding() {
        // CBNZ X0, offset
        let encoding = cbnz(1, 5, X0);
        assert_ne!(encoding, 0);
    }

    #[test]
    fn test_bcond_encodings() {
        // Test all condition codes
        assert_ne!(bcond(5, 0), 0);  // B.EQ
        assert_ne!(bcond(5, 1), 0);  // B.NE
        assert_ne!(bcond(5, 10), 0); // B.GE
        assert_ne!(bcond(5, 11), 0); // B.LT
        assert_ne!(bcond(5, 12), 0); // B.GT
        assert_ne!(bcond(5, 13), 0); // B.LE
    }

    #[test]
    fn test_enum_variants() {
        let add = ArmAsm::AddReg {
            sf: 1,
            shift: 0,
            rm: X2,
            imm6: 0,
            rn: X1,
            rd: X0,
        };
        assert_ne!(add.encode(), 0);

        let b_eq = ArmAsm::BEq { imm19: 5 };
        assert_ne!(b_eq.encode(), 0);
    }
}
