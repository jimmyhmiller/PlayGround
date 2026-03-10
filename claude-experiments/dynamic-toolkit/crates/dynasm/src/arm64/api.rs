//! High-level constructor functions for ARM64 instructions.
//!
//! These derive `sf`, `size`, `opc`, etc. from register types so callers
//! don't need to manually set encoding fields.

use super::inst::*;
use super::reg::{Arm64Reg, RegSize, SP, XZR};

impl Arm64Inst {
    // === Arithmetic ===

    /// ADD Rd, Rn, Rm
    pub fn add(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::AddShifted {
            sf: rd.sf(), shift: 0, rm, imm6: 0, rn, rd,
        }
    }

    /// ADD Rd, Rn, #imm12
    pub fn add_imm(rd: Arm64Reg, rn: Arm64Reg, imm12: i32) -> Self {
        Arm64Inst::AddImm {
            sf: rd.sf(), sh: 0, imm12, rn, rd,
        }
    }

    /// SUB Rd, Rn, Rm
    pub fn sub(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::SubShifted {
            sf: rd.sf(), shift: 0, rm, imm6: 0, rn, rd,
        }
    }

    /// SUB Rd, Rn, #imm12
    pub fn sub_imm(rd: Arm64Reg, rn: Arm64Reg, imm12: i32) -> Self {
        Arm64Inst::SubImm {
            sf: rd.sf(), sh: 0, imm12, rn, rd,
        }
    }

    /// CMP Rn, Rm  (alias for SUBS XZR, Rn, Rm)
    pub fn cmp(rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::CmpShifted {
            sf: rn.sf(), shift: 0, rm, imm6: 0, rn,
        }
    }

    /// CMP Rn, #imm12  (alias for SUBS XZR/WZR, Rn, #imm12)
    pub fn cmp_imm(rn: Arm64Reg, imm12: i32) -> Self {
        // Encoded as SUBS (immediate) with Rd = zero register
        let zr = match rn.size {
            RegSize::W32 => super::reg::WZR,
            RegSize::X64 => XZR,
        };
        // We need a SUBS immediate variant. Use the raw SubsImm instruction.
        Arm64Inst::SubsImm {
            sf: rn.sf(), sh: 0, imm12, rn, rd: zr,
        }
    }

    /// CSET Rd, cond
    pub fn cset(rd: Arm64Reg, cond: Arm64Cond) -> Self {
        // CSET encodes the inverted condition
        Arm64Inst::Cset {
            sf: rd.sf(),
            cond: cond.invert() as i32,
            rd,
        }
    }

    // === Move ===

    /// MOV Rd, Rm (register to register)
    pub fn mov(rd: Arm64Reg, rm: Arm64Reg) -> Self {
        // If either register is SP, use the ADD-based MOV
        if rd == SP || rm == SP {
            Arm64Inst::MovSp { sf: rd.sf(), rn: rm, rd }
        } else {
            Arm64Inst::MovReg { sf: rd.sf(), rm, rd }
        }
    }

    /// MOVZ Rd, #imm16, LSL #shift
    pub fn movz(rd: Arm64Reg, imm16: u16, shift: u8) -> Self {
        assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
        Arm64Inst::Movz {
            sf: rd.sf(),
            hw: (shift / 16) as i32,
            imm16: imm16 as i32,
            rd,
        }
    }

    /// MOVK Rd, #imm16, LSL #shift
    pub fn movk(rd: Arm64Reg, imm16: u16, shift: u8) -> Self {
        assert!(shift == 0 || shift == 16 || shift == 32 || shift == 48);
        Arm64Inst::Movk {
            sf: rd.sf(),
            hw: (shift / 16) as i32,
            imm16: imm16 as i32,
            rd,
        }
    }

    /// Load a full 64-bit immediate into a register using MOVZ + MOVK sequence.
    /// Returns a Vec of 1-4 instructions.
    pub fn mov_imm64(rd: Arm64Reg, value: u64) -> Vec<Self> {
        let mut insts = Vec::new();
        let mut first = true;
        for i in 0..4u8 {
            let chunk = ((value >> (i * 16)) & 0xFFFF) as u16;
            if chunk != 0 || (first && i == 3) {
                if first {
                    insts.push(Arm64Inst::movz(rd, chunk, i * 16));
                    first = false;
                } else {
                    insts.push(Arm64Inst::movk(rd, chunk, i * 16));
                }
            }
        }
        if insts.is_empty() {
            insts.push(Arm64Inst::movz(rd, 0, 0));
        }
        insts
    }

    // === Load/Store ===

    /// LDR Rt, [Rn, #offset]  (unsigned offset, scaled)
    pub fn ldr(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        let size = rt.size_bits();
        let scale = if size == 3 { 8 } else { 4 };
        assert!(offset >= 0 && offset % scale == 0, "offset must be aligned and non-negative for unsigned offset LDR");
        let imm12 = offset / scale;
        Arm64Inst::LdrImm {
            size, imm9: 0, rn, rt, imm12, mode: LdrImmMode::UnsignedOffset,
        }
    }

    /// LDUR Rt, [Rn, #offset]  (unscaled offset, supports negative)
    pub fn ldur(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        Arm64Inst::LdurGen {
            size: rt.size_bits(), imm9: offset, rn, rt,
        }
    }

    /// STR Rt, [Rn, #offset]  (unsigned offset, scaled)
    pub fn str(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        let size = rt.size_bits();
        let scale = if size == 3 { 8 } else { 4 };
        assert!(offset >= 0 && offset % scale == 0, "offset must be aligned and non-negative for unsigned offset STR");
        let imm12 = offset / scale;
        Arm64Inst::StrImm {
            size, imm9: 0, rn, rt, imm12, mode: StrImmMode::UnsignedOffset,
        }
    }

    /// STUR Rt, [Rn, #offset]  (unscaled offset, supports negative)
    pub fn stur(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        Arm64Inst::SturGen {
            size: rt.size_bits(), imm9: offset, rn, rt,
        }
    }

    /// LDP Rt1, Rt2, [Rn, #offset]  (signed offset mode)
    pub fn ldp(rt1: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, offset: i32, mode: LdpMode) -> Self {
        let opc = rt1.opc();
        let scale = if opc == 2 { 8 } else { 4 };
        let imm7 = offset / scale;
        Arm64Inst::LdpGen { opc, imm7, rt2, rn, rt: rt1, mode }
    }

    /// STP Rt1, Rt2, [Rn, #offset]  (signed offset mode)
    pub fn stp(rt1: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, offset: i32, mode: StpMode) -> Self {
        let opc = rt1.opc();
        let scale = if opc == 2 { 8 } else { 4 };
        let imm7 = offset / scale;
        Arm64Inst::StpGen { opc, imm7, rt2, rn, rt: rt1, mode }
    }

    /// LDRB Wt, [Xn, #offset]  (unsigned offset)
    pub fn ldrb(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        assert!(offset >= 0, "LDRB unsigned offset must be non-negative");
        Arm64Inst::Ldrb { imm12: offset, rn, rt }
    }

    // === Branch ===

    /// BL offset (in instructions, not bytes — will be multiplied by 4 internally)
    pub fn bl(offset_insns: i32) -> Self {
        Arm64Inst::Bl { imm26: offset_insns }
    }

    /// BLR Xn
    pub fn blr(rn: Arm64Reg) -> Self {
        Arm64Inst::Blr { rn }
    }

    /// RET  (defaults to X30)
    pub fn ret() -> Self {
        Arm64Inst::Ret { rn: super::reg::X30 }
    }

    /// B.cond offset (in instructions)
    pub fn b_cond(cond: Arm64Cond, offset_insns: i32) -> Self {
        Arm64Inst::BCond { imm19: offset_insns, cond: cond as i32 }
    }

    /// CBZ Rt, offset (in instructions)
    pub fn cbz(rt: Arm64Reg, offset_insns: i32) -> Self {
        Arm64Inst::Cbz { sf: rt.sf(), imm19: offset_insns, rt }
    }

    /// CBNZ Rt, offset (in instructions)
    pub fn cbnz(rt: Arm64Reg, offset_insns: i32) -> Self {
        Arm64Inst::Cbnz { sf: rt.sf(), imm19: offset_insns, rt }
    }

    /// BRK #imm16
    pub fn brk(imm16: u16) -> Self {
        Arm64Inst::Brk { imm16: imm16 as i32 }
    }

    // === Logical ===

    /// AND Rd, Rn, Rm
    pub fn and(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::AndShifted {
            sf: rd.sf(), shift: 0, rm, imm6: 0, rn, rd,
        }
    }

    /// ORR Rd, Rn, Rm
    pub fn orr(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::OrrShifted {
            sf: rd.sf(), shift: 0, rm, imm6: 0, rn, rd,
        }
    }

    /// EOR Rd, Rn, Rm
    pub fn eor(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::EorShifted {
            sf: rd.sf(), shift: 0, rm, imm6: 0, rn, rd,
        }
    }

    // === Multiply/Divide ===

    /// MUL Rd, Rn, Rm  (alias for MADD Rd, Rn, Rm, XZR)
    pub fn mul(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        let zr = match rd.size {
            RegSize::W32 => super::reg::WZR,
            RegSize::X64 => XZR,
        };
        Arm64Inst::Madd { sf: rd.sf(), rm, ra: zr, rn, rd }
    }

    /// SDIV Rd, Rn, Rm
    pub fn sdiv(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::Sdiv { sf: rd.sf(), rm, rn, rd }
    }
}
