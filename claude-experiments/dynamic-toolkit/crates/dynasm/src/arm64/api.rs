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
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
            rd,
        }
    }

    /// ADD Rd, Rn, #imm12
    pub fn add_imm(rd: Arm64Reg, rn: Arm64Reg, imm12: i32) -> Self {
        Arm64Inst::AddImm {
            sf: rd.sf(),
            sh: 0,
            imm12,
            rn,
            rd,
        }
    }

    /// SUB Rd, Rn, Rm
    pub fn sub(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::SubShifted {
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
            rd,
        }
    }

    /// SUB Rd, Rn, #imm12
    pub fn sub_imm(rd: Arm64Reg, rn: Arm64Reg, imm12: i32) -> Self {
        Arm64Inst::SubImm {
            sf: rd.sf(),
            sh: 0,
            imm12,
            rn,
            rd,
        }
    }

    /// CMP Rn, Rm  (alias for SUBS XZR, Rn, Rm)
    pub fn cmp(rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::CmpShifted {
            sf: rn.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
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
            sf: rn.sf(),
            sh: 0,
            imm12,
            rn,
            rd: zr,
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
            Arm64Inst::MovSp {
                sf: rd.sf(),
                rn: rm,
                rd,
            }
        } else {
            Arm64Inst::MovReg {
                sf: rd.sf(),
                rm,
                rd,
            }
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
        assert!(
            offset >= 0 && offset % scale == 0,
            "offset must be aligned and non-negative for unsigned offset LDR"
        );
        let imm12 = offset / scale;
        Arm64Inst::LdrImm {
            size,
            imm9: 0,
            rn,
            rt,
            imm12,
            mode: LdrImmMode::UnsignedOffset,
        }
    }

    /// LDUR Rt, [Rn, #offset]  (unscaled offset, supports negative)
    pub fn ldur(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        Arm64Inst::LdurGen {
            size: rt.size_bits(),
            imm9: offset,
            rn,
            rt,
        }
    }

    /// STR Rt, [Rn, #offset]  (unsigned offset, scaled)
    pub fn str(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        let size = rt.size_bits();
        let scale = if size == 3 { 8 } else { 4 };
        assert!(
            offset >= 0 && offset % scale == 0,
            "offset must be aligned and non-negative for unsigned offset STR"
        );
        let imm12 = offset / scale;
        Arm64Inst::StrImm {
            size,
            imm9: 0,
            rn,
            rt,
            imm12,
            mode: StrImmMode::UnsignedOffset,
        }
    }

    /// STUR Rt, [Rn, #offset]  (unscaled offset, supports negative)
    pub fn stur(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        Arm64Inst::SturGen {
            size: rt.size_bits(),
            imm9: offset,
            rn,
            rt,
        }
    }

    /// LDP Rt1, Rt2, [Rn, #offset]  (signed offset mode)
    pub fn ldp(rt1: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, offset: i32, mode: LdpMode) -> Self {
        let opc = rt1.opc();
        let scale = if opc == 2 { 8 } else { 4 };
        let imm7 = offset / scale;
        Arm64Inst::LdpGen {
            opc,
            imm7,
            rt2,
            rn,
            rt: rt1,
            mode,
        }
    }

    /// STP Rt1, Rt2, [Rn, #offset]  (signed offset mode)
    pub fn stp(rt1: Arm64Reg, rt2: Arm64Reg, rn: Arm64Reg, offset: i32, mode: StpMode) -> Self {
        let opc = rt1.opc();
        let scale = if opc == 2 { 8 } else { 4 };
        let imm7 = offset / scale;
        Arm64Inst::StpGen {
            opc,
            imm7,
            rt2,
            rn,
            rt: rt1,
            mode,
        }
    }

    /// LDRB Wt, [Xn, #offset]  (unsigned offset)
    pub fn ldrb(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        assert!(offset >= 0, "LDRB unsigned offset must be non-negative");
        Arm64Inst::Ldrb {
            imm12: offset,
            rn,
            rt,
        }
    }

    // === Branch ===

    /// BL offset (in instructions, not bytes — will be multiplied by 4 internally)
    pub fn bl(offset_insns: i32) -> Self {
        Arm64Inst::Bl {
            imm26: offset_insns,
        }
    }

    /// BLR Xn
    pub fn blr(rn: Arm64Reg) -> Self {
        Arm64Inst::Blr { rn }
    }

    /// RET  (defaults to X30)
    pub fn ret() -> Self {
        Arm64Inst::Ret {
            rn: super::reg::X30,
        }
    }

    /// B.cond offset (in instructions)
    pub fn b_cond(cond: Arm64Cond, offset_insns: i32) -> Self {
        Arm64Inst::BCond {
            imm19: offset_insns,
            cond: cond as i32,
        }
    }

    /// CBZ Rt, offset (in instructions)
    pub fn cbz(rt: Arm64Reg, offset_insns: i32) -> Self {
        Arm64Inst::Cbz {
            sf: rt.sf(),
            imm19: offset_insns,
            rt,
        }
    }

    /// CBNZ Rt, offset (in instructions)
    pub fn cbnz(rt: Arm64Reg, offset_insns: i32) -> Self {
        Arm64Inst::Cbnz {
            sf: rt.sf(),
            imm19: offset_insns,
            rt,
        }
    }

    /// BRK #imm16
    pub fn brk(imm16: u16) -> Self {
        Arm64Inst::Brk {
            imm16: imm16 as i32,
        }
    }

    // === Logical ===

    /// AND Rd, Rn, Rm
    pub fn and(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::AndShifted {
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
            rd,
        }
    }

    /// ORR Rd, Rn, Rm
    pub fn orr(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::OrrShifted {
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
            rd,
        }
    }

    /// EOR Rd, Rn, Rm
    pub fn eor(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::EorShifted {
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn,
            rd,
        }
    }

    // === Multiply/Divide ===

    /// MUL Rd, Rn, Rm  (alias for MADD Rd, Rn, Rm, XZR)
    pub fn mul(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        let zr = match rd.size {
            RegSize::W32 => super::reg::WZR,
            RegSize::X64 => XZR,
        };
        Arm64Inst::Madd {
            sf: rd.sf(),
            rm,
            ra: zr,
            rn,
            rd,
        }
    }

    /// SDIV Rd, Rn, Rm
    pub fn sdiv(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::Sdiv {
            sf: rd.sf(),
            rm,
            rn,
            rd,
        }
    }

    /// UDIV Rd, Rn, Rm
    pub fn udiv(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::Udiv {
            sf: rd.sf(),
            rm,
            rn,
            rd,
        }
    }

    /// B offset (unconditional branch, offset in instructions)
    pub fn b(offset_insns: i32) -> Self {
        Arm64Inst::B {
            imm26: offset_insns,
        }
    }

    /// CSEL Rd, Rn, Rm, cond
    pub fn csel(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, cond: Arm64Cond) -> Self {
        Arm64Inst::Csel {
            sf: rd.sf(),
            rm,
            cond: cond as i32,
            rn,
            rd,
        }
    }

    /// MVN Rd, Rm  (alias for ORN Rd, XZR, Rm)
    pub fn mvn(rd: Arm64Reg, rm: Arm64Reg) -> Self {
        let zr = match rd.size {
            RegSize::W32 => super::reg::WZR,
            RegSize::X64 => XZR,
        };
        Arm64Inst::OrnShifted {
            sf: rd.sf(),
            shift: 0,
            rm,
            imm6: 0,
            rn: zr,
            rd,
        }
    }

    /// SXTW Xd, Wn  (sign-extend 32 to 64)
    pub fn sxtw(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        Arm64Inst::Sbfm {
            sf: 1,
            n: 0,
            immr: 0,
            imms: 31,
            rn,
            rd,
        }
    }

    /// SXTB Xd, Wn  (sign-extend 8 to 64)
    pub fn sxtb(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        Arm64Inst::Sbfm {
            sf: rd.sf(),
            n: if rd.sf() == 1 { 1 } else { 0 },
            immr: 0,
            imms: 7,
            rn,
            rd,
        }
    }

    /// UXTB Wd, Wn  (zero-extend byte)
    pub fn uxtb(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        Arm64Inst::Ubfm {
            sf: 0,
            n: 0,
            immr: 0,
            imms: 7,
            rn,
            rd,
        }
    }

    /// SCVTF Dd, Xn (signed int to double)
    pub fn scvtf_to_double(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        // ftype=01 for double, sf from source reg
        Arm64Inst::ScvtfIntToFloat {
            sf: rn.sf(),
            ftype: 1,
            rn,
            rd,
        }
    }

    /// FCVTZS Xd, Dn (double to signed int, round toward zero)
    pub fn fcvtzs_from_double(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        Arm64Inst::FcvtzsFloatToInt {
            sf: rd.sf(),
            ftype: 1,
            rn,
            rd,
        }
    }

    /// FNEG Dd, Dn
    pub fn fneg(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        Arm64Inst::FnegFloat { ftype: 1, rn, rd } // ftype=1 for double
    }

    /// FCSEL Dd, Dn, Dm, cond
    pub fn fcsel(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg, cond: Arm64Cond) -> Self {
        Arm64Inst::FcselFloat {
            ftype: 1,
            rm,
            cond: cond as i32,
            rn,
            rd,
        }
    }

    /// LDR Dt, [Xn, #offset]  (FP, unsigned offset, 8-byte aligned for D regs)
    pub fn ldr_fp(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        assert!(
            offset >= 0 && offset % 8 == 0,
            "FP LDR offset must be 8-byte aligned and non-negative"
        );
        Arm64Inst::LdrFpImm {
            size: 3,
            imm12: offset / 8,
            rn,
            rt,
        }
    }

    /// STR Dt, [Xn, #offset]  (FP, unsigned offset, 8-byte aligned for D regs)
    pub fn str_fp(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        assert!(
            offset >= 0 && offset % 8 == 0,
            "FP STR offset must be 8-byte aligned and non-negative"
        );
        Arm64Inst::StrFpImm {
            size: 3,
            imm12: offset / 8,
            rn,
            rt,
        }
    }

    /// LDUR Dt, [Xn, #offset]  (FP, unscaled offset)
    pub fn ldur_fp(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        Arm64Inst::LdurFp {
            size: 3,
            imm9: offset,
            rn,
            rt,
        }
    }

    /// STUR Dt, [Xn, #offset]  (FP, unscaled offset)
    pub fn stur_fp(rt: Arm64Reg, rn: Arm64Reg, offset: i32) -> Self {
        Arm64Inst::SturFp {
            size: 3,
            imm9: offset,
            rn,
            rt,
        }
    }

    /// FMOV Dd, Xn  (general to float)
    pub fn fmov_gp_to_fp(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        // sf=1, ftype=01 (double), rmode=00, opcode=111
        Arm64Inst::FmovFloatGen {
            sf: 1,
            ftype: 1,
            rmode: 0,
            opcode: 7,
            rn,
            rd,
        }
    }

    /// FMOV Xd, Dn  (float to general)
    pub fn fmov_fp_to_gp(rd: Arm64Reg, rn: Arm64Reg) -> Self {
        // sf=1, ftype=01 (double), rmode=00, opcode=110
        Arm64Inst::FmovFloatGen {
            sf: 1,
            ftype: 1,
            rmode: 0,
            opcode: 6,
            rn,
            rd,
        }
    }

    /// FCMP Dn, Dm (double compare, sets flags)
    pub fn fcmp_double(rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::FcmpFloat { ftype: 1, rm, rn }
    }

    /// FADD Dd, Dn, Dm (double)
    pub fn fadd(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::FaddFloat {
            ftype: 1,
            rm,
            rn,
            rd,
        }
    }

    /// FSUB Dd, Dn, Dm (double)
    pub fn fsub(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::FsubFloat {
            ftype: 1,
            rm,
            rn,
            rd,
        }
    }

    /// FMUL Dd, Dn, Dm (double)
    pub fn fmul(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::FmulFloat {
            ftype: 1,
            rm,
            rn,
            rd,
        }
    }

    /// FDIV Dd, Dn, Dm (double)
    pub fn fdiv(rd: Arm64Reg, rn: Arm64Reg, rm: Arm64Reg) -> Self {
        Arm64Inst::FdivFloat {
            ftype: 1,
            rm,
            rn,
            rd,
        }
    }
}
