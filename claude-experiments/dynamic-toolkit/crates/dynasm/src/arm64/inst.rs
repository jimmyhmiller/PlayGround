use super::encoding::truncate_imm;
use super::reg::Arm64Reg;

/// Addressing mode selector for LDP.
#[derive(Debug, PartialEq, Eq)]
pub enum LdpMode {
    PostIndex,
    PreIndex,
    SignedOffset,
}

/// Addressing mode selector for LDR immediate.
#[derive(Debug, PartialEq, Eq)]
pub enum LdrImmMode {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

/// Addressing mode selector for STP.
#[derive(Debug, PartialEq, Eq)]
pub enum StpMode {
    PostIndex,
    PreIndex,
    SignedOffset,
}

/// Addressing mode selector for STR immediate.
#[derive(Debug, PartialEq, Eq)]
pub enum StrImmMode {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

/// ARM64 condition codes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Arm64Cond {
    EQ = 0b0000,
    NE = 0b0001,
    CS = 0b0010, // HS
    CC = 0b0011, // LO
    MI = 0b0100,
    PL = 0b0101,
    VS = 0b0110,
    VC = 0b0111,
    HI = 0b1000,
    LS = 0b1001,
    GE = 0b1010,
    LT = 0b1011,
    GT = 0b1100,
    LE = 0b1101,
    AL = 0b1110,
}

impl Arm64Cond {
    /// Invert the condition (flip the least-significant bit).
    pub fn invert(self) -> Self {
        use Arm64Cond::*;
        match self {
            EQ => NE,
            NE => EQ,
            CS => CC,
            CC => CS,
            MI => PL,
            PL => MI,
            VS => VC,
            VC => VS,
            HI => LS,
            LS => HI,
            GE => LT,
            LT => GE,
            GT => LE,
            LE => GT,
            AL => AL,
        }
    }
}

/// ARM64 instruction set — raw encoding variants.
///
/// Each variant maps directly to an ARM64 instruction encoding.
/// For a high-level constructor API, see [`super::api`].
#[derive(Debug)]
pub enum Arm64Inst {
    // === Address generation ===
    Adr {
        immlo: i32,
        immhi: i32,
        rd: Arm64Reg,
    },

    // === Arithmetic (immediate) ===
    AddImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    SubImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Arithmetic (shifted register) ===
    AddShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    SubShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    SubsShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    SubsImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === CMP (shifted register) — alias of SUBS with Rd=XZR ===
    CmpShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
    },

    // === CSET — alias of CSINC ===
    Cset {
        sf: i32,
        cond: i32,
        rd: Arm64Reg,
    },

    // === Logical (immediate) ===
    AndImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Logical (shifted register) ===
    AndShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    OrrShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    EorShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Move ===
    MovReg {
        sf: i32,
        rm: Arm64Reg,
        rd: Arm64Reg,
    },
    MovSp {
        sf: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    Movz {
        sf: i32,
        hw: i32,
        imm16: i32,
        rd: Arm64Reg,
    },
    Movk {
        sf: i32,
        hw: i32,
        imm16: i32,
        rd: Arm64Reg,
    },

    // === Shift ===
    LslReg {
        sf: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    LslImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    LsrReg {
        sf: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    LsrImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    AsrReg {
        sf: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    AsrImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Multiply ===
    Madd {
        sf: i32,
        rm: Arm64Reg,
        ra: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    Msub {
        sf: i32,
        rm: Arm64Reg,
        ra: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    Sdiv {
        sf: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Branch ===
    Bl {
        imm26: i32,
    },
    Blr {
        rn: Arm64Reg,
    },
    Ret {
        rn: Arm64Reg,
    },
    BCond {
        imm19: i32,
        cond: i32,
    },
    Cbz {
        sf: i32,
        imm19: i32,
        rt: Arm64Reg,
    },
    Cbnz {
        sf: i32,
        imm19: i32,
        rt: Arm64Reg,
    },
    Brk {
        imm16: i32,
    },

    // === Load/Store ===
    LdrImm {
        size: i32,
        imm9: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
        imm12: i32,
        mode: LdrImmMode,
    },
    LdrReg {
        size: i32,
        rm: Arm64Reg,
        option: i32,
        s: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    LdurGen {
        size: i32,
        imm9: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    Ldar {
        size: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    LdpGen {
        opc: i32,
        imm7: i32,
        rt2: Arm64Reg,
        rn: Arm64Reg,
        rt: Arm64Reg,
        mode: LdpMode,
    },

    StrImm {
        size: i32,
        imm9: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
        imm12: i32,
        mode: StrImmMode,
    },
    StrReg {
        size: i32,
        rm: Arm64Reg,
        option: i32,
        s: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    SturGen {
        size: i32,
        imm9: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    Stlr {
        size: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    StpGen {
        opc: i32,
        imm7: i32,
        rt2: Arm64Reg,
        rn: Arm64Reg,
        rt: Arm64Reg,
        mode: StpMode,
    },

    // === Byte load (LDRB) ===
    Ldrb {
        imm12: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },

    // === Atomic ===
    Cas {
        size: i32,
        l: i32,
        rs: Arm64Reg,
        o0: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },

    // === Unconditional branch ===
    B {
        imm26: i32,
    },

    // === Conditional select ===
    Csel {
        sf: i32,
        rm: Arm64Reg,
        cond: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === ORN (shifted register) — for MVN ===
    OrnShifted {
        sf: i32,
        shift: i32,
        rm: Arm64Reg,
        imm6: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Bitfield ===
    Sbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    Ubfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Unsigned divide ===
    Udiv {
        sf: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Int/Float conversions ===
    ScvtfIntToFloat {
        sf: i32,
        ftype: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FcvtzsFloatToInt {
        sf: i32,
        ftype: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Float negate ===
    FnegFloat {
        ftype: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === Float conditional select ===
    FcselFloat {
        ftype: i32,
        rm: Arm64Reg,
        cond: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },

    // === FP load/store (V=1) ===
    LdrFpImm {
        size: i32,
        imm12: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    StrFpImm {
        size: i32,
        imm12: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    LdurFp {
        size: i32,
        imm9: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },
    SturFp {
        size: i32,
        imm9: i32,
        rn: Arm64Reg,
        rt: Arm64Reg,
    },

    // === Move NOT ===
    Movn {
        sf: i32,
        hw: i32,
        imm16: i32,
        rd: Arm64Reg,
    },

    // === Floating point ===
    FmovFloat {
        ftype: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FmovFloatGen {
        sf: i32,
        ftype: i32,
        rmode: i32,
        opcode: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FaddFloat {
        ftype: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FsubFloat {
        ftype: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FmulFloat {
        ftype: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FdivFloat {
        ftype: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FrintzFloat {
        ftype: i32,
        rn: Arm64Reg,
        rd: Arm64Reg,
    },
    FcmpFloat {
        ftype: i32,
        rm: Arm64Reg,
        rn: Arm64Reg,
    },
}

impl Arm64Inst {
    /// Encode this instruction to a 32-bit word.
    pub fn encode(&self) -> u32 {
        match self {
            Arm64Inst::Adr { immlo, immhi, rd } => {
                0b0_00_10000_0000000000000000000_00000
                    | (((*immlo as u32) & 0x3) << 29)
                    | (((*immhi as u32) & 0x7FFFF) << 5)
                    | (rd << 0)
            }
            Arm64Inst::AddImm {
                sf,
                sh,
                imm12,
                rn,
                rd,
            } => {
                0b0_0_0_100010_0_000000000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*sh as u32) << 22)
                    | (truncate_imm::<_, 12>(*imm12) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::SubImm {
                sf,
                sh,
                imm12,
                rn,
                rd,
            } => {
                0b0_1_0_100010_0_000000000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*sh as u32) << 22)
                    | (truncate_imm::<_, 12>(*imm12) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::AddShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_0_0_01011_00_0_00000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::SubShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_1_0_01011_00_0_00000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::SubsImm {
                sf,
                sh,
                imm12,
                rn,
                rd,
            } => {
                // SUBS (immediate) = SUB (immediate) with S flag (bit 29)
                0b0_1_1_100010_0_000000000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*sh as u32) << 22)
                    | (truncate_imm::<_, 12>(*imm12) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::SubsShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_1_1_01011_00_0_00000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::CmpShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
            } => {
                0b0_1_1_01011_00_0_00000_000000_00000_11111
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
            }
            Arm64Inst::Cset { sf, cond, rd } => {
                0b0_0_0_11010100_11111_0000_0_1_11111_00000
                    | ((*sf as u32) << 31)
                    | ((*cond as u32) << 12)
                    | (rd << 0)
            }
            Arm64Inst::AndImm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_00_100100_0_000000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*n as u32) << 22)
                    | ((*immr as u32) << 16)
                    | ((*imms as u32) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::AndShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_00_01010_00_0_00000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::OrrShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_01_01010_00_0_00000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::EorShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_10_01010_00_0_00000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::MovReg { sf, rm, rd } => {
                0b0_01_01010_00_0_00000_000000_11111_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rd << 0)
            }
            Arm64Inst::MovSp { sf, rn, rd } => {
                0b0_0_0_100010_0_000000000000_00000_00000
                    | ((*sf as u32) << 31)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Movz { sf, hw, imm16, rd } => {
                0b0_10_100101_00_0000000000000000_00000
                    | ((*sf as u32) << 31)
                    | ((*hw as u32) << 21)
                    | ((*imm16 as u32) << 5)
                    | (rd << 0)
            }
            Arm64Inst::Movk { sf, hw, imm16, rd } => {
                0b0_11_100101_00_0000000000000000_00000
                    | ((*sf as u32) << 31)
                    | ((*hw as u32) << 21)
                    | ((*imm16 as u32) << 5)
                    | (rd << 0)
            }
            Arm64Inst::LslReg { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_00_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::LslImm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_10_100110_0_000000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*n as u32) << 22)
                    | ((*immr as u32) << 16)
                    | ((*imms as u32) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::LsrReg { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_01_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::LsrImm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_10_100110_0_000000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*n as u32) << 22)
                    | ((*immr as u32) << 16)
                    | ((*imms as u32) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::AsrReg { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_10_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::AsrImm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_00_100110_0_000000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*n as u32) << 22)
                    | ((*immr as u32) << 16)
                    | ((*imms as u32) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Madd { sf, rm, ra, rn, rd } => {
                0b0_00_11011_000_00000_0_00000_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (ra << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Msub { sf, rm, ra, rn, rd } => {
                0b0_00_11011_000_00000_1_00000_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (ra << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Sdiv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_00001_1_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Bl { imm26 } => {
                0b1_00101_00000000000000000000000000 | (truncate_imm::<_, 26>(*imm26) << 0)
            }
            Arm64Inst::Blr { rn } => 0b1101011_0_0_01_11111_0000_0_0_00000_00000 | (rn << 5),
            Arm64Inst::Ret { rn } => 0b1101011_0_0_10_11111_0000_0_0_00000_00000 | (rn << 5),
            Arm64Inst::BCond { imm19, cond } => {
                0b0101010_0_0000000000000000000_0_0000
                    | (truncate_imm::<_, 19>(*imm19) << 5)
                    | ((*cond as u32) << 0)
            }
            Arm64Inst::Cbz { sf, imm19, rt } => {
                0b0_011010_0_0000000000000000000_00000
                    | ((*sf as u32) << 31)
                    | (truncate_imm::<_, 19>(*imm19) << 5)
                    | (rt << 0)
            }
            Arm64Inst::Cbnz { sf, imm19, rt } => {
                0b0_011010_1_0000000000000000000_00000
                    | ((*sf as u32) << 31)
                    | (truncate_imm::<_, 19>(*imm19) << 5)
                    | (rt << 0)
            }
            Arm64Inst::Brk { imm16 } => {
                0b11010100_001_0000000000000000_000_00 | ((*imm16 as u32) << 5)
            }
            Arm64Inst::LdrImm {
                size,
                imm9,
                rn,
                rt,
                imm12,
                mode,
            } => match mode {
                LdrImmMode::PostIndex => {
                    0b00_111_0_00_01_0_000000000_01_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdrImmMode::PreIndex => {
                    0b00_111_0_00_01_0_000000000_11_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdrImmMode::UnsignedOffset => {
                    0b00_111_0_01_01_000000000000_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 12>(*imm12) << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            Arm64Inst::LdrReg {
                size,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_0_00_01_1_00000_000_0_10_00000_00000
                    | ((*size as u32) << 30)
                    | (rm << 16)
                    | ((*option as u32) << 13)
                    | ((*s as u32) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::LdurGen { size, imm9, rn, rt } => {
                0b00_111_0_00_01_0_000000000_00_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 9>(*imm9) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::Ldar { size, rn, rt } => {
                0b00_001000_1_1_0_11111_1_11111_00000_00000
                    | ((*size as u32) << 30)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::LdpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                mode,
            } => match mode {
                LdpMode::PostIndex => {
                    0b00_101_0_001_1_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdpMode::PreIndex => {
                    0b00_101_0_011_1_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdpMode::SignedOffset => {
                    0b00_101_0_010_1_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            Arm64Inst::StrImm {
                size,
                imm9,
                rn,
                rt,
                imm12,
                mode,
            } => match mode {
                StrImmMode::PostIndex => {
                    0b00_111_0_00_00_0_000000000_01_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                StrImmMode::PreIndex => {
                    0b00_111_0_00_00_0_000000000_11_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                StrImmMode::UnsignedOffset => {
                    0b00_111_0_01_00_000000000000_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 12>(*imm12) << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            Arm64Inst::StrReg {
                size,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_0_00_00_1_00000_000_0_10_00000_00000
                    | ((*size as u32) << 30)
                    | (rm << 16)
                    | ((*option as u32) << 13)
                    | ((*s as u32) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::SturGen { size, imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_00_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 9>(*imm9) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::Stlr { size, rn, rt } => {
                0b00_001000_1_0_0_11111_1_11111_00000_00000
                    | ((*size as u32) << 30)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::StpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                mode,
            } => match mode {
                StpMode::PostIndex => {
                    0b00_101_0_001_0_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                StpMode::PreIndex => {
                    0b00_101_0_011_0_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                StpMode::SignedOffset => {
                    0b00_101_0_010_0_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            Arm64Inst::Ldrb { imm12, rn, rt } => {
                // LDRB (unsigned offset): size=00, opc=01
                0b00_111_0_01_01_000000000000_00000_00000
                    | (truncate_imm::<_, 12>(*imm12) << 10)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::Cas {
                size,
                l,
                rs,
                o0,
                rn,
                rt,
            } => {
                0b00_0010001_0_1_00000_0_11111_00000_00000
                    | ((*size as u32) << 30)
                    | ((*l as u32) << 22)
                    | (rs << 16)
                    | ((*o0 as u32) << 15)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::B { imm26 } => {
                // B imm26: 000101 imm26
                0b0_00101_00000000000000000000000000 | (truncate_imm::<_, 26>(*imm26) << 0)
            }
            Arm64Inst::Csel {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                // CSEL: sf 0 0 11010100 rm cond 0 0 rn rd
                0b0_0_0_11010100_00000_0000_0_0_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | ((*cond as u32) << 12)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::OrnShifted {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                // ORN: sf 01 01010 shift 1 rm imm6 rn rd
                0b0_01_01010_00_1_00000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*shift as u32) << 22)
                    | (rm << 16)
                    | (truncate_imm::<_, 6>(*imm6) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Sbfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                // SBFM: sf 00 100110 N immr imms rn rd
                0b0_00_100110_0_000000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*n as u32) << 22)
                    | ((*immr as u32) << 16)
                    | ((*imms as u32) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Ubfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                // UBFM: sf 10 100110 N immr imms rn rd
                0b0_10_100110_0_000000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*n as u32) << 22)
                    | ((*immr as u32) << 16)
                    | ((*imms as u32) << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::Udiv { sf, rm, rn, rd } => {
                // UDIV: sf 0 0 11010110 rm 00001 0 rn rd
                0b0_0_0_11010110_00000_00001_0_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::ScvtfIntToFloat { sf, ftype, rn, rd } => {
                // SCVTF (scalar, integer): sf 0 0 11110 ftype 1 00 010 000000 rn rd
                0b0_0_0_11110_00_1_00_010_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*ftype as u32) << 22)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FcvtzsFloatToInt { sf, ftype, rn, rd } => {
                // FCVTZS (scalar, integer): sf 0 0 11110 ftype 1 11 000 000000 rn rd
                0b0_0_0_11110_00_1_11_000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*ftype as u32) << 22)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FnegFloat { ftype, rn, rd } => {
                // FNEG: 0 0 0 11110 ftype 1 0000 10 10000 rn rd
                0b0_0_0_11110_00_1_000010_10000_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FcselFloat {
                ftype,
                rm,
                cond,
                rn,
                rd,
            } => {
                // FCSEL: 0 0 0 11110 ftype 1 rm cond 11 rn rd
                0b0_0_0_11110_00_1_00000_0000_11_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | ((*cond as u32) << 12)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::LdrFpImm {
                size,
                imm12,
                rn,
                rt,
            } => {
                // LDR (SIMD&FP, unsigned offset): size 1 11 1 1 01 01 imm12 rn rt
                0b00_111_1_01_01_000000000000_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 12>(*imm12) << 10)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::StrFpImm {
                size,
                imm12,
                rn,
                rt,
            } => {
                // STR (SIMD&FP, unsigned offset): size 1 11 1 1 01 00 imm12 rn rt
                0b00_111_1_01_00_000000000000_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 12>(*imm12) << 10)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::LdurFp { size, imm9, rn, rt } => {
                // LDUR (SIMD&FP): size 1 11 1 00 01 0 imm9 00 rn rt
                0b00_111_1_00_01_0_000000000_00_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 9>(*imm9) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::SturFp { size, imm9, rn, rt } => {
                // STUR (SIMD&FP): size 1 11 1 00 00 0 imm9 00 rn rt
                0b00_111_1_00_00_0_000000000_00_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 9>(*imm9) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            Arm64Inst::Movn { sf, hw, imm16, rd } => {
                0b0_00_100101_00_0000000000000000_00000
                    | ((*sf as u32) << 31)
                    | ((*hw as u32) << 21)
                    | ((*imm16 as u32) << 5)
                    | (rd << 0)
            }
            Arm64Inst::FmovFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0000_00_10000_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FmovFloatGen {
                sf,
                ftype,
                rmode,
                opcode,
                rn,
                rd,
            } => {
                0b0_0_0_11110_00_1_00_000_000000_00000_00000
                    | ((*sf as u32) << 31)
                    | ((*ftype as u32) << 22)
                    | ((*rmode as u32) << 19)
                    | ((*opcode as u32) << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FaddFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_001_0_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FsubFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_001_1_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FmulFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_0_000_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FdivFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_0001_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FrintzFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001011_10000_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rn << 5)
                    | (rd << 0)
            }
            Arm64Inst::FcmpFloat { ftype, rm, rn } => {
                0x1E202000 | ((*ftype as u32) << 22) | (rm << 16) | (rn << 5)
            }
        }
    }
}
