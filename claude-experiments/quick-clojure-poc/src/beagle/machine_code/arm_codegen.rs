#![allow(clippy::identity_op)]
#![allow(clippy::unusual_byte_groupings)]

use std::ops::Shl;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Size {
    S32,
    S64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Register {
    pub size: Size,
    pub index: u8,
}

impl Register {
    pub fn sf(&self) -> i32 {
        match self.size {
            Size::S32 => 0,
            Size::S64 => 1,
        }
    }

    pub fn from_index(index: usize) -> Register {
        Register {
            index: index as u8,
            size: Size::S64,
        }
    }
}

impl Register {
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

pub const SP: Register = Register {
    index: 31,
    size: Size::S64,
};

pub fn truncate_imm<T: Into<i32>, const WIDTH: usize>(imm: T) -> u32 {
    let value: i32 = imm.into();
    let masked = (value as u32) & ((1 << WIDTH) - 1);

    // Assert that we didn't drop any bits by truncating.
    if value >= 0 {
        assert_eq!(value as u32, masked);
    } else {
        assert_eq!(value as u32, masked | (u32::MAX << WIDTH));
    }

    masked
}

pub const X0: Register = Register {
    index: 0,
    size: Size::S64,
};
pub const X1: Register = Register {
    index: 1,
    size: Size::S64,
};
pub const X2: Register = Register {
    index: 2,
    size: Size::S64,
};
pub const X3: Register = Register {
    index: 3,
    size: Size::S64,
};
pub const X4: Register = Register {
    index: 4,
    size: Size::S64,
};
pub const X5: Register = Register {
    index: 5,
    size: Size::S64,
};
pub const X6: Register = Register {
    index: 6,
    size: Size::S64,
};
pub const X7: Register = Register {
    index: 7,
    size: Size::S64,
};
pub const X8: Register = Register {
    index: 8,
    size: Size::S64,
};
pub const X9: Register = Register {
    index: 9,
    size: Size::S64,
};
pub const X10: Register = Register {
    index: 10,
    size: Size::S64,
};
pub const X11: Register = Register {
    index: 11,
    size: Size::S64,
};
pub const X12: Register = Register {
    index: 12,
    size: Size::S64,
};
pub const X13: Register = Register {
    index: 13,
    size: Size::S64,
};
pub const X14: Register = Register {
    index: 14,
    size: Size::S64,
};
pub const X15: Register = Register {
    index: 15,
    size: Size::S64,
};
pub const X16: Register = Register {
    index: 16,
    size: Size::S64,
};
pub const X17: Register = Register {
    index: 17,
    size: Size::S64,
};
pub const X18: Register = Register {
    index: 18,
    size: Size::S64,
};
pub const X19: Register = Register {
    index: 19,
    size: Size::S64,
};
pub const X20: Register = Register {
    index: 20,
    size: Size::S64,
};
pub const X21: Register = Register {
    index: 21,
    size: Size::S64,
};
pub const X22: Register = Register {
    index: 22,
    size: Size::S64,
};
pub const X23: Register = Register {
    index: 23,
    size: Size::S64,
};
pub const X24: Register = Register {
    index: 24,
    size: Size::S64,
};
pub const X25: Register = Register {
    index: 25,
    size: Size::S64,
};
pub const X26: Register = Register {
    index: 26,
    size: Size::S64,
};
pub const X27: Register = Register {
    index: 27,
    size: Size::S64,
};
pub const X28: Register = Register {
    index: 28,
    size: Size::S64,
};
pub const X29: Register = Register {
    index: 29,
    size: Size::S64,
};
pub const X30: Register = Register {
    index: 30,
    size: Size::S64,
};
pub const ZERO_REGISTER: Register = Register {
    index: 31,
    size: Size::S64,
};
#[derive(Debug)]
pub enum ArmAsm {
    /// ADR -- A64
    /// Form PC-relative address
    /// ADR  <Xd>, <label>
    Adr {
        immlo: i32,
        immhi: i32,
        rd: Register,
    },
    /// ADD (immediate) -- A64
    /// Add (immediate)
    /// ADD  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
    /// ADD  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
    AddAddsubImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Register,
        rd: Register,
    },
    /// ADD (shifted register) -- A64
    /// Add (shifted register)
    /// ADD  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ADD  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    AddAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// AND (immediate) -- A64
    /// Bitwise AND (immediate)
    /// AND  <Wd|WSP>, <Wn>, #<imm>
    /// AND  <Xd|SP>, <Xn>, #<imm>
    AndLogImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// AND (shifted register) -- A64
    /// Bitwise AND (shifted register)
    /// AND  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// AND  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    AndLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// B.cond -- A64
    /// Branch conditionally
    /// B.<cond>  <label>
    BCond { imm19: i32, cond: i32 },
    /// BL -- A64
    /// Branch with Link
    /// BL  <label>
    Bl { imm26: i32 },
    /// BLR -- A64
    /// Branch with Link to Register
    /// BLR  <Xn>
    Blr { rn: Register },
    /// BRK -- A64
    /// Breakpoint instruction
    /// BRK  #<imm>
    Brk { imm16: i32 },
    /// CAS, CASA, CASAL, CASL -- A64
    /// Compare and Swap word or doubleword in memory
    /// CAS  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASA  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASAL  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASL  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CAS  <Xs>, <Xt>, [<Xn|SP>{,#0}]
    /// CASA  <Xs>, <Xt>, [<Xn|SP>{,#0}]
    /// CASAL  <Xs>, <Xt>, [<Xn|SP>{,#0}]
    /// CASL  <Xs>, <Xt>, [<Xn|SP>{,#0}]
    Cas {
        size: i32,
        l: i32,
        rs: Register,
        o0: i32,
        rn: Register,
        rt: Register,
    },
    /// CMP (shifted register) -- A64
    /// Compare (shifted register)
    /// CMP  <Wn>, <Wm>{, <shift> #<amount>}
    /// SUBS WZR, <Wn>, <Wm> {, <shift> #<amount>}
    /// CMP  <Xn>, <Xm>{, <shift> #<amount>}
    /// SUBS XZR, <Xn>, <Xm> {, <shift> #<amount>}
    CmpSubsAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
    },
    /// CSET -- A64
    /// Conditional Set
    /// CSET  <Wd>, <cond>
    /// CSINC <Wd>, WZR, WZR, invert(<cond>)
    /// CSET  <Xd>, <cond>
    /// CSINC <Xd>, XZR, XZR, invert(<cond>)
    CsetCsinc { sf: i32, cond: i32, rd: Register },
    /// LDAR -- A64
    /// Load-Acquire Register
    /// LDAR  <Wt>, [<Xn|SP>{,#0}]
    /// LDAR  <Xt>, [<Xn|SP>{,#0}]
    Ldar {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// LDP -- A64
    /// Load Pair of Registers
    /// LDP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
    /// LDP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
    /// LDP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
    /// LDP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
    /// LDP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
    /// LDP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    LdpGen {
        opc: i32,
        imm7: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
        class_selector: LdpGenSelector,
    },
    /// LDR (immediate) -- A64
    /// Load Register (immediate)
    /// LDR  <Wt>, [<Xn|SP>], #<simm>
    /// LDR  <Xt>, [<Xn|SP>], #<simm>
    /// LDR  <Wt>, [<Xn|SP>, #<simm>]!
    /// LDR  <Xt>, [<Xn|SP>, #<simm>]!
    /// LDR  <Wt>, [<Xn|SP>{, #<pimm>}]
    /// LDR  <Xt>, [<Xn|SP>{, #<pimm>}]
    LdrImmGen {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: LdrImmGenSelector,
    },
    /// LDR (register) -- A64
    /// Load Register (register)
    /// LDR  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// LDR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    LdrRegGen {
        size: i32,
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// LDUR -- A64
    /// Load Register (unscaled)
    /// LDUR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDUR  <Xt>, [<Xn|SP>{, #<simm>}]
    LdurGen {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LSL (register) -- A64
    /// Logical Shift Left (register)
    /// LSL  <Wd>, <Wn>, <Wm>
    /// LSLV <Wd>, <Wn>, <Wm>
    /// LSL  <Xd>, <Xn>, <Xm>
    /// LSLV <Xd>, <Xn>, <Xm>
    LslLslv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// LSL (immediate) -- A64
    /// Logical Shift Left (immediate)
    /// LSL  <Wd>, <Wn>, #<shift>
    /// UBFM <Wd>, <Wn>, #(-<shift> MOD 32), #(31-<shift>)
    /// LSL  <Xd>, <Xn>, #<shift>
    /// UBFM <Xd>, <Xn>, #(-<shift> MOD 64), #(63-<shift>)
    LslUbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// LSR (immediate) -- A64
    /// Logical Shift Right (immediate)
    /// LSR  <Wd>, <Wn>, #<shift>
    /// UBFM <Wd>, <Wn>, #<shift>, #31
    /// LSR  <Xd>, <Xn>, #<shift>
    /// UBFM <Xd>, <Xn>, #<shift>, #63
    LsrUbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// LSR (register) -- A64
    /// Logical Shift Right (register)
    /// LSR  <Wd>, <Wn>, <Wm>
    /// LSRV <Wd>, <Wn>, <Wm>
    /// LSR  <Xd>, <Xn>, <Xm>
    /// LSRV <Xd>, <Xn>, <Xm>
    LsrLsrv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// ASR (register) -- A64
    /// Arithmetic Shift Right (register)
    /// ASR  <Wd>, <Wn>, <Wm>
    /// ASRV <Wd>, <Wn>, <Wm>
    /// ASR  <Xd>, <Xn>, <Xm>
    /// ASRV <Xd>, <Xn>, <Xm>
    AsrAsrv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// ASR (immediate) -- A64
    /// Arithmetic Shift Right (immediate)
    /// ASR  <Wd>, <Wn>, #<shift>
    /// SBFM <Wd>, <Wn>, #<shift>, #31
    /// ASR  <Xd>, <Xn>, #<shift>
    /// SBFM <Xd>, <Xn>, #<shift>, #63
    AsrSbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// EOR (shifted register) -- A64
    /// Bitwise Exclusive OR (shifted register)
    /// EOR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// EOR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    EorLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// MADD -- A64
    /// Multiply-Add
    /// MADD  <Wd>, <Wn>, <Wm>, <Wa>
    /// MADD  <Xd>, <Xn>, <Xm>, <Xa>
    Madd {
        sf: i32,
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// MOV (to/from SP) -- A64
    /// MOV  <Wd|WSP>, <Wn|WSP>
    /// ADD <Wd|WSP>, <Wn|WSP>, #0
    /// MOV  <Xd|SP>, <Xn|SP>
    /// ADD <Xd|SP>, <Xn|SP>, #0
    MovAddAddsubImm { sf: i32, rn: Register, rd: Register },
    /// MOV (register) -- A64
    /// Move (register)
    /// MOV  <Wd>, <Wm>
    /// ORR <Wd>, WZR, <Wm>
    /// MOV  <Xd>, <Xm>
    /// ORR <Xd>, XZR, <Xm>
    MovOrrLogShift { sf: i32, rm: Register, rd: Register },
    /// MOVK -- A64
    /// Move wide with keep
    /// MOVK  <Wd>, #<imm>{, LSL #<shift>}
    /// MOVK  <Xd>, #<imm>{, LSL #<shift>}
    Movk {
        sf: i32,
        hw: i32,
        imm16: i32,
        rd: Register,
    },
    /// MOVZ -- A64
    /// Move wide with zero
    /// MOVZ  <Wd>, #<imm>{, LSL #<shift>}
    /// MOVZ  <Xd>, #<imm>{, LSL #<shift>}
    Movz {
        sf: i32,
        hw: i32,
        imm16: i32,
        rd: Register,
    },
    /// ORR (shifted register) -- A64
    /// Bitwise OR (shifted register)
    /// ORR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ORR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    OrrLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// RET -- A64
    /// Return from subroutine
    /// RET  {<Xn>}
    Ret { rn: Register },
    /// SDIV -- A64
    /// Signed Divide
    /// SDIV  <Wd>, <Wn>, <Wm>
    /// SDIV  <Xd>, <Xn>, <Xm>
    Sdiv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// STLR -- A64
    /// Store-Release Register
    /// STLR  <Wt>, [<Xn|SP>{,#0}]
    /// STLR  <Xt>, [<Xn|SP>{,#0}]
    Stlr {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// STP -- A64
    /// Store Pair of Registers
    /// STP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
    /// STP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
    /// STP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
    /// STP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
    /// STP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
    /// STP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    StpGen {
        opc: i32,
        imm7: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
        class_selector: StpGenSelector,
    },
    /// STR (immediate) -- A64
    /// Store Register (immediate)
    /// STR  <Wt>, [<Xn|SP>], #<simm>
    /// STR  <Xt>, [<Xn|SP>], #<simm>
    /// STR  <Wt>, [<Xn|SP>, #<simm>]!
    /// STR  <Xt>, [<Xn|SP>, #<simm>]!
    /// STR  <Wt>, [<Xn|SP>{, #<pimm>}]
    /// STR  <Xt>, [<Xn|SP>{, #<pimm>}]
    StrImmGen {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: StrImmGenSelector,
    },
    /// STR (register) -- A64
    /// Store Register (register)
    /// STR  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// STR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    StrRegGen {
        size: i32,
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// STUR -- A64
    /// Store Register (unscaled)
    /// STUR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// STUR  <Xt>, [<Xn|SP>{, #<simm>}]
    SturGen {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// SUB (immediate) -- A64
    /// Subtract (immediate)
    /// SUB  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
    /// SUB  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
    SubAddsubImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Register,
        rd: Register,
    },
    /// SUB (shifted register) -- A64
    /// Subtract (shifted register)
    /// SUB  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// SUB  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    SubAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// SUBS (shifted register) -- A64
    /// Subtract (shifted register), setting flags
    /// SUBS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// SUBS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    SubsAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// FMOV (register) -- A64
    /// Floating-point Move register without conversion
    /// FMOV  <Hd>, <Hn>
    /// FMOV  <Sd>, <Sn>
    /// FMOV  <Dd>, <Dn>
    FmovFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FMOV (general) -- A64
    /// Floating-point Move to or from general-purpose register without conversion
    /// FMOV  <Wd>, <Hn>
    /// FMOV  <Xd>, <Hn>
    /// FMOV  <Hd>, <Wn>
    /// FMOV  <Sd>, <Wn>
    /// FMOV  <Wd>, <Sn>
    /// FMOV  <Hd>, <Xn>
    /// FMOV  <Dd>, <Xn>
    /// FMOV  <Vd>.D[1], <Xn>
    /// FMOV  <Xd>, <Dn>
    /// FMOV  <Xd>, <Vn>.D[1]
    FmovFloatGen {
        sf: i32,
        ftype: i32,
        rmode: i32,
        opcode: i32,
        rn: Register,
        rd: Register,
    },
    /// FADD (scalar) -- A64
    /// Floating-point Add (scalar)
    /// FADD  <Hd>, <Hn>, <Hm>
    /// FADD  <Sd>, <Sn>, <Sm>
    /// FADD  <Dd>, <Dn>, <Dm>
    FaddFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FSUB (scalar) -- A64
    /// Floating-point Subtract (scalar)
    /// FSUB  <Hd>, <Hn>, <Hm>
    /// FSUB  <Sd>, <Sn>, <Sm>
    /// FSUB  <Dd>, <Dn>, <Dm>
    FsubFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FMUL (scalar) -- A64
    /// Floating-point Multiply (scalar)
    /// FMUL  <Hd>, <Hn>, <Hm>
    /// FMUL  <Sd>, <Sn>, <Sm>
    /// FMUL  <Dd>, <Dn>, <Dm>
    FmulFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FDIV (scalar) -- A64
    /// Floating-point Divide (scalar)
    /// FDIV  <Hd>, <Hn>, <Hm>
    /// FDIV  <Sd>, <Sn>, <Sm>
    /// FDIV  <Dd>, <Dn>, <Dm>
    FdivFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
}
#[derive(Debug, PartialEq, Eq)]
pub enum LdpGenSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LdrImmGenSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StpGenSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StrImmGenSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}
impl ArmAsm {
    pub fn encode(&self) -> u32 {
        match self {
            ArmAsm::Adr { immlo, immhi, rd } => {
                0b0_00_10000_0000000000000000000_00000
                    | ((*immlo as u32) << 29)
                    | ((*immhi as u32) << 5)
                    | (rd << 0)
            }
            ArmAsm::AddAddsubImm {
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
            ArmAsm::AddAddsubShift {
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
            ArmAsm::AndLogImm {
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
            ArmAsm::AndLogShift {
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
            ArmAsm::BCond { imm19, cond } => {
                0b0101010_0_0000000000000000000_0_0000
                    | (truncate_imm::<_, 19>(*imm19) << 5)
                    | ((*cond as u32) << 0)
            }
            ArmAsm::Bl { imm26 } => {
                0b1_00101_00000000000000000000000000 | (truncate_imm::<_, 26>(*imm26) << 0)
            }
            ArmAsm::Blr { rn } => 0b1101011_0_0_01_11111_0000_0_0_00000_00000 | (rn << 5),
            ArmAsm::Brk { imm16 } => {
                0b11010100_001_0000000000000000_000_00 | ((*imm16 as u32) << 5)
            }
            ArmAsm::Cas {
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
            ArmAsm::CmpSubsAddsubShift {
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
            ArmAsm::CsetCsinc { sf, cond, rd } => {
                0b0_0_0_11010100_11111_0000_0_1_11111_00000
                    | ((*sf as u32) << 31)
                    | ((*cond as u32) << 12)
                    | (rd << 0)
            }
            ArmAsm::Ldar { size, rn, rt } => {
                0b00_001000_1_1_0_11111_1_11111_00000_00000
                    | ((*size as u32) << 30)
                    | (rn << 5)
                    | (rt << 0)
            }
            ArmAsm::LdpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                class_selector,
            } => match class_selector {
                LdpGenSelector::PostIndex => {
                    0b00_101_0_001_1_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdpGenSelector::PreIndex => {
                    0b00_101_0_011_1_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdpGenSelector::SignedOffset => {
                    0b00_101_0_010_1_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            ArmAsm::LdrImmGen {
                size,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                LdrImmGenSelector::PostIndex => {
                    0b00_111_0_00_01_0_000000000_01_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdrImmGenSelector::PreIndex => {
                    0b00_111_0_00_01_0_000000000_11_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                LdrImmGenSelector::UnsignedOffset => {
                    0b00_111_0_01_01_000000000000_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 12>(*imm12) << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            ArmAsm::LdrRegGen {
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
            ArmAsm::LdurGen { size, imm9, rn, rt } => {
                0b00_111_0_00_01_0_000000000_00_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 9>(*imm9) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            ArmAsm::LsrLsrv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_01_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::AsrAsrv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_10_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::AsrSbfm {
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
            ArmAsm::LslUbfm {
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
            ArmAsm::LslLslv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_00_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::LsrUbfm {
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
            ArmAsm::EorLogShift {
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
            ArmAsm::Madd { sf, rm, ra, rn, rd } => {
                0b0_00_11011_000_00000_0_00000_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (ra << 10)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::MovAddAddsubImm { sf, rn, rd } => {
                0b0_0_0_100010_0_000000000000_00000_00000
                    | ((*sf as u32) << 31)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::MovOrrLogShift { sf, rm, rd } => {
                0b0_01_01010_00_0_00000_000000_11111_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rd << 0)
            }
            ArmAsm::Movk { sf, hw, imm16, rd } => {
                0b0_11_100101_00_0000000000000000_00000
                    | ((*sf as u32) << 31)
                    | ((*hw as u32) << 21)
                    | ((*imm16 as u32) << 5)
                    | (rd << 0)
            }
            ArmAsm::Movz { sf, hw, imm16, rd } => {
                0b0_10_100101_00_0000000000000000_00000
                    | ((*sf as u32) << 31)
                    | ((*hw as u32) << 21)
                    | ((*imm16 as u32) << 5)
                    | (rd << 0)
            }
            ArmAsm::OrrLogShift {
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
            ArmAsm::Ret { rn } => 0b1101011_0_0_10_11111_0000_0_0_00000_00000 | (rn << 5),
            ArmAsm::Sdiv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_00001_1_00000_00000
                    | ((*sf as u32) << 31)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::Stlr { size, rn, rt } => {
                0b00_001000_1_0_0_11111_1_11111_00000_00000
                    | ((*size as u32) << 30)
                    | (rn << 5)
                    | (rt << 0)
            }
            ArmAsm::StpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                class_selector,
            } => match class_selector {
                StpGenSelector::PostIndex => {
                    0b00_101_0_001_0_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                StpGenSelector::PreIndex => {
                    0b00_101_0_011_0_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
                StpGenSelector::SignedOffset => {
                    0b00_101_0_010_0_0000000_00000_00000_00000
                        | ((*opc as u32) << 30)
                        | (truncate_imm::<_, 7>(*imm7) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            ArmAsm::StrImmGen {
                size,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                StrImmGenSelector::PostIndex => {
                    0b00_111_0_00_00_0_000000000_01_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                StrImmGenSelector::PreIndex => {
                    0b00_111_0_00_00_0_000000000_11_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 9>(*imm9) << 12)
                        | (rn << 5)
                        | (rt << 0)
                }
                StrImmGenSelector::UnsignedOffset => {
                    0b00_111_0_01_00_000000000000_00000_00000
                        | ((*size as u32) << 30)
                        | (truncate_imm::<_, 12>(*imm12) << 10)
                        | (rn << 5)
                        | (rt << 0)
                }
            },
            ArmAsm::StrRegGen {
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
            ArmAsm::SturGen { size, imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_00_00000_00000
                    | ((*size as u32) << 30)
                    | (truncate_imm::<_, 9>(*imm9) << 12)
                    | (rn << 5)
                    | (rt << 0)
            }
            ArmAsm::SubAddsubImm {
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
            ArmAsm::SubAddsubShift {
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
            ArmAsm::SubsAddsubShift {
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
            ArmAsm::FmovFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0000_00_10000_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::FmovFloatGen {
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
            ArmAsm::FaddFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_001_0_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::FsubFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_001_1_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::FmulFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_0_000_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
            ArmAsm::FdivFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_0001_10_00000_00000
                    | ((*ftype as u32) << 22)
                    | (rm << 16)
                    | (rn << 5)
                    | (rd << 0)
            }
        }
    }
}
