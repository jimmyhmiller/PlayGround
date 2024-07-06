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
    /// ADC -- A64
    /// Add with Carry
    /// ADC  <Wd>, <Wn>, <Wm>
    /// ADC  <Xd>, <Xn>, <Xm>
    Adc {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// ADCS -- A64
    /// Add with Carry, setting flags
    /// ADCS  <Wd>, <Wn>, <Wm>
    /// ADCS  <Xd>, <Xn>, <Xm>
    Adcs {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// ADD (extended register) -- A64
    /// Add (extended register)
    /// ADD  <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// ADD  <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    AddAddsubExt {
        sf: i32,
        rm: Register,
        option: i32,
        imm3: i32,
        rn: Register,
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
    /// ADDG -- A64
    /// Add with Tag
    /// ADDG  <Xd|SP>, <Xn|SP>, #<uimm6>, #<uimm4>
    Addg {
        uimm6: i32,
        uimm4: i32,
        xn: i32,
        xd: i32,
    },
    /// ADDS (extended register) -- A64
    /// Add (extended register), setting flags
    /// ADDS  <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// ADDS  <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    AddsAddsubExt {
        sf: i32,
        rm: Register,
        option: i32,
        imm3: i32,
        rn: Register,
        rd: Register,
    },
    /// ADDS (immediate) -- A64
    /// Add (immediate), setting flags
    /// ADDS  <Wd>, <Wn|WSP>, #<imm>{, <shift>}
    /// ADDS  <Xd>, <Xn|SP>, #<imm>{, <shift>}
    AddsAddsubImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Register,
        rd: Register,
    },
    /// ADDS (shifted register) -- A64
    /// Add (shifted register), setting flags
    /// ADDS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ADDS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    AddsAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// ADR -- A64
    /// Form PC-relative address
    /// ADR  <Xd>, <label>
    Adr {
        immlo: i32,
        immhi: i32,
        rd: Register,
    },
    /// ADRP -- A64
    /// Form PC-relative address to 4KB page
    /// ADRP  <Xd>, <label>
    Adrp {
        immlo: i32,
        immhi: i32,
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
    /// ANDS (immediate) -- A64
    /// Bitwise AND (immediate), setting flags
    /// ANDS  <Wd>, <Wn>, #<imm>
    /// ANDS  <Xd>, <Xn>, #<imm>
    AndsLogImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// ANDS (shifted register) -- A64
    /// Bitwise AND (shifted register), setting flags
    /// ANDS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ANDS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    AndsLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
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
    /// ASRV -- A64
    /// Arithmetic Shift Right Variable
    /// ASRV  <Wd>, <Wn>, <Wm>
    /// ASRV  <Xd>, <Xn>, <Xm>
    Asrv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// AT -- A64
    /// Address Translate
    /// AT  <at_op>, <Xt>
    /// SYS #<op1>, C7, <Cm>, #<op2>, <Xt>
    AtSys {
        op1: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// AUTDA, AUTDZA -- A64
    /// Authenticate Data address, using key A
    /// AUTDA  <Xd>, <Xn|SP>
    /// AUTDZA  <Xd>
    Autda { z: i32, rn: Register, rd: Register },
    /// AUTDB, AUTDZB -- A64
    /// Authenticate Data address, using key B
    /// AUTDB  <Xd>, <Xn|SP>
    /// AUTDZB  <Xd>
    Autdb { z: i32, rn: Register, rd: Register },
    /// AUTIA, AUTIA1716, AUTIASP, AUTIAZ, AUTIZA -- A64
    /// Authenticate Instruction address, using key A
    /// AUTIA  <Xd>, <Xn|SP>
    /// AUTIZA  <Xd>
    /// AUTIA1716
    /// AUTIASP
    /// AUTIAZ
    Autia {
        z: i32,
        rn: Register,
        rd: Register,
        crm: i32,
        op2: i32,
        class_selector: AutiaSelector,
    },
    /// AUTIB, AUTIB1716, AUTIBSP, AUTIBZ, AUTIZB -- A64
    /// Authenticate Instruction address, using key B
    /// AUTIB  <Xd>, <Xn|SP>
    /// AUTIZB  <Xd>
    /// AUTIB1716
    /// AUTIBSP
    /// AUTIBZ
    Autib {
        z: i32,
        rn: Register,
        rd: Register,
        crm: i32,
        op2: i32,
        class_selector: AutibSelector,
    },
    /// AXFLAG -- A64
    /// Convert floating-point condition flags from Arm to external format
    /// AXFLAG
    Axflag,
    /// B.cond -- A64
    /// Branch conditionally
    /// B.<cond>  <label>
    BCond { imm19: i32, cond: i32 },
    /// B -- A64
    /// Branch
    /// B  <label>
    BUncond { imm26: i32 },
    /// BC.cond -- A64
    /// Branch Consistent conditionally
    /// BC.<cond>  <label>
    BcCond { imm19: i32, cond: i32 },
    /// BFC -- A64
    /// Bitfield Clear
    /// BFC  <Wd>, #<lsb>, #<width>
    /// BFM <Wd>, WZR, #(-<lsb> MOD 32), #(<width>-1)
    /// BFC  <Xd>, #<lsb>, #<width>
    /// BFM <Xd>, XZR, #(-<lsb> MOD 64), #(<width>-1)
    BfcBfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rd: Register,
    },
    /// BFI -- A64
    /// Bitfield Insert
    /// BFI  <Wd>, <Wn>, #<lsb>, #<width>
    /// BFM  <Wd>, <Wn>, #(-<lsb> MOD 32), #(<width>-1)
    /// BFI  <Xd>, <Xn>, #<lsb>, #<width>
    /// BFM  <Xd>, <Xn>, #(-<lsb> MOD 64), #(<width>-1)
    BfiBfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// BFM -- A64
    /// Bitfield Move
    /// BFM  <Wd>, <Wn>, #<immr>, #<imms>
    /// BFM  <Xd>, <Xn>, #<immr>, #<imms>
    Bfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// BFXIL -- A64
    /// Bitfield extract and insert at low end
    /// BFXIL  <Wd>, <Wn>, #<lsb>, #<width>
    /// BFM  <Wd>, <Wn>, #<lsb>, #(<lsb>+<width>-1)
    /// BFXIL  <Xd>, <Xn>, #<lsb>, #<width>
    /// BFM  <Xd>, <Xn>, #<lsb>, #(<lsb>+<width>-1)
    BfxilBfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// BIC (shifted register) -- A64
    /// Bitwise Bit Clear (shifted register)
    /// BIC  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// BIC  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    BicLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// BICS (shifted register) -- A64
    /// Bitwise Bit Clear (shifted register), setting flags
    /// BICS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// BICS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    Bics {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// BL -- A64
    /// Branch with Link
    /// BL  <label>
    Bl { imm26: i32 },
    /// BLR -- A64
    /// Branch with Link to Register
    /// BLR  <Xn>
    Blr { rn: Register },
    /// BLRAA, BLRAAZ, BLRAB, BLRABZ -- A64
    /// Branch with Link to Register, with pointer authentication
    /// BLRAAZ  <Xn>
    /// BLRAA  <Xn>, <Xm|SP>
    /// BLRABZ  <Xn>
    /// BLRAB  <Xn>, <Xm|SP>
    Blra {
        z: i32,
        m: i32,
        rn: Register,
        rm: Register,
    },
    /// BR -- A64
    /// Branch to Register
    /// BR  <Xn>
    Br { rn: Register },
    /// BRAA, BRAAZ, BRAB, BRABZ -- A64
    /// Branch to Register, with pointer authentication
    /// BRAAZ  <Xn>
    /// BRAA  <Xn>, <Xm|SP>
    /// BRABZ  <Xn>
    /// BRAB  <Xn>, <Xm|SP>
    Bra {
        z: i32,
        m: i32,
        rn: Register,
        rm: Register,
    },
    /// BRK -- A64
    /// Breakpoint instruction
    /// BRK  #<imm>
    Brk { imm16: i32 },
    /// BTI -- A64
    /// Branch Target Identification
    /// BTI  {<targets>}
    Bti { op2: i32 },
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
    /// CASB, CASAB, CASALB, CASLB -- A64
    /// Compare and Swap byte in memory
    /// CASAB  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASALB  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASB  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASLB  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    Casb {
        l: i32,
        rs: Register,
        o0: i32,
        rn: Register,
        rt: Register,
    },
    /// CASH, CASAH, CASALH, CASLH -- A64
    /// Compare and Swap halfword in memory
    /// CASAH  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASALH  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASH  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// CASLH  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    Cash {
        l: i32,
        rs: Register,
        o0: i32,
        rn: Register,
        rt: Register,
    },
    /// CASP, CASPA, CASPAL, CASPL -- A64
    /// Compare and Swap Pair of words or doublewords in memory
    /// CASP  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
    /// CASPA  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
    /// CASPAL  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
    /// CASPL  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
    /// CASP  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
    /// CASPA  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
    /// CASPAL  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
    /// CASPL  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
    Casp {
        sz: i32,
        l: i32,
        rs: Register,
        o0: i32,
        rn: Register,
        rt: Register,
    },
    /// CBNZ -- A64
    /// Compare and Branch on Nonzero
    /// CBNZ  <Wt>, <label>
    /// CBNZ  <Xt>, <label>
    Cbnz { sf: i32, imm19: i32, rt: Register },
    /// CBZ -- A64
    /// Compare and Branch on Zero
    /// CBZ  <Wt>, <label>
    /// CBZ  <Xt>, <label>
    Cbz { sf: i32, imm19: i32, rt: Register },
    /// CCMN (immediate) -- A64
    /// Conditional Compare Negative (immediate)
    /// CCMN  <Wn>, #<imm>, #<nzcv>, <cond>
    /// CCMN  <Xn>, #<imm>, #<nzcv>, <cond>
    CcmnImm {
        sf: i32,
        imm5: i32,
        cond: i32,
        rn: Register,
        nzcv: i32,
    },
    /// CCMN (register) -- A64
    /// Conditional Compare Negative (register)
    /// CCMN  <Wn>, <Wm>, #<nzcv>, <cond>
    /// CCMN  <Xn>, <Xm>, #<nzcv>, <cond>
    CcmnReg {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        nzcv: i32,
    },
    /// CCMP (immediate) -- A64
    /// Conditional Compare (immediate)
    /// CCMP  <Wn>, #<imm>, #<nzcv>, <cond>
    /// CCMP  <Xn>, #<imm>, #<nzcv>, <cond>
    CcmpImm {
        sf: i32,
        imm5: i32,
        cond: i32,
        rn: Register,
        nzcv: i32,
    },
    /// CCMP (register) -- A64
    /// Conditional Compare (register)
    /// CCMP  <Wn>, <Wm>, #<nzcv>, <cond>
    /// CCMP  <Xn>, <Xm>, #<nzcv>, <cond>
    CcmpReg {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        nzcv: i32,
    },
    /// CFINV -- A64
    /// Invert Carry Flag
    /// CFINV
    Cfinv,
    /// CFP -- A64
    /// Control Flow Prediction Restriction by Context
    /// CFP  RCTX, <Xt>
    /// SYS #3, C7, C3, #4, <Xt>
    CfpSys { rt: Register },
    /// CINC -- A64
    /// Conditional Increment
    /// CINC  <Wd>, <Wn>, <cond>
    /// CSINC <Wd>, <Wn>, <Wn>, invert(<cond>)
    /// CINC  <Xd>, <Xn>, <cond>
    /// CSINC <Xd>, <Xn>, <Xn>, invert(<cond>)
    CincCsinc {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// CINV -- A64
    /// Conditional Invert
    /// CINV  <Wd>, <Wn>, <cond>
    /// CSINV <Wd>, <Wn>, <Wn>, invert(<cond>)
    /// CINV  <Xd>, <Xn>, <cond>
    /// CSINV <Xd>, <Xn>, <Xn>, invert(<cond>)
    CinvCsinv {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// CLREX -- A64
    /// Clear Exclusive
    /// CLREX  {#<imm>}
    Clrex { crm: i32 },
    /// CLS -- A64
    /// Count Leading Sign bits
    /// CLS  <Wd>, <Wn>
    /// CLS  <Xd>, <Xn>
    ClsInt { sf: i32, rn: Register, rd: Register },
    /// CLZ -- A64
    /// Count Leading Zeros
    /// CLZ  <Wd>, <Wn>
    /// CLZ  <Xd>, <Xn>
    ClzInt { sf: i32, rn: Register, rd: Register },
    /// CMN (extended register) -- A64
    /// Compare Negative (extended register)
    /// CMN  <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// ADDS WZR, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// CMN  <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    /// ADDS XZR, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    CmnAddsAddsubExt {
        sf: i32,
        rm: Register,
        option: i32,
        imm3: i32,
        rn: Register,
    },
    /// CMN (immediate) -- A64
    /// Compare Negative (immediate)
    /// CMN  <Wn|WSP>, #<imm>{, <shift>}
    /// ADDS WZR, <Wn|WSP>, #<imm> {, <shift>}
    /// CMN  <Xn|SP>, #<imm>{, <shift>}
    /// ADDS XZR, <Xn|SP>, #<imm> {, <shift>}
    CmnAddsAddsubImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Register,
    },
    /// CMN (shifted register) -- A64
    /// Compare Negative (shifted register)
    /// CMN  <Wn>, <Wm>{, <shift> #<amount>}
    /// ADDS WZR, <Wn>, <Wm> {, <shift> #<amount>}
    /// CMN  <Xn>, <Xm>{, <shift> #<amount>}
    /// ADDS XZR, <Xn>, <Xm> {, <shift> #<amount>}
    CmnAddsAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
    },
    /// CMP (extended register) -- A64
    /// Compare (extended register)
    /// CMP  <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// SUBS WZR, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// CMP  <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    /// SUBS XZR, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    CmpSubsAddsubExt {
        sf: i32,
        rm: Register,
        option: i32,
        imm3: i32,
        rn: Register,
    },
    /// CMP (immediate) -- A64
    /// Compare (immediate)
    /// CMP  <Wn|WSP>, #<imm>{, <shift>}
    /// SUBS WZR, <Wn|WSP>, #<imm> {, <shift>}
    /// CMP  <Xn|SP>, #<imm>{, <shift>}
    /// SUBS XZR, <Xn|SP>, #<imm> {, <shift>}
    CmpSubsAddsubImm {
        sf: i32,
        sh: i32,
        imm12: i32,
        rn: Register,
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
    /// CMPP -- A64
    /// Compare with Tag
    /// CMPP  <Xn|SP>, <Xm|SP>
    /// SUBPS XZR, <Xn|SP>, <Xm|SP>
    CmppSubps { xm: i32, xn: i32 },
    /// CNEG -- A64
    /// Conditional Negate
    /// CNEG  <Wd>, <Wn>, <cond>
    /// CSNEG <Wd>, <Wn>, <Wn>, invert(<cond>)
    /// CNEG  <Xd>, <Xn>, <cond>
    /// CSNEG <Xd>, <Xn>, <Xn>, invert(<cond>)
    CnegCsneg {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// CPP -- A64
    /// Cache Prefetch Prediction Restriction by Context
    /// CPP  RCTX, <Xt>
    /// SYS #3, C7, C3, #7, <Xt>
    CppSys { rt: Register },
    /// CPYFP, CPYFM, CPYFE -- A64
    /// Memory Copy Forward-only
    /// CPYFE  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFM  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFP  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfp {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPN, CPYFMN, CPYFEN -- A64
    /// Memory Copy Forward-only, reads and writes non-temporal
    /// CPYFEN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfpn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPRN, CPYFMRN, CPYFERN -- A64
    /// Memory Copy Forward-only, reads non-temporal
    /// CPYFERN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfprn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPRT, CPYFMRT, CPYFERT -- A64
    /// Memory Copy Forward-only, reads unprivileged
    /// CPYFERT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMRT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPRT  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfprt {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPRTN, CPYFMRTN, CPYFERTN -- A64
    /// Memory Copy Forward-only, reads unprivileged, reads and writes non-temporal
    /// CPYFERTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMRTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPRTN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfprtn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPRTRN, CPYFMRTRN, CPYFERTRN -- A64
    /// Memory Copy Forward-only, reads unprivileged and non-temporal
    /// CPYFERTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMRTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPRTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfprtrn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPRTWN, CPYFMRTWN, CPYFERTWN -- A64
    /// Memory Copy Forward-only, reads unprivileged, writes non-temporal
    /// CPYFERTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMRTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPRTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfprtwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPT, CPYFMT, CPYFET -- A64
    /// Memory Copy Forward-only, reads and writes unprivileged
    /// CPYFET  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPT  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfpt {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPTN, CPYFMTN, CPYFETN -- A64
    /// Memory Copy Forward-only, reads and writes unprivileged and non-temporal
    /// CPYFETN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPTN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfptn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPTRN, CPYFMTRN, CPYFETRN -- A64
    /// Memory Copy Forward-only, reads and writes unprivileged, reads non-temporal
    /// CPYFETRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfptrn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPTWN, CPYFMTWN, CPYFETWN -- A64
    /// Memory Copy Forward-only, reads and writes unprivileged, writes non-temporal
    /// CPYFETWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfptwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPWN, CPYFMWN, CPYFEWN -- A64
    /// Memory Copy Forward-only, writes non-temporal
    /// CPYFEWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfpwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPWT, CPYFMWT, CPYFEWT -- A64
    /// Memory Copy Forward-only, writes unprivileged
    /// CPYFEWT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMWT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPWT  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfpwt {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPWTN, CPYFMWTN, CPYFEWTN -- A64
    /// Memory Copy Forward-only, writes unprivileged, reads and writes non-temporal
    /// CPYFEWTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMWTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPWTN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfpwtn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPWTRN, CPYFMWTRN, CPYFEWTRN -- A64
    /// Memory Copy Forward-only, writes unprivileged, reads non-temporal
    /// CPYFEWTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMWTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPWTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfpwtrn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYFPWTWN, CPYFMWTWN, CPYFEWTWN -- A64
    /// Memory Copy Forward-only, writes unprivileged and non-temporal
    /// CPYFEWTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFMWTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYFPWTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyfpwtwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYP, CPYM, CPYE -- A64
    /// Memory Copy
    /// CPYE  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYM  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYP  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyp {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPN, CPYMN, CPYEN -- A64
    /// Memory Copy, reads and writes non-temporal
    /// CPYEN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpypn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPRN, CPYMRN, CPYERN -- A64
    /// Memory Copy, reads non-temporal
    /// CPYERN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyprn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPRT, CPYMRT, CPYERT -- A64
    /// Memory Copy, reads unprivileged
    /// CPYERT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMRT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPRT  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyprt {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPRTN, CPYMRTN, CPYERTN -- A64
    /// Memory Copy, reads unprivileged, reads and writes non-temporal
    /// CPYERTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMRTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPRTN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyprtn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPRTRN, CPYMRTRN, CPYERTRN -- A64
    /// Memory Copy, reads unprivileged and non-temporal
    /// CPYERTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMRTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPRTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyprtrn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPRTWN, CPYMRTWN, CPYERTWN -- A64
    /// Memory Copy, reads unprivileged, writes non-temporal
    /// CPYERTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMRTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPRTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyprtwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPT, CPYMT, CPYET -- A64
    /// Memory Copy, reads and writes unprivileged
    /// CPYET  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPT  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpypt {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPTN, CPYMTN, CPYETN -- A64
    /// Memory Copy, reads and writes unprivileged and non-temporal
    /// CPYETN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPTN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyptn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPTRN, CPYMTRN, CPYETRN -- A64
    /// Memory Copy, reads and writes unprivileged, reads non-temporal
    /// CPYETRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyptrn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPTWN, CPYMTWN, CPYETWN -- A64
    /// Memory Copy, reads and writes unprivileged, writes non-temporal
    /// CPYETWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpyptwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPWN, CPYMWN, CPYEWN -- A64
    /// Memory Copy, writes non-temporal
    /// CPYEWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpypwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPWT, CPYMWT, CPYEWT -- A64
    /// Memory Copy, writes unprivileged
    /// CPYEWT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMWT  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPWT  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpypwt {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPWTN, CPYMWTN, CPYEWTN -- A64
    /// Memory Copy, writes unprivileged, reads and writes non-temporal
    /// CPYEWTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMWTN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPWTN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpypwtn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPWTRN, CPYMWTRN, CPYEWTRN -- A64
    /// Memory Copy, writes unprivileged, reads non-temporal
    /// CPYEWTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMWTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPWTRN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpypwtrn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CPYPWTWN, CPYMWTWN, CPYEWTWN -- A64
    /// Memory Copy, writes unprivileged and non-temporal
    /// CPYEWTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYMWTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    /// CPYPWTWN  [<Xd>]!, [<Xs>]!, <Xn>!
    Cpypwtwn {
        sz: i32,
        op1: i32,
        rs: Register,
        rn: Register,
        rd: Register,
    },
    /// CRC32B, CRC32H, CRC32W, CRC32X -- A64
    /// CRC32 checksum
    /// CRC32B  <Wd>, <Wn>, <Wm>
    /// CRC32H  <Wd>, <Wn>, <Wm>
    /// CRC32W  <Wd>, <Wn>, <Wm>
    /// CRC32X  <Wd>, <Wn>, <Xm>
    Crc32 {
        sf: i32,
        rm: Register,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// CRC32CB, CRC32CH, CRC32CW, CRC32CX -- A64
    /// CRC32C checksum
    /// CRC32CB  <Wd>, <Wn>, <Wm>
    /// CRC32CH  <Wd>, <Wn>, <Wm>
    /// CRC32CW  <Wd>, <Wn>, <Wm>
    /// CRC32CX  <Wd>, <Wn>, <Xm>
    Crc32c {
        sf: i32,
        rm: Register,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// CSDB -- A64
    /// Consumption of Speculative Data Barrier
    /// CSDB
    Csdb,
    /// CSEL -- A64
    /// Conditional Select
    /// CSEL  <Wd>, <Wn>, <Wm>, <cond>
    /// CSEL  <Xd>, <Xn>, <Xm>, <cond>
    Csel {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// CSET -- A64
    /// Conditional Set
    /// CSET  <Wd>, <cond>
    /// CSINC <Wd>, WZR, WZR, invert(<cond>)
    /// CSET  <Xd>, <cond>
    /// CSINC <Xd>, XZR, XZR, invert(<cond>)
    CsetCsinc { sf: i32, cond: i32, rd: Register },
    /// CSETM -- A64
    /// Conditional Set Mask
    /// CSETM  <Wd>, <cond>
    /// CSINV <Wd>, WZR, WZR, invert(<cond>)
    /// CSETM  <Xd>, <cond>
    /// CSINV <Xd>, XZR, XZR, invert(<cond>)
    CsetmCsinv { sf: i32, cond: i32, rd: Register },
    /// CSINC -- A64
    /// Conditional Select Increment
    /// CSINC  <Wd>, <Wn>, <Wm>, <cond>
    /// CSINC  <Xd>, <Xn>, <Xm>, <cond>
    Csinc {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// CSINV -- A64
    /// Conditional Select Invert
    /// CSINV  <Wd>, <Wn>, <Wm>, <cond>
    /// CSINV  <Xd>, <Xn>, <Xm>, <cond>
    Csinv {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// CSNEG -- A64
    /// Conditional Select Negation
    /// CSNEG  <Wd>, <Wn>, <Wm>, <cond>
    /// CSNEG  <Xd>, <Xn>, <Xm>, <cond>
    Csneg {
        sf: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// DC -- A64
    /// Data Cache operation
    /// DC  <dc_op>, <Xt>
    /// SYS #<op1>, C7, <Cm>, #<op2>, <Xt>
    DcSys {
        op1: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// DCPS1 -- A64
    /// Debug Change PE State to EL1.
    /// DCPS1  {#<imm>}
    Dcps1 { imm16: i32 },
    /// DCPS2 -- A64
    /// Debug Change PE State to EL2.
    /// DCPS2  {#<imm>}
    Dcps2 { imm16: i32 },
    /// DCPS3 -- A64
    /// Debug Change PE State to EL3
    /// DCPS3  {#<imm>}
    Dcps3 { imm16: i32 },
    /// DGH -- A64
    /// Data Gathering Hint
    /// DGH
    Dgh,
    /// DMB -- A64
    /// Data Memory Barrier
    /// DMB  <option>|#<imm>
    Dmb { crm: i32 },
    /// DRPS -- A64

    /// DRPS
    Drps,
    /// DSB -- A64
    /// Data Synchronization Barrier
    /// DSB  <option>|#<imm>
    /// DSB  <option>nXS|#<imm>
    Dsb {
        crm: i32,
        imm2: i32,
        class_selector: DsbSelector,
    },
    /// DVP -- A64
    /// Data Value Prediction Restriction by Context
    /// DVP  RCTX, <Xt>
    /// SYS #3, C7, C3, #5, <Xt>
    DvpSys { rt: Register },
    /// EON (shifted register) -- A64
    /// Bitwise Exclusive OR NOT (shifted register)
    /// EON  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// EON  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    Eon {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// EOR (immediate) -- A64
    /// Bitwise Exclusive OR (immediate)
    /// EOR  <Wd|WSP>, <Wn>, #<imm>
    /// EOR  <Xd|SP>, <Xn>, #<imm>
    EorLogImm {
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
    /// ERET -- A64
    /// Exception Return
    /// ERET
    Eret,
    /// ERETAA, ERETAB -- A64
    /// Exception Return, with pointer authentication
    /// ERETAA
    /// ERETAB
    Ereta { m: i32 },
    /// ESB -- A64
    /// Error Synchronization Barrier
    /// ESB
    Esb,
    /// EXTR -- A64
    /// Extract register
    /// EXTR  <Wd>, <Wn>, <Wm>, #<lsb>
    /// EXTR  <Xd>, <Xn>, <Xm>, #<lsb>
    Extr {
        sf: i32,
        n: i32,
        rm: Register,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// GMI -- A64
    /// Tag Mask Insert
    /// GMI  <Xd>, <Xn|SP>, <Xm>
    Gmi { xm: i32, xn: i32, xd: i32 },
    /// HINT -- A64
    /// Hint instruction
    /// HINT  #<imm>
    Hint { crm: i32, op2: i32 },
    /// HLT -- A64
    /// Halt instruction
    /// HLT  #<imm>
    Hlt { imm16: i32 },
    /// HVC -- A64
    /// Hypervisor Call
    /// HVC  #<imm>
    Hvc { imm16: i32 },
    /// IC -- A64
    /// Instruction Cache operation
    /// IC  <ic_op>{, <Xt>}
    /// SYS #<op1>, C7, <Cm>, #<op2>{, <Xt>}
    IcSys {
        op1: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// IRG -- A64
    /// Insert Random Tag
    /// IRG  <Xd|SP>, <Xn|SP>{, <Xm>}
    Irg { xm: i32, xn: i32, xd: i32 },
    /// ISB -- A64
    /// Instruction Synchronization Barrier
    /// ISB  {<option>|#<imm>}
    Isb { crm: i32 },
    /// LD64B -- A64
    /// Single-copy Atomic 64-byte Load
    /// LD64B  <Xt>, [<Xn|SP> {,#0}]
    Ld64b { rn: Register, rt: Register },
    /// LDADD, LDADDA, LDADDAL, LDADDL -- A64
    /// Atomic add on word or doubleword in memory
    /// LDADD  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADD  <Xs>, <Xt>, [<Xn|SP>]
    /// LDADDA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDADDAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDADDL  <Xs>, <Xt>, [<Xn|SP>]
    Ldadd {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDADDB, LDADDAB, LDADDALB, LDADDLB -- A64
    /// Atomic add on byte in memory
    /// LDADDAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDLB  <Ws>, <Wt>, [<Xn|SP>]
    Ldaddb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDADDH, LDADDAH, LDADDALH, LDADDLH -- A64
    /// Atomic add on halfword in memory
    /// LDADDAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDADDLH  <Ws>, <Wt>, [<Xn|SP>]
    Ldaddh {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDAPR -- A64
    /// Load-Acquire RCpc Register
    /// LDAPR  <Wt>, [<Xn|SP> {,#0}]
    /// LDAPR  <Xt>, [<Xn|SP> {,#0}]
    Ldapr {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAPRB -- A64
    /// Load-Acquire RCpc Register Byte
    /// LDAPRB  <Wt>, [<Xn|SP> {,#0}]
    Ldaprb { rn: Register, rt: Register },
    /// LDAPRH -- A64
    /// Load-Acquire RCpc Register Halfword
    /// LDAPRH  <Wt>, [<Xn|SP> {,#0}]
    Ldaprh { rn: Register, rt: Register },
    /// LDAPUR -- A64
    /// Load-Acquire RCpc Register (unscaled)
    /// LDAPUR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDAPUR  <Xt>, [<Xn|SP>{, #<simm>}]
    LdapurGen {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAPURB -- A64
    /// Load-Acquire RCpc Register Byte (unscaled)
    /// LDAPURB  <Wt>, [<Xn|SP>{, #<simm>}]
    Ldapurb {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAPURH -- A64
    /// Load-Acquire RCpc Register Halfword (unscaled)
    /// LDAPURH  <Wt>, [<Xn|SP>{, #<simm>}]
    Ldapurh {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAPURSB -- A64
    /// Load-Acquire RCpc Register Signed Byte (unscaled)
    /// LDAPURSB  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDAPURSB  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldapursb {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAPURSH -- A64
    /// Load-Acquire RCpc Register Signed Halfword (unscaled)
    /// LDAPURSH  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDAPURSH  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldapursh {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAPURSW -- A64
    /// Load-Acquire RCpc Register Signed Word (unscaled)
    /// LDAPURSW  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldapursw {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAR -- A64
    /// Load-Acquire Register
    /// LDAR  <Wt>, [<Xn|SP>{,#0}]
    /// LDAR  <Xt>, [<Xn|SP>{,#0}]
    Ldar {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// LDARB -- A64
    /// Load-Acquire Register Byte
    /// LDARB  <Wt>, [<Xn|SP>{,#0}]
    Ldarb { rn: Register, rt: Register },
    /// LDARH -- A64
    /// Load-Acquire Register Halfword
    /// LDARH  <Wt>, [<Xn|SP>{,#0}]
    Ldarh { rn: Register, rt: Register },
    /// LDAXP -- A64
    /// Load-Acquire Exclusive Pair of Registers
    /// LDAXP  <Wt1>, <Wt2>, [<Xn|SP>{,#0}]
    /// LDAXP  <Xt1>, <Xt2>, [<Xn|SP>{,#0}]
    Ldaxp {
        sz: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
    },
    /// LDAXR -- A64
    /// Load-Acquire Exclusive Register
    /// LDAXR  <Wt>, [<Xn|SP>{,#0}]
    /// LDAXR  <Xt>, [<Xn|SP>{,#0}]
    Ldaxr {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// LDAXRB -- A64
    /// Load-Acquire Exclusive Register Byte
    /// LDAXRB  <Wt>, [<Xn|SP>{,#0}]
    Ldaxrb { rn: Register, rt: Register },
    /// LDAXRH -- A64
    /// Load-Acquire Exclusive Register Halfword
    /// LDAXRH  <Wt>, [<Xn|SP>{,#0}]
    Ldaxrh { rn: Register, rt: Register },
    /// LDCLR, LDCLRA, LDCLRAL, LDCLRL -- A64
    /// Atomic bit clear on word or doubleword in memory
    /// LDCLR  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLR  <Xs>, <Xt>, [<Xn|SP>]
    /// LDCLRA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDCLRAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDCLRL  <Xs>, <Xt>, [<Xn|SP>]
    Ldclr {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDCLRB, LDCLRAB, LDCLRALB, LDCLRLB -- A64
    /// Atomic bit clear on byte in memory
    /// LDCLRAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRLB  <Ws>, <Wt>, [<Xn|SP>]
    Ldclrb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDCLRH, LDCLRAH, LDCLRALH, LDCLRLH -- A64
    /// Atomic bit clear on halfword in memory
    /// LDCLRAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDCLRLH  <Ws>, <Wt>, [<Xn|SP>]
    Ldclrh {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDEOR, LDEORA, LDEORAL, LDEORL -- A64
    /// Atomic exclusive OR on word or doubleword in memory
    /// LDEOR  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEOR  <Xs>, <Xt>, [<Xn|SP>]
    /// LDEORA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDEORAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDEORL  <Xs>, <Xt>, [<Xn|SP>]
    Ldeor {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDEORB, LDEORAB, LDEORALB, LDEORLB -- A64
    /// Atomic exclusive OR on byte in memory
    /// LDEORAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORLB  <Ws>, <Wt>, [<Xn|SP>]
    Ldeorb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDEORH, LDEORAH, LDEORALH, LDEORLH -- A64
    /// Atomic exclusive OR on halfword in memory
    /// LDEORAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDEORLH  <Ws>, <Wt>, [<Xn|SP>]
    Ldeorh {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDG -- A64
    /// Load Allocation Tag
    /// LDG  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldg { imm9: i32, xn: i32, xt: i32 },
    /// LDGM -- A64
    /// Load Tag Multiple
    /// LDGM  <Xt>, [<Xn|SP>]
    Ldgm { xn: i32, xt: i32 },
    /// LDLAR -- A64
    /// Load LOAcquire Register
    /// LDLAR  <Wt>, [<Xn|SP>{,#0}]
    /// LDLAR  <Xt>, [<Xn|SP>{,#0}]
    Ldlar {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// LDLARB -- A64
    /// Load LOAcquire Register Byte
    /// LDLARB  <Wt>, [<Xn|SP>{,#0}]
    Ldlarb { rn: Register, rt: Register },
    /// LDLARH -- A64
    /// Load LOAcquire Register Halfword
    /// LDLARH  <Wt>, [<Xn|SP>{,#0}]
    Ldlarh { rn: Register, rt: Register },
    /// LDNP -- A64
    /// Load Pair of Registers, with non-temporal hint
    /// LDNP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
    /// LDNP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    LdnpGen {
        opc: i32,
        imm7: i32,
        rt2: Register,
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
    /// LDPSW -- A64
    /// Load Pair of Registers Signed Word
    /// LDPSW  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
    /// LDPSW  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
    /// LDPSW  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    Ldpsw {
        imm7: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
        class_selector: LdpswSelector,
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
    /// LDR (literal) -- A64
    /// Load Register (literal)
    /// LDR  <Wt>, <label>
    /// LDR  <Xt>, <label>
    LdrLitGen { opc: i32, imm19: i32, rt: Register },
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
    /// LDRAA, LDRAB -- A64
    /// Load Register, with pointer authentication
    /// LDRAA  <Xt>, [<Xn|SP>{, #<simm>}]
    /// LDRAA  <Xt>, [<Xn|SP>{, #<simm>}]!
    /// LDRAB  <Xt>, [<Xn|SP>{, #<simm>}]
    /// LDRAB  <Xt>, [<Xn|SP>{, #<simm>}]!
    Ldra {
        m: i32,
        s: i32,
        imm9: i32,
        w: i32,
        rn: Register,
        rt: Register,
    },
    /// LDRB (immediate) -- A64
    /// Load Register Byte (immediate)
    /// LDRB  <Wt>, [<Xn|SP>], #<simm>
    /// LDRB  <Wt>, [<Xn|SP>, #<simm>]!
    /// LDRB  <Wt>, [<Xn|SP>{, #<pimm>}]
    LdrbImm {
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: LdrbImmSelector,
    },
    /// LDRB (register) -- A64
    /// Load Register Byte (register)
    /// LDRB  <Wt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
    /// LDRB  <Wt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
    LdrbReg {
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// LDRH (immediate) -- A64
    /// Load Register Halfword (immediate)
    /// LDRH  <Wt>, [<Xn|SP>], #<simm>
    /// LDRH  <Wt>, [<Xn|SP>, #<simm>]!
    /// LDRH  <Wt>, [<Xn|SP>{, #<pimm>}]
    LdrhImm {
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: LdrhImmSelector,
    },
    /// LDRH (register) -- A64
    /// Load Register Halfword (register)
    /// LDRH  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    LdrhReg {
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// LDRSB (immediate) -- A64
    /// Load Register Signed Byte (immediate)
    /// LDRSB  <Wt>, [<Xn|SP>], #<simm>
    /// LDRSB  <Xt>, [<Xn|SP>], #<simm>
    /// LDRSB  <Wt>, [<Xn|SP>, #<simm>]!
    /// LDRSB  <Xt>, [<Xn|SP>, #<simm>]!
    /// LDRSB  <Wt>, [<Xn|SP>{, #<pimm>}]
    /// LDRSB  <Xt>, [<Xn|SP>{, #<pimm>}]
    LdrsbImm {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: LdrsbImmSelector,
    },
    /// LDRSB (register) -- A64
    /// Load Register Signed Byte (register)
    /// LDRSB  <Wt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
    /// LDRSB  <Wt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
    /// LDRSB  <Xt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
    /// LDRSB  <Xt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
    LdrsbReg {
        opc: i32,
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// LDRSH (immediate) -- A64
    /// Load Register Signed Halfword (immediate)
    /// LDRSH  <Wt>, [<Xn|SP>], #<simm>
    /// LDRSH  <Xt>, [<Xn|SP>], #<simm>
    /// LDRSH  <Wt>, [<Xn|SP>, #<simm>]!
    /// LDRSH  <Xt>, [<Xn|SP>, #<simm>]!
    /// LDRSH  <Wt>, [<Xn|SP>{, #<pimm>}]
    /// LDRSH  <Xt>, [<Xn|SP>{, #<pimm>}]
    LdrshImm {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: LdrshImmSelector,
    },
    /// LDRSH (register) -- A64
    /// Load Register Signed Halfword (register)
    /// LDRSH  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// LDRSH  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    LdrshReg {
        opc: i32,
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// LDRSW (immediate) -- A64
    /// Load Register Signed Word (immediate)
    /// LDRSW  <Xt>, [<Xn|SP>], #<simm>
    /// LDRSW  <Xt>, [<Xn|SP>, #<simm>]!
    /// LDRSW  <Xt>, [<Xn|SP>{, #<pimm>}]
    LdrswImm {
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: LdrswImmSelector,
    },
    /// LDRSW (literal) -- A64
    /// Load Register Signed Word (literal)
    /// LDRSW  <Xt>, <label>
    LdrswLit { imm19: i32, rt: Register },
    /// LDRSW (register) -- A64
    /// Load Register Signed Word (register)
    /// LDRSW  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    LdrswReg {
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// LDSET, LDSETA, LDSETAL, LDSETL -- A64
    /// Atomic bit set on word or doubleword in memory
    /// LDSET  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSET  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSETA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSETAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSETL  <Xs>, <Xt>, [<Xn|SP>]
    Ldset {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSETB, LDSETAB, LDSETALB, LDSETLB -- A64
    /// Atomic bit set on byte in memory
    /// LDSETAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETLB  <Ws>, <Wt>, [<Xn|SP>]
    Ldsetb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSETH, LDSETAH, LDSETALH, LDSETLH -- A64
    /// Atomic bit set on halfword in memory
    /// LDSETAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSETLH  <Ws>, <Wt>, [<Xn|SP>]
    Ldseth {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSMAX, LDSMAXA, LDSMAXAL, LDSMAXL -- A64
    /// Atomic signed maximum on word or doubleword in memory
    /// LDSMAX  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAX  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSMAXA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSMAXAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSMAXL  <Xs>, <Xt>, [<Xn|SP>]
    Ldsmax {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSMAXB, LDSMAXAB, LDSMAXALB, LDSMAXLB -- A64
    /// Atomic signed maximum on byte in memory
    /// LDSMAXAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXLB  <Ws>, <Wt>, [<Xn|SP>]
    Ldsmaxb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSMAXH, LDSMAXAH, LDSMAXALH, LDSMAXLH -- A64
    /// Atomic signed maximum on halfword in memory
    /// LDSMAXAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMAXLH  <Ws>, <Wt>, [<Xn|SP>]
    Ldsmaxh {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSMIN, LDSMINA, LDSMINAL, LDSMINL -- A64
    /// Atomic signed minimum on word or doubleword in memory
    /// LDSMIN  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMIN  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSMINA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSMINAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDSMINL  <Xs>, <Xt>, [<Xn|SP>]
    Ldsmin {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSMINB, LDSMINAB, LDSMINALB, LDSMINLB -- A64
    /// Atomic signed minimum on byte in memory
    /// LDSMINAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINLB  <Ws>, <Wt>, [<Xn|SP>]
    Ldsminb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDSMINH, LDSMINAH, LDSMINALH, LDSMINLH -- A64
    /// Atomic signed minimum on halfword in memory
    /// LDSMINAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDSMINLH  <Ws>, <Wt>, [<Xn|SP>]
    Ldsminh {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDTR -- A64
    /// Load Register (unprivileged)
    /// LDTR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDTR  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldtr {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDTRB -- A64
    /// Load Register Byte (unprivileged)
    /// LDTRB  <Wt>, [<Xn|SP>{, #<simm>}]
    Ldtrb {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDTRH -- A64
    /// Load Register Halfword (unprivileged)
    /// LDTRH  <Wt>, [<Xn|SP>{, #<simm>}]
    Ldtrh {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDTRSB -- A64
    /// Load Register Signed Byte (unprivileged)
    /// LDTRSB  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDTRSB  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldtrsb {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDTRSH -- A64
    /// Load Register Signed Halfword (unprivileged)
    /// LDTRSH  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDTRSH  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldtrsh {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDTRSW -- A64
    /// Load Register Signed Word (unprivileged)
    /// LDTRSW  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldtrsw {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDUMAX, LDUMAXA, LDUMAXAL, LDUMAXL -- A64
    /// Atomic unsigned maximum on word or doubleword in memory
    /// LDUMAX  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAX  <Xs>, <Xt>, [<Xn|SP>]
    /// LDUMAXA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDUMAXAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDUMAXL  <Xs>, <Xt>, [<Xn|SP>]
    Ldumax {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDUMAXB, LDUMAXAB, LDUMAXALB, LDUMAXLB -- A64
    /// Atomic unsigned maximum on byte in memory
    /// LDUMAXAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXLB  <Ws>, <Wt>, [<Xn|SP>]
    Ldumaxb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDUMAXH, LDUMAXAH, LDUMAXALH, LDUMAXLH -- A64
    /// Atomic unsigned maximum on halfword in memory
    /// LDUMAXAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMAXLH  <Ws>, <Wt>, [<Xn|SP>]
    Ldumaxh {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDUMIN, LDUMINA, LDUMINAL, LDUMINL -- A64
    /// Atomic unsigned minimum on word or doubleword in memory
    /// LDUMIN  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINA  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINAL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINL  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMIN  <Xs>, <Xt>, [<Xn|SP>]
    /// LDUMINA  <Xs>, <Xt>, [<Xn|SP>]
    /// LDUMINAL  <Xs>, <Xt>, [<Xn|SP>]
    /// LDUMINL  <Xs>, <Xt>, [<Xn|SP>]
    Ldumin {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDUMINB, LDUMINAB, LDUMINALB, LDUMINLB -- A64
    /// Atomic unsigned minimum on byte in memory
    /// LDUMINAB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINALB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINB  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINLB  <Ws>, <Wt>, [<Xn|SP>]
    Lduminb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// LDUMINH, LDUMINAH, LDUMINALH, LDUMINLH -- A64
    /// Atomic unsigned minimum on halfword in memory
    /// LDUMINAH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINALH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINH  <Ws>, <Wt>, [<Xn|SP>]
    /// LDUMINLH  <Ws>, <Wt>, [<Xn|SP>]
    Lduminh {
        a: i32,
        r: Register,
        rs: Register,
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
    /// LDURB -- A64
    /// Load Register Byte (unscaled)
    /// LDURB  <Wt>, [<Xn|SP>{, #<simm>}]
    Ldurb {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDURH -- A64
    /// Load Register Halfword (unscaled)
    /// LDURH  <Wt>, [<Xn|SP>{, #<simm>}]
    Ldurh {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDURSB -- A64
    /// Load Register Signed Byte (unscaled)
    /// LDURSB  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDURSB  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldursb {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDURSH -- A64
    /// Load Register Signed Halfword (unscaled)
    /// LDURSH  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDURSH  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldursh {
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDURSW -- A64
    /// Load Register Signed Word (unscaled)
    /// LDURSW  <Xt>, [<Xn|SP>{, #<simm>}]
    Ldursw {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// LDXP -- A64
    /// Load Exclusive Pair of Registers
    /// LDXP  <Wt1>, <Wt2>, [<Xn|SP>{,#0}]
    /// LDXP  <Xt1>, <Xt2>, [<Xn|SP>{,#0}]
    Ldxp {
        sz: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
    },
    /// LDXR -- A64
    /// Load Exclusive Register
    /// LDXR  <Wt>, [<Xn|SP>{,#0}]
    /// LDXR  <Xt>, [<Xn|SP>{,#0}]
    Ldxr {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// LDXRB -- A64
    /// Load Exclusive Register Byte
    /// LDXRB  <Wt>, [<Xn|SP>{,#0}]
    Ldxrb { rn: Register, rt: Register },
    /// LDXRH -- A64
    /// Load Exclusive Register Halfword
    /// LDXRH  <Wt>, [<Xn|SP>{,#0}]
    Ldxrh { rn: Register, rt: Register },
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
    /// LSLV -- A64
    /// Logical Shift Left Variable
    /// LSLV  <Wd>, <Wn>, <Wm>
    /// LSLV  <Xd>, <Xn>, <Xm>
    Lslv {
        sf: i32,
        rm: Register,
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
    /// LSRV -- A64
    /// Logical Shift Right Variable
    /// LSRV  <Wd>, <Wn>, <Wm>
    /// LSRV  <Xd>, <Xn>, <Xm>
    Lsrv {
        sf: i32,
        rm: Register,
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
    /// MNEG -- A64
    /// Multiply-Negate
    /// MNEG  <Wd>, <Wn>, <Wm>
    /// MSUB <Wd>, <Wn>, <Wm>, WZR
    /// MNEG  <Xd>, <Xn>, <Xm>
    /// MSUB <Xd>, <Xn>, <Xm>, XZR
    MnegMsub {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// MOV (to/from SP) -- A64

    /// MOV  <Wd|WSP>, <Wn|WSP>
    /// ADD <Wd|WSP>, <Wn|WSP>, #0
    /// MOV  <Xd|SP>, <Xn|SP>
    /// ADD <Xd|SP>, <Xn|SP>, #0
    MovAddAddsubImm { sf: i32, rn: Register, rd: Register },
    /// MOV (inverted wide immediate) -- A64
    /// Move (inverted wide immediate)
    /// MOV  <Wd>, #<imm>
    /// MOVN <Wd>, #<imm16>, LSL #<shift>
    /// MOV  <Xd>, #<imm>
    /// MOVN <Xd>, #<imm16>, LSL #<shift>
    MovMovn {
        sf: i32,
        hw: i32,
        imm16: i32,
        rd: Register,
    },
    /// MOV (wide immediate) -- A64
    /// Move (wide immediate)
    /// MOV  <Wd>, #<imm>
    /// MOVZ <Wd>, #<imm16>, LSL #<shift>
    /// MOV  <Xd>, #<imm>
    /// MOVZ <Xd>, #<imm16>, LSL #<shift>
    MovMovz {
        sf: i32,
        hw: i32,
        imm16: i32,
        rd: Register,
    },
    /// MOV (bitmask immediate) -- A64
    /// Move (bitmask immediate)
    /// MOV  <Wd|WSP>, #<imm>
    /// ORR <Wd|WSP>, WZR, #<imm>
    /// MOV  <Xd|SP>, #<imm>
    /// ORR <Xd|SP>, XZR, #<imm>
    MovOrrLogImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rd: Register,
    },
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
    /// MOVN -- A64
    /// Move wide with NOT
    /// MOVN  <Wd>, #<imm>{, LSL #<shift>}
    /// MOVN  <Xd>, #<imm>{, LSL #<shift>}
    Movn {
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
    /// MRS -- A64
    /// Move System Register
    /// MRS  <Xt>, (<systemreg>|S<op0>_<op1>_<Cn>_<Cm>_<op2>)
    Mrs {
        o0: i32,
        op1: i32,
        crn: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// MSR (immediate) -- A64
    /// Move immediate value to Special Register
    /// MSR  <pstatefield>, #<imm>
    MsrImm { op1: i32, crm: i32, op2: i32 },
    /// MSR (register) -- A64
    /// Move general-purpose register to System Register
    /// MSR  (<systemreg>|S<op0>_<op1>_<Cn>_<Cm>_<op2>), <Xt>
    MsrReg {
        o0: i32,
        op1: i32,
        crn: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// MSUB -- A64
    /// Multiply-Subtract
    /// MSUB  <Wd>, <Wn>, <Wm>, <Wa>
    /// MSUB  <Xd>, <Xn>, <Xm>, <Xa>
    Msub {
        sf: i32,
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// MUL -- A64

    /// MUL  <Wd>, <Wn>, <Wm>
    /// MADD <Wd>, <Wn>, <Wm>, WZR
    /// MUL  <Xd>, <Xn>, <Xm>
    /// MADD <Xd>, <Xn>, <Xm>, XZR
    MulMadd {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// MVN -- A64
    /// Bitwise NOT
    /// MVN  <Wd>, <Wm>{, <shift> #<amount>}
    /// ORN <Wd>, WZR, <Wm>{, <shift> #<amount>}
    /// MVN  <Xd>, <Xm>{, <shift> #<amount>}
    /// ORN <Xd>, XZR, <Xm>{, <shift> #<amount>}
    MvnOrnLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rd: Register,
    },
    /// NEG (shifted register) -- A64
    /// Negate (shifted register)
    /// NEG  <Wd>, <Wm>{, <shift> #<amount>}
    /// SUB  <Wd>, WZR, <Wm> {, <shift> #<amount>}
    /// NEG  <Xd>, <Xm>{, <shift> #<amount>}
    /// SUB  <Xd>, XZR, <Xm> {, <shift> #<amount>}
    NegSubAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rd: Register,
    },
    /// NEGS -- A64
    /// Negate, setting flags
    /// NEGS  <Wd>, <Wm>{, <shift> #<amount>}
    /// SUBS <Wd>, WZR, <Wm> {, <shift> #<amount>}
    /// NEGS  <Xd>, <Xm>{, <shift> #<amount>}
    /// SUBS <Xd>, XZR, <Xm> {, <shift> #<amount>}
    NegsSubsAddsubShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rd: Register,
    },
    /// NGC -- A64
    /// Negate with Carry
    /// NGC  <Wd>, <Wm>
    /// SBC <Wd>, WZR, <Wm>
    /// NGC  <Xd>, <Xm>
    /// SBC <Xd>, XZR, <Xm>
    NgcSbc { sf: i32, rm: Register, rd: Register },
    /// NGCS -- A64
    /// Negate with Carry, setting flags
    /// NGCS  <Wd>, <Wm>
    /// SBCS <Wd>, WZR, <Wm>
    /// NGCS  <Xd>, <Xm>
    /// SBCS <Xd>, XZR, <Xm>
    NgcsSbcs { sf: i32, rm: Register, rd: Register },
    /// NOP -- A64
    /// No Operation
    /// NOP
    Nop,
    /// ORN (shifted register) -- A64
    /// Bitwise OR NOT (shifted register)
    /// ORN  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ORN  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    OrnLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
    /// ORR (immediate) -- A64
    /// Bitwise OR (immediate)
    /// ORR  <Wd|WSP>, <Wn>, #<imm>
    /// ORR  <Xd|SP>, <Xn>, #<imm>
    OrrLogImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
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
    /// PACDA, PACDZA -- A64
    /// Pointer Authentication Code for Data address, using key A
    /// PACDA  <Xd>, <Xn|SP>
    /// PACDZA  <Xd>
    Pacda { z: i32, rn: Register, rd: Register },
    /// PACDB, PACDZB -- A64
    /// Pointer Authentication Code for Data address, using key B
    /// PACDB  <Xd>, <Xn|SP>
    /// PACDZB  <Xd>
    Pacdb { z: i32, rn: Register, rd: Register },
    /// PACGA -- A64
    /// Pointer Authentication Code, using Generic key
    /// PACGA  <Xd>, <Xn>, <Xm|SP>
    Pacga {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// PACIA, PACIA1716, PACIASP, PACIAZ, PACIZA -- A64
    /// Pointer Authentication Code for Instruction address, using key A
    /// PACIA  <Xd>, <Xn|SP>
    /// PACIZA  <Xd>
    /// PACIA1716
    /// PACIASP
    /// PACIAZ
    Pacia {
        z: i32,
        rn: Register,
        rd: Register,
        crm: i32,
        op2: i32,
        class_selector: PaciaSelector,
    },
    /// PACIB, PACIB1716, PACIBSP, PACIBZ, PACIZB -- A64
    /// Pointer Authentication Code for Instruction address, using key B
    /// PACIB  <Xd>, <Xn|SP>
    /// PACIZB  <Xd>
    /// PACIB1716
    /// PACIBSP
    /// PACIBZ
    Pacib {
        z: i32,
        rn: Register,
        rd: Register,
        crm: i32,
        op2: i32,
        class_selector: PacibSelector,
    },
    /// PRFM (immediate) -- A64
    /// Prefetch Memory (immediate)
    /// PRFM  (<prfop>|#<imm5>), [<Xn|SP>{, #<pimm>}]
    PrfmImm {
        imm12: i32,
        rn: Register,
        rt: Register,
    },
    /// PRFM (literal) -- A64
    /// Prefetch Memory (literal)
    /// PRFM  (<prfop>|#<imm5>), <label>
    PrfmLit { imm19: i32, rt: Register },
    /// PRFM (register) -- A64
    /// Prefetch Memory (register)
    /// PRFM  (<prfop>|#<imm5>), [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    PrfmReg {
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// PRFUM -- A64
    /// Prefetch Memory (unscaled offset)
    /// PRFUM (<prfop>|#<imm5>), [<Xn|SP>{, #<simm>}]
    Prfum {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// PSB CSYNC -- A64
    /// Profiling Synchronization Barrier
    /// PSB CSYNC
    Psb,
    /// PSSBB -- A64
    /// Physical Speculative Store Bypass Barrier
    /// PSSBB
    /// DSB #4
    PssbbDsb,
    /// RBIT -- A64
    /// Reverse Bits
    /// RBIT  <Wd>, <Wn>
    /// RBIT  <Xd>, <Xn>
    RbitInt { sf: i32, rn: Register, rd: Register },
    /// RET -- A64
    /// Return from subroutine
    /// RET  {<Xn>}
    Ret { rn: Register },
    /// RETAA, RETAB -- A64
    /// Return from subroutine, with pointer authentication
    /// RETAA
    /// RETAB
    Reta { m: i32 },
    /// REV -- A64
    /// Reverse Bytes
    /// REV  <Wd>, <Wn>
    /// REV  <Xd>, <Xn>
    Rev {
        sf: i32,
        opc: i32,
        rn: Register,
        rd: Register,
    },
    /// REV16 -- A64
    /// Reverse bytes in 16-bit halfwords
    /// REV16  <Wd>, <Wn>
    /// REV16  <Xd>, <Xn>
    Rev16Int { sf: i32, rn: Register, rd: Register },
    /// REV32 -- A64
    /// Reverse bytes in 32-bit words
    /// REV32  <Xd>, <Xn>
    Rev32Int { rn: Register, rd: Register },
    /// REV64 -- A64
    /// Reverse Bytes
    /// REV64  <Xd>, <Xn>
    /// REV  <Xd>, <Xn>
    Rev64Rev { rn: Register, rd: Register },
    /// RMIF -- A64
    /// Rotate, Mask Insert Flags
    /// RMIF  <Xn>, #<shift>, #<mask>
    Rmif { imm6: i32, rn: Register, mask: i32 },
    /// ROR (immediate) -- A64
    /// Rotate right (immediate)
    /// ROR  <Wd>, <Ws>, #<shift>
    /// EXTR <Wd>, <Ws>, <Ws>, #<shift>
    /// ROR  <Xd>, <Xs>, #<shift>
    /// EXTR <Xd>, <Xs>, <Xs>, #<shift>
    RorExtr {
        sf: i32,
        n: i32,
        rm: Register,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// ROR (register) -- A64
    /// Rotate Right (register)
    /// ROR  <Wd>, <Wn>, <Wm>
    /// RORV <Wd>, <Wn>, <Wm>
    /// ROR  <Xd>, <Xn>, <Xm>
    /// RORV <Xd>, <Xn>, <Xm>
    RorRorv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// RORV -- A64
    /// Rotate Right Variable
    /// RORV  <Wd>, <Wn>, <Wm>
    /// RORV  <Xd>, <Xn>, <Xm>
    Rorv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SB -- A64
    /// Speculation Barrier
    /// SB
    Sb,
    /// SBC -- A64
    /// Subtract with Carry
    /// SBC  <Wd>, <Wn>, <Wm>
    /// SBC  <Xd>, <Xn>, <Xm>
    Sbc {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SBCS -- A64
    /// Subtract with Carry, setting flags
    /// SBCS  <Wd>, <Wn>, <Wm>
    /// SBCS  <Xd>, <Xn>, <Xm>
    Sbcs {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SBFIZ -- A64
    /// Signed Bitfield Insert in Zero
    /// SBFIZ  <Wd>, <Wn>, #<lsb>, #<width>
    /// SBFM <Wd>, <Wn>, #(-<lsb> MOD 32), #(<width>-1)
    /// SBFIZ  <Xd>, <Xn>, #<lsb>, #<width>
    /// SBFM <Xd>, <Xn>, #(-<lsb> MOD 64), #(<width>-1)
    SbfizSbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// SBFM -- A64
    /// Signed Bitfield Move
    /// SBFM  <Wd>, <Wn>, #<immr>, #<imms>
    /// SBFM  <Xd>, <Xn>, #<immr>, #<imms>
    Sbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// SBFX -- A64
    /// Signed Bitfield Extract
    /// SBFX  <Wd>, <Wn>, #<lsb>, #<width>
    /// SBFM <Wd>, <Wn>, #<lsb>, #(<lsb>+<width>-1)
    /// SBFX  <Xd>, <Xn>, #<lsb>, #<width>
    /// SBFM <Xd>, <Xn>, #<lsb>, #(<lsb>+<width>-1)
    SbfxSbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
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
    /// SETF8, SETF16 -- A64
    /// Evaluation of 8 or 16 bit flag values
    /// SETF8  <Wn>
    /// SETF16  <Wn>
    Setf { sz: i32, rn: Register },
    /// SETGP, SETGM, SETGE -- A64
    /// Memory Set with tag setting
    /// SETGE  [<Xd>]!, <Xn>!, <Xs>
    /// SETGM  [<Xd>]!, <Xn>!, <Xs>
    /// SETGP  [<Xd>]!, <Xn>!, <Xs>
    Setgp {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SETGPN, SETGMN, SETGEN -- A64
    /// Memory Set with tag setting, non-temporal
    /// SETGEN  [<Xd>]!, <Xn>!, <Xs>
    /// SETGMN  [<Xd>]!, <Xn>!, <Xs>
    /// SETGPN  [<Xd>]!, <Xn>!, <Xs>
    Setgpn {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SETGPT, SETGMT, SETGET -- A64
    /// Memory Set with tag setting, unprivileged
    /// SETGET  [<Xd>]!, <Xn>!, <Xs>
    /// SETGMT  [<Xd>]!, <Xn>!, <Xs>
    /// SETGPT  [<Xd>]!, <Xn>!, <Xs>
    Setgpt {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SETGPTN, SETGMTN, SETGETN -- A64
    /// Memory Set with tag setting, unprivileged and non-temporal
    /// SETGETN  [<Xd>]!, <Xn>!, <Xs>
    /// SETGMTN  [<Xd>]!, <Xn>!, <Xs>
    /// SETGPTN  [<Xd>]!, <Xn>!, <Xs>
    Setgptn {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SETP, SETM, SETE -- A64
    /// Memory Set
    /// SETE  [<Xd>]!, <Xn>!, <Xs>
    /// SETM  [<Xd>]!, <Xn>!, <Xs>
    /// SETP  [<Xd>]!, <Xn>!, <Xs>
    Setp {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SETPN, SETMN, SETEN -- A64
    /// Memory Set, non-temporal
    /// SETEN  [<Xd>]!, <Xn>!, <Xs>
    /// SETMN  [<Xd>]!, <Xn>!, <Xs>
    /// SETPN  [<Xd>]!, <Xn>!, <Xs>
    Setpn {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SETPT, SETMT, SETET -- A64
    /// Memory Set, unprivileged
    /// SETET  [<Xd>]!, <Xn>!, <Xs>
    /// SETMT  [<Xd>]!, <Xn>!, <Xs>
    /// SETPT  [<Xd>]!, <Xn>!, <Xs>
    Setpt {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SETPTN, SETMTN, SETETN -- A64
    /// Memory Set, unprivileged and non-temporal
    /// SETETN  [<Xd>]!, <Xn>!, <Xs>
    /// SETMTN  [<Xd>]!, <Xn>!, <Xs>
    /// SETPTN  [<Xd>]!, <Xn>!, <Xs>
    Setptn {
        sz: i32,
        rs: Register,
        op2: i32,
        rn: Register,
        rd: Register,
    },
    /// SEV -- A64
    /// Send Event
    /// SEV
    Sev,
    /// SEVL -- A64
    /// Send Event Local
    /// SEVL
    Sevl,
    /// SMADDL -- A64
    /// Signed Multiply-Add Long
    /// SMADDL  <Xd>, <Wn>, <Wm>, <Xa>
    Smaddl {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// SMC -- A64
    /// Secure Monitor Call
    /// SMC  #<imm>
    Smc { imm16: i32 },
    /// SMNEGL -- A64
    /// Signed Multiply-Negate Long
    /// SMNEGL  <Xd>, <Wn>, <Wm>
    /// SMSUBL <Xd>, <Wn>, <Wm>, XZR
    SmneglSmsubl {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMSUBL -- A64
    /// Signed Multiply-Subtract Long
    /// SMSUBL  <Xd>, <Wn>, <Wm>, <Xa>
    Smsubl {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// SMULH -- A64
    /// Signed Multiply High
    /// SMULH  <Xd>, <Xn>, <Xm>
    Smulh {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMULL -- A64
    /// Signed Multiply Long
    /// SMULL  <Xd>, <Wn>, <Wm>
    /// SMADDL <Xd>, <Wn>, <Wm>, XZR
    SmullSmaddl {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SSBB -- A64
    /// Speculative Store Bypass Barrier
    /// SSBB
    /// DSB #0
    SsbbDsb,
    /// ST2G -- A64
    /// Store Allocation Tags
    /// ST2G  <Xt|SP>, [<Xn|SP>], #<simm>
    /// ST2G  <Xt|SP>, [<Xn|SP>, #<simm>]!
    /// ST2G  <Xt|SP>, [<Xn|SP>{, #<simm>}]
    St2g {
        imm9: i32,
        xn: i32,
        xt: i32,
        class_selector: St2gSelector,
    },
    /// ST64B -- A64
    /// Single-copy Atomic 64-byte Store without Return
    /// ST64B  <Xt>, [<Xn|SP> {,#0}]
    St64b { rn: Register, rt: Register },
    /// ST64BV -- A64
    /// Single-copy Atomic 64-byte Store with Return
    /// ST64BV  <Xs>, <Xt>, [<Xn|SP>]
    St64bv {
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// ST64BV0 -- A64
    /// Single-copy Atomic 64-byte EL0 Store with Return
    /// ST64BV0  <Xs>, <Xt>, [<Xn|SP>]
    St64bv0 {
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// STADD, STADDL -- A64
    /// Atomic add on word or doubleword in memory, without return
    /// STADD  <Ws>, [<Xn|SP>]
    /// LDADD <Ws>, WZR, [<Xn|SP>]
    /// STADDL  <Ws>, [<Xn|SP>]
    /// LDADDL <Ws>, WZR, [<Xn|SP>]
    /// STADD  <Xs>, [<Xn|SP>]
    /// LDADD <Xs>, XZR, [<Xn|SP>]
    /// STADDL  <Xs>, [<Xn|SP>]
    /// LDADDL <Xs>, XZR, [<Xn|SP>]
    StaddLdadd {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STADDB, STADDLB -- A64
    /// Atomic add on byte in memory, without return
    /// STADDB  <Ws>, [<Xn|SP>]
    /// LDADDB <Ws>, WZR, [<Xn|SP>]
    /// STADDLB  <Ws>, [<Xn|SP>]
    /// LDADDLB <Ws>, WZR, [<Xn|SP>]
    StaddbLdaddb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STADDH, STADDLH -- A64
    /// Atomic add on halfword in memory, without return
    /// STADDH  <Ws>, [<Xn|SP>]
    /// LDADDH <Ws>, WZR, [<Xn|SP>]
    /// STADDLH  <Ws>, [<Xn|SP>]
    /// LDADDLH <Ws>, WZR, [<Xn|SP>]
    StaddhLdaddh {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STCLR, STCLRL -- A64
    /// Atomic bit clear on word or doubleword in memory, without return
    /// STCLR  <Ws>, [<Xn|SP>]
    /// LDCLR <Ws>, WZR, [<Xn|SP>]
    /// STCLRL  <Ws>, [<Xn|SP>]
    /// LDCLRL <Ws>, WZR, [<Xn|SP>]
    /// STCLR  <Xs>, [<Xn|SP>]
    /// LDCLR <Xs>, XZR, [<Xn|SP>]
    /// STCLRL  <Xs>, [<Xn|SP>]
    /// LDCLRL <Xs>, XZR, [<Xn|SP>]
    StclrLdclr {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STCLRB, STCLRLB -- A64
    /// Atomic bit clear on byte in memory, without return
    /// STCLRB  <Ws>, [<Xn|SP>]
    /// LDCLRB <Ws>, WZR, [<Xn|SP>]
    /// STCLRLB  <Ws>, [<Xn|SP>]
    /// LDCLRLB <Ws>, WZR, [<Xn|SP>]
    StclrbLdclrb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STCLRH, STCLRLH -- A64
    /// Atomic bit clear on halfword in memory, without return
    /// STCLRH  <Ws>, [<Xn|SP>]
    /// LDCLRH <Ws>, WZR, [<Xn|SP>]
    /// STCLRLH  <Ws>, [<Xn|SP>]
    /// LDCLRLH <Ws>, WZR, [<Xn|SP>]
    StclrhLdclrh {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STEOR, STEORL -- A64
    /// Atomic exclusive OR on word or doubleword in memory, without return
    /// STEOR  <Ws>, [<Xn|SP>]
    /// LDEOR <Ws>, WZR, [<Xn|SP>]
    /// STEORL  <Ws>, [<Xn|SP>]
    /// LDEORL <Ws>, WZR, [<Xn|SP>]
    /// STEOR  <Xs>, [<Xn|SP>]
    /// LDEOR <Xs>, XZR, [<Xn|SP>]
    /// STEORL  <Xs>, [<Xn|SP>]
    /// LDEORL <Xs>, XZR, [<Xn|SP>]
    SteorLdeor {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STEORB, STEORLB -- A64
    /// Atomic exclusive OR on byte in memory, without return
    /// STEORB  <Ws>, [<Xn|SP>]
    /// LDEORB <Ws>, WZR, [<Xn|SP>]
    /// STEORLB  <Ws>, [<Xn|SP>]
    /// LDEORLB <Ws>, WZR, [<Xn|SP>]
    SteorbLdeorb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STEORH, STEORLH -- A64
    /// Atomic exclusive OR on halfword in memory, without return
    /// STEORH  <Ws>, [<Xn|SP>]
    /// LDEORH <Ws>, WZR, [<Xn|SP>]
    /// STEORLH  <Ws>, [<Xn|SP>]
    /// LDEORLH <Ws>, WZR, [<Xn|SP>]
    SteorhLdeorh {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STG -- A64
    /// Store Allocation Tag
    /// STG  <Xt|SP>, [<Xn|SP>], #<simm>
    /// STG  <Xt|SP>, [<Xn|SP>, #<simm>]!
    /// STG  <Xt|SP>, [<Xn|SP>{, #<simm>}]
    Stg {
        imm9: i32,
        xn: i32,
        xt: i32,
        class_selector: StgSelector,
    },
    /// STGM -- A64
    /// Store Tag Multiple
    /// STGM  <Xt>, [<Xn|SP>]
    Stgm { xn: i32, xt: i32 },
    /// STGP -- A64
    /// Store Allocation Tag and Pair of registers
    /// STGP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
    /// STGP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
    /// STGP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    Stgp {
        simm7: i32,
        xt2: i32,
        xn: i32,
        xt: i32,
        class_selector: StgpSelector,
    },
    /// STLLR -- A64
    /// Store LORelease Register
    /// STLLR  <Wt>, [<Xn|SP>{,#0}]
    /// STLLR  <Xt>, [<Xn|SP>{,#0}]
    Stllr {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// STLLRB -- A64
    /// Store LORelease Register Byte
    /// STLLRB  <Wt>, [<Xn|SP>{,#0}]
    Stllrb { rn: Register, rt: Register },
    /// STLLRH -- A64
    /// Store LORelease Register Halfword
    /// STLLRH  <Wt>, [<Xn|SP>{,#0}]
    Stllrh { rn: Register, rt: Register },
    /// STLR -- A64
    /// Store-Release Register
    /// STLR  <Wt>, [<Xn|SP>{,#0}]
    /// STLR  <Xt>, [<Xn|SP>{,#0}]
    Stlr {
        size: i32,
        rn: Register,
        rt: Register,
    },
    /// STLRB -- A64
    /// Store-Release Register Byte
    /// STLRB  <Wt>, [<Xn|SP>{,#0}]
    Stlrb { rn: Register, rt: Register },
    /// STLRH -- A64
    /// Store-Release Register Halfword
    /// STLRH  <Wt>, [<Xn|SP>{,#0}]
    Stlrh { rn: Register, rt: Register },
    /// STLUR -- A64
    /// Store-Release Register (unscaled)
    /// STLUR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// STLUR  <Xt>, [<Xn|SP>{, #<simm>}]
    StlurGen {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STLURB -- A64
    /// Store-Release Register Byte (unscaled)
    /// STLURB  <Wt>, [<Xn|SP>{, #<simm>}]
    Stlurb {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STLURH -- A64
    /// Store-Release Register Halfword (unscaled)
    /// STLURH  <Wt>, [<Xn|SP>{, #<simm>}]
    Stlurh {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STLXP -- A64
    /// Store-Release Exclusive Pair of registers
    /// STLXP  <Ws>, <Wt1>, <Wt2>, [<Xn|SP>{,#0}]
    /// STLXP  <Ws>, <Xt1>, <Xt2>, [<Xn|SP>{,#0}]
    Stlxp {
        sz: i32,
        rs: Register,
        rt2: Register,
        rn: Register,
        rt: Register,
    },
    /// STLXR -- A64
    /// Store-Release Exclusive Register
    /// STLXR  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// STLXR  <Ws>, <Xt>, [<Xn|SP>{,#0}]
    Stlxr {
        size: i32,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// STLXRB -- A64
    /// Store-Release Exclusive Register Byte
    /// STLXRB  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    Stlxrb {
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// STLXRH -- A64
    /// Store-Release Exclusive Register Halfword
    /// STLXRH  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    Stlxrh {
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// STNP -- A64
    /// Store Pair of Registers, with non-temporal hint
    /// STNP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
    /// STNP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    StnpGen {
        opc: i32,
        imm7: i32,
        rt2: Register,
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
    /// STRB (immediate) -- A64
    /// Store Register Byte (immediate)
    /// STRB  <Wt>, [<Xn|SP>], #<simm>
    /// STRB  <Wt>, [<Xn|SP>, #<simm>]!
    /// STRB  <Wt>, [<Xn|SP>{, #<pimm>}]
    StrbImm {
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: StrbImmSelector,
    },
    /// STRB (register) -- A64
    /// Store Register Byte (register)
    /// STRB  <Wt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
    /// STRB  <Wt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
    StrbReg {
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// STRH (immediate) -- A64
    /// Store Register Halfword (immediate)
    /// STRH  <Wt>, [<Xn|SP>], #<simm>
    /// STRH  <Wt>, [<Xn|SP>, #<simm>]!
    /// STRH  <Wt>, [<Xn|SP>{, #<pimm>}]
    StrhImm {
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: StrhImmSelector,
    },
    /// STRH (register) -- A64
    /// Store Register Halfword (register)
    /// STRH  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    StrhReg {
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
    },
    /// STSET, STSETL -- A64
    /// Atomic bit set on word or doubleword in memory, without return
    /// STSET  <Ws>, [<Xn|SP>]
    /// LDSET <Ws>, WZR, [<Xn|SP>]
    /// STSETL  <Ws>, [<Xn|SP>]
    /// LDSETL <Ws>, WZR, [<Xn|SP>]
    /// STSET  <Xs>, [<Xn|SP>]
    /// LDSET <Xs>, XZR, [<Xn|SP>]
    /// STSETL  <Xs>, [<Xn|SP>]
    /// LDSETL <Xs>, XZR, [<Xn|SP>]
    StsetLdset {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSETB, STSETLB -- A64
    /// Atomic bit set on byte in memory, without return
    /// STSETB  <Ws>, [<Xn|SP>]
    /// LDSETB <Ws>, WZR, [<Xn|SP>]
    /// STSETLB  <Ws>, [<Xn|SP>]
    /// LDSETLB <Ws>, WZR, [<Xn|SP>]
    StsetbLdsetb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSETH, STSETLH -- A64
    /// Atomic bit set on halfword in memory, without return
    /// STSETH  <Ws>, [<Xn|SP>]
    /// LDSETH <Ws>, WZR, [<Xn|SP>]
    /// STSETLH  <Ws>, [<Xn|SP>]
    /// LDSETLH <Ws>, WZR, [<Xn|SP>]
    StsethLdseth {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSMAX, STSMAXL -- A64
    /// Atomic signed maximum on word or doubleword in memory, without return
    /// STSMAX  <Ws>, [<Xn|SP>]
    /// LDSMAX <Ws>, WZR, [<Xn|SP>]
    /// STSMAXL  <Ws>, [<Xn|SP>]
    /// LDSMAXL <Ws>, WZR, [<Xn|SP>]
    /// STSMAX  <Xs>, [<Xn|SP>]
    /// LDSMAX <Xs>, XZR, [<Xn|SP>]
    /// STSMAXL  <Xs>, [<Xn|SP>]
    /// LDSMAXL <Xs>, XZR, [<Xn|SP>]
    StsmaxLdsmax {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSMAXB, STSMAXLB -- A64
    /// Atomic signed maximum on byte in memory, without return
    /// STSMAXB  <Ws>, [<Xn|SP>]
    /// LDSMAXB <Ws>, WZR, [<Xn|SP>]
    /// STSMAXLB  <Ws>, [<Xn|SP>]
    /// LDSMAXLB <Ws>, WZR, [<Xn|SP>]
    StsmaxbLdsmaxb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSMAXH, STSMAXLH -- A64
    /// Atomic signed maximum on halfword in memory, without return
    /// STSMAXH  <Ws>, [<Xn|SP>]
    /// LDSMAXH <Ws>, WZR, [<Xn|SP>]
    /// STSMAXLH  <Ws>, [<Xn|SP>]
    /// LDSMAXLH <Ws>, WZR, [<Xn|SP>]
    StsmaxhLdsmaxh {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSMIN, STSMINL -- A64
    /// Atomic signed minimum on word or doubleword in memory, without return
    /// STSMIN  <Ws>, [<Xn|SP>]
    /// LDSMIN <Ws>, WZR, [<Xn|SP>]
    /// STSMINL  <Ws>, [<Xn|SP>]
    /// LDSMINL <Ws>, WZR, [<Xn|SP>]
    /// STSMIN  <Xs>, [<Xn|SP>]
    /// LDSMIN <Xs>, XZR, [<Xn|SP>]
    /// STSMINL  <Xs>, [<Xn|SP>]
    /// LDSMINL <Xs>, XZR, [<Xn|SP>]
    StsminLdsmin {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSMINB, STSMINLB -- A64
    /// Atomic signed minimum on byte in memory, without return
    /// STSMINB  <Ws>, [<Xn|SP>]
    /// LDSMINB <Ws>, WZR, [<Xn|SP>]
    /// STSMINLB  <Ws>, [<Xn|SP>]
    /// LDSMINLB <Ws>, WZR, [<Xn|SP>]
    StsminbLdsminb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STSMINH, STSMINLH -- A64
    /// Atomic signed minimum on halfword in memory, without return
    /// STSMINH  <Ws>, [<Xn|SP>]
    /// LDSMINH <Ws>, WZR, [<Xn|SP>]
    /// STSMINLH  <Ws>, [<Xn|SP>]
    /// LDSMINLH <Ws>, WZR, [<Xn|SP>]
    StsminhLdsminh {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STTR -- A64
    /// Store Register (unprivileged)
    /// STTR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// STTR  <Xt>, [<Xn|SP>{, #<simm>}]
    Sttr {
        size: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STTRB -- A64
    /// Store Register Byte (unprivileged)
    /// STTRB  <Wt>, [<Xn|SP>{, #<simm>}]
    Sttrb {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STTRH -- A64
    /// Store Register Halfword (unprivileged)
    /// STTRH  <Wt>, [<Xn|SP>{, #<simm>}]
    Sttrh {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STUMAX, STUMAXL -- A64
    /// Atomic unsigned maximum on word or doubleword in memory, without return
    /// STUMAX  <Ws>, [<Xn|SP>]
    /// LDUMAX <Ws>, WZR, [<Xn|SP>]
    /// STUMAXL  <Ws>, [<Xn|SP>]
    /// LDUMAXL <Ws>, WZR, [<Xn|SP>]
    /// STUMAX  <Xs>, [<Xn|SP>]
    /// LDUMAX <Xs>, XZR, [<Xn|SP>]
    /// STUMAXL  <Xs>, [<Xn|SP>]
    /// LDUMAXL <Xs>, XZR, [<Xn|SP>]
    StumaxLdumax {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STUMAXB, STUMAXLB -- A64
    /// Atomic unsigned maximum on byte in memory, without return
    /// STUMAXB  <Ws>, [<Xn|SP>]
    /// LDUMAXB <Ws>, WZR, [<Xn|SP>]
    /// STUMAXLB  <Ws>, [<Xn|SP>]
    /// LDUMAXLB <Ws>, WZR, [<Xn|SP>]
    StumaxbLdumaxb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STUMAXH, STUMAXLH -- A64
    /// Atomic unsigned maximum on halfword in memory, without return
    /// STUMAXH  <Ws>, [<Xn|SP>]
    /// LDUMAXH <Ws>, WZR, [<Xn|SP>]
    /// STUMAXLH  <Ws>, [<Xn|SP>]
    /// LDUMAXLH <Ws>, WZR, [<Xn|SP>]
    StumaxhLdumaxh {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STUMIN, STUMINL -- A64
    /// Atomic unsigned minimum on word or doubleword in memory, without return
    /// STUMIN  <Ws>, [<Xn|SP>]
    /// LDUMIN <Ws>, WZR, [<Xn|SP>]
    /// STUMINL  <Ws>, [<Xn|SP>]
    /// LDUMINL <Ws>, WZR, [<Xn|SP>]
    /// STUMIN  <Xs>, [<Xn|SP>]
    /// LDUMIN <Xs>, XZR, [<Xn|SP>]
    /// STUMINL  <Xs>, [<Xn|SP>]
    /// LDUMINL <Xs>, XZR, [<Xn|SP>]
    StuminLdumin {
        size: i32,
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STUMINB, STUMINLB -- A64
    /// Atomic unsigned minimum on byte in memory, without return
    /// STUMINB  <Ws>, [<Xn|SP>]
    /// LDUMINB <Ws>, WZR, [<Xn|SP>]
    /// STUMINLB  <Ws>, [<Xn|SP>]
    /// LDUMINLB <Ws>, WZR, [<Xn|SP>]
    StuminbLduminb {
        r: Register,
        rs: Register,
        rn: Register,
    },
    /// STUMINH, STUMINLH -- A64
    /// Atomic unsigned minimum on halfword in memory, without return
    /// STUMINH  <Ws>, [<Xn|SP>]
    /// LDUMINH <Ws>, WZR, [<Xn|SP>]
    /// STUMINLH  <Ws>, [<Xn|SP>]
    /// LDUMINLH <Ws>, WZR, [<Xn|SP>]
    StuminhLduminh {
        r: Register,
        rs: Register,
        rn: Register,
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
    /// STURB -- A64
    /// Store Register Byte (unscaled)
    /// STURB  <Wt>, [<Xn|SP>{, #<simm>}]
    Sturb {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STURH -- A64
    /// Store Register Halfword (unscaled)
    /// STURH  <Wt>, [<Xn|SP>{, #<simm>}]
    Sturh {
        imm9: i32,
        rn: Register,
        rt: Register,
    },
    /// STXP -- A64
    /// Store Exclusive Pair of registers
    /// STXP  <Ws>, <Wt1>, <Wt2>, [<Xn|SP>{,#0}]
    /// STXP  <Ws>, <Xt1>, <Xt2>, [<Xn|SP>{,#0}]
    Stxp {
        sz: i32,
        rs: Register,
        rt2: Register,
        rn: Register,
        rt: Register,
    },
    /// STXR -- A64
    /// Store Exclusive Register
    /// STXR  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    /// STXR  <Ws>, <Xt>, [<Xn|SP>{,#0}]
    Stxr {
        size: i32,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// STXRB -- A64
    /// Store Exclusive Register Byte
    /// STXRB  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    Stxrb {
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// STXRH -- A64
    /// Store Exclusive Register Halfword
    /// STXRH  <Ws>, <Wt>, [<Xn|SP>{,#0}]
    Stxrh {
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// STZ2G -- A64
    /// Store Allocation Tags, Zeroing
    /// STZ2G  <Xt|SP>, [<Xn|SP>], #<simm>
    /// STZ2G  <Xt|SP>, [<Xn|SP>, #<simm>]!
    /// STZ2G  <Xt|SP>, [<Xn|SP>{, #<simm>}]
    Stz2g {
        imm9: i32,
        xn: i32,
        xt: i32,
        class_selector: Stz2gSelector,
    },
    /// STZG -- A64
    /// Store Allocation Tag, Zeroing
    /// STZG  <Xt|SP>, [<Xn|SP>], #<simm>
    /// STZG  <Xt|SP>, [<Xn|SP>, #<simm>]!
    /// STZG  <Xt|SP>, [<Xn|SP>{, #<simm>}]
    Stzg {
        imm9: i32,
        xn: i32,
        xt: i32,
        class_selector: StzgSelector,
    },
    /// STZGM -- A64
    /// Store Tag and Zero Multiple
    /// STZGM  <Xt>, [<Xn|SP>]
    Stzgm { xn: i32, xt: i32 },
    /// SUB (extended register) -- A64
    /// Subtract (extended register)
    /// SUB  <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// SUB  <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    SubAddsubExt {
        sf: i32,
        rm: Register,
        option: i32,
        imm3: i32,
        rn: Register,
        rd: Register,
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
    /// SUBG -- A64
    /// Subtract with Tag
    /// SUBG  <Xd|SP>, <Xn|SP>, #<uimm6>, #<uimm4>
    Subg {
        uimm6: i32,
        uimm4: i32,
        xn: i32,
        xd: i32,
    },
    /// SUBP -- A64
    /// Subtract Pointer
    /// SUBP  <Xd>, <Xn|SP>, <Xm|SP>
    Subp { xm: i32, xn: i32, xd: i32 },
    /// SUBPS -- A64
    /// Subtract Pointer, setting Flags
    /// SUBPS  <Xd>, <Xn|SP>, <Xm|SP>
    Subps { xm: i32, xn: i32, xd: i32 },
    /// SUBS (extended register) -- A64
    /// Subtract (extended register), setting flags
    /// SUBS  <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
    /// SUBS  <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
    SubsAddsubExt {
        sf: i32,
        rm: Register,
        option: i32,
        imm3: i32,
        rn: Register,
        rd: Register,
    },
    /// SUBS (immediate) -- A64
    /// Subtract (immediate), setting flags
    /// SUBS  <Wd>, <Wn|WSP>, #<imm>{, <shift>}
    /// SUBS  <Xd>, <Xn|SP>, #<imm>{, <shift>}
    SubsAddsubImm {
        sf: i32,
        sh: i32,
        imm12: i32,
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
    /// SVC -- A64
    /// Supervisor Call
    /// SVC  #<imm>
    Svc { imm16: i32 },
    /// SWP, SWPA, SWPAL, SWPL -- A64
    /// Swap word or doubleword in memory
    /// SWP  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPA  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPAL  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPL  <Ws>, <Wt>, [<Xn|SP>]
    /// SWP  <Xs>, <Xt>, [<Xn|SP>]
    /// SWPA  <Xs>, <Xt>, [<Xn|SP>]
    /// SWPAL  <Xs>, <Xt>, [<Xn|SP>]
    /// SWPL  <Xs>, <Xt>, [<Xn|SP>]
    Swp {
        size: i32,
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// SWPB, SWPAB, SWPALB, SWPLB -- A64
    /// Swap byte in memory
    /// SWPAB  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPALB  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPB  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPLB  <Ws>, <Wt>, [<Xn|SP>]
    Swpb {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// SWPH, SWPAH, SWPALH, SWPLH -- A64
    /// Swap halfword in memory
    /// SWPAH  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPALH  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPH  <Ws>, <Wt>, [<Xn|SP>]
    /// SWPLH  <Ws>, <Wt>, [<Xn|SP>]
    Swph {
        a: i32,
        r: Register,
        rs: Register,
        rn: Register,
        rt: Register,
    },
    /// SXTB -- A64
    /// Signed Extend Byte
    /// SXTB  <Wd>, <Wn>
    /// SBFM <Wd>, <Wn>, #0, #7
    /// SXTB  <Xd>, <Wn>
    /// SBFM <Xd>, <Xn>, #0, #7
    SxtbSbfm {
        sf: i32,
        n: i32,
        rn: Register,
        rd: Register,
    },
    /// SXTH -- A64
    /// Sign Extend Halfword
    /// SXTH  <Wd>, <Wn>
    /// SBFM <Wd>, <Wn>, #0, #15
    /// SXTH  <Xd>, <Wn>
    /// SBFM <Xd>, <Xn>, #0, #15
    SxthSbfm {
        sf: i32,
        n: i32,
        rn: Register,
        rd: Register,
    },
    /// SXTW -- A64
    /// Sign Extend Word
    /// SXTW  <Xd>, <Wn>
    /// SBFM <Xd>, <Xn>, #0, #31
    SxtwSbfm { rn: Register, rd: Register },
    /// SYS -- A64
    /// System instruction
    /// SYS  #<op1>, <Cn>, <Cm>, #<op2>{, <Xt>}
    Sys {
        op1: i32,
        crn: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// SYSL -- A64
    /// System instruction with result
    /// SYSL  <Xt>, #<op1>, <Cn>, <Cm>, #<op2>
    Sysl {
        op1: i32,
        crn: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// TBNZ -- A64
    /// Test bit and Branch if Nonzero
    /// TBNZ  <R><t>, #<imm>, <label>
    Tbnz {
        b5: i32,
        b40: i32,
        imm14: i32,
        rt: Register,
    },
    /// TBZ -- A64
    /// Test bit and Branch if Zero
    /// TBZ  <R><t>, #<imm>, <label>
    Tbz {
        b5: i32,
        b40: i32,
        imm14: i32,
        rt: Register,
    },
    /// TLBI -- A64
    /// TLB Invalidate operation
    /// TLBI  <tlbi_op>{, <Xt>}
    /// SYS #<op1>, C8, <Cm>, #<op2>{, <Xt>}
    TlbiSys {
        op1: i32,
        crm: i32,
        op2: i32,
        rt: Register,
    },
    /// TSB CSYNC -- A64
    /// Trace Synchronization Barrier
    /// TSB CSYNC
    Tsb,
    /// TST (immediate) -- A64

    /// TST  <Wn>, #<imm>
    /// ANDS WZR, <Wn>, #<imm>
    /// TST  <Xn>, #<imm>
    /// ANDS XZR, <Xn>, #<imm>
    TstAndsLogImm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
    },
    /// TST (shifted register) -- A64
    /// Test (shifted register)
    /// TST  <Wn>, <Wm>{, <shift> #<amount>}
    /// ANDS WZR, <Wn>, <Wm>{, <shift> #<amount>}
    /// TST  <Xn>, <Xm>{, <shift> #<amount>}
    /// ANDS XZR, <Xn>, <Xm>{, <shift> #<amount>}
    TstAndsLogShift {
        sf: i32,
        shift: i32,
        rm: Register,
        imm6: i32,
        rn: Register,
    },
    /// UBFIZ -- A64
    /// Unsigned Bitfield Insert in Zero
    /// UBFIZ  <Wd>, <Wn>, #<lsb>, #<width>
    /// UBFM <Wd>, <Wn>, #(-<lsb> MOD 32), #(<width>-1)
    /// UBFIZ  <Xd>, <Xn>, #<lsb>, #<width>
    /// UBFM <Xd>, <Xn>, #(-<lsb> MOD 64), #(<width>-1)
    UbfizUbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// UBFM -- A64
    /// Unsigned Bitfield Move
    /// UBFM  <Wd>, <Wn>, #<immr>, #<imms>
    /// UBFM  <Xd>, <Xn>, #<immr>, #<imms>
    Ubfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// UBFX -- A64
    /// Unsigned Bitfield Extract
    /// UBFX  <Wd>, <Wn>, #<lsb>, #<width>
    /// UBFM <Wd>, <Wn>, #<lsb>, #(<lsb>+<width>-1)
    /// UBFX  <Xd>, <Xn>, #<lsb>, #<width>
    /// UBFM <Xd>, <Xn>, #<lsb>, #(<lsb>+<width>-1)
    UbfxUbfm {
        sf: i32,
        n: i32,
        immr: i32,
        imms: i32,
        rn: Register,
        rd: Register,
    },
    /// UDF -- A64
    /// Permanently Undefined
    /// UDF  #<imm>
    UdfPermUndef { imm16: i32 },
    /// UDIV -- A64
    /// Unsigned Divide
    /// UDIV  <Wd>, <Wn>, <Wm>
    /// UDIV  <Xd>, <Xn>, <Xm>
    Udiv {
        sf: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMADDL -- A64
    /// Unsigned Multiply-Add Long
    /// UMADDL  <Xd>, <Wn>, <Wm>, <Xa>
    Umaddl {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// UMNEGL -- A64
    /// Unsigned Multiply-Negate Long
    /// UMNEGL  <Xd>, <Wn>, <Wm>
    /// UMSUBL <Xd>, <Wn>, <Wm>, XZR
    UmneglUmsubl {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMSUBL -- A64
    /// Unsigned Multiply-Subtract Long
    /// UMSUBL  <Xd>, <Wn>, <Wm>, <Xa>
    Umsubl {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// UMULH -- A64
    /// Unsigned Multiply High
    /// UMULH  <Xd>, <Xn>, <Xm>
    Umulh {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMULL -- A64
    /// Unsigned Multiply Long
    /// UMULL  <Xd>, <Wn>, <Wm>
    /// UMADDL <Xd>, <Wn>, <Wm>, XZR
    UmullUmaddl {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UXTB -- A64
    /// Unsigned Extend Byte
    /// UXTB  <Wd>, <Wn>
    /// UBFM <Wd>, <Wn>, #0, #7
    UxtbUbfm { rn: Register, rd: Register },
    /// UXTH -- A64
    /// Unsigned Extend Halfword
    /// UXTH  <Wd>, <Wn>
    /// UBFM <Wd>, <Wn>, #0, #15
    UxthUbfm { rn: Register, rd: Register },
    /// WFE -- A64
    /// Wait For Event
    /// WFE
    Wfe,
    /// WFET -- A64
    /// Wait For Event with Timeout
    /// WFET  <Xt>
    Wfet { rd: Register },
    /// WFI -- A64
    /// Wait For Interrupt
    /// WFI
    Wfi,
    /// WFIT -- A64
    /// Wait For Interrupt with Timeout
    /// WFIT  <Xt>
    Wfit { rd: Register },
    /// XAFLAG -- A64
    /// Convert floating-point condition flags from external format to Arm format
    /// XAFLAG
    Xaflag,
    /// XPACD, XPACI, XPACLRI -- A64
    /// Strip Pointer Authentication Code
    /// XPACD  <Xd>
    /// XPACI  <Xd>
    /// XPACLRI
    Xpac {
        d: i32,
        rd: Register,
        class_selector: XpacSelector,
    },
    /// YIELD -- A64
    /// YIELD
    /// YIELD
    Yield,
}
#[derive(Debug)]
pub enum AutiaSelector {
    Integer,
    System,
}

#[derive(Debug)]
pub enum AutibSelector {
    Integer,
    System,
}

#[derive(Debug)]
pub enum DsbSelector {
    MemoryBarrier,
    MemoryNxsBarrier,
}

#[derive(Debug)]
pub enum LdpGenSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum LdpswSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum LdrImmGenSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum LdrbImmSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum LdrhImmSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum LdrsbImmSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum LdrshImmSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum LdrswImmSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum PaciaSelector {
    Integer,
    System,
}

#[derive(Debug)]
pub enum PacibSelector {
    Integer,
    System,
}

#[derive(Debug)]
pub enum St2gSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum StgSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum StgpSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum StpGenSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum StrImmGenSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum StrbImmSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum StrhImmSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
}

#[derive(Debug)]
pub enum Stz2gSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum StzgSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum XpacSelector {
    Integer,
    System,
}
impl ArmAsm {
    pub fn encode(&self) -> u32 {
        match self {
            ArmAsm::Adc { sf, rm, rn, rd } => {
                0b0_0_0_11010000_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Adcs { sf, rm, rn, rd } => {
                0b0_0_1_11010000_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AddAddsubExt {
                sf,
                rm,
                option,
                imm3,
                rn,
                rd,
            } => {
                0b0_0_0_01011_00_1_00000_000_000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*option as u32) << 13
                    | truncate_imm::<_, 3>(*imm3) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AddAddsubImm {
                sf,
                sh,
                imm12,
                rn,
                rd,
            } => {
                0b0_0_0_100010_0_000000000000_00000_00000
                    | (*sf as u32) << 31
                    | (*sh as u32) << 22
                    | truncate_imm::<_, 12>(*imm12) << 10
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Addg {
                uimm6,
                uimm4,
                xn,
                xd,
            } => {
                0b1_0_0_100011_0_000000_00_0000_00000_00000
                    | (*uimm6 as u32) << 16
                    | (*uimm4 as u32) << 10
                    | (*xn as u32) << 5
                    | (*xd as u32) << 0
            }
            ArmAsm::AddsAddsubExt {
                sf,
                rm,
                option,
                imm3,
                rn,
                rd,
            } => {
                0b0_0_1_01011_00_1_00000_000_000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*option as u32) << 13
                    | truncate_imm::<_, 3>(*imm3) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AddsAddsubImm {
                sf,
                sh,
                imm12,
                rn,
                rd,
            } => {
                0b0_0_1_100010_0_000000000000_00000_00000
                    | (*sf as u32) << 31
                    | (*sh as u32) << 22
                    | truncate_imm::<_, 12>(*imm12) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AddsAddsubShift {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_0_1_01011_00_0_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Adr { immlo, immhi, rd } => {
                0b0_00_10000_0000000000000000000_00000
                    | (*immlo as u32) << 29
                    | (*immhi as u32) << 5
                    | rd << 0
            }
            ArmAsm::Adrp { immlo, immhi, rd } => {
                0b1_00_10000_0000000000000000000_00000
                    | (*immlo as u32) << 29
                    | (*immhi as u32) << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AndsLogImm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_11_100100_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AndsLogShift {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_11_01010_00_0_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AsrAsrv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_10_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Asrv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_10_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AtSys { op1, crm, op2, rt } => {
                0b1101010100_0_01_000_0111_0000_000_00000
                    | (*op1 as u32) << 16
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::Autda { z, rn, rd } => {
                0b1_1_0_11010110_00001_0_0_0_110_00000_00000 | (*z as u32) << 13 | rn << 5 | rd << 0
            }
            ArmAsm::Autdb { z, rn, rd } => {
                0b1_1_0_11010110_00001_0_0_0_111_00000_00000 | (*z as u32) << 13 | rn << 5 | rd << 0
            }
            ArmAsm::Autia {
                z,
                rn,
                rd,
                crm,
                op2,
                class_selector,
            } => match class_selector {
                AutiaSelector::Integer => {
                    0b1_1_0_11010110_00001_0_0_0_100_00000_00000
                        | (*z as u32) << 13
                        | rn << 5
                        | rd << 0
                }
                AutiaSelector::System => {
                    0b1101010100_0_00_011_0010_0000_000_11111
                        | (*crm as u32) << 8
                        | (*op2 as u32) << 5
                }
            },
            ArmAsm::Autib {
                z,
                rn,
                rd,
                crm,
                op2,
                class_selector,
            } => match class_selector {
                AutibSelector::Integer => {
                    0b1_1_0_11010110_00001_0_0_0_101_00000_00000
                        | (*z as u32) << 13
                        | rn << 5
                        | rd << 0
                }
                AutibSelector::System => {
                    0b1101010100_0_00_011_0010_0000_000_11111
                        | (*crm as u32) << 8
                        | (*op2 as u32) << 5
                }
            },
            ArmAsm::Axflag {} => 0b1101010100_0_00_000_0100_0000_010_11111,
            ArmAsm::BCond { imm19, cond } => {
                0b0101010_0_0000000000000000000_0_0000
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | (*cond as u32) << 0
            }
            ArmAsm::BUncond { imm26 } => {
                0b0_00101_00000000000000000000000000 | truncate_imm::<_, 26>(*imm26) << 0
            }
            ArmAsm::BcCond { imm19, cond } => {
                0b0101010_0_0000000000000000000_1_0000
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | (*cond as u32) << 0
            }
            ArmAsm::BfcBfm {
                sf,
                n,
                immr,
                imms,
                rd,
            } => {
                0b0_01_100110_0_000000_000000_11111_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rd << 0
            }
            ArmAsm::BfiBfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_01_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Bfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_01_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::BfxilBfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_01_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::BicLogShift {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_00_01010_00_1_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Bics {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_11_01010_00_1_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Bl { imm26 } => {
                0b1_00101_00000000000000000000000000 | truncate_imm::<_, 26>(*imm26) << 0
            }
            ArmAsm::Blr { rn } => 0b1101011_0_0_01_11111_0000_0_0_00000_00000 | rn << 5,
            ArmAsm::Blra { z, m, rn, rm } => {
                0b1101011_0_0_01_11111_0000_1_0_00000_00000
                    | (*z as u32) << 24
                    | (*m as u32) << 10
                    | rn << 5
                    | rm << 0
            }
            ArmAsm::Br { rn } => 0b1101011_0_0_00_11111_0000_0_0_00000_00000 | rn << 5,
            ArmAsm::Bra { z, m, rn, rm } => {
                0b1101011_0_0_00_11111_0000_1_0_00000_00000
                    | (*z as u32) << 24
                    | (*m as u32) << 10
                    | rn << 5
                    | rm << 0
            }
            ArmAsm::Brk { imm16 } => 0b11010100_001_0000000000000000_000_00 | (*imm16 as u32) << 5,
            ArmAsm::Bti { op2 } => 0b1101010100_0_00_011_0010_0100_000_11111 | (*op2 as u32) << 5,
            ArmAsm::Cas {
                size,
                l,
                rs,
                o0,
                rn,
                rt,
            } => {
                0b00_0010001_0_1_00000_0_11111_00000_00000
                    | (*size as u32) << 30
                    | (*l as u32) << 22
                    | rs << 16
                    | (*o0 as u32) << 15
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Casb { l, rs, o0, rn, rt } => {
                0b00_0010001_0_1_00000_0_11111_00000_00000
                    | (*l as u32) << 22
                    | rs << 16
                    | (*o0 as u32) << 15
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Cash { l, rs, o0, rn, rt } => {
                0b01_0010001_0_1_00000_0_11111_00000_00000
                    | (*l as u32) << 22
                    | rs << 16
                    | (*o0 as u32) << 15
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Casp {
                sz,
                l,
                rs,
                o0,
                rn,
                rt,
            } => {
                0b0_0_001000_0_0_1_00000_0_11111_00000_00000
                    | (*sz as u32) << 30
                    | (*l as u32) << 22
                    | rs << 16
                    | (*o0 as u32) << 15
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Cbnz { sf, imm19, rt } => {
                0b0_011010_1_0000000000000000000_00000
                    | (*sf as u32) << 31
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | rt << 0
            }
            ArmAsm::Cbz { sf, imm19, rt } => {
                0b0_011010_0_0000000000000000000_00000
                    | (*sf as u32) << 31
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | rt << 0
            }
            ArmAsm::CcmnImm {
                sf,
                imm5,
                cond,
                rn,
                nzcv,
            } => {
                0b0_0_1_11010010_00000_0000_1_0_00000_0_0000
                    | (*sf as u32) << 31
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | (*nzcv as u32) << 0
            }
            ArmAsm::CcmnReg {
                sf,
                rm,
                cond,
                rn,
                nzcv,
            } => {
                0b0_0_1_11010010_00000_0000_0_0_00000_0_0000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | (*nzcv as u32) << 0
            }
            ArmAsm::CcmpImm {
                sf,
                imm5,
                cond,
                rn,
                nzcv,
            } => {
                0b0_1_1_11010010_00000_0000_1_0_00000_0_0000
                    | (*sf as u32) << 31
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | (*nzcv as u32) << 0
            }
            ArmAsm::CcmpReg {
                sf,
                rm,
                cond,
                rn,
                nzcv,
            } => {
                0b0_1_1_11010010_00000_0000_0_0_00000_0_0000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | (*nzcv as u32) << 0
            }
            ArmAsm::Cfinv {} => 0b1101010100_0_0_0_000_0100_0000_000_11111,
            ArmAsm::CfpSys { rt } => 0b1101010100_0_01_011_0111_0011_100_00000 | rt << 0,
            ArmAsm::CincCsinc {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_0_0_11010100_00000_0000_0_1_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::CinvCsinv {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_1_0_11010100_00000_0000_0_0_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Clrex { crm } => 0b1101010100_0_00_011_0011_0000_010_11111 | (*crm as u32) << 8,
            ArmAsm::ClsInt { sf, rn, rd } => {
                0b0_1_0_11010110_00000_00010_1_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::ClzInt { sf, rn, rd } => {
                0b0_1_0_11010110_00000_00010_0_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::CmnAddsAddsubExt {
                sf,
                rm,
                option,
                imm3,
                rn,
            } => {
                0b0_0_1_01011_00_1_00000_000_000_00000_11111
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*option as u32) << 13
                    | truncate_imm::<_, 3>(*imm3) << 10
                    | rn << 5
            }
            ArmAsm::CmnAddsAddsubImm { sf, sh, imm12, rn } => {
                0b0_0_1_100010_0_000000000000_00000_11111
                    | (*sf as u32) << 31
                    | (*sh as u32) << 22
                    | truncate_imm::<_, 12>(*imm12) << 10
                    | rn << 5
            }
            ArmAsm::CmnAddsAddsubShift {
                sf,
                shift,
                rm,
                imm6,
                rn,
            } => {
                0b0_0_1_01011_00_0_00000_000000_00000_11111
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
            }
            ArmAsm::CmpSubsAddsubExt {
                sf,
                rm,
                option,
                imm3,
                rn,
            } => {
                0b0_1_1_01011_00_1_00000_000_000_00000_11111
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*option as u32) << 13
                    | truncate_imm::<_, 3>(*imm3) << 10
                    | rn << 5
            }
            ArmAsm::CmpSubsAddsubImm { sf, sh, imm12, rn } => {
                0b0_1_1_100010_0_000000000000_00000_11111
                    | (*sf as u32) << 31
                    | (*sh as u32) << 22
                    | truncate_imm::<_, 12>(*imm12) << 10
                    | rn << 5
            }
            ArmAsm::CmpSubsAddsubShift {
                sf,
                shift,
                rm,
                imm6,
                rn,
            } => {
                0b0_1_1_01011_00_0_00000_000000_00000_11111
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
            }
            ArmAsm::CmppSubps { xm, xn } => {
                0b1_0_1_11010110_00000_0_0_0_0_0_0_00000_11111
                    | (*xm as u32) << 16
                    | (*xn as u32) << 5
            }
            ArmAsm::CnegCsneg {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_1_0_11010100_00000_0000_0_1_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::CppSys { rt } => 0b1101010100_0_01_011_0111_0011_111_00000 | rt << 0,
            ArmAsm::Cpyfp {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfpn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1100_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfprn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1000_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfprt {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0010_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfprtn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1110_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfprtrn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1010_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfprtwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0110_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfpt {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0011_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfptn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1111_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfptrn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1011_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfptwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0111_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfpwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0100_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfpwt {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0001_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfpwtn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1101_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfpwtrn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_1001_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyfpwtwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_0_01_00_0_00000_0101_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyp {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpypn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1100_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyprn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1000_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyprt {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0010_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyprtn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1110_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyprtrn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1010_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyprtwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0110_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpypt {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0011_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyptn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1111_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyptrn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1011_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpyptwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0111_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpypwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0100_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpypwt {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0001_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpypwtn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1101_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpypwtrn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_1001_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Cpypwtwn {
                sz,
                op1,
                rs,
                rn,
                rd,
            } => {
                0b00_011_1_01_00_0_00000_0101_01_00000_00000
                    | (*sz as u32) << 30
                    | (*op1 as u32) << 22
                    | rs << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Crc32 { sf, rm, sz, rn, rd } => {
                0b0_0_0_11010110_00000_010_0_00_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*sz as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Crc32c { sf, rm, sz, rn, rd } => {
                0b0_0_0_11010110_00000_010_1_00_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*sz as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Csdb {} => 0b1101010100_0_00_011_0010_0010_100_11111,
            ArmAsm::Csel {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_0_0_11010100_00000_0000_0_0_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::CsetCsinc { sf, cond, rd } => {
                0b0_0_0_11010100_11111_0000_0_1_11111_00000
                    | (*sf as u32) << 31
                    | (*cond as u32) << 12
                    | rd << 0
            }
            ArmAsm::CsetmCsinv { sf, cond, rd } => {
                0b0_1_0_11010100_11111_0000_0_0_11111_00000
                    | (*sf as u32) << 31
                    | (*cond as u32) << 12
                    | rd << 0
            }
            ArmAsm::Csinc {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_0_0_11010100_00000_0000_0_1_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Csinv {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_1_0_11010100_00000_0000_0_0_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Csneg {
                sf,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_1_0_11010100_00000_0000_0_1_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::DcSys { op1, crm, op2, rt } => {
                0b1101010100_0_01_000_0111_0000_000_00000
                    | (*op1 as u32) << 16
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::Dcps1 { imm16 } => {
                0b11010100_101_0000000000000000_000_01 | (*imm16 as u32) << 5
            }
            ArmAsm::Dcps2 { imm16 } => {
                0b11010100_101_0000000000000000_000_10 | (*imm16 as u32) << 5
            }
            ArmAsm::Dcps3 { imm16 } => {
                0b11010100_101_0000000000000000_000_11 | (*imm16 as u32) << 5
            }
            ArmAsm::Dgh {} => 0b1101010100_0_00_011_0010_0000_110_11111,
            ArmAsm::Dmb { crm } => 0b1101010100_0_00_011_0011_0000_1_01_11111 | (*crm as u32) << 8,
            ArmAsm::Drps {} => 0b1101011_0101_11111_000000_11111_00000,
            ArmAsm::Dsb {
                crm,
                imm2,
                class_selector,
            } => match class_selector {
                DsbSelector::MemoryBarrier => {
                    0b1101010100_0_00_011_0011_0000_1_00_11111 | (*crm as u32) << 8
                }
                DsbSelector::MemoryNxsBarrier => {
                    0b11010101000000110011_00_10_0_01_11111 | (*imm2 as u32) << 10
                }
            },
            ArmAsm::DvpSys { rt } => 0b1101010100_0_01_011_0111_0011_101_00000 | rt << 0,
            ArmAsm::Eon {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_10_01010_00_1_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::EorLogImm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_10_100100_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Eret {} => 0b1101011_0_100_11111_0000_0_0_11111_00000,
            ArmAsm::Ereta { m } => 0b1101011_0_100_11111_0000_1_0_11111_11111 | (*m as u32) << 10,
            ArmAsm::Esb {} => 0b1101010100_0_00_011_0010_0010_000_11111,
            ArmAsm::Extr {
                sf,
                n,
                rm,
                imms,
                rn,
                rd,
            } => {
                0b0_00_100111_0_0_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | rm << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Gmi { xm, xn, xd } => {
                0b1_0_0_11010110_00000_0_0_0_1_0_1_00000_00000
                    | (*xm as u32) << 16
                    | (*xn as u32) << 5
                    | (*xd as u32) << 0
            }
            ArmAsm::Hint { crm, op2 } => {
                0b1101010100_0_00_011_0010_0000_000_11111 | (*crm as u32) << 8 | (*op2 as u32) << 5
            }
            ArmAsm::Hlt { imm16 } => 0b11010100_010_0000000000000000_000_00 | (*imm16 as u32) << 5,
            ArmAsm::Hvc { imm16 } => 0b11010100_000_0000000000000000_000_10 | (*imm16 as u32) << 5,
            ArmAsm::IcSys { op1, crm, op2, rt } => {
                0b1101010100_0_01_000_0111_0000_000_00000
                    | (*op1 as u32) << 16
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::Irg { xm, xn, xd } => {
                0b1_0_0_11010110_00000_0_0_0_1_0_0_00000_00000
                    | (*xm as u32) << 16
                    | (*xn as u32) << 5
                    | (*xd as u32) << 0
            }
            ArmAsm::Isb { crm } => 0b1101010100_0_00_011_0011_0000_1_10_11111 | (*crm as u32) << 8,
            ArmAsm::Ld64b { rn, rt } => {
                0b11_111_0_00_0_0_1_11111_1_101_00_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldadd {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_000_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldaddb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_000_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldaddh { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_000_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldapr { size, rn, rt } => {
                0b00_111_0_00_1_0_1_11111_1_100_00_00000_00000
                    | (*size as u32) << 30
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldaprb { rn, rt } => {
                0b00_111_0_00_1_0_1_11111_1_100_00_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldaprh { rn, rt } => {
                0b01_111_0_00_1_0_1_11111_1_100_00_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::LdapurGen { size, imm9, rn, rt } => {
                0b00_011001_01_0_000000000_00_00000_00000
                    | (*size as u32) << 30
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldapurb { imm9, rn, rt } => {
                0b00_011001_01_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldapurh { imm9, rn, rt } => {
                0b01_011001_01_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldapursb { opc, imm9, rn, rt } => {
                0b00_011001_00_0_000000000_00_00000_00000
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldapursh { opc, imm9, rn, rt } => {
                0b01_011001_00_0_000000000_00_00000_00000
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldapursw { imm9, rn, rt } => {
                0b10_011001_10_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldar { size, rn, rt } => {
                0b00_001000_1_1_0_11111_1_11111_00000_00000
                    | (*size as u32) << 30
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldarb { rn, rt } => {
                0b00_001000_1_1_0_11111_1_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldarh { rn, rt } => {
                0b01_001000_1_1_0_11111_1_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldaxp { sz, rt2, rn, rt } => {
                0b1_0_001000_0_1_1_11111_1_00000_00000_00000
                    | (*sz as u32) << 30
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldaxr { size, rn, rt } => {
                0b00_001000_0_1_0_11111_1_11111_00000_00000
                    | (*size as u32) << 30
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldaxrb { rn, rt } => {
                0b00_001000_0_1_0_11111_1_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldaxrh { rn, rt } => {
                0b01_001000_0_1_0_11111_1_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldclr {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_001_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldclrb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_001_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldclrh { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_001_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldeor {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_010_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldeorb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_010_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldeorh { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_010_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldg { imm9, xn, xt } => {
                0b11011001_0_1_1_000000000_0_0_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | (*xn as u32) << 5
                    | (*xt as u32) << 0
            }
            ArmAsm::Ldgm { xn, xt } => {
                0b11011001_1_1_1_0_0_0_0_0_0_0_0_0_0_0_00000_00000
                    | (*xn as u32) << 5
                    | (*xt as u32) << 0
            }
            ArmAsm::Ldlar { size, rn, rt } => {
                0b00_001000_1_1_0_11111_0_11111_00000_00000
                    | (*size as u32) << 30
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldlarb { rn, rt } => {
                0b00_001000_1_1_0_11111_0_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldlarh { rn, rt } => {
                0b01_001000_1_1_0_11111_0_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::LdnpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
            } => {
                0b00_101_0_000_1_0000000_00000_00000_00000
                    | (*opc as u32) << 30
                    | truncate_imm::<_, 7>(*imm7) << 15
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
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
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                LdpGenSelector::PreIndex => {
                    0b00_101_0_011_1_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                LdpGenSelector::SignedOffset => {
                    0b00_101_0_010_1_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ldpsw {
                imm7,
                rt2,
                rn,
                rt,
                class_selector,
            } => match class_selector {
                LdpswSelector::PostIndex => {
                    0b01_101_0_001_1_0000000_00000_00000_00000
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                LdpswSelector::PreIndex => {
                    0b01_101_0_011_1_0000000_00000_00000_00000
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                LdpswSelector::SignedOffset => {
                    0b01_101_0_010_1_0000000_00000_00000_00000
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
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
                        | (*size as u32) << 30
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrImmGenSelector::PreIndex => {
                    0b00_111_0_00_01_0_000000000_11_00000_00000
                        | (*size as u32) << 30
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrImmGenSelector::UnsignedOffset => {
                    0b00_111_0_01_01_000000000000_00000_00000
                        | (*size as u32) << 30
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::LdrLitGen { opc, imm19, rt } => {
                0b00_011_0_00_0000000000000000000_00000
                    | (*opc as u32) << 30
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | rt << 0
            }
            ArmAsm::LdrRegGen {
                size,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_0_00_01_1_00000_000_0_10_00000_00000
                    | (*size as u32) << 30
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldra {
                m,
                s,
                imm9,
                w,
                rn,
                rt,
            } => {
                0b11_111_0_00_0_0_1_000000000_0_1_00000_00000
                    | (*m as u32) << 23
                    | (*s as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | (*w as u32) << 11
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::LdrbImm {
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                LdrbImmSelector::PostIndex => {
                    0b00_111_0_00_01_0_000000000_01_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrbImmSelector::PreIndex => {
                    0b00_111_0_00_01_0_000000000_11_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrbImmSelector::UnsignedOffset => {
                    0b00_111_0_01_01_000000000000_00000_00000
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::LdrbReg {
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_0_00_01_1_00000_000_0_10_00000_00000
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::LdrhImm {
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                LdrhImmSelector::PostIndex => {
                    0b01_111_0_00_01_0_000000000_01_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrhImmSelector::PreIndex => {
                    0b01_111_0_00_01_0_000000000_11_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrhImmSelector::UnsignedOffset => {
                    0b01_111_0_01_01_000000000000_00000_00000
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::LdrhReg {
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b01_111_0_00_01_1_00000_000_0_10_00000_00000
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::LdrsbImm {
                opc,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                LdrsbImmSelector::PostIndex => {
                    0b00_111_0_00_00_0_000000000_01_00000_00000
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrsbImmSelector::PreIndex => {
                    0b00_111_0_00_00_0_000000000_11_00000_00000
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrsbImmSelector::UnsignedOffset => {
                    0b00_111_0_01_00_000000000000_00000_00000
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::LdrsbReg {
                opc,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_0_00_00_1_00000_000_0_10_00000_00000
                    | (*opc as u32) << 22
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::LdrshImm {
                opc,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                LdrshImmSelector::PostIndex => {
                    0b01_111_0_00_00_0_000000000_01_00000_00000
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrshImmSelector::PreIndex => {
                    0b01_111_0_00_00_0_000000000_11_00000_00000
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrshImmSelector::UnsignedOffset => {
                    0b01_111_0_01_00_000000000000_00000_00000
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::LdrshReg {
                opc,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b01_111_0_00_00_1_00000_000_0_10_00000_00000
                    | (*opc as u32) << 22
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::LdrswImm {
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                LdrswImmSelector::PostIndex => {
                    0b10_111_0_00_10_0_000000000_01_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrswImmSelector::PreIndex => {
                    0b10_111_0_00_10_0_000000000_11_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrswImmSelector::UnsignedOffset => {
                    0b10_111_0_01_10_000000000000_00000_00000
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::LdrswLit { imm19, rt } => {
                0b10_011_0_00_0000000000000000000_00000
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | rt << 0
            }
            ArmAsm::LdrswReg {
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b10_111_0_00_10_1_00000_000_0_10_00000_00000
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldset {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_011_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldsetb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_011_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldseth { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_011_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldsmax {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_100_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldsmaxb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_100_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldsmaxh { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_100_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldsmin {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_101_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldsminb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_101_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldsminh { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_101_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldtr { size, imm9, rn, rt } => {
                0b00_111_0_00_01_0_000000000_10_00000_00000
                    | (*size as u32) << 30
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldtrb { imm9, rn, rt } => {
                0b00_111_0_00_01_0_000000000_10_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldtrh { imm9, rn, rt } => {
                0b01_111_0_00_01_0_000000000_10_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldtrsb { opc, imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_10_00000_00000
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldtrsh { opc, imm9, rn, rt } => {
                0b01_111_0_00_00_0_000000000_10_00000_00000
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldtrsw { imm9, rn, rt } => {
                0b10_111_0_00_10_0_000000000_10_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldumax {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_110_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldumaxb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_110_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldumaxh { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_110_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldumin {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_0_111_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Lduminb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_0_111_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Lduminh { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_0_111_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::LdurGen { size, imm9, rn, rt } => {
                0b00_111_0_00_01_0_000000000_00_00000_00000
                    | (*size as u32) << 30
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldurb { imm9, rn, rt } => {
                0b00_111_0_00_01_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldurh { imm9, rn, rt } => {
                0b01_111_0_00_01_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldursb { opc, imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_00_00000_00000
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldursh { opc, imm9, rn, rt } => {
                0b01_111_0_00_00_0_000000000_00_00000_00000
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldursw { imm9, rn, rt } => {
                0b10_111_0_00_10_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldxp { sz, rt2, rn, rt } => {
                0b1_0_001000_0_1_1_11111_0_00000_00000_00000
                    | (*sz as u32) << 30
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldxr { size, rn, rt } => {
                0b00_001000_0_1_0_11111_0_11111_00000_00000
                    | (*size as u32) << 30
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Ldxrb { rn, rt } => {
                0b00_001000_0_1_0_11111_0_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Ldxrh { rn, rt } => {
                0b01_001000_0_1_0_11111_0_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::LslLslv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_00_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Lslv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_00_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::LsrLsrv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_01_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Lsrv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_01_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Madd { sf, rm, ra, rn, rd } => {
                0b0_00_11011_000_00000_0_00000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MnegMsub { sf, rm, rn, rd } => {
                0b0_00_11011_000_00000_1_11111_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MovAddAddsubImm { sf, rn, rd } => {
                0b0_0_0_100010_0_000000000000_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::MovMovn { sf, hw, imm16, rd } => {
                0b0_00_100101_00_0000000000000000_00000
                    | (*sf as u32) << 31
                    | (*hw as u32) << 21
                    | (*imm16 as u32) << 5
                    | rd << 0
            }
            ArmAsm::MovMovz { sf, hw, imm16, rd } => {
                0b0_10_100101_00_0000000000000000_00000
                    | (*sf as u32) << 31
                    | (*hw as u32) << 21
                    | (*imm16 as u32) << 5
                    | rd << 0
            }
            ArmAsm::MovOrrLogImm {
                sf,
                n,
                immr,
                imms,
                rd,
            } => {
                0b0_01_100100_0_000000_000000_11111_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rd << 0
            }
            ArmAsm::MovOrrLogShift { sf, rm, rd } => {
                0b0_01_01010_00_0_00000_000000_11111_00000 | (*sf as u32) << 31 | rm << 16 | rd << 0
            }
            ArmAsm::Movk { sf, hw, imm16, rd } => {
                0b0_11_100101_00_0000000000000000_00000
                    | (*sf as u32) << 31
                    | (*hw as u32) << 21
                    | (*imm16 as u32) << 5
                    | rd << 0
            }
            ArmAsm::Movn { sf, hw, imm16, rd } => {
                0b0_00_100101_00_0000000000000000_00000
                    | (*sf as u32) << 31
                    | (*hw as u32) << 21
                    | (*imm16 as u32) << 5
                    | rd << 0
            }
            ArmAsm::Movz { sf, hw, imm16, rd } => {
                0b0_10_100101_00_0000000000000000_00000
                    | (*sf as u32) << 31
                    | (*hw as u32) << 21
                    | (*imm16 as u32) << 5
                    | rd << 0
            }
            ArmAsm::Mrs {
                o0,
                op1,
                crn,
                crm,
                op2,
                rt,
            } => {
                0b1101010100_1_1_0_000_0000_0000_000_00000
                    | (*o0 as u32) << 19
                    | (*op1 as u32) << 16
                    | (*crn as u32) << 12
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::MsrImm { op1, crm, op2 } => {
                0b1101010100_0_00_000_0100_0000_000_11111
                    | (*op1 as u32) << 16
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
            }
            ArmAsm::MsrReg {
                o0,
                op1,
                crn,
                crm,
                op2,
                rt,
            } => {
                0b1101010100_0_1_0_000_0000_0000_000_00000
                    | (*o0 as u32) << 19
                    | (*op1 as u32) << 16
                    | (*crn as u32) << 12
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::Msub { sf, rm, ra, rn, rd } => {
                0b0_00_11011_000_00000_1_00000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MulMadd { sf, rm, rn, rd } => {
                0b0_00_11011_000_00000_0_11111_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MvnOrnLogShift {
                sf,
                shift,
                rm,
                imm6,
                rd,
            } => {
                0b0_01_01010_00_1_00000_000000_11111_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rd << 0
            }
            ArmAsm::NegSubAddsubShift {
                sf,
                shift,
                rm,
                imm6,
                rd,
            } => {
                0b0_1_0_01011_00_0_00000_000000_11111_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rd << 0
            }
            ArmAsm::NegsSubsAddsubShift {
                sf,
                shift,
                rm,
                imm6,
                rd,
            } => {
                0b0_1_1_01011_00_0_00000_000000_11111_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rd << 0
            }
            ArmAsm::NgcSbc { sf, rm, rd } => {
                0b0_1_0_11010000_00000_000000_11111_00000 | (*sf as u32) << 31 | rm << 16 | rd << 0
            }
            ArmAsm::NgcsSbcs { sf, rm, rd } => {
                0b0_1_1_11010000_00000_000000_11111_00000 | (*sf as u32) << 31 | rm << 16 | rd << 0
            }
            ArmAsm::Nop {} => 0b1101010100_0_00_011_0010_0000_000_11111,
            ArmAsm::OrnLogShift {
                sf,
                shift,
                rm,
                imm6,
                rn,
                rd,
            } => {
                0b0_01_01010_00_1_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::OrrLogImm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_01_100100_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Pacda { z, rn, rd } => {
                0b1_1_0_11010110_00001_0_0_0_010_00000_00000 | (*z as u32) << 13 | rn << 5 | rd << 0
            }
            ArmAsm::Pacdb { z, rn, rd } => {
                0b1_1_0_11010110_00001_0_0_0_011_00000_00000 | (*z as u32) << 13 | rn << 5 | rd << 0
            }
            ArmAsm::Pacga { rm, rn, rd } => {
                0b1_0_0_11010110_00000_001100_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Pacia {
                z,
                rn,
                rd,
                crm,
                op2,
                class_selector,
            } => match class_selector {
                PaciaSelector::Integer => {
                    0b1_1_0_11010110_00001_0_0_0_000_00000_00000
                        | (*z as u32) << 13
                        | rn << 5
                        | rd << 0
                }
                PaciaSelector::System => {
                    0b1101010100_0_00_011_0010_0000_000_11111
                        | (*crm as u32) << 8
                        | (*op2 as u32) << 5
                }
            },
            ArmAsm::Pacib {
                z,
                rn,
                rd,
                crm,
                op2,
                class_selector,
            } => match class_selector {
                PacibSelector::Integer => {
                    0b1_1_0_11010110_00001_0_0_0_001_00000_00000
                        | (*z as u32) << 13
                        | rn << 5
                        | rd << 0
                }
                PacibSelector::System => {
                    0b1101010100_0_00_011_0010_0000_000_11111
                        | (*crm as u32) << 8
                        | (*op2 as u32) << 5
                }
            },
            ArmAsm::PrfmImm { imm12, rn, rt } => {
                0b11_111_0_01_10_000000000000_00000_00000
                    | truncate_imm::<_, 12>(*imm12) << 10
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::PrfmLit { imm19, rt } => {
                0b11_011_0_00_0000000000000000000_00000
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | rt << 0
            }
            ArmAsm::PrfmReg {
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b11_111_0_00_10_1_00000_000_0_10_00000_00000
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Prfum { imm9, rn, rt } => {
                0b11_111_0_00_10_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Psb {} => 0b1101010100_0_00_011_0010_0010_001_11111,
            ArmAsm::PssbbDsb {} => 0b1101010100_0_00_011_0011_0100_1_00_11111,
            ArmAsm::RbitInt { sf, rn, rd } => {
                0b0_1_0_11010110_00000_0000_00_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::Ret { rn } => 0b1101011_0_0_10_11111_0000_0_0_00000_00000 | rn << 5,
            ArmAsm::Reta { m } => 0b1101011_0_0_10_11111_0000_1_0_11111_11111 | (*m as u32) << 10,
            ArmAsm::Rev { sf, opc, rn, rd } => {
                0b0_1_0_11010110_00000_0000_00_00000_00000
                    | (*sf as u32) << 31
                    | (*opc as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Rev16Int { sf, rn, rd } => {
                0b0_1_0_11010110_00000_0000_01_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::Rev32Int { rn, rd } => {
                0b1_1_0_11010110_00000_0000_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Rev64Rev { rn, rd } => {
                0b1_1_0_11010110_00000_0000_11_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Rmif { imm6, rn, mask } => {
                0b1_0_1_11010000_000000_00001_00000_0_0000
                    | truncate_imm::<_, 6>(*imm6) << 15
                    | rn << 5
                    | (*mask as u32) << 0
            }
            ArmAsm::RorExtr {
                sf,
                n,
                rm,
                imms,
                rn,
                rd,
            } => {
                0b0_00_100111_0_0_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | rm << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::RorRorv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_11_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Rorv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_0010_11_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sb {} => 0b1101010100_0_00_011_0011_0000_1_11_11111,
            ArmAsm::Sbc { sf, rm, rn, rd } => {
                0b0_1_0_11010000_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sbcs { sf, rm, rn, rd } => {
                0b0_1_1_11010000_00000_000000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SbfizSbfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_00_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sbfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_00_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SbfxSbfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_00_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sdiv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_00001_1_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setf { sz, rn } => {
                0b0_0_1_11010000_000000_0_0010_00000_0_1101 | (*sz as u32) << 14 | rn << 5
            }
            ArmAsm::Setgp {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_1_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setgpn {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_1_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setgpt {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_1_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setgptn {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_1_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setp {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_0_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setpn {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_0_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setpt {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_0_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Setptn {
                sz,
                rs,
                op2,
                rn,
                rd,
            } => {
                0b00_011_0_01_11_0_00000_0000_01_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | (*op2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sev {} => 0b1101010100_0_00_011_0010_0000_100_11111,
            ArmAsm::Sevl {} => 0b1101010100_0_00_011_0010_0000_101_11111,
            ArmAsm::Smaddl { rm, ra, rn, rd } => {
                0b1_00_11011_0_01_00000_0_00000_00000_00000
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Smc { imm16 } => 0b11010100_000_0000000000000000_000_11 | (*imm16 as u32) << 5,
            ArmAsm::SmneglSmsubl { rm, rn, rd } => {
                0b1_00_11011_0_01_00000_1_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Smsubl { rm, ra, rn, rd } => {
                0b1_00_11011_0_01_00000_1_00000_00000_00000
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Smulh { rm, rn, rd } => {
                0b1_00_11011_0_10_00000_0_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::SmullSmaddl { rm, rn, rd } => {
                0b1_00_11011_0_01_00000_0_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::SsbbDsb {} => 0b1101010100_0_00_011_0011_0000_1_00_11111,
            ArmAsm::St2g {
                imm9,
                xn,
                xt,
                class_selector,
            } => match class_selector {
                St2gSelector::PostIndex => {
                    0b11011001_1_0_1_000000000_0_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                St2gSelector::PreIndex => {
                    0b11011001_1_0_1_000000000_1_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                St2gSelector::SignedOffset => {
                    0b11011001_1_0_1_000000000_1_0_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
            },
            ArmAsm::St64b { rn, rt } => {
                0b11_111_0_00_0_0_1_11111_1_001_00_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::St64bv { rs, rn, rt } => {
                0b11_111_0_00_0_0_1_00000_1_011_00_00000_00000 | rs << 16 | rn << 5 | rt << 0
            }
            ArmAsm::St64bv0 { rs, rn, rt } => {
                0b11_111_0_00_0_0_1_00000_1_010_00_00000_00000 | rs << 16 | rn << 5 | rt << 0
            }
            ArmAsm::StaddLdadd { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_000_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::StaddbLdaddb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_000_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StaddhLdaddh { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_000_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StclrLdclr { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_001_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::StclrbLdclrb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_001_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StclrhLdclrh { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_001_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::SteorLdeor { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_010_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::SteorbLdeorb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_010_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::SteorhLdeorh { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_010_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::Stg {
                imm9,
                xn,
                xt,
                class_selector,
            } => match class_selector {
                StgSelector::PostIndex => {
                    0b11011001_0_0_1_000000000_0_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                StgSelector::PreIndex => {
                    0b11011001_0_0_1_000000000_1_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                StgSelector::SignedOffset => {
                    0b11011001_0_0_1_000000000_1_0_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
            },
            ArmAsm::Stgm { xn, xt } => {
                0b11011001_1_0_1_0_0_0_0_0_0_0_0_0_0_0_00000_00000
                    | (*xn as u32) << 5
                    | (*xt as u32) << 0
            }
            ArmAsm::Stgp {
                simm7,
                xt2,
                xn,
                xt,
                class_selector,
            } => match class_selector {
                StgpSelector::PostIndex => {
                    0b0_1_101_0_001_0_0000000_00000_00000_00000
                        | (*simm7 as u32) << 15
                        | (*xt2 as u32) << 10
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                StgpSelector::PreIndex => {
                    0b0_1_101_0_011_0_0000000_00000_00000_00000
                        | (*simm7 as u32) << 15
                        | (*xt2 as u32) << 10
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                StgpSelector::SignedOffset => {
                    0b0_1_101_0_010_0_0000000_00000_00000_00000
                        | (*simm7 as u32) << 15
                        | (*xt2 as u32) << 10
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
            },
            ArmAsm::Stllr { size, rn, rt } => {
                0b00_001000_1_0_0_11111_0_11111_00000_00000
                    | (*size as u32) << 30
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stllrb { rn, rt } => {
                0b00_001000_1_0_0_11111_0_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Stllrh { rn, rt } => {
                0b01_001000_1_0_0_11111_0_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Stlr { size, rn, rt } => {
                0b00_001000_1_0_0_11111_1_11111_00000_00000
                    | (*size as u32) << 30
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stlrb { rn, rt } => {
                0b00_001000_1_0_0_11111_1_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::Stlrh { rn, rt } => {
                0b01_001000_1_0_0_11111_1_11111_00000_00000 | rn << 5 | rt << 0
            }
            ArmAsm::StlurGen { size, imm9, rn, rt } => {
                0b00_011001_00_0_000000000_00_00000_00000
                    | (*size as u32) << 30
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stlurb { imm9, rn, rt } => {
                0b00_011001_00_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stlurh { imm9, rn, rt } => {
                0b01_011001_00_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stlxp {
                sz,
                rs,
                rt2,
                rn,
                rt,
            } => {
                0b1_0_001000_0_0_1_00000_1_00000_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stlxr { size, rs, rn, rt } => {
                0b00_001000_0_0_0_00000_1_11111_00000_00000
                    | (*size as u32) << 30
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stlxrb { rs, rn, rt } => {
                0b00_001000_0_0_0_00000_1_11111_00000_00000 | rs << 16 | rn << 5 | rt << 0
            }
            ArmAsm::Stlxrh { rs, rn, rt } => {
                0b01_001000_0_0_0_00000_1_11111_00000_00000 | rs << 16 | rn << 5 | rt << 0
            }
            ArmAsm::StnpGen {
                opc,
                imm7,
                rt2,
                rn,
                rt,
            } => {
                0b00_101_0_000_0_0000000_00000_00000_00000
                    | (*opc as u32) << 30
                    | truncate_imm::<_, 7>(*imm7) << 15
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
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
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                StpGenSelector::PreIndex => {
                    0b00_101_0_011_0_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                StpGenSelector::SignedOffset => {
                    0b00_101_0_010_0_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
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
                        | (*size as u32) << 30
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrImmGenSelector::PreIndex => {
                    0b00_111_0_00_00_0_000000000_11_00000_00000
                        | (*size as u32) << 30
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrImmGenSelector::UnsignedOffset => {
                    0b00_111_0_01_00_000000000000_00000_00000
                        | (*size as u32) << 30
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
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
                    | (*size as u32) << 30
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::StrbImm {
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                StrbImmSelector::PostIndex => {
                    0b00_111_0_00_00_0_000000000_01_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrbImmSelector::PreIndex => {
                    0b00_111_0_00_00_0_000000000_11_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrbImmSelector::UnsignedOffset => {
                    0b00_111_0_01_00_000000000000_00000_00000
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::StrbReg {
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_0_00_00_1_00000_000_0_10_00000_00000
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::StrhImm {
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                StrhImmSelector::PostIndex => {
                    0b01_111_0_00_00_0_000000000_01_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrhImmSelector::PreIndex => {
                    0b01_111_0_00_00_0_000000000_11_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrhImmSelector::UnsignedOffset => {
                    0b01_111_0_01_00_000000000000_00000_00000
                        | truncate_imm::<_, 12>(*imm12) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::StrhReg {
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b01_111_0_00_00_1_00000_000_0_10_00000_00000
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::StsetLdset { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_011_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::StsetbLdsetb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_011_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StsethLdseth { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_011_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StsmaxLdsmax { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_100_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::StsmaxbLdsmaxb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_100_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StsmaxhLdsmaxh { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_100_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StsminLdsmin { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_101_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::StsminbLdsminb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_101_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StsminhLdsminh { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_101_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::Sttr { size, imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_10_00000_00000
                    | (*size as u32) << 30
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Sttrb { imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_10_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Sttrh { imm9, rn, rt } => {
                0b01_111_0_00_00_0_000000000_10_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::StumaxLdumax { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_110_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::StumaxbLdumaxb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_110_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StumaxhLdumaxh { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_110_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StuminLdumin { size, r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_111_00_00000_11111
                    | (*size as u32) << 30
                    | r << 22
                    | rs << 16
                    | rn << 5
            }
            ArmAsm::StuminbLduminb { r, rs, rn } => {
                0b00_111_0_00_0_0_1_00000_0_111_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::StuminhLduminh { r, rs, rn } => {
                0b01_111_0_00_0_0_1_00000_0_111_00_00000_11111 | r << 22 | rs << 16 | rn << 5
            }
            ArmAsm::SturGen { size, imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_00_00000_00000
                    | (*size as u32) << 30
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Sturb { imm9, rn, rt } => {
                0b00_111_0_00_00_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Sturh { imm9, rn, rt } => {
                0b01_111_0_00_00_0_000000000_00_00000_00000
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stxp {
                sz,
                rs,
                rt2,
                rn,
                rt,
            } => {
                0b1_0_001000_0_0_1_00000_0_00000_00000_00000
                    | (*sz as u32) << 30
                    | rs << 16
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stxr { size, rs, rn, rt } => {
                0b00_001000_0_0_0_00000_0_11111_00000_00000
                    | (*size as u32) << 30
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Stxrb { rs, rn, rt } => {
                0b00_001000_0_0_0_00000_0_11111_00000_00000 | rs << 16 | rn << 5 | rt << 0
            }
            ArmAsm::Stxrh { rs, rn, rt } => {
                0b01_001000_0_0_0_00000_0_11111_00000_00000 | rs << 16 | rn << 5 | rt << 0
            }
            ArmAsm::Stz2g {
                imm9,
                xn,
                xt,
                class_selector,
            } => match class_selector {
                Stz2gSelector::PostIndex => {
                    0b11011001_1_1_1_000000000_0_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                Stz2gSelector::PreIndex => {
                    0b11011001_1_1_1_000000000_1_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                Stz2gSelector::SignedOffset => {
                    0b11011001_1_1_1_000000000_1_0_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
            },
            ArmAsm::Stzg {
                imm9,
                xn,
                xt,
                class_selector,
            } => match class_selector {
                StzgSelector::PostIndex => {
                    0b11011001_0_1_1_000000000_0_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                StzgSelector::PreIndex => {
                    0b11011001_0_1_1_000000000_1_1_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
                StzgSelector::SignedOffset => {
                    0b11011001_0_1_1_000000000_1_0_00000_00000
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | (*xn as u32) << 5
                        | (*xt as u32) << 0
                }
            },
            ArmAsm::Stzgm { xn, xt } => {
                0b11011001_0_0_1_0_0_0_0_0_0_0_0_0_0_0_00000_00000
                    | (*xn as u32) << 5
                    | (*xt as u32) << 0
            }
            ArmAsm::SubAddsubExt {
                sf,
                rm,
                option,
                imm3,
                rn,
                rd,
            } => {
                0b0_1_0_01011_00_1_00000_000_000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*option as u32) << 13
                    | truncate_imm::<_, 3>(*imm3) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SubAddsubImm {
                sf,
                sh,
                imm12,
                rn,
                rd,
            } => {
                0b0_1_0_100010_0_000000000000_00000_00000
                    | (*sf as u32) << 31
                    | (*sh as u32) << 22
                    | truncate_imm::<_, 12>(*imm12) << 10
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Subg {
                uimm6,
                uimm4,
                xn,
                xd,
            } => {
                0b1_1_0_100011_0_000000_00_0000_00000_00000
                    | (*uimm6 as u32) << 16
                    | (*uimm4 as u32) << 10
                    | (*xn as u32) << 5
                    | (*xd as u32) << 0
            }
            ArmAsm::Subp { xm, xn, xd } => {
                0b1_0_0_11010110_00000_0_0_0_0_0_0_00000_00000
                    | (*xm as u32) << 16
                    | (*xn as u32) << 5
                    | (*xd as u32) << 0
            }
            ArmAsm::Subps { xm, xn, xd } => {
                0b1_0_1_11010110_00000_0_0_0_0_0_0_00000_00000
                    | (*xm as u32) << 16
                    | (*xn as u32) << 5
                    | (*xd as u32) << 0
            }
            ArmAsm::SubsAddsubExt {
                sf,
                rm,
                option,
                imm3,
                rn,
                rd,
            } => {
                0b0_1_1_01011_00_1_00000_000_000_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | (*option as u32) << 13
                    | truncate_imm::<_, 3>(*imm3) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SubsAddsubImm {
                sf,
                sh,
                imm12,
                rn,
                rd,
            } => {
                0b0_1_1_100010_0_000000000000_00000_00000
                    | (*sf as u32) << 31
                    | (*sh as u32) << 22
                    | truncate_imm::<_, 12>(*imm12) << 10
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Svc { imm16 } => 0b11010100_000_0000000000000000_000_01 | (*imm16 as u32) << 5,
            ArmAsm::Swp {
                size,
                a,
                r,
                rs,
                rn,
                rt,
            } => {
                0b00_111_0_00_0_0_1_00000_1_000_00_00000_00000
                    | (*size as u32) << 30
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Swpb { a, r, rs, rn, rt } => {
                0b00_111_0_00_0_0_1_00000_1_000_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::Swph { a, r, rs, rn, rt } => {
                0b01_111_0_00_0_0_1_00000_1_000_00_00000_00000
                    | (*a as u32) << 23
                    | r << 22
                    | rs << 16
                    | rn << 5
                    | rt << 0
            }
            ArmAsm::SxtbSbfm { sf, n, rn, rd } => {
                0b0_00_100110_0_000000_000111_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SxthSbfm { sf, n, rn, rd } => {
                0b0_00_100110_0_000000_001111_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SxtwSbfm { rn, rd } => {
                0b1_00_100110_1_000000_011111_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Sys {
                op1,
                crn,
                crm,
                op2,
                rt,
            } => {
                0b1101010100_0_01_000_0000_0000_000_00000
                    | (*op1 as u32) << 16
                    | (*crn as u32) << 12
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::Sysl {
                op1,
                crn,
                crm,
                op2,
                rt,
            } => {
                0b1101010100_1_01_000_0000_0000_000_00000
                    | (*op1 as u32) << 16
                    | (*crn as u32) << 12
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::Tbnz { b5, b40, imm14, rt } => {
                0b0_011011_1_00000_00000000000000_00000
                    | (*b5 as u32) << 31
                    | (*b40 as u32) << 19
                    | truncate_imm::<_, 14>(*imm14) << 5
                    | rt << 0
            }
            ArmAsm::Tbz { b5, b40, imm14, rt } => {
                0b0_011011_0_00000_00000000000000_00000
                    | (*b5 as u32) << 31
                    | (*b40 as u32) << 19
                    | truncate_imm::<_, 14>(*imm14) << 5
                    | rt << 0
            }
            ArmAsm::TlbiSys { op1, crm, op2, rt } => {
                0b1101010100_0_01_000_1000_0000_000_00000
                    | (*op1 as u32) << 16
                    | (*crm as u32) << 8
                    | (*op2 as u32) << 5
                    | rt << 0
            }
            ArmAsm::Tsb {} => 0b1101010100_0_00_011_0010_0010_010_11111,
            ArmAsm::TstAndsLogImm {
                sf,
                n,
                immr,
                imms,
                rn,
            } => {
                0b0_11_100100_0_000000_000000_00000_11111
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
            }
            ArmAsm::TstAndsLogShift {
                sf,
                shift,
                rm,
                imm6,
                rn,
            } => {
                0b0_11_01010_00_0_00000_000000_00000_11111
                    | (*sf as u32) << 31
                    | (*shift as u32) << 22
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
            }
            ArmAsm::UbfizUbfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_10_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Ubfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_10_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UbfxUbfm {
                sf,
                n,
                immr,
                imms,
                rn,
                rd,
            } => {
                0b0_10_100110_0_000000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*n as u32) << 22
                    | (*immr as u32) << 16
                    | (*imms as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UdfPermUndef { imm16 } => {
                0b0000000000000000_0000000000000000 | (*imm16 as u32) << 0
            }
            ArmAsm::Udiv { sf, rm, rn, rd } => {
                0b0_0_0_11010110_00000_00001_0_00000_00000
                    | (*sf as u32) << 31
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Umaddl { rm, ra, rn, rd } => {
                0b1_00_11011_1_01_00000_0_00000_00000_00000
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmneglUmsubl { rm, rn, rd } => {
                0b1_00_11011_1_01_00000_1_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Umsubl { rm, ra, rn, rd } => {
                0b1_00_11011_1_01_00000_1_00000_00000_00000
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Umulh { rm, rn, rd } => {
                0b1_00_11011_1_10_00000_0_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::UmullUmaddl { rm, rn, rd } => {
                0b1_00_11011_1_01_00000_0_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::UxtbUbfm { rn, rd } => {
                0b0_10_100110_0_000000_000111_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::UxthUbfm { rn, rd } => {
                0b0_10_100110_0_000000_001111_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Wfe {} => 0b1101010100_0_00_011_0010_0000_010_11111,
            ArmAsm::Wfet { rd } => 0b11010101000000110001_0000_000_00000 | rd << 0,
            ArmAsm::Wfi {} => 0b1101010100_0_00_011_0010_0000_011_11111,
            ArmAsm::Wfit { rd } => 0b11010101000000110001_0000_001_00000 | rd << 0,
            ArmAsm::Xaflag {} => 0b1101010100_0_00_000_0100_0000_001_11111,
            ArmAsm::Xpac {
                d,
                rd,
                class_selector,
            } => match class_selector {
                XpacSelector::Integer => {
                    0b1_1_0_11010110_00001_0_1_000_0_11111_00000 | (*d as u32) << 10 | rd << 0
                }
                XpacSelector::System => 0b1101010100_0_00_011_0010_0000_111_11111,
            },
            ArmAsm::Yield {} => 0b1101010100_0_00_011_0010_0000_001_11111,
        }
    }
}
