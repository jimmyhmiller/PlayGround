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
    /// ABS -- A64
    /// Absolute value (vector)
    /// ABS  <V><d>, <V><n>
    /// ABS  <Vd>.<T>, <Vn>.<T>
    AbsAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: AbsAdvsimdSelector,
    },
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
    /// ADD (vector) -- A64
    /// Add (vector)
    /// ADD  <V><d>, <V><n>, <V><m>
    /// ADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    AddAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: AddAdvsimdSelector,
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
    /// ADDHN, ADDHN2 -- A64
    /// Add returning High Narrow
    /// ADDHN{2}  <Vd>.<Tb>, <Vn>.<Ta>, <Vm>.<Ta>
    AddhnAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// ADDP (scalar) -- A64
    /// Add Pair of elements (scalar)
    /// ADDP  <V><d>, <Vn>.<T>
    AddpAdvsimdPair {
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// ADDP (vector) -- A64
    /// Add Pairwise (vector)
    /// ADDP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    AddpAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
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
    /// ADDV -- A64
    /// Add across Vector
    /// ADDV  <V><d>, <Vn>.<T>
    AddvAdvsimd {
        q: i32,
        size: i32,
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
    /// AESD -- A64
    /// AES single round decryption
    /// AESD  <Vd>.16B, <Vn>.16B
    AesdAdvsimd { rn: Register, rd: Register },
    /// AESE -- A64
    /// AES single round encryption
    /// AESE  <Vd>.16B, <Vn>.16B
    AeseAdvsimd { rn: Register, rd: Register },
    /// AESIMC -- A64
    /// AES inverse mix columns
    /// AESIMC  <Vd>.16B, <Vn>.16B
    AesimcAdvsimd { rn: Register, rd: Register },
    /// AESMC -- A64
    /// AES mix columns
    /// AESMC  <Vd>.16B, <Vn>.16B
    AesmcAdvsimd { rn: Register, rd: Register },
    /// AND (vector) -- A64
    /// Bitwise AND (vector)
    /// AND  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    AndAdvsimd {
        q: i32,
        rm: Register,
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
    /// BCAX -- A64
    /// Bit Clear and XOR
    /// BCAX  <Vd>.16B, <Vn>.16B, <Vm>.16B, <Va>.16B
    BcaxAdvsimd {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
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
    /// BFCVT -- A64
    /// Floating-point convert from single-precision to BFloat16 format (scalar)
    /// BFCVT  <Hd>, <Sn>
    BfcvtFloat { rn: Register, rd: Register },
    /// BFCVTN, BFCVTN2 -- A64
    /// Floating-point convert from single-precision to BFloat16 format (vector)
    /// BFCVTN{2}  <Vd>.<Ta>, <Vn>.4S
    BfcvtnAdvsimd { q: i32, rn: Register, rd: Register },
    /// BFDOT (by element) -- A64
    /// BFloat16 floating-point dot product (vector, by element)
    /// BFDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.2H[<index>]
    BfdotAdvsimdElt {
        q: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// BFDOT (vector) -- A64
    /// BFloat16 floating-point dot product (vector)
    /// BFDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    BfdotAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
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
    /// BFMLALB, BFMLALT (by element) -- A64
    /// BFloat16 floating-point widening multiply-add long (by element)
    /// BFMLAL<bt>  <Vd>.4S, <Vn>.8H, <Vm>.H[<index>]
    BfmlalAdvsimdElt {
        q: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// BFMLALB, BFMLALT (vector) -- A64
    /// BFloat16 floating-point widening multiply-add long (vector)
    /// BFMLAL<bt>  <Vd>.4S, <Vn>.8H, <Vm>.8H
    BfmlalAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// BFMMLA -- A64
    /// BFloat16 floating-point matrix multiply-accumulate into 2x2 matrix
    /// BFMMLA  <Vd>.4S, <Vn>.8H, <Vm>.8H
    BfmmlaAdvsimd {
        rm: Register,
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
    /// BIC (vector, immediate) -- A64
    /// Bitwise bit Clear (vector, immediate)
    /// BIC  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    /// BIC  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    BicAdvsimdImm {
        q: i32,
        a: i32,
        b: i32,
        c: i32,
        cmode: i32,
        d: i32,
        e: i32,
        f: i32,
        g: i32,
        h: i32,
        rd: Register,
    },
    /// BIC (vector, register) -- A64
    /// Bitwise bit Clear (vector, register)
    /// BIC  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    BicAdvsimdReg {
        q: i32,
        rm: Register,
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
    /// BIF -- A64
    /// Bitwise Insert if False
    /// BIF  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    BifAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// BIT -- A64
    /// Bitwise Insert if True
    /// BIT  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    BitAdvsimd {
        q: i32,
        rm: Register,
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
    /// BSL -- A64
    /// Bitwise Select
    /// BSL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    BslAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
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
    /// CLS (vector) -- A64
    /// Count Leading Sign bits (vector)
    /// CLS  <Vd>.<T>, <Vn>.<T>
    ClsAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// CLS -- A64
    /// Count Leading Sign bits
    /// CLS  <Wd>, <Wn>
    /// CLS  <Xd>, <Xn>
    ClsInt { sf: i32, rn: Register, rd: Register },
    /// CLZ (vector) -- A64
    /// Count Leading Zero bits (vector)
    /// CLZ  <Vd>.<T>, <Vn>.<T>
    ClzAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// CLZ -- A64
    /// Count Leading Zeros
    /// CLZ  <Wd>, <Wn>
    /// CLZ  <Xd>, <Xn>
    ClzInt { sf: i32, rn: Register, rd: Register },
    /// CMEQ (register) -- A64
    /// Compare bitwise Equal (vector)
    /// CMEQ  <V><d>, <V><n>, <V><m>
    /// CMEQ  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    CmeqAdvsimdReg {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmeqAdvsimdRegSelector,
    },
    /// CMEQ (zero) -- A64
    /// Compare bitwise Equal to zero (vector)
    /// CMEQ  <V><d>, <V><n>, #0
    /// CMEQ  <Vd>.<T>, <Vn>.<T>, #0
    CmeqAdvsimdZero {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmeqAdvsimdZeroSelector,
    },
    /// CMGE (register) -- A64
    /// Compare signed Greater than or Equal (vector)
    /// CMGE  <V><d>, <V><n>, <V><m>
    /// CMGE  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    CmgeAdvsimdReg {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmgeAdvsimdRegSelector,
    },
    /// CMGE (zero) -- A64
    /// Compare signed Greater than or Equal to zero (vector)
    /// CMGE  <V><d>, <V><n>, #0
    /// CMGE  <Vd>.<T>, <Vn>.<T>, #0
    CmgeAdvsimdZero {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmgeAdvsimdZeroSelector,
    },
    /// CMGT (register) -- A64
    /// Compare signed Greater than (vector)
    /// CMGT  <V><d>, <V><n>, <V><m>
    /// CMGT  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    CmgtAdvsimdReg {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmgtAdvsimdRegSelector,
    },
    /// CMGT (zero) -- A64
    /// Compare signed Greater than zero (vector)
    /// CMGT  <V><d>, <V><n>, #0
    /// CMGT  <Vd>.<T>, <Vn>.<T>, #0
    CmgtAdvsimdZero {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmgtAdvsimdZeroSelector,
    },
    /// CMHI (register) -- A64
    /// Compare unsigned Higher (vector)
    /// CMHI  <V><d>, <V><n>, <V><m>
    /// CMHI  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    CmhiAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmhiAdvsimdSelector,
    },
    /// CMHS (register) -- A64
    /// Compare unsigned Higher or Same (vector)
    /// CMHS  <V><d>, <V><n>, <V><m>
    /// CMHS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    CmhsAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmhsAdvsimdSelector,
    },
    /// CMLE (zero) -- A64
    /// Compare signed Less than or Equal to zero (vector)
    /// CMLE  <V><d>, <V><n>, #0
    /// CMLE  <Vd>.<T>, <Vn>.<T>, #0
    CmleAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmleAdvsimdSelector,
    },
    /// CMLT (zero) -- A64
    /// Compare signed Less than zero (vector)
    /// CMLT  <V><d>, <V><n>, #0
    /// CMLT  <Vd>.<T>, <Vn>.<T>, #0
    CmltAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmltAdvsimdSelector,
    },
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
    /// CMTST -- A64
    /// Compare bitwise Test bits nonzero (vector)
    /// CMTST  <V><d>, <V><n>, <V><m>
    /// CMTST  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    CmtstAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: CmtstAdvsimdSelector,
    },
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
    /// CNT -- A64
    /// Population Count per byte
    /// CNT  <Vd>.<T>, <Vn>.<T>
    CntAdvsimd {
        q: i32,
        size: i32,
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
    /// DUP (element) -- A64
    /// Duplicate vector element to vector or scalar
    /// DUP  <V><d>, <Vn>.<T>[<index>]
    /// DUP  <Vd>.<T>, <Vn>.<Ts>[<index>]
    DupAdvsimdElt {
        imm5: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: DupAdvsimdEltSelector,
    },
    /// DUP (general) -- A64
    /// Duplicate general-purpose register to vector
    /// DUP  <Vd>.<T>, <R><n>
    DupAdvsimdGen {
        q: i32,
        imm5: i32,
        rn: Register,
        rd: Register,
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
    /// EOR3 -- A64
    /// Three-way Exclusive OR
    /// EOR3  <Vd>.16B, <Vn>.16B, <Vm>.16B, <Va>.16B
    Eor3Advsimd {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// EOR (vector) -- A64
    /// Bitwise Exclusive OR (vector)
    /// EOR  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    EorAdvsimd {
        q: i32,
        rm: Register,
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
    /// EXT -- A64
    /// Extract vector from pair of vectors
    /// EXT  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<index>
    ExtAdvsimd {
        q: i32,
        rm: Register,
        imm4: i32,
        rn: Register,
        rd: Register,
    },
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
    /// FABD -- A64
    /// Floating-point Absolute Difference (vector)
    /// FABD  <Hd>, <Hn>, <Hm>
    /// FABD  <V><d>, <V><n>, <V><m>
    /// FABD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FABD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FabdAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FabdAdvsimdSelector,
    },
    /// FABS (vector) -- A64
    /// Floating-point Absolute value (vector)
    /// FABS  <Vd>.<T>, <Vn>.<T>
    /// FABS  <Vd>.<T>, <Vn>.<T>
    FabsAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FabsAdvsimdSelector,
    },
    /// FABS (scalar) -- A64
    /// Floating-point Absolute value (scalar)
    /// FABS  <Hd>, <Hn>
    /// FABS  <Sd>, <Sn>
    /// FABS  <Dd>, <Dn>
    FabsFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FACGE -- A64
    /// Floating-point Absolute Compare Greater than or Equal (vector)
    /// FACGE  <Hd>, <Hn>, <Hm>
    /// FACGE  <V><d>, <V><n>, <V><m>
    /// FACGE  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FACGE  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FacgeAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FacgeAdvsimdSelector,
    },
    /// FACGT -- A64
    /// Floating-point Absolute Compare Greater than (vector)
    /// FACGT  <Hd>, <Hn>, <Hm>
    /// FACGT  <V><d>, <V><n>, <V><m>
    /// FACGT  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FACGT  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FacgtAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FacgtAdvsimdSelector,
    },
    /// FADD (vector) -- A64
    /// Floating-point Add (vector)
    /// FADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FaddAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FaddAdvsimdSelector,
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
    /// FADDP (scalar) -- A64
    /// Floating-point Add Pair of elements (scalar)
    /// FADDP  <V><d>, <Vn>.<T>
    /// FADDP  <V><d>, <Vn>.<T>
    FaddpAdvsimdPair {
        sz: i32,
        rn: Register,
        rd: Register,
        class_selector: FaddpAdvsimdPairSelector,
    },
    /// FADDP (vector) -- A64
    /// Floating-point Add Pairwise (vector)
    /// FADDP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FADDP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FaddpAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FaddpAdvsimdVecSelector,
    },
    /// FCADD -- A64
    /// Floating-point Complex Add
    /// FCADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<rotate>
    FcaddAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rot: i32,
        rn: Register,
        rd: Register,
    },
    /// FCCMP -- A64
    /// Floating-point Conditional quiet Compare (scalar)
    /// FCCMP  <Hn>, <Hm>, #<nzcv>, <cond>
    /// FCCMP  <Sn>, <Sm>, #<nzcv>, <cond>
    /// FCCMP  <Dn>, <Dm>, #<nzcv>, <cond>
    FccmpFloat {
        ftype: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        nzcv: i32,
    },
    /// FCCMPE -- A64
    /// Floating-point Conditional signaling Compare (scalar)
    /// FCCMPE  <Hn>, <Hm>, #<nzcv>, <cond>
    /// FCCMPE  <Sn>, <Sm>, #<nzcv>, <cond>
    /// FCCMPE  <Dn>, <Dm>, #<nzcv>, <cond>
    FccmpeFloat {
        ftype: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        nzcv: i32,
    },
    /// FCMEQ (register) -- A64
    /// Floating-point Compare Equal (vector)
    /// FCMEQ  <Hd>, <Hn>, <Hm>
    /// FCMEQ  <V><d>, <V><n>, <V><m>
    /// FCMEQ  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FCMEQ  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FcmeqAdvsimdReg {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmeqAdvsimdRegSelector,
    },
    /// FCMEQ (zero) -- A64
    /// Floating-point Compare Equal to zero (vector)
    /// FCMEQ  <Hd>, <Hn>, #0.0
    /// FCMEQ  <V><d>, <V><n>, #0.0
    /// FCMEQ  <Vd>.<T>, <Vn>.<T>, #0.0
    /// FCMEQ  <Vd>.<T>, <Vn>.<T>, #0.0
    FcmeqAdvsimdZero {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmeqAdvsimdZeroSelector,
    },
    /// FCMGE (register) -- A64
    /// Floating-point Compare Greater than or Equal (vector)
    /// FCMGE  <Hd>, <Hn>, <Hm>
    /// FCMGE  <V><d>, <V><n>, <V><m>
    /// FCMGE  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FCMGE  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FcmgeAdvsimdReg {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmgeAdvsimdRegSelector,
    },
    /// FCMGE (zero) -- A64
    /// Floating-point Compare Greater than or Equal to zero (vector)
    /// FCMGE  <Hd>, <Hn>, #0.0
    /// FCMGE  <V><d>, <V><n>, #0.0
    /// FCMGE  <Vd>.<T>, <Vn>.<T>, #0.0
    /// FCMGE  <Vd>.<T>, <Vn>.<T>, #0.0
    FcmgeAdvsimdZero {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmgeAdvsimdZeroSelector,
    },
    /// FCMGT (register) -- A64
    /// Floating-point Compare Greater than (vector)
    /// FCMGT  <Hd>, <Hn>, <Hm>
    /// FCMGT  <V><d>, <V><n>, <V><m>
    /// FCMGT  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FCMGT  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FcmgtAdvsimdReg {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmgtAdvsimdRegSelector,
    },
    /// FCMGT (zero) -- A64
    /// Floating-point Compare Greater than zero (vector)
    /// FCMGT  <Hd>, <Hn>, #0.0
    /// FCMGT  <V><d>, <V><n>, #0.0
    /// FCMGT  <Vd>.<T>, <Vn>.<T>, #0.0
    /// FCMGT  <Vd>.<T>, <Vn>.<T>, #0.0
    FcmgtAdvsimdZero {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmgtAdvsimdZeroSelector,
    },
    /// FCMLA (by element) -- A64
    /// Floating-point Complex Multiply Accumulate (by element)
    /// FCMLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>], #<rotate>
    /// FCMLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>], #<rotate>
    FcmlaAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        rot: i32,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// FCMLA -- A64
    /// Floating-point Complex Multiply Accumulate
    /// FCMLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>, #<rotate>
    FcmlaAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rot: i32,
        rn: Register,
        rd: Register,
    },
    /// FCMLE (zero) -- A64
    /// Floating-point Compare Less than or Equal to zero (vector)
    /// FCMLE  <Hd>, <Hn>, #0.0
    /// FCMLE  <V><d>, <V><n>, #0.0
    /// FCMLE  <Vd>.<T>, <Vn>.<T>, #0.0
    /// FCMLE  <Vd>.<T>, <Vn>.<T>, #0.0
    FcmleAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmleAdvsimdSelector,
    },
    /// FCMLT (zero) -- A64
    /// Floating-point Compare Less than zero (vector)
    /// FCMLT  <Hd>, <Hn>, #0.0
    /// FCMLT  <V><d>, <V><n>, #0.0
    /// FCMLT  <Vd>.<T>, <Vn>.<T>, #0.0
    /// FCMLT  <Vd>.<T>, <Vn>.<T>, #0.0
    FcmltAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcmltAdvsimdSelector,
    },
    /// FCMP -- A64
    /// Floating-point quiet Compare (scalar)
    /// FCMP  <Hn>, <Hm>
    /// FCMP  <Hn>, #0.0
    /// FCMP  <Sn>, <Sm>
    /// FCMP  <Sn>, #0.0
    /// FCMP  <Dn>, <Dm>
    /// FCMP  <Dn>, #0.0
    FcmpFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        opc: i32,
    },
    /// FCMPE -- A64
    /// Floating-point signaling Compare (scalar)
    /// FCMPE  <Hn>, <Hm>
    /// FCMPE  <Hn>, #0.0
    /// FCMPE  <Sn>, <Sm>
    /// FCMPE  <Sn>, #0.0
    /// FCMPE  <Dn>, <Dm>
    /// FCMPE  <Dn>, #0.0
    FcmpeFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        opc: i32,
    },
    /// FCSEL -- A64
    /// Floating-point Conditional Select (scalar)
    /// FCSEL  <Hd>, <Hn>, <Hm>, <cond>
    /// FCSEL  <Sd>, <Sn>, <Sm>, <cond>
    /// FCSEL  <Dd>, <Dn>, <Dm>, <cond>
    FcselFloat {
        ftype: i32,
        rm: Register,
        cond: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVT -- A64
    /// Floating-point Convert precision (scalar)
    /// FCVT  <Sd>, <Hn>
    /// FCVT  <Dd>, <Hn>
    /// FCVT  <Hd>, <Sn>
    /// FCVT  <Dd>, <Sn>
    /// FCVT  <Hd>, <Dn>
    /// FCVT  <Sd>, <Dn>
    FcvtFloat {
        ftype: i32,
        opc: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTAS (vector) -- A64
    /// Floating-point Convert to Signed integer, rounding to nearest with ties to Away (vector)
    /// FCVTAS  <Hd>, <Hn>
    /// FCVTAS  <V><d>, <V><n>
    /// FCVTAS  <Vd>.<T>, <Vn>.<T>
    /// FCVTAS  <Vd>.<T>, <Vn>.<T>
    FcvtasAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtasAdvsimdSelector,
    },
    /// FCVTAS (scalar) -- A64
    /// Floating-point Convert to Signed integer, rounding to nearest with ties to Away (scalar)
    /// FCVTAS  <Wd>, <Hn>
    /// FCVTAS  <Xd>, <Hn>
    /// FCVTAS  <Wd>, <Sn>
    /// FCVTAS  <Xd>, <Sn>
    /// FCVTAS  <Wd>, <Dn>
    /// FCVTAS  <Xd>, <Dn>
    FcvtasFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTAU (vector) -- A64
    /// Floating-point Convert to Unsigned integer, rounding to nearest with ties to Away (vector)
    /// FCVTAU  <Hd>, <Hn>
    /// FCVTAU  <V><d>, <V><n>
    /// FCVTAU  <Vd>.<T>, <Vn>.<T>
    /// FCVTAU  <Vd>.<T>, <Vn>.<T>
    FcvtauAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtauAdvsimdSelector,
    },
    /// FCVTAU (scalar) -- A64
    /// Floating-point Convert to Unsigned integer, rounding to nearest with ties to Away (scalar)
    /// FCVTAU  <Wd>, <Hn>
    /// FCVTAU  <Xd>, <Hn>
    /// FCVTAU  <Wd>, <Sn>
    /// FCVTAU  <Xd>, <Sn>
    /// FCVTAU  <Wd>, <Dn>
    /// FCVTAU  <Xd>, <Dn>
    FcvtauFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTL, FCVTL2 -- A64
    /// Floating-point Convert to higher precision Long (vector)
    /// FCVTL{2}  <Vd>.<Ta>, <Vn>.<Tb>
    FcvtlAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTMS (vector) -- A64
    /// Floating-point Convert to Signed integer, rounding toward Minus infinity (vector)
    /// FCVTMS  <Hd>, <Hn>
    /// FCVTMS  <V><d>, <V><n>
    /// FCVTMS  <Vd>.<T>, <Vn>.<T>
    /// FCVTMS  <Vd>.<T>, <Vn>.<T>
    FcvtmsAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtmsAdvsimdSelector,
    },
    /// FCVTMS (scalar) -- A64
    /// Floating-point Convert to Signed integer, rounding toward Minus infinity (scalar)
    /// FCVTMS  <Wd>, <Hn>
    /// FCVTMS  <Xd>, <Hn>
    /// FCVTMS  <Wd>, <Sn>
    /// FCVTMS  <Xd>, <Sn>
    /// FCVTMS  <Wd>, <Dn>
    /// FCVTMS  <Xd>, <Dn>
    FcvtmsFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTMU (vector) -- A64
    /// Floating-point Convert to Unsigned integer, rounding toward Minus infinity (vector)
    /// FCVTMU  <Hd>, <Hn>
    /// FCVTMU  <V><d>, <V><n>
    /// FCVTMU  <Vd>.<T>, <Vn>.<T>
    /// FCVTMU  <Vd>.<T>, <Vn>.<T>
    FcvtmuAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtmuAdvsimdSelector,
    },
    /// FCVTMU (scalar) -- A64
    /// Floating-point Convert to Unsigned integer, rounding toward Minus infinity (scalar)
    /// FCVTMU  <Wd>, <Hn>
    /// FCVTMU  <Xd>, <Hn>
    /// FCVTMU  <Wd>, <Sn>
    /// FCVTMU  <Xd>, <Sn>
    /// FCVTMU  <Wd>, <Dn>
    /// FCVTMU  <Xd>, <Dn>
    FcvtmuFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTN, FCVTN2 -- A64
    /// Floating-point Convert to lower precision Narrow (vector)
    /// FCVTN{2}  <Vd>.<Tb>, <Vn>.<Ta>
    FcvtnAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTNS (vector) -- A64
    /// Floating-point Convert to Signed integer, rounding to nearest with ties to even (vector)
    /// FCVTNS  <Hd>, <Hn>
    /// FCVTNS  <V><d>, <V><n>
    /// FCVTNS  <Vd>.<T>, <Vn>.<T>
    /// FCVTNS  <Vd>.<T>, <Vn>.<T>
    FcvtnsAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtnsAdvsimdSelector,
    },
    /// FCVTNS (scalar) -- A64
    /// Floating-point Convert to Signed integer, rounding to nearest with ties to even (scalar)
    /// FCVTNS  <Wd>, <Hn>
    /// FCVTNS  <Xd>, <Hn>
    /// FCVTNS  <Wd>, <Sn>
    /// FCVTNS  <Xd>, <Sn>
    /// FCVTNS  <Wd>, <Dn>
    /// FCVTNS  <Xd>, <Dn>
    FcvtnsFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTNU (vector) -- A64
    /// Floating-point Convert to Unsigned integer, rounding to nearest with ties to even (vector)
    /// FCVTNU  <Hd>, <Hn>
    /// FCVTNU  <V><d>, <V><n>
    /// FCVTNU  <Vd>.<T>, <Vn>.<T>
    /// FCVTNU  <Vd>.<T>, <Vn>.<T>
    FcvtnuAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtnuAdvsimdSelector,
    },
    /// FCVTNU (scalar) -- A64
    /// Floating-point Convert to Unsigned integer, rounding to nearest with ties to even (scalar)
    /// FCVTNU  <Wd>, <Hn>
    /// FCVTNU  <Xd>, <Hn>
    /// FCVTNU  <Wd>, <Sn>
    /// FCVTNU  <Xd>, <Sn>
    /// FCVTNU  <Wd>, <Dn>
    /// FCVTNU  <Xd>, <Dn>
    FcvtnuFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTPS (vector) -- A64
    /// Floating-point Convert to Signed integer, rounding toward Plus infinity (vector)
    /// FCVTPS  <Hd>, <Hn>
    /// FCVTPS  <V><d>, <V><n>
    /// FCVTPS  <Vd>.<T>, <Vn>.<T>
    /// FCVTPS  <Vd>.<T>, <Vn>.<T>
    FcvtpsAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtpsAdvsimdSelector,
    },
    /// FCVTPS (scalar) -- A64
    /// Floating-point Convert to Signed integer, rounding toward Plus infinity (scalar)
    /// FCVTPS  <Wd>, <Hn>
    /// FCVTPS  <Xd>, <Hn>
    /// FCVTPS  <Wd>, <Sn>
    /// FCVTPS  <Xd>, <Sn>
    /// FCVTPS  <Wd>, <Dn>
    /// FCVTPS  <Xd>, <Dn>
    FcvtpsFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTPU (vector) -- A64
    /// Floating-point Convert to Unsigned integer, rounding toward Plus infinity (vector)
    /// FCVTPU  <Hd>, <Hn>
    /// FCVTPU  <V><d>, <V><n>
    /// FCVTPU  <Vd>.<T>, <Vn>.<T>
    /// FCVTPU  <Vd>.<T>, <Vn>.<T>
    FcvtpuAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtpuAdvsimdSelector,
    },
    /// FCVTPU (scalar) -- A64
    /// Floating-point Convert to Unsigned integer, rounding toward Plus infinity (scalar)
    /// FCVTPU  <Wd>, <Hn>
    /// FCVTPU  <Xd>, <Hn>
    /// FCVTPU  <Wd>, <Sn>
    /// FCVTPU  <Xd>, <Sn>
    /// FCVTPU  <Wd>, <Dn>
    /// FCVTPU  <Xd>, <Dn>
    FcvtpuFloat {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTXN, FCVTXN2 -- A64
    /// Floating-point Convert to lower precision Narrow, rounding to odd (vector)
    /// FCVTXN  <Vb><d>, <Va><n>
    /// FCVTXN{2}  <Vd>.<Tb>, <Vn>.<Ta>
    FcvtxnAdvsimd {
        sz: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: FcvtxnAdvsimdSelector,
    },
    /// FCVTZS (vector, fixed-point) -- A64
    /// Floating-point Convert to Signed fixed-point, rounding toward Zero (vector)
    /// FCVTZS  <V><d>, <V><n>, #<fbits>
    /// FCVTZS  <Vd>.<T>, <Vn>.<T>, #<fbits>
    FcvtzsAdvsimdFix {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: FcvtzsAdvsimdFixSelector,
    },
    /// FCVTZS (vector, integer) -- A64
    /// Floating-point Convert to Signed integer, rounding toward Zero (vector)
    /// FCVTZS  <Hd>, <Hn>
    /// FCVTZS  <V><d>, <V><n>
    /// FCVTZS  <Vd>.<T>, <Vn>.<T>
    /// FCVTZS  <Vd>.<T>, <Vn>.<T>
    FcvtzsAdvsimdInt {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtzsAdvsimdIntSelector,
    },
    /// FCVTZS (scalar, fixed-point) -- A64
    /// Floating-point Convert to Signed fixed-point, rounding toward Zero (scalar)
    /// FCVTZS  <Wd>, <Hn>, #<fbits>
    /// FCVTZS  <Xd>, <Hn>, #<fbits>
    /// FCVTZS  <Wd>, <Sn>, #<fbits>
    /// FCVTZS  <Xd>, <Sn>, #<fbits>
    /// FCVTZS  <Wd>, <Dn>, #<fbits>
    /// FCVTZS  <Xd>, <Dn>, #<fbits>
    FcvtzsFloatFix {
        sf: i32,
        ftype: i32,
        scale: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTZS (scalar, integer) -- A64
    /// Floating-point Convert to Signed integer, rounding toward Zero (scalar)
    /// FCVTZS  <Wd>, <Hn>
    /// FCVTZS  <Xd>, <Hn>
    /// FCVTZS  <Wd>, <Sn>
    /// FCVTZS  <Xd>, <Sn>
    /// FCVTZS  <Wd>, <Dn>
    /// FCVTZS  <Xd>, <Dn>
    FcvtzsFloatInt {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTZU (vector, fixed-point) -- A64
    /// Floating-point Convert to Unsigned fixed-point, rounding toward Zero (vector)
    /// FCVTZU  <V><d>, <V><n>, #<fbits>
    /// FCVTZU  <Vd>.<T>, <Vn>.<T>, #<fbits>
    FcvtzuAdvsimdFix {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: FcvtzuAdvsimdFixSelector,
    },
    /// FCVTZU (vector, integer) -- A64
    /// Floating-point Convert to Unsigned integer, rounding toward Zero (vector)
    /// FCVTZU  <Hd>, <Hn>
    /// FCVTZU  <V><d>, <V><n>
    /// FCVTZU  <Vd>.<T>, <Vn>.<T>
    /// FCVTZU  <Vd>.<T>, <Vn>.<T>
    FcvtzuAdvsimdInt {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FcvtzuAdvsimdIntSelector,
    },
    /// FCVTZU (scalar, fixed-point) -- A64
    /// Floating-point Convert to Unsigned fixed-point, rounding toward Zero (scalar)
    /// FCVTZU  <Wd>, <Hn>, #<fbits>
    /// FCVTZU  <Xd>, <Hn>, #<fbits>
    /// FCVTZU  <Wd>, <Sn>, #<fbits>
    /// FCVTZU  <Xd>, <Sn>, #<fbits>
    /// FCVTZU  <Wd>, <Dn>, #<fbits>
    /// FCVTZU  <Xd>, <Dn>, #<fbits>
    FcvtzuFloatFix {
        sf: i32,
        ftype: i32,
        scale: i32,
        rn: Register,
        rd: Register,
    },
    /// FCVTZU (scalar, integer) -- A64
    /// Floating-point Convert to Unsigned integer, rounding toward Zero (scalar)
    /// FCVTZU  <Wd>, <Hn>
    /// FCVTZU  <Xd>, <Hn>
    /// FCVTZU  <Wd>, <Sn>
    /// FCVTZU  <Xd>, <Sn>
    /// FCVTZU  <Wd>, <Dn>
    /// FCVTZU  <Xd>, <Dn>
    FcvtzuFloatInt {
        sf: i32,
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FDIV (vector) -- A64
    /// Floating-point Divide (vector)
    /// FDIV  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FDIV  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FdivAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FdivAdvsimdSelector,
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
    /// FJCVTZS -- A64
    /// Floating-point Javascript Convert to Signed fixed-point, rounding toward Zero
    /// FJCVTZS  <Wd>, <Dn>
    Fjcvtzs { rn: Register, rd: Register },
    /// FMADD -- A64
    /// Floating-point fused Multiply-Add (scalar)
    /// FMADD  <Hd>, <Hn>, <Hm>, <Ha>
    /// FMADD  <Sd>, <Sn>, <Sm>, <Sa>
    /// FMADD  <Dd>, <Dn>, <Dm>, <Da>
    FmaddFloat {
        ftype: i32,
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// FMAX (vector) -- A64
    /// Floating-point Maximum (vector)
    /// FMAX  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMAX  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmaxAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmaxAdvsimdSelector,
    },
    /// FMAX (scalar) -- A64
    /// Floating-point Maximum (scalar)
    /// FMAX  <Hd>, <Hn>, <Hm>
    /// FMAX  <Sd>, <Sn>, <Sm>
    /// FMAX  <Dd>, <Dn>, <Dm>
    FmaxFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FMAXNM (vector) -- A64
    /// Floating-point Maximum Number (vector)
    /// FMAXNM  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMAXNM  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmaxnmAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmaxnmAdvsimdSelector,
    },
    /// FMAXNM (scalar) -- A64
    /// Floating-point Maximum Number (scalar)
    /// FMAXNM  <Hd>, <Hn>, <Hm>
    /// FMAXNM  <Sd>, <Sn>, <Sm>
    /// FMAXNM  <Dd>, <Dn>, <Dm>
    FmaxnmFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FMAXNMP (scalar) -- A64
    /// Floating-point Maximum Number of Pair of elements (scalar)
    /// FMAXNMP  <V><d>, <Vn>.<T>
    /// FMAXNMP  <V><d>, <Vn>.<T>
    FmaxnmpAdvsimdPair {
        sz: i32,
        rn: Register,
        rd: Register,
        class_selector: FmaxnmpAdvsimdPairSelector,
    },
    /// FMAXNMP (vector) -- A64
    /// Floating-point Maximum Number Pairwise (vector)
    /// FMAXNMP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMAXNMP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmaxnmpAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmaxnmpAdvsimdVecSelector,
    },
    /// FMAXNMV -- A64
    /// Floating-point Maximum Number across Vector
    /// FMAXNMV  <V><d>, <Vn>.<T>
    /// FMAXNMV  <V><d>, <Vn>.<T>
    FmaxnmvAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmaxnmvAdvsimdSelector,
    },
    /// FMAXP (scalar) -- A64
    /// Floating-point Maximum of Pair of elements (scalar)
    /// FMAXP  <V><d>, <Vn>.<T>
    /// FMAXP  <V><d>, <Vn>.<T>
    FmaxpAdvsimdPair {
        sz: i32,
        rn: Register,
        rd: Register,
        class_selector: FmaxpAdvsimdPairSelector,
    },
    /// FMAXP (vector) -- A64
    /// Floating-point Maximum Pairwise (vector)
    /// FMAXP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMAXP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmaxpAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmaxpAdvsimdVecSelector,
    },
    /// FMAXV -- A64
    /// Floating-point Maximum across Vector
    /// FMAXV  <V><d>, <Vn>.<T>
    /// FMAXV  <V><d>, <Vn>.<T>
    FmaxvAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmaxvAdvsimdSelector,
    },
    /// FMIN (vector) -- A64
    /// Floating-point minimum (vector)
    /// FMIN  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMIN  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FminAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FminAdvsimdSelector,
    },
    /// FMIN (scalar) -- A64
    /// Floating-point Minimum (scalar)
    /// FMIN  <Hd>, <Hn>, <Hm>
    /// FMIN  <Sd>, <Sn>, <Sm>
    /// FMIN  <Dd>, <Dn>, <Dm>
    FminFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FMINNM (vector) -- A64
    /// Floating-point Minimum Number (vector)
    /// FMINNM  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMINNM  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FminnmAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FminnmAdvsimdSelector,
    },
    /// FMINNM (scalar) -- A64
    /// Floating-point Minimum Number (scalar)
    /// FMINNM  <Hd>, <Hn>, <Hm>
    /// FMINNM  <Sd>, <Sn>, <Sm>
    /// FMINNM  <Dd>, <Dn>, <Dm>
    FminnmFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FMINNMP (scalar) -- A64
    /// Floating-point Minimum Number of Pair of elements (scalar)
    /// FMINNMP  <V><d>, <Vn>.<T>
    /// FMINNMP  <V><d>, <Vn>.<T>
    FminnmpAdvsimdPair {
        sz: i32,
        rn: Register,
        rd: Register,
        class_selector: FminnmpAdvsimdPairSelector,
    },
    /// FMINNMP (vector) -- A64
    /// Floating-point Minimum Number Pairwise (vector)
    /// FMINNMP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMINNMP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FminnmpAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FminnmpAdvsimdVecSelector,
    },
    /// FMINNMV -- A64
    /// Floating-point Minimum Number across Vector
    /// FMINNMV  <V><d>, <Vn>.<T>
    /// FMINNMV  <V><d>, <Vn>.<T>
    FminnmvAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FminnmvAdvsimdSelector,
    },
    /// FMINP (scalar) -- A64
    /// Floating-point Minimum of Pair of elements (scalar)
    /// FMINP  <V><d>, <Vn>.<T>
    /// FMINP  <V><d>, <Vn>.<T>
    FminpAdvsimdPair {
        sz: i32,
        rn: Register,
        rd: Register,
        class_selector: FminpAdvsimdPairSelector,
    },
    /// FMINP (vector) -- A64
    /// Floating-point Minimum Pairwise (vector)
    /// FMINP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMINP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FminpAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FminpAdvsimdVecSelector,
    },
    /// FMINV -- A64
    /// Floating-point Minimum across Vector
    /// FMINV  <V><d>, <Vn>.<T>
    /// FMINV  <V><d>, <Vn>.<T>
    FminvAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FminvAdvsimdSelector,
    },
    /// FMLA (by element) -- A64
    /// Floating-point fused Multiply-Add to accumulator (by element)
    /// FMLA  <Hd>, <Hn>, <Vm>.H[<index>]
    /// FMLA  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// FMLA  <Vd>.<T>, <Vn>.<T>, <Vm>.H[<index>]
    /// FMLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    FmlaAdvsimdElt {
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FmlaAdvsimdEltSelector,
    },
    /// FMLA (vector) -- A64
    /// Floating-point fused Multiply-Add to accumulator (vector)
    /// FMLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmlaAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmlaAdvsimdVecSelector,
    },
    /// FMLAL, FMLAL2 (by element) -- A64
    /// Floating-point fused Multiply-Add Long to accumulator (by element)
    /// FMLAL  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.H[<index>]
    /// FMLAL2  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.H[<index>]
    FmlalAdvsimdElt {
        q: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        class_selector: FmlalAdvsimdEltSelector,
    },
    /// FMLAL, FMLAL2 (vector) -- A64
    /// Floating-point fused Multiply-Add Long to accumulator (vector)
    /// FMLAL  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    /// FMLAL2  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    FmlalAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        class_selector: FmlalAdvsimdVecSelector,
    },
    /// FMLS (by element) -- A64
    /// Floating-point fused Multiply-Subtract from accumulator (by element)
    /// FMLS  <Hd>, <Hn>, <Vm>.H[<index>]
    /// FMLS  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// FMLS  <Vd>.<T>, <Vn>.<T>, <Vm>.H[<index>]
    /// FMLS  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    FmlsAdvsimdElt {
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FmlsAdvsimdEltSelector,
    },
    /// FMLS (vector) -- A64
    /// Floating-point fused Multiply-Subtract from accumulator (vector)
    /// FMLS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMLS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmlsAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmlsAdvsimdVecSelector,
    },
    /// FMLSL, FMLSL2 (by element) -- A64
    /// Floating-point fused Multiply-Subtract Long from accumulator (by element)
    /// FMLSL  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.H[<index>]
    /// FMLSL2  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.H[<index>]
    FmlslAdvsimdElt {
        q: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        class_selector: FmlslAdvsimdEltSelector,
    },
    /// FMLSL, FMLSL2 (vector) -- A64
    /// Floating-point fused Multiply-Subtract Long from accumulator (vector)
    /// FMLSL  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    /// FMLSL2  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    FmlslAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        class_selector: FmlslAdvsimdVecSelector,
    },
    /// FMOV (vector, immediate) -- A64
    /// Floating-point move immediate (vector)
    /// FMOV  <Vd>.<T>, #<imm>
    /// FMOV  <Vd>.<T>, #<imm>
    /// FMOV  <Vd>.2D, #<imm>
    FmovAdvsimd {
        q: i32,
        a: i32,
        b: i32,
        c: i32,
        d: i32,
        e: i32,
        f: i32,
        g: i32,
        h: i32,
        rd: Register,
        op: i32,
        class_selector: FmovAdvsimdSelector,
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
    /// FMOV (scalar, immediate) -- A64
    /// Floating-point move immediate (scalar)
    /// FMOV  <Hd>, #<imm>
    /// FMOV  <Sd>, #<imm>
    /// FMOV  <Dd>, #<imm>
    FmovFloatImm { ftype: i32, imm8: i32, rd: Register },
    /// FMSUB -- A64
    /// Floating-point Fused Multiply-Subtract (scalar)
    /// FMSUB  <Hd>, <Hn>, <Hm>, <Ha>
    /// FMSUB  <Sd>, <Sn>, <Sm>, <Sa>
    /// FMSUB  <Dd>, <Dn>, <Dm>, <Da>
    FmsubFloat {
        ftype: i32,
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// FMUL (by element) -- A64
    /// Floating-point Multiply (by element)
    /// FMUL  <Hd>, <Hn>, <Vm>.H[<index>]
    /// FMUL  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// FMUL  <Vd>.<T>, <Vn>.<T>, <Vm>.H[<index>]
    /// FMUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    FmulAdvsimdElt {
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FmulAdvsimdEltSelector,
    },
    /// FMUL (vector) -- A64
    /// Floating-point Multiply (vector)
    /// FMUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmulAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FmulAdvsimdVecSelector,
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
    /// FMULX (by element) -- A64
    /// Floating-point Multiply extended (by element)
    /// FMULX  <Hd>, <Hn>, <Vm>.H[<index>]
    /// FMULX  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// FMULX  <Vd>.<T>, <Vn>.<T>, <Vm>.H[<index>]
    /// FMULX  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    FmulxAdvsimdElt {
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FmulxAdvsimdEltSelector,
    },
    /// FMULX -- A64
    /// Floating-point Multiply extended
    /// FMULX  <Hd>, <Hn>, <Hm>
    /// FMULX  <V><d>, <V><n>, <V><m>
    /// FMULX  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FMULX  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FmulxAdvsimdVec {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FmulxAdvsimdVecSelector,
    },
    /// FNEG (vector) -- A64
    /// Floating-point Negate (vector)
    /// FNEG  <Vd>.<T>, <Vn>.<T>
    /// FNEG  <Vd>.<T>, <Vn>.<T>
    FnegAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FnegAdvsimdSelector,
    },
    /// FNEG (scalar) -- A64
    /// Floating-point Negate (scalar)
    /// FNEG  <Hd>, <Hn>
    /// FNEG  <Sd>, <Sn>
    /// FNEG  <Dd>, <Dn>
    FnegFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FNMADD -- A64
    /// Floating-point Negated fused Multiply-Add (scalar)
    /// FNMADD  <Hd>, <Hn>, <Hm>, <Ha>
    /// FNMADD  <Sd>, <Sn>, <Sm>, <Sa>
    /// FNMADD  <Dd>, <Dn>, <Dm>, <Da>
    FnmaddFloat {
        ftype: i32,
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// FNMSUB -- A64
    /// Floating-point Negated fused Multiply-Subtract (scalar)
    /// FNMSUB  <Hd>, <Hn>, <Hm>, <Ha>
    /// FNMSUB  <Sd>, <Sn>, <Sm>, <Sa>
    /// FNMSUB  <Dd>, <Dn>, <Dm>, <Da>
    FnmsubFloat {
        ftype: i32,
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// FNMUL (scalar) -- A64
    /// Floating-point Multiply-Negate (scalar)
    /// FNMUL  <Hd>, <Hn>, <Hm>
    /// FNMUL  <Sd>, <Sn>, <Sm>
    /// FNMUL  <Dd>, <Dn>, <Dm>
    FnmulFloat {
        ftype: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// FRECPE -- A64
    /// Floating-point Reciprocal Estimate
    /// FRECPE  <Hd>, <Hn>
    /// FRECPE  <V><d>, <V><n>
    /// FRECPE  <Vd>.<T>, <Vn>.<T>
    /// FRECPE  <Vd>.<T>, <Vn>.<T>
    FrecpeAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FrecpeAdvsimdSelector,
    },
    /// FRECPS -- A64
    /// Floating-point Reciprocal Step
    /// FRECPS  <Hd>, <Hn>, <Hm>
    /// FRECPS  <V><d>, <V><n>, <V><m>
    /// FRECPS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FRECPS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FrecpsAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FrecpsAdvsimdSelector,
    },
    /// FRECPX -- A64
    /// Floating-point Reciprocal exponent (scalar)
    /// FRECPX  <Hd>, <Hn>
    /// FRECPX  <V><d>, <V><n>
    FrecpxAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrecpxAdvsimdSelector,
    },
    /// FRINT32X (vector) -- A64
    /// Floating-point Round to 32-bit Integer, using current rounding mode (vector)
    /// FRINT32X  <Vd>.<T>, <Vn>.<T>
    Frint32xAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINT32X (scalar) -- A64
    /// Floating-point Round to 32-bit Integer, using current rounding mode (scalar)
    /// FRINT32X  <Sd>, <Sn>
    /// FRINT32X  <Dd>, <Dn>
    Frint32xFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINT32Z (vector) -- A64
    /// Floating-point Round to 32-bit Integer toward Zero (vector)
    /// FRINT32Z  <Vd>.<T>, <Vn>.<T>
    Frint32zAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINT32Z (scalar) -- A64
    /// Floating-point Round to 32-bit Integer toward Zero (scalar)
    /// FRINT32Z  <Sd>, <Sn>
    /// FRINT32Z  <Dd>, <Dn>
    Frint32zFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINT64X (vector) -- A64
    /// Floating-point Round to 64-bit Integer, using current rounding mode (vector)
    /// FRINT64X  <Vd>.<T>, <Vn>.<T>
    Frint64xAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINT64X (scalar) -- A64
    /// Floating-point Round to 64-bit Integer, using current rounding mode (scalar)
    /// FRINT64X  <Sd>, <Sn>
    /// FRINT64X  <Dd>, <Dn>
    Frint64xFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINT64Z (vector) -- A64
    /// Floating-point Round to 64-bit Integer toward Zero (vector)
    /// FRINT64Z  <Vd>.<T>, <Vn>.<T>
    Frint64zAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINT64Z (scalar) -- A64
    /// Floating-point Round to 64-bit Integer toward Zero (scalar)
    /// FRINT64Z  <Sd>, <Sn>
    /// FRINT64Z  <Dd>, <Dn>
    Frint64zFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINTA (vector) -- A64
    /// Floating-point Round to Integral, to nearest with ties to Away (vector)
    /// FRINTA  <Vd>.<T>, <Vn>.<T>
    /// FRINTA  <Vd>.<T>, <Vn>.<T>
    FrintaAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrintaAdvsimdSelector,
    },
    /// FRINTA (scalar) -- A64
    /// Floating-point Round to Integral, to nearest with ties to Away (scalar)
    /// FRINTA  <Hd>, <Hn>
    /// FRINTA  <Sd>, <Sn>
    /// FRINTA  <Dd>, <Dn>
    FrintaFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINTI (vector) -- A64
    /// Floating-point Round to Integral, using current rounding mode (vector)
    /// FRINTI  <Vd>.<T>, <Vn>.<T>
    /// FRINTI  <Vd>.<T>, <Vn>.<T>
    FrintiAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrintiAdvsimdSelector,
    },
    /// FRINTI (scalar) -- A64
    /// Floating-point Round to Integral, using current rounding mode (scalar)
    /// FRINTI  <Hd>, <Hn>
    /// FRINTI  <Sd>, <Sn>
    /// FRINTI  <Dd>, <Dn>
    FrintiFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINTM (vector) -- A64
    /// Floating-point Round to Integral, toward Minus infinity (vector)
    /// FRINTM  <Vd>.<T>, <Vn>.<T>
    /// FRINTM  <Vd>.<T>, <Vn>.<T>
    FrintmAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrintmAdvsimdSelector,
    },
    /// FRINTM (scalar) -- A64
    /// Floating-point Round to Integral, toward Minus infinity (scalar)
    /// FRINTM  <Hd>, <Hn>
    /// FRINTM  <Sd>, <Sn>
    /// FRINTM  <Dd>, <Dn>
    FrintmFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINTN (vector) -- A64
    /// Floating-point Round to Integral, to nearest with ties to even (vector)
    /// FRINTN  <Vd>.<T>, <Vn>.<T>
    /// FRINTN  <Vd>.<T>, <Vn>.<T>
    FrintnAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrintnAdvsimdSelector,
    },
    /// FRINTN (scalar) -- A64
    /// Floating-point Round to Integral, to nearest with ties to even (scalar)
    /// FRINTN  <Hd>, <Hn>
    /// FRINTN  <Sd>, <Sn>
    /// FRINTN  <Dd>, <Dn>
    FrintnFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINTP (vector) -- A64
    /// Floating-point Round to Integral, toward Plus infinity (vector)
    /// FRINTP  <Vd>.<T>, <Vn>.<T>
    /// FRINTP  <Vd>.<T>, <Vn>.<T>
    FrintpAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrintpAdvsimdSelector,
    },
    /// FRINTP (scalar) -- A64
    /// Floating-point Round to Integral, toward Plus infinity (scalar)
    /// FRINTP  <Hd>, <Hn>
    /// FRINTP  <Sd>, <Sn>
    /// FRINTP  <Dd>, <Dn>
    FrintpFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINTX (vector) -- A64
    /// Floating-point Round to Integral exact, using current rounding mode (vector)
    /// FRINTX  <Vd>.<T>, <Vn>.<T>
    /// FRINTX  <Vd>.<T>, <Vn>.<T>
    FrintxAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrintxAdvsimdSelector,
    },
    /// FRINTX (scalar) -- A64
    /// Floating-point Round to Integral exact, using current rounding mode (scalar)
    /// FRINTX  <Hd>, <Hn>
    /// FRINTX  <Sd>, <Sn>
    /// FRINTX  <Dd>, <Dn>
    FrintxFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRINTZ (vector) -- A64
    /// Floating-point Round to Integral, toward Zero (vector)
    /// FRINTZ  <Vd>.<T>, <Vn>.<T>
    /// FRINTZ  <Vd>.<T>, <Vn>.<T>
    FrintzAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FrintzAdvsimdSelector,
    },
    /// FRINTZ (scalar) -- A64
    /// Floating-point Round to Integral, toward Zero (scalar)
    /// FRINTZ  <Hd>, <Hn>
    /// FRINTZ  <Sd>, <Sn>
    /// FRINTZ  <Dd>, <Dn>
    FrintzFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FRSQRTE -- A64
    /// Floating-point Reciprocal Square Root Estimate
    /// FRSQRTE  <Hd>, <Hn>
    /// FRSQRTE  <V><d>, <V><n>
    /// FRSQRTE  <Vd>.<T>, <Vn>.<T>
    /// FRSQRTE  <Vd>.<T>, <Vn>.<T>
    FrsqrteAdvsimd {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FrsqrteAdvsimdSelector,
    },
    /// FRSQRTS -- A64
    /// Floating-point Reciprocal Square Root Step
    /// FRSQRTS  <Hd>, <Hn>, <Hm>
    /// FRSQRTS  <V><d>, <V><n>, <V><m>
    /// FRSQRTS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FRSQRTS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FrsqrtsAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: FrsqrtsAdvsimdSelector,
    },
    /// FSQRT (vector) -- A64
    /// Floating-point Square Root (vector)
    /// FSQRT  <Vd>.<T>, <Vn>.<T>
    /// FSQRT  <Vd>.<T>, <Vn>.<T>
    FsqrtAdvsimd {
        q: i32,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FsqrtAdvsimdSelector,
    },
    /// FSQRT (scalar) -- A64
    /// Floating-point Square Root (scalar)
    /// FSQRT  <Hd>, <Hn>
    /// FSQRT  <Sd>, <Sn>
    /// FSQRT  <Dd>, <Dn>
    FsqrtFloat {
        ftype: i32,
        rn: Register,
        rd: Register,
    },
    /// FSUB (vector) -- A64
    /// Floating-point Subtract (vector)
    /// FSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    /// FSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    FsubAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        sz: i32,
        class_selector: FsubAdvsimdSelector,
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
    /// INS (element) -- A64
    /// Insert vector element from another vector element
    /// INS  <Vd>.<Ts>[<index1>], <Vn>.<Ts>[<index2>]
    InsAdvsimdElt {
        imm5: i32,
        imm4: i32,
        rn: Register,
        rd: Register,
    },
    /// INS (general) -- A64
    /// Insert vector element from general-purpose register
    /// INS  <Vd>.<Ts>[<index>], <R><n>
    InsAdvsimdGen {
        imm5: i32,
        rn: Register,
        rd: Register,
    },
    /// IRG -- A64
    /// Insert Random Tag
    /// IRG  <Xd|SP>, <Xn|SP>{, <Xm>}
    Irg { xm: i32, xn: i32, xd: i32 },
    /// ISB -- A64
    /// Instruction Synchronization Barrier
    /// ISB  {<option>|#<imm>}
    Isb { crm: i32 },
    /// LD1 (multiple structures) -- A64
    /// Load multiple single-element structures to one, two, three, or four registers
    /// LD1  { <Vt>.<T> }, [<Xn|SP>]
    /// LD1  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
    /// LD1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>]
    /// LD1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
    /// LD1  { <Vt>.<T> }, [<Xn|SP>], <imm>
    /// LD1  { <Vt>.<T> }, [<Xn|SP>], <Xm>
    /// LD1  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>
    /// LD1  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <Xm>
    /// LD1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <imm>
    /// LD1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <Xm>
    /// LD1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
    /// LD1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <Xm>
    Ld1AdvsimdMult {
        q: i32,
        opcode: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld1AdvsimdMultSelector,
    },
    /// LD1 (single structure) -- A64
    /// Load one single-element structure to one lane of one register
    /// LD1  { <Vt>.B }[<index>], [<Xn|SP>]
    /// LD1  { <Vt>.H }[<index>], [<Xn|SP>]
    /// LD1  { <Vt>.S }[<index>], [<Xn|SP>]
    /// LD1  { <Vt>.D }[<index>], [<Xn|SP>]
    /// LD1  { <Vt>.B }[<index>], [<Xn|SP>], #1
    /// LD1  { <Vt>.B }[<index>], [<Xn|SP>], <Xm>
    /// LD1  { <Vt>.H }[<index>], [<Xn|SP>], #2
    /// LD1  { <Vt>.H }[<index>], [<Xn|SP>], <Xm>
    /// LD1  { <Vt>.S }[<index>], [<Xn|SP>], #4
    /// LD1  { <Vt>.S }[<index>], [<Xn|SP>], <Xm>
    /// LD1  { <Vt>.D }[<index>], [<Xn|SP>], #8
    /// LD1  { <Vt>.D }[<index>], [<Xn|SP>], <Xm>
    Ld1AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld1AdvsimdSnglSelector,
    },
    /// LD1R -- A64
    /// Load one single-element structure and Replicate to all lanes (of one register)
    /// LD1R  { <Vt>.<T> }, [<Xn|SP>]
    /// LD1R  { <Vt>.<T> }, [<Xn|SP>], <imm>
    /// LD1R  { <Vt>.<T> }, [<Xn|SP>], <Xm>
    Ld1rAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld1rAdvsimdSelector,
    },
    /// LD2 (multiple structures) -- A64
    /// Load multiple 2-element structures to two registers
    /// LD2  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
    /// LD2  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>
    /// LD2  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <Xm>
    Ld2AdvsimdMult {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld2AdvsimdMultSelector,
    },
    /// LD2 (single structure) -- A64
    /// Load single 2-element structure to one lane of two registers
    /// LD2  { <Vt>.B, <Vt2>.B }[<index>], [<Xn|SP>]
    /// LD2  { <Vt>.H, <Vt2>.H }[<index>], [<Xn|SP>]
    /// LD2  { <Vt>.S, <Vt2>.S }[<index>], [<Xn|SP>]
    /// LD2  { <Vt>.D, <Vt2>.D }[<index>], [<Xn|SP>]
    /// LD2  { <Vt>.B, <Vt2>.B }[<index>], [<Xn|SP>], #2
    /// LD2  { <Vt>.B, <Vt2>.B }[<index>], [<Xn|SP>], <Xm>
    /// LD2  { <Vt>.H, <Vt2>.H }[<index>], [<Xn|SP>], #4
    /// LD2  { <Vt>.H, <Vt2>.H }[<index>], [<Xn|SP>], <Xm>
    /// LD2  { <Vt>.S, <Vt2>.S }[<index>], [<Xn|SP>], #8
    /// LD2  { <Vt>.S, <Vt2>.S }[<index>], [<Xn|SP>], <Xm>
    /// LD2  { <Vt>.D, <Vt2>.D }[<index>], [<Xn|SP>], #16
    /// LD2  { <Vt>.D, <Vt2>.D }[<index>], [<Xn|SP>], <Xm>
    Ld2AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld2AdvsimdSnglSelector,
    },
    /// LD2R -- A64
    /// Load single 2-element structure and Replicate to all lanes of two registers
    /// LD2R  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
    /// LD2R  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>
    /// LD2R  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <Xm>
    Ld2rAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld2rAdvsimdSelector,
    },
    /// LD3 (multiple structures) -- A64
    /// Load multiple 3-element structures to three registers
    /// LD3  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>]
    /// LD3  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <imm>
    /// LD3  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <Xm>
    Ld3AdvsimdMult {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld3AdvsimdMultSelector,
    },
    /// LD3 (single structure) -- A64
    /// Load single 3-element structure to one lane of three registers
    /// LD3  { <Vt>.B, <Vt2>.B, <Vt3>.B }[<index>], [<Xn|SP>]
    /// LD3  { <Vt>.H, <Vt2>.H, <Vt3>.H }[<index>], [<Xn|SP>]
    /// LD3  { <Vt>.S, <Vt2>.S, <Vt3>.S }[<index>], [<Xn|SP>]
    /// LD3  { <Vt>.D, <Vt2>.D, <Vt3>.D }[<index>], [<Xn|SP>]
    /// LD3  { <Vt>.B, <Vt2>.B, <Vt3>.B }[<index>], [<Xn|SP>], #3
    /// LD3  { <Vt>.B, <Vt2>.B, <Vt3>.B }[<index>], [<Xn|SP>], <Xm>
    /// LD3  { <Vt>.H, <Vt2>.H, <Vt3>.H }[<index>], [<Xn|SP>], #6
    /// LD3  { <Vt>.H, <Vt2>.H, <Vt3>.H }[<index>], [<Xn|SP>], <Xm>
    /// LD3  { <Vt>.S, <Vt2>.S, <Vt3>.S }[<index>], [<Xn|SP>], #12
    /// LD3  { <Vt>.S, <Vt2>.S, <Vt3>.S }[<index>], [<Xn|SP>], <Xm>
    /// LD3  { <Vt>.D, <Vt2>.D, <Vt3>.D }[<index>], [<Xn|SP>], #24
    /// LD3  { <Vt>.D, <Vt2>.D, <Vt3>.D }[<index>], [<Xn|SP>], <Xm>
    Ld3AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld3AdvsimdSnglSelector,
    },
    /// LD3R -- A64
    /// Load single 3-element structure and Replicate to all lanes of three registers
    /// LD3R  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>]
    /// LD3R  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <imm>
    /// LD3R  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <Xm>
    Ld3rAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld3rAdvsimdSelector,
    },
    /// LD4 (multiple structures) -- A64
    /// Load multiple 4-element structures to four registers
    /// LD4  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
    /// LD4  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
    /// LD4  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <Xm>
    Ld4AdvsimdMult {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld4AdvsimdMultSelector,
    },
    /// LD4 (single structure) -- A64
    /// Load single 4-element structure to one lane of four registers
    /// LD4  { <Vt>.B, <Vt2>.B, <Vt3>.B, <Vt4>.B }[<index>], [<Xn|SP>]
    /// LD4  { <Vt>.H, <Vt2>.H, <Vt3>.H, <Vt4>.H }[<index>], [<Xn|SP>]
    /// LD4  { <Vt>.S, <Vt2>.S, <Vt3>.S, <Vt4>.S }[<index>], [<Xn|SP>]
    /// LD4  { <Vt>.D, <Vt2>.D, <Vt3>.D, <Vt4>.D }[<index>], [<Xn|SP>]
    /// LD4  { <Vt>.B, <Vt2>.B, <Vt3>.B, <Vt4>.B }[<index>], [<Xn|SP>], #4
    /// LD4  { <Vt>.B, <Vt2>.B, <Vt3>.B, <Vt4>.B }[<index>], [<Xn|SP>], <Xm>
    /// LD4  { <Vt>.H, <Vt2>.H, <Vt3>.H, <Vt4>.H }[<index>], [<Xn|SP>], #8
    /// LD4  { <Vt>.H, <Vt2>.H, <Vt3>.H, <Vt4>.H }[<index>], [<Xn|SP>], <Xm>
    /// LD4  { <Vt>.S, <Vt2>.S, <Vt3>.S, <Vt4>.S }[<index>], [<Xn|SP>], #16
    /// LD4  { <Vt>.S, <Vt2>.S, <Vt3>.S, <Vt4>.S }[<index>], [<Xn|SP>], <Xm>
    /// LD4  { <Vt>.D, <Vt2>.D, <Vt3>.D, <Vt4>.D }[<index>], [<Xn|SP>], #32
    /// LD4  { <Vt>.D, <Vt2>.D, <Vt3>.D, <Vt4>.D }[<index>], [<Xn|SP>], <Xm>
    Ld4AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld4AdvsimdSnglSelector,
    },
    /// LD4R -- A64
    /// Load single 4-element structure and Replicate to all lanes of four registers
    /// LD4R  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
    /// LD4R  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
    /// LD4R  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <Xm>
    Ld4rAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: Ld4rAdvsimdSelector,
    },
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
    /// LDNP (SIMD&FP) -- A64
    /// Load Pair of SIMD&FP registers, with Non-temporal hint
    /// LDNP  <St1>, <St2>, [<Xn|SP>{, #<imm>}]
    /// LDNP  <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]
    /// LDNP  <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]
    LdnpFpsimd {
        opc: i32,
        imm7: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
    },
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
    /// LDP (SIMD&FP) -- A64
    /// Load Pair of SIMD&FP registers
    /// LDP  <St1>, <St2>, [<Xn|SP>], #<imm>
    /// LDP  <Dt1>, <Dt2>, [<Xn|SP>], #<imm>
    /// LDP  <Qt1>, <Qt2>, [<Xn|SP>], #<imm>
    /// LDP  <St1>, <St2>, [<Xn|SP>, #<imm>]!
    /// LDP  <Dt1>, <Dt2>, [<Xn|SP>, #<imm>]!
    /// LDP  <Qt1>, <Qt2>, [<Xn|SP>, #<imm>]!
    /// LDP  <St1>, <St2>, [<Xn|SP>{, #<imm>}]
    /// LDP  <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]
    /// LDP  <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]
    LdpFpsimd {
        opc: i32,
        imm7: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
        class_selector: LdpFpsimdSelector,
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
    /// LDR (immediate, SIMD&FP) -- A64
    /// Load SIMD&FP Register (immediate offset)
    /// LDR  <Bt>, [<Xn|SP>], #<simm>
    /// LDR  <Ht>, [<Xn|SP>], #<simm>
    /// LDR  <St>, [<Xn|SP>], #<simm>
    /// LDR  <Dt>, [<Xn|SP>], #<simm>
    /// LDR  <Qt>, [<Xn|SP>], #<simm>
    /// LDR  <Bt>, [<Xn|SP>, #<simm>]!
    /// LDR  <Ht>, [<Xn|SP>, #<simm>]!
    /// LDR  <St>, [<Xn|SP>, #<simm>]!
    /// LDR  <Dt>, [<Xn|SP>, #<simm>]!
    /// LDR  <Qt>, [<Xn|SP>, #<simm>]!
    /// LDR  <Bt>, [<Xn|SP>{, #<pimm>}]
    /// LDR  <Ht>, [<Xn|SP>{, #<pimm>}]
    /// LDR  <St>, [<Xn|SP>{, #<pimm>}]
    /// LDR  <Dt>, [<Xn|SP>{, #<pimm>}]
    /// LDR  <Qt>, [<Xn|SP>{, #<pimm>}]
    LdrImmFpsimd {
        size: i32,
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: LdrImmFpsimdSelector,
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
    /// LDR (literal, SIMD&FP) -- A64
    /// Load SIMD&FP Register (PC-relative literal)
    /// LDR  <St>, <label>
    /// LDR  <Dt>, <label>
    /// LDR  <Qt>, <label>
    LdrLitFpsimd { opc: i32, imm19: i32, rt: Register },
    /// LDR (literal) -- A64
    /// Load Register (literal)
    /// LDR  <Wt>, <label>
    /// LDR  <Xt>, <label>
    LdrLitGen { opc: i32, imm19: i32, rt: Register },
    /// LDR (register, SIMD&FP) -- A64
    /// Load SIMD&FP Register (register offset)
    /// LDR  <Bt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
    /// LDR  <Bt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
    /// LDR  <Ht>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// LDR  <St>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// LDR  <Dt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// LDR  <Qt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    LdrRegFpsimd {
        size: i32,
        opc: i32,
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
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
    /// LDUR (SIMD&FP) -- A64
    /// Load SIMD&FP Register (unscaled offset)
    /// LDUR  <Bt>, [<Xn|SP>{, #<simm>}]
    /// LDUR  <Ht>, [<Xn|SP>{, #<simm>}]
    /// LDUR  <St>, [<Xn|SP>{, #<simm>}]
    /// LDUR  <Dt>, [<Xn|SP>{, #<simm>}]
    /// LDUR  <Qt>, [<Xn|SP>{, #<simm>}]
    LdurFpsimd {
        size: i32,
        opc: i32,
        imm9: i32,
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
    /// MLA (by element) -- A64
    /// Multiply-Add to accumulator (vector, by element)
    /// MLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    MlaAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// MLA (vector) -- A64
    /// Multiply-Add to accumulator (vector)
    /// MLA  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    MlaAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// MLS (by element) -- A64
    /// Multiply-Subtract from accumulator (vector, by element)
    /// MLS  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    MlsAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// MLS (vector) -- A64
    /// Multiply-Subtract from accumulator (vector)
    /// MLS  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    MlsAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
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
    /// MOV (scalar) -- A64
    /// Move vector element to scalar
    /// MOV  <V><d>, <Vn>.<T>[<index>]
    /// DUP  <V><d>, <Vn>.<T>[<index>]
    MovDupAdvsimdElt {
        imm5: i32,
        rn: Register,
        rd: Register,
    },
    /// MOV (element) -- A64
    /// Move vector element to another vector element
    /// MOV  <Vd>.<Ts>[<index1>], <Vn>.<Ts>[<index2>]
    /// INS  <Vd>.<Ts>[<index1>], <Vn>.<Ts>[<index2>]
    MovInsAdvsimdElt {
        imm5: i32,
        imm4: i32,
        rn: Register,
        rd: Register,
    },
    /// MOV (from general) -- A64
    /// Move general-purpose register to a vector element
    /// MOV  <Vd>.<Ts>[<index>], <R><n>
    /// INS  <Vd>.<Ts>[<index>], <R><n>
    MovInsAdvsimdGen {
        imm5: i32,
        rn: Register,
        rd: Register,
    },
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
    /// MOV (vector) -- A64
    /// Move vector
    /// MOV  <Vd>.<T>, <Vn>.<T>
    /// ORR  <Vd>.<T>, <Vn>.<T>, <Vn>.<T>
    MovOrrAdvsimdReg {
        q: i32,
        rm: Register,
        rn: Register,
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
    /// MOV (to general) -- A64
    /// Move vector element to general-purpose register
    /// MOV  <Wd>, <Vn>.S[<index>]
    /// UMOV <Wd>, <Vn>.S[<index>]
    /// MOV  <Xd>, <Vn>.D[<index>]
    /// UMOV <Xd>, <Vn>.D[<index>]
    MovUmovAdvsimd {
        q: i32,
        imm5: i32,
        rn: Register,
        rd: Register,
    },
    /// MOVI -- A64
    /// Move Immediate (vector)
    /// MOVI  <Vd>.<T>, #<imm8>{, LSL #0}
    /// MOVI  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    /// MOVI  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    /// MOVI  <Vd>.<T>, #<imm8>, MSL #<amount>
    /// MOVI  <Dd>, #<imm>
    /// MOVI  <Vd>.2D, #<imm>
    MoviAdvsimd {
        q: i32,
        op: i32,
        a: i32,
        b: i32,
        c: i32,
        cmode: i32,
        d: i32,
        e: i32,
        f: i32,
        g: i32,
        h: i32,
        rd: Register,
    },
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
    /// MUL (by element) -- A64
    /// Multiply (vector, by element)
    /// MUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    MulAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// MUL (vector) -- A64
    /// Multiply (vector)
    /// MUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    MulAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
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
    /// Bitwise NOT (vector)
    /// MVN  <Vd>.<T>, <Vn>.<T>
    /// NOT  <Vd>.<T>, <Vn>.<T>
    MvnNotAdvsimd { q: i32, rn: Register, rd: Register },
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
    /// MVNI -- A64
    /// Move inverted Immediate (vector)
    /// MVNI  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    /// MVNI  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    /// MVNI  <Vd>.<T>, #<imm8>, MSL #<amount>
    MvniAdvsimd {
        q: i32,
        a: i32,
        b: i32,
        c: i32,
        cmode: i32,
        d: i32,
        e: i32,
        f: i32,
        g: i32,
        h: i32,
        rd: Register,
    },
    /// NEG (vector) -- A64
    /// Negate (vector)
    /// NEG  <V><d>, <V><n>
    /// NEG  <Vd>.<T>, <Vn>.<T>
    NegAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: NegAdvsimdSelector,
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
    /// NOT -- A64
    /// Bitwise NOT (vector)
    /// NOT  <Vd>.<T>, <Vn>.<T>
    NotAdvsimd { q: i32, rn: Register, rd: Register },
    /// ORN (vector) -- A64
    /// Bitwise inclusive OR NOT (vector)
    /// ORN  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    OrnAdvsimd {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
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
    /// ORR (vector, immediate) -- A64
    /// Bitwise inclusive OR (vector, immediate)
    /// ORR  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    /// ORR  <Vd>.<T>, #<imm8>{, LSL #<amount>}
    OrrAdvsimdImm {
        q: i32,
        a: i32,
        b: i32,
        c: i32,
        cmode: i32,
        d: i32,
        e: i32,
        f: i32,
        g: i32,
        h: i32,
        rd: Register,
    },
    /// ORR (vector, register) -- A64
    /// Bitwise inclusive OR (vector, register)
    /// ORR  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    OrrAdvsimdReg {
        q: i32,
        rm: Register,
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
    /// PMUL -- A64
    /// Polynomial Multiply
    /// PMUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    PmulAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// PMULL, PMULL2 -- A64
    /// Polynomial Multiply Long
    /// PMULL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    PmullAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
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
    /// RADDHN, RADDHN2 -- A64
    /// Rounding Add returning High Narrow
    /// RADDHN{2}  <Vd>.<Tb>, <Vn>.<Ta>, <Vm>.<Ta>
    RaddhnAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// RAX1 -- A64
    /// Rotate and Exclusive OR
    /// RAX1  <Vd>.2D, <Vn>.2D, <Vm>.2D
    Rax1Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// RBIT (vector) -- A64
    /// Reverse Bit order (vector)
    /// RBIT  <Vd>.<T>, <Vn>.<T>
    RbitAdvsimd { q: i32, rn: Register, rd: Register },
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
    /// REV16 (vector) -- A64
    /// Reverse elements in 16-bit halfwords (vector)
    /// REV16  <Vd>.<T>, <Vn>.<T>
    Rev16Advsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// REV16 -- A64
    /// Reverse bytes in 16-bit halfwords
    /// REV16  <Wd>, <Wn>
    /// REV16  <Xd>, <Xn>
    Rev16Int { sf: i32, rn: Register, rd: Register },
    /// REV32 (vector) -- A64
    /// Reverse elements in 32-bit words (vector)
    /// REV32  <Vd>.<T>, <Vn>.<T>
    Rev32Advsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// REV32 -- A64
    /// Reverse bytes in 32-bit words
    /// REV32  <Xd>, <Xn>
    Rev32Int { rn: Register, rd: Register },
    /// REV64 -- A64
    /// Reverse elements in 64-bit doublewords (vector)
    /// REV64  <Vd>.<T>, <Vn>.<T>
    Rev64Advsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
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
    /// RSHRN, RSHRN2 -- A64
    /// Rounding Shift Right Narrow (immediate)
    /// RSHRN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    RshrnAdvsimd {
        q: i32,
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
    },
    /// RSUBHN, RSUBHN2 -- A64
    /// Rounding Subtract returning High Narrow
    /// RSUBHN{2}  <Vd>.<Tb>, <Vn>.<Ta>, <Vm>.<Ta>
    RsubhnAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SABA -- A64
    /// Signed Absolute difference and Accumulate
    /// SABA  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SabaAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SABAL, SABAL2 -- A64
    /// Signed Absolute difference and Accumulate Long
    /// SABAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SabalAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SABD -- A64
    /// Signed Absolute Difference
    /// SABD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SabdAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SABDL, SABDL2 -- A64
    /// Signed Absolute Difference Long
    /// SABDL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SabdlAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SADALP -- A64
    /// Signed Add and Accumulate Long Pairwise
    /// SADALP  <Vd>.<Ta>, <Vn>.<Tb>
    SadalpAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// SADDL, SADDL2 -- A64
    /// Signed Add Long (vector)
    /// SADDL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SaddlAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SADDLP -- A64
    /// Signed Add Long Pairwise
    /// SADDLP  <Vd>.<Ta>, <Vn>.<Tb>
    SaddlpAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// SADDLV -- A64
    /// Signed Add Long across Vector
    /// SADDLV  <V><d>, <Vn>.<T>
    SaddlvAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// SADDW, SADDW2 -- A64
    /// Signed Add Wide
    /// SADDW{2}  <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
    SaddwAdvsimd {
        q: i32,
        size: i32,
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
    /// SCVTF (vector, fixed-point) -- A64
    /// Signed fixed-point Convert to Floating-point (vector)
    /// SCVTF  <V><d>, <V><n>, #<fbits>
    /// SCVTF  <Vd>.<T>, <Vn>.<T>, #<fbits>
    ScvtfAdvsimdFix {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: ScvtfAdvsimdFixSelector,
    },
    /// SCVTF (vector, integer) -- A64
    /// Signed integer Convert to Floating-point (vector)
    /// SCVTF  <Hd>, <Hn>
    /// SCVTF  <V><d>, <V><n>
    /// SCVTF  <Vd>.<T>, <Vn>.<T>
    /// SCVTF  <Vd>.<T>, <Vn>.<T>
    ScvtfAdvsimdInt {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: ScvtfAdvsimdIntSelector,
    },
    /// SCVTF (scalar, fixed-point) -- A64
    /// Signed fixed-point Convert to Floating-point (scalar)
    /// SCVTF  <Hd>, <Wn>, #<fbits>
    /// SCVTF  <Sd>, <Wn>, #<fbits>
    /// SCVTF  <Dd>, <Wn>, #<fbits>
    /// SCVTF  <Hd>, <Xn>, #<fbits>
    /// SCVTF  <Sd>, <Xn>, #<fbits>
    /// SCVTF  <Dd>, <Xn>, #<fbits>
    ScvtfFloatFix {
        sf: i32,
        ftype: i32,
        scale: i32,
        rn: Register,
        rd: Register,
    },
    /// SCVTF (scalar, integer) -- A64
    /// Signed integer Convert to Floating-point (scalar)
    /// SCVTF  <Hd>, <Wn>
    /// SCVTF  <Sd>, <Wn>
    /// SCVTF  <Dd>, <Wn>
    /// SCVTF  <Hd>, <Xn>
    /// SCVTF  <Sd>, <Xn>
    /// SCVTF  <Dd>, <Xn>
    ScvtfFloatInt {
        sf: i32,
        ftype: i32,
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
    /// SDOT (by element) -- A64
    /// Dot Product signed arithmetic (vector, by element)
    /// SDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.4B[<index>]
    SdotAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// SDOT (vector) -- A64
    /// Dot Product signed arithmetic (vector)
    /// SDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SdotAdvsimdVec {
        q: i32,
        size: i32,
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
    /// SHA1C -- A64
    /// SHA1 hash update (choose)
    /// SHA1C  <Qd>, <Sn>, <Vm>.4S
    Sha1cAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA1H -- A64
    /// SHA1 fixed rotate
    /// SHA1H  <Sd>, <Sn>
    Sha1hAdvsimd { rn: Register, rd: Register },
    /// SHA1M -- A64
    /// SHA1 hash update (majority)
    /// SHA1M  <Qd>, <Sn>, <Vm>.4S
    Sha1mAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA1P -- A64
    /// SHA1 hash update (parity)
    /// SHA1P  <Qd>, <Sn>, <Vm>.4S
    Sha1pAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA1SU0 -- A64
    /// SHA1 schedule update 0
    /// SHA1SU0  <Vd>.4S, <Vn>.4S, <Vm>.4S
    Sha1su0Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA1SU1 -- A64
    /// SHA1 schedule update 1
    /// SHA1SU1  <Vd>.4S, <Vn>.4S
    Sha1su1Advsimd { rn: Register, rd: Register },
    /// SHA256H2 -- A64
    /// SHA256 hash update (part 2)
    /// SHA256H2  <Qd>, <Qn>, <Vm>.4S
    Sha256h2Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA256H -- A64
    /// SHA256 hash update (part 1)
    /// SHA256H  <Qd>, <Qn>, <Vm>.4S
    Sha256hAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA256SU0 -- A64
    /// SHA256 schedule update 0
    /// SHA256SU0  <Vd>.4S, <Vn>.4S
    Sha256su0Advsimd { rn: Register, rd: Register },
    /// SHA256SU1 -- A64
    /// SHA256 schedule update 1
    /// SHA256SU1  <Vd>.4S, <Vn>.4S, <Vm>.4S
    Sha256su1Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA512H2 -- A64
    /// SHA512 Hash update part 2
    /// SHA512H2  <Qd>, <Qn>, <Vm>.2D
    Sha512h2Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA512H -- A64
    /// SHA512 Hash update part 1
    /// SHA512H  <Qd>, <Qn>, <Vm>.2D
    Sha512hAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHA512SU0 -- A64
    /// SHA512 Schedule Update 0
    /// SHA512SU0  <Vd>.2D, <Vn>.2D
    Sha512su0Advsimd { rn: Register, rd: Register },
    /// SHA512SU1 -- A64
    /// SHA512 Schedule Update 1
    /// SHA512SU1  <Vd>.2D, <Vn>.2D, <Vm>.2D
    Sha512su1Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHADD -- A64
    /// Signed Halving Add
    /// SHADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    ShaddAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SHL -- A64
    /// Shift Left (immediate)
    /// SHL  <V><d>, <V><n>, #<shift>
    /// SHL  <Vd>.<T>, <Vn>.<T>, #<shift>
    ShlAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: ShlAdvsimdSelector,
    },
    /// SHLL, SHLL2 -- A64
    /// Shift Left Long (by element size)
    /// SHLL{2}  <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
    ShllAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// SHRN, SHRN2 -- A64
    /// Shift Right Narrow (immediate)
    /// SHRN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    ShrnAdvsimd {
        q: i32,
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
    },
    /// SHSUB -- A64
    /// Signed Halving Subtract
    /// SHSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    ShsubAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SLI -- A64
    /// Shift Left and Insert (immediate)
    /// SLI  <V><d>, <V><n>, #<shift>
    /// SLI  <Vd>.<T>, <Vn>.<T>, #<shift>
    SliAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SliAdvsimdSelector,
    },
    /// SM3PARTW1 -- A64
    /// SM3PARTW1
    /// SM3PARTW1  <Vd>.4S, <Vn>.4S, <Vm>.4S
    Sm3partw1Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SM3PARTW2 -- A64
    /// SM3PARTW2
    /// SM3PARTW2  <Vd>.4S, <Vn>.4S, <Vm>.4S
    Sm3partw2Advsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SM3SS1 -- A64
    /// SM3SS1
    /// SM3SS1  <Vd>.4S, <Vn>.4S, <Vm>.4S, <Va>.4S
    Sm3ss1Advsimd {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// SM3TT1A -- A64
    /// SM3TT1A
    /// SM3TT1A  <Vd>.4S, <Vn>.4S, <Vm>.S[<imm2>]
    Sm3tt1aAdvsimd {
        rm: Register,
        imm2: i32,
        rn: Register,
        rd: Register,
    },
    /// SM3TT1B -- A64
    /// SM3TT1B
    /// SM3TT1B  <Vd>.4S, <Vn>.4S, <Vm>.S[<imm2>]
    Sm3tt1bAdvsimd {
        rm: Register,
        imm2: i32,
        rn: Register,
        rd: Register,
    },
    /// SM3TT2A -- A64
    /// SM3TT2A
    /// SM3TT2A  <Vd>.4S, <Vn>.4S, <Vm>.S[<imm2>]
    Sm3tt2aAdvsimd {
        rm: Register,
        imm2: i32,
        rn: Register,
        rd: Register,
    },
    /// SM3TT2B -- A64
    /// SM3TT2B
    /// SM3TT2B  <Vd>.4S, <Vn>.4S, <Vm>.S[<imm2>]
    Sm3tt2bAdvsimd {
        rm: Register,
        imm2: i32,
        rn: Register,
        rd: Register,
    },
    /// SM4E -- A64
    /// SM4 Encode
    /// SM4E  <Vd>.4S, <Vn>.4S
    Sm4eAdvsimd { rn: Register, rd: Register },
    /// SM4EKEY -- A64
    /// SM4 Key
    /// SM4EKEY  <Vd>.4S, <Vn>.4S, <Vm>.4S
    Sm4ekeyAdvsimd {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMADDL -- A64
    /// Signed Multiply-Add Long
    /// SMADDL  <Xd>, <Wn>, <Wm>, <Xa>
    Smaddl {
        rm: Register,
        ra: Register,
        rn: Register,
        rd: Register,
    },
    /// SMAX -- A64
    /// Signed Maximum (vector)
    /// SMAX  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SmaxAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMAXP -- A64
    /// Signed Maximum Pairwise
    /// SMAXP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SmaxpAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMAXV -- A64
    /// Signed Maximum across Vector
    /// SMAXV  <V><d>, <Vn>.<T>
    SmaxvAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// SMC -- A64
    /// Secure Monitor Call
    /// SMC  #<imm>
    Smc { imm16: i32 },
    /// SMIN -- A64
    /// Signed Minimum (vector)
    /// SMIN  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SminAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMINP -- A64
    /// Signed Minimum Pairwise
    /// SMINP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SminpAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMINV -- A64
    /// Signed Minimum across Vector
    /// SMINV  <V><d>, <Vn>.<T>
    SminvAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// SMLAL, SMLAL2 (by element) -- A64
    /// Signed Multiply-Add Long (vector, by element)
    /// SMLAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    SmlalAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// SMLAL, SMLAL2 (vector) -- A64
    /// Signed Multiply-Add Long (vector)
    /// SMLAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SmlalAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMLSL, SMLSL2 (by element) -- A64
    /// Signed Multiply-Subtract Long (vector, by element)
    /// SMLSL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    SmlslAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// SMLSL, SMLSL2 (vector) -- A64
    /// Signed Multiply-Subtract Long (vector)
    /// SMLSL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SmlslAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMMLA (vector) -- A64
    /// Signed 8-bit integer matrix multiply-accumulate (vector)
    /// SMMLA  <Vd>.4S, <Vn>.16B, <Vm>.16B
    SmmlaAdvsimdVec {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMNEGL -- A64
    /// Signed Multiply-Negate Long
    /// SMNEGL  <Xd>, <Wn>, <Wm>
    /// SMSUBL <Xd>, <Wn>, <Wm>, XZR
    SmneglSmsubl {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SMOV -- A64
    /// Signed Move vector element to general-purpose register
    /// SMOV  <Wd>, <Vn>.<Ts>[<index>]
    /// SMOV  <Xd>, <Vn>.<Ts>[<index>]
    SmovAdvsimd {
        q: i32,
        imm5: i32,
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
    /// SMULL, SMULL2 (by element) -- A64
    /// Signed Multiply Long (vector, by element)
    /// SMULL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    SmullAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// SMULL, SMULL2 (vector) -- A64
    /// Signed Multiply Long (vector)
    /// SMULL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SmullAdvsimdVec {
        q: i32,
        size: i32,
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
    /// SQABS -- A64
    /// Signed saturating Absolute value
    /// SQABS  <V><d>, <V><n>
    /// SQABS  <Vd>.<T>, <Vn>.<T>
    SqabsAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqabsAdvsimdSelector,
    },
    /// SQADD -- A64
    /// Signed saturating Add
    /// SQADD  <V><d>, <V><n>, <V><m>
    /// SQADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqaddAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqaddAdvsimdSelector,
    },
    /// SQDMLAL, SQDMLAL2 (by element) -- A64
    /// Signed saturating Doubling Multiply-Add Long (by element)
    /// SQDMLAL  <Va><d>, <Vb><n>, <Vm>.<Ts>[<index>]
    /// SQDMLAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    SqdmlalAdvsimdElt {
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmlalAdvsimdEltSelector,
    },
    /// SQDMLAL, SQDMLAL2 (vector) -- A64
    /// Signed saturating Doubling Multiply-Add Long
    /// SQDMLAL  <Va><d>, <Vb><n>, <Vb><m>
    /// SQDMLAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SqdmlalAdvsimdVec {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmlalAdvsimdVecSelector,
    },
    /// SQDMLSL, SQDMLSL2 (by element) -- A64
    /// Signed saturating Doubling Multiply-Subtract Long (by element)
    /// SQDMLSL  <Va><d>, <Vb><n>, <Vm>.<Ts>[<index>]
    /// SQDMLSL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    SqdmlslAdvsimdElt {
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmlslAdvsimdEltSelector,
    },
    /// SQDMLSL, SQDMLSL2 (vector) -- A64
    /// Signed saturating Doubling Multiply-Subtract Long
    /// SQDMLSL  <Va><d>, <Vb><n>, <Vb><m>
    /// SQDMLSL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SqdmlslAdvsimdVec {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmlslAdvsimdVecSelector,
    },
    /// SQDMULH (by element) -- A64
    /// Signed saturating Doubling Multiply returning High half (by element)
    /// SQDMULH  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// SQDMULH  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    SqdmulhAdvsimdElt {
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmulhAdvsimdEltSelector,
    },
    /// SQDMULH (vector) -- A64
    /// Signed saturating Doubling Multiply returning High half
    /// SQDMULH  <V><d>, <V><n>, <V><m>
    /// SQDMULH  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqdmulhAdvsimdVec {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmulhAdvsimdVecSelector,
    },
    /// SQDMULL, SQDMULL2 (by element) -- A64
    /// Signed saturating Doubling Multiply Long (by element)
    /// SQDMULL  <Va><d>, <Vb><n>, <Vm>.<Ts>[<index>]
    /// SQDMULL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    SqdmullAdvsimdElt {
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmullAdvsimdEltSelector,
    },
    /// SQDMULL, SQDMULL2 (vector) -- A64
    /// Signed saturating Doubling Multiply Long
    /// SQDMULL  <Va><d>, <Vb><n>, <Vb><m>
    /// SQDMULL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SqdmullAdvsimdVec {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqdmullAdvsimdVecSelector,
    },
    /// SQNEG -- A64
    /// Signed saturating Negate
    /// SQNEG  <V><d>, <V><n>
    /// SQNEG  <Vd>.<T>, <Vn>.<T>
    SqnegAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqnegAdvsimdSelector,
    },
    /// SQRDMLAH (by element) -- A64
    /// Signed Saturating Rounding Doubling Multiply Accumulate returning High Half (by element)
    /// SQRDMLAH  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// SQRDMLAH  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    SqrdmlahAdvsimdElt {
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrdmlahAdvsimdEltSelector,
    },
    /// SQRDMLAH (vector) -- A64
    /// Signed Saturating Rounding Doubling Multiply Accumulate returning High Half (vector)
    /// SQRDMLAH  <V><d>, <V><n>, <V><m>
    /// SQRDMLAH  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqrdmlahAdvsimdVec {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrdmlahAdvsimdVecSelector,
    },
    /// SQRDMLSH (by element) -- A64
    /// Signed Saturating Rounding Doubling Multiply Subtract returning High Half (by element)
    /// SQRDMLSH  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// SQRDMLSH  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    SqrdmlshAdvsimdElt {
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrdmlshAdvsimdEltSelector,
    },
    /// SQRDMLSH (vector) -- A64
    /// Signed Saturating Rounding Doubling Multiply Subtract returning High Half (vector)
    /// SQRDMLSH  <V><d>, <V><n>, <V><m>
    /// SQRDMLSH  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqrdmlshAdvsimdVec {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrdmlshAdvsimdVecSelector,
    },
    /// SQRDMULH (by element) -- A64
    /// Signed saturating Rounding Doubling Multiply returning High half (by element)
    /// SQRDMULH  <V><d>, <V><n>, <Vm>.<Ts>[<index>]
    /// SQRDMULH  <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]
    SqrdmulhAdvsimdElt {
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrdmulhAdvsimdEltSelector,
    },
    /// SQRDMULH (vector) -- A64
    /// Signed saturating Rounding Doubling Multiply returning High half
    /// SQRDMULH  <V><d>, <V><n>, <V><m>
    /// SQRDMULH  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqrdmulhAdvsimdVec {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrdmulhAdvsimdVecSelector,
    },
    /// SQRSHL -- A64
    /// Signed saturating Rounding Shift Left (register)
    /// SQRSHL  <V><d>, <V><n>, <V><m>
    /// SQRSHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqrshlAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrshlAdvsimdSelector,
    },
    /// SQRSHRN, SQRSHRN2 -- A64
    /// Signed saturating Rounded Shift Right Narrow (immediate)
    /// SQRSHRN  <Vb><d>, <Va><n>, #<shift>
    /// SQRSHRN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    SqrshrnAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrshrnAdvsimdSelector,
    },
    /// SQRSHRUN, SQRSHRUN2 -- A64
    /// Signed saturating Rounded Shift Right Unsigned Narrow (immediate)
    /// SQRSHRUN  <Vb><d>, <Va><n>, #<shift>
    /// SQRSHRUN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    SqrshrunAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqrshrunAdvsimdSelector,
    },
    /// SQSHL (immediate) -- A64
    /// Signed saturating Shift Left (immediate)
    /// SQSHL  <V><d>, <V><n>, #<shift>
    /// SQSHL  <Vd>.<T>, <Vn>.<T>, #<shift>
    SqshlAdvsimdImm {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqshlAdvsimdImmSelector,
    },
    /// SQSHL (register) -- A64
    /// Signed saturating Shift Left (register)
    /// SQSHL  <V><d>, <V><n>, <V><m>
    /// SQSHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqshlAdvsimdReg {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqshlAdvsimdRegSelector,
    },
    /// SQSHLU -- A64
    /// Signed saturating Shift Left Unsigned (immediate)
    /// SQSHLU  <V><d>, <V><n>, #<shift>
    /// SQSHLU  <Vd>.<T>, <Vn>.<T>, #<shift>
    SqshluAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqshluAdvsimdSelector,
    },
    /// SQSHRN, SQSHRN2 -- A64
    /// Signed saturating Shift Right Narrow (immediate)
    /// SQSHRN  <Vb><d>, <Va><n>, #<shift>
    /// SQSHRN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    SqshrnAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqshrnAdvsimdSelector,
    },
    /// SQSHRUN, SQSHRUN2 -- A64
    /// Signed saturating Shift Right Unsigned Narrow (immediate)
    /// SQSHRUN  <Vb><d>, <Va><n>, #<shift>
    /// SQSHRUN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    SqshrunAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqshrunAdvsimdSelector,
    },
    /// SQSUB -- A64
    /// Signed saturating Subtract
    /// SQSUB  <V><d>, <V><n>, <V><m>
    /// SQSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SqsubAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqsubAdvsimdSelector,
    },
    /// SQXTN, SQXTN2 -- A64
    /// Signed saturating extract Narrow
    /// SQXTN  <Vb><d>, <Va><n>
    /// SQXTN{2}  <Vd>.<Tb>, <Vn>.<Ta>
    SqxtnAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqxtnAdvsimdSelector,
    },
    /// SQXTUN, SQXTUN2 -- A64
    /// Signed saturating extract Unsigned Narrow
    /// SQXTUN  <Vb><d>, <Va><n>
    /// SQXTUN{2}  <Vd>.<Tb>, <Vn>.<Ta>
    SqxtunAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SqxtunAdvsimdSelector,
    },
    /// SRHADD -- A64
    /// Signed Rounding Halving Add
    /// SRHADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SrhaddAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SRI -- A64
    /// Shift Right and Insert (immediate)
    /// SRI  <V><d>, <V><n>, #<shift>
    /// SRI  <Vd>.<T>, <Vn>.<T>, #<shift>
    SriAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SriAdvsimdSelector,
    },
    /// SRSHL -- A64
    /// Signed Rounding Shift Left (register)
    /// SRSHL  <V><d>, <V><n>, <V><m>
    /// SRSHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SrshlAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SrshlAdvsimdSelector,
    },
    /// SRSHR -- A64
    /// Signed Rounding Shift Right (immediate)
    /// SRSHR  <V><d>, <V><n>, #<shift>
    /// SRSHR  <Vd>.<T>, <Vn>.<T>, #<shift>
    SrshrAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SrshrAdvsimdSelector,
    },
    /// SRSRA -- A64
    /// Signed Rounding Shift Right and Accumulate (immediate)
    /// SRSRA  <V><d>, <V><n>, #<shift>
    /// SRSRA  <Vd>.<T>, <Vn>.<T>, #<shift>
    SrsraAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SrsraAdvsimdSelector,
    },
    /// SSBB -- A64
    /// Speculative Store Bypass Barrier
    /// SSBB
    /// DSB #0
    SsbbDsb,
    /// SSHL -- A64
    /// Signed Shift Left (register)
    /// SSHL  <V><d>, <V><n>, <V><m>
    /// SSHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SshlAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SshlAdvsimdSelector,
    },
    /// SSHLL, SSHLL2 -- A64
    /// Signed Shift Left Long (immediate)
    /// SSHLL{2}  <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
    SshllAdvsimd {
        q: i32,
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
    },
    /// SSHR -- A64
    /// Signed Shift Right (immediate)
    /// SSHR  <V><d>, <V><n>, #<shift>
    /// SSHR  <Vd>.<T>, <Vn>.<T>, #<shift>
    SshrAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SshrAdvsimdSelector,
    },
    /// SSRA -- A64
    /// Signed Shift Right and Accumulate (immediate)
    /// SSRA  <V><d>, <V><n>, #<shift>
    /// SSRA  <Vd>.<T>, <Vn>.<T>, #<shift>
    SsraAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SsraAdvsimdSelector,
    },
    /// SSUBL, SSUBL2 -- A64
    /// Signed Subtract Long
    /// SSUBL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    SsublAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// SSUBW, SSUBW2 -- A64
    /// Signed Subtract Wide
    /// SSUBW{2}  <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
    SsubwAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// ST1 (multiple structures) -- A64
    /// Store multiple single-element structures from one, two, three, or four registers
    /// ST1  { <Vt>.<T> }, [<Xn|SP>]
    /// ST1  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
    /// ST1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>]
    /// ST1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
    /// ST1  { <Vt>.<T> }, [<Xn|SP>], <imm>
    /// ST1  { <Vt>.<T> }, [<Xn|SP>], <Xm>
    /// ST1  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>
    /// ST1  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <Xm>
    /// ST1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <imm>
    /// ST1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <Xm>
    /// ST1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
    /// ST1  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <Xm>
    St1AdvsimdMult {
        q: i32,
        opcode: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St1AdvsimdMultSelector,
    },
    /// ST1 (single structure) -- A64
    /// Store a single-element structure from one lane of one register
    /// ST1  { <Vt>.B }[<index>], [<Xn|SP>]
    /// ST1  { <Vt>.H }[<index>], [<Xn|SP>]
    /// ST1  { <Vt>.S }[<index>], [<Xn|SP>]
    /// ST1  { <Vt>.D }[<index>], [<Xn|SP>]
    /// ST1  { <Vt>.B }[<index>], [<Xn|SP>], #1
    /// ST1  { <Vt>.B }[<index>], [<Xn|SP>], <Xm>
    /// ST1  { <Vt>.H }[<index>], [<Xn|SP>], #2
    /// ST1  { <Vt>.H }[<index>], [<Xn|SP>], <Xm>
    /// ST1  { <Vt>.S }[<index>], [<Xn|SP>], #4
    /// ST1  { <Vt>.S }[<index>], [<Xn|SP>], <Xm>
    /// ST1  { <Vt>.D }[<index>], [<Xn|SP>], #8
    /// ST1  { <Vt>.D }[<index>], [<Xn|SP>], <Xm>
    St1AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St1AdvsimdSnglSelector,
    },
    /// ST2 (multiple structures) -- A64
    /// Store multiple 2-element structures from two registers
    /// ST2  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>]
    /// ST2  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <imm>
    /// ST2  { <Vt>.<T>, <Vt2>.<T> }, [<Xn|SP>], <Xm>
    St2AdvsimdMult {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St2AdvsimdMultSelector,
    },
    /// ST2 (single structure) -- A64
    /// Store single 2-element structure from one lane of two registers
    /// ST2  { <Vt>.B, <Vt2>.B }[<index>], [<Xn|SP>]
    /// ST2  { <Vt>.H, <Vt2>.H }[<index>], [<Xn|SP>]
    /// ST2  { <Vt>.S, <Vt2>.S }[<index>], [<Xn|SP>]
    /// ST2  { <Vt>.D, <Vt2>.D }[<index>], [<Xn|SP>]
    /// ST2  { <Vt>.B, <Vt2>.B }[<index>], [<Xn|SP>], #2
    /// ST2  { <Vt>.B, <Vt2>.B }[<index>], [<Xn|SP>], <Xm>
    /// ST2  { <Vt>.H, <Vt2>.H }[<index>], [<Xn|SP>], #4
    /// ST2  { <Vt>.H, <Vt2>.H }[<index>], [<Xn|SP>], <Xm>
    /// ST2  { <Vt>.S, <Vt2>.S }[<index>], [<Xn|SP>], #8
    /// ST2  { <Vt>.S, <Vt2>.S }[<index>], [<Xn|SP>], <Xm>
    /// ST2  { <Vt>.D, <Vt2>.D }[<index>], [<Xn|SP>], #16
    /// ST2  { <Vt>.D, <Vt2>.D }[<index>], [<Xn|SP>], <Xm>
    St2AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St2AdvsimdSnglSelector,
    },
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
    /// ST3 (multiple structures) -- A64
    /// Store multiple 3-element structures from three registers
    /// ST3  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>]
    /// ST3  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <imm>
    /// ST3  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T> }, [<Xn|SP>], <Xm>
    St3AdvsimdMult {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St3AdvsimdMultSelector,
    },
    /// ST3 (single structure) -- A64
    /// Store single 3-element structure from one lane of three registers
    /// ST3  { <Vt>.B, <Vt2>.B, <Vt3>.B }[<index>], [<Xn|SP>]
    /// ST3  { <Vt>.H, <Vt2>.H, <Vt3>.H }[<index>], [<Xn|SP>]
    /// ST3  { <Vt>.S, <Vt2>.S, <Vt3>.S }[<index>], [<Xn|SP>]
    /// ST3  { <Vt>.D, <Vt2>.D, <Vt3>.D }[<index>], [<Xn|SP>]
    /// ST3  { <Vt>.B, <Vt2>.B, <Vt3>.B }[<index>], [<Xn|SP>], #3
    /// ST3  { <Vt>.B, <Vt2>.B, <Vt3>.B }[<index>], [<Xn|SP>], <Xm>
    /// ST3  { <Vt>.H, <Vt2>.H, <Vt3>.H }[<index>], [<Xn|SP>], #6
    /// ST3  { <Vt>.H, <Vt2>.H, <Vt3>.H }[<index>], [<Xn|SP>], <Xm>
    /// ST3  { <Vt>.S, <Vt2>.S, <Vt3>.S }[<index>], [<Xn|SP>], #12
    /// ST3  { <Vt>.S, <Vt2>.S, <Vt3>.S }[<index>], [<Xn|SP>], <Xm>
    /// ST3  { <Vt>.D, <Vt2>.D, <Vt3>.D }[<index>], [<Xn|SP>], #24
    /// ST3  { <Vt>.D, <Vt2>.D, <Vt3>.D }[<index>], [<Xn|SP>], <Xm>
    St3AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St3AdvsimdSnglSelector,
    },
    /// ST4 (multiple structures) -- A64
    /// Store multiple 4-element structures from four registers
    /// ST4  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>]
    /// ST4  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <imm>
    /// ST4  { <Vt>.<T>, <Vt2>.<T>, <Vt3>.<T>, <Vt4>.<T> }, [<Xn|SP>], <Xm>
    St4AdvsimdMult {
        q: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St4AdvsimdMultSelector,
    },
    /// ST4 (single structure) -- A64
    /// Store single 4-element structure from one lane of four registers
    /// ST4  { <Vt>.B, <Vt2>.B, <Vt3>.B, <Vt4>.B }[<index>], [<Xn|SP>]
    /// ST4  { <Vt>.H, <Vt2>.H, <Vt3>.H, <Vt4>.H }[<index>], [<Xn|SP>]
    /// ST4  { <Vt>.S, <Vt2>.S, <Vt3>.S, <Vt4>.S }[<index>], [<Xn|SP>]
    /// ST4  { <Vt>.D, <Vt2>.D, <Vt3>.D, <Vt4>.D }[<index>], [<Xn|SP>]
    /// ST4  { <Vt>.B, <Vt2>.B, <Vt3>.B, <Vt4>.B }[<index>], [<Xn|SP>], #4
    /// ST4  { <Vt>.B, <Vt2>.B, <Vt3>.B, <Vt4>.B }[<index>], [<Xn|SP>], <Xm>
    /// ST4  { <Vt>.H, <Vt2>.H, <Vt3>.H, <Vt4>.H }[<index>], [<Xn|SP>], #8
    /// ST4  { <Vt>.H, <Vt2>.H, <Vt3>.H, <Vt4>.H }[<index>], [<Xn|SP>], <Xm>
    /// ST4  { <Vt>.S, <Vt2>.S, <Vt3>.S, <Vt4>.S }[<index>], [<Xn|SP>], #16
    /// ST4  { <Vt>.S, <Vt2>.S, <Vt3>.S, <Vt4>.S }[<index>], [<Xn|SP>], <Xm>
    /// ST4  { <Vt>.D, <Vt2>.D, <Vt3>.D, <Vt4>.D }[<index>], [<Xn|SP>], #32
    /// ST4  { <Vt>.D, <Vt2>.D, <Vt3>.D, <Vt4>.D }[<index>], [<Xn|SP>], <Xm>
    St4AdvsimdSngl {
        q: i32,
        opcode: i32,
        s: i32,
        size: i32,
        rn: Register,
        rt: Register,
        rm: Register,
        class_selector: St4AdvsimdSnglSelector,
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
    /// STNP (SIMD&FP) -- A64
    /// Store Pair of SIMD&FP registers, with Non-temporal hint
    /// STNP  <St1>, <St2>, [<Xn|SP>{, #<imm>}]
    /// STNP  <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]
    /// STNP  <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]
    StnpFpsimd {
        opc: i32,
        imm7: i32,
        rt2: Register,
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
    /// STP (SIMD&FP) -- A64
    /// Store Pair of SIMD&FP registers
    /// STP  <St1>, <St2>, [<Xn|SP>], #<imm>
    /// STP  <Dt1>, <Dt2>, [<Xn|SP>], #<imm>
    /// STP  <Qt1>, <Qt2>, [<Xn|SP>], #<imm>
    /// STP  <St1>, <St2>, [<Xn|SP>, #<imm>]!
    /// STP  <Dt1>, <Dt2>, [<Xn|SP>, #<imm>]!
    /// STP  <Qt1>, <Qt2>, [<Xn|SP>, #<imm>]!
    /// STP  <St1>, <St2>, [<Xn|SP>{, #<imm>}]
    /// STP  <Dt1>, <Dt2>, [<Xn|SP>{, #<imm>}]
    /// STP  <Qt1>, <Qt2>, [<Xn|SP>{, #<imm>}]
    StpFpsimd {
        opc: i32,
        imm7: i32,
        rt2: Register,
        rn: Register,
        rt: Register,
        class_selector: StpFpsimdSelector,
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
    /// STR (immediate, SIMD&FP) -- A64
    /// Store SIMD&FP register (immediate offset)
    /// STR  <Bt>, [<Xn|SP>], #<simm>
    /// STR  <Ht>, [<Xn|SP>], #<simm>
    /// STR  <St>, [<Xn|SP>], #<simm>
    /// STR  <Dt>, [<Xn|SP>], #<simm>
    /// STR  <Qt>, [<Xn|SP>], #<simm>
    /// STR  <Bt>, [<Xn|SP>, #<simm>]!
    /// STR  <Ht>, [<Xn|SP>, #<simm>]!
    /// STR  <St>, [<Xn|SP>, #<simm>]!
    /// STR  <Dt>, [<Xn|SP>, #<simm>]!
    /// STR  <Qt>, [<Xn|SP>, #<simm>]!
    /// STR  <Bt>, [<Xn|SP>{, #<pimm>}]
    /// STR  <Ht>, [<Xn|SP>{, #<pimm>}]
    /// STR  <St>, [<Xn|SP>{, #<pimm>}]
    /// STR  <Dt>, [<Xn|SP>{, #<pimm>}]
    /// STR  <Qt>, [<Xn|SP>{, #<pimm>}]
    StrImmFpsimd {
        size: i32,
        opc: i32,
        imm9: i32,
        rn: Register,
        rt: Register,
        imm12: i32,
        class_selector: StrImmFpsimdSelector,
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
    /// STR (register, SIMD&FP) -- A64
    /// Store SIMD&FP register (register offset)
    /// STR  <Bt>, [<Xn|SP>, (<Wm>|<Xm>), <extend> {<amount>}]
    /// STR  <Bt>, [<Xn|SP>, <Xm>{, LSL <amount>}]
    /// STR  <Ht>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// STR  <St>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// STR  <Dt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    /// STR  <Qt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
    StrRegFpsimd {
        size: i32,
        opc: i32,
        rm: Register,
        option: i32,
        s: i32,
        rn: Register,
        rt: Register,
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
    /// STUR (SIMD&FP) -- A64
    /// Store SIMD&FP register (unscaled offset)
    /// STUR  <Bt>, [<Xn|SP>{, #<simm>}]
    /// STUR  <Ht>, [<Xn|SP>{, #<simm>}]
    /// STUR  <St>, [<Xn|SP>{, #<simm>}]
    /// STUR  <Dt>, [<Xn|SP>{, #<simm>}]
    /// STUR  <Qt>, [<Xn|SP>{, #<simm>}]
    SturFpsimd {
        size: i32,
        opc: i32,
        imm9: i32,
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
    /// SUB (vector) -- A64
    /// Subtract (vector)
    /// SUB  <V><d>, <V><n>, <V><m>
    /// SUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    SubAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SubAdvsimdSelector,
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
    /// SUBHN, SUBHN2 -- A64
    /// Subtract returning High Narrow
    /// SUBHN{2}  <Vd>.<Tb>, <Vn>.<Ta>, <Vm>.<Ta>
    SubhnAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
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
    /// SUDOT (by element) -- A64
    /// Dot product with signed and unsigned integers (vector, by element)
    /// SUDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.4B[<index>]
    SudotAdvsimdElt {
        q: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// SUQADD -- A64
    /// Signed saturating Accumulate of Unsigned value
    /// SUQADD  <V><d>, <V><n>
    /// SUQADD  <Vd>.<T>, <Vn>.<T>
    SuqaddAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: SuqaddAdvsimdSelector,
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
    /// SXTL, SXTL2 -- A64
    /// Signed extend Long
    /// SXTL{2}  <Vd>.<Ta>, <Vn>.<Tb>
    /// SSHLL{2}  <Vd>.<Ta>, <Vn>.<Tb>, #0
    SxtlSshllAdvsimd {
        q: i32,
        immh: i32,
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
    /// TBL -- A64
    /// Table vector Lookup
    /// TBL  <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B }, <Vm>.<Ta>
    /// TBL  <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B }, <Vm>.<Ta>
    /// TBL  <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B, <Vn+3>.16B }, <Vm>.<Ta>
    /// TBL  <Vd>.<Ta>, { <Vn>.16B }, <Vm>.<Ta>
    TblAdvsimd {
        q: i32,
        rm: Register,
        len: i32,
        rn: Register,
        rd: Register,
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
    /// TBX -- A64
    /// Table vector lookup extension
    /// TBX  <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B }, <Vm>.<Ta>
    /// TBX  <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B }, <Vm>.<Ta>
    /// TBX  <Vd>.<Ta>, { <Vn>.16B, <Vn+1>.16B, <Vn+2>.16B, <Vn+3>.16B }, <Vm>.<Ta>
    /// TBX  <Vd>.<Ta>, { <Vn>.16B }, <Vm>.<Ta>
    TbxAdvsimd {
        q: i32,
        rm: Register,
        len: i32,
        rn: Register,
        rd: Register,
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
    /// TRN1 -- A64
    /// Transpose vectors (primary)
    /// TRN1  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    Trn1Advsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// TRN2 -- A64
    /// Transpose vectors (secondary)
    /// TRN2  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    Trn2Advsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
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
    /// UABA -- A64
    /// Unsigned Absolute difference and Accumulate
    /// UABA  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UabaAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UABAL, UABAL2 -- A64
    /// Unsigned Absolute difference and Accumulate Long
    /// UABAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UabalAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UABD -- A64
    /// Unsigned Absolute Difference (vector)
    /// UABD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UabdAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UABDL, UABDL2 -- A64
    /// Unsigned Absolute Difference Long
    /// UABDL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UabdlAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UADALP -- A64
    /// Unsigned Add and Accumulate Long Pairwise
    /// UADALP  <Vd>.<Ta>, <Vn>.<Tb>
    UadalpAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// UADDL, UADDL2 -- A64
    /// Unsigned Add Long (vector)
    /// UADDL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UaddlAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UADDLP -- A64
    /// Unsigned Add Long Pairwise
    /// UADDLP  <Vd>.<Ta>, <Vn>.<Tb>
    UaddlpAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// UADDLV -- A64
    /// Unsigned sum Long across Vector
    /// UADDLV  <V><d>, <Vn>.<T>
    UaddlvAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// UADDW, UADDW2 -- A64
    /// Unsigned Add Wide
    /// UADDW{2}  <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
    UaddwAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
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
    /// UCVTF (vector, fixed-point) -- A64
    /// Unsigned fixed-point Convert to Floating-point (vector)
    /// UCVTF  <V><d>, <V><n>, #<fbits>
    /// UCVTF  <Vd>.<T>, <Vn>.<T>, #<fbits>
    UcvtfAdvsimdFix {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UcvtfAdvsimdFixSelector,
    },
    /// UCVTF (vector, integer) -- A64
    /// Unsigned integer Convert to Floating-point (vector)
    /// UCVTF  <Hd>, <Hn>
    /// UCVTF  <V><d>, <V><n>
    /// UCVTF  <Vd>.<T>, <Vn>.<T>
    /// UCVTF  <Vd>.<T>, <Vn>.<T>
    UcvtfAdvsimdInt {
        rn: Register,
        rd: Register,
        sz: i32,
        q: i32,
        class_selector: UcvtfAdvsimdIntSelector,
    },
    /// UCVTF (scalar, fixed-point) -- A64
    /// Unsigned fixed-point Convert to Floating-point (scalar)
    /// UCVTF  <Hd>, <Wn>, #<fbits>
    /// UCVTF  <Sd>, <Wn>, #<fbits>
    /// UCVTF  <Dd>, <Wn>, #<fbits>
    /// UCVTF  <Hd>, <Xn>, #<fbits>
    /// UCVTF  <Sd>, <Xn>, #<fbits>
    /// UCVTF  <Dd>, <Xn>, #<fbits>
    UcvtfFloatFix {
        sf: i32,
        ftype: i32,
        scale: i32,
        rn: Register,
        rd: Register,
    },
    /// UCVTF (scalar, integer) -- A64
    /// Unsigned integer Convert to Floating-point (scalar)
    /// UCVTF  <Hd>, <Wn>
    /// UCVTF  <Sd>, <Wn>
    /// UCVTF  <Dd>, <Wn>
    /// UCVTF  <Hd>, <Xn>
    /// UCVTF  <Sd>, <Xn>
    /// UCVTF  <Dd>, <Xn>
    UcvtfFloatInt {
        sf: i32,
        ftype: i32,
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
    /// UDOT (by element) -- A64
    /// Dot Product unsigned arithmetic (vector, by element)
    /// UDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.4B[<index>]
    UdotAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// UDOT (vector) -- A64
    /// Dot Product unsigned arithmetic (vector)
    /// UDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UdotAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UHADD -- A64
    /// Unsigned Halving Add
    /// UHADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UhaddAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UHSUB -- A64
    /// Unsigned Halving Subtract
    /// UHSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UhsubAdvsimd {
        q: i32,
        size: i32,
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
    /// UMAX -- A64
    /// Unsigned Maximum (vector)
    /// UMAX  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UmaxAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMAXP -- A64
    /// Unsigned Maximum Pairwise
    /// UMAXP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UmaxpAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMAXV -- A64
    /// Unsigned Maximum across Vector
    /// UMAXV  <V><d>, <Vn>.<T>
    UmaxvAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// UMIN -- A64
    /// Unsigned Minimum (vector)
    /// UMIN  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UminAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMINP -- A64
    /// Unsigned Minimum Pairwise
    /// UMINP  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UminpAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMINV -- A64
    /// Unsigned Minimum across Vector
    /// UMINV  <V><d>, <Vn>.<T>
    UminvAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// UMLAL, UMLAL2 (by element) -- A64
    /// Unsigned Multiply-Add Long (vector, by element)
    /// UMLAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    UmlalAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// UMLAL, UMLAL2 (vector) -- A64
    /// Unsigned Multiply-Add Long (vector)
    /// UMLAL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UmlalAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMLSL, UMLSL2 (by element) -- A64
    /// Unsigned Multiply-Subtract Long (vector, by element)
    /// UMLSL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    UmlslAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// UMLSL, UMLSL2 (vector) -- A64
    /// Unsigned Multiply-Subtract Long (vector)
    /// UMLSL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UmlslAdvsimdVec {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UMMLA (vector) -- A64
    /// Unsigned 8-bit integer matrix multiply-accumulate (vector)
    /// UMMLA  <Vd>.4S, <Vn>.16B, <Vm>.16B
    UmmlaAdvsimdVec {
        rm: Register,
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
    /// UMOV -- A64
    /// Unsigned Move vector element to general-purpose register
    /// UMOV  <Wd>, <Vn>.<Ts>[<index>]
    /// UMOV  <Xd>, <Vn>.<Ts>[<index>]
    UmovAdvsimd {
        q: i32,
        imm5: i32,
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
    /// UMULL, UMULL2 (by element) -- A64
    /// Unsigned Multiply Long (vector, by element)
    /// UMULL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Ts>[<index>]
    UmullAdvsimdElt {
        q: i32,
        size: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// UMULL, UMULL2 (vector) -- A64
    /// Unsigned Multiply long (vector)
    /// UMULL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UmullAdvsimdVec {
        q: i32,
        size: i32,
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
    /// UQADD -- A64
    /// Unsigned saturating Add
    /// UQADD  <V><d>, <V><n>, <V><m>
    /// UQADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UqaddAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqaddAdvsimdSelector,
    },
    /// UQRSHL -- A64
    /// Unsigned saturating Rounding Shift Left (register)
    /// UQRSHL  <V><d>, <V><n>, <V><m>
    /// UQRSHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UqrshlAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqrshlAdvsimdSelector,
    },
    /// UQRSHRN, UQRSHRN2 -- A64
    /// Unsigned saturating Rounded Shift Right Narrow (immediate)
    /// UQRSHRN  <Vb><d>, <Va><n>, #<shift>
    /// UQRSHRN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    UqrshrnAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqrshrnAdvsimdSelector,
    },
    /// UQSHL (immediate) -- A64
    /// Unsigned saturating Shift Left (immediate)
    /// UQSHL  <V><d>, <V><n>, #<shift>
    /// UQSHL  <Vd>.<T>, <Vn>.<T>, #<shift>
    UqshlAdvsimdImm {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqshlAdvsimdImmSelector,
    },
    /// UQSHL (register) -- A64
    /// Unsigned saturating Shift Left (register)
    /// UQSHL  <V><d>, <V><n>, <V><m>
    /// UQSHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UqshlAdvsimdReg {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqshlAdvsimdRegSelector,
    },
    /// UQSHRN, UQSHRN2 -- A64
    /// Unsigned saturating Shift Right Narrow (immediate)
    /// UQSHRN  <Vb><d>, <Va><n>, #<shift>
    /// UQSHRN{2}  <Vd>.<Tb>, <Vn>.<Ta>, #<shift>
    UqshrnAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqshrnAdvsimdSelector,
    },
    /// UQSUB -- A64
    /// Unsigned saturating Subtract
    /// UQSUB  <V><d>, <V><n>, <V><m>
    /// UQSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UqsubAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqsubAdvsimdSelector,
    },
    /// UQXTN, UQXTN2 -- A64
    /// Unsigned saturating extract Narrow
    /// UQXTN  <Vb><d>, <Va><n>
    /// UQXTN{2}  <Vd>.<Tb>, <Vn>.<Ta>
    UqxtnAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UqxtnAdvsimdSelector,
    },
    /// URECPE -- A64
    /// Unsigned Reciprocal Estimate
    /// URECPE  <Vd>.<T>, <Vn>.<T>
    UrecpeAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// URHADD -- A64
    /// Unsigned Rounding Halving Add
    /// URHADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UrhaddAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// URSHL -- A64
    /// Unsigned Rounding Shift Left (register)
    /// URSHL  <V><d>, <V><n>, <V><m>
    /// URSHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UrshlAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UrshlAdvsimdSelector,
    },
    /// URSHR -- A64
    /// Unsigned Rounding Shift Right (immediate)
    /// URSHR  <V><d>, <V><n>, #<shift>
    /// URSHR  <Vd>.<T>, <Vn>.<T>, #<shift>
    UrshrAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UrshrAdvsimdSelector,
    },
    /// URSQRTE -- A64
    /// Unsigned Reciprocal Square Root Estimate
    /// URSQRTE  <Vd>.<T>, <Vn>.<T>
    UrsqrteAdvsimd {
        q: i32,
        sz: i32,
        rn: Register,
        rd: Register,
    },
    /// URSRA -- A64
    /// Unsigned Rounding Shift Right and Accumulate (immediate)
    /// URSRA  <V><d>, <V><n>, #<shift>
    /// URSRA  <Vd>.<T>, <Vn>.<T>, #<shift>
    UrsraAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UrsraAdvsimdSelector,
    },
    /// USDOT (by element) -- A64
    /// Dot Product with unsigned and signed integers (vector, by element)
    /// USDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.4B[<index>]
    UsdotAdvsimdElt {
        q: i32,
        l: i32,
        m: i32,
        rm: Register,
        h: i32,
        rn: Register,
        rd: Register,
    },
    /// USDOT (vector) -- A64
    /// Dot Product with unsigned and signed integers (vector)
    /// USDOT  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UsdotAdvsimdVec {
        q: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// USHL -- A64
    /// Unsigned Shift Left (register)
    /// USHL  <V><d>, <V><n>, <V><m>
    /// USHL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    UshlAdvsimd {
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UshlAdvsimdSelector,
    },
    /// USHLL, USHLL2 -- A64
    /// Unsigned Shift Left Long (immediate)
    /// USHLL{2}  <Vd>.<Ta>, <Vn>.<Tb>, #<shift>
    UshllAdvsimd {
        q: i32,
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
    },
    /// USHR -- A64
    /// Unsigned Shift Right (immediate)
    /// USHR  <V><d>, <V><n>, #<shift>
    /// USHR  <Vd>.<T>, <Vn>.<T>, #<shift>
    UshrAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UshrAdvsimdSelector,
    },
    /// USMMLA (vector) -- A64
    /// Unsigned and signed 8-bit integer matrix multiply-accumulate (vector)
    /// USMMLA  <Vd>.4S, <Vn>.16B, <Vm>.16B
    UsmmlaAdvsimdVec {
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// USQADD -- A64
    /// Unsigned saturating Accumulate of Signed value
    /// USQADD  <V><d>, <V><n>
    /// USQADD  <Vd>.<T>, <Vn>.<T>
    UsqaddAdvsimd {
        size: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UsqaddAdvsimdSelector,
    },
    /// USRA -- A64
    /// Unsigned Shift Right and Accumulate (immediate)
    /// USRA  <V><d>, <V><n>, #<shift>
    /// USRA  <Vd>.<T>, <Vn>.<T>, #<shift>
    UsraAdvsimd {
        immh: i32,
        immb: i32,
        rn: Register,
        rd: Register,
        q: i32,
        class_selector: UsraAdvsimdSelector,
    },
    /// USUBL, USUBL2 -- A64
    /// Unsigned Subtract Long
    /// USUBL{2}  <Vd>.<Ta>, <Vn>.<Tb>, <Vm>.<Tb>
    UsublAdvsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// USUBW, USUBW2 -- A64
    /// Unsigned Subtract Wide
    /// USUBW{2}  <Vd>.<Ta>, <Vn>.<Ta>, <Vm>.<Tb>
    UsubwAdvsimd {
        q: i32,
        size: i32,
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
    /// UXTL, UXTL2 -- A64
    /// Unsigned extend Long
    /// UXTL{2}  <Vd>.<Ta>, <Vn>.<Tb>
    /// USHLL{2}  <Vd>.<Ta>, <Vn>.<Tb>, #0
    UxtlUshllAdvsimd {
        q: i32,
        immh: i32,
        rn: Register,
        rd: Register,
    },
    /// UZP1 -- A64
    /// Unzip vectors (primary)
    /// UZP1  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    Uzp1Advsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// UZP2 -- A64
    /// Unzip vectors (secondary)
    /// UZP2  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    Uzp2Advsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
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
    /// XAR -- A64
    /// Exclusive OR and Rotate
    /// XAR  <Vd>.2D, <Vn>.2D, <Vm>.2D, #<imm6>
    XarAdvsimd {
        rm: Register,
        imm6: i32,
        rn: Register,
        rd: Register,
    },
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
    /// XTN, XTN2 -- A64
    /// Extract Narrow
    /// XTN{2}  <Vd>.<Tb>, <Vn>.<Ta>
    XtnAdvsimd {
        q: i32,
        size: i32,
        rn: Register,
        rd: Register,
    },
    /// YIELD -- A64
    /// YIELD
    /// YIELD
    Yield,
    /// ZIP1 -- A64
    /// Zip vectors (primary)
    /// ZIP1  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    Zip1Advsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
    /// ZIP2 -- A64
    /// Zip vectors (secondary)
    /// ZIP2  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
    Zip2Advsimd {
        q: i32,
        size: i32,
        rm: Register,
        rn: Register,
        rd: Register,
    },
}
#[derive(Debug)]
pub enum AbsAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum AddAdvsimdSelector {
    Scalar,
    Vector,
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
pub enum CmeqAdvsimdRegSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmeqAdvsimdZeroSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmgeAdvsimdRegSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmgeAdvsimdZeroSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmgtAdvsimdRegSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmgtAdvsimdZeroSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmhiAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmhsAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmleAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmltAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum CmtstAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum DsbSelector {
    MemoryBarrier,
    MemoryNxsBarrier,
}

#[derive(Debug)]
pub enum DupAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum FabdAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FabsAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FacgeAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FacgtAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FaddAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FaddpAdvsimdPairSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FaddpAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmeqAdvsimdRegSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmeqAdvsimdZeroSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmgeAdvsimdRegSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmgeAdvsimdZeroSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmgtAdvsimdRegSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmgtAdvsimdZeroSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmleAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcmltAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtasAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtauAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtmsAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtmuAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtnsAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtnuAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtpsAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtpuAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtxnAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum FcvtzsAdvsimdFixSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum FcvtzsAdvsimdIntSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FcvtzuAdvsimdFixSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum FcvtzuAdvsimdIntSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FdivAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxnmAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxnmpAdvsimdPairSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxnmpAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxnmvAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxpAdvsimdPairSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxpAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmaxvAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminnmAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminnmpAdvsimdPairSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminnmpAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminnmvAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminpAdvsimdPairSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminpAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FminvAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmlaAdvsimdEltSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmlaAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmlalAdvsimdEltSelector {
    Fmlal,
    Fmlal2,
}

#[derive(Debug)]
pub enum FmlalAdvsimdVecSelector {
    Fmlal,
    Fmlal2,
}

#[derive(Debug)]
pub enum FmlsAdvsimdEltSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmlsAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmlslAdvsimdEltSelector {
    Fmlsl,
    Fmlsl2,
}

#[derive(Debug)]
pub enum FmlslAdvsimdVecSelector {
    Fmlsl,
    Fmlsl2,
}

#[derive(Debug)]
pub enum FmovAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmulAdvsimdEltSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmulAdvsimdVecSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmulxAdvsimdEltSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FmulxAdvsimdVecSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FnegAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrecpeAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrecpsAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrecpxAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrintaAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrintiAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrintmAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrintnAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrintpAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrintxAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrintzAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrsqrteAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FrsqrtsAdvsimdSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FsqrtAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum FsubAdvsimdSelector {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum Ld1AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld1AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld1rAdvsimdSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld2AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld2AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld2rAdvsimdSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld3AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld3AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld3rAdvsimdSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld4AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld4AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum Ld4rAdvsimdSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum LdpFpsimdSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
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
pub enum LdrImmFpsimdSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
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
pub enum NegAdvsimdSelector {
    Scalar,
    Vector,
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
pub enum ScvtfAdvsimdFixSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum ScvtfAdvsimdIntSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum ShlAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SliAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqabsAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqaddAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmlalAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmlalAdvsimdVecSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmlslAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmlslAdvsimdVecSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmulhAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmulhAdvsimdVecSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmullAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqdmullAdvsimdVecSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqnegAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrdmlahAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrdmlahAdvsimdVecSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrdmlshAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrdmlshAdvsimdVecSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrdmulhAdvsimdEltSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrdmulhAdvsimdVecSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrshlAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrshrnAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqrshrunAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqshlAdvsimdImmSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqshlAdvsimdRegSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqshluAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqshrnAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqshrunAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqsubAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqxtnAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SqxtunAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SriAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SrshlAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SrshrAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SrsraAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SshlAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SshrAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SsraAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum St1AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum St1AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum St2AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum St2AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum St2gSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
}

#[derive(Debug)]
pub enum St3AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum St3AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum St4AdvsimdMultSelector {
    NoOffset,
    PostIndex,
}

#[derive(Debug)]
pub enum St4AdvsimdSnglSelector {
    NoOffset,
    PostIndex,
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
pub enum StpFpsimdSelector {
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
pub enum StrImmFpsimdSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
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
pub enum SubAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum SuqaddAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UcvtfAdvsimdFixSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UcvtfAdvsimdIntSelector {
    ScalarHalfPrecision,
    ScalarSinglePrecisionAndDoublePrecision,
    VectorHalfPrecision,
    VectorSinglePrecisionAndDoublePrecision,
}

#[derive(Debug)]
pub enum UqaddAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UqrshlAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UqrshrnAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UqshlAdvsimdImmSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UqshlAdvsimdRegSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UqshrnAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UqsubAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UqxtnAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UrshlAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UrshrAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UrsraAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UshlAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UshrAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UsqaddAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum UsraAdvsimdSelector {
    Scalar,
    Vector,
}

#[derive(Debug)]
pub enum XpacSelector {
    Integer,
    System,
}
impl ArmAsm {
    pub fn encode(&self) -> u32 {
        match self {
            ArmAsm::AbsAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                AbsAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_10000_01011_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                AbsAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_10000_01011_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
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
            ArmAsm::AddAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                AddAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_1_00000_10000_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                AddAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_10000_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
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
            ArmAsm::AddhnAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_01_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AddpAdvsimdPair { size, rn, rd } => {
                0b01_0_11110_00_11000_11011_10_00000_00000
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::AddpAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_10111_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
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
            ArmAsm::AddvAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_11000_11011_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
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
            ArmAsm::AesdAdvsimd { rn, rd } => {
                0b01001110_00_10100_0010_1_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::AeseAdvsimd { rn, rd } => {
                0b01001110_00_10100_0010_0_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::AesimcAdvsimd { rn, rd } => {
                0b01001110_00_10100_0011_1_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::AesmcAdvsimd { rn, rd } => {
                0b01001110_00_10100_0011_0_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::AndAdvsimd { q, rm, rn, rd } => {
                0b0_0_0_01110_00_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
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
            ArmAsm::BcaxAdvsimd { rm, ra, rn, rd } => {
                0b110011100_01_00000_0_00000_00000_00000 | rm << 16 | ra << 10 | rn << 5 | rd << 0
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
            ArmAsm::BfcvtFloat { rn, rd } => {
                0b0_0_0_11110_01_1_000110_10000_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::BfcvtnAdvsimd { q, rn, rd } => {
                0b0_0_0_01110_10_10000_10110_10_00000_00000 | (*q as u32) << 30 | rn << 5 | rd << 0
            }
            ArmAsm::BfdotAdvsimdElt {
                q,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_01_0_0_0000_1111_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::BfdotAdvsimdVec { q, rm, rn, rd } => {
                0b0_0_1_01110_01_0_00000_1_1111_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
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
            ArmAsm::BfmlalAdvsimdElt {
                q,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_11_0_0_0000_1_1_11_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::BfmlalAdvsimdVec { q, rm, rn, rd } => {
                0b0_0_1_01110_11_0_00000_1_11_1_1_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::BfmmlaAdvsimd { rm, rn, rd } => {
                0b0_1_1_01110_01_0_00000_1_1101_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
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
            ArmAsm::BicAdvsimdImm {
                q,
                a,
                b,
                c,
                cmode,
                d,
                e,
                f,
                g,
                h,
                rd,
            } => {
                0b0_0_1_0111100000_0_0_0_0000_0_1_0_0_0_0_0_00000
                    | (*q as u32) << 30
                    | (*a as u32) << 18
                    | (*b as u32) << 17
                    | (*c as u32) << 16
                    | (*cmode as u32) << 12
                    | (*d as u32) << 9
                    | (*e as u32) << 8
                    | (*f as u32) << 7
                    | (*g as u32) << 6
                    | (*h as u32) << 5
                    | rd << 0
            }
            ArmAsm::BicAdvsimdReg { q, rm, rn, rd } => {
                0b0_0_0_01110_01_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
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
            ArmAsm::BifAdvsimd { q, rm, rn, rd } => {
                0b0_0_1_01110_11_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::BitAdvsimd { q, rm, rn, rd } => {
                0b0_0_1_01110_10_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
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
            ArmAsm::BslAdvsimd { q, rm, rn, rd } => {
                0b0_0_1_01110_01_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
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
            ArmAsm::ClsAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_10000_00100_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::ClsInt { sf, rn, rd } => {
                0b0_1_0_11010110_00000_00010_1_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::ClzAdvsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_10000_00100_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::ClzInt { sf, rn, rd } => {
                0b0_1_0_11010110_00000_00010_0_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::CmeqAdvsimdReg {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmeqAdvsimdRegSelector::Scalar => {
                    0b01_1_11110_00_1_00000_10001_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                CmeqAdvsimdRegSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_10001_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmeqAdvsimdZero {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmeqAdvsimdZeroSelector::Scalar => {
                    0b01_0_11110_00_10000_0100_1_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                CmeqAdvsimdZeroSelector::Vector => {
                    0b0_0_0_01110_00_10000_0100_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmgeAdvsimdReg {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmgeAdvsimdRegSelector::Scalar => {
                    0b01_0_11110_00_1_00000_0011_1_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                CmgeAdvsimdRegSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_0011_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmgeAdvsimdZero {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmgeAdvsimdZeroSelector::Scalar => {
                    0b01_1_11110_00_10000_0100_0_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                CmgeAdvsimdZeroSelector::Vector => {
                    0b0_0_1_01110_00_10000_0100_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmgtAdvsimdReg {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmgtAdvsimdRegSelector::Scalar => {
                    0b01_0_11110_00_1_00000_0011_0_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                CmgtAdvsimdRegSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_0011_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmgtAdvsimdZero {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmgtAdvsimdZeroSelector::Scalar => {
                    0b01_0_11110_00_10000_0100_0_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                CmgtAdvsimdZeroSelector::Vector => {
                    0b0_0_0_01110_00_10000_0100_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmhiAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmhiAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_0011_0_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                CmhiAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_0011_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmhsAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmhsAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_0011_1_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                CmhsAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_0011_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmleAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmleAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_10000_0100_1_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                CmleAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_10000_0100_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::CmltAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmltAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_10000_01010_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                CmltAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_10000_01010_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
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
            ArmAsm::CmtstAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                CmtstAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_1_00000_10001_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                CmtstAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_10001_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
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
            ArmAsm::CntAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_10000_00101_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
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
            ArmAsm::DupAdvsimdElt {
                imm5,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                DupAdvsimdEltSelector::Scalar => {
                    0b01_0_11110000_00000_0_0000_1_00000_00000
                        | truncate_imm::<_, 5>(*imm5) << 16
                        | rn << 5
                        | rd << 0
                }
                DupAdvsimdEltSelector::Vector => {
                    0b0_0_0_01110000_00000_0_0000_1_00000_00000
                        | (*q as u32) << 30
                        | truncate_imm::<_, 5>(*imm5) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::DupAdvsimdGen { q, imm5, rn, rd } => {
                0b0_0_0_01110000_00000_0_0001_1_00000_00000
                    | (*q as u32) << 30
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | rn << 5
                    | rd << 0
            }
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
            ArmAsm::Eor3Advsimd { rm, ra, rn, rd } => {
                0b110011100_00_00000_0_00000_00000_00000 | rm << 16 | ra << 10 | rn << 5 | rd << 0
            }
            ArmAsm::EorAdvsimd { q, rm, rn, rd } => {
                0b0_0_1_01110_00_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
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
            ArmAsm::ExtAdvsimd {
                q,
                rm,
                imm4,
                rn,
                rd,
            } => {
                0b0_0_101110_00_0_00000_0_0000_0_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | (*imm4 as u32) << 11
                    | rn << 5
                    | rd << 0
            }
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
            ArmAsm::FabdAdvsimd {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FabdAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_10_00000_00_010_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FabdAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_1_00000_11010_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FabdAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_10_00000_00_010_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FabdAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_1_00000_11010_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FabsAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FabsAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_111100_01111_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FabsAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_01111_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FabsFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0000_01_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FacgeAdvsimd {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FacgeAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_0_10_00000_00_10_1_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FacgeAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_1_00000_1110_1_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FacgeAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_0_10_00000_00_10_1_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FacgeAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_1_00000_1110_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FacgtAdvsimd {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FacgtAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_10_00000_00_10_1_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FacgtAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_1_00000_1110_1_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FacgtAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_10_00000_00_10_1_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FacgtAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_1_00000_1110_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FaddAdvsimd {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FaddAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_0_10_00000_00_010_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FaddAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_1_00000_11010_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FaddFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_001_0_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FaddpAdvsimdPair {
                sz,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FaddpAdvsimdPairSelector::HalfPrecision => {
                    0b01_0_11110_0_0_11000_01101_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FaddpAdvsimdPairSelector::SinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_11000_01101_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FaddpAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FaddpAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_1_01110_0_10_00000_00_010_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FaddpAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_1_00000_11010_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcaddAdvsimdVec {
                q,
                size,
                rm,
                rot,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_0_00000_1_11_0_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | (*rot as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FccmpFloat {
                ftype,
                rm,
                cond,
                rn,
                nzcv,
            } => {
                0b0_0_0_11110_00_1_00000_0000_01_00000_0_0000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | (*nzcv as u32) << 0
            }
            ArmAsm::FccmpeFloat {
                ftype,
                rm,
                cond,
                rn,
                nzcv,
            } => {
                0b0_0_0_11110_00_1_00000_0000_01_00000_1_0000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | (*nzcv as u32) << 0
            }
            ArmAsm::FcmeqAdvsimdReg {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmeqAdvsimdRegSelector::ScalarHalfPrecision => {
                    0b01_0_11110_0_10_00000_00_10_0_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FcmeqAdvsimdRegSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_0_0_1_00000_1110_0_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FcmeqAdvsimdRegSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_0_10_00000_00_10_0_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FcmeqAdvsimdRegSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_1_00000_1110_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmeqAdvsimdZero {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmeqAdvsimdZeroSelector::ScalarHalfPrecision => {
                    0b01_0_11110_1_111100_0110_1_10_00000_00000 | rn << 5 | rd << 0
                }
                FcmeqAdvsimdZeroSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_10000_0110_1_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcmeqAdvsimdZeroSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_1_111100_0110_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcmeqAdvsimdZeroSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_0110_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmgeAdvsimdReg {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmgeAdvsimdRegSelector::ScalarHalfPrecision => {
                    0b01_1_11110_0_10_00000_00_10_0_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FcmgeAdvsimdRegSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_1_00000_1110_0_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FcmgeAdvsimdRegSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_0_10_00000_00_10_0_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FcmgeAdvsimdRegSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_1_00000_1110_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmgeAdvsimdZero {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmgeAdvsimdZeroSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_111100_0110_0_10_00000_00000 | rn << 5 | rd << 0
                }
                FcmgeAdvsimdZeroSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_10000_0110_0_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcmgeAdvsimdZeroSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_111100_0110_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcmgeAdvsimdZeroSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_0110_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmgtAdvsimdReg {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmgtAdvsimdRegSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_10_00000_00_10_0_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FcmgtAdvsimdRegSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_1_00000_1110_0_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FcmgtAdvsimdRegSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_10_00000_00_10_0_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FcmgtAdvsimdRegSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_1_00000_1110_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmgtAdvsimdZero {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmgtAdvsimdZeroSelector::ScalarHalfPrecision => {
                    0b01_0_11110_1_111100_0110_0_10_00000_00000 | rn << 5 | rd << 0
                }
                FcmgtAdvsimdZeroSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_10000_0110_0_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcmgtAdvsimdZeroSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_1_111100_0110_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcmgtAdvsimdZeroSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_0110_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmlaAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                rot,
                h,
                rn,
                rd,
            } => {
                0b0_0_1_01111_00_0_0_0000_0_00_1_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*rot as u32) << 13
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcmlaAdvsimdVec {
                q,
                size,
                rm,
                rot,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_0_00000_1_10_00_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | (*rot as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcmleAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmleAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_111100_0110_1_10_00000_00000 | rn << 5 | rd << 0
                }
                FcmleAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_10000_0110_1_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcmleAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_111100_0110_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcmleAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_0110_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmltAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcmltAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_1_111100_01110_10_00000_00000 | rn << 5 | rd << 0
                }
                FcmltAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_10000_01110_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcmltAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_1_111100_01110_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcmltAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_01110_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcmpFloat { ftype, rm, rn, opc } => {
                0b0_0_0_11110_00_1_00000_00_1000_00000_00_000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | (*opc as u32) << 3
            }
            ArmAsm::FcmpeFloat { ftype, rm, rn, opc } => {
                0b0_0_0_11110_00_1_00000_00_1000_00000_00_000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | (*opc as u32) << 3
            }
            ArmAsm::FcselFloat {
                ftype,
                rm,
                cond,
                rn,
                rd,
            } => {
                0b0_0_0_11110_00_1_00000_0000_11_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | (*cond as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtFloat { ftype, opc, rn, rd } => {
                0b0_0_0_11110_00_1_0001_00_10000_00000_00000
                    | (*ftype as u32) << 22
                    | (*opc as u32) << 15
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtasAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtasAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_0_111100_11100_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtasAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_0_0_10000_11100_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtasAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_0_111100_11100_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtasAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_10000_11100_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtasFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_00_100_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtauAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtauAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_0_111100_11100_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtauAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_10000_11100_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtauAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_0_111100_11100_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtauAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_10000_11100_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtauFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_00_101_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtlAdvsimd { q, sz, rn, rd } => {
                0b0_0_0_01110_0_0_10000_10111_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtmsAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtmsAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_0_111100_1101_1_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtmsAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_0_0_10000_1101_1_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtmsAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_0_111100_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtmsAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_10000_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtmsFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_10_000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtmuAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtmuAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_0_111100_1101_1_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtmuAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_10000_1101_1_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtmuAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_0_111100_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtmuAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_10000_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtmuFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_10_001_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtnAdvsimd { q, sz, rn, rd } => {
                0b0_0_0_01110_0_0_10000_10110_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtnsAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtnsAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_0_111100_1101_0_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtnsAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_0_0_10000_1101_0_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtnsAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_0_111100_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtnsAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_10000_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtnsFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_00_000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtnuAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtnuAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_0_111100_1101_0_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtnuAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_10000_1101_0_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtnuAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_0_111100_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtnuAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_10000_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtnuFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_00_001_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtpsAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtpsAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_1_111100_1101_0_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtpsAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_10000_1101_0_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtpsAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_1_111100_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtpsAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtpsFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_01_000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtpuAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtpuAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_111100_1101_0_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtpuAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_10000_1101_0_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtpuAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_111100_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtpuAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_1101_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtpuFloat { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_01_001_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtxnAdvsimd {
                sz,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                FcvtxnAdvsimdSelector::Scalar => {
                    0b01_1_11110_0_0_10000_10110_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtxnAdvsimdSelector::Vector => {
                    0b0_0_1_01110_0_0_10000_10110_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtzsAdvsimdFix {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                FcvtzsAdvsimdFixSelector::Scalar => {
                    0b01_0_111110_0000_000_11111_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                FcvtzsAdvsimdFixSelector::Vector => {
                    0b0_0_0_011110_0000_000_11111_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtzsAdvsimdInt {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtzsAdvsimdIntSelector::ScalarHalfPrecision => {
                    0b01_0_11110_1_111100_1101_1_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtzsAdvsimdIntSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_10000_1101_1_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtzsAdvsimdIntSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_1_111100_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtzsAdvsimdIntSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtzsFloatFix {
                sf,
                ftype,
                scale,
                rn,
                rd,
            } => {
                0b0_0_0_11110_00_0_11_000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | (*scale as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtzsFloatInt { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_11_000_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtzuAdvsimdFix {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                FcvtzuAdvsimdFixSelector::Scalar => {
                    0b01_1_111110_0000_000_11111_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                FcvtzuAdvsimdFixSelector::Vector => {
                    0b0_0_1_011110_0000_000_11111_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtzuAdvsimdInt {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FcvtzuAdvsimdIntSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_111100_1101_1_10_00000_00000 | rn << 5 | rd << 0
                }
                FcvtzuAdvsimdIntSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_10000_1101_1_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FcvtzuAdvsimdIntSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_111100_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FcvtzuAdvsimdIntSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_1101_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FcvtzuFloatFix {
                sf,
                ftype,
                scale,
                rn,
                rd,
            } => {
                0b0_0_0_11110_00_0_11_001_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | (*scale as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FcvtzuFloatInt { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_11_001_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FdivAdvsimd {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FdivAdvsimdSelector::HalfPrecision => {
                    0b0_0_1_01110_0_10_00000_00_111_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FdivAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_1_00000_11111_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FdivFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_0001_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Fjcvtzs { rn, rd } => {
                0b0_0_0_11110_01_1_11_110_000000_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::FmaddFloat {
                ftype,
                rm,
                ra,
                rn,
                rd,
            } => {
                0b0_0_0_11111_00_0_00000_0_00000_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FmaxAdvsimd {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmaxAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_0_10_00000_00_110_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmaxAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_1_00000_11110_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmaxFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_01_00_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FmaxnmAdvsimd {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmaxnmAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_0_10_00000_00_000_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmaxnmAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_1_00000_11000_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmaxnmFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_01_10_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FmaxnmpAdvsimdPair {
                sz,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FmaxnmpAdvsimdPairSelector::HalfPrecision => {
                    0b01_0_11110_0_0_11000_01100_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FmaxnmpAdvsimdPairSelector::SinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_11000_01100_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmaxnmpAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmaxnmpAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_1_01110_0_10_00000_00_000_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmaxnmpAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_1_00000_11000_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmaxnmvAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmaxnmvAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_0_0_11000_01100_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FmaxnmvAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_11000_01100_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmaxpAdvsimdPair {
                sz,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FmaxpAdvsimdPairSelector::HalfPrecision => {
                    0b01_0_11110_0_0_11000_01111_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FmaxpAdvsimdPairSelector::SinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_11000_01111_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmaxpAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmaxpAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_1_01110_0_10_00000_00_110_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmaxpAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_1_00000_11110_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmaxvAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmaxvAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_0_0_11000_01111_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FmaxvAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_11000_01111_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminAdvsimd {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FminAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_10_00000_00_110_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FminAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_1_00000_11110_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_01_01_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FminnmAdvsimd {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FminnmAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_10_00000_00_000_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FminnmAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_1_00000_11000_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminnmFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_01_11_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FminnmpAdvsimdPair {
                sz,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FminnmpAdvsimdPairSelector::HalfPrecision => {
                    0b01_0_11110_1_0_11000_01100_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FminnmpAdvsimdPairSelector::SinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_11000_01100_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminnmpAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FminnmpAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_1_01110_1_10_00000_00_000_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FminnmpAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_1_00000_11000_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminnmvAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FminnmvAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_0_11000_01100_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FminnmvAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_11000_01100_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminpAdvsimdPair {
                sz,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FminpAdvsimdPairSelector::HalfPrecision => {
                    0b01_0_11110_1_0_11000_01111_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FminpAdvsimdPairSelector::SinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_11000_01111_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminpAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FminpAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_1_01110_1_10_00000_00_110_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FminpAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_1_00000_11110_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FminvAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FminvAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_0_11000_01111_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FminvAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_11000_01111_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlaAdvsimdElt {
                l,
                m,
                rm,
                h,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FmlaAdvsimdEltSelector::ScalarHalfPrecision => {
                    0b01_0_11111_00_0_0_0000_0_0_01_0_0_00000_00000
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlaAdvsimdEltSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11111_1_0_0_0_0000_0_0_01_0_0_00000_00000
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlaAdvsimdEltSelector::VectorHalfPrecision => {
                    0b0_0_0_01111_00_0_0_0000_0_0_01_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlaAdvsimdEltSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01111_1_0_0_0_0000_0_0_01_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlaAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmlaAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_0_01110_0_10_00000_00_001_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmlaAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_1_00000_11001_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlalAdvsimdElt {
                q,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FmlalAdvsimdEltSelector::Fmlal => {
                    0b0_0_0_01111_1_0_0_0_0000_0_0_00_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlalAdvsimdEltSelector::Fmlal2 => {
                    0b0_0_1_01111_1_0_0_0_0000_1_0_00_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlalAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FmlalAdvsimdVecSelector::Fmlal => {
                    0b0_0_0_01110_0_0_1_00000_1_1101_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmlalAdvsimdVecSelector::Fmlal2 => {
                    0b0_0_1_01110_0_0_1_00000_1_1001_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlsAdvsimdElt {
                l,
                m,
                rm,
                h,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FmlsAdvsimdEltSelector::ScalarHalfPrecision => {
                    0b01_0_11111_00_0_0_0000_0_1_01_0_0_00000_00000
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlsAdvsimdEltSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11111_1_0_0_0_0000_0_1_01_0_0_00000_00000
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlsAdvsimdEltSelector::VectorHalfPrecision => {
                    0b0_0_0_01111_00_0_0_0000_0_1_01_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlsAdvsimdEltSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01111_1_0_0_0_0000_0_1_01_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlsAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmlsAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_0_01110_1_10_00000_00_001_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmlsAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_1_00000_11001_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlslAdvsimdElt {
                q,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FmlslAdvsimdEltSelector::Fmlsl => {
                    0b0_0_0_01111_1_0_0_0_0000_0_1_00_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmlslAdvsimdEltSelector::Fmlsl2 => {
                    0b0_0_1_01111_1_0_0_0_0000_1_1_00_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmlslAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                class_selector,
            } => match class_selector {
                FmlslAdvsimdVecSelector::Fmlsl => {
                    0b0_0_0_01110_1_0_1_00000_1_1101_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmlslAdvsimdVecSelector::Fmlsl2 => {
                    0b0_0_1_01110_1_0_1_00000_1_1001_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmovAdvsimd {
                q,
                a,
                b,
                c,
                d,
                e,
                f,
                g,
                h,
                rd,
                op,
                class_selector,
            } => match class_selector {
                FmovAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_0111100000_0_0_0_1111_1_1_0_0_0_0_0_00000
                        | (*q as u32) << 30
                        | (*a as u32) << 18
                        | (*b as u32) << 17
                        | (*c as u32) << 16
                        | (*d as u32) << 9
                        | (*e as u32) << 8
                        | (*f as u32) << 7
                        | (*g as u32) << 6
                        | (*h as u32) << 5
                        | rd << 0
                }
                FmovAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_0111100000_0_0_0_1111_0_1_0_0_0_0_0_00000
                        | (*q as u32) << 30
                        | (*op as u32) << 29
                        | (*a as u32) << 18
                        | (*b as u32) << 17
                        | (*c as u32) << 16
                        | (*d as u32) << 9
                        | (*e as u32) << 8
                        | (*f as u32) << 7
                        | (*g as u32) << 6
                        | (*h as u32) << 5
                        | rd << 0
                }
            },
            ArmAsm::FmovFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0000_00_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
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
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | (*rmode as u32) << 19
                    | (*opcode as u32) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FmovFloatImm { ftype, imm8, rd } => {
                0b0_0_0_11110_00_1_00000000_100_00000_00000
                    | (*ftype as u32) << 22
                    | (*imm8 as u32) << 13
                    | rd << 0
            }
            ArmAsm::FmsubFloat {
                ftype,
                rm,
                ra,
                rn,
                rd,
            } => {
                0b0_0_0_11111_00_0_00000_1_00000_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FmulAdvsimdElt {
                l,
                m,
                rm,
                h,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FmulAdvsimdEltSelector::ScalarHalfPrecision => {
                    0b01_0_11111_00_0_0_0000_1001_0_0_00000_00000
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmulAdvsimdEltSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11111_1_0_0_0_0000_1001_0_0_00000_00000
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmulAdvsimdEltSelector::VectorHalfPrecision => {
                    0b0_0_0_01111_00_0_0_0000_1001_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmulAdvsimdEltSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01111_1_0_0_0_0000_1001_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmulAdvsimdVec {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FmulAdvsimdVecSelector::HalfPrecision => {
                    0b0_0_1_01110_0_10_00000_00_011_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmulAdvsimdVecSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_1_00000_11011_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmulFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_0_000_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FmulxAdvsimdElt {
                l,
                m,
                rm,
                h,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FmulxAdvsimdEltSelector::ScalarHalfPrecision => {
                    0b01_1_11111_00_0_0_0000_1001_0_0_00000_00000
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmulxAdvsimdEltSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11111_1_0_0_0_0000_1001_0_0_00000_00000
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmulxAdvsimdEltSelector::VectorHalfPrecision => {
                    0b0_0_1_01111_00_0_0_0000_1001_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                FmulxAdvsimdEltSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01111_1_0_0_0_0000_1001_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FmulxAdvsimdVec {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FmulxAdvsimdVecSelector::ScalarHalfPrecision => {
                    0b01_0_11110_0_10_00000_00_011_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FmulxAdvsimdVecSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_0_0_1_00000_11011_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmulxAdvsimdVecSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_0_10_00000_00_011_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FmulxAdvsimdVecSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_1_00000_11011_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FnegAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FnegAdvsimdSelector::HalfPrecision => {
                    0b0_0_1_01110_1_111100_01111_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FnegAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_01111_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FnegFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0000_10_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FnmaddFloat {
                ftype,
                rm,
                ra,
                rn,
                rd,
            } => {
                0b0_0_0_11111_00_1_00000_0_00000_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FnmsubFloat {
                ftype,
                rm,
                ra,
                rn,
                rd,
            } => {
                0b0_0_0_11111_00_1_00000_1_00000_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FnmulFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_1_000_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrecpeAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FrecpeAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_1_111100_11101_10_00000_00000 | rn << 5 | rd << 0
                }
                FrecpeAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_10000_11101_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FrecpeAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_1_111100_11101_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrecpeAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_11101_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrecpsAdvsimd {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FrecpsAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_0_10_00000_00_111_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FrecpsAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_0_0_1_00000_11111_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FrecpsAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_0_10_00000_00_111_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FrecpsAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_1_00000_11111_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrecpxAdvsimd {
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrecpxAdvsimdSelector::HalfPrecision => {
                    0b01_0_11110_1_111100_11111_10_00000_00000 | rn << 5 | rd << 0
                }
                FrecpxAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_10000_11111_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::Frint32xAdvsimd { q, sz, rn, rd } => {
                0b0_0_1_01110_0_0_10000_1111_0_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Frint32xFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0100_01_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Frint32zAdvsimd { q, sz, rn, rd } => {
                0b0_0_0_01110_0_0_10000_1111_0_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Frint32zFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0100_00_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Frint64xAdvsimd { q, sz, rn, rd } => {
                0b0_0_1_01110_0_0_10000_1111_1_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Frint64xFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0100_11_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Frint64zAdvsimd { q, sz, rn, rd } => {
                0b0_0_0_01110_0_0_10000_1111_1_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Frint64zFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0100_10_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrintaAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrintaAdvsimdSelector::HalfPrecision => {
                    0b0_0_1_01110_0_111100_1100_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrintaAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_10000_1100_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrintaFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001_100_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrintiAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrintiAdvsimdSelector::HalfPrecision => {
                    0b0_0_1_01110_1_111100_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrintiAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrintiFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001_111_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrintmAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrintmAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_0_111100_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrintmAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_10000_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrintmFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001_010_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrintnAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrintnAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_0_111100_1100_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrintnAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_10000_1100_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrintnFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001_000_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrintpAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrintpAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_111100_1100_0_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrintpAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_1100_0_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrintpFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001_001_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrintxAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrintxAdvsimdSelector::HalfPrecision => {
                    0b0_0_1_01110_0_111100_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrintxAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_10000_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrintxFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001_110_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrintzAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FrintzAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_111100_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrintzAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_10000_1100_1_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrintzFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_001_011_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FrsqrteAdvsimd {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FrsqrteAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_1_11110_1_111100_11101_10_00000_00000 | rn << 5 | rd << 0
                }
                FrsqrteAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_1_0_10000_11101_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                FrsqrteAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_1_111100_11101_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FrsqrteAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_11101_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FrsqrtsAdvsimd {
                rm,
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                FrsqrtsAdvsimdSelector::ScalarHalfPrecision => {
                    0b01_0_11110_1_10_00000_00_111_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
                }
                FrsqrtsAdvsimdSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_1_0_1_00000_11111_1_00000_00000
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FrsqrtsAdvsimdSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_1_10_00000_00_111_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FrsqrtsAdvsimdSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_1_00000_11111_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FsqrtAdvsimd {
                q,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FsqrtAdvsimdSelector::HalfPrecision => {
                    0b0_0_1_01110_1_111100_11111_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                FsqrtAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_1_0_10000_11111_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FsqrtFloat { ftype, rn, rd } => {
                0b0_0_0_11110_00_1_0000_11_10000_00000_00000
                    | (*ftype as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::FsubAdvsimd {
                q,
                rm,
                rn,
                rd,
                sz,
                class_selector,
            } => match class_selector {
                FsubAdvsimdSelector::HalfPrecision => {
                    0b0_0_0_01110_1_10_00000_00_010_1_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                FsubAdvsimdSelector::SinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_1_0_1_00000_11010_1_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::FsubFloat { ftype, rm, rn, rd } => {
                0b0_0_0_11110_00_1_00000_001_1_10_00000_00000
                    | (*ftype as u32) << 22
                    | rm << 16
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
            ArmAsm::InsAdvsimdElt { imm5, imm4, rn, rd } => {
                0b0_1_1_01110000_00000_0_0000_1_00000_00000
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | (*imm4 as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::InsAdvsimdGen { imm5, rn, rd } => {
                0b0_1_0_01110000_00000_0_0011_1_00000_00000
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Irg { xm, xn, xd } => {
                0b1_0_0_11010110_00000_0_0_0_1_0_0_00000_00000
                    | (*xm as u32) << 16
                    | (*xn as u32) << 5
                    | (*xd as u32) << 0
            }
            ArmAsm::Isb { crm } => 0b1101010100_0_00_011_0011_0000_1_10_11111 | (*crm as u32) << 8,
            ArmAsm::Ld1AdvsimdMult {
                q,
                opcode,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld1AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_1_000000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld1AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_1_0_00000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld1AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld1AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_1_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld1AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_1_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld1rAdvsimd {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld1rAdvsimdSelector::NoOffset => {
                    0b0_0_0011010_1_0_00000_110_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld1rAdvsimdSelector::PostIndex => {
                    0b0_0_0011011_1_0_00000_110_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld2AdvsimdMult {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld2AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_1_000000_1000_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld2AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_1_0_00000_1000_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld2AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld2AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_1_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld2AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_1_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld2rAdvsimd {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld2rAdvsimdSelector::NoOffset => {
                    0b0_0_0011010_1_1_00000_110_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld2rAdvsimdSelector::PostIndex => {
                    0b0_0_0011011_1_1_00000_110_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld3AdvsimdMult {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld3AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_1_000000_0100_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld3AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_1_0_00000_0100_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld3AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld3AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_1_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld3AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_1_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld3rAdvsimd {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld3rAdvsimdSelector::NoOffset => {
                    0b0_0_0011010_1_0_00000_111_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld3rAdvsimdSelector::PostIndex => {
                    0b0_0_0011011_1_0_00000_111_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld4AdvsimdMult {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld4AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_1_000000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld4AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_1_0_00000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld4AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld4AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_1_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld4AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_1_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::Ld4rAdvsimd {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                Ld4rAdvsimdSelector::NoOffset => {
                    0b0_0_0011010_1_1_00000_111_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                Ld4rAdvsimdSelector::PostIndex => {
                    0b0_0_0011011_1_1_00000_111_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
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
            ArmAsm::LdnpFpsimd {
                opc,
                imm7,
                rt2,
                rn,
                rt,
            } => {
                0b00_101_1_000_1_0000000_00000_00000_00000
                    | (*opc as u32) << 30
                    | truncate_imm::<_, 7>(*imm7) << 15
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
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
            ArmAsm::LdpFpsimd {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                class_selector,
            } => match class_selector {
                LdpFpsimdSelector::PostIndex => {
                    0b00_101_1_001_1_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                LdpFpsimdSelector::PreIndex => {
                    0b00_101_1_011_1_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                LdpFpsimdSelector::SignedOffset => {
                    0b00_101_1_010_1_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
            },
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
            ArmAsm::LdrImmFpsimd {
                size,
                opc,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                LdrImmFpsimdSelector::PostIndex => {
                    0b00_111_1_00_00_0_000000000_01_00000_00000
                        | (*size as u32) << 30
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrImmFpsimdSelector::PreIndex => {
                    0b00_111_1_00_00_0_000000000_11_00000_00000
                        | (*size as u32) << 30
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                LdrImmFpsimdSelector::UnsignedOffset => {
                    0b00_111_1_01_00_000000000000_00000_00000
                        | (*size as u32) << 30
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 12>(*imm12) << 10
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
            ArmAsm::LdrLitFpsimd { opc, imm19, rt } => {
                0b00_011_1_00_0000000000000000000_00000
                    | (*opc as u32) << 30
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | rt << 0
            }
            ArmAsm::LdrLitGen { opc, imm19, rt } => {
                0b00_011_0_00_0000000000000000000_00000
                    | (*opc as u32) << 30
                    | truncate_imm::<_, 19>(*imm19) << 5
                    | rt << 0
            }
            ArmAsm::LdrRegFpsimd {
                size,
                opc,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_1_00_00_1_00000_000_0_10_00000_00000
                    | (*size as u32) << 30
                    | (*opc as u32) << 22
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
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
            ArmAsm::LdurFpsimd {
                size,
                opc,
                imm9,
                rn,
                rt,
            } => {
                0b00_111_1_00_00_0_000000000_00_00000_00000
                    | (*size as u32) << 30
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
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
            ArmAsm::MlaAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_1_01111_00_0_0_0000_0_0_00_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MlaAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_10010_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MlsAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_1_01111_00_0_0_0000_0_1_00_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MlsAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_10010_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
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
            ArmAsm::MovDupAdvsimdElt { imm5, rn, rd } => {
                0b01_0_11110000_00000_0_0000_1_00000_00000
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MovInsAdvsimdElt { imm5, imm4, rn, rd } => {
                0b0_1_1_01110000_00000_0_0000_1_00000_00000
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | (*imm4 as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MovInsAdvsimdGen { imm5, rn, rd } => {
                0b0_1_0_01110000_00000_0_0011_1_00000_00000
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | rn << 5
                    | rd << 0
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
            ArmAsm::MovOrrAdvsimdReg { q, rm, rn, rd } => {
                0b0_0_0_01110_10_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
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
            ArmAsm::MovUmovAdvsimd { q, imm5, rn, rd } => {
                0b0_0_0_01110000_00000_0_01_1_1_1_00000_00000
                    | (*q as u32) << 30
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MoviAdvsimd {
                q,
                op,
                a,
                b,
                c,
                cmode,
                d,
                e,
                f,
                g,
                h,
                rd,
            } => {
                0b0_0_0_0111100000_0_0_0_0000_0_1_0_0_0_0_0_00000
                    | (*q as u32) << 30
                    | (*op as u32) << 29
                    | (*a as u32) << 18
                    | (*b as u32) << 17
                    | (*c as u32) << 16
                    | (*cmode as u32) << 12
                    | (*d as u32) << 9
                    | (*e as u32) << 8
                    | (*f as u32) << 7
                    | (*g as u32) << 6
                    | (*h as u32) << 5
                    | rd << 0
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
            ArmAsm::MulAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_00_0_0_0000_1000_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::MulAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_10011_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
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
            ArmAsm::MvnNotAdvsimd { q, rn, rd } => {
                0b0_0_1_01110_00_10000_00101_10_00000_00000 | (*q as u32) << 30 | rn << 5 | rd << 0
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
            ArmAsm::MvniAdvsimd {
                q,
                a,
                b,
                c,
                cmode,
                d,
                e,
                f,
                g,
                h,
                rd,
            } => {
                0b0_0_1_0111100000_0_0_0_0000_0_1_0_0_0_0_0_00000
                    | (*q as u32) << 30
                    | (*a as u32) << 18
                    | (*b as u32) << 17
                    | (*c as u32) << 16
                    | (*cmode as u32) << 12
                    | (*d as u32) << 9
                    | (*e as u32) << 8
                    | (*f as u32) << 7
                    | (*g as u32) << 6
                    | (*h as u32) << 5
                    | rd << 0
            }
            ArmAsm::NegAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                NegAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_10000_01011_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                NegAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_10000_01011_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
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
            ArmAsm::NotAdvsimd { q, rn, rd } => {
                0b0_0_1_01110_00_10000_00101_10_00000_00000 | (*q as u32) << 30 | rn << 5 | rd << 0
            }
            ArmAsm::OrnAdvsimd { q, rm, rn, rd } => {
                0b0_0_0_01110_11_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
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
            ArmAsm::OrrAdvsimdImm {
                q,
                a,
                b,
                c,
                cmode,
                d,
                e,
                f,
                g,
                h,
                rd,
            } => {
                0b0_0_0_0111100000_0_0_0_0000_0_1_0_0_0_0_0_00000
                    | (*q as u32) << 30
                    | (*a as u32) << 18
                    | (*b as u32) << 17
                    | (*c as u32) << 16
                    | (*cmode as u32) << 12
                    | (*d as u32) << 9
                    | (*e as u32) << 8
                    | (*f as u32) << 7
                    | (*g as u32) << 6
                    | (*h as u32) << 5
                    | rd << 0
            }
            ArmAsm::OrrAdvsimdReg { q, rm, rn, rd } => {
                0b0_0_0_01110_10_1_00000_00011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
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
            ArmAsm::PmulAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_10011_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::PmullAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_1110_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
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
            ArmAsm::RaddhnAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_01_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Rax1Advsimd { rm, rn, rd } => {
                0b11001110011_00000_1_0_00_11_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::RbitAdvsimd { q, rn, rd } => {
                0b0_0_1_01110_01_10000_00101_10_00000_00000 | (*q as u32) << 30 | rn << 5 | rd << 0
            }
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
            ArmAsm::Rev16Advsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_10000_0000_1_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Rev16Int { sf, rn, rd } => {
                0b0_1_0_11010110_00000_0000_01_00000_00000 | (*sf as u32) << 31 | rn << 5 | rd << 0
            }
            ArmAsm::Rev32Advsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_10000_0000_0_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Rev32Int { rn, rd } => {
                0b1_1_0_11010110_00000_0000_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Rev64Advsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_10000_0000_0_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
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
            ArmAsm::RshrnAdvsimd {
                q,
                immh,
                immb,
                rn,
                rd,
            } => {
                0b0_0_0_011110_0000_000_1000_1_1_00000_00000
                    | (*q as u32) << 30
                    | (*immh as u32) << 19
                    | (*immb as u32) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::RsubhnAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_01_1_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SabaAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_0111_1_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SabalAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_01_0_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SabdAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_0111_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SabdlAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_01_1_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SadalpAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_10000_00_1_10_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SaddlAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_00_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SaddlpAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_10000_00_0_10_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SaddlvAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_11000_00011_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SaddwAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_00_0_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
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
            ArmAsm::ScvtfAdvsimdFix {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                ScvtfAdvsimdFixSelector::Scalar => {
                    0b01_0_111110_0000_000_11100_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                ScvtfAdvsimdFixSelector::Vector => {
                    0b0_0_0_011110_0000_000_11100_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::ScvtfAdvsimdInt {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                ScvtfAdvsimdIntSelector::ScalarHalfPrecision => {
                    0b01_0_11110_0_111100_11101_10_00000_00000 | rn << 5 | rd << 0
                }
                ScvtfAdvsimdIntSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_0_11110_0_0_10000_11101_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                ScvtfAdvsimdIntSelector::VectorHalfPrecision => {
                    0b0_0_0_01110_0_111100_11101_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                ScvtfAdvsimdIntSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_0_01110_0_0_10000_11101_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::ScvtfFloatFix {
                sf,
                ftype,
                scale,
                rn,
                rd,
            } => {
                0b0_0_0_11110_00_0_00_010_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | (*scale as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::ScvtfFloatInt { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_00_010_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
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
            ArmAsm::SdotAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_00_0_0_0000_1110_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SdotAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_0_00000_1_0010_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
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
            ArmAsm::Sha1cAdvsimd { rm, rn, rd } => {
                0b01011110_00_0_00000_0_000_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha1hAdvsimd { rn, rd } => {
                0b01011110_00_10100_00000_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Sha1mAdvsimd { rm, rn, rd } => {
                0b01011110_00_0_00000_0_010_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha1pAdvsimd { rm, rn, rd } => {
                0b01011110_00_0_00000_0_001_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha1su0Advsimd { rm, rn, rd } => {
                0b01011110_00_0_00000_0_011_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha1su1Advsimd { rn, rd } => {
                0b01011110_00_10100_00001_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Sha256h2Advsimd { rm, rn, rd } => {
                0b01011110_00_0_00000_0_10_1_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha256hAdvsimd { rm, rn, rd } => {
                0b01011110_00_0_00000_0_10_0_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha256su0Advsimd { rn, rd } => {
                0b01011110_00_10100_00010_10_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Sha256su1Advsimd { rm, rn, rd } => {
                0b01011110_00_0_00000_0_110_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha512h2Advsimd { rm, rn, rd } => {
                0b11001110011_00000_1_0_00_01_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha512hAdvsimd { rm, rn, rd } => {
                0b11001110011_00000_1_0_00_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sha512su0Advsimd { rn, rd } => {
                0b11001110110000001000_00_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Sha512su1Advsimd { rm, rn, rd } => {
                0b11001110011_00000_1_0_00_10_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::ShaddAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_00000_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::ShlAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                ShlAdvsimdSelector::Scalar => {
                    0b01_0_111110_0000_000_01010_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                ShlAdvsimdSelector::Vector => {
                    0b0_0_0_011110_0000_000_01010_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::ShllAdvsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_10000_10011_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::ShrnAdvsimd {
                q,
                immh,
                immb,
                rn,
                rd,
            } => {
                0b0_0_0_011110_0000_000_1000_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*immh as u32) << 19
                    | (*immb as u32) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::ShsubAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_00100_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SliAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SliAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_01010_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SliAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_01010_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::Sm3partw1Advsimd { rm, rn, rd } => {
                0b11001110011_00000_1_1_00_00_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sm3partw2Advsimd { rm, rn, rd } => {
                0b11001110011_00000_1_1_00_01_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Sm3ss1Advsimd { rm, ra, rn, rd } => {
                0b110011100_10_00000_0_00000_00000_00000 | rm << 16 | ra << 10 | rn << 5 | rd << 0
            }
            ArmAsm::Sm3tt1aAdvsimd { rm, imm2, rn, rd } => {
                0b11001110010_00000_10_00_00_00000_00000
                    | rm << 16
                    | (*imm2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sm3tt1bAdvsimd { rm, imm2, rn, rd } => {
                0b11001110010_00000_10_00_01_00000_00000
                    | rm << 16
                    | (*imm2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sm3tt2aAdvsimd { rm, imm2, rn, rd } => {
                0b11001110010_00000_10_00_10_00000_00000
                    | rm << 16
                    | (*imm2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sm3tt2bAdvsimd { rm, imm2, rn, rd } => {
                0b11001110010_00000_10_00_11_00000_00000
                    | rm << 16
                    | (*imm2 as u32) << 12
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Sm4eAdvsimd { rn, rd } => {
                0b11001110110000001000_01_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::Sm4ekeyAdvsimd { rm, rn, rd } => {
                0b11001110011_00000_1_1_00_10_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::Smaddl { rm, ra, rn, rd } => {
                0b1_00_11011_0_01_00000_0_00000_00000_00000
                    | rm << 16
                    | ra << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmaxAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_0110_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmaxpAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_1010_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmaxvAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_11000_0_1010_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Smc { imm16 } => 0b11010100_000_0000000000000000_000_11 | (*imm16 as u32) << 5,
            ArmAsm::SminAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_0110_1_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SminpAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_1010_1_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SminvAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_11000_1_1010_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmlalAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_00_0_0_0000_0_0_10_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmlalAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_10_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmlslAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_00_0_0_0000_0_1_10_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmlslAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_10_1_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmmlaAdvsimdVec { rm, rn, rd } => {
                0b0_1_0_01110_10_0_00000_1_010_0_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::SmneglSmsubl { rm, rn, rd } => {
                0b1_00_11011_0_01_00000_1_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::SmovAdvsimd { q, imm5, rn, rd } => {
                0b0_0_0_01110000_00000_0_01_0_1_1_00000_00000
                    | (*q as u32) << 30
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | rn << 5
                    | rd << 0
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
            ArmAsm::SmullAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_00_0_0_0000_1010_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmullAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_1_1_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SmullSmaddl { rm, rn, rd } => {
                0b1_00_11011_0_01_00000_0_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::SqabsAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqabsAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_10000_00111_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                SqabsAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_10000_00111_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqaddAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqaddAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_1_00000_00001_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqaddAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_00001_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmlalAdvsimdElt {
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmlalAdvsimdEltSelector::Scalar => {
                    0b01_0_11111_00_0_0_0000_0_0_11_0_0_00000_00000
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                SqdmlalAdvsimdEltSelector::Vector => {
                    0b0_0_0_01111_00_0_0_0000_0_0_11_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmlalAdvsimdVec {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmlalAdvsimdVecSelector::Scalar => {
                    0b01_0_11110_00_1_00000_10_0_1_00_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqdmlalAdvsimdVecSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_10_0_1_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmlslAdvsimdElt {
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmlslAdvsimdEltSelector::Scalar => {
                    0b01_0_11111_00_0_0_0000_0_1_11_0_0_00000_00000
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                SqdmlslAdvsimdEltSelector::Vector => {
                    0b0_0_0_01111_00_0_0_0000_0_1_11_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmlslAdvsimdVec {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmlslAdvsimdVecSelector::Scalar => {
                    0b01_0_11110_00_1_00000_10_1_1_00_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqdmlslAdvsimdVecSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_10_1_1_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmulhAdvsimdElt {
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmulhAdvsimdEltSelector::Scalar => {
                    0b01_0_11111_00_0_0_0000_110_0_0_0_00000_00000
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                SqdmulhAdvsimdEltSelector::Vector => {
                    0b0_0_0_01111_00_0_0_0000_110_0_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmulhAdvsimdVec {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmulhAdvsimdVecSelector::Scalar => {
                    0b01_0_11110_00_1_00000_10110_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqdmulhAdvsimdVecSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_10110_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmullAdvsimdElt {
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmullAdvsimdEltSelector::Scalar => {
                    0b01_0_11111_00_0_0_0000_1011_0_0_00000_00000
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                SqdmullAdvsimdEltSelector::Vector => {
                    0b0_0_0_01111_00_0_0_0000_1011_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqdmullAdvsimdVec {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqdmullAdvsimdVecSelector::Scalar => {
                    0b01_0_11110_00_1_00000_1101_00_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqdmullAdvsimdVecSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_1101_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqnegAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqnegAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_10000_00111_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                SqnegAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_10000_00111_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrdmlahAdvsimdElt {
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrdmlahAdvsimdEltSelector::Scalar => {
                    0b01_1_11111_00_0_0_0000_11_0_1_0_0_00000_00000
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                SqrdmlahAdvsimdEltSelector::Vector => {
                    0b0_0_1_01111_00_0_0_0000_11_0_1_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrdmlahAdvsimdVec {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrdmlahAdvsimdVecSelector::Scalar => {
                    0b01_1_11110_00_0_00000_1_000_0_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqrdmlahAdvsimdVecSelector::Vector => {
                    0b0_0_1_01110_00_0_00000_1_000_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrdmlshAdvsimdElt {
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrdmlshAdvsimdEltSelector::Scalar => {
                    0b01_1_11111_00_0_0_0000_11_1_1_0_0_00000_00000
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                SqrdmlshAdvsimdEltSelector::Vector => {
                    0b0_0_1_01111_00_0_0_0000_11_1_1_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrdmlshAdvsimdVec {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrdmlshAdvsimdVecSelector::Scalar => {
                    0b01_1_11110_00_0_00000_1_000_1_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqrdmlshAdvsimdVecSelector::Vector => {
                    0b0_0_1_01110_00_0_00000_1_000_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrdmulhAdvsimdElt {
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrdmulhAdvsimdEltSelector::Scalar => {
                    0b01_0_11111_00_0_0_0000_110_1_0_0_00000_00000
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
                SqrdmulhAdvsimdEltSelector::Vector => {
                    0b0_0_0_01111_00_0_0_0000_110_1_0_0_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | (*l as u32) << 21
                        | (*m as u32) << 20
                        | rm << 16
                        | (*h as u32) << 11
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrdmulhAdvsimdVec {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrdmulhAdvsimdVecSelector::Scalar => {
                    0b01_1_11110_00_1_00000_10110_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqrdmulhAdvsimdVecSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_10110_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrshlAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrshlAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_1_00000_010_1_1_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqrshlAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_010_1_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrshrnAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrshrnAdvsimdSelector::Scalar => {
                    0b01_0_111110_0000_000_1001_1_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SqrshrnAdvsimdSelector::Vector => {
                    0b0_0_0_011110_0000_000_1001_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqrshrunAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqrshrunAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_1000_1_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SqrshrunAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_1000_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqshlAdvsimdImm {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqshlAdvsimdImmSelector::Scalar => {
                    0b01_0_111110_0000_000_011_1_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SqshlAdvsimdImmSelector::Vector => {
                    0b0_0_0_011110_0000_000_011_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqshlAdvsimdReg {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqshlAdvsimdRegSelector::Scalar => {
                    0b01_0_11110_00_1_00000_010_0_1_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqshlAdvsimdRegSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_010_0_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqshluAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqshluAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_011_0_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SqshluAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_011_0_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqshrnAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqshrnAdvsimdSelector::Scalar => {
                    0b01_0_111110_0000_000_1001_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SqshrnAdvsimdSelector::Vector => {
                    0b0_0_0_011110_0000_000_1001_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqshrunAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqshrunAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_1000_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SqshrunAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_1000_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqsubAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqsubAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_1_00000_00101_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SqsubAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_00101_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqxtnAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqxtnAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_10000_10100_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                SqxtnAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_10000_10100_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SqxtunAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SqxtunAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_10000_10010_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                SqxtunAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_10000_10010_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SrhaddAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_00010_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SriAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SriAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_01000_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SriAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_01000_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SrshlAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SrshlAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_1_00000_010_1_0_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SrshlAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_010_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SrshrAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SrshrAdvsimdSelector::Scalar => {
                    0b01_0_111110_0000_000_00_1_0_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SrshrAdvsimdSelector::Vector => {
                    0b0_0_0_011110_0000_000_00_1_0_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SrsraAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SrsraAdvsimdSelector::Scalar => {
                    0b01_0_111110_0000_000_00_1_1_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SrsraAdvsimdSelector::Vector => {
                    0b0_0_0_011110_0000_000_00_1_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SsbbDsb {} => 0b1101010100_0_00_011_0011_0000_1_00_11111,
            ArmAsm::SshlAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SshlAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_1_00000_010_0_0_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SshlAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_1_00000_010_0_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SshllAdvsimd {
                q,
                immh,
                immb,
                rn,
                rd,
            } => {
                0b0_0_0_011110_0000_000_10100_1_00000_00000
                    | (*q as u32) << 30
                    | (*immh as u32) << 19
                    | (*immb as u32) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SshrAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SshrAdvsimdSelector::Scalar => {
                    0b01_0_111110_0000_000_00_0_0_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SshrAdvsimdSelector::Vector => {
                    0b0_0_0_011110_0000_000_00_0_0_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SsraAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SsraAdvsimdSelector::Scalar => {
                    0b01_0_111110_0000_000_00_0_1_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                SsraAdvsimdSelector::Vector => {
                    0b0_0_0_011110_0000_000_00_0_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::SsublAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_00_1_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SsubwAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_00_1_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::St1AdvsimdMult {
                q,
                opcode,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St1AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_0_000000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St1AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_0_0_00000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::St1AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St1AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_0_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St1AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_0_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::St2AdvsimdMult {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St2AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_0_000000_1000_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St2AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_0_0_00000_1000_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::St2AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St2AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_0_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St2AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_0_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
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
            ArmAsm::St3AdvsimdMult {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St3AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_0_000000_0100_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St3AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_0_0_00000_0100_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::St3AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St3AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_0_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St3AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_0_0_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::St4AdvsimdMult {
                q,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St4AdvsimdMultSelector::NoOffset => {
                    0b0_0_0011000_0_000000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St4AdvsimdMultSelector::PostIndex => {
                    0b0_0_0011001_0_0_00000_0000_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
            },
            ArmAsm::St4AdvsimdSngl {
                q,
                opcode,
                s,
                size,
                rn,
                rt,
                rm,
                class_selector,
            } => match class_selector {
                St4AdvsimdSnglSelector::NoOffset => {
                    0b0_0_0011010_0_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
                }
                St4AdvsimdSnglSelector::PostIndex => {
                    0b0_0_0011011_0_1_00000_000_0_00_00000_00000
                        | (*q as u32) << 30
                        | rm << 16
                        | (*opcode as u32) << 13
                        | (*s as u32) << 12
                        | (*size as u32) << 10
                        | rn << 5
                        | rt << 0
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
            ArmAsm::StnpFpsimd {
                opc,
                imm7,
                rt2,
                rn,
                rt,
            } => {
                0b00_101_1_000_0_0000000_00000_00000_00000
                    | (*opc as u32) << 30
                    | truncate_imm::<_, 7>(*imm7) << 15
                    | rt2 << 10
                    | rn << 5
                    | rt << 0
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
            ArmAsm::StpFpsimd {
                opc,
                imm7,
                rt2,
                rn,
                rt,
                class_selector,
            } => match class_selector {
                StpFpsimdSelector::PostIndex => {
                    0b00_101_1_001_0_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                StpFpsimdSelector::PreIndex => {
                    0b00_101_1_011_0_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
                StpFpsimdSelector::SignedOffset => {
                    0b00_101_1_010_0_0000000_00000_00000_00000
                        | (*opc as u32) << 30
                        | truncate_imm::<_, 7>(*imm7) << 15
                        | rt2 << 10
                        | rn << 5
                        | rt << 0
                }
            },
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
            ArmAsm::StrImmFpsimd {
                size,
                opc,
                imm9,
                rn,
                rt,
                imm12,
                class_selector,
            } => match class_selector {
                StrImmFpsimdSelector::PostIndex => {
                    0b00_111_1_00_00_0_000000000_01_00000_00000
                        | (*size as u32) << 30
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrImmFpsimdSelector::PreIndex => {
                    0b00_111_1_00_00_0_000000000_11_00000_00000
                        | (*size as u32) << 30
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 9>(*imm9) << 12
                        | rn << 5
                        | rt << 0
                }
                StrImmFpsimdSelector::UnsignedOffset => {
                    0b00_111_1_01_00_000000000000_00000_00000
                        | (*size as u32) << 30
                        | (*opc as u32) << 22
                        | truncate_imm::<_, 12>(*imm12) << 10
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
            ArmAsm::StrRegFpsimd {
                size,
                opc,
                rm,
                option,
                s,
                rn,
                rt,
            } => {
                0b00_111_1_00_00_1_00000_000_0_10_00000_00000
                    | (*size as u32) << 30
                    | (*opc as u32) << 22
                    | rm << 16
                    | (*option as u32) << 13
                    | (*s as u32) << 12
                    | rn << 5
                    | rt << 0
            }
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
            ArmAsm::SturFpsimd {
                size,
                opc,
                imm9,
                rn,
                rt,
            } => {
                0b00_111_1_00_00_0_000000000_00_00000_00000
                    | (*size as u32) << 30
                    | (*opc as u32) << 22
                    | truncate_imm::<_, 9>(*imm9) << 12
                    | rn << 5
                    | rt << 0
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
            ArmAsm::SubAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SubAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_10000_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                SubAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_10000_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
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
            ArmAsm::SubhnAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_0_01110_00_1_00000_01_1_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
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
            ArmAsm::SudotAdvsimdElt {
                q,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_0_0_0_0_0000_1111_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::SuqaddAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                SuqaddAdvsimdSelector::Scalar => {
                    0b01_0_11110_00_10000_00011_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                SuqaddAdvsimdSelector::Vector => {
                    0b0_0_0_01110_00_10000_00011_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
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
            ArmAsm::SxtlSshllAdvsimd { q, immh, rn, rd } => {
                0b0_0_0_011110_0000_000_10100_1_00000_00000
                    | (*q as u32) << 30
                    | (*immh as u32) << 19
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
            ArmAsm::TblAdvsimd { q, rm, len, rn, rd } => {
                0b0_0_001110_00_0_00000_0_00_0_00_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | (*len as u32) << 13
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Tbnz { b5, b40, imm14, rt } => {
                0b0_011011_1_00000_00000000000000_00000
                    | (*b5 as u32) << 31
                    | (*b40 as u32) << 19
                    | truncate_imm::<_, 14>(*imm14) << 5
                    | rt << 0
            }
            ArmAsm::TbxAdvsimd { q, rm, len, rn, rd } => {
                0b0_0_001110_00_0_00000_0_00_1_00_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | (*len as u32) << 13
                    | rn << 5
                    | rd << 0
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
            ArmAsm::Trn1Advsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_001110_00_0_00000_0_0_10_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Trn2Advsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_001110_00_0_00000_0_1_10_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
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
            ArmAsm::UabaAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_0111_1_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UabalAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_01_0_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UabdAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_0111_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UabdlAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_01_1_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UadalpAdvsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_10000_00_1_10_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UaddlAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_00_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UaddlpAdvsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_10000_00_0_10_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UaddlvAdvsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_11000_00011_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UaddwAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_00_0_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
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
            ArmAsm::UcvtfAdvsimdFix {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UcvtfAdvsimdFixSelector::Scalar => {
                    0b01_1_111110_0000_000_11100_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UcvtfAdvsimdFixSelector::Vector => {
                    0b0_0_1_011110_0000_000_11100_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UcvtfAdvsimdInt {
                rn,
                rd,
                sz,
                q,
                class_selector,
            } => match class_selector {
                UcvtfAdvsimdIntSelector::ScalarHalfPrecision => {
                    0b01_1_11110_0_111100_11101_10_00000_00000 | rn << 5 | rd << 0
                }
                UcvtfAdvsimdIntSelector::ScalarSinglePrecisionAndDoublePrecision => {
                    0b01_1_11110_0_0_10000_11101_10_00000_00000
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                UcvtfAdvsimdIntSelector::VectorHalfPrecision => {
                    0b0_0_1_01110_0_111100_11101_10_00000_00000
                        | (*q as u32) << 30
                        | rn << 5
                        | rd << 0
                }
                UcvtfAdvsimdIntSelector::VectorSinglePrecisionAndDoublePrecision => {
                    0b0_0_1_01110_0_0_10000_11101_10_00000_00000
                        | (*q as u32) << 30
                        | (*sz as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UcvtfFloatFix {
                sf,
                ftype,
                scale,
                rn,
                rd,
            } => {
                0b0_0_0_11110_00_0_00_011_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
                    | (*scale as u32) << 10
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UcvtfFloatInt { sf, ftype, rn, rd } => {
                0b0_0_0_11110_00_1_00_011_000000_00000_00000
                    | (*sf as u32) << 31
                    | (*ftype as u32) << 22
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
            ArmAsm::UdotAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_1_01111_00_0_0_0000_1110_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UdotAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_0_00000_1_0010_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UhaddAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_00000_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UhsubAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_00100_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
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
            ArmAsm::UmaxAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_0110_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmaxpAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_1010_0_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmaxvAdvsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_11000_0_1010_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UminAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_0110_1_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UminpAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_1010_1_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UminvAdvsimd { q, size, rn, rd } => {
                0b0_0_1_01110_00_11000_1_1010_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmlalAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_1_01111_00_0_0_0000_0_0_10_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmlalAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_10_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmlslAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_1_01111_00_0_0_0000_0_1_10_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmlslAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_10_1_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmmlaAdvsimdVec { rm, rn, rd } => {
                0b0_1_1_01110_10_0_00000_1_010_0_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::UmneglUmsubl { rm, rn, rd } => {
                0b1_00_11011_1_01_00000_1_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::UmovAdvsimd { q, imm5, rn, rd } => {
                0b0_0_0_01110000_00000_0_01_1_1_1_00000_00000
                    | (*q as u32) << 30
                    | truncate_imm::<_, 5>(*imm5) << 16
                    | rn << 5
                    | rd << 0
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
            ArmAsm::UmullAdvsimdElt {
                q,
                size,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_1_01111_00_0_0_0000_1010_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmullAdvsimdVec {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_1_1_0_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UmullUmaddl { rm, rn, rd } => {
                0b1_00_11011_1_01_00000_0_11111_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::UqaddAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqaddAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_00001_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                UqaddAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_00001_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UqrshlAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqrshlAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_010_1_1_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                UqrshlAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_010_1_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UqrshrnAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqrshrnAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_1001_1_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UqrshrnAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_1001_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UqshlAdvsimdImm {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqshlAdvsimdImmSelector::Scalar => {
                    0b01_1_111110_0000_000_011_1_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UqshlAdvsimdImmSelector::Vector => {
                    0b0_0_1_011110_0000_000_011_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UqshlAdvsimdReg {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqshlAdvsimdRegSelector::Scalar => {
                    0b01_1_11110_00_1_00000_010_0_1_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                UqshlAdvsimdRegSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_010_0_1_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UqshrnAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqshrnAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_1001_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UqshrnAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_1001_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UqsubAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqsubAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_00101_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                UqsubAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_00101_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UqxtnAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UqxtnAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_10000_10100_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                UqxtnAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_10000_10100_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UrecpeAdvsimd { q, sz, rn, rd } => {
                0b0_0_0_01110_1_0_10000_11100_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UrhaddAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_00010_1_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UrshlAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UrshlAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_010_1_0_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                UrshlAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_010_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UrshrAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UrshrAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_00_1_0_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UrshrAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_00_1_0_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UrsqrteAdvsimd { q, sz, rn, rd } => {
                0b0_0_1_01110_1_0_10000_11100_10_00000_00000
                    | (*q as u32) << 30
                    | (*sz as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UrsraAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UrsraAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_00_1_1_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UrsraAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_00_1_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UsdotAdvsimdElt {
                q,
                l,
                m,
                rm,
                h,
                rn,
                rd,
            } => {
                0b0_0_0_01111_1_0_0_0_0000_1111_0_0_00000_00000
                    | (*q as u32) << 30
                    | (*l as u32) << 21
                    | (*m as u32) << 20
                    | rm << 16
                    | (*h as u32) << 11
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UsdotAdvsimdVec { q, rm, rn, rd } => {
                0b0_0_0_01110_10_0_00000_1_0011_1_00000_00000
                    | (*q as u32) << 30
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UshlAdvsimd {
                size,
                rm,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UshlAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_1_00000_010_0_0_1_00000_00000
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
                UshlAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_1_00000_010_0_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rm << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UshllAdvsimd {
                q,
                immh,
                immb,
                rn,
                rd,
            } => {
                0b0_0_1_011110_0000_000_10100_1_00000_00000
                    | (*q as u32) << 30
                    | (*immh as u32) << 19
                    | (*immb as u32) << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UshrAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UshrAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_00_0_0_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UshrAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_00_0_0_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UsmmlaAdvsimdVec { rm, rn, rd } => {
                0b0_1_0_01110_10_0_00000_1_010_1_1_00000_00000 | rm << 16 | rn << 5 | rd << 0
            }
            ArmAsm::UsqaddAdvsimd {
                size,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UsqaddAdvsimdSelector::Scalar => {
                    0b01_1_11110_00_10000_00011_10_00000_00000
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
                UsqaddAdvsimdSelector::Vector => {
                    0b0_0_1_01110_00_10000_00011_10_00000_00000
                        | (*q as u32) << 30
                        | (*size as u32) << 22
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UsraAdvsimd {
                immh,
                immb,
                rn,
                rd,
                q,
                class_selector,
            } => match class_selector {
                UsraAdvsimdSelector::Scalar => {
                    0b01_1_111110_0000_000_00_0_1_0_1_00000_00000
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
                UsraAdvsimdSelector::Vector => {
                    0b0_0_1_011110_0000_000_00_0_1_0_1_00000_00000
                        | (*q as u32) << 30
                        | (*immh as u32) << 19
                        | (*immb as u32) << 16
                        | rn << 5
                        | rd << 0
                }
            },
            ArmAsm::UsublAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_00_1_0_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UsubwAdvsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_1_01110_00_1_00000_00_1_1_00_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::UxtbUbfm { rn, rd } => {
                0b0_10_100110_0_000000_000111_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::UxthUbfm { rn, rd } => {
                0b0_10_100110_0_000000_001111_00000_00000 | rn << 5 | rd << 0
            }
            ArmAsm::UxtlUshllAdvsimd { q, immh, rn, rd } => {
                0b0_0_1_011110_0000_000_10100_1_00000_00000
                    | (*q as u32) << 30
                    | (*immh as u32) << 19
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Uzp1Advsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_001110_00_0_00000_0_0_01_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Uzp2Advsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_001110_00_0_00000_0_1_01_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Wfe {} => 0b1101010100_0_00_011_0010_0000_010_11111,
            ArmAsm::Wfet { rd } => 0b11010101000000110001_0000_000_00000 | rd << 0,
            ArmAsm::Wfi {} => 0b1101010100_0_00_011_0010_0000_011_11111,
            ArmAsm::Wfit { rd } => 0b11010101000000110001_0000_001_00000 | rd << 0,
            ArmAsm::Xaflag {} => 0b1101010100_0_00_000_0100_0000_001_11111,
            ArmAsm::XarAdvsimd { rm, imm6, rn, rd } => {
                0b11001110100_00000_000000_00000_00000
                    | rm << 16
                    | truncate_imm::<_, 6>(*imm6) << 10
                    | rn << 5
                    | rd << 0
            }
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
            ArmAsm::XtnAdvsimd { q, size, rn, rd } => {
                0b0_0_0_01110_00_10000_10010_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Yield {} => 0b1101010100_0_00_011_0010_0000_001_11111,
            ArmAsm::Zip1Advsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_001110_00_0_00000_0_0_11_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
            ArmAsm::Zip2Advsimd {
                q,
                size,
                rm,
                rn,
                rd,
            } => {
                0b0_0_001110_00_0_00000_0_1_11_10_00000_00000
                    | (*q as u32) << 30
                    | (*size as u32) << 22
                    | rm << 16
                    | rn << 5
                    | rd << 0
            }
        }
    }
}
