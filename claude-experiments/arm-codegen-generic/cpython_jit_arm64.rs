
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



/// ADD (shifted register) -- A64
/// Add (shifted register)
/// ADD  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
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

/// B.cond -- A64
/// Branch conditionally
/// B.<cond>  <label>
pub fn bcond(imm19: i32, cond: i32) -> u32 {
    let mut result = 0b0101010_0_0000000000000000000_0_0000;
    result |= truncate_imm::<_, 19>(imm19) << 5;
    result |= (cond as u32) << 0;
    result
}

/// CBNZ -- A64
/// Compare and Branch on Nonzero
/// CBNZ  <Wt>, <label>
/// CBNZ  <Xt>, <label>
pub fn cbnz(sf: i32, imm19: i32, rt: Register) -> u32 {
    let mut result = 0b0_011010_1_0000000000000000000_00000;
    result |= (sf as u32) << 31;
    result |= truncate_imm::<_, 19>(imm19) << 5;
    result |= (rt.encode() as u32) << 0;
    result
}

/// CMP (shifted register) -- A64
/// Compare (shifted register)
/// CMP  <Wn>, <Wm>{, <shift> #<amount>}
/// SUBS WZR, <Wn>, <Wm> {, <shift> #<amount>}
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

/// SUB (shifted register) -- A64
/// Subtract (shifted register)
/// SUB  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
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





} // namespace arm_asm
