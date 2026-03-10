use std::ops::Shl;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RegSize {
    W32,
    X64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Arm64Reg {
    pub index: u8,
    pub size: RegSize,
}

impl Arm64Reg {
    pub const fn new(index: u8, size: RegSize) -> Self {
        Self { index, size }
    }

    pub fn sf(&self) -> i32 {
        match self.size {
            RegSize::W32 => 0,
            RegSize::X64 => 1,
        }
    }

    pub fn encode(&self) -> u8 {
        self.index
    }

    pub fn from_index(index: usize) -> Arm64Reg {
        Arm64Reg {
            index: index as u8,
            size: RegSize::X64,
        }
    }

    /// Returns the size encoding for load/store instructions (2 = 32-bit, 3 = 64-bit).
    pub fn size_bits(&self) -> i32 {
        match self.size {
            RegSize::W32 => 2,
            RegSize::X64 => 3,
        }
    }

    /// Returns the opc for LDP/STP (0 = 32-bit, 2 = 64-bit).
    pub fn opc(&self) -> i32 {
        match self.size {
            RegSize::W32 => 0,
            RegSize::X64 => 2,
        }
    }
}

impl Shl<u32> for &Arm64Reg {
    type Output = u32;
    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

impl Shl<u32> for Arm64Reg {
    type Output = u32;
    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

// 64-bit registers
pub const X0: Arm64Reg = Arm64Reg::new(0, RegSize::X64);
pub const X1: Arm64Reg = Arm64Reg::new(1, RegSize::X64);
pub const X2: Arm64Reg = Arm64Reg::new(2, RegSize::X64);
pub const X3: Arm64Reg = Arm64Reg::new(3, RegSize::X64);
pub const X4: Arm64Reg = Arm64Reg::new(4, RegSize::X64);
pub const X5: Arm64Reg = Arm64Reg::new(5, RegSize::X64);
pub const X6: Arm64Reg = Arm64Reg::new(6, RegSize::X64);
pub const X7: Arm64Reg = Arm64Reg::new(7, RegSize::X64);
pub const X8: Arm64Reg = Arm64Reg::new(8, RegSize::X64);
pub const X9: Arm64Reg = Arm64Reg::new(9, RegSize::X64);
pub const X10: Arm64Reg = Arm64Reg::new(10, RegSize::X64);
pub const X11: Arm64Reg = Arm64Reg::new(11, RegSize::X64);
pub const X12: Arm64Reg = Arm64Reg::new(12, RegSize::X64);
pub const X13: Arm64Reg = Arm64Reg::new(13, RegSize::X64);
pub const X14: Arm64Reg = Arm64Reg::new(14, RegSize::X64);
pub const X15: Arm64Reg = Arm64Reg::new(15, RegSize::X64);
pub const X16: Arm64Reg = Arm64Reg::new(16, RegSize::X64);
pub const X17: Arm64Reg = Arm64Reg::new(17, RegSize::X64);
pub const X18: Arm64Reg = Arm64Reg::new(18, RegSize::X64);
pub const X19: Arm64Reg = Arm64Reg::new(19, RegSize::X64);
pub const X20: Arm64Reg = Arm64Reg::new(20, RegSize::X64);
pub const X21: Arm64Reg = Arm64Reg::new(21, RegSize::X64);
pub const X22: Arm64Reg = Arm64Reg::new(22, RegSize::X64);
pub const X23: Arm64Reg = Arm64Reg::new(23, RegSize::X64);
pub const X24: Arm64Reg = Arm64Reg::new(24, RegSize::X64);
pub const X25: Arm64Reg = Arm64Reg::new(25, RegSize::X64);
pub const X26: Arm64Reg = Arm64Reg::new(26, RegSize::X64);
pub const X27: Arm64Reg = Arm64Reg::new(27, RegSize::X64);
pub const X28: Arm64Reg = Arm64Reg::new(28, RegSize::X64);
pub const X29: Arm64Reg = Arm64Reg::new(29, RegSize::X64);
pub const X30: Arm64Reg = Arm64Reg::new(30, RegSize::X64);

// 32-bit registers
pub const W0: Arm64Reg = Arm64Reg::new(0, RegSize::W32);
pub const W1: Arm64Reg = Arm64Reg::new(1, RegSize::W32);
pub const W2: Arm64Reg = Arm64Reg::new(2, RegSize::W32);
pub const W3: Arm64Reg = Arm64Reg::new(3, RegSize::W32);
pub const W4: Arm64Reg = Arm64Reg::new(4, RegSize::W32);
pub const W5: Arm64Reg = Arm64Reg::new(5, RegSize::W32);
pub const W6: Arm64Reg = Arm64Reg::new(6, RegSize::W32);
pub const W7: Arm64Reg = Arm64Reg::new(7, RegSize::W32);
pub const W8: Arm64Reg = Arm64Reg::new(8, RegSize::W32);
pub const W9: Arm64Reg = Arm64Reg::new(9, RegSize::W32);
pub const W10: Arm64Reg = Arm64Reg::new(10, RegSize::W32);
pub const W11: Arm64Reg = Arm64Reg::new(11, RegSize::W32);
pub const W12: Arm64Reg = Arm64Reg::new(12, RegSize::W32);
pub const W13: Arm64Reg = Arm64Reg::new(13, RegSize::W32);
pub const W14: Arm64Reg = Arm64Reg::new(14, RegSize::W32);
pub const W15: Arm64Reg = Arm64Reg::new(15, RegSize::W32);
pub const W16: Arm64Reg = Arm64Reg::new(16, RegSize::W32);
pub const W17: Arm64Reg = Arm64Reg::new(17, RegSize::W32);
pub const W18: Arm64Reg = Arm64Reg::new(18, RegSize::W32);
pub const W19: Arm64Reg = Arm64Reg::new(19, RegSize::W32);
pub const W20: Arm64Reg = Arm64Reg::new(20, RegSize::W32);
pub const W21: Arm64Reg = Arm64Reg::new(21, RegSize::W32);
pub const W22: Arm64Reg = Arm64Reg::new(22, RegSize::W32);
pub const W23: Arm64Reg = Arm64Reg::new(23, RegSize::W32);
pub const W24: Arm64Reg = Arm64Reg::new(24, RegSize::W32);
pub const W25: Arm64Reg = Arm64Reg::new(25, RegSize::W32);
pub const W26: Arm64Reg = Arm64Reg::new(26, RegSize::W32);
pub const W27: Arm64Reg = Arm64Reg::new(27, RegSize::W32);
pub const W28: Arm64Reg = Arm64Reg::new(28, RegSize::W32);
pub const W29: Arm64Reg = Arm64Reg::new(29, RegSize::W32);
pub const W30: Arm64Reg = Arm64Reg::new(30, RegSize::W32);

// Special registers
pub const SP: Arm64Reg = Arm64Reg::new(31, RegSize::X64);
pub const XZR: Arm64Reg = Arm64Reg::new(31, RegSize::X64);
pub const WZR: Arm64Reg = Arm64Reg::new(31, RegSize::W32);
