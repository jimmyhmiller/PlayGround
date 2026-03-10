use std::ops::Shl;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Size {
    S8,
    S16,
    S32,
    S64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct X64Reg {
    pub index: u8,
    pub size: Size,
}

impl X64Reg {
    pub const fn new(index: u8, size: Size) -> Self {
        Self { index, size }
    }

    pub fn encode(&self) -> u8 {
        self.index
    }

    pub fn from_index(index: usize) -> X64Reg {
        X64Reg {
            index: index as u8,
            size: Size::S64,
        }
    }

    pub fn needs_rex_ext(&self) -> bool {
        self.index >= 8
    }
}

impl Shl<u32> for &X64Reg {
    type Output = u32;
    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

impl Shl<u32> for X64Reg {
    type Output = u32;
    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}

/// XMM register for SSE operations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct XmmReg {
    pub index: u8,
}

impl XmmReg {
    pub const fn new(index: u8) -> Self {
        Self { index }
    }

    pub fn needs_rex_ext(&self) -> bool {
        self.index >= 8
    }
}

// 64-bit registers
pub const RAX: X64Reg = X64Reg::new(0, Size::S64);
pub const RCX: X64Reg = X64Reg::new(1, Size::S64);
pub const RDX: X64Reg = X64Reg::new(2, Size::S64);
pub const RBX: X64Reg = X64Reg::new(3, Size::S64);
pub const RSP: X64Reg = X64Reg::new(4, Size::S64);
pub const RBP: X64Reg = X64Reg::new(5, Size::S64);
pub const RSI: X64Reg = X64Reg::new(6, Size::S64);
pub const RDI: X64Reg = X64Reg::new(7, Size::S64);
pub const R8: X64Reg = X64Reg::new(8, Size::S64);
pub const R9: X64Reg = X64Reg::new(9, Size::S64);
pub const R10: X64Reg = X64Reg::new(10, Size::S64);
pub const R11: X64Reg = X64Reg::new(11, Size::S64);
pub const R12: X64Reg = X64Reg::new(12, Size::S64);
pub const R13: X64Reg = X64Reg::new(13, Size::S64);
pub const R14: X64Reg = X64Reg::new(14, Size::S64);
pub const R15: X64Reg = X64Reg::new(15, Size::S64);

// 32-bit registers
pub const EAX: X64Reg = X64Reg::new(0, Size::S32);
pub const ECX: X64Reg = X64Reg::new(1, Size::S32);
pub const EDX: X64Reg = X64Reg::new(2, Size::S32);
pub const EBX: X64Reg = X64Reg::new(3, Size::S32);

// 8-bit registers
pub const AL: X64Reg = X64Reg::new(0, Size::S8);
pub const CL: X64Reg = X64Reg::new(1, Size::S8);

// XMM registers
pub const XMM0: XmmReg = XmmReg::new(0);
pub const XMM1: XmmReg = XmmReg::new(1);
pub const XMM2: XmmReg = XmmReg::new(2);
pub const XMM3: XmmReg = XmmReg::new(3);
pub const XMM4: XmmReg = XmmReg::new(4);
pub const XMM5: XmmReg = XmmReg::new(5);
pub const XMM6: XmmReg = XmmReg::new(6);
pub const XMM7: XmmReg = XmmReg::new(7);
pub const XMM8: XmmReg = XmmReg::new(8);
pub const XMM9: XmmReg = XmmReg::new(9);
pub const XMM10: XmmReg = XmmReg::new(10);
pub const XMM11: XmmReg = XmmReg::new(11);
pub const XMM12: XmmReg = XmmReg::new(12);
pub const XMM13: XmmReg = XmmReg::new(13);
pub const XMM14: XmmReg = XmmReg::new(14);
pub const XMM15: XmmReg = XmmReg::new(15);
