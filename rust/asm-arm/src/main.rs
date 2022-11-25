use std::mem;
// use memmap2::{MmapMut, Mmap};
use mmap_rs::{Error, MmapOptions, MmapMut, Mmap};

// Notes:
// Can't make things RWX
// This is big endian


struct Page {
    mem_write: Option<MmapMut>,
    mem_exec: Option<Mmap>,
}

impl Page {
    fn new() -> Result<Self, Error> {
        let mem = MmapOptions::new(MmapOptions::page_size().0)
        .map_mut()?;

        Ok(Self {
            mem_write: Some(mem),
            mem_exec: None,
        })
    }

    fn write_u32_be(&mut self, offset: usize, data: u32) -> Result<(), Error> {
        self.writeable()?;
        let memory = &mut self.mem_write.as_mut().unwrap()[..];
        for (i, byte) in data.to_be_bytes().iter().enumerate() {
            memory[offset as usize + i] = *byte;
        }
        Ok(())
    }

    fn write_u32_le(&mut self, offset: usize, data: u32) -> Result<(), Error> {
        self.writeable()?;
        let memory = &mut self.mem_write.as_mut().unwrap()[..];
        for (i, byte) in data.to_le_bytes().iter().enumerate() {
            memory[offset as usize + i] = *byte;
        }
        Ok(())
    }


    fn executable(&mut self) -> Result<(), Error> {
        if let Some(m) = self.mem_write.take() {
            let m = m.make_exec().unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap executable: {}", e);
            });
            self.mem_exec = Some(m);
        }
        Ok(())
    }

    fn writeable(&mut self) -> Result<(), Error> {
        if let Some(m) = self.mem_exec.take() {
            let m = m.make_mut().unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap writeable: {}", e);
            });
            self.mem_write = Some(m);
        }
        Ok(())
    }

    fn get_function(&mut self) -> Result<extern "C" fn() -> u64, Error> {
        self.writeable()?;
        let size = self.mem_write.as_ref().unwrap().size();
        self.mem_write.as_mut().unwrap().flush(0..size)?;
        self.executable()?;
        Ok(unsafe {
            mem::transmute(self.mem_exec.as_ref().unwrap().as_ptr())
        })
    }

}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BitSize {
    B32,
    B64,
}

impl BitSize {
    fn sf(&self) -> u8  {
        match self {
            BitSize::B32 => 0,
            BitSize::B64 => 1,
        }
    }
}
// https://github.com/ziglang/zig/blob/master/src/arch/aarch64/bits.zig
// Borrowed structure from zig


// Going to try something that will use more memory than it needs
// But I don't particularly care about that.

trait Width {
    fn value(&self) -> u32;
    fn size(&self) -> u8;
}


struct U1(u8);
impl Width for U1 {
    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        1
    }
}
struct U2(u8);
impl Width for U2 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        2
    }
}
struct U3(u8);
impl Width for U3 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        3
    }
}
struct U4(u8);
impl Width for U4 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        4
    }
}
struct U5(u8);
impl Width for U5 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        5
    }
}
struct U6(u8);
impl Width for U6 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        6
    }
}
struct U7(u8);
impl Width for U7 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        7
    }
}


struct U12(u16);
impl Width for U12 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        12
    }
}
struct U19(u32);
impl Width for U19 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        19
    }
}
struct U26(u32);
impl Width for U26 {

    fn value(&self) -> u32 {
        self.0 as u32
    }
    fn size(&self) -> u8 {
        26
    }
}

impl Width for u8 {
    fn value(&self) -> u32 {
        *self as u32
    }
    fn size(&self) -> u8 {
        8
    }
}
impl Width for u16 {
    fn value(&self) -> u32 {
        *self as u32
    }
    fn size(&self) -> u8 {
        16
    }
}

impl Width for u32 {
    fn value(&self) -> u32 {
        *self
    }
    fn size(&self) -> u8 {
        32
    }
}

macro_rules! encode_instruction {

    (@repeat $instruction:ident, $offset:ident,) => {

    };


    (@repeat $instruction:ident, $offset:ident, $head:ident, $($tail:ident,)*) => {
        $instruction |= $head.value() << $offset;
        // This is an incorrect warning
        #[allow(unused_assignments)]
        {
            $offset += $head.size();
        }
        encode_instruction!(@repeat $instruction, $offset, $($tail,)*);
    };

    ($($n:ident),*) => {
        {
            let mut instruction : u32 = 0;
            let mut offset : u8 = 0;
            encode_instruction!(@repeat instruction, offset, $($n,)*);
            instruction
        }
    };


}


// There is a part of my that wants to do a prodcedural macro for a custom derive.
// Mostly because I think it would be fun.
// There is another part of me that thinks that would be a waste of time.

impl BaseInstructionKinds {
    fn encode_instruction(&self) -> u32 {
        match self {
            BaseInstructionKinds::MoveWideImmediate { rd, imm16, shift: hw, fixed, opc, sf } => {
                encode_instruction!(rd, imm16, hw, fixed, opc, sf)
            }
            BaseInstructionKinds::PcRelativeAddress { rd, immhi, fixed, immlo, op } => {
                encode_instruction!(rd, immhi, fixed, immlo, op)
            }
            BaseInstructionKinds::LoadStoreRegister { rt, rn, offset, opc, op1, v, fixed, size } => {
                encode_instruction!(rt, rn, offset, opc, op1, v, fixed, size)
            }
            BaseInstructionKinds::LoadStoreRegisterPair { rt1, rn, rt2, imm7, load, encoding, fixed, opc } => {
                encode_instruction!(rt1, rn, rt2, imm7, load, encoding, fixed, opc)
            }
            BaseInstructionKinds::LoadLiteral { rt, imm19, fixed, opc } => {
                encode_instruction!(rt, imm19, fixed, opc)
            }
            BaseInstructionKinds::ExceptionGeneration { ll, op2, imm16, opc, fixed } => {
                encode_instruction!(ll, op2, imm16, opc, fixed)
            }
            BaseInstructionKinds::UnconditionalBranchRegister { op4, rn, op3, op2, opc, fixed } => {
                encode_instruction!(op4, rn, op3, op2, opc, fixed)
            }
            BaseInstructionKinds::UnconditionalBranchImmediate { imm26, fixed, op } =>{
                encode_instruction!(imm26, fixed, op)
            }
            BaseInstructionKinds::NoOperation { fixed } => {
                encode_instruction!(fixed)
            }
            BaseInstructionKinds::LogicalShiftedRegister { rd, rn, imm6, rm, n, shift, fixed, opc, sf } => {
                encode_instruction!(rd, rn, imm6, rm, n, shift, fixed, opc, sf)
            }
            BaseInstructionKinds::AddSubtractImmediate { rd, rn, imm12, sh, fixed, s, op, sf } => {
                encode_instruction!(rd, rn, imm12, sh, fixed, s, op, sf)
            }
            BaseInstructionKinds::LogicalImmediate { rd, rn, imms, immr, n, fixed, opc, sf } => {
                encode_instruction!(rd, rn, imms, immr, n, fixed, opc, sf)
            }
            BaseInstructionKinds::Bitfield { rd, rn, imms, immr, n, fixed, opc, sf } => {
                encode_instruction!(rd, rn, imms, immr, n, fixed, opc, sf)
            }
            BaseInstructionKinds::AddSubtractShiftedRegister { rd, rn, imm6, rm, fixed_1, shift, fixed_2, s, op, sf } => {
                encode_instruction!(rd, rn, imm6, rm, fixed_1, shift, fixed_2, s, op, sf)
            }
            BaseInstructionKinds::AddSubtractExtendedRegister { rd, rn, imm3, option, rm, fixed, s, op, sf } => {
                encode_instruction!(rd, rn, imm3, option, rm, fixed, s, op, sf)
            }
            BaseInstructionKinds::ConditionalBranch { cond, o0, imm19, o1, fixed } => {
                encode_instruction!(cond, o0, imm19, o1, fixed)
            }
            BaseInstructionKinds::CompareAndBranch { rt, imm19, op, fixed, sf } => {
                encode_instruction!(rt, imm19, op, fixed, sf)
            }
            BaseInstructionKinds::ConditionalSelect { rd, rn, op2, cond, rm, fixed, s, op, sf } => {
                encode_instruction!(rd, rn, op2, cond, rm, fixed, s, op, sf)
            }
            BaseInstructionKinds::DataProcessing3Source { rd, rn, ra, o0, rm, op31, fixed, op54, sf } => {
                encode_instruction!(rd, rn, ra, o0, rm, op31, fixed, op54, sf)
            }
            BaseInstructionKinds::DataProcessing2Source { rd, rn, opcode, rm, fixed_1, s, fixed_2, sf } => {
                encode_instruction!(rd, rn, opcode, rm, fixed_1, s, fixed_2, sf)
            }
        }
    }
}


#[allow(unused)]
enum BaseInstructionKinds {
    MoveWideImmediate {
        rd: Register,
        imm16: u16,
        shift: Shift,
        fixed: U6, // 0b100101
        opc: U2,
        sf: U1,
    },

    PcRelativeAddress {
        rd: Register,
        immhi: U19,
        fixed: U5, // 0b10000
        immlo: U2,
        op: U1,
    },

    LoadStoreRegister {
        rt: Register,
        rn: Register,
        offset: U12,
        opc: U2,
        op1: U2,
        v: U1,
        fixed: U3, // 0b111
        size: U2,
    },

    LoadStoreRegisterPair {
        rt1: U5,
        rn: Register,
        rt2: U5,
        imm7: U7,
        load: U1,
        encoding: U2,
        fixed: U5, // 0b101_0_0
        opc: U2,
    },

    LoadLiteral {
        rt: Register,
        imm19: U19,
        fixed: U6, // 0b011_0_00
        opc: U2,
    },

    ExceptionGeneration {
        ll: U2,
        op2: U3,
        imm16: u16,
        opc: U3,
        fixed: u8, // 0b1101_0100
    },

    UnconditionalBranchRegister {
        op4: U5,
        rn: Register,
        op3: U6,
        op2: U5,
        opc: U4,
        fixed: U7, // 0b1101_011
    },

    UnconditionalBranchImmediate {
        imm26: U26,
        fixed: U5, // 0b00101
        op: U1,
    },

    NoOperation {
        fixed: u32, // 0b1101010100_0_00_011_0010_0000_000_11111
    },

    LogicalShiftedRegister {
        rd: Register,
        rn: Register,
        imm6: U6,
        rm: Register,
        n: U1,
        shift: Shift,
        fixed: U5, // 0b01010
        opc: U2,
        sf: U1,
    },

    AddSubtractImmediate {
        rd: Register,
        rn: Register,
        imm12: U12,
        sh: U1,
        fixed: U6, // 0b100010
        s: U1,
        op: U1,
        sf: U1,
    },

    LogicalImmediate {
        rd: Register,
        rn: Register,
        imms: U6,
        immr: U6,
        n: U1,
        fixed: U6, // 0b100100
        opc: U2,
        sf: U1,
    },

    Bitfield {
        rd: Register,
        rn: Register,
        imms: U6,
        immr: U6,
        n: U1,
        fixed: U6, // 0b100110
        opc: U2,
        sf: U1,
    },

    AddSubtractShiftedRegister {
        rd: Register,
        rn: Register,
        imm6: U6,
        rm: Register,
        fixed_1: U1, // 0b0
        shift: Shift,
        fixed_2: U5, // 0b01011
        s: U1,
        op: U1,
        sf: U1,
    },

    AddSubtractExtendedRegister {
        rd: Register,
        rn: Register,
        imm3: U3,
        option: U3,
        rm: Register,
        fixed: u8, // 0b01011_00_1
        s: U1,
        op: U1,
        sf: U1,
    },

    ConditionalBranch {
        cond: U4,
        o0: U1,
        imm19: U19,
        o1: U1,
        fixed: U7, // 0b0101010
    },

    CompareAndBranch {
        rt: Register,
        imm19: U19,
        op: U1,
        fixed: U6, // 0b011010
        sf: U1,
    },

    ConditionalSelect {
        rd: Register,
        rn: Register,
        op2: U2,
        cond: U4,
        rm: Register,
        fixed: u8, // 0b11010100
        s: U1,
        op: U1,
        sf: U1,
    },

    DataProcessing3Source {
        rd: Register,
        rn: Register,
        ra: Register,
        o0: U1,
        rm: Register,
        op31: U3,
        fixed: U5, // 0b11011
        op54: U2,
        sf: U1,
    },

    DataProcessing2Source {
        rd: Register,
        rn: Register,
        opcode: U6,
        rm: Register,
        fixed_1: u8, // 0b11010110
        s: U1,
        fixed_2: U1, // 0b0
        sf: U1,
    },
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Register {
    index: u8,
    size: BitSize,
}

impl Width for Register {

    fn value(&self) -> u32 {
        self.index as u32
    }
    fn size(&self) -> u8 {
        5
    }
}


// LSL
#[derive(Debug, Clone, Copy)]
enum Shift {
    S0 = 0b00,
    S16 = 0b01,
    S32 = 0b10,
    S48 = 0b11,
}

impl Width for Shift {
    fn value(&self) -> u32 {
        *self as u32
    }
    fn size(&self) -> u8 {
        2
    }
}

fn encode_ret() -> u32 {
    0xC0035FD6
}






fn encode_movz_16(destination: Register, value: u16) -> u32 {

    let new = BaseInstructionKinds::MoveWideImmediate {
        rd: destination,
        imm16: value,
        shift:Shift::S0,
        fixed: U6(0b100101), // TODO: Fix this
        opc: U2(0b10),
        sf: U1(destination.size.sf()),
    }.encode_instruction();

    let old = 0
    | (destination.size.sf() as u32) << 31
    | 0b10 << 29
    | 0b1000 << 25
    | 0b101 << 23
    | (Shift::S0 as u32) << 21
    | (value as u32) << 5
    | destination.index as u32;

    assert_eq!(new, old);
    new
}
fn encode_movk(destination: Register, shift: Shift, value: u16) -> u32 {
    let new = BaseInstructionKinds::MoveWideImmediate {
        rd: destination,
        imm16: value,
        shift,
        fixed: U6(0b100101), // TODO: Fix this
        opc: U2(0b11),
        sf: U1(destination.size.sf()),
    }.encode_instruction();

    let old = 0
    | (destination.size.sf() as u32) << 31
    | 0b11 << 29
    | 0b1000 << 25
    | 0b101 << 23
    | (shift as u32) << 21
    | (value as u32) << 5
    | destination.index as u32;

    assert_eq!(old, new);
    new
}


// TODO:
// This is the longest possible encoding and we don't
// even short circuit here when we could.
fn encode_u64(destination: Register, value: u64) -> [u32; 4] {
    let mut value = value;
    let mut result = [0; 4];
    result[0] = encode_movz_16(destination, value as u16 & 0xffff);
    value >>= 16;
    result[1] = encode_movk(destination, Shift::S16, value as u16 & 0xffff);
    value >>= 16;
    result[2] = encode_movk(destination, Shift::S32, value as u16 & 0xffff);
    value >>= 16;
    result[3] = encode_movk(destination, Shift::S48, value as u16 & 0xffff);
    result
}

fn main() -> Result<(), Error> {

    let mut page = Page::new()?;


    // 0
    // | destination.size.sf() << 31
    // | 0b10 << 29
    // | 0b1000 << 25
    // | 0b101 << 23
    // | (Shift::S0 as u32) << 21
    // | (value as u32) << 5
    // | destination.index as u32

    let reg = &Register { index: 0, size: BitSize::B64};
    let value = 10 as u16 & 0xffff;


    let encoded = encode_movz_16(reg.clone(), 10);
    println!("ret: {:#x}", encoded);


    // rd: Register,
    // imm16: u16,
    // hw: U2,
    // fixed: U6, // 0b100101
    // opc: U2,
    // sf: U1,


    // let move_2_to_w0 = 0xE00280D2;
    // let movz2 = encode_movz_16(&Register { index: 0, size: BitSize::B64 }, 1);
    // let movk2 = encode_movk(&Register { index: 0, size: BitSize::B64 }, Shift::S48, 1);

    let mov = encode_u64(Register { index: 0, size: BitSize::B64 }, 0x1234567890abcdef);
    page.write_u32_le(0, mov[0])?;
    page.write_u32_le(4, mov[1])?;
    page.write_u32_le(8, mov[2])?;
    page.write_u32_le(12, mov[3])?;

    // println!("{:#x}", movk2);
    let ret: u32 = encode_ret();
    // page.write_u32_le(0, movk2)?;
    page.write_u32_be(16, ret)?;

    let main_fn = page.get_function()?;
    println!("Hello, world! {:#x}", main_fn());

    return Ok(())
}
