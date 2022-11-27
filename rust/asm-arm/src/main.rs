
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

    #[allow(dead_code)]
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


#[allow(dead_code)]
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


#[derive(Debug, Clone, Copy)]
struct Value<const SIZE: u8>(u32);

impl<const SIZE: u8> Value<SIZE> {
    fn value(&self) -> u32 {
        self.0
    }
    const fn size(&self) -> u8 {
        SIZE
    }
}

#[derive(Debug, Clone, Copy)]
struct Fixed<const SIZE: u8, const VALUE : u32>();

impl<const SIZE: u8, const VALUE: u32> Fixed<SIZE, VALUE> {
    const fn value(&self) -> u32 {
        VALUE
    }
    fn size(&self) -> u8 {
        SIZE
    }
}

#[derive(Debug, Clone, Copy)]
struct Register {
    index: u8,
    size: BitSize,
}

impl Register {
    fn value(&self) -> u32 {
        self.index as u32
    }
    fn size(&self) -> u8 {
        5
    }
}


#[derive(Debug, Clone, Copy)]
struct Shift(u32);

impl Shift {
    fn value(&self) -> u32 {
        self.0
    }
    fn size(&self) -> u8 {
        2
    }

    fn shift_0() -> Shift {
        Shift(0b00)
    }
    fn shift_16() -> Shift {
        Shift(0b01)
    }
    fn shift_32() -> Shift {
        Shift(0b10)
    }
    fn shift_48() -> Shift {
        Shift(0b11)
    }
}

type U1 = Value<1>;
type U2 = Value<2>;
type U3 = Value<3>;
type U4 = Value<4>;
type U5 = Value<5>;
type U6 = Value<6>;
type U7 = Value<7>;
// type U8 = Value<8>;
type U12 = Value<12>;
type U16 = Value<16>;
type U19 = Value<19>;
type U26 = Value<26>;
// type U32 = Value<32>;


macro_rules! encode_instruction {

    (@repeat $instruction:expr, $offset:expr,) => {

    };

    (@repeat $instruction:ident, $offset:ident, $head:expr, $($tail:expr,)*) => {
        $instruction |= $head.value() << $offset;
        #[allow(unused_assignments)] {
            $offset += $head.size();
        }
        encode_instruction!(@repeat $instruction, $offset, $($tail,)*);
    };

    ($($n:expr),*) => {
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

impl BaseInstructionKind {
    fn encode_instruction(&self) -> u32 {
        match self {
            BaseInstructionKind::MoveWideImmediate { rd, imm16, shift: hw, fixed, opc, sf } => {
                encode_instruction!(rd, imm16, hw, fixed, opc, sf)
            }
            BaseInstructionKind::PcRelativeAddress { rd, immhi, fixed, immlo, op } => {
                encode_instruction!(rd, immhi, fixed, immlo, op)
            }
            BaseInstructionKind::LoadStoreRegister { rt, rn, offset, opc, op1, v, fixed, size } => {
                encode_instruction!(rt, rn, offset, opc, op1, v, fixed, size)
            }
            BaseInstructionKind::LoadStoreRegisterPair { rt1, rn, rt2, imm7, load, encoding, fixed, opc } => {
                encode_instruction!(rt1, rn, rt2, imm7, load, encoding, fixed, opc)
            }
            BaseInstructionKind::LoadLiteral { rt, imm19, fixed, opc } => {
                encode_instruction!(rt, imm19, fixed, opc)
            }
            BaseInstructionKind::ExceptionGeneration { ll, op2, imm16, opc, fixed } => {
                encode_instruction!(ll, op2, imm16, opc, fixed)
            }
            BaseInstructionKind::UnconditionalBranchRegister { op4, rn, op3, op2, opc, fixed } => {
                encode_instruction!(op4, rn, op3, op2, opc, fixed)
            }
            BaseInstructionKind::UnconditionalBranchImmediate { imm26, fixed, op } =>{
                encode_instruction!(imm26, fixed, op)
            }
            BaseInstructionKind::NoOperation { fixed } => {
                encode_instruction!(fixed)
            }
            BaseInstructionKind::LogicalShiftedRegister { rd, rn, imm6, rm, n, shift, fixed, opc, sf } => {
                encode_instruction!(rd, rn, imm6, rm, n, shift, fixed, opc, sf)
            }
            BaseInstructionKind::AddSubtractImmediate { rd, rn, imm12, sh, fixed, s, op, sf } => {
                encode_instruction!(rd, rn, imm12, sh, fixed, s, op, sf)
            }
            BaseInstructionKind::LogicalImmediate { rd, rn, imms, immr, n, fixed, opc, sf } => {
                encode_instruction!(rd, rn, imms, immr, n, fixed, opc, sf)
            }
            BaseInstructionKind::Bitfield { rd, rn, imms, immr, n, fixed, opc, sf } => {
                encode_instruction!(rd, rn, imms, immr, n, fixed, opc, sf)
            }
            BaseInstructionKind::AddSubtractShiftedRegister { rd, rn, imm6, rm, fixed_1, shift, fixed_2, s, op, sf } => {
                encode_instruction!(rd, rn, imm6, rm, fixed_1, shift, fixed_2, s, op, sf)
            }
            BaseInstructionKind::AddSubtractExtendedRegister { rd, rn, imm3, option, rm, fixed, s, op, sf } => {
                encode_instruction!(rd, rn, imm3, option, rm, fixed, s, op, sf)
            }
            BaseInstructionKind::ConditionalBranch { cond, o0, imm19, o1, fixed } => {
                encode_instruction!(cond, o0, imm19, o1, fixed)
            }
            BaseInstructionKind::CompareAndBranch { rt, imm19, op, fixed, sf } => {
                encode_instruction!(rt, imm19, op, fixed, sf)
            }
            BaseInstructionKind::ConditionalSelect { rd, rn, op2, cond, rm, fixed, s, op, sf } => {
                encode_instruction!(rd, rn, op2, cond, rm, fixed, s, op, sf)
            }
            BaseInstructionKind::DataProcessing3Source { rd, rn, ra, o0, rm, op31, fixed, op54, sf } => {
                encode_instruction!(rd, rn, ra, o0, rm, op31, fixed, op54, sf)
            }
            BaseInstructionKind::DataProcessing2Source { rd, rn, opcode, rm, fixed_1, s, fixed_2, sf } => {
                encode_instruction!(rd, rn, opcode, rm, fixed_1, s, fixed_2, sf)
            }
        }
    }
}


#[allow(unused)]
enum BaseInstructionKind {
    MoveWideImmediate {
        rd: Register,
        imm16: U16,
        shift: Shift,
        fixed: Fixed<6, 0b100101>,
        opc: U2,
        sf: U1,
    },

    PcRelativeAddress {
        rd: Register,
        immhi: U19,
        fixed: Fixed<5, 0b10000>,
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
        fixed: Fixed<3, 0b111>,
        size: U2,
    },

    LoadStoreRegisterPair {
        rt1: U5,
        rn: Register,
        rt2: U5,
        imm7: U7,
        load: U1,
        encoding: U2,
        fixed: Fixed<5, 0b101_0_0>,
        opc: U2,
    },

    LoadLiteral {
        rt: Register,
        imm19: U19,
        fixed: Fixed<6, 0b011_0_00>,
        opc: U2,
    },

    ExceptionGeneration {
        ll: U2,
        op2: U3,
        imm16: U16,
        opc: U3,
        fixed: Fixed<8, 0b1101_0100>
    },

    UnconditionalBranchRegister {
        op4: U5,
        rn: Register,
        op3: U6,
        op2: U5,
        opc: U4,
        fixed: Fixed<7, 0b1101_011>,
    },

    UnconditionalBranchImmediate {
        imm26: U26,
        fixed: Fixed<5, 0b00101>,
        op: U1,
    },

    NoOperation {
        fixed: Fixed<32, 0b1101010100_0_00_011_0010_0000_000_11111>,
    },

    LogicalShiftedRegister {
        rd: Register,
        rn: Register,
        imm6: U6,
        rm: Register,
        n: U1,
        shift: Shift,
        fixed: Fixed<5, 0b01010>,
        opc: U2,
        sf: U1,
    },

    AddSubtractImmediate {
        rd: Register,
        rn: Register,
        imm12: U12,
        sh: U1,
        fixed: Fixed<6, 0b100010>,
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
        fixed: Fixed<6, 0b100100>,
        opc: U2,
        sf: U1,
    },

    Bitfield {
        rd: Register,
        rn: Register,
        imms: U6,
        immr: U6,
        n: U1,
        fixed: Fixed<6, 0b100110>,
        opc: U2,
        sf: U1,
    },

    AddSubtractShiftedRegister {
        rd: Register,
        rn: Register,
        imm6: U6,
        rm: Register,
        fixed_1: Fixed<1, 0b0>,
        shift: Shift,
        fixed_2: Fixed<5, 0b01011>,
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
        fixed: Fixed<8, 0b01011_00_1>,
        s: U1,
        op: U1,
        sf: U1,
    },

    ConditionalBranch {
        cond: U4,
        o0: U1,
        imm19: U19,
        o1: U1,
        fixed: Fixed<7, 0b0101010>,
    },

    CompareAndBranch {
        rt: Register,
        imm19: U19,
        op: U1,
        fixed: Fixed<6, 0b011010>,
        sf: U1,
    },

    ConditionalSelect {
        rd: Register,
        rn: Register,
        op2: U2,
        cond: U4,
        rm: Register,
        fixed: Fixed<8, 0b11010100>,
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
        fixed: Fixed<5, 0b11011>,
        op54: U2,
        sf: U1,
    },

    DataProcessing2Source {
        rd: Register,
        rn: Register,
        opcode: U6,
        rm: Register,
        fixed_1: Fixed<8, 0b11010110>,
        s: U1,
        fixed_2: Fixed<1, 0b0>,
        sf: U1,
    },
}

fn encode_ret() -> u32 {
    BaseInstructionKind::UnconditionalBranchRegister {
        op4: Value(0b00000),
        rn: Register { index: 30, size: BitSize::B64 },
        op3: Value(0b000000),
        op2: Value(0b11111),
        opc: Value(0b0010),
        fixed: Fixed(),
    }.encode_instruction()
}

fn encode_movz_16(destination: Register, value: u16) -> u32 {

    BaseInstructionKind::MoveWideImmediate {
        rd: destination,
        imm16: Value(value as u32),
        shift: Shift::shift_0(),
        fixed: Fixed(),
        opc: Value(0b10),
        sf: Value(destination.size.sf() as u32),
    }.encode_instruction()
}
fn encode_movk(destination: Register, shift: Shift, value: U16) -> u32 {
    BaseInstructionKind::MoveWideImmediate {
        rd: destination,
        imm16: value,
        shift,
        fixed: Fixed(),
        opc: Value(0b11),
        sf: Value(destination.size.sf() as u32),
    }.encode_instruction()
}


// TODO:
// This is the longest possible encoding and we don't
// even short circuit here when we could.
fn encode_u64(destination: Register, value: u64) -> [u32; 4] {
    let mut value = value;
    let mut result = [0; 4];
    result[0] = encode_movz_16(destination, value as u16 & 0xffff);
    value >>= 16;
    result[1] = encode_movk(destination, Shift::shift_16(), Value((value as u16 & 0xffff) as u32));
    value >>= 16;
    result[2] = encode_movk(destination, Shift::shift_32(), Value((value as u16 & 0xffff) as u32));
    value >>= 16;
    result[3] = encode_movk(destination, Shift::shift_48(), Value((value as u16 & 0xffff) as u32));
    result
}

fn main() -> Result<(), Error> {

    let mut page = Page::new()?;

    let mov = encode_u64(Register { index: 0, size: BitSize::B64 }, 0x1234567890abcdef);
    page.write_u32_le(0, mov[0])?;
    page.write_u32_le(4, mov[1])?;
    page.write_u32_le(8, mov[2])?;
    page.write_u32_le(12, mov[3])?;

    page.write_u32_le(16, encode_ret())?;


    let main_fn = page.get_function()?;
    println!("Hello, world! {:#x}", main_fn());

    return Ok(())
}
