use std::convert::TryInto;
use mmap::MemoryMap;
use mmap::MapOption;
use std::{mem};


// https://github.com/sdiehl/tinyjit/blob/master/src/Assembler.hs

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Register {
    RAX,
    RBX,
    RCX,
    RDX,
    RBP,
    RSP,
    RSI,
    RDI,
    // These require some different encoding
    // Need to understand these better
    // R8,
    // R9,
    // R10,
    // R11,
    // R12,
    // R13,
    // R14,
    // R15,
}

impl Register {
    fn index(&self) -> u8 {
        match self {
            Register::RAX => 0,
            Register::RCX => 1,
            Register::RDX => 2,
            Register::RBX => 3,
            Register::RSP => 4,
            Register::RBP => 5,
            Register::RSI => 6,
            Register::RDI => 7,
        }
    }
}


#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Val {
    Int(i32),
    Reg(Register),
    AddrReg(Register),
    AddrRegOffset(Register, i32),
}


#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Instructions {
    Ret,
    Mov(Val, Val),
    Add(Val, Val),
    Sub(Val, Val),
    Mul(Val), // unsigned?
    IMul(Val, Val), // signed?
    Xor(Val, Val),
    Inc(Val),
    Dec(Val),
    Push(Val),
    Pop(Val),
    Call(Val),
    Loop(Val),
    Nop,
    Syscall,
    // Probably need to do some jumps
}


#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Emitter<'a> {
    // Going to assume a page for the moment
    // Obviously not good enough forever
    memory: &'a mut [u8; 4096],
    instruction_index: usize,
}


#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Rex {
    w: bool,
    r: bool,
    b: bool,
    x: bool,
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Mode {
    M11,
    M10,
    M01,
    M00,
}

impl Mode {
    fn to_bytes(self) -> u8 {
        match self {
            Mode::M11 => 0b11000000,
            Mode::M10 => 0b10000000,
            Mode::M01 => 0b01000000,
            Mode::M00 => 0b00000000,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct ModRM {
    mode: Mode,
    reg: Val,
    rm: Val,
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Instruction {
    opcode: u8,
    rex: Option<Rex>,
    modrm: Option<ModRM>,
    offset: Option<i32>,
}

#[allow(dead_code)]
impl Instruction {
    fn encode(self, bytes: &mut[u8]) -> usize {
        let mut i = 0;
        if self.rex.is_some() {
            let rex = self.rex.unwrap();
            bytes[i] = 0b0100 << 4 | 
                      (rex.w as u8) << 3 |
                      (rex.r as u8) << 2 |
                      (rex.b as u8) << 1 |
                      (rex.x as u8) << 0;
            i += 1;
        }
        bytes[i] = self.opcode;
        i += 1;
        if self.modrm.is_some() {
            let modrm = self.modrm.unwrap();
            match (modrm.reg, modrm.rm)  {
                (Val::Reg(r1), Val::Reg(r2)) => {
                    bytes[i] = modrm.mode.to_bytes() | (r1.index() << 3) | r2.index();
                    i += 1;
                }
                x => unimplemented!("Didn't handle case {:?}", x),
            }
        }
        if self.offset.is_some() {
            let offset = self.offset.unwrap();
            let offset_bytes : [u8; 4] = offset.to_le_bytes();
            for j in 0..4 {
                bytes[i] = offset_bytes[j];
                i += 1;
            }
        }
        i
    }
}


#[allow(dead_code)]
impl Emitter<'_> {
    fn emit(&mut self, bytes: &[u8]) {
        for (i, byte) in bytes.iter().enumerate() {
            self.memory[self.instruction_index + i] = *byte;
            // println!("{:#04x}", self.memory[self.instruction_index + i]);
        }
        self.instruction_index += bytes.len();
    }

    fn imm(&mut self, i : i32) {
        self.emit(&i.to_le_bytes());
    }

    fn ret(&mut self) {
        self.emit(&[0xC3]);
    }

    fn mov(&mut self, val1: Val, val2: Val) {
        // There are more compact ways to do this.
        // But being verabose helps me understand.
        match (val1, val2) {
            (Val::Reg(dst), Val::Int(src)) => {
                self.emit(&[0x48]);
                self.emit(&[0xC7]);
                self.emit(&[0xC0 | (dst.index() & 7)]);
                self.imm(src);
            }
            (Val::Reg(dst), Val::AddrReg(Register::RSP)) => {
                // This is Rex.W
                self.emit(&[0x48]);
                self.emit(&[0x8b]);
                self.emit(&[(Register::RSP.index()) | dst.index() << 3]);
                // This is a SIB where the scale is 0
                // the index is rsp
                // and the base is rsp
                // or at least I think...
                self.emit(&[0x24]);
            }
            (Val::Reg(dst), Val::AddrReg(src)) => {
                // This is Rex.W
                self.emit(&[0x48]);
                self.emit(&[0x8b]);
                self.emit(&[(src.index()) | dst.index() << 3]);
            }
            (Val::Reg(dst), Val::AddrRegOffset(src, offset)) => {
                // This is Rex.W
                self.emit(&[0x48]);
                self.emit(&[0x8b]);
                self.emit(&[0x80 | src.index() | (dst.index() << 3)]);
                if offset != 0 {
                    self.imm(offset);
                }
            }
            (Val::AddrReg(Register::RSP), Val::Reg(src)) => {
                // This is Rex.W
                self.emit(&[0x48]);
                self.emit(&[0x89]);
                self.emit(&[(Register::RSP.index()) | src.index() << 3]);
                // This is a SIB where the scale is 0
                // the index is rsp
                // and the base is rsp
                // or at least I think...
                self.emit(&[0x24]);
            }
            (Val::AddrReg(dst), Val::Reg(src)) => {
                // This is Rex.W
                self.emit(&[0x48]);
                self.emit(&[0x89]);
                self.emit(&[(dst.index()) | src.index() << 3]);
            }
            ( Val::AddrRegOffset(dst, offset), Val::Reg(src)) => {
                // This is Rex.W
                self.emit(&[0x48]);
                self.emit(&[0x89]);
                self.emit(&[0x80 | dst.index() | (src.index() << 3)]);
                if offset != 0 {
                    self.imm(offset);
                }
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.emit(&[0x48]);
                self.emit(&[0x89]);
                // This is the MODRM
                self.emit(&[0xC0 | dst.index() | (src.index() << 3)]);
            }
            _ => panic!("Mov not implemented for that combination")
        }
    }

    fn add(&mut self, val1: Val, val2: Val) {
        self.emit(&[0x48]);
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.emit(&[0x05]);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.emit(&[0x81]);
                self.emit(&[0xC0 | (dst.index() & 7)]);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.emit(&[0x81]);
                self.emit(&[0xC0 | (src.index()) | (dst.index() << 3)]);
            }
            _ => panic!("add not implemented for that combination")
        }
    }

    fn push(&mut self, val: Val) {
        match val {
            Val::Int(i) => {
                self.emit(&[0x68]);
                self.imm(i);
            },
            Val::Reg(reg) => {
                self.emit(&[0x50 | reg.index()])
            },
             _ => panic!("push not implemented for that combination")
        }
    }
    fn pop(&mut self, val: Val) {
        match val {
            Val::Int(i) => {
                self.emit(&[0x8f]);
                self.imm(i);
            },
            Val::Reg(reg) => {
                self.emit(&[0x58 | reg.index()])
            },
             _ => panic!("push not implemented for that combination")
        }
    }
}


#[allow(dead_code)]
const RAX: Val = Val::Reg(Register::RAX);

#[allow(dead_code)]
const RSP: Val = Val::Reg(Register::RSP);

#[allow(dead_code)]
const RDI: Val = Val::Reg(Register::RDI);

#[allow(dead_code)]
const RBX: Val = Val::Reg(Register::RBX);

#[allow(dead_code)]
const RCX: Val = Val::Reg(Register::RCX);


fn main() {
    let m = MemoryMap::new(4096, &[MapOption::MapReadable, MapOption::MapWritable, MapOption::MapExecutable]).unwrap();
    let mut my_memory : &mut [u8; 4096] = unsafe { std::slice::from_raw_parts_mut(m.data(), m.len()).try_into().expect("wrong size")};

    let e = &mut Emitter{
        memory: &mut my_memory,
        instruction_index: 0
    };

    // Things are encoding properly. But I'm not doing anything that makes
    // sense. Need to make program that writes to memory and reads it back.
    // e.mov(Val::Reg(Register::RBX), Val::Int(0));
    // e.mov(Val::Reg(Register::RAX), Val::Int(22));
     // RBP is used for RIP here??
    e.mov(RAX, Val::Int(42));
    e.add(RDI, Val::Int(64));
    e.mov(RBX, RSP);
    e.mov(RSP, RDI);
    // e.mov(Val::AddrRegOffset(Register::RDI, 64), Val::Reg(Register::RAX));
    // e.mov(Val::Reg(Register::RAX), Val::AddrRegOffset(Register::RDI, 64));
    e.push(Val::Reg(Register::RAX));
    // e.pop(Val::Reg(Register::RAX));
    e.mov(RSP, RBX);
    e.mov(RAX, Val::Int(43));
    // e.mov(Val::Reg(Register::RAX), Val::Int(0));
    // e.mov(Val::AddrRegOffset(Register::RBP, 64), Val::Reg(Register::RAX));
    // e.mov(Val::Reg(Register::RAX), Val::Int(0));
    // RBP is used for RIP here??
    // e.mov(Val::Reg(Register::RAX), Val::AddrRegOffset(Register::RBP, 64));
    // e.add(Val::Reg(Register::RBX), Val::Reg(Register::RAX));
    // e.add(Val::Reg(Register::RAX), Val::Int(-1));
    e.ret();

    for i in 0..e.instruction_index {
        print!("{:02x}", e.memory[i]);
    }

    let mut output : [u8;8] = [0;8];
    println!("\n");
    let instructions = Instruction {
        rex: Some(Rex {w: true, r: false, x: false, b: false}),
        opcode: 0x89,
        modrm: Some(ModRM {
            mode: Mode::M00,
            reg: RDI,
            rm: RAX,
        }),
        offset: None,
    }.encode(&mut output);
    for i in 0..(instructions) {
        print!("{:02x}", output[i]);
    }
    println!("");
    


    let instructions = Instruction {
        rex: None,
        opcode: 0xFF,
        modrm: Some(ModRM {
            mode: Mode::M11,
            reg: RSP,
            rm: RDI,
        }),
        offset: None,
    }.encode(&mut output);
    println!("instructions {}", instructions);
    for i in 0..(instructions) {
        print!("{:02x}", output[i]);
    }
    println!("");




    let instructions = Instruction {
        rex: Some(Rex {w: true, r: false, x: false, b: false}),
        opcode: 0x8D,
        modrm: Some(ModRM {
            mode: Mode::M00,
            reg: RSP,
            rm: RDI,
        }),
        offset: None,
    }.encode(&mut output);
    println!("instructions {}", instructions);
    for i in 0..(instructions) {
        print!("{:02x}", output[i]);
    }
    println!("");


    let instructions = Instruction {
        rex: Some(Rex {w: true, r: false, x: false, b: false}),
        opcode: 0xFF,
        modrm: Some(ModRM {
            mode: Mode::M11,
            reg: RCX, // Consider allowing register to just be u8
            // this is really just 1, not the register
            rm: RDI,
        }),
        offset: None,
    }.encode(&mut output);
    println!("instructions {}", instructions);
    for i in 0..(instructions) {
        print!("{:02x}", output[i]);
    }
    println!("");



    let instructions = Instruction {
        rex: Some(Rex {w: true, r: false, x: false, b: false}),
        opcode: 0x69,
        modrm: Some(ModRM {
            mode: Mode::M11,
            reg: RCX, // Consider allowing register to just be u8
            // this is really just 1, not the register
            rm: RDI,
        }),
        offset: Some(42),
    }.encode(&mut output);
    println!("instructions {}", instructions);
    for i in 0..(instructions) {
        print!("{:02x}", output[i]);
    }
    println!("");


    // This is working well, but need to figure out more than just 64bit
    // Need to figure out the whole SIB thing (can I just consider that an offset)
    // Need to figure out how I want to code this.
    // Need to deal with things that aren't really registers in modrm


    let main_fn: extern "C" fn(*mut u8) -> i64 = unsafe { mem::transmute(m.data()) };
    println!("Hello, world! {:}", main_fn(m.data()));
    println!("{}", u64::from_le_bytes(e.memory[(64-8)..64].try_into().expect("Wrong size")));
    
}
