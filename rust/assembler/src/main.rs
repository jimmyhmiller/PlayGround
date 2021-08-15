use mmap::MapOption;
use mmap::MemoryMap;
use std::convert::TryInto;
use std::mem;
use std::io::{self, Write};
use std::process::Command;

// https://github.com/sdiehl/tinyjit/blob/master/src/Assembler.hs

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
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

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
#[allow(dead_code)]
enum Val {
    Int(i32),
    U8(u8),
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
    Mul(Val),       // unsigned?
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

const REX_W: Rex = Rex {
    w: true,
    r: false,
    b: false,
    x: false,
};

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Mode {
    M11,
    M10,
    M01,
    M00,
}

impl Mode {
    fn into_bytes(self) -> u8 {
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
enum Scale {
    S11,
    S10,
    S01,
    S00,
}

impl Scale {
    fn into_bytes(self) -> u8 {
        match self {
            Scale::S11 => 0b11000000,
            Scale::S10 => 0b10000000,
            Scale::S01 => 0b01000000,
            Scale::S00 => 0b00000000,
        }
    }
}

// Probably use this in modrm too.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[allow(dead_code)]
enum RegOrU3 {
    Reg(Register),
    U3(u8),
}

impl RegOrU3 {
    fn into_bytes(self) -> u8 {
        match self {
            RegOrU3::U3(u8) => u8,
            RegOrU3::Reg(reg) => reg.index(),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Sib {
    scale: Scale,
    index: RegOrU3,
    base: RegOrU3,
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct Instruction {
    opcode: u8,
    rex: Option<Rex>,
    modrm: Option<ModRM>,
    imm32: Option<u32>,
}

#[allow(dead_code)]
impl Instruction {
    fn encode(self, bytes: &mut [u8]) -> usize {
        let mut i = 0;
        if self.rex.is_some() {
            let rex = self.rex.unwrap();
            bytes[i] = 0b0100 << 4
                | (rex.w as u8) << 3
                | (rex.r as u8) << 2
                | (rex.b as u8) << 1
                | (rex.x as u8);
            i += 1;
        }
        bytes[i] = self.opcode;
        i += 1;
        if self.modrm.is_some() {
            let modrm = self.modrm.unwrap();
            match (modrm.reg, modrm.rm) {
                (Val::Reg(r1), Val::Reg(r2)) => {
                    bytes[i] = modrm.mode.into_bytes() | (r1.index() << 3) | r2.index();
                    i += 1;
                }
                (Val::U8(u), Val::Reg(r2)) => {
                    bytes[i] = modrm.mode.into_bytes() | (u << 3) | r2.index();
                    i += 1;
                }
                (Val::Reg(r1), Val::U8(u)) => {
                    bytes[i] = modrm.mode.into_bytes() | (r1.index() << 3) | u;
                    i += 1;
                }
                x => unimplemented!("Didn't handle case {:?}", x),
            }
        }
        if self.imm32.is_some() {
            let imm32 = self.imm32.unwrap();
            let imm32_bytes: [u8; 4] = imm32.to_le_bytes();
            for byte in &imm32_bytes {
                bytes[i] = *byte;
                i += 1;
            }
        }
        i
    }
}

#[allow(dead_code)]
impl Emitter<'_> {
    fn emit_u8(&mut self, byte: u8) {
        self.memory[self.instruction_index] = byte;
        self.instruction_index += 1;
    }

    fn opcode(&mut self, opcode: u8) {
        self.emit_u8(opcode);
    }

    fn rex(&mut self, rex: Rex) {
        self.emit_u8(
            0b0100 << 4
                | (rex.w as u8) << 3
                | (rex.r as u8) << 2
                | (rex.b as u8) << 1
                | (rex.x as u8),
        );
    }

    fn modrm(&mut self, modrm: ModRM) {
        match (modrm.reg, modrm.rm) {
            (Val::Reg(r1), Val::Reg(r2)) => {
                self.emit_u8(modrm.mode.into_bytes() | (r1.index() << 3) | r2.index());
            }
            (Val::U8(u), Val::Reg(r2)) => {
                self.emit_u8(modrm.mode.into_bytes() | (u << 3) | r2.index());
            }
            (Val::Reg(r1), Val::U8(u)) => {
                self.emit_u8(modrm.mode.into_bytes() | (r1.index() << 3) | u);
            }
            x => unimplemented!("Didn't handle case {:?}", x),
        }
    }

    fn sib(&mut self, sib: Sib) {
        self.emit_u8(sib.scale.into_bytes() | sib.index.into_bytes() << 3 | sib.base.into_bytes());
    }

    fn emit(&mut self, bytes: &[u8]) {
        for (i, byte) in bytes.iter().enumerate() {
            self.memory[self.instruction_index + i] = *byte;
            // println!("{:#04x}", self.memory[self.instruction_index + i]);
        }
        self.instruction_index += bytes.len();
    }

    fn imm(&mut self, i: i32) {
        self.emit(&i.to_le_bytes());
    }

    fn ret(&mut self) {
        self.opcode(0xC3);
    }

    // I could make it so there are multiple moves depending on type

    fn mov(&mut self, val1: Val, val2: Val) {
        // I might have gotten some of the src vs dst wrong :(

        match (val1, val2) {
            (Val::Reg(dst), Val::Int(src)) => {
                // Need to deal with extended registers
                self.rex(REX_W);
                self.opcode(0xC7);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(0),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::AddrReg(Register::RSP)) => {
                self.rex(REX_W);
                self.opcode(0x8b);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::U8(4),
                    // Need to rethink what modrm takes, because it doesn't care about addrReg I don't think
                    // I am just turning it into reg here becasue then the code will work.
                    rm: Val::Reg(dst),
                });
                self.emit(&[(Register::RSP.index()) | dst.index() << 3]);
                // This seems like a completely special case. But is there a way for me to see it isn't?
                // Maybe it jsut is.
                self.sib(Sib {
                    scale: Scale::S00,
                    index: RegOrU3::Reg(Register::RSP),
                    base: RegOrU3::Reg(Register::RSP),
                });
            }
            (Val::Reg(dst), Val::AddrReg(src)) => {
                self.rex(REX_W);
                self.opcode(0x8b);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            (Val::Reg(dst), Val::AddrRegOffset(src, offset)) => {
                self.rex(REX_W);
                self.opcode(0x8b);
                self.modrm(ModRM {
                    mode: Mode::M10,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
                assert!(offset != 0);
                self.imm(offset);
            }
            (Val::AddrReg(Register::RSP), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::U8(4),
                    rm: Val::Reg(src),
                });
                // Seems like special case, is it?
                self.sib(Sib {
                    scale: Scale::S00,
                    index: RegOrU3::Reg(Register::RSP),
                    base: RegOrU3::Reg(Register::RSP),
                });
            }
            (Val::AddrReg(dst), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M00,
                    reg: Val::Reg(src),
                    rm: Val::Reg(dst),
                });
            }
            (Val::AddrRegOffset(dst, offset), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M10,
                    reg: Val::Reg(src),
                    rm: Val::Reg(dst),
                });
                assert!(offset != 0);
                self.imm(offset);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x89);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(src),
                    rm: Val::Reg(dst),
                });
            }
            _ => panic!("Mov not implemented for that combination"),
        }
    }

    fn add(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x05);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(0),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x03);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn sub(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x2D);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(5),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x2B);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn imul(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0xF7);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(5),
                    rm: Val::Reg(src),
                });
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x69);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x69);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
                self.imm(1);

            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn and(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x25);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(4),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x23);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }

    fn or(&mut self, val1: Val, val2: Val) {
        match (val1, val2) {
            (Val::Reg(Register::RAX), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x0D);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Int(src)) => {
                self.rex(REX_W);
                self.opcode(0x81);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::U8(1),
                    rm: Val::Reg(dst),
                });
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.rex(REX_W);
                self.opcode(0x0B);
                self.modrm(ModRM {
                    mode: Mode::M11,
                    reg: Val::Reg(dst),
                    rm: Val::Reg(src),
                });
            }
            _ => panic!("add not implemented for that combination"),
        }
    }



    fn push(&mut self, val: Val) {
        match val {
            Val::Int(i) => {
                self.emit(&[0x68]);
                self.imm(i);
            }
            Val::Reg(reg) => self.emit(&[0x50 | reg.index()]),
            _ => panic!("push not implemented for that combination"),
        }
    }
    fn pop(&mut self, val: Val) {
        match val {
            Val::Int(i) => {
                self.emit(&[0x8f]);
                self.imm(i);
            }
            Val::Reg(reg) => self.emit(&[0x58 | reg.index()]),
            _ => panic!("push not implemented for that combination"),
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
const RSI: Val = Val::Reg(Register::RSI);


#[allow(dead_code)]
const RBX: Val = Val::Reg(Register::RBX);

#[allow(dead_code)]
const RCX: Val = Val::Reg(Register::RCX);

fn main() {
    let m = MemoryMap::new(
        4096,
        &[
            MapOption::MapReadable,
            MapOption::MapWritable,
            MapOption::MapExecutable,
        ],
    )
    .unwrap();
    let mut my_memory: &mut [u8; 4096] = unsafe {
        std::slice::from_raw_parts_mut(m.data(), m.len())
            .try_into()
            .expect("wrong size")
    };

    let e = &mut Emitter {
        memory: &mut my_memory,
        instruction_index: 0,
    };

    // If my code gets too big, I could overwrite this offset
    let mem_offset = 4096-128;
    let mem_offset_size : usize = mem_offset.try_into().unwrap();

    // Things are encoding properly. But I'm not doing anything that makes
    // sense. Need to make program that writes to memory and reads it back.
    // e.mov(Val::Reg(Register::RBX), Val::Int(0));
    // e.mov(Val::Reg(Register::RAX), Val::Int(22));
    // RBP is used for RIP here??

    e.mov(RAX, Val::Int(42));
    e.sub(RAX, Val::Int(1));
    e.mov(RSI, RDI);
    // e.add(RDI, Val::Int(64));
    // e.mov(RBX, RSP);
    // e.mov(RSP, RDI);
    e.mov(Val::AddrRegOffset(Register::RDI, mem_offset), Val::Reg(Register::RAX));
    e.mov(RAX, Val::AddrRegOffset(Register::RDI, mem_offset));
    e.add(RDI, Val::Int(mem_offset));
    e.sub(RSI, Val::Int(1));
    e.imul(RSI, Val::Int(2));
    e.mov(RBX, Val::AddrReg(Register::RDI));
    e.mov(Val::AddrReg(Register::RDI), RBX);
    // e.mov(Val::AddrRegOffset(Register::RDI, 0), RBX);
    e.push(Val::Reg(Register::RAX));
    e.pop(Val::Reg(Register::RAX));
    e.mov(RSI, RBX);
    // e.mov(RAX, Val::Int(43));
    e.and(RAX, RAX);
    e.and(RAX, Val::Int(1));
    e.and(RAX, RDI);
    e.or(RAX, RAX);
    e.or(RAX, Val::Int(1));
    e.or(RAX, RSI);
    e.add(RAX, RBX);
    e.add(RAX, Val::Int(1));
    e.imul(RAX, Val::Int(2));

    e.ret();


   let result =  e.memory.iter().take(e.instruction_index).fold(String::new(),
        |res, byte| res + &format!("{:02x}", byte)
    );

    println!("{}", result);
    println!();
   
    let output =  Command::new("yaxdis")
            .arg("-a")
            .arg("x86_64")
            .arg(result)
            .output()
            .expect("failed to execute process");

    println!("{}", String::from_utf8(output.stdout).unwrap());

    io::stdout().flush().unwrap();
    // This is working well, but need to figure out more than just 64bit
    // Need to figure out the whole SIB thing (can I just consider that an offset)
    // Need to figure out how I want to code this.
    // Need to deal with things that aren't really registers in modrm

    let main_fn: extern "C" fn(*mut u8) -> i64 = unsafe { mem::transmute(m.data()) };
    println!("Hello, world! {:}", main_fn(m.data()));
    println!(
        "{}",
        // I had been looking at a different address because I change rsp
        // and pushing which goes in the other direction
        u64::from_le_bytes(e.memory[mem_offset_size..(mem_offset_size+8)].try_into().expect("Wrong size"))
    );
}
