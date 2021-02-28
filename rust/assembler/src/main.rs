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
    RDP,
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
            Register::RBX => 1,
            Register::RCX => 2,
            Register::RDX => 3,
            Register::RDP => 4,
            Register::RSP => 5,
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
    Addr(i32)
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


#[allow(dead_code)]
impl Emitter<'_> {
    fn emit(&mut self, bytes: &[u8]) {
        for (i, byte) in bytes.iter().enumerate() {
            self.memory[self.instruction_index + i] = *byte;
            println!("here {}, {:#04x}", self.instruction_index + i, self.memory[self.instruction_index + i]);
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
        self.emit(&[0x48]);
        match (val1, val2) {
            (Val::Reg(dst), Val::Int(src)) => {
                self.emit(&[0xC7]);
                self.emit(&[0xC0 | (dst.index() & 7)]);
                self.imm(src);
            }
            (Val::Reg(_dst), Val::Addr(src)) => {
                // This is definitely wrong because dst isn't used.
                self.emit(&[0xC7]);
                self.emit(&[0xC7]);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.emit(&[0x89]);
                self.emit(&[0xC0 | (src.index() << 3) | dst.index()]);
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
                self.emit(&[0x01]);
                self.emit(&[0xC0 | (dst.index() & 7)]);
                self.imm(src);
            }
            (Val::Reg(dst), Val::Reg(src)) => {
                self.emit(&[0x01]);
                self.emit(&[0xC0 | (src.index() << 3) | dst.index()]);
            }
            _ => panic!("Mov not implemented for that combination")
        }
    }
}





fn main() {
    let m = MemoryMap::new(4096, &[MapOption::MapReadable, MapOption::MapWritable, MapOption::MapExecutable]).unwrap();
    let mut my_memory : &mut [u8; 4096] = unsafe { std::slice::from_raw_parts_mut(m.data(), m.len()).try_into().expect("wrong size")};

    let e = &mut Emitter{
        memory: &mut my_memory,
        instruction_index: 0
    };

    println!("{}", e.instruction_index);


    e.mov(Val::Reg(Register::RBX), Val::Int(20));
    e.mov(Val::Reg(Register::RAX), Val::Int(22));
    e.add(Val::Reg(Register::RAX), Val::Reg(Register::RBX));
    e.add(Val::Reg(Register::RAX), Val::Int(-1));
    e.ret();



    let main_fn: extern "C" fn() -> i64 = unsafe { mem::transmute(m.data()) };
    println!("Hello, world! {:}", main_fn());

}
