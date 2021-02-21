use std::convert::TryInto;
use mmap::MemoryMap;
use mmap::MapOption;
use std::{mem};

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
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

// Need to figure out mov_reg_reg

fn assign_32(memory: &mut [u8; 4096], instruction_index : usize, immediate: i32) {
    for (i, byte) in immediate.to_le_bytes().iter().enumerate() {
        memory[instruction_index + 1 + i] = *byte;
    }
}

#[allow(dead_code)]
// How to do this for mov 64?
fn mov_imm(memory: &mut [u8; 4096], instruction_index : usize, reg: Register, immediate: i32) {
    memory[instruction_index] = 0xB8 + reg as u8;
    assign_32(memory, instruction_index, immediate);
}

#[allow(dead_code)]
fn add_imm(memory: &mut [u8; 4096], instruction_index : usize, reg: Register, immediate: i32) {
    memory[instruction_index] = if reg == Register::RAX {
        5
    } else {
        0x81 + reg as u8
    };
    assign_32(memory, instruction_index, immediate);
}

fn ret(memory: &mut [u8; 4096], instruction_index : usize) {
    memory[instruction_index] = 0xC3;
}


fn main() {
    let m = MemoryMap::new(4096, &[MapOption::MapReadable, MapOption::MapWritable, MapOption::MapExecutable]).unwrap();
    // println!("{}", m.len());
    let my_memory : &mut [u8; 4096] = unsafe { std::slice::from_raw_parts_mut(m.data(), m.len()).try_into().expect("wrong size")};
    mov_imm(my_memory, 0, Register::RAX, 20);
    add_imm(my_memory, 5, Register::RAX, 22);
    ret(my_memory, 10);
    let main_fn: extern "C" fn() -> i64 = unsafe { mem::transmute(m.data()) };
    println!("Hello, world! {:}", main_fn());
}
