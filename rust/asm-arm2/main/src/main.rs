use std::{error::Error, mem, time::Instant};

mod arm;
pub mod ast;
pub mod common;
pub mod ir;

use crate::arm::LowLevelArm;



use ir::Ir;
use mmap_rs::MmapOptions;

fn use_the_assembler(n: i64, arm: &mut LowLevelArm) -> Result<(), Box<dyn Error>> {
    let mut buffer = MmapOptions::new(MmapOptions::page_size())?.map_mut()?;
    let memory = &mut buffer[..];

    let bytes = arm.compile_to_bytes();

    for (index, byte) in bytes.iter().enumerate() {
        memory[index] = *byte;
    }

    let size = buffer.size();
    buffer.flush(0..size)?;

    let exec = buffer.make_exec().unwrap_or_else(|(_map, e)| {
        panic!("Failed to make mmap executable: {}", e);
    });

    let f: fn(i64) -> u64 = unsafe { mem::transmute(exec.as_ref().as_ptr()) };

    let time = Instant::now();

    let result1 = f(n);
    println!("Our time {:?}", time.elapsed());
    let time = Instant::now();
    let result2 = fib_rust(n as usize);
    println!("Rust time {:?}", time.elapsed());
    println!("{} {}", result1, result2);

    Ok(())
}

fn fib_rust(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    fib_rust(n - 1) + fib_rust(n - 2)
}





fn main() -> Result<(), Box<dyn Error>> {
    let new_fib = ast::fib2();

    let mut new_ir: Ir = new_fib.compile();

    let mut lang = new_ir.compile();
    use_the_assembler(30, &mut lang)?;
    Ok(())
}

// TODO:
// Runtime?
//     Function in our language names and calling them.
//     Built-in functions
//     Stack
//     Heap
// Parser
// Debugging
