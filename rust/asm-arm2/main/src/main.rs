use std::{error::Error, mem, time::Instant};

mod arm;
pub mod ast;
pub mod common;
pub mod ir;

use crate::arm::LowLevelArm;

use mmap_rs::{Mmap, MmapOptions};

fn compile_arm(arm: &mut LowLevelArm) -> Result<Mmap, Box<dyn Error>> {
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
    Ok(exec)
}

fn test_fib(n: i64) -> Result<(), Box<dyn Error>> {
    let fib: ast::Ast = ast::fib();
    let mut fib: ir::Ir = fib.compile(|ir| {});
    let mut fib = fib.compile();
    let exec = compile_arm(&mut fib)?;

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
    let hello_ast = ast::hello_world();
    let mut hello_ir = hello_ast.compile(|ir| {
        ir.add_function("print", ir::print_value as *const u8);
    });
    let mut hello = hello_ir.compile();

    let mem = compile_arm(&mut hello)?;
    let f: fn() -> u64 = unsafe { mem::transmute(mem.as_ref().as_ptr()) };
    println!("{:?}", f());
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
