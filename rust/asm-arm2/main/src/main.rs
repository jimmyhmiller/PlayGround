use std::{error::Error, time::Instant};

mod arm;
pub mod ast;
pub mod common;
pub mod compiler;
pub mod ir;
pub mod parser;

use crate::{compiler::Compiler, parser::Parser, ir::BuiltInTypes};

fn test_fib(compiler: &mut Compiler, n: u64) -> Result<(), Box<dyn Error>> {
    let fib: ast::Ast = parser::fib();
    let mut fib: ir::Ir = fib.compile(compiler);
    let mut fib = fib.compile();
    let fib = compiler.add_function("fib", &fib.compile_to_bytes())?;

    let time = Instant::now();

    let result1 = compiler.run1(fib, n as u64)?;
    println!("Our time {:?}", time.elapsed());
    let time = Instant::now();
    let result2 = fib_rust(n as usize);
    println!("Rust time {:?}", time.elapsed());
    println!("{} {}", BuiltInTypes::untag(result1 as usize), result2);

    Ok(())
}

fn fib_rust(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    fib_rust(n - 1) + fib_rust(n - 2)
}

// Do these need to be extern "C"?
fn allocate_array(compiler: *mut Compiler, value: usize) -> usize {
    let value = BuiltInTypes::untag(value);
    let compiler = unsafe { &mut *compiler };
    compiler.allocate(value).unwrap()
}

fn array_store(compiler: *mut Compiler, array: usize, index: usize, value: usize) -> usize {
    let compiler = unsafe { &mut *compiler };
    compiler.array_store(array, index, value).unwrap()
}

fn array_get(compiler: *mut Compiler, array: usize, index: usize) -> usize {
    let compiler = unsafe { &mut *compiler };
    compiler.array_get(array, index).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {

    let mut compiler = Compiler::new();
    // Very inefficient way to do array stuff
    // but working
    compiler.add_foreign_function("print", ir::print_value as *const u8)?;
    compiler.add_builtin_function("allocate_array", allocate_array as *const u8)?;
    compiler.add_builtin_function("array_store", array_store as *const u8)?;
    compiler.add_builtin_function("array_get", array_get as *const u8)?;

    let hello_ast = parse! {
        fn hello(x) {
            array_get(array_store(array_store(allocate_array(x), 0, 42), 1, "hello"), 1)
        }
    };

    // println!("{:#?}", hello_ast);

    let mut hello_ir = hello_ast.compile(&mut compiler);
    println!("{:#?}", hello_ir);
    let mut hello = hello_ir.compile();

    // let hello2_ast = ast::hello_world2();
    // let mut hello2_ir = hello2_ast.compile(&mut compiler);
    // let mut hello2 = hello2_ir.compile();
// 
    let hello = compiler.add_function("hello", &hello.compile_to_bytes())?;

    compiler.print(compiler.run1(hello, 16).unwrap() as usize);
    // compiler.print(compiler.run1(hello, 32).unwrap() as usize);
    // println!("Got here");


    // compiler.overwrite_function(hello, &hello2.compile_to_bytes())?;

    // println!("{}", compiler.run(hello)?);

    test_fib(&mut compiler, 32)?;
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

// TODO: Make variables
// Should we allow reassignment?
// Need to add guards against type errors
// Need to do some looping construct
// Need to do load and store in the actual
// compiler instead of cheating
// Think about protocols
// Think about how to implementing interesting
// data structures in the language itself
// Consider checked and uncheck stuff


// There is clearly a bug going on with my lifetimes
// of registers.
// I had to add a bunch of registers that I shouldn't need from the looks
// of things. But I need to look closer