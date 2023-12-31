use std::{error::Error, time::Instant};

mod arm;
pub mod ast;
pub mod common;
pub mod compiler;
pub mod ir;
pub mod parser;

use crate::{compiler::Compiler, parser::Parser};

fn test_fib(compiler: &mut Compiler, n: u64) -> Result<(), Box<dyn Error>> {
    let fib: ast::Ast = ast::fib();
    let mut fib: ir::Ir = fib.compile(compiler);
    let mut fib = fib.compile();
    let fib = compiler.add_function("fib", &fib.compile_to_bytes())?;

    let time = Instant::now();

    let result1 = compiler.run1(fib, n)?;
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

// Do these need to be extern "C"?
fn test_builtin(compiler: *const Compiler) -> usize {
    let compiler = unsafe { &*compiler };
    println!("{:?}", compiler);
    42
}

fn main() -> Result<(), Box<dyn Error>> {

    let mut compiler = Compiler::new();
    compiler.add_foreign_function("print", ir::print_value as *const u8)?;
    compiler.add_builtin_function("test", test_builtin as *const u8)?;

    let mut parser = Parser::new(String::from("
    fn hello() {
        print(\"Hello World!\")
    }"));

    let hello_ast = parser.parse();

    // let hello_ast = ast::hello_world();
    let mut hello_ir = hello_ast.compile(&mut compiler);
    let mut hello = hello_ir.compile();

    let hello2_ast = ast::hello_world2();
    let mut hello2_ir = hello2_ast.compile(&mut compiler);
    let mut hello2 = hello2_ir.compile();

    let hello = compiler.add_function("hello", &hello.compile_to_bytes())?;

    println!("{}", compiler.run(hello)?);

    compiler.overwrite_function(hello, &hello2.compile_to_bytes())?;

    println!("{}", compiler.run(hello)?);

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
// Think about variables
// Should we allow reassignment?
