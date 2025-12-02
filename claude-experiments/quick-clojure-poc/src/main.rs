mod value;
mod reader;
mod clojure_ast;
mod eval;
mod jit;

// Our own IR and compiler
mod ir;
mod compiler;
mod arm_codegen;

use std::io::{self, Write};
use crate::reader::read;
use crate::clojure_ast::{analyze, Expr};
use crate::jit::JitCompiler;
use crate::compiler::Compiler;

fn print_help() {
    println!("\nClojure REPL Commands:");
    println!("  (+ 1 2)           - Execute expression");
    println!("  :ast (+ 1 2)      - Show AST");
    println!("  :ir (+ 1 2)       - Show IR instructions");
    println!("  :asm (+ 1 2)      - Show ARM64 machine code");
    println!("  :help             - Show this help");
    println!("  :quit             - Exit REPL");
    println!();
}

fn print_ast(ast: &Expr, indent: usize) {
    let prefix = "  ".repeat(indent);
    match ast {
        Expr::Literal(v) => println!("{}Literal({:?})", prefix, v),
        Expr::Var(name) => println!("{}Var({})", prefix, name),
        Expr::Quote(v) => println!("{}Quote({:?})", prefix, v),
        Expr::Def { name, value } => {
            println!("{}Def", prefix);
            println!("{}  name: {}", prefix, name);
            println!("{}  value:", prefix);
            print_ast(value, indent + 2);
        }
        Expr::If { test, then, else_ } => {
            println!("{}If", prefix);
            println!("{}  test:", prefix);
            print_ast(test, indent + 2);
            println!("{}  then:", prefix);
            print_ast(then, indent + 2);
            if let Some(e) = else_ {
                println!("{}  else:", prefix);
                print_ast(e, indent + 2);
            }
        }
        Expr::Do { exprs } => {
            println!("{}Do", prefix);
            for (i, expr) in exprs.iter().enumerate() {
                println!("{}  [{}]:", prefix, i);
                print_ast(expr, indent + 2);
            }
        }
        Expr::Call { func, args } => {
            println!("{}Call", prefix);
            println!("{}  func:", prefix);
            print_ast(func, indent + 2);
            println!("{}  args:", prefix);
            for (i, arg) in args.iter().enumerate() {
                println!("{}    [{}]:", prefix, i);
                print_ast(arg, indent + 3);
            }
        }
    }
}

fn print_machine_code(code: &[u32]) {
    println!("\nMachine Code ({} instructions, {} bytes):", code.len(), code.len() * 4);
    for (i, instruction) in code.iter().enumerate() {
        println!("  {:04x}: {:08x}  ; {}", i * 4, instruction, disassemble_arm64(*instruction));
    }
}

fn disassemble_arm64(inst: u32) -> String {
    // Simple ARM64 disassembly
    match inst {
        0xD65F03C0 => "ret".to_string(),
        i if (i & 0xFFE00000) == 0xD2800000 => {
            let rd = i & 0x1F;
            let imm = (i >> 5) & 0xFFFF;
            format!("movz x{}, #{}", rd, imm)
        }
        i if (i & 0xFFE00000) == 0xF2A00000 => {
            let rd = i & 0x1F;
            let imm = (i >> 5) & 0xFFFF;
            format!("movk x{}, #{}, lsl #16", rd, imm)
        }
        i if (i & 0xFFE00000) == 0x8B000000 => {
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("add x{}, x{}, x{}", rd, rn, rm)
        }
        i if (i & 0xFFE00000) == 0xCB000000 => {
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("sub x{}, x{}, x{}", rd, rn, rm)
        }
        i if (i & 0xFFE00000) == 0x9B000000 => {
            let rd = i & 0x1F;
            let rn = (i >> 5) & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("mul x{}, x{}, x{}", rd, rn, rm)
        }
        i if (i & 0xFFE00000) == 0xAA000000 => {
            let rd = i & 0x1F;
            let rm = (i >> 16) & 0x1F;
            format!("mov x{}, x{}", rd, rm)
        }
        _ => format!("<unknown: {:08x}>", inst),
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Clojure → ARM64 JIT Compiler                               ║");
    println!("║  Multi-stage compilation: Reader → AST → IR → Machine Code  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    print_help();

    let mut jit = JitCompiler::new();

    loop {
        print!("λ ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }

                // Check for REPL commands
                if input == ":help" {
                    print_help();
                    continue;
                }

                if input == ":quit" || input == ":exit" {
                    break;
                }

                // Parse command and code
                let (command, code) = if input.starts_with(':') {
                    let parts: Vec<&str> = input.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        (parts[0], parts[1])
                    } else {
                        eprintln!("Usage: {} <expression>", parts[0]);
                        continue;
                    }
                } else {
                    ("", input)
                };

                // Read and analyze
                match read(code) {
                    Ok(value) => {
                        match analyze(&value) {
                            Ok(ast) => {
                                match command {
                                    ":ast" => {
                                        println!("\nAST:");
                                        print_ast(&ast, 0);
                                        println!();
                                    }
                                    ":ir" => {
                                        let mut compiler = Compiler::new();
                                        match compiler.compile(&ast) {
                                            Ok(_) => {
                                                let instructions = compiler.finish();
                                                println!("\nIR ({} instructions):", instructions.len());
                                                for (i, inst) in instructions.iter().enumerate() {
                                                    println!("  {:3}: {:?}", i, inst);
                                                }
                                                println!();
                                            }
                                            Err(e) => eprintln!("Compile error: {}", e),
                                        }
                                    }
                                    ":asm" | ":machine" => {
                                        // Get the machine code from JIT
                                        let code = match jit.get_machine_code(&ast) {
                                            Ok(code) => code,
                                            Err(e) => {
                                                eprintln!("Compile error: {}", e);
                                                continue;
                                            }
                                        };
                                        print_machine_code(&code);
                                        println!();
                                    }
                                    "" => {
                                        // Normal execution
                                        match jit.compile_and_run(&ast) {
                                            Ok(result) => println!("{}", result),
                                            Err(e) => eprintln!("Error: {}", e),
                                        }
                                    }
                                    _ => {
                                        eprintln!("Unknown command: {}", command);
                                        eprintln!("Type :help for available commands");
                                    }
                                }
                            }
                            Err(e) => eprintln!("Parse error: {}", e),
                        }
                    }
                    Err(e) => eprintln!("Read error: {}", e),
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

    println!("\n👋 Goodbye!");
}
