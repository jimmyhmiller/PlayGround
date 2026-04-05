use std::env;
use std::fs;
use std::process;

use lox::vm::{InterpretResult, VM};

fn main() {
    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => repl(),
        2 => run_file(&args[1]),
        _ => {
            eprintln!("Usage: lox [path]");
            process::exit(64);
        }
    }
}

fn repl() {
    let mut vm = VM::new();
    let mut line = String::new();
    loop {
        print!("> ");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
        line.clear();
        if std::io::stdin().read_line(&mut line).unwrap() == 0 {
            println!();
            break;
        }
        vm.interpret(&line);
    }
}

fn run_file(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Could not read file \"{}\": {}", path, e);
            process::exit(74);
        }
    };

    let mut vm = VM::new();
    match vm.interpret(&source) {
        InterpretResult::Ok => {}
        InterpretResult::CompileError => process::exit(65),
        InterpretResult::RuntimeError => process::exit(70),
    }
}
