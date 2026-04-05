use std::env;
use std::fs;
use std::process;

use lox::vm::{InterpretResult, VM};

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut use_jit = false;
    let mut file_path = None;

    for arg in &args[1..] {
        match arg.as_str() {
            "--jit" => use_jit = true,
            _ => file_path = Some(arg.clone()),
        }
    }

    let mut vm = VM::new();
    vm.use_jit = use_jit;

    match file_path {
        None => repl(&mut vm),
        Some(path) => run_file(&mut vm, &path),
    }
}

fn repl(vm: &mut VM) {
    let mut accumulated = String::new();
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
        accumulated.push_str(&line);
        // Recompile everything each time so functions from earlier
        // lines are available.
        vm.reset();
        vm.interpret(&accumulated);
    }
}

fn run_file(vm: &mut VM, path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Could not read file \"{}\": {}", path, e);
            process::exit(74);
        }
    };

    match vm.interpret(&source) {
        InterpretResult::Ok => {}
        InterpretResult::CompileError => process::exit(65),
        InterpretResult::RuntimeError => process::exit(70),
    }
}
