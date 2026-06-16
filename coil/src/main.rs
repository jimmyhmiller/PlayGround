//! CLI: `coil [--emit-ir] <file.coil>`
//!
//! Default: JIT-compile the file and run `main`, printing its i64 result.
//! `--emit-ir`: print the generated LLVM IR instead (no execution).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (mode, path) = match args.as_slice() {
        [_, flag, path] if flag == "--emit-ir" => (Mode::EmitIr, path.clone()),
        [_, flag, path] if flag == "--expand" => (Mode::Expand, path.clone()),
        [_, path] => (Mode::Run, path.clone()),
        _ => {
            eprintln!("usage: coil [--emit-ir | --expand] <file.coil>");
            return ExitCode::from(2);
        }
    };

    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let result = match mode {
        Mode::EmitIr => coil::emit_ir(&src),
        Mode::Expand => coil::expand_to_string(&src),
        Mode::Run => coil::run_source(&src).map(|r| r.to_string()),
    };
    match result {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

enum Mode {
    Run,
    EmitIr,
    Expand,
}
