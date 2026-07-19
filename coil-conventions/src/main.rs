//! CLI: `coil [--emit-ir] <file.coil>`
//!
//! Default: JIT-compile the file and run `main`, printing its i64 result.
//! `--emit-ir`: print the generated LLVM IR instead (no execution).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (emit_ir, path) = match args.as_slice() {
        [_, flag, path] if flag == "--emit-ir" => (true, path.clone()),
        [_, path] => (false, path.clone()),
        _ => {
            eprintln!("usage: coil [--emit-ir] <file.coil>");
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

    if emit_ir {
        match coil::emit_ir(&src) {
            Ok(ir) => {
                print!("{ir}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        }
    } else {
        match coil::run_source(&src) {
            Ok(result) => {
                println!("{result}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        }
    }
}
