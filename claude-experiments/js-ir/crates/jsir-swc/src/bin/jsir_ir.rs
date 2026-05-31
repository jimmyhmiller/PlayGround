//! CLI: lower a JS file to JSIR IR text (the `source2ast,ast2hir` of upstream
//! `jsir_gen --output_type=ir`), for differential parity testing.
//!
//! Usage:
//!   jsir_ir <file.js>              print our IR for the file
//!   jsir_ir --roundtrip <file.js>  print ir_to_source(source_to_ir(file))

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (roundtrip, path) = match args.as_slice() {
        [_, flag, p] if flag == "--roundtrip" => (true, p.clone()),
        [_, p] => (false, p.clone()),
        _ => {
            eprintln!("usage: jsir_ir [--roundtrip] <file.js>");
            return ExitCode::from(2);
        }
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let ir = match jsir_swc::source_to_ir(&src) {
        Ok(op) => op,
        Err(e) => {
            eprintln!("source_to_ir: {e}");
            return ExitCode::from(1);
        }
    };
    if roundtrip {
        match jsir_swc::ir_to_source(&ir) {
            Ok(js) => print!("{js}"),
            Err(e) => {
                eprintln!("ir_to_source: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        print!("{}", ir.print());
    }
    ExitCode::SUCCESS
}
