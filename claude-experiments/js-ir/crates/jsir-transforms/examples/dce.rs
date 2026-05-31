//! `cargo run --release --example dce -p jsir-transforms -- <file.js>`
//!
//! Runs our DCE on a JS file and prints the eliminated-code report plus the
//! rewritten source. Used for the capability study against terser/esbuild.

use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src = match args.get(1) {
        Some(p) => std::fs::read_to_string(p).expect("read file"),
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).expect("read stdin");
            s
        }
    };

    let ir = match jsir_swc::source_to_ir(&src) {
        Ok(ir) => ir,
        Err(e) => {
            eprintln!("parse/lower failed: {e}");
            std::process::exit(1);
        }
    };
    let (out, stats) = jsir_transforms::eliminate_dead_code(&ir);
    let js = jsir_swc::ir_to_source(&out).expect("lift");

    eprintln!("--- DCE report ---");
    eprintln!("if (const-true)  -> consequent : {}", stats.if_taken_consequent);
    eprintln!("if (const-false) -> alternate  : {}", stats.if_taken_alternate);
    eprintln!("while (const-false) removed    : {}", stats.while_removed);
    eprintln!("unreachable statements dropped : {}", stats.unreachable_statements);
    eprintln!("dead constant ops removed      : {}", stats.dead_values_removed);
    eprintln!("input bytes  : {}", src.len());
    eprintln!("output bytes : {}", js.len());
    eprintln!("------------------");

    print!("{js}");
}
