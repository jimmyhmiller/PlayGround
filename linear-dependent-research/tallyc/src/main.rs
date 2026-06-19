//! `tally` CLI.  Currently: `tally check <file>` -- parse and run the
//! linear/permission checker, printing diagnostics.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("check") => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: tally check <file>");
                return ExitCode::FAILURE;
            };
            let src = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match tally::check_str(&src) {
                Ok(()) => {
                    println!("ok: {path} type-checks (no leaks, no use-after-free)");
                    ExitCode::SUCCESS
                }
                Err(diags) => {
                    eprintln!("{path}: rejected ({} error(s)):", diags.len());
                    for d in &diags {
                        eprintln!("  - {d}");
                    }
                    ExitCode::FAILURE
                }
            }
        }
        Some("dep") => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: tally dep <file>");
                return ExitCode::FAILURE;
            };
            let src = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match tally::surface::check_program(&src) {
                Ok(prog) => {
                    println!(
                        "ok: {path} elaborates and type-checks ({} datatype(s), {} def(s))",
                        prog.sig.datas.len(),
                        prog.defs.len()
                    );
                    // normalize a `main` if present
                    if let Some(nf) = prog.normalize("main") {
                        println!("main = {}", tally::surface::pretty(&nf));
                    }
                    ExitCode::SUCCESS
                }
                Err(diags) => {
                    eprintln!("{path}: rejected ({} error(s)):", diags.len());
                    for d in &diags {
                        eprintln!("  - {d}");
                    }
                    ExitCode::FAILURE
                }
            }
        }
        Some("lang") => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: tally lang <file>");
                return ExitCode::FAILURE;
            };
            let src = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match tally::rust_surface::check_program(&src) {
                Ok(prog) => {
                    println!(
                        "ok: {path} elaborates and type-checks ({} datatype(s), {} def(s))",
                        prog.sig.datas.len(),
                        prog.defs.len()
                    );
                    if let Some(nf) = prog.normalize("main") {
                        println!("main = {}", tally::rust_surface::pretty(&nf));
                    }
                    ExitCode::SUCCESS
                }
                Err(diags) => {
                    eprintln!("{path}: rejected ({} error(s)):", diags.len());
                    for d in &diags {
                        eprintln!("  - {d}");
                    }
                    ExitCode::FAILURE
                }
            }
        }
        Some("run") => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: tally run <file>");
                return ExitCode::FAILURE;
            };
            let src = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            run_native(&src, path)
        }
        _ => {
            eprintln!(
                "lambda-Tally compiler\nusage:\n  tally check <file>   (low-level linear checker)\n  tally dep <file>     (dependent surface, ML syntax)\n  tally lang <file>    (v1.0 surface: ML types, Rust terms)\n  tally run <file>     (type-check + COMPILE main to native, run it)"
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "llvm")]
fn run_native(src: &str, path: &str) -> ExitCode {
    let prog = match tally::rust_surface::check_program(src) {
        Ok(p) => p,
        Err(diags) => {
            eprintln!("{path}: rejected ({} error(s)):", diags.len());
            for d in &diags {
                eprintln!("  - {d}");
            }
            return ExitCode::FAILURE;
        }
    };
    let Some((_, _, body)) = prog.defs.iter().find(|(n, _, _)| n == "main") else {
        eprintln!("{path}: no `main` to run");
        return ExitCode::FAILURE;
    };
    match tally::dep_codegen::run_nat(&prog.sig, body) {
        Ok(v) => {
            println!("{path}: type-checks, compiled to native, ran → {v}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{path}: cannot compile to native: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "llvm"))]
fn run_native(_src: &str, _path: &str) -> ExitCode {
    eprintln!("`tally run` needs the native backend; rebuild with `--features llvm`");
    ExitCode::FAILURE
}
