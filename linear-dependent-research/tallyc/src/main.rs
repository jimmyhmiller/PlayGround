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
        _ => {
            eprintln!(
                "lambda-Tally compiler\nusage:\n  tally check <file>   (low-level linear checker)\n  tally dep <file>     (dependent+linear surface language)"
            );
            ExitCode::FAILURE
        }
    }
}
