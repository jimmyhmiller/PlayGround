//! coil CLI.
//!
//! Front-end subcommands (no MLIR required):
//!
//!   coil read   <file>   read each top-level form and pretty-print it
//!   coil expand <file>   read + expand surface sugar, print the core forms
//!   coil check  <file>   read + expand and report the core form count
//!
//! With no subcommand, `coil <file>` behaves like `read`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, path) = match args.as_slice() {
        [path] => ("read", path.as_str()),
        [cmd, path] => (cmd.as_str(), path.as_str()),
        _ => {
            eprintln!("usage: coil [read|check] <file.coil>");
            return ExitCode::from(2);
        }
    };

    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("coil: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let forms = match coil::read_all(&src) {
        Ok(forms) => forms,
        Err(e) => {
            eprintln!("coil: {e}");
            return ExitCode::FAILURE;
        }
    };

    match cmd {
        "read" => {
            for form in &forms {
                println!("{}", coil::print(form));
            }
            ExitCode::SUCCESS
        }
        "expand" | "check" => match coil::expand_all(&forms) {
            Ok(core) => {
                if cmd == "expand" {
                    for form in &core {
                        println!("{}", coil::print(form));
                    }
                } else {
                    println!("ok: {} core form(s)", core.len());
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("coil: {e}");
                ExitCode::FAILURE
            }
        },
        other => {
            eprintln!("coil: unknown command `{other}`");
            ExitCode::from(2)
        }
    }
}
