//! coil CLI.
//!
//! Until the elaborator lands, the useful subcommands are reader-facing:
//!
//!   coil read  <file>   read each top-level form and pretty-print it
//!   coil check <file>   read and report form count / first error
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

    match coil::read_all(&src) {
        Ok(forms) => match cmd {
            "read" => {
                for form in &forms {
                    println!("{}", coil::print(form));
                }
                ExitCode::SUCCESS
            }
            "check" => {
                println!("ok: {} top-level form(s)", forms.len());
                ExitCode::SUCCESS
            }
            other => {
                eprintln!("coil: unknown command `{other}`");
                ExitCode::from(2)
            }
        },
        Err(e) => {
            eprintln!("coil: {e}");
            ExitCode::FAILURE
        }
    }
}
