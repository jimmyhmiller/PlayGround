//! Clojure REPL entry point.
//!
//! Today: invoke as `clojure -e EXPR`. Reads `EXPR`, evaluates, prints
//! the result. Interactive REPL comes later.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == "-e" {
        let mut engine = clojure::Engine::new();
        let result = engine.eval(&args[2]);
        println!("{}", engine.print(result));
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "usage: {} -e EXPR",
            args.first().map(String::as_str).unwrap_or("clojure")
        );
        ExitCode::FAILURE
    }
}
