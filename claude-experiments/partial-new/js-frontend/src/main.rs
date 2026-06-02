//! CLI: partially evaluate a real JavaScript file.
//!
//!   js-frontend <file.js> [input ...]
//!
//! Parses the file with SWC, lowers the supported subset, specializes
//! `function main(input)` with the engine, prints the residual program, and (for
//! any integer inputs given) runs both the residual and the reference
//! interpreter and checks they agree.

use std::process::ExitCode;

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().collect();
    let emit_js = raw.iter().any(|a| a == "--js");
    let args: Vec<String> = raw.into_iter().filter(|a| a != "--js").collect();
    if args.len() < 2 {
        eprintln!("usage: {} [--js] <file.js> [input ...]", args[0]);
        eprintln!("  --js   print the residual as JavaScript instead of the IR");
        return ExitCode::FAILURE;
    }
    let path = &args[1];
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let (vm, prog) = match js_frontend::specialize(&src) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    if emit_js {
        match js_frontend::to_js(&src) {
            Ok(js) => {
                println!("--- residual as JavaScript ({} block(s)) ---", prog.blocks.len());
                print!("{js}");
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        println!("--- residual program ({} block(s)) ---", prog.blocks.len());
        print!("{}", vm.dump(&prog));
    }

    let inputs: Vec<i64> = args[2..].iter().filter_map(|s| s.parse().ok()).collect();
    if !inputs.is_empty() {
        println!("\n--- evaluation (residual vs reference) ---");
        for inp in inputs {
            let got = vm.run_residual(&prog, inp);
            let reference = vm.run_reference(inp);
            let ok = if got == reference { "ok" } else { "MISMATCH" };
            println!("  main({inp}) = {got:?}   [{ok}]");
            if got != reference {
                eprintln!("    reference said {reference:?}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
