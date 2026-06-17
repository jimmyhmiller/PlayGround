//! autostruct — a little language with no declared data types.
//!
//! You write programs against abstract collections (`collection()`) using
//! high-level operations. The compiler analyzes how each collection is *used*
//! and picks the best concrete data structure for it. Then it either runs the
//! program with those structures or compiles it to specialized JavaScript.

mod analysis;
mod ast;
mod codegen_js;
mod interp;
mod lexer;
mod parser;
mod value;

use std::process::exit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        exit(2);
    }

    let (cmd, file) = match args[1].as_str() {
        "run" | "analyze" | "js" | "bench" => {
            if args.len() < 3 {
                eprintln!("error: '{}' needs a file argument", args[1]);
                usage();
                exit(2);
            }
            (args[1].as_str(), args[2].as_str())
        }
        // Allow `autostruct <file>` as shorthand for `run`.
        maybe_file => ("run", maybe_file),
    };

    let naive = args.iter().any(|a| a == "--naive");

    if let Err(e) = real_main(cmd, file, naive) {
        eprintln!("error: {}", e);
        exit(1);
    }
}

fn real_main(cmd: &str, file: &str, naive: bool) -> Result<(), String> {
    let src = std::fs::read_to_string(file).map_err(|e| format!("cannot read {}: {}", file, e))?;
    let toks = lexer::lex(&src)?;
    let program = parser::parse(toks)?;
    let analysis = analysis::analyze(&program);

    match cmd {
        "analyze" => {
            print_report(&analysis);
            Ok(())
        }
        "run" => {
            if !naive {
                print_report(&analysis);
                println!("──────────────────────────────────────────────");
                println!("output:");
            }
            let mut interp = interp::Interp::new(&analysis, naive);
            interp.run(&program)?;
            for line in &interp.output {
                println!("{}", line);
            }
            Ok(())
        }
        "js" => {
            let js = codegen_js::generate(&program, &analysis);
            print!("{}", js);
            Ok(())
        }
        "bench" => run_bench(&program, &analysis),
        _ => {
            usage();
            Ok(())
        }
    }
}

fn print_report(analysis: &analysis::Analysis) {
    println!("══════════════════════════════════════════════");
    println!(" data-structure inference report");
    println!("══════════════════════════════════════════════");
    if analysis.selections.is_empty() {
        println!(" (no inferred collections in this program)");
        return;
    }
    for sel in &analysis.selections {
        let ops: Vec<&str> = sel.ops.iter().map(|s| s.as_str()).collect();
        println!();
        println!(" collection `{}`", sel.name);
        println!(
            "   used as : {}",
            if ops.is_empty() {
                "(unused)".to_string()
            } else {
                ops.join(", ")
            }
        );
        println!("   reason  : {}", sel.reason);
        println!("   chosen  : {}", sel.specialized);
        println!("   (naive  : {})", sel.naive);
    }
}

/// Run the program twice — specialized vs. naive — and report the speedup.
fn run_bench(program: &ast::Program, analysis: &analysis::Analysis) -> Result<(), String> {
    print_report(analysis);
    println!("──────────────────────────────────────────────");
    println!("benchmark: specialized structures vs. naive linear structures");
    println!();

    // Warm + measure naive.
    let start = Instant::now();
    let mut naive = interp::Interp::new(analysis, true);
    naive.run(program)?;
    let naive_time = start.elapsed();

    let start = Instant::now();
    let mut spec = interp::Interp::new(analysis, false);
    spec.run(program)?;
    let spec_time = start.elapsed();

    if naive.output != spec.output {
        return Err("internal: specialized and naive runs disagree on output!".into());
    }

    println!("program output:");
    for line in &spec.output {
        println!("  {}", line);
    }
    println!();
    println!("  naive (linear structures) : {:?}", naive_time);
    println!("  specialized (inferred)    : {:?}", spec_time);
    let ratio = naive_time.as_secs_f64() / spec_time.as_secs_f64().max(1e-9);
    println!("  speedup                   : {:.1}x faster", ratio);
    Ok(())
}

fn usage() {
    eprintln!(
        "autostruct — infer the best data structure from usage

USAGE:
    autostruct <command> <file.shape> [--naive]

COMMANDS:
    run <file>       analyze, then run with the inferred data structures
    run <file> --naive   run with naive linear structures instead
    analyze <file>   print the inference report only
    js <file>        compile to specialized JavaScript (prints to stdout)
    bench <file>     run specialized vs. naive and report the speedup

You can also write `autostruct <file>` as shorthand for `run`."
    );
}
