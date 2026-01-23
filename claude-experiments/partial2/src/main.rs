use std::env;
use std::fs;
use std::process;

use oxc_allocator::Allocator;

use jspartial::eval::Evaluator;
use jspartial::env::Value;
use jspartial::parse;
use jspartial::residual::emit_residual;

fn print_help() {
    eprintln!("jspartial - JavaScript Partial Evaluator");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    jspartial [OPTIONS] <input.js>");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("    -d, --dynamic <name>    Mark a variable as dynamic (can be repeated)");
    eprintln!("    -r, --max-recursion <n> Maximum recursion depth (default: 100)");
    eprintln!("    --debug-bindings        Output variable bindings as JSON (instead of residual)");
    eprintln!("    --debug-pretty          Output bindings as annotated source (static=green, dynamic=red)");
    eprintln!("    --trace <file>          Write execution trace to JSON file");
    eprintln!("    -h, --help              Print this help message");
    eprintln!("    -v, --verbose           Show additional information");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("    jspartial program.js");
    eprintln!("    jspartial -d input -d config program.js");
    eprintln!("    jspartial --dynamic userInput script.js");
    eprintln!("    jspartial -r 500 program.js");
}

struct Args {
    filename: String,
    dynamic_vars: Vec<String>,
    verbose: bool,
    max_recursion_depth: usize,
    debug_bindings: bool,
    debug_pretty: bool,
    trace_file: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        return Err("No input file specified".to_string());
    }

    let mut dynamic_vars = Vec::new();
    let mut verbose = false;
    let mut max_recursion_depth: usize = 100;
    let mut debug_bindings = false;
    let mut debug_pretty = false;
    let mut trace_file = None;
    let mut filename = None;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            }
            "-v" | "--verbose" => {
                verbose = true;
                i += 1;
            }
            "-d" | "--dynamic" => {
                if i + 1 >= args.len() {
                    return Err(format!("Option {} requires a variable name", args[i]));
                }
                dynamic_vars.push(args[i + 1].clone());
                i += 2;
            }
            "-r" | "--max-recursion" => {
                if i + 1 >= args.len() {
                    return Err(format!("Option {} requires a number", args[i]));
                }
                max_recursion_depth = args[i + 1].parse()
                    .map_err(|_| format!("Invalid recursion depth: {}", args[i + 1]))?;
                i += 2;
            }
            "--debug-bindings" => {
                debug_bindings = true;
                i += 1;
            }
            "--debug-pretty" => {
                debug_pretty = true;
                i += 1;
            }
            "--trace" => {
                if i + 1 >= args.len() {
                    return Err(format!("Option {} requires a file path", args[i]));
                }
                trace_file = Some(args[i + 1].clone());
                i += 2;
            }
            arg if arg.starts_with('-') => {
                return Err(format!("Unknown option: {}", arg));
            }
            _ => {
                if filename.is_some() {
                    return Err("Multiple input files not supported".to_string());
                }
                filename = Some(args[i].clone());
                i += 1;
            }
        }
    }

    match filename {
        Some(f) => Ok(Args {
            filename: f,
            dynamic_vars,
            verbose,
            max_recursion_depth,
            debug_bindings,
            debug_pretty,
            trace_file,
        }),
        None => Err("No input file specified".to_string()),
    }
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            print_help();
            process::exit(1);
        }
    };

    // Read source file
    let source = match fs::read_to_string(&args.filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading '{}': {}", args.filename, e);
            process::exit(1);
        }
    };

    if args.verbose {
        eprintln!("--- Input ---");
        eprintln!("{}", source);
        if !args.dynamic_vars.is_empty() {
            eprintln!("--- Dynamic variables ---");
            for var in &args.dynamic_vars {
                eprintln!("  {}", var);
            }
        }
        eprintln!();
    }

    // Parse the source
    let allocator = Allocator::default();
    let program = parse(&allocator, &source);

    // Create evaluator and mark dynamic variables
    let mut evaluator = Evaluator::new();
    evaluator.max_recursion_depth = args.max_recursion_depth;

    // Enable tracing if requested
    if args.trace_file.is_some() {
        evaluator.trace_enabled = true;
        evaluator.set_source(&source);
    }

    for var in &args.dynamic_vars {
        evaluator.env.define(var, Value::Dynamic(var.clone()));
    }

    // Evaluate the program
    if let Err(e) = evaluator.eval_program(&program) {
        eprintln!("Evaluation error: {}", e);
        process::exit(1);
    }

    // Write trace file if requested
    if let Some(trace_path) = &args.trace_file {
        let trace_json = evaluator.execution_trace.to_json_with_source(Some(&source));
        if let Err(e) = fs::write(trace_path, &trace_json) {
            eprintln!("Error writing trace file '{}': {}", trace_path, e);
            process::exit(1);
        }
        eprintln!("Wrote {} trace events to {}", evaluator.execution_trace.entries.len(), trace_path);
    }

    // Output debug bindings if requested
    if args.debug_bindings {
        println!("{}", evaluator.debug_bindings_json());
        return;
    }

    // Output pretty debug if requested
    if args.debug_pretty {
        println!("{}", evaluator.debug_pretty_print());
        return;
    }

    // Generate residual code
    let residual = match emit_residual(&evaluator.trace, true) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Residual generation error: {}", e);
            process::exit(1);
        }
    };

    if args.verbose {
        eprintln!("--- Residual ---");
    }

    println!("{}", residual);
}
