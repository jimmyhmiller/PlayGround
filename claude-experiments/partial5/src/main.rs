//! Partial evaluator for obfuscated JavaScript
//!
//! This tool performs full partial evaluation on JavaScript code
//! that uses control flow flattening (while-switch state machines).

use std::env;
use std::fs;

use partial5::parser::parse_js;
use partial5::statemachine::find_state_machines_in_stmts;
use partial5::trace::{new_env, Evaluator, TracedOp, cleanup_residual_stmts};
use partial5::cfg::{state_machine_to_cfg, print_cfg};
use partial5::emit;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.js> [--trace] [--cfg] [--quiet] [-d var1,var2,...] [--closure-vars var1,var2,...]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --trace              Show execution trace");
        eprintln!("  --cfg                Extract and print CFGs from state machines");
        eprintln!("  --quiet              Reduce debug output");
        eprintln!("  -d vars              Mark variables as dynamic (comma-separated)");
        eprintln!("  --closure-vars vars  Variables to treat as closure captures during callback specialization");
        std::process::exit(1);
    }

    let input_file = &args[1];
    let show_trace = args.contains(&"--trace".to_string());
    let show_cfg = args.contains(&"--cfg".to_string());
    let quiet = args.contains(&"--quiet".to_string());

    // Parse dynamic variables
    let mut dynamic_vars = Vec::new();
    let mut closure_vars = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if arg == "-d" && i + 1 < args.len() {
            for var in args[i + 1].split(',') {
                dynamic_vars.push(var.trim().to_string());
            }
        }
        if arg == "--closure-vars" && i + 1 < args.len() {
            for var in args[i + 1].split(',') {
                closure_vars.push(var.trim().to_string());
            }
        }
    }

    // Read input file
    let source = match fs::read_to_string(input_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", input_file, e);
            std::process::exit(1);
        }
    };

    // Parse JavaScript
    let module = match parse_js(&source) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    if !quiet {
        eprintln!("Parsed {} top-level items", module.body.len());
    }

    // Find state machines (for analysis)
    if show_cfg {
        let stmts: Vec<_> = module
            .body
            .iter()
            .filter_map(|item| {
                if let swc_ecma_ast::ModuleItem::Stmt(stmt) = item {
                    Some(stmt.clone())
                } else {
                    None
                }
            })
            .collect();

        let state_machines = find_state_machines_in_stmts(&stmts);
        eprintln!("Found {} state machines", state_machines.len());

        for (i, (pos, sm)) in state_machines.iter().enumerate() {
            eprintln!();
            eprintln!("State machine #{} at position {}:", i, pos);
            eprintln!("  state_var: {}", sm.state_var);
            eprintln!("  mask: {:?}", sm.mask);
            eprintln!("  cases: {:?}", sm.cases.keys().collect::<Vec<_>>());

            let cfg = state_machine_to_cfg(sm);
            eprintln!();
            eprintln!("{}", print_cfg(&cfg));
        }
    }

    // Create evaluator and execute
    let env = new_env();
    let mut eval = Evaluator::new(env);
    eval.debug = !quiet;

    // Mark dynamic variables
    for var in &dynamic_vars {
        if !quiet {
            eprintln!("Marking '{}' as dynamic", var);
        }
        eval.mark_dynamic(var);
    }

    // Set closure variable hints
    if !closure_vars.is_empty() {
        if !quiet {
            eprintln!("Closure variable hints: {:?}", closure_vars);
        }
        eval.closure_var_hints = closure_vars;
    }

    // Execute all statements
    if !quiet {
        eprintln!();
        eprintln!("Evaluating...");
    }

    for item in &module.body {
        if let swc_ecma_ast::ModuleItem::Stmt(stmt) = item {
            eval.eval_stmt(stmt);

            if eval.should_stop() {
                eprintln!("Warning: Reached max step limit ({})", eval.max_steps);
                break;
            }
        }
    }

    eprintln!("Executed {} steps", eval.steps);

    // Check for milestones
    let mut text_decoder_count = 0;
    let mut text_decoder_decode_count = 0;
    for op in &eval.trace {
        match op {
            TracedOp::TextDecoderNew => text_decoder_count += 1,
            TracedOp::TextDecoderDecode(_) => text_decoder_decode_count += 1,
            _ => {}
        }
    }

    if text_decoder_count > 0 {
        eprintln!("TextDecoder instantiated {} times", text_decoder_count);
    }
    if text_decoder_decode_count > 0 {
        eprintln!("TextDecoder.decode() called {} times", text_decoder_decode_count);
    }

    if show_trace {
        eprintln!();
        eprintln!("Trace ({} operations):", eval.trace.len());
        for (i, op) in eval.trace.iter().take(100).enumerate() {
            eprintln!("  {}: {:?}", i, op);
        }
        if eval.trace.len() > 100 {
            eprintln!("  ... ({} more)", eval.trace.len() - 100);
        }
    }

    // Show final variable values
    eprintln!();
    eprintln!("Final variable values:");
    let env_ref = eval.env.borrow();
    let mut vars: Vec<_> = env_ref.iter().collect();
    vars.sort_by_key(|(k, _)| *k);

    for (name, value) in vars.iter().take(100) {
        eprintln!("  {} = {}", name, value);
    }
    if vars.len() > 100 {
        eprintln!("  ... ({} more variables)", vars.len() - 100);
    }

    // Output the residual program (specialized code)
    // Need to drop env_ref before accessing eval.residual
    drop(env_ref);

    if !eval.residual.is_empty() {
        // Clean up the residual: remove dead stores, simplify expressions
        let cleaned_residual = cleanup_residual_stmts(eval.residual);
        eprintln!();
        eprintln!("Residual program: {} statements", cleaned_residual.len());
        let code = emit::emit_stmts(&cleaned_residual);
        println!("{}", code);
    } else {
        eprintln!();
        eprintln!("No residual code generated (all code was statically evaluated)");
    }
}
