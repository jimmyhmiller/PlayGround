use std::collections::HashMap;
use std::env;
use std::fs;

use partial3::ast::{pretty_print, Expr};
use partial3::color_print::{pretty_print_colored, print_legend};
use partial3::js_to_lisp;
use partial3::opaque::OpaqueRegistry;
use partial3::parse;
use partial3::partial::{new_penv, partial_eval, set_gas, with_opaque_registry, PValue};
use partial3::value::Value;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <file.lisp|file.js> [OPTIONS]", args[0]);
        eprintln!();
        eprintln!("Supports both Lisp (.lisp) and JavaScript (.js) files.");
        eprintln!("JavaScript files are automatically transpiled to Lisp before partial evaluation.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --dynamic var1,var2,...  Mark variables as dynamic (unknown at compile time)");
        eprintln!("  --gas N                  Set gas limit for evaluation (default: 10000000)");
        eprintln!("  --show-lisp              Show the Lisp representation (useful for JS files)");
        eprintln!("  --opaque                 Show opaque calls that receive static arguments");
        eprintln!("  --builtins               Enable built-in handlers for JS constructors");
        eprintln!("  --color                  Colorize output (green=static, yellow=dynamic)");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} examples/static.lisp", args[0]);
        eprintln!("  {} examples/closure.js --dynamic input", args[0]);
        eprintln!("  {} examples/power.js --show-lisp", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];

    // Parse flags
    let mut dynamic_vars: Vec<String> = Vec::new();
    let mut show_lisp = false;
    let mut show_opaque = false;
    let mut use_builtins = false;
    #[allow(unused_assignments)]
    let mut use_color = false;
    let mut parse_only = false;
    let mut gas_limit: Option<usize> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dynamic" => {
                if i + 1 < args.len() {
                    dynamic_vars.extend(args[i + 1].split(',').map(|s| s.trim().to_string()));
                    i += 2;
                } else {
                    eprintln!("Error: --dynamic requires a comma-separated list of variable names");
                    std::process::exit(1);
                }
            }
            "--show-lisp" => {
                show_lisp = true;
                i += 1;
            }
            "--opaque" => {
                show_opaque = true;
                i += 1;
            }
            "--builtins" => {
                use_builtins = true;
                i += 1;
            }
            "--color" => {
                use_color = true;
                i += 1;
            }
            "--parse-only" => {
                parse_only = true;
                i += 1;
            }
            "--gas" => {
                if i + 1 < args.len() {
                    match args[i + 1].replace("_", "").parse::<usize>() {
                        Ok(n) => gas_limit = Some(n),
                        Err(_) => {
                            eprintln!("Error: --gas requires a positive integer (e.g., --gas 1000000)");
                            std::process::exit(1);
                        }
                    }
                    i += 2;
                } else {
                    eprintln!("Error: --gas requires a value (e.g., --gas 1000000)");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // Read the file
    let input = match fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", filename, e);
            std::process::exit(1);
        }
    };

    // Parse based on file extension
    let is_js = filename.ends_with(".js");
    let expr = if is_js {
        match js_to_lisp::js_to_lisp(&input) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("JavaScript parse error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        match parse::parse(&input) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Lisp parse error: {}", e);
                std::process::exit(1);
            }
        }
    };

    // Optionally show Lisp representation
    if show_lisp {
        println!("Lisp representation:");
        println!("{}", expr);
        println!();
    }

    // If parse-only, we're done
    if parse_only {
        println!("Parse successful! Expression size: {} chars", format!("{}", expr).len());
        return;
    }

    // Build initial partial environment
    let penv = new_penv();
    for var in &dynamic_vars {
        penv.borrow_mut()
            .insert(var.clone(), PValue::Dynamic(Expr::Var(var.clone())));
    }

    // Set gas limit (default: 10 million operations)
    if let Some(gas) = gas_limit {
        set_gas(gas);
    } else {
        set_gas(10_000_000);
    }

    // Perform partial evaluation (with or without builtins)
    let result = if use_builtins {
        let registry = OpaqueRegistry::with_builtins();
        eprintln!("Using builtin handlers: {:?}", registry.handler_names());
        with_opaque_registry(registry, || partial_eval(&expr, &penv))
    } else {
        partial_eval(&expr, &penv)
    };

    // Output result
    match result {
        PValue::Static(v) => {
            println!("Fully evaluated:");
            print_value(&v);
        }
        PValue::StaticNamed { value, .. } => {
            // Named static value - still fully evaluated, just has a name attached
            println!("Fully evaluated:");
            print_value(&value);
        }
        PValue::Dynamic(ref e) | PValue::DynamicNamed { expr: ref e, .. } => {
            println!("Residual program:");
            if use_color {
                print_legend();
                println!();
                println!("{}", pretty_print_colored(&result));
            } else {
                println!("{}", pretty_print(e));
            }

            if show_opaque {
                println!();
                println!("=== Opaque calls with static arguments ===");
                let mut opaque_calls: HashMap<String, usize> = HashMap::new();
                collect_opaque_calls(e, &mut opaque_calls);

                if opaque_calls.is_empty() {
                    println!("(none found)");
                } else {
                    let mut sorted: Vec<_> = opaque_calls.iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
                    for (call, count) in sorted {
                        println!("  {:4}x  {}", count, call);
                    }
                }
            }
        }
    }
}

/// Check if an expression is "static" (a literal value)
fn is_static_expr(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Undefined | Expr::Null
    )
}

/// Collect opaque calls (new, method calls) that have static arguments
fn collect_opaque_calls(expr: &Expr, calls: &mut HashMap<String, usize>) {
    match expr {
        // new Constructor(args) - report if all args are static (including zero args)
        Expr::New(ctor, args) => {
            let all_static = args.iter().all(|a| is_static_expr(a));
            if all_static {
                let ctor_name = match ctor.as_ref() {
                    Expr::Var(name) => name.clone(),
                    Expr::PropAccess(_, prop) => prop.clone(),
                    other => other.to_string(),
                };
                let key = if args.is_empty() {
                    format!("(new {})", ctor_name)
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| a.to_string()).collect();
                    format!("(new {} {})", ctor_name, arg_strs.join(" "))
                };
                *calls.entry(key).or_insert(0) += 1;
            }
            // Recurse into constructor and args
            collect_opaque_calls(ctor, calls);
            for arg in args {
                collect_opaque_calls(arg, calls);
            }
        }

        // Method calls: (call (prop obj "method") args)
        Expr::Call(callee, args) => {
            if let Expr::PropAccess(_obj, method) = callee.as_ref() {
                let static_args: Vec<_> = args.iter().filter(|a| is_static_expr(a)).collect();
                if !static_args.is_empty() {
                    let static_arg_strs: Vec<_> = static_args.iter().map(|a| a.to_string()).collect();
                    let key = format!("(call .{} [static: {}])", method, static_arg_strs.join(", "));
                    *calls.entry(key).or_insert(0) += 1;
                }
            }
            // Recurse
            collect_opaque_calls(callee, calls);
            for arg in args {
                collect_opaque_calls(arg, calls);
            }
        }

        // Recurse into all other expression types
        Expr::BinOp(_, left, right) => {
            collect_opaque_calls(left, calls);
            collect_opaque_calls(right, calls);
        }
        Expr::If(cond, then_b, else_b) => {
            collect_opaque_calls(cond, calls);
            collect_opaque_calls(then_b, calls);
            collect_opaque_calls(else_b, calls);
        }
        Expr::Let(_, value, body) => {
            collect_opaque_calls(value, calls);
            collect_opaque_calls(body, calls);
        }
        Expr::Fn(_, body) => {
            collect_opaque_calls(body, calls);
        }
        Expr::Begin(exprs) => {
            for e in exprs {
                collect_opaque_calls(e, calls);
            }
        }
        Expr::While(cond, body) => {
            collect_opaque_calls(cond, calls);
            collect_opaque_calls(body, calls);
        }
        Expr::For { init, cond, update, body } => {
            if let Some(i) = init {
                collect_opaque_calls(i, calls);
            }
            if let Some(c) = cond {
                collect_opaque_calls(c, calls);
            }
            if let Some(u) = update {
                collect_opaque_calls(u, calls);
            }
            collect_opaque_calls(body, calls);
        }
        Expr::Set(_, value) => {
            collect_opaque_calls(value, calls);
        }
        Expr::Array(elems) => {
            for e in elems {
                collect_opaque_calls(e, calls);
            }
        }
        Expr::Index(arr, idx) => {
            collect_opaque_calls(arr, calls);
            collect_opaque_calls(idx, calls);
        }
        Expr::Len(arr) => {
            collect_opaque_calls(arr, calls);
        }
        Expr::Object(props) => {
            for (_, v) in props {
                collect_opaque_calls(v, calls);
            }
        }
        Expr::PropAccess(obj, _) => {
            collect_opaque_calls(obj, calls);
        }
        Expr::PropSet(obj, _, value) => {
            collect_opaque_calls(obj, calls);
            collect_opaque_calls(value, calls);
        }
        Expr::ComputedAccess(obj, key) => {
            collect_opaque_calls(obj, calls);
            collect_opaque_calls(key, calls);
        }
        Expr::ComputedSet(obj, key, value) => {
            collect_opaque_calls(obj, calls);
            collect_opaque_calls(key, calls);
            collect_opaque_calls(value, calls);
        }
        Expr::LogNot(inner) => {
            collect_opaque_calls(inner, calls);
        }
        Expr::BitNot(inner) => {
            collect_opaque_calls(inner, calls);
        }
        Expr::Throw(inner) => {
            collect_opaque_calls(inner, calls);
        }
        Expr::Switch { discriminant, cases, default } => {
            collect_opaque_calls(discriminant, calls);
            for (cv, body) in cases {
                collect_opaque_calls(cv, calls);
                for e in body {
                    collect_opaque_calls(e, calls);
                }
            }
            if let Some(d) = default {
                for e in d {
                    collect_opaque_calls(e, calls);
                }
            }
        }
        Expr::TryCatch { try_block, catch_block, finally_block, .. } => {
            collect_opaque_calls(try_block, calls);
            collect_opaque_calls(catch_block, calls);
            if let Some(fb) = finally_block {
                collect_opaque_calls(fb, calls);
            }
        }
        // Leaf nodes - no recursion needed
        Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Var(_) |
        Expr::Undefined | Expr::Null | Expr::Break | Expr::Continue | Expr::Opaque(_) => {}
    }
}

fn print_value(v: &Value) {
    match v {
        Value::Int(n) => println!("{}", n),
        Value::Bool(b) => println!("{}", b),
        Value::String(s) => println!("\"{}\"", s),
        Value::Undefined => println!("undefined"),
        Value::Null => println!("null"),
        Value::Array(elements) => {
            print!("[");
            for (i, elem) in elements.borrow().iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print_value_inline(elem);
            }
            println!("]");
        }
        Value::Object(obj) => {
            print!("{{");
            let borrowed = obj.borrow();
            for (i, (k, val)) in borrowed.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}: ", k);
                print_value_inline(val);
            }
            println!("}}");
        }
        Value::Closure { params, body, .. } => {
            println!("(fn ({}) {})", params.join(" "), body);
        }
        Value::Opaque { label, .. } => {
            println!("<opaque: {}>", label);
        }
    }
}

fn print_value_inline(v: &Value) {
    match v {
        Value::Int(n) => print!("{}", n),
        Value::Bool(b) => print!("{}", b),
        Value::String(s) => print!("\"{}\"", s),
        Value::Undefined => print!("undefined"),
        Value::Null => print!("null"),
        Value::Array(elements) => {
            print!("[");
            for (i, elem) in elements.borrow().iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print_value_inline(elem);
            }
            print!("]");
        }
        Value::Object(obj) => {
            print!("{{");
            let borrowed = obj.borrow();
            for (i, (k, val)) in borrowed.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}: ", k);
                print_value_inline(val);
            }
            print!("}}");
        }
        Value::Closure { params, body, .. } => {
            print!("(fn ({}) {})", params.join(" "), body);
        }
        Value::Opaque { label, .. } => {
            print!("<opaque: {}>", label);
        }
    }
}
