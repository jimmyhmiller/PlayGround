//! dataless — a modern realization of R. M. Balzer's "Dataless Programming"
//! (RAND Memorandum RM-5290-ARPA, 1967).
//!
//! A program is written once, in terms of logic alone, using a single canonical
//! reference form `name(handle)` for both data and functions. The data
//! *representation* lives in a separate declarations file and can be swapped
//! (ARRAY <-> LIST <-> DOUBLE_LIST, or stored field <-> computed function)
//! without changing one character of the program. Same program, same output;
//! only the performance and storage change.

mod ast;
mod interp;
mod lexer;
mod parser;
mod repr;
mod value;

use std::process::exit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        exit(2);
    }
    if let Err(e) = run(&args) {
        eprintln!("error: {}", e);
        exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    match args[1].as_str() {
        "run" => {
            let prog = req(args, 2, "run")?;
            let decl = flag(args, "--decl").ok_or("`run` needs --decl <file>")?;
            let trace = args.iter().any(|a| a == "--trace");
            cmd_run(&prog, &decl, trace)
        }
        "compare" => {
            // compare <program> --decl a.decl --decl b.decl [--decl c.decl ...]
            let prog = req(args, 2, "compare")?;
            let decls = flags(args, "--decl");
            if decls.len() < 2 {
                return Err("`compare` needs at least two --decl <file> arguments".into());
            }
            cmd_compare(&prog, &decls)
        }
        "decls" => {
            let decl = req(args, 2, "decls")?;
            cmd_decls(&decl)
        }
        _ => {
            usage();
            Err(format!("unknown command `{}`", args[1]))
        }
    }
}

fn cmd_run(prog_path: &str, decl_path: &str, trace: bool) -> Result<(), String> {
    let (program, decls) = load(prog_path, decl_path)?;
    let mut interp = interp::Interp::from_decls(&decls)?;
    let start = Instant::now();
    interp.run(&program)?;
    let elapsed = start.elapsed();
    for line in &interp.output {
        println!("{}", line);
    }
    if trace {
        eprintln!("──────────────────────────────────────────────");
        eprintln!("representation trace ({}):", decl_path);
        for c in &interp.colls {
            eprintln!(
                "  collection `{}` as {}: {} members, {} positional steps",
                c.name,
                c.rep.name(),
                c.len(),
                c.steps
            );
        }
        eprintln!("  total positional/shift steps: {}", interp.total_steps());
        eprintln!("  wall time: {:?}", elapsed);
    }
    Ok(())
}

/// Run the SAME program under several declaration files and prove the output is
/// identical while the cost differs. This is the paper's central claim, made
/// executable.
fn cmd_compare(prog_path: &str, decl_paths: &[String]) -> Result<(), String> {
    let src = read(prog_path)?;
    let program = parser::parse_program(lexer::lex(&src)?)?;

    println!("program : {}", prog_path);
    println!("claim   : identical output across all representations; only cost differs.");
    println!("──────────────────────────────────────────────");

    let mut baseline: Option<Vec<String>> = None;
    let mut all_match = true;

    for path in decl_paths {
        let dsrc = read(path)?;
        let decls = parser::parse_decls(lexer::lex(&dsrc)?)?;
        let mut interp = interp::Interp::from_decls(&decls)?;
        let start = Instant::now();
        interp.run(&program)?;
        let elapsed = start.elapsed();

        let reps: Vec<String> = interp
            .colls
            .iter()
            .map(|c| format!("{}={}", c.name, c.rep.name()))
            .collect();
        println!("declarations: {}", path);
        println!("  representations : {}", reps.join(", "));
        println!(
            "  positional steps: {}   wall: {:?}",
            interp.total_steps(),
            elapsed
        );

        match &baseline {
            None => baseline = Some(interp.output.clone()),
            Some(b) => {
                if *b != interp.output {
                    all_match = false;
                    println!("  >>> OUTPUT DIFFERS FROM BASELINE <<<");
                }
            }
        }
    }

    println!("──────────────────────────────────────────────");
    if all_match {
        println!("RESULT: every representation produced identical output. ✓");
        println!();
        println!("output:");
        for line in baseline.unwrap_or_default() {
            println!("  {}", line);
        }
        Ok(())
    } else {
        Err("representations disagreed on output (this should never happen)".into())
    }
}

fn cmd_decls(decl_path: &str) -> Result<(), String> {
    let dsrc = read(decl_path)?;
    let decls = parser::parse_decls(lexer::lex(&dsrc)?)?;
    println!("declarations in {}:", decl_path);
    for d in &decls {
        match d {
            ast::Decl::Collection { name, rep, fields } => {
                println!("  collection `{}` as {}", name, rep.name());
                for f in fields {
                    println!("      {} : {:?}", f.name, f.ty);
                }
            }
            ast::Decl::Computed { name, param, .. } => {
                println!("  computed `{}({})`  (a data reference backed by a function)", name, param);
            }
        }
    }
    Ok(())
}

fn load(prog_path: &str, decl_path: &str) -> Result<(ast::Program, ast::Decls), String> {
    let psrc = read(prog_path)?;
    let dsrc = read(decl_path)?;
    let program = parser::parse_program(lexer::lex(&psrc)?)?;
    let decls = parser::parse_decls(lexer::lex(&dsrc)?)?;
    Ok((program, decls))
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {}", path, e))
}

fn req(args: &[String], i: usize, cmd: &str) -> Result<String, String> {
    args.get(i)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("`{}` needs a file argument", cmd))
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}

fn flags(args: &[String], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..args.len() {
        if args[i] == name {
            if let Some(v) = args.get(i + 1) {
                out.push(v.clone());
            }
        }
    }
    out
}

fn usage() {
    eprintln!(
        "dataless — Balzer's Dataless Programming (RM-5290), made executable

USAGE:
    dataless run     <program.dl> --decl <decls.decl> [--trace]
    dataless compare <program.dl> --decl <a.decl> --decl <b.decl> [--decl ...]
    dataless decls   <decls.decl>

The same <program.dl> runs against any declarations. `compare` runs it under
several declaration files and proves the output is identical while the cost
differs — the central claim of Dataless Programming."
    );
}
