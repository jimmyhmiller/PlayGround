//! `tally` CLI — the dependent + linear surface language.

use std::process::ExitCode;

fn main() -> ExitCode {
    // The dependent kernel normalizes terms by recursion; deep recursion in a
    // user program (e.g. a big eliminator) wants a deep native stack to type-check
    // and lower. Run the whole driver on a large-stack worker thread. (The runtime
    // it EMITS uses native loops and never recurses on data size.)
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(run_cli)
        .expect("spawn compiler thread")
        .join()
        .expect("compiler thread panicked")
}

fn run_cli() -> ExitCode {
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
            match tally::rust_surface::check_program(&src) {
                Ok(prog) => {
                    let total = prog.totality.iter().filter(|(_, t, _)| *t).count();
                    println!(
                        "ok: {path} type-checks ({} datatype(s), {} def(s)) — no leaks, no use-after-free",
                        prog.sig.datas.len(),
                        prog.defs.len()
                    );
                    println!(
                        "    totality: {total}/{} fn(s) total",
                        prog.totality.len()
                    );
                    for (name, is_total, reason) in &prog.totality {
                        if *is_total {
                            println!("      • {name}: total");
                        } else {
                            println!(
                                "      • {name}: partial — {}",
                                reason.as_deref().unwrap_or("not total")
                            );
                        }
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
        Some("run") => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: tally run <file>");
                return ExitCode::FAILURE;
            };
            let src = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            run_native(&src, path)
        }
        Some("build") => {
            // tally build <file> [-o <out>] [-O0|-O1|-O2|-O3]
            let Some(path) = args.get(2) else {
                eprintln!("usage: tally build <file> [-o <out>] [-O2]");
                return ExitCode::FAILURE;
            };
            let mut out: Option<String> = None;
            let mut opt: u32 = 2;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "-o" => {
                        out = args.get(i + 1).cloned();
                        i += 2;
                    }
                    "-O0" => { opt = 0; i += 1; }
                    "-O1" => { opt = 1; i += 1; }
                    "-O2" => { opt = 2; i += 1; }
                    "-O3" => { opt = 3; i += 1; }
                    other => {
                        eprintln!("tally build: unknown flag `{other}`");
                        return ExitCode::FAILURE;
                    }
                }
            }
            let src = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let out = out.unwrap_or_else(|| {
                std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "a.out".into())
            });
            build_native(&src, path, &out, opt)
        }
        _ => {
            eprintln!(
                "lambda-Tally compiler\nusage:\n  tally check <file>   (type-check: dependent + linear, no leaks/use-after-free)\n  tally run <file>     (type-check + JIT-compile main to native, run it)\n  tally build <file>   (type-check + AOT-compile to a native executable)\n     [-o out] [-O0|-O1|-O2|-O3]"
            );
            ExitCode::FAILURE
        }
    }
}

fn run_native(src: &str, path: &str) -> ExitCode {
    let prog = match tally::rust_surface::check_program(src) {
        Ok(p) => p,
        Err(diags) => {
            eprintln!("{path}: rejected ({} error(s)):", diags.len());
            for d in &diags {
                eprintln!("  - {d}");
            }
            return ExitCode::FAILURE;
        }
    };
    let Some((_, _, body)) = prog.defs.iter().find(|(n, _, _)| n == "main") else {
        eprintln!("{path}: no `main` to run");
        return ExitCode::FAILURE;
    };
    match tally::dep_codegen::run_main(&prog.sig, body) {
        Ok(v) => {
            println!("{path}: type-checks, compiled to native, ran → {v}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{path}: cannot compile to native: {e}");
            ExitCode::FAILURE
        }
    }
}

fn build_native(src: &str, path: &str, out: &str, opt: u32) -> ExitCode {
    use inkwell::OptimizationLevel;
    let prog = match tally::rust_surface::check_program(src) {
        Ok(p) => p,
        Err(diags) => {
            eprintln!("{path}: rejected ({} error(s)):", diags.len());
            for d in &diags {
                eprintln!("  - {d}");
            }
            return ExitCode::FAILURE;
        }
    };
    let Some((_, _, body)) = prog.defs.iter().find(|(n, _, _)| n == "main") else {
        eprintln!("{path}: no `main` to build");
        return ExitCode::FAILURE;
    };
    let opt = match opt {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        3 => OptimizationLevel::Aggressive,
        _ => OptimizationLevel::Default,
    };
    let obj = format!("{out}.o");
    let obj_path = std::path::Path::new(&obj);
    if let Err(e) = tally::dep_codegen::build_object(&prog.sig, body, obj_path, opt) {
        eprintln!("{path}: cannot compile to object: {e}");
        return ExitCode::FAILURE;
    }
    // link the object into a standalone executable with the system C toolchain.
    let status = std::process::Command::new("cc")
        .arg(&obj)
        .arg("-o")
        .arg(out)
        .status();
    match status {
        Ok(s) if s.success() => {
            println!("{path}: type-checks, AOT-compiled → {out}");
            ExitCode::SUCCESS
        }
        Ok(s) => {
            eprintln!("{path}: linker (cc) failed with {s}");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("{path}: cannot invoke linker `cc`: {e}");
            ExitCode::FAILURE
        }
    }
}
