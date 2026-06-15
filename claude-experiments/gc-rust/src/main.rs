//! gc-rust compiler driver (`gcr`).
//!
//! Usage:
//!   gcr check <file.gcr>     lex + parse + resolve, report errors
//!   gcr parse <file.gcr>     print the item summary
//!
//! Later phases add `build` (AOT) and `run` (JIT).

use std::process::ExitCode;

use gcrust::ast::ItemKind;
use gcrust::codegen::{build_executable, jit_run_i64};
use gcrust::compile::parse_with_prelude;
use gcrust::lexer::lex;
use gcrust::lower::lower_program;
use gcrust::parser::parse_module;
use gcrust::resolve::resolve_module;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gcr <check|parse|run|build> <file.gcr> [-o <out>]");
        return ExitCode::FAILURE;
    }
    let cmd = &args[1];
    let path = &args[2];
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gcr: cannot read {}: {}", path, e);
            return ExitCode::FAILURE;
        }
    };

    // The `parse` subcommand shows only the user's own items; every other path
    // injects the prelude so Option/Result/helpers are in scope without the user
    // declaring them.
    let module = if cmd == "parse" {
        let tokens = match lex(&src) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("gcr: lex error at {:?}: {}", e.span, e.msg);
                return ExitCode::FAILURE;
            }
        };
        match parse_module(&tokens) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("gcr: parse error at {:?}: {}", e.span, e.msg);
                return ExitCode::FAILURE;
            }
        }
    } else {
        match parse_with_prelude(&src) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                return ExitCode::FAILURE;
            }
        }
    };

    match cmd.as_str() {
        "parse" => {
            for item in &module.items {
                let (kind, name) = match &item.kind {
                    ItemKind::Fn(f) => ("fn", f.name.clone()),
                    ItemKind::Struct(s) => (if s.is_value { "value struct" } else { "struct" }, s.name.clone()),
                    ItemKind::Enum(e) => (if e.is_value { "value enum" } else { "enum" }, e.name.clone()),
                    ItemKind::Trait(t) => ("trait", t.name.clone()),
                    ItemKind::Impl(b) => ("impl", format!("{:?}", b.self_ty.kind)),
                    ItemKind::TypeAlias(a) => ("type", a.name.clone()),
                    ItemKind::Const(c) => ("const", c.name.clone()),
                    ItemKind::Mod(m) => ("mod", m.name.clone()),
                    ItemKind::Use(u) => ("use", u.segments.join("::")),
                };
                println!("{kind} {name}");
            }
            ExitCode::SUCCESS
        }
        "check" => match resolve_module(module) {
            Ok(r) => {
                println!("ok: {} top-level symbols", r.globals.by_path.len());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("gcr: resolve error at {:?}: {}", e.span, e.msg);
                ExitCode::FAILURE
            }
        },
        "run" => {
            let resolved = match resolve_module(module) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            let prog = match lower_program(&resolved.globals) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            match jit_run_i64(&prog) {
                Ok(v) => {
                    println!("{}", v);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("gcr: codegen error: {}", e.0);
                    ExitCode::FAILURE
                }
            }
        }
        "build" => {
            // Output path: `gcr build foo.gcr -o foo`. Defaults to the source
            // stem with no extension if `-o` is omitted.
            let out = match parse_output_flag(&args) {
                Ok(o) => o,
                Err(msg) => {
                    eprintln!("gcr: {}", msg);
                    return ExitCode::FAILURE;
                }
            };
            let out = out.unwrap_or_else(|| {
                std::path::Path::new(path)
                    .file_stem()
                    .map(|s| std::path::PathBuf::from(s))
                    .unwrap_or_else(|| std::path::PathBuf::from("a.out"))
            });

            let resolved = match resolve_module(module) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            let prog = match lower_program(&resolved.globals) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            match build_executable(&prog, &out) {
                Ok(()) => {
                    println!("gcr: wrote {}", out.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("gcr: build error: {}", e.0);
                    ExitCode::FAILURE
                }
            }
        }
        other => {
            eprintln!("gcr: unknown command `{}`", other);
            ExitCode::FAILURE
        }
    }
}

/// Parse an optional `-o <path>` (or `--output <path>`) flag from the argument
/// vector. Returns `Ok(Some(path))` when present, `Ok(None)` when absent, and
/// `Err` when `-o` is given without a following value.
fn parse_output_flag(args: &[String]) -> Result<Option<std::path::PathBuf>, String> {
    let mut it = args.iter().skip(3);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--output" => {
                let v = it
                    .next()
                    .ok_or_else(|| "`-o` requires an output path".to_string())?;
                return Ok(Some(std::path::PathBuf::from(v)));
            }
            other => return Err(format!("unexpected argument `{}`", other)),
        }
    }
    Ok(None)
}
