//! gc-rust compiler driver (`gcr`).
//!
//! Usage:
//!   gcr check <file.gcr>     lex + parse + resolve, report errors
//!   gcr parse <file.gcr>     print the item summary
//!
//! Later phases add `build` (AOT) and `run` (JIT).

use std::process::ExitCode;

use gcrust::ast::ItemKind;
use gcrust::codegen::jit_run_i64;
use gcrust::lexer::lex;
use gcrust::lower::lower_program;
use gcrust::parser::parse_module;
use gcrust::resolve::resolve_module;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gcr <check|parse> <file.gcr>");
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

    let tokens = match lex(&src) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("gcr: lex error at {:?}: {}", e.span, e.msg);
            return ExitCode::FAILURE;
        }
    };
    let module = match parse_module(&tokens) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gcr: parse error at {:?}: {}", e.span, e.msg);
            return ExitCode::FAILURE;
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
                    eprintln!("gcr: resolve error at {:?}: {}", e.span, e.msg);
                    return ExitCode::FAILURE;
                }
            };
            let prog = match lower_program(&resolved.globals) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("gcr: type error at {:?}: {}", e.span, e.msg);
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
        other => {
            eprintln!("gcr: unknown command `{}`", other);
            ExitCode::FAILURE
        }
    }
}
