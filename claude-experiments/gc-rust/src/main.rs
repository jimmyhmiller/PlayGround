//! gc-rust compiler driver (`gcr`).
//!
//! Usage:
//!   gcr check <file.gcr>     lex + parse + resolve, report errors
//!   gcr parse <file.gcr>     print the item summary
//!
//! Later phases add `build` (AOT) and `run` (JIT).

use std::process::ExitCode;

use gcrust::ast::ItemKind;
use gcrust::codegen::{build_executable, jit_run_i64, jit_run_i64_gc};
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
    let arg_path = &args[2];

    // Resolve the entry file. If the argument is a directory (or a `gcr.toml`),
    // read the manifest and use its `entry`; otherwise it's a source file path.
    let path: String = {
        let p = std::path::Path::new(arg_path);
        if p.is_dir() || p.file_name().map(|n| n == "gcr.toml").unwrap_or(false) {
            let dir = if p.is_dir() { p } else { p.parent().unwrap_or(std::path::Path::new(".")) };
            match gcrust::manifest::Manifest::load(dir) {
                Ok(m) => {
                    println!("gcr: building `{}` v{}", m.name, m.version);
                    m.entry_path().to_string_lossy().into_owned()
                }
                Err(e) => {
                    eprintln!("gcr: {}", e.0);
                    return ExitCode::FAILURE;
                }
            }
        } else {
            arg_path.clone()
        }
    };
    let path = &path;

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
        // Load the file (plus any `mod foo;` sibling files) and inject the prelude.
        match gcrust::compile::parse_file_with_prelude(std::path::Path::new(path)) {
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
        "check" => {
            // `check` runs the full front end: resolve + typecheck + monomorphize.
            // It first runs the multi-error check pass (every non-generic function
            // checked independently), so it can report ALL broken functions at
            // once, then the demand-driven lower pass to catch anything reachable
            // only through a generic instantiation.
            let resolved = match resolve_module(module) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            let nsyms = resolved.globals.by_path.len();
            if let Err(errs) = gcrust::lower::check_program(&resolved.globals) {
                report_errors(path, &src, &errs);
                return ExitCode::FAILURE;
            }
            match lower_program(&resolved.globals) {
                Ok(_) => {
                    println!("ok: {} top-level symbols, type-checks", nsyms);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    ExitCode::FAILURE
                }
            }
        }
        "run" => {
            let resolved = match resolve_module(module) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            if let Err(errs) = gcrust::lower::check_program(&resolved.globals) {
                report_errors(path, &src, &errs);
                return ExitCode::FAILURE;
            }
            let prog = match lower_program(&resolved.globals) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            // `--gc-stress` forces a collection at every allocation — the
            // strongest test that the precise relocating GC keeps roots correct.
            let stress = args.iter().any(|a| a == "--gc-stress");
            let result = if stress { jit_run_i64_gc(&prog, true) } else { jit_run_i64(&prog) };
            match result {
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
            if let Err(errs) = gcrust::lower::check_program(&resolved.globals) {
                report_errors(path, &src, &errs);
                return ExitCode::FAILURE;
            }
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

/// Render every collected error (the multi-error check pass), each with its own
/// caret, separated by blank lines, then a summary count.
fn report_errors(path: &str, src: &str, errs: &[gcrust::lower::LowerError]) {
    for e in errs {
        eprintln!("{}\n", gcrust::diag::render(path, src, e.span, &e.msg));
    }
    let n = errs.len();
    eprintln!("gcr: {} error{} found", n, if n == 1 { "" } else { "s" });
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
