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

    // `emit` has its own argument shape (`gcr emit <stage> <file>`), so it is
    // dispatched before the generic `<cmd> <file>` path resolution below.
    if cmd == "emit" {
        return run_emit(&args);
    }
    // `eval` is the scratchpad/REPL path: it may synthesize a `main` from a
    // bare trailing expression, so it reads + rewrites the source itself rather
    // than going through the normal module parse below.
    if cmd == "eval" {
        return run_eval(&args);
    }
    // `heap-diff` consumes two JSON heap snapshots (from `GCR_HEAP_DUMP=json`),
    // not a `.gcr` source — dispatch it before the source-file path resolution.
    if cmd == "heap-diff" {
        return run_heap_diff(&args);
    }

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
    // `module` + the SourceMap (one entry per source: user / mod files / prelude),
    // attached to the lowered program (run/build) so spans resolve to their real
    // file:line:col.
    let (module, sources) = if cmd == "parse" {
        let tokens = match lex(&src) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("gcr: lex error at {:?}: {}", e.span, e.msg);
                return ExitCode::FAILURE;
            }
        };
        match parse_module(&tokens) {
            Ok(m) => (m, vec![gcrust::core::SourceEntry { path: path.to_string(), text: src.clone() }]),
            Err(e) => {
                eprintln!("gcr: parse error at {:?}: {}", e.span, e.msg);
                return ExitCode::FAILURE;
            }
        }
    } else {
        // Load the file (plus any `mod foo;` sibling files) and inject the prelude.
        match gcrust::compile::parse_file_with_prelude(std::path::Path::new(path)) {
            Ok(ms) => ms,
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
            let mut prog = match lower_program(&resolved.globals) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            // Attach the source so interned spans resolve to file:line:col (alloc
            // sites, and later DWARF). Cheap; only the driver has src+path.
            prog.sources = sources;
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
            // Extra linker args: `--link-arg <arg>` (repeatable) are passed
            // straight to `cc`. FFI programs use this to link native libs — the
            // self-hosting compiler links libLLVM this way.
            let link_args = parse_link_args(&args);
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
            let mut prog = match lower_program(&resolved.globals) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            // Attach source for span→file:line:col resolution (see `run`).
            prog.sources = sources;
            match build_executable(&prog, &out, &link_args) {
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

/// `gcr emit <stage> <file>` — dump a single pipeline stage as JSON (the
/// compiler "X-ray"). `tokens`/`ast` need only lex+parse; `core`/`layout`/`mono`
/// run the full front end (resolve + typecheck + monomorphize) to a Core IR.
/// `gcr heap-diff <before.json> <after.json>` — diff two heap snapshots
/// (`GCR_HEAP_DUMP=json`) to surface growth, the leak-hunting workflow. Reports
/// per-type byte/count deltas (sorted by Δbytes) plus the summary delta. A type
/// whose retained bytes climb across snapshots is the classic leak signature;
/// with two snapshots this shows the delta, and over a series (run repeatedly)
/// a monotonically-growing type is the suspect.
fn run_heap_diff(args: &[String]) -> ExitCode {
    use std::collections::BTreeMap;
    if args.len() < 4 {
        eprintln!("usage: gcr heap-diff <before.json> <after.json>");
        return ExitCode::FAILURE;
    }
    let read = |p: &str| -> Result<serde_json::Value, String> {
        let s = std::fs::read_to_string(p).map_err(|e| format!("cannot read {p}: {e}"))?;
        serde_json::from_str(&s).map_err(|e| format!("invalid snapshot JSON in {p}: {e}"))
    };
    let (a, b) = match (read(&args[2]), read(&args[3])) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("gcr: {e}");
            return ExitCode::FAILURE;
        }
    };

    // (count, bytes) per type name.
    let by_type = |v: &serde_json::Value| -> BTreeMap<String, (i64, i64)> {
        let mut m = BTreeMap::new();
        if let Some(arr) = v["summary"]["by_type"].as_array() {
            for e in arr {
                let name = e["name"].as_str().unwrap_or("?").to_string();
                let count = e["count"].as_i64().unwrap_or(0);
                let bytes = e["bytes"].as_i64().unwrap_or(0);
                m.insert(name, (count, bytes));
            }
        }
        m
    };
    let am = by_type(&a);
    let bm = by_type(&b);

    let mut names: Vec<String> = am.keys().chain(bm.keys()).cloned().collect();
    names.sort();
    names.dedup();
    // (name, dcount, dbytes, a_bytes, b_bytes)
    let mut rows: Vec<(String, i64, i64, i64, i64)> = names
        .into_iter()
        .map(|n| {
            let (ac, ab) = am.get(&n).copied().unwrap_or((0, 0));
            let (bc, bb) = bm.get(&n).copied().unwrap_or((0, 0));
            (n, bc - ac, bb - ab, ab, bb)
        })
        .collect();
    rows.sort_by(|x, y| y.2.cmp(&x.2).then(x.0.cmp(&y.0)));

    let field = |v: &serde_json::Value, k: &str| v["summary"][k].as_i64().unwrap_or(0);
    println!("gc-rust heap-diff: {} -> {}", args[2], args[3]);
    println!(
        "  objects {} -> {} (Δ{:+})   bytes {} -> {} (Δ{:+})   reachable {} -> {}",
        field(&a, "objects"),
        field(&b, "objects"),
        field(&b, "objects") - field(&a, "objects"),
        field(&a, "bytes"),
        field(&b, "bytes"),
        field(&b, "bytes") - field(&a, "bytes"),
        field(&a, "reachable_objects"),
        field(&b, "reachable_objects"),
    );
    println!("  by type (Δbytes desc):");
    println!("  {:>14}  {:>10}  {:<24}  {}", "Δbytes", "Δcount", "type", "a_bytes -> b_bytes");
    for (name, dcount, dbytes, ab, bb) in &rows {
        if *dbytes == 0 && *dcount == 0 {
            continue; // unchanged
        }
        let tag = if *ab == 0 {
            "NEW"
        } else if *bb == 0 {
            "GONE"
        } else if *dbytes > 0 {
            "GREW"
        } else {
            "shrank"
        };
        println!(
            "  {:>+14}  {:>+10}  {:<24}  {} -> {}   [{}]",
            dbytes, dcount, name, ab, bb, tag
        );
    }
    let grown: i64 = rows.iter().filter(|r| r.2 > 0).map(|r| r.2).sum();
    println!(
        "  net growth across {} changed types: {:+} bytes",
        rows.iter().filter(|r| r.1 != 0 || r.2 != 0).count(),
        grown + rows.iter().filter(|r| r.2 < 0).map(|r| r.2).sum::<i64>()
    );
    ExitCode::SUCCESS
}

fn run_emit(args: &[String]) -> ExitCode {
    const USAGE: &str = "usage: gcr emit <tokens|ast|core|layout|reflect|mono|llvm> <file.gcr> [--no-opt (llvm)]";
    let stage = match args.get(2) {
        Some(s) => s.as_str(),
        None => {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let arg_path = match args.get(3) {
        Some(p) => p.clone(),
        None => {
            eprintln!("gcr: emit needs a file. {USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let path = match resolve_entry(&arg_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gcr: {e}");
            return ExitCode::FAILURE;
        }
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gcr: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    match stage {
        // Raw lexer output — no prelude, no parse.
        "tokens" => match lex(&src) {
            Ok(toks) => {
                println!("{}", gcrust::emit::pretty(&toks));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("gcr: lex error at {:?}: {}", e.span, e.msg);
                ExitCode::FAILURE
            }
        },
        // Surface AST of the user's OWN module (prelude omitted, like `gcr parse`,
        // so the dump is the program you wrote, not 200 prelude items).
        "ast" => {
            let toks = match lex(&src) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("gcr: lex error at {:?}: {}", e.span, e.msg);
                    return ExitCode::FAILURE;
                }
            };
            match parse_module(&toks) {
                Ok(m) => {
                    println!("{}", gcrust::emit::pretty(&m));
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("gcr: parse error at {:?}: {}", e.span, e.msg);
                    ExitCode::FAILURE
                }
            }
        }
        // Monomorphic Core IR, its curated layout/mono views, and the LLVM IR.
        // These need the full front end with the prelude injected.
        "core" | "layout" | "reflect" | "mono" | "llvm" => {
            let (module, _) = match gcrust::compile::parse_file_with_prelude(std::path::Path::new(&path)) {
                Ok(ms) => ms,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(&path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            let resolved = match resolve_module(module) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(&path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            if let Err(errs) = gcrust::lower::check_program(&resolved.globals) {
                report_errors(&path, &src, &errs);
                return ExitCode::FAILURE;
            }
            let prog = match lower_program(&resolved.globals) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", gcrust::diag::render(&path, &src, e.span, &e.msg));
                    return ExitCode::FAILURE;
                }
            };
            // `llvm` is raw IR text (not JSON): default to the optimized IR that
            // actually runs; `--no-opt` shows the naive pre-O2 IR.
            if stage == "llvm" {
                let optimize = !args.iter().any(|a| a == "--no-opt");
                return match gcrust::codegen::emit_llvm_ir(&prog, optimize) {
                    Ok(ir) => {
                        println!("{ir}");
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("gcr: codegen error: {}", e.0);
                        ExitCode::FAILURE
                    }
                };
            }
            let out = match stage {
                "core" => gcrust::emit::pretty(&prog),
                "layout" => gcrust::emit::pretty(&gcrust::emit::layouts_view(&prog)),
                "reflect" => gcrust::emit::pretty(&gcrust::emit::reflect_view(&prog)),
                "mono" => gcrust::emit::pretty(&gcrust::emit::mono_table(&prog)),
                _ => unreachable!(),
            };
            println!("{out}");
            ExitCode::SUCCESS
        }
        // No silent stubs: an unsupported stage is a clear, loud error.
        other => {
            eprintln!(
                "gcr: emit stage `{other}` is not implemented. \
                 Available: tokens, ast, core, layout, reflect, mono, llvm. \
                 (resolved/types/asm not yet wired — see src/emit.rs.)"
            );
            ExitCode::FAILURE
        }
    }
}

/// Resolve an `emit` path argument: a plain source file is used as-is; a
/// directory or a `gcr.toml` is read as a project manifest and its `entry` is
/// used. Mirrors the resolution `main` does for the other subcommands.
fn resolve_entry(arg: &str) -> Result<String, String> {
    let p = std::path::Path::new(arg);
    if p.is_dir() || p.file_name().map(|n| n == "gcr.toml").unwrap_or(false) {
        let dir = if p.is_dir() { p } else { p.parent().unwrap_or(std::path::Path::new(".")) };
        let m = gcrust::manifest::Manifest::load(dir).map_err(|e| e.0)?;
        Ok(m.entry_path().to_string_lossy().into_owned())
    } else {
        Ok(arg.to_string())
    }
}

/// `gcr eval <file>` — the scratchpad / REPL evaluator. Compiles + JIT-runs the
/// program and emits a single JSON object describing the result, so the editor
/// scratchpad can render `<value> : <type>` or a compile error inline. When the
/// buffer is (or ends in) a bare expression with no `main`, a `main` is
/// synthesized around it. A compile error is a *result* (reported in the JSON),
/// not a process failure, so this exits SUCCESS unless the file is unreadable.
fn run_eval(args: &[String]) -> ExitCode {
    let arg_path = match args.get(2) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: gcr eval <file.gcr> [--expr-file <f>] [--gc-stress]");
            return ExitCode::FAILURE;
        }
    };
    let path = match resolve_entry(&arg_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gcr: {e}");
            return ExitCode::FAILURE;
        }
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gcr: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let stress = args.iter().any(|a| a == "--gc-stress");
    // `--expr-file <f>`: evaluate the expression in <f> against the definitions
    // in <file> (the file's own `main` is neutralized). This is the editor's
    // "eval the selection" path — the selection is the expression, the buffer
    // is the context. Passed via a file (not an arg) so multi-line selections
    // need no escaping.
    let expr = args.iter().position(|a| a == "--expr-file").and_then(|i| args.get(i + 1));
    let out = match expr {
        Some(ef) => match std::fs::read_to_string(ef) {
            Ok(e) => eval_expr_in_context(&src, e.trim(), stress),
            Err(e) => serde_json::json!({ "ok": false, "stage": "read", "error": format!("expr file: {e}") }),
        },
        None => eval_source(&src, stress),
    };
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".into()));
    ExitCode::SUCCESS
}

/// Evaluate `expr` with the definitions from `context` in scope (the editor's
/// "eval the selection" path). The context's own `main` is renamed so the
/// synthesized expr-`main` is the sole entry point.
fn eval_expr_in_context(context: &str, expr: &str, stress: bool) -> serde_json::Value {
    use serde_json::json;
    if expr.is_empty() {
        return json!({ "ok": true, "kind": "no_result", "message": "empty selection" });
    }
    let program = wrap_expr(&neutralize_main(context), expr);
    match compile_and_run(&program, stress) {
        Ok(v) => json!({ "ok": true, "kind": "value", "value": v.to_string(), "type": "i64", "wrapped": true, "selection": true }),
        Err((stage, msg)) => json!({ "ok": false, "stage": stage, "error": msg, "selection": true }),
    }
}

/// Rename a top-level `fn main` to a dead name so a synthesized expr-`main` can
/// be the sole entry. Matches `fn main` only when not followed by an identifier
/// char (so `fn mainframe` is left alone). No `main` → returned unchanged.
fn neutralize_main(src: &str) -> String {
    const NEEDLE: &str = "fn main";
    let mut start = 0;
    while let Some(rel) = src[start..].find(NEEDLE) {
        let at = start + rel;
        let after = src[at + NEEDLE.len()..].chars().next();
        let is_boundary = after.map_or(true, |c| !(c.is_alphanumeric() || c == '_'));
        if is_boundary {
            return format!("{}fn __scratch_prev_main{}", &src[..at], &src[at + NEEDLE.len()..]);
        }
        start = at + NEEDLE.len();
    }
    src.to_string()
}

/// Decide what to compile (the buffer as-is, or a synthesized `main` around a
/// trailing expression), run it, and return the scratchpad JSON result.
fn eval_source(src: &str, stress: bool) -> serde_json::Value {
    use serde_json::json;
    let (program_src, wrapped, result_ty) = match gcrust::compile::parse_with_prelude(src) {
        // Parses as a whole module: run `main` if present, else it's a
        // definitions-only buffer with nothing to evaluate yet.
        Ok((module, _)) => match main_ret_type(&module) {
            Some(ty) => (src.to_string(), false, ty),
            None => {
                return json!({
                    "ok": true,
                    "kind": "no_result",
                    "message": "type-checks — no `main` or trailing expression to evaluate",
                });
            }
        },
        // Doesn't parse as a module — most likely a trailing expression (not a
        // valid top-level item). Split items from the trailing expression and
        // wrap it in a `main`. If the wrapped form still won't parse, the
        // original error is the honest one (it points into the user's text).
        Err(orig_err) => {
            let (head, tail) = split_trailing_expr(src);
            if tail.is_empty() || gcrust::compile::parse_with_prelude(&wrap_expr(head, tail)).is_err() {
                return json!({ "ok": false, "stage": "parse", "error": orig_err.msg });
            }
            (wrap_expr(head, tail), true, "i64".to_string())
        }
    };

    match compile_and_run(&program_src, stress) {
        Ok(v) => json!({
            "ok": true,
            "kind": "value",
            "value": v.to_string(),
            "type": result_ty,
            "wrapped": wrapped,
        }),
        Err((stage, msg)) => json!({
            "ok": false,
            "stage": stage,
            "error": msg,
            "wrapped": wrapped,
        }),
    }
}

/// Synthesize `<items> fn main() -> i64 { <expr> }`. The result repr is i64
/// because the JIT entry (`jit_run_i64`) returns an i64; a non-i64 expression
/// surfaces as a type error, which the scratchpad reports.
fn wrap_expr(head: &str, expr: &str) -> String {
    format!("{head}\nfn main() -> i64 {{\n{expr}\n}}\n")
}

/// Run the full front end + JIT on a program string. On error, returns the
/// pipeline stage that failed and a concise message (no source rendering — the
/// scratchpad shows this in a one-line strip).
fn compile_and_run(src: &str, stress: bool) -> Result<i64, (String, String)> {
    let (module, _) = gcrust::compile::parse_with_prelude(src).map_err(|e| ("parse".into(), e.msg))?;
    let resolved = resolve_module(module).map_err(|e| ("resolve".into(), e.msg))?;
    if let Err(errs) = gcrust::lower::check_program(&resolved.globals) {
        let msg = errs.iter().map(|e| e.msg.clone()).collect::<Vec<_>>().join("; ");
        return Err(("typecheck".into(), msg));
    }
    let prog = lower_program(&resolved.globals).map_err(|e| ("lower".into(), e.msg))?;
    let res = if stress { jit_run_i64_gc(&prog, true) } else { jit_run_i64(&prog) };
    res.map_err(|e| ("codegen".into(), e.0))
}

/// The declared return type of a user `main`, rendered as a source-level string
/// (e.g. `i64`, `bool`, `Vec<i64>`). `None` when there is no `main`.
fn main_ret_type(module: &gcrust::ast::Module) -> Option<String> {
    module.items.iter().find_map(|it| match &it.kind {
        ItemKind::Fn(f) if f.name == "main" => {
            Some(f.ret.as_ref().map(type_to_string).unwrap_or_else(|| "()".to_string()))
        }
        _ => None,
    })
}

/// Render an AST type as a source-level string for the scratchpad readout.
fn type_to_string(t: &gcrust::ast::Type) -> String {
    use gcrust::ast::TypeKind::*;
    let join = |ts: &[gcrust::ast::Type]| ts.iter().map(type_to_string).collect::<Vec<_>>().join(", ");
    match &t.kind {
        Path(p, generics) => {
            let base = p.segments.join("::");
            if generics.is_empty() { base } else { format!("{base}<{}>", join(generics)) }
        }
        Tuple(ts) => format!("({})", join(ts)),
        Array(elem, _) => format!("[{}; N]", type_to_string(elem)),
        Fn(params, ret) => format!(
            "fn({}){}",
            join(params),
            ret.as_ref().map(|r| format!(" -> {}", type_to_string(r))).unwrap_or_default()
        ),
        ExternFn(params, ret) => format!(
            "extern fn({}){}",
            join(params),
            ret.as_ref().map(|r| format!(" -> {}", type_to_string(r))).unwrap_or_default()
        ),
        SelfType => "Self".to_string(),
    }
}

/// Split a buffer into `(items, trailing_expression)` by brace depth: the
/// trailing expression is everything after the last top-level `}` (i.e. after
/// the final top-level item). With no items, the whole buffer is the
/// expression. Line comments and string literals are skipped so their braces
/// don't throw off the depth count.
fn split_trailing_expr(src: &str) -> (&str, &str) {
    let bytes = src.as_bytes();
    let mut depth: i32 = 0;
    let mut last_top_close = 0usize;
    let mut i = 0;
    let (mut in_line_comment, mut in_str) = (false, false);
    while i < bytes.len() {
        let c = bytes[i];
        if in_line_comment {
            if c == b'\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }
        if in_str {
            match c {
                b'\\' => i += 1, // skip the escaped char
                b'"' => in_str = false,
                _ => {}
            }
            i += 1;
            continue;
        }
        match c {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                in_line_comment = true;
                i += 1;
            }
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    last_top_close = i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    (&src[..last_top_close], src[last_top_close..].trim())
}

/// Collect every `--link-arg <value>` (repeatable) from the argument vector.
/// Passed verbatim to `cc` at link time. Used by FFI programs — notably the
/// self-hosting compiler, which links libLLVM.
fn parse_link_args(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--link-arg" {
            if let Some(v) = it.next() {
                out.push(v.clone());
            }
        }
    }
    out
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
