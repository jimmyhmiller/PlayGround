//! gc-rust compiler driver (`gcr`).
//!
//! Usage:
//!   gcr check <file.gcr>     lex + parse + resolve, report errors
//!   gcr parse <file.gcr>     print the item summary
//!
//! Later phases add `build` (AOT) and `run` (JIT).

use std::process::ExitCode;

use gcrust::ast::ItemKind;
use gcrust::codegen::{build_executable_level, jit_run_i64, jit_run_i64_gc};
use gcrust::lexer::lex;
use gcrust::lower::lower_program;
use gcrust::parser::parse_module;
use gcrust::resolve::resolve_module;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gcr <new|check|parse|run|build|emit|eval|heap|heap-diff> [file.gcr | project-dir]");
        return ExitCode::FAILURE;
    }
    let cmd = &args[1];

    // `new` scaffolds a fresh project directory (`gcr.toml` + `src/main.gcr`).
    if cmd == "new" {
        return run_new(&args);
    }
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
    // `heap` runs a program and emits its program-end heap snapshot as JSON — the
    // data source for `gcr_heap.ft` (the heap-explorer widget). Own dispatch so
    // it can redirect the dump cleanly to `--out` (the widget reads that file).
    if cmd == "heap" {
        return run_heap(&args);
    }
    // `bench` runs ANY gc-rust program(s) and emits a general metrics schema
    // (runtime + compile + size + GC cycles/pauses + allocation churn + peak
    // heap) — the data source for the composable bench toolkit. Own dispatch:
    // it builds + runs each program itself.
    if cmd == "bench" {
        return run_bench(&args);
    }

    // Resolve the entry file and (when in a project) its manifest. The first
    // non-flag token after the subcommand is the path: a `.gcr` file, a project
    // directory, or a `gcr.toml`. With NO path, discover a `gcr.toml` upward from
    // the current directory (cargo-style `gcr build` / `gcr run`).
    let arg_path = args.get(2).filter(|a| !a.starts_with('-'));
    let manifest: Option<gcrust::manifest::Manifest>;
    // `project_mode` = the build is driven BY the manifest (entry + output name +
    // banner come from it). A bare-file build still *discovers* a manifest for its
    // `[link]` config, but keeps the file as the entry and the file stem as the
    // output — so `gcr build other.gcr` in a project links the same libraries.
    let project_mode: bool;
    let path: String = match arg_path {
        Some(arg_path) => {
        let p = std::path::Path::new(arg_path);
        if p.is_dir() || p.file_name().map(|n| n == "gcr.toml").unwrap_or(false) {
            let dir = if p.is_dir() { p } else { p.parent().unwrap_or(std::path::Path::new(".")) };
            match gcrust::manifest::Manifest::load(dir) {
                Ok(m) => {
                    let entry = m.entry_path().to_string_lossy().into_owned();
                    manifest = Some(m);
                    project_mode = true;
                    entry
                }
                Err(e) => {
                    eprintln!("gcr: {}", e.0);
                    return ExitCode::FAILURE;
                }
            }
        } else {
            // Bare source file — discover a project manifest (link config only).
            manifest = gcrust::manifest::Manifest::discover(p);
            project_mode = false;
            arg_path.clone()
        }
        }
        None => {
            // No path argument — discover a project `gcr.toml` from the cwd.
            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            match gcrust::manifest::Manifest::discover(&cwd) {
                Some(m) => {
                    let entry = m.entry_path().to_string_lossy().into_owned();
                    manifest = Some(m);
                    project_mode = true;
                    entry
                }
                None => {
                    eprintln!(
                        "gcr: no file given and no `gcr.toml` found in {} — \
                         pass a `.gcr` file or run inside a project (try `gcr new <name>`)",
                        cwd.display()
                    );
                    return ExitCode::FAILURE;
                }
            }
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

            // Project mode (a `gcr.toml` was found): build a native executable —
            // linking the manifest's `[link]` libraries — and run it, forwarding
            // its exit code. This is the only way native FFI deps (raylib, …)
            // resolve; the JIT can't link them. Mirrors `cargo run`.
            if let Some(m) = &manifest {
                let mut link_args = parse_link_args(&args);
                link_args.extend(m.link_args());
                let out = default_output(path, Some(m), project_mode);
                if let Err(e) = build_executable_level(
                    &prog,
                    &out,
                    &link_args,
                    gcrust::codegen::DebugLevel::LineTables,
                ) {
                    eprintln!("gcr: build error: {}", e.0);
                    return ExitCode::FAILURE;
                }
                match std::process::Command::new(&out).status() {
                    Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
                    Err(e) => {
                        eprintln!("gcr: failed to run {}: {}", out.display(), e);
                        ExitCode::FAILURE
                    }
                }
            } else {
                // Bare-file run: JIT execute and print the result.
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
        }
        "build" => {
            // Linker args: the project manifest's `[link]` section (libs / paths
            // / frameworks), plus any `--link-arg <arg>` (repeatable) from the CLI.
            // A project build needs no CLI link flags at all.
            let mut link_args = parse_link_args(&args);
            if let Some(m) = &manifest {
                if project_mode {
                    println!("gcr: building `{}` v{}", m.name, m.version);
                }
                link_args.extend(m.link_args());
            }
            // Output path: `gcr build foo.gcr -o foo`. In a project it defaults to
            // `target/<name>`; for a bare file, the source stem.
            let out = match parse_output_flag(&args) {
                Ok(o) => o,
                Err(msg) => {
                    eprintln!("gcr: {}", msg);
                    return ExitCode::FAILURE;
                }
            };
            let out = out.unwrap_or_else(|| default_output(path, manifest.as_ref(), project_mode));

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
            // `--debug` → full DWARF (debugger P3): unoptimized + local-variable
            // DIEs, so `lldb`'s `frame variable` shows source names/values.
            // Default stays line-tables-only (P2: stepping/breakpoints under O2).
            let level = if args.iter().any(|a| a == "--debug") {
                gcrust::codegen::DebugLevel::Full
            } else {
                gcrust::codegen::DebugLevel::LineTables
            };
            match build_executable_level(&prog, &out, &link_args, level) {
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
/// `gcr bench <prog.gcr> [more.gcr …] [--runs N] [--json <out>]` — the GENERAL
/// benchmark runner. Builds + runs each gc-rust program and emits one metrics
/// document in the `gcr-bench/1` schema: every program is a `group` carrying a
/// `gc-rust` series whose `values` hold runtime (mean/stddev/min/max over N
/// runs), compile time, binary size, and the runtime-only numbers no native
/// binary exposes — GC cycles + pause distribution, allocation churn, peak heap.
/// This is "run any program, get any benchmark": the composable bench views
/// (gcr_bench_*) render whatever metrics this produces.
fn run_bench(args: &[String]) -> ExitCode {
    use serde_json::{json, Value};
    // Collect program paths, skipping flags AND their values (`--runs N`,
    // `--json <path>`) so a flag's argument isn't mistaken for a program.
    let mut progs: Vec<String> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        let a = &args[i];
        if a == "--runs" || a == "--json" || a == "--vary" || a == "--nursery" {
            i += 2;
            continue;
        }
        if a.starts_with('-') {
            i += 1;
            continue;
        }
        progs.push(a.clone());
        i += 1;
    }
    if progs.is_empty() {
        eprintln!("usage: gcr bench <prog.gcr> [more.gcr …] [--runs N] [--json <out>] \
                   [--vary ENV=v1,v2,…] [--nursery m1,m2,…]");
        return ExitCode::FAILURE;
    }
    // `--vary ENV=v1,v2,…` runs each program once PER value (a series per value),
    // with that env var set — so general mode gets grouped bars over a tuning
    // axis. `--nursery m1,m2,…` is sugar for `--vary GCR_NURSERY_MB=…` with nicer
    // labels (the gc-rust GC nursery size, a knob no native binary exposes).
    let vary: Option<(String, Vec<String>)> = parse_vary(args);
    let runs: usize = args
        .iter()
        .position(|a| a == "--runs")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(8)
        .max(1);
    let json_out = args.iter().position(|a| a == "--json").and_then(|i| args.get(i + 1)).cloned();

    let self_exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gcr bench: cannot find own executable: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tmp = std::env::temp_dir().join(format!("gcr_bench_run_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    if let Err(e) = std::fs::create_dir_all(&tmp) {
        eprintln!("gcr bench: cannot create work dir: {e}");
        return ExitCode::FAILURE;
    }

    // The metric catalog — what each value means, so the views are schema-driven
    // (units + lower_better drive formatting, winners, and colors).
    let metrics = json!([
        {"key": "wall_ms", "label": "runtime", "unit": "ms", "lower_better": true, "dist": true},
        {"key": "compile_ms", "label": "compile", "unit": "ms", "lower_better": true},
        {"key": "binary_kb", "label": "binary size", "unit": "KB", "lower_better": true},
        {"key": "gc_minor", "label": "minor GCs", "unit": "", "lower_better": true},
        {"key": "gc_major", "label": "major GCs", "unit": "", "lower_better": true},
        {"key": "gc_pause_max_ms", "label": "max GC pause", "unit": "ms", "lower_better": true},
        {"key": "gc_pause_total_ms", "label": "total GC pause", "unit": "ms", "lower_better": true},
        {"key": "alloc_objects", "label": "allocations", "unit": "", "lower_better": true},
        {"key": "alloc_bytes", "label": "alloc churn", "unit": "B", "lower_better": true},
        {"key": "peak_heap_bytes", "label": "peak heap", "unit": "B", "lower_better": true}
    ]);

    let mut groups: Vec<Value> = Vec::new();
    for prog in &progs {
        let path = match resolve_entry(prog) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("gcr bench: {e}");
                continue;
            }
        };
        let name = std::path::Path::new(&path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| prog.clone());
        let bin = tmp.join(&name);

        // Compile (timed). Shells our own `gcr build -o`, so the link config and
        // staticlib refresh are handled exactly as a normal build.
        let t0 = std::time::Instant::now();
        let build = std::process::Command::new(&self_exe)
            .args(["build", &path, "-o", &bin.to_string_lossy()])
            .output();
        let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let built_ok = matches!(&build, Ok(o) if o.status.success());
        if !built_ok {
            let err = build.map(|o| String::from_utf8_lossy(&o.stderr).into_owned()).unwrap_or_default();
            eprintln!("gcr bench: {name}: build failed: {}", err.trim());
            groups.push(json!({"name": name, "error": "build failed",
                "series": [{"label": "gc-rust", "values": {}}]}));
            continue;
        }
        let binary_kb = std::fs::metadata(&bin).map(|m| m.len() as f64 / 1024.0).unwrap_or(0.0);

        // One series per variant value (or a single "gc-rust" series with no
        // `--vary`). Compile/size are program-level, so they're measured once and
        // repeated across variants.
        let variant_vals: Vec<Option<String>> = match &vary {
            Some((_, vals)) => vals.iter().map(|v| Some(v.clone())).collect(),
            None => vec![None],
        };
        let env_name = vary.as_ref().map(|(n, _)| n.clone());
        let mut series: Vec<Value> = Vec::new();
        for variant in &variant_vals {
            // Wall-clock runs (output suppressed so I/O doesn't skew timing).
            let mut times_ms: Vec<f64> = Vec::with_capacity(runs);
            for _ in 0..runs {
                let mut cmd = std::process::Command::new(&bin);
                if let (Some(n), Some(v)) = (&env_name, variant) {
                    cmd.env(n, v);
                }
                let t = std::time::Instant::now();
                if cmd.output().is_err() {
                    break;
                }
                times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let (mean, stddev, mn, mx) = wall_stats(&times_ms);

            // One extra run to capture the runtime-only metrics (GC/alloc/heap).
            let mfile = tmp.join(format!("{name}.metrics.json"));
            let mut mcmd = std::process::Command::new(&bin);
            mcmd.env("GCR_METRICS_FILE", &mfile);
            if let (Some(n), Some(v)) = (&env_name, variant) {
                mcmd.env(n, v);
            }
            let _ = mcmd.output();
            let m: Value = std::fs::read_to_string(&mfile)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_else(|| json!({}));
            let mv = |k: &str| m.get(k).cloned().unwrap_or(json!(0));

            let values = json!({
                "wall_ms": {"mean": mean, "stddev": stddev, "min": mn, "max": mx},
                "compile_ms": compile_ms,
                "binary_kb": binary_kb,
                "gc_minor": mv("gc_minor"),
                "gc_major": mv("gc_major"),
                "gc_pause_max_ms": mv("gc_pause_max_ms"),
                "gc_pause_total_ms": mv("gc_pause_total_ms"),
                "alloc_objects": mv("alloc_objects"),
                "alloc_bytes": mv("alloc_bytes"),
                "peak_heap_bytes": mv("peak_heap_bytes")
            });
            let label = variant_label(&env_name, variant);
            series.push(json!({"label": label, "values": values}));
        }
        groups.push(json!({"name": name, "series": series}));
    }
    let _ = std::fs::remove_dir_all(&tmp);

    let mut meta = json!({"kind": "programs", "runs": runs});
    if let Some((n, vals)) = &vary {
        meta["vary"] = json!({"env": n, "values": vals});
    }
    let doc = json!({
        "schema": "gcr-bench/1",
        "meta": meta,
        "metrics": metrics,
        "groups": groups
    });
    let text = serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".into());
    match json_out {
        Some(o) => {
            if let Err(e) = std::fs::write(&o, &text) {
                eprintln!("gcr bench: cannot write {o}: {e}");
                return ExitCode::FAILURE;
            }
            println!("{o}");
        }
        None => println!("{text}"),
    }
    ExitCode::SUCCESS
}

/// Parse the bench variant axis: `--vary ENV=v1,v2,…` → (ENV, [v1,v2,…]), or
/// `--nursery m1,m2,…` as sugar for `--vary GCR_NURSERY_MB=…`. `None` if neither
/// is given (a single un-varied run).
fn parse_vary(args: &[String]) -> Option<(String, Vec<String>)> {
    if let Some(spec) = args.iter().position(|a| a == "--vary").and_then(|i| args.get(i + 1)) {
        if let Some((env, csv)) = spec.split_once('=') {
            let vals: Vec<String> = csv.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            if !vals.is_empty() {
                return Some((env.to_string(), vals));
            }
        }
    }
    if let Some(csv) = args.iter().position(|a| a == "--nursery").and_then(|i| args.get(i + 1)) {
        let vals: Vec<String> = csv.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
        if !vals.is_empty() {
            return Some(("GCR_NURSERY_MB".to_string(), vals));
        }
    }
    None
}

/// A series label for a variant value (or "gc-rust" for an un-varied run).
fn variant_label(env: &Option<String>, value: &Option<String>) -> String {
    match (env, value) {
        (Some(n), Some(v)) if n == "GCR_NURSERY_MB" => format!("nursery {v}MB"),
        (Some(n), Some(v)) => format!("{n}={v}"),
        _ => "gc-rust".to_string(),
    }
}

/// (mean, sample-stddev, min, max) of a slice of millisecond timings.
fn wall_stats(xs: &[f64]) -> (f64, f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = if xs.len() > 1 {
        xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let mn = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, var.sqrt(), mn, mx)
}

/// `gcr heap <file.gcr> [--out <path>] [--gc-stress]` — run a program and emit
/// its program-end heap snapshot as JSON (the same schema as `GCR_HEAP_DUMP=json`
/// and `gcr heap-diff`). With `--out`, the JSON is written there (clean, never
/// interleaved with program output) and the path is echoed on stdout; this is the
/// contract the heap-explorer widget (`gcr_heap.ft`) drives. Without `--out`, the
/// JSON is printed to stdout after the program's own output.
fn run_heap(args: &[String]) -> ExitCode {
    let arg_path = match args.iter().skip(2).find(|a| !a.starts_with('-')) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: gcr heap <file.gcr> [--out <path>] [--gc-stress]");
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
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned();
    // `--series <path>` bundles EVERY in-program `heap_snapshot()` into one JSON
    // `{ "snapshots": [ <snap>, … ] }` (in call order) — the data for the
    // explorer's growth/leak tab + snapshot scrubber. Falls back to a 1-element
    // series from the program-end dump for programs that never snapshot.
    let series = args.iter().position(|a| a == "--series").and_then(|i| args.get(i + 1)).cloned();

    // We capture TWO kinds of snapshot and prefer the richer one:
    //  1. In-program `heap_snapshot()` calls land (numbered) in a temp dir via
    //     GCR_HEAP_SNAPSHOT_DIR — these are MID-EXECUTION, so they carry live
    //     stack roots + real retainer chains (the whole point of a heap explorer).
    //  2. The program-end dump (GCR_HEAP_DUMP_FILE) — a fallback for programs that
    //     never call `heap_snapshot()`; at end, roots = globals only.
    // The widget gets the last in-program snapshot if any, else the end dump.
    let unique = std::process::id();
    let snap_dir = std::env::temp_dir().join(format!("gcr_heap_snaps_{unique}"));
    let _ = std::fs::remove_dir_all(&snap_dir);
    if let Err(e) = std::fs::create_dir_all(&snap_dir) {
        eprintln!("gcr heap: cannot create snapshot dir {}: {e}", snap_dir.display());
        return ExitCode::FAILURE;
    }
    let end_dump = std::env::temp_dir().join(format!("gcr_heap_end_{unique}.json"));
    // The dumps fire inside the JIT run, keyed on these env vars (see codegen.rs /
    // runtime.rs). Safe: single-threaded here — set before any thread is spawned.
    unsafe {
        std::env::set_var("GCR_HEAP_SNAPSHOT_DIR", &snap_dir);
        std::env::set_var("GCR_HEAP_DUMP", "json");
        std::env::set_var("GCR_HEAP_DUMP_FILE", &end_dump);
    }
    let run = compile_and_run(&src, stress);
    if let Err((stage, msg)) = run {
        eprintln!("gcr heap: {stage}: {msg}");
        let _ = std::fs::remove_dir_all(&snap_dir);
        return ExitCode::FAILURE;
    }
    // `--series` mode: bundle every in-program snapshot (in call order) into one
    // `{ "snapshots": [ … ] }` file, falling back to the end dump if none.
    if let Some(series_path) = series {
        let mut files = all_snapshots(&snap_dir);
        if files.is_empty() {
            files.push(end_dump.clone());
        }
        let mut parts: Vec<String> = Vec::with_capacity(files.len());
        for f in &files {
            match std::fs::read_to_string(f) {
                Ok(j) => parts.push(j.trim().to_string()),
                Err(e) => {
                    eprintln!("gcr heap: cannot read snapshot {}: {e}", f.display());
                    let _ = std::fs::remove_dir_all(&snap_dir);
                    return ExitCode::FAILURE;
                }
            }
        }
        // Each part is itself a complete JSON object, so joining with commas
        // inside an array yields valid JSON without re-serialization.
        let bundle = format!("{{\"snapshots\":[{}]}}", parts.join(","));
        let _ = std::fs::remove_dir_all(&snap_dir);
        let _ = std::fs::remove_file(&end_dump);
        if let Err(e) = std::fs::write(&series_path, &bundle) {
            eprintln!("gcr heap: cannot write {series_path}: {e}");
            return ExitCode::FAILURE;
        }
        println!("{series_path}");
        return ExitCode::SUCCESS;
    }

    // Single-snapshot mode: highest-numbered in-program snapshot, else the
    // program-end dump.
    let chosen = latest_snapshot(&snap_dir).unwrap_or(end_dump.clone());
    let json = match std::fs::read_to_string(&chosen) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("gcr heap: no snapshot produced ({}: {e})", chosen.display());
            let _ = std::fs::remove_dir_all(&snap_dir);
            return ExitCode::FAILURE;
        }
    };
    let _ = std::fs::remove_dir_all(&snap_dir);
    let _ = std::fs::remove_file(&end_dump);
    match out {
        // `--out` given (the widget's path): write the JSON there, echo the path.
        Some(o) => {
            if let Err(e) = std::fs::write(&o, &json) {
                eprintln!("gcr heap: cannot write {o}: {e}");
                return ExitCode::FAILURE;
            }
            println!("{o}");
        }
        // No `--out`: surface the JSON on stdout for scripting.
        None => print!("{json}"),
    }
    ExitCode::SUCCESS
}

/// Every `snapshot-NNNN.json` in `dir`, sorted by sequence (call order). Names
/// are zero-padded, so lexical order == numeric order.
fn all_snapshots(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut snaps: Vec<(String, std::path::PathBuf)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) {
                if name.starts_with("snapshot-") && name.ends_with(".json") {
                    snaps.push((name, path));
                }
            }
        }
    }
    snaps.sort_by(|a, b| a.0.cmp(&b.0));
    snaps.into_iter().map(|(_, p)| p).collect()
}

/// The highest-numbered `snapshot-NNNN.json` in `dir` (the last in-program
/// `heap_snapshot()` taken), or `None` if the program took no snapshots.
fn latest_snapshot(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    all_snapshots(dir).into_iter().next_back()
}

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

/// The default output path for `build`/`run` when `-o` is not given. In project
/// mode it is `<manifest-dir>/target/<name>` (the directory is created); for a
/// bare-file build (even one inside a project) it is the source stem.
fn default_output(
    path: &str,
    manifest: Option<&gcrust::manifest::Manifest>,
    project_mode: bool,
) -> std::path::PathBuf {
    if project_mode {
        if let Some(m) = manifest {
            let target = m.dir.join("target");
            let _ = std::fs::create_dir_all(&target);
            return target.join(&m.name);
        }
    }
    std::path::Path::new(path)
        .file_stem()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("a.out"))
}

/// `gcr new <name>` — scaffold a project: `<name>/gcr.toml` + `<name>/src/main.gcr`.
fn run_new(args: &[String]) -> ExitCode {
    let name = match args.get(2) {
        Some(n) if !n.starts_with('-') => n.clone(),
        _ => {
            eprintln!("usage: gcr new <name>");
            return ExitCode::FAILURE;
        }
    };
    let dir = std::path::PathBuf::from(&name);
    if dir.exists() {
        eprintln!("gcr: `{}` already exists", name);
        return ExitCode::FAILURE;
    }
    let src_dir = dir.join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        eprintln!("gcr: cannot create {}: {}", src_dir.display(), e);
        return ExitCode::FAILURE;
    }
    let toml = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nentry = \"src/main.gcr\"\n\n\
         # Link native libraries (uncomment + edit as needed):\n\
         # [link]\n\
         # libs = [\"raylib\"]\n\
         # lib-paths = [\"/opt/homebrew/lib\"]\n\
         # frameworks = [\"Cocoa\", \"OpenGL\"]\n"
    );
    let main_src = format!(
        "fn main() -> i64 {{\n  print_line(\"hello from {name}\");\n  0\n}}\n"
    );
    if let Err(e) = std::fs::write(dir.join("gcr.toml"), toml) {
        eprintln!("gcr: {}", e);
        return ExitCode::FAILURE;
    }
    if let Err(e) = std::fs::write(src_dir.join("main.gcr"), main_src) {
        eprintln!("gcr: {}", e);
        return ExitCode::FAILURE;
    }
    println!("gcr: created project `{name}`");
    println!("     cd {name} && gcr run");
    ExitCode::SUCCESS
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
