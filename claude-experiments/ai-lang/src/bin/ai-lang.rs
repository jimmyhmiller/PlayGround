//! `ai-lang` — the generic runner.
//!
//! A program is a hash in a content-addressed codebase. Names are
//! mutable surface aliases. The codebase lives on disk as a directory:
//!
//!   defs/<hex>.def     canonical bytes per def
//!   types/<hex>.type   cached TypeScheme per def
//!   names.txt          name → hash
//!
//! ## Commands
//!
//!   ai-lang add <file.ail>                   ingest source into the codebase
//!   ai-lang ls                                list named defs
//!   ai-lang run <name> [--nodes=N] [-- ...]   JIT and invoke `<name>() -> Int`
//!   ai-lang serve                             become a worker for `at()` calls
//!
//! ## Codebase location
//!
//! Defaults to `./.ai-lang`. Override with `--codebase <path>` or env
//! `AI_LANG_CODEBASE=<path>`.

use ai_lang::Hash;
use ai_lang::ast::Def;
use ai_lang::codebase::Codebase;
use ai_lang::codegen::{
    CompiledModule, IncrementalJit, Jit, def_symbol, init_native_target,
};
use ai_lang::io_externs::{set_user_args, set_worker_nodes};
use ai_lang::knowledge::KnowledgeBase;
use ai_lang::net::{
    bind, build_at_runtime_binding, clear_at_conn_cache, clear_current_at_binding,
    clear_current_knowledge_base, clear_current_runtime, install_current_at_binding,
    install_current_knowledge_base, install_current_runtime, serve_with_install,
};
use ai_lang::parser::parse_module;
use ai_lang::resolve::{AtBinding, ExternSig, ResolvedDef, ResolvedModule, resolve_module};
use ai_lang::runtime::{Runtime, Thread};
use ai_lang::stdlib::SOURCE as STDLIB;
use ai_lang::typecheck::{TypeCache, typecheck_module};
use inkwell::context::Context;
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

// =============================================================================
// CLI
// =============================================================================

fn usage(code: i32) -> ! {
    let s = "\
ai-lang — generic runner for content-addressed ai-lang programs

usage:
  ai-lang add <file.ail>                   ingest source into the codebase
  ai-lang ls                                list named defs in the codebase
  ai-lang run <name> [--nodes=N] [-- ...]  invoke a `<name>() -> Int` and print it
  ai-lang serve                             become a worker for at() (empty runtime)

options (run/add/ls):
  --codebase <path>      override codebase root (default ./.ai-lang)
                         also honoured via AI_LANG_CODEBASE env

run-only:
  --nodes=<N>            spawn N `ai-lang serve` subprocesses, expose them
                         to the program via `node_count()` / `get_node_port(i)`
";
    let _ = io::stderr().write_all(s.as_bytes());
    std::process::exit(code);
}

fn cb_root(opt: Option<&String>) -> PathBuf {
    if let Some(p) = opt {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("AI_LANG_CODEBASE") {
        return PathBuf::from(p);
    }
    PathBuf::from(".ai-lang")
}

/// Parse `--codebase <path>` and `--nodes=<N>` out of `args` (in place).
/// Returns (codebase_override, nodes_override, remaining_after_dashdash).
fn split_flags(args: &mut Vec<String>) -> (Option<String>, Option<usize>, Vec<String>) {
    let mut codebase: Option<String> = None;
    let mut nodes: Option<usize> = None;
    let mut post_dashdash: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--" {
            post_dashdash = args.split_off(i + 1);
            args.pop(); // drop the "--" itself
            break;
        } else if a == "--codebase" {
            if i + 1 >= args.len() {
                eprintln!("--codebase: missing value");
                std::process::exit(2);
            }
            codebase = Some(args[i + 1].clone());
            args.drain(i..i + 2);
        } else if let Some(rest) = a.strip_prefix("--codebase=") {
            codebase = Some(rest.to_owned());
            args.remove(i);
        } else if let Some(rest) = a.strip_prefix("--nodes=") {
            nodes = Some(rest.parse().unwrap_or_else(|_| {
                eprintln!("--nodes=<N>: bad integer");
                std::process::exit(2);
            }));
            args.remove(i);
        } else if a == "--nodes" {
            if i + 1 >= args.len() {
                eprintln!("--nodes: missing value");
                std::process::exit(2);
            }
            nodes = Some(args[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("--nodes <N>: bad integer");
                std::process::exit(2);
            }));
            args.drain(i..i + 2);
        } else {
            i += 1;
        }
    }
    (codebase, nodes, post_dashdash)
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage(2);
    }
    // args[0] is the binary path; args[1] is the subcommand.
    let subcommand = args.remove(1);
    // Drop binary path.
    args.remove(0);
    let (cb_opt, nodes_opt, post_args) = split_flags(&mut args);
    let cb_path = cb_root(cb_opt.as_ref());

    match subcommand.as_str() {
        "add" => {
            if args.len() != 1 {
                usage(2);
            }
            cmd_add(&args[0], &cb_path);
        }
        "ls" => {
            cmd_ls(&cb_path);
        }
        "run" => {
            if args.is_empty() {
                usage(2);
            }
            cmd_run(&args[0], nodes_opt.unwrap_or(0), post_args, &cb_path);
        }
        "serve" => {
            cmd_serve();
        }
        other => {
            eprintln!("unknown subcommand: {}", other);
            usage(2);
        }
    }
}

// =============================================================================
// `add`
// =============================================================================

fn cmd_add(file: &str, cb_path: &PathBuf) {
    init_native_target().expect("init native target");
    let user_src = std::fs::read_to_string(file)
        .unwrap_or_else(|e| {
            eprintln!("read {}: {}", file, e);
            std::process::exit(1);
        });
    let full_src = format!("{}\n{}", STDLIB, user_src);
    let m = parse_module(&full_src).unwrap_or_else(|e| {
        eprintln!("parse: {:?}", e);
        std::process::exit(1);
    });
    let r = resolve_module(&m).unwrap_or_else(|e| {
        eprintln!("resolve: {:?}", e);
        std::process::exit(1);
    });
    let mut tc = TypeCache::new();
    typecheck_module(&r, &mut tc).unwrap_or_else(|e| {
        eprintln!("typecheck: {:?}", e);
        std::process::exit(1);
    });

    let mut cb = Codebase::open(cb_path).expect("open codebase");
    cb.store_resolved_module(&r).expect("store resolved module");
    let typed = cb.store_typecache(&tc).expect("store typecache");
    eprintln!(
        "added {} defs to {} ({} new typeschemes cached)",
        r.defs.len(),
        cb_path.display(),
        typed,
    );
}

// =============================================================================
// `ls`
// =============================================================================

fn cmd_ls(cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let mut entries: Vec<(&String, &Hash)> = cb.names().iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (name, hash) in entries {
        println!("{:<32}  {}", name, &hash.to_hex()[..16]);
    }
    eprintln!("\n  {} names in {}", cb.names().len(), cb_path.display());
}

// =============================================================================
// `run`
// =============================================================================

fn cmd_run(name: &str, num_nodes: usize, user_args: Vec<String>, cb_path: &PathBuf) {
    init_native_target().expect("init native target");

    let cb = Codebase::open(cb_path).expect("open codebase");
    let root_hash = match cb.get_name(name) {
        Some(h) => h,
        None => {
            eprintln!("unknown name: {} (try `ai-lang ls`)", name);
            std::process::exit(1);
        }
    };

    // Reconstruct a ResolvedModule from every named def in the
    // codebase. Compiling everything (not just deps of the root) is
    // simpler than threading a dep-walker through here — codegen and
    // KB::build handle the cycle / order details from a flat Vec.
    let mut defs: Vec<ResolvedDef> = Vec::with_capacity(cb.names().len());
    let mut entries: Vec<(&String, &Hash)> = cb.names().iter().collect();
    entries.sort_by(|a, b| a.1.to_hex().cmp(&b.1.to_hex()));
    for (name, hash) in entries {
        let def = cb.load_def(hash).expect("load_def");
        defs.push(ResolvedDef {
            name: name.clone(),
            hash: *hash,
            def,
        });
    }

    let externs = stdlib_externs();
    let at_binding = at_binding_from_codebase(&cb);
    let rm = ResolvedModule {
        defs,
        at_binding,
        externs,
    };

    // Spawn workers (if requested) and publish their ports to the
    // global table that `get_node_port(i)` reads from.
    let mut servers: Vec<ServerChild> = Vec::new();
    if num_nodes > 0 {
        eprintln!("spawning {} worker processes...", num_nodes);
        let t0 = std::time::Instant::now();
        for i in 0..num_nodes {
            servers.push(spawn_worker(i).expect("spawn worker"));
        }
        let ports: Vec<u16> = servers.iter().map(|s| s.port).collect();
        eprintln!(
            "  {} workers ready in {:.1}s (first 6 PIDs: {:?})",
            num_nodes,
            t0.elapsed().as_secs_f64(),
            &servers.iter().map(|s| s.child.id()).take(6).collect::<Vec<_>>(),
        );
        set_worker_nodes(ports);
    }

    // Publish user args.
    set_user_args(user_args.clone());

    // JIT the reconstructed module and call <name>() -> Int.
    let ctx = Context::create();
    let cm = CompiledModule::build(&ctx, &rm).expect("build module");
    let rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let jit = Jit::new(cm, &rt).expect("jit");

    install_current_runtime(&rt);
    // For at() to work end-to-end, the client also needs a knowledge
    // base to ship to servers on NeedCode. Build it from the reconstructed
    // resolved module — this captures every def + every reachable lambda.
    let kb = KnowledgeBase::build(&rm);
    install_current_knowledge_base(&kb);
    if let Some(rb) = rm
        .at_binding
        .as_ref()
        .and_then(|ab| build_at_runtime_binding(&rt, ab))
    {
        install_current_at_binding(&rb);
    }

    let entry = unsafe {
        jit.engine
            .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(
                &root_hash,
            ))
            .unwrap_or_else(|_| {
                eprintln!("`{}` is not a `() -> Int` function", name);
                std::process::exit(1);
            })
    };
    let result = unsafe { entry.call(rt.thread_ptr()) };
    println!("{}", result);

    clear_at_conn_cache();
    clear_current_runtime();
    clear_current_knowledge_base();
    clear_current_at_binding();
    drop(servers); // kills workers
}

// =============================================================================
// `serve`
// =============================================================================

fn cmd_serve() {
    init_native_target().expect("init native target");

    let ctx = Context::create();
    let empty_m = parse_module("").expect("parse empty");
    let empty_r = resolve_module(&empty_m).expect("resolve empty");
    let empty_cm = CompiledModule::build(&ctx, &empty_r).expect("build empty");
    let mut rt = Runtime::new_with_metadata(
        empty_cm.closure_type_infos.clone(),
        empty_cm.shape_registry.clone(),
        empty_cm.shape_meta.clone(),
        empty_cm.shape_by_type_id.clone(),
    );
    let mut jit = IncrementalJit::new(empty_cm, &rt).expect("incremental jit");

    let listener = bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    println!("READY {}", port);
    let _ = io::stdout().flush();

    // Watchdog: parent's stdin pipe closes when parent dies → we exit.
    std::thread::spawn(|| {
        let stdin = io::stdin();
        let mut buf = [0u8; 64];
        loop {
            match stdin.lock().read(&mut buf) {
                Ok(0) | Err(_) => std::process::exit(0),
                Ok(_) => {}
            }
        }
    });

    eprintln!(
        "[ai-lang serve pid={} port={}] empty runtime, awaiting code",
        std::process::id(),
        port
    );

    loop {
        let mut stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(e) => {
                eprintln!("[ai-lang serve] accept: {}", e);
                break;
            }
        };
        let _ = stream.set_nodelay(true);
        loop {
            match unsafe { serve_with_install(&mut rt, &mut jit, &mut stream) } {
                Ok(()) => continue,
                Err(e) => {
                    eprintln!(
                        "[ai-lang serve pid={}] error on connection: {}",
                        std::process::id(),
                        e
                    );
                    break;
                }
            }
        }
    }
}

// =============================================================================
// Codebase helpers
// =============================================================================

/// Re-resolve the stdlib in isolation to recover its `externs` map.
/// User-declared externs are not yet supported by the codebase
/// runner — they'd need their own on-disk table.
fn stdlib_externs() -> HashMap<String, ExternSig> {
    let m = parse_module(STDLIB).expect("parse stdlib");
    let r = resolve_module(&m).expect("resolve stdlib");
    r.externs
}

/// Rebuild the resolver's `AtBinding` from named enum/struct defs in the
/// codebase. Returns `None` if any of `Result`/`Failure`/`Node` is
/// missing (the program doesn't use `at()`).
fn at_binding_from_codebase(cb: &Codebase) -> Option<AtBinding> {
    let result_hash = cb.get_name("Result")?;
    let failure_hash = cb.get_name("Failure")?;
    let node_hash = cb.get_name("Node")?;
    let result_def = cb.load_def(&result_hash).ok()?;
    let failure_def = cb.load_def(&failure_hash).ok()?;
    let result_variants = match result_def {
        Def::Enum { variants, .. } => variants,
        _ => return None,
    };
    let failure_variants = match failure_def {
        Def::Enum { variants, .. } => variants,
        _ => return None,
    };
    let find = |vs: &[(String, _)], n: &str| -> Option<u32> {
        vs.iter().position(|(name, _)| name == n).map(|i| i as u32)
    };
    Some(AtBinding {
        result_hash,
        failure_hash,
        node_hash,
        ok_variant_index: find(&result_variants, "Ok")?,
        err_variant_index: find(&result_variants, "Err")?,
        unreachable_variant_index: find(&failure_variants, "Unreachable")?,
        crashed_variant_index: find(&failure_variants, "Crashed")?,
        code_missing_variant_index: find(&failure_variants, "CodeMissing")?,
        cancelled_variant_index: find(&failure_variants, "Cancelled")?,
    })
}

// =============================================================================
// Worker spawn
// =============================================================================

struct ServerChild {
    child: Child,
    port: u16,
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_worker(_idx: usize) -> io::Result<ServerChild> {
    let exe = std::env::current_exe()?;
    let mut child = Command::new(exe)
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child.stdout.take().expect("stdout piped");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let port: u16 = line
        .trim()
        .strip_prefix("READY ")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("worker bad READY line: {:?}", line),
            )
        })?;
    drop(reader);
    Ok(ServerChild { child, port })
}
