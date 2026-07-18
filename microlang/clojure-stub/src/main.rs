//! `microclj` — the command-line entry point, shaped like the `clojure` CLI:
//!
//!   microclj                      an interactive REPL
//!   microclj file.clj [args…]     run a file (`*command-line-args*` bound)
//!   microclj -e "(+ 1 2)"         evaluate an expression and print it
//!   microclj --jit …              run on the native JIT tier (feature "jit")
//!
//! The load path is `$MICROLANG_PATH` (colon-separated, default `.`), plus the
//! script's own directory when running a file.

use std::io::{BufRead, Write};

use microlang::{LowBitModel, Runtime, TreeWalk};

fn usage() -> ! {
    eprintln!(
        "usage: microclj [--jit] [file.clj args… | -e EXPR]\n\
         \n\
         no arguments        start a REPL (:repl/quit or ctrl-D to exit)\n\
         file.clj args…      run a file; args are *command-line-args*\n\
         -e EXPR             evaluate EXPR and print the result\n\
         --nrepl [PORT]      start an nREPL server (default port 7888)\n\
         --jit               use the native JIT tier (requires the `jit` feature)"
    );
    std::process::exit(2)
}

enum Mode {
    Repl,
    Eval(String),
    File(String, Vec<String>),
    Nrepl(u16),
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut jit = false;
    let mut opt = false;
    // Consume leading tier flags in any order: --jit selects the native tier,
    // --opt wraps the chosen tier in the fixnum-specializing nanopass optimizer.
    loop {
        match args.first().map(String::as_str) {
            Some("--jit") => { jit = true; args.remove(0); }
            Some("--opt") => { opt = true; args.remove(0); }
            _ => break,
        }
    }
    let mode = match args.first().map(String::as_str) {
        None => Mode::Repl,
        Some("--help") | Some("-h") => usage(),
        Some("-e") => match args.get(1) {
            Some(e) => Mode::Eval(e.clone()),
            None => usage(),
        },
        Some("--nrepl") => {
            let port = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(7888);
            Mode::Nrepl(port)
        }
        Some(_) => Mode::File(args[0].clone(), args[1..].to_vec()),
    };

    if jit {
        #[cfg(feature = "jit")]
        {
            let backend = microlang::jit_cranelift::JitCranelift::<LowBitModel>::new();
            if opt {
                return drive(&microlang::Optimized::new(backend), mode);
            }
            return drive(&backend, mode);
        }
        #[cfg(not(feature = "jit"))]
        {
            eprintln!("--jit requires building with `--features jit`");
            std::process::exit(2);
        }
    }
    if opt {
        return drive(&microlang::Optimized::new(TreeWalk), mode);
    }
    drive(&TreeWalk, mode)
}

fn drive(cs: &dyn microlang::CodeSpace<LowBitModel>, mode: Mode) {
    let mut rt = Runtime::<LowBitModel>::new();
    let mut paths = clojure_stub::default_load_paths();
    if let Mode::File(f, _) = &mode {
        if let Some(dir) = std::path::Path::new(f).parent() {
            if !dir.as_os_str().is_empty() {
                paths.push(dir.to_path_buf());
            }
        }
    }
    // ./deps.edn, exactly like the clojure CLI: its :paths and :local/root
    // deps join the load path (nREPL etc. arrive as ordinary dependencies).
    let deps_edn = std::path::Path::new("deps.edn");
    if deps_edn.is_file() {
        let src = std::fs::read_to_string(deps_edn).unwrap_or_else(|e| {
            eprintln!("microclj: cannot read deps.edn: {e}");
            std::process::exit(1)
        });
        match clojure_stub::deps_edn_paths(&mut rt, &src, std::path::Path::new(".")) {
            Ok(mut ps) => paths.append(&mut ps),
            Err(e) => {
                eprintln!("microclj: {e}");
                std::process::exit(1)
            }
        }
    }
    let mut session = clojure_stub::Session::new(&mut rt, cs, paths);

    // MICROLANG_STATS=1: dump the process-wide performance counters at exit
    // (the same numbers `(%stats)` reads — see microlang::stats).
    struct StatsDump;
    impl Drop for StatsDump {
        fn drop(&mut self) {
            use std::sync::atomic::Ordering::Relaxed;
            eprintln!(
                "[stats] native-invokes={} interp-invokes={} dispatch-shim-calls={} jit-compiles={}",
                microlang::stats::NATIVE_INVOKES.load(Relaxed),
                microlang::stats::INTERP_INVOKES.load(Relaxed),
                microlang::stats::DISPATCH_SHIM_CALLS.load(Relaxed),
                microlang::stats::JIT_COMPILES.load(Relaxed),
            );
        }
    }
    let _stats_dump =
        std::env::var_os("MICROLANG_STATS").filter(|v| v != "0").map(|_| StatsDump);

    match mode {
        Mode::Eval(expr) => {
            let v = session.eval(&mut rt, cs, &expr);
            println!("{}", clojure_stub::clj_str(&rt, v));
        }
        Mode::File(f, argv) => {
            let src = std::fs::read_to_string(&f).unwrap_or_else(|e| {
                eprintln!("microclj: cannot read {f}: {e}");
                std::process::exit(1)
            });
            // Bind *command-line-args* (a seq of strings, nil when empty) INTO
            // clojure.core, so it resolves from whatever ns the script enters.
            let bind = format!(
                "(in-ns 'clojure.core) \
                 (def ^:dynamic *command-line-args* (seq (list {}))) \
                 (in-ns 'user)",
                argv.iter().map(|a| clj_string_literal(a)).collect::<Vec<_>>().join(" ")
            );
            session.eval(&mut rt, cs, &bind);
            session.eval(&mut rt, cs, &src);
        }
        Mode::Repl => repl(&mut rt, cs, &mut session),
        Mode::Nrepl(port) => {
            // The nREPL server is a LIBRARY, not baked into the binary: it (and
            // the real nrepl bencode it uses) must be on the load path — via
            // deps.edn, like any dependency.
            let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                session.eval(&mut rt, cs, "(require 'microclj.nrepl-server)");
            }));
            if loaded.is_err() {
                eprintln!(
                    "microclj: cannot load microclj.nrepl-server.\n\
                     Add the nREPL libraries to your deps.edn, e.g.\n\
                     {{:deps {{nrepl/nrepl {{:mvn/version \"1.3.1\"}}\n\
                             microclj/nrepl-server {{:local/root \"…/clojure-stub/libs\"}}}}}}"
                );
                std::process::exit(1);
            }
            session.eval(&mut rt, cs, &format!("(microclj.nrepl-server/start-server! {port})"));
        }
    }
}

/// A Clojure string literal for `s` (escaped enough for arbitrary argv).
fn clj_string_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn repl(
    rt: &mut Runtime<LowBitModel>,
    cs: &dyn microlang::CodeSpace<LowBitModel>,
    session: &mut clojure_stub::Session,
) {
    // Errors reach us as panics (uncaught throws); print them as one clean
    // line, not a Rust backtrace.
    std::panic::set_hook(Box::new(|_| {}));
    let stdin = std::io::stdin();
    let mut input = String::new();
    loop {
        if input.is_empty() {
            print!("{}=> ", session.current_ns());
        } else {
            print!("{}.. ", " ".repeat(session.current_ns().len()));
        }
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!();
                return; // EOF
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("microclj: {e}");
                return;
            }
        }
        if input.is_empty() && line.trim() == ":repl/quit" {
            return;
        }
        input.push_str(&line);
        if !forms_complete(&input) {
            continue; // keep reading until the parens balance
        }
        let src = std::mem::take(&mut input);
        if src.trim().is_empty() {
            continue;
        }
        // An uncaught throw (or reader error) panics; catch it so the REPL — and
        // the session state — survive. Definitions completed before the error keep.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let v = session.eval(rt, cs, &src);
            clojure_stub::clj_str(rt, v)
        }));
        match result {
            Ok(s) => println!("{s}"),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "error".into());
                eprintln!("; error: {}", msg.lines().next().unwrap_or(""));
            }
        }
    }
}

/// Are all delimiters in `src` balanced (outside strings/chars/comments)?
/// Cheap structural check so the REPL knows when a multi-line form is done.
fn forms_complete(src: &str) -> bool {
    let mut depth: i64 = 0;
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ';' => {
                for c2 in chars.by_ref() {
                    if c2 == '\n' {
                        break;
                    }
                }
            }
            '"' => {
                while let Some(c2) = chars.next() {
                    match c2 {
                        '\\' => {
                            chars.next();
                        }
                        '"' => break,
                        _ => {}
                    }
                }
            }
            '\\' => {
                chars.next(); // a character literal: skip the escaped head char
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
    }
    depth <= 0
}
