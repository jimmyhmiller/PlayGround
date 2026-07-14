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
    if args.first().map(String::as_str) == Some("--jit") {
        jit = true;
        args.remove(0);
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
            return drive(&backend, mode);
        }
        #[cfg(not(feature = "jit"))]
        {
            eprintln!("--jit requires building with `--features jit`");
            std::process::exit(2);
        }
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
    let mut session = clojure_stub::Session::new(&mut rt, cs, paths);

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
            // the REAL nrepl bencode + the server over it, then serve forever
            for src in clojure_stub::NREPL_SOURCES {
                session.eval(&mut rt, cs, src);
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
