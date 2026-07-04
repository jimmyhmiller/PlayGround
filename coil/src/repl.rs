//! `coil repl` — an interactive session over the normal AOT pipeline, wire-
//! compatible with Emacs' inferior-lisp mode (see `emacs/coil-mode.el`).
//!
//! There is no interpreter and no JIT — nothing that can drift from the compiled
//! language. The REPL accumulates top-level definitions as SOURCE, and every
//! expression becomes a tiny program compiled by the same pipeline `coil run`
//! uses. Definitions persist and REDEFINE by name (send a `defn` again to
//! replace it — the Emacs `C-M-x` workflow). An expression's value is printed
//! by GENERATED code chosen from its inferred type (`repl_infer_tail` runs the
//! real checker over a probe function), so what prints is exactly what the type
//! system says.
//!
//! Two evaluation modes:
//!
//!   - LIVE (default; evcxr-style): each eval is compiled to a dylib and
//!     `dlopen`ed into THIS process — hot code loading, not interpretation. The
//!     process is long-lived, so malloc'd memory persists across inputs, and
//!     `(def name EXPR)` binds a value for later inputs: the generated eval
//!     stores it in a heap cell and publishes the cell's pointer into a slot
//!     table; every later eval's generated prologue reads the slots back, typed
//!     from the driver's records. A binding is an immutable snapshot (rebind
//!     with another `def`); persistent MUTABLE state is an explicit pointer —
//!     `(def c (alloc-heap T))` then `(store! c …)` — which is Coil's normal
//!     mutation-is-visible discipline. The trade: a crashing eval takes the
//!     session down (it IS this process).
//!
//!   - `--isolate`: each expression runs in a fresh process (the generated
//!     program gets a `main` and runs like `coil run`). Nothing persists
//!     between inputs; nothing can hurt the session.
//!
//! The value display generates one `show` helper per concrete type (memoized by
//! type, so recursive sums terminate): scalars via the `fmt` specs, strings
//! quoted, slices/arrays as `[e0 e1 …]`, structs as `(Name :field v …)`, sums as
//! `(Variant v…)` via `match`. A type with no honest rendering (SIMD vectors,
//! function pointers, >64-bit ints, `:bits` structs) prints as `#<its-type>` —
//! visible, never silently wrong.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use crate::ast::{Layout, Program, Type};
use crate::check::ty_str;
use crate::reader::{self, Sexp, SexpKind};

/// Reserved names spliced into generated programs. `coil-repl--` is the REPL's
/// namespace prefix; defining these yourself in a session is unsupported.
const PROBE: &str = "coil-repl--probe";

/// The stdlib the REPL brings into every session so the everyday API — slices
/// and strings (`slice-new`/`slice-len`/`str-len`/…) and formatting (`printf`/
/// `print-u`/…) — works BARE, with no manual imports. Each is imported BOTH
/// aliased (`coil-repl-*`, which the generated value-printing helpers use — so
/// they stay correct even if you redefine `slice-len` in your session) AND with
/// `:use *` (which exposes the bare names to you). A same-named session `defn`
/// cleanly wins over the `:use *` name — no shadowing error. Keep the aliases in
/// sync with the `coil-repl-*` references in the show-helper generator.
const REPL_IMPORTS: &str = "\
(import \"fmt.coil\"   :as coil-repl-fmt   :use *)\n\
(import \"io.coil\"    :as coil-repl-io    :use *)\n\
(import \"slice.coil\" :as coil-repl-slice :use *)\n\
(import \"str.coil\"   :as coil-repl-str   :use *)\n";

/// Options for a REPL session (mirrors the `coil run` link surface — an extern
/// against a C library needs its `-l` at every expression's link step).
pub struct Opts {
    pub link_flags: Vec<String>,
    /// `--isolate`: run each expression in a fresh PROCESS instead of loading it
    /// into the live session — no persistent bindings, but a crashing
    /// expression can't take the session down.
    pub isolate: bool,
}

/// One accumulated top-level definition: its redefinition key (`None` = only
/// deduplicated by exact text) and its source text.
struct Def {
    key: Option<String>,
    src: String,
}

// ---------------------------------------------------------------------------
// Live mode: the host process state. Each eval compiles to a DYLIB that is
// dlopen'd into this process and never unloaded (so anything a stored value
// points at — a static string, a function — stays valid for the session).
// Bindings created by `(def name EXPR)` live in `env`: slot i holds a pointer
// to a heap cell the generated Coil code `alloc-heap`ed and wrote the value
// into. The generated prologue of every later eval reads the slots back,
// TYPED (the driver records each binding's inferred type).
// ---------------------------------------------------------------------------

extern "C" {
    fn dlopen(path: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, sym: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn dlerror() -> *mut std::ffi::c_char;
}
const RTLD_NOW: i32 = 0x2; // same value on macOS and Linux
#[cfg(target_os = "macos")]
const RTLD_LOCAL: i32 = 0x4; // Linux's RTLD_LOCAL is 0 (and 0x4 means RTLD_NOLOAD there!)
#[cfg(not(target_os = "macos"))]
const RTLD_LOCAL: i32 = 0;

/// One live binding: `(def name EXPR)`'s name, its inferred type, and which
/// `env` slot holds the pointer to its value cell.
struct Binding {
    name: String,
    ty: Type,
    slot: usize,
}

struct Host {
    bindings: Vec<Binding>,
    /// Slot table passed to every eval's entry function. Grows monotonically;
    /// a rebound name gets a NEW slot (the old cell leaks — REPL policy).
    env: Vec<*mut u8>,
    /// dlopen handles, held (never closed) for the session's lifetime.
    libs: Vec<*mut std::ffi::c_void>,
    /// The dylib files backing `libs` — kept on disk while loaded, removed at drop.
    lib_files: Vec<PathBuf>,
}

impl Host {
    fn new() -> Host {
        Host { bindings: Vec::new(), env: Vec::new(), libs: Vec::new(), lib_files: Vec::new() }
    }

    /// dlopen `path` and call its `sym` entry with the slot table (padded to
    /// `n_slots`). The handle is retained for the session.
    fn load_and_run(&mut self, path: &PathBuf, sym: &str, n_slots: usize) -> Result<(), String> {
        let cpath = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| "dylib path contains a NUL byte".to_string())?;
        let csym = std::ffi::CString::new(sym).expect("entry symbol has no NUL");
        while self.env.len() < n_slots {
            self.env.push(std::ptr::null_mut());
        }
        unsafe {
            let handle = dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL);
            if handle.is_null() {
                let msg = std::ffi::CStr::from_ptr(dlerror()).to_string_lossy().into_owned();
                return Err(format!("dlopen failed: {msg}"));
            }
            self.libs.push(handle);
            self.lib_files.push(path.clone());
            let f = dlsym(handle, csym.as_ptr());
            if f.is_null() {
                return Err(format!("entry symbol '{sym}' not found in compiled eval"));
            }
            let entry: extern "C" fn(*mut *mut u8) -> i64 = std::mem::transmute(f);
            entry(self.env.as_mut_ptr());
        }
        Ok(())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // The mappings die with the process; only the temp files need cleanup.
        for f in &self.lib_files {
            let _ = std::fs::remove_file(f);
        }
    }
}

struct Session {
    defs: Vec<Def>,
    /// Serial number for generated temp executables (avoids racing a previous
    /// still-running eval's file).
    evals: u64,
    link_flags: Vec<String>,
    /// `Some` = live mode (the default): evals load into this process and
    /// `(def …)` bindings persist. `None` = `--isolate` fresh-process mode.
    host: Option<Host>,
}

impl Session {
    fn new(link_flags: Vec<String>, live: bool) -> Session {
        Session { defs: Vec::new(), evals: 0, link_flags, host: live.then(Host::new) }
    }

    /// The session program's source, optionally excluding the user's `main`
    /// (expression evaluation generates its own entry point; a colliding user
    /// `main` is left out there — see `handle_expr`).
    fn source(&self, exclude_main: bool) -> String {
        let mut s = String::from("(module repl)\n");
        for d in &self.defs {
            if exclude_main && d.key.as_deref() == Some("defn main") {
                continue;
            }
            s.push_str(&d.src);
            s.push('\n');
        }
        s
    }

    /// The session with `src` added (replacing the same-keyed definition if one
    /// exists, in place — Coil programs are order-sensitive reading material for
    /// humans even if the compiler reorders). Returns the candidate source and
    /// the index the def would land at; `commit` applies it.
    fn candidate(&self, key: &Option<String>, src: &str) -> (String, Option<usize>) {
        let at = key
            .as_ref()
            .and_then(|k| self.defs.iter().position(|d| d.key.as_deref() == Some(k.as_str())));
        let mut s = String::from("(module repl)\n");
        for (i, d) in self.defs.iter().enumerate() {
            if Some(i) == at {
                s.push_str(src);
            } else {
                s.push_str(&d.src);
            }
            s.push('\n');
        }
        if at.is_none() {
            s.push_str(src);
            s.push('\n');
        }
        (s, at)
    }

    fn commit(&mut self, key: Option<String>, src: String, at: Option<usize>) {
        match at {
            Some(i) => self.defs[i] = Def { key, src },
            None => self.defs.push(Def { key, src }),
        }
    }
}

/// Entry point for `coil repl`: stdin/stdout, exit code.
pub fn run_stdio(opts: Opts) -> i32 {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    run(stdin.lock(), stdout.lock(), opts)
}

/// The REPL loop over explicit streams. All REPL chatter (prompts, values,
/// errors) goes to `out`; a compiled expression's process inherits the real
/// stdout/stderr, which under Emacs is the same comint pipe.
pub fn run<R: BufRead, W: Write>(mut input: R, mut out: W, opts: Opts) -> i32 {
    let mut session = Session::new(opts.link_flags, !opts.isolate);
    let _ = if opts.isolate {
        writeln!(
            out,
            "Coil REPL (--isolate) — definitions persist (and redefine by name); each\n\
             expression runs in a fresh process. :help for commands, :quit or C-d to exit."
        )
    } else {
        writeln!(
            out,
            "Coil REPL — definitions persist (and redefine by name); (def x EXPR) binds\n\
             a value for later inputs. :help for commands, :quit or C-d to exit."
        )
    };
    loop {
        let _ = write!(out, "coil> ");
        let _ = out.flush();
        let mut buf = String::new();
        match input.read_line(&mut buf) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                let _ = writeln!(out, "error reading input: {e}");
                break;
            }
        }
        // Accumulate lines until every opened delimiter/string is closed.
        while input_incomplete(&buf) {
            let _ = write!(out, "....> ");
            let _ = out.flush();
            let mut line = String::new();
            match input.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => buf.push_str(&line),
                Err(_) => break,
            }
        }
        let text = buf.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(cmd) = text.strip_prefix(':') {
            if handle_command(cmd, &mut session, &mut out) {
                break;
            }
            continue;
        }
        handle_input(text, &mut session, &mut out);
    }
    let _ = writeln!(out);
    0
}

/// Is `buf` a prefix of a form — i.e. does reading it fail only because the
/// input ENDS too soon (unclosed delimiter or string)? A malformed-but-complete
/// input reads as complete so its error surfaces immediately.
fn input_incomplete(buf: &str) -> bool {
    match reader::read_all(buf, 0) {
        Ok(_) => false,
        Err(d) => {
            let m = d.msg.as_str();
            m.contains("unclosed") || m.contains("unexpected end of input") || m.contains("unterminated")
        }
    }
}

/// Handle a `:command`. Returns true to exit the REPL.
fn handle_command<W: Write>(cmd: &str, session: &mut Session, out: &mut W) -> bool {
    let (name, rest) = match cmd.find(char::is_whitespace) {
        Some(i) => (&cmd[..i], cmd[i..].trim()),
        None => (cmd, ""),
    };
    match name {
        "q" | "quit" | "exit" => return true,
        "help" | "h" => {
            let _ = writeln!(
                out,
                "input:      a top-level definition (defn/defstruct/defsum/deftrait/impl/\n\
                 \x20           extern/const/import/…) persists in the session and redefines\n\
                 \x20           by name; anything else is an expression — it is compiled and\n\
                 \x20           run, and its value printed by inferred type.\n\
                 \x20           (def x EXPR) binds x's VALUE for later inputs (live mode).\n\
                 commands:   :help          this text\n\
                 \x20           :type EXPR    infer and print EXPR's type (nothing runs)\n\
                 \x20           :session      print the accumulated program (and bindings)\n\
                 \x20           :load FILE    add FILE's top-level forms to the session\n\
                 \x20           :run          build the session's `main` and run it (like\n\
                 \x20                         `coil run`) — how you drive a loaded example\n\
                 \x20           :reset        forget every definition and binding\n\
                 \x20           :quit         exit (C-d also works)\n\
                 notes:      live mode (the default) loads each eval into THIS process, so\n\
                 \x20           (def …) bindings and malloc'd memory persist — a mutable cell\n\
                 \x20           is (def c (alloc-heap T)) then (store! c …); a crash takes the\n\
                 \x20           session with it. --isolate runs each expression in a fresh\n\
                 \x20           process instead: crash-proof, but nothing persists and a\n\
                 \x20           session `main` is excluded from expression programs there."
            );
        }
        "reset" => {
            session.defs.clear();
            if let Some(h) = &mut session.host {
                // Forget bindings; the cells and loaded dylibs stay (something
                // printed earlier may still be looked at — and code can't unload).
                h.bindings.clear();
            }
            let _ = writeln!(out, "session cleared");
        }
        "session" => {
            let _ = write!(out, "{}", session.source(false));
            if let Some(h) = &session.host {
                for b in &h.bindings {
                    let _ = writeln!(out, "; (def {}) : {}", b.name, ty_str(&b.ty));
                }
            }
        }
        "type" => {
            if rest.is_empty() {
                let _ = writeln!(out, "usage: :type EXPR");
            } else {
                match infer(session, rest) {
                    Ok((ty, _)) => {
                        let _ = writeln!(out, "{}", ty_str(&strip_refs(&ty)));
                    }
                    Err(e) => print_err(out, &e),
                }
            }
        }
        "load" => {
            if rest.is_empty() {
                let _ = writeln!(out, "usage: :load path/to/file.coil");
                return false;
            }
            load_file(rest.trim_matches('"'), session, out);
        }
        "run" => {
            run_session_main(session, out);
        }
        _ => {
            let _ = writeln!(out, "unknown command :{name} (:help lists them)");
        }
    }
    false
}

/// `:load FILE` — add every top-level form of a Coil source file to the session
/// (its `(module …)` header is dropped; the session owns the module). The whole
/// file is validated as one candidate: it either all enters the session or none
/// of it does.
fn load_file<W: Write>(path: &str, session: &mut Session, out: &mut W) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(out, "error reading {path}: {e}");
            return;
        }
    };
    let forms = match reader::read_all(&src, 0) {
        Ok(f) => f,
        Err(d) => {
            let _ = writeln!(out, "error reading {path}: {}", d.msg);
            return;
        }
    };
    // Stage all forms, then validate once.
    let mut staged: Vec<(Option<String>, String)> = Vec::new();
    for form in &forms {
        if head_of(form).as_deref() == Some("module") {
            continue;
        }
        let text = form_text(&src, form);
        staged.push((def_key(form), text.to_string()));
    }
    // Validation-only clone (no host: `check_source` is all that runs on it).
    let mut trial = Session::new(session.link_flags.clone(), false);
    trial.defs = session
        .defs
        .iter()
        .map(|d| Def { key: d.key.clone(), src: d.src.clone() })
        .collect();
    for (key, text) in &staged {
        let (_, at) = trial.candidate(key, text);
        trial.commit(key.clone(), text.clone(), at);
    }
    match crate::check_source(&trial.source(false)) {
        Ok(()) => {
            let n = staged.len();
            session.defs = trial.defs;
            let _ = writeln!(out, "loaded {n} form(s) from {path}");
        }
        Err(e) => print_err(out, &e),
    }
}

/// Handle one complete non-command input: every form it contains, in order.
fn handle_input<W: Write>(text: &str, session: &mut Session, out: &mut W) {
    let forms = match reader::read_all(text, 0) {
        Ok(f) => f,
        Err(d) => {
            let _ = writeln!(out, "error: {}", d.msg);
            return;
        }
    };
    for form in &forms {
        handle_form(text, form_text(text, form), form, session, out);
    }
}

/// Split `(def name EXPR)` into the name and the expression's source text
/// (sliced out of the full input by span, so nesting/newlines survive intact).
fn def_parts<'a>(input: &'a str, form: &Sexp) -> Result<(String, &'a str), String> {
    let items = match &form.kind {
        SexpKind::List(items) => items,
        _ => return Err("def: expected (def name EXPR)".to_string()),
    };
    if items.len() != 3 {
        return Err("def: expected exactly (def name EXPR)".to_string());
    }
    let name = match &items[1].kind {
        SexpKind::Sym(s) => s.clone(),
        _ => return Err("def: the name must be a symbol".to_string()),
    };
    if name.starts_with("coil-repl--") {
        return Err("def: the coil-repl-- prefix is reserved".to_string());
    }
    let (lo, hi) = (items[2].span.lo as usize, items[2].span.hi as usize);
    if lo > hi || hi > input.len() {
        return Err("def: internal — bad expression span".to_string());
    }
    Ok((name, &input[lo..hi]))
}

/// The source text of one form (sliced back out of the input by its span).
fn form_text<'a>(input: &'a str, form: &Sexp) -> &'a str {
    let (lo, hi) = (form.span.lo as usize, form.span.hi as usize);
    if lo <= hi && hi <= input.len() {
        &input[lo..hi]
    } else {
        input
    }
}

fn head_of(form: &Sexp) -> Option<String> {
    if let SexpKind::List(items) = &form.kind {
        if let Some(Sexp { kind: SexpKind::Sym(h), .. }) = items.first() {
            return Some(h.clone());
        }
    }
    None
}

/// Top-level definition heads. `derive` is a stdlib macro but always a
/// definition when it appears at top level, so it classifies here too.
const DEF_HEADS: &[&str] = &[
    "defn", "defstruct", "defsum", "deftrait", "impl", "extern", "const", "defcc", "import",
    "include", "export", "export-c", "static-assert", "meta", "derive",
];

/// The redefinition key for a definition form: which earlier session entry this
/// one replaces. `None` = appended (deduplicated only if the text is identical).
fn def_key(form: &Sexp) -> Option<String> {
    let items = match &form.kind {
        SexpKind::List(items) => items,
        _ => return None,
    };
    let head = match items.first().map(|s| &s.kind) {
        Some(SexpKind::Sym(h)) => h.as_str(),
        _ => return None,
    };
    let sym_at = |i: usize| match items.get(i).map(|s| &s.kind) {
        Some(SexpKind::Sym(s)) => Some(s.clone()),
        _ => None,
    };
    match head {
        "defn" | "defstruct" | "defsum" | "deftrait" | "const" | "defcc" | "extern" => {
            Some(format!("{head} {}", sym_at(1)?))
        }
        "impl" => Some(format!("impl {} {}", sym_at(1)?, sym_at(2)?)),
        "derive" => Some(format!("impl {} {}", sym_at(1)?, sym_at(2)?)), // (derive Eq T) generates (impl Eq T …)
        "import" => match items.get(1).map(|s| &s.kind) {
            Some(SexpKind::Str(p)) => Some(format!("import {p}")),
            _ => None,
        },
        _ => None,
    }
}

fn handle_form<W: Write>(input: &str, text: &str, form: &Sexp, session: &mut Session, out: &mut W) {
    let head = head_of(form);
    if head.as_deref() == Some("module") {
        let _ = writeln!(out, "; the REPL owns the module header — ignored");
        return;
    }
    // `(def name EXPR)` — bind a value into the live session.
    if head.as_deref() == Some("def") {
        let r = def_parts(input, form).and_then(|(name, expr)| {
            if session.host.is_some() {
                eval_live(expr, Some(&name), session, out)
            } else {
                Err("(def …) needs the live session — run `coil repl` without --isolate".to_string())
            }
        });
        if let Err(e) = r {
            print_err(out, &e);
        }
        return;
    }
    if head.as_deref().is_some_and(|h| DEF_HEADS.contains(&h)) {
        match try_def(text, form, session) {
            Ok(msg) => {
                let _ = writeln!(out, "{msg}");
            }
            Err(e) => print_err(out, &e),
        }
        return;
    }
    // Expression. If it fails to infer, it may still be a top-level macro call
    // that GENERATES definitions (e.g. a user's derive-like macro) — try it as a
    // definition before reporting; the expression error stays primary.
    let r = if session.host.is_some() {
        eval_live(text, None, session, out)
    } else {
        handle_expr(text, session, out)
    };
    match r {
        Ok(()) => {}
        Err(expr_err) => {
            if head.is_some() {
                if let Ok(msg) = try_def(text, form, session) {
                    let _ = writeln!(out, "{msg}");
                    return;
                }
            }
            print_err(out, &expr_err);
        }
    }
}

/// Validate `text` as a definition against the whole session; commit on success.
/// Returns the acknowledgement line to print.
fn try_def(text: &str, form: &Sexp, session: &mut Session) -> Result<String, String> {
    let key = def_key(form);
    // A keyless duplicate (same text already present) is a no-op — resending a
    // buffer region shouldn't stack identical imports/exports.
    if key.is_none() && session.defs.iter().any(|d| d.src == text) {
        return Ok("ok (unchanged)".to_string());
    }
    let (candidate, at) = session.candidate(&key, text);
    crate::check_source(&candidate).map_err(|e| e.replace("<source>", "repl"))?;
    let mut ack = match &key {
        Some(k) if k.starts_with("import ") => "ok".to_string(),
        Some(k) => {
            let name = k.split_whitespace().last().unwrap_or(k);
            format!("#'repl/{name}")
        }
        None => "ok".to_string(),
    };
    // REDEFINING a struct/sum changes its layout; a live binding of that type
    // would reinterpret its old cell with the new layout — silent corruption.
    // Drop the affected bindings, loudly.
    if at.is_some() {
        if let Some(k) = key.as_deref() {
            if let Some((kind, name)) = k.split_once(' ') {
                if matches!(kind, "defstruct" | "defsum") {
                    if let Some(h) = &mut session.host {
                        let dropped: Vec<String> = h
                            .bindings
                            .iter()
                            .filter(|b| type_mentions(&b.ty, name))
                            .map(|b| b.name.clone())
                            .collect();
                        if !dropped.is_empty() {
                            h.bindings.retain(|b| !type_mentions(&b.ty, name));
                            ack.push_str(&format!(
                                "\n; dropped binding(s) {} — their type's layout was redefined",
                                dropped.join(", ")
                            ));
                        }
                    }
                }
            }
        }
    }
    session.commit(key, text.to_string(), at);
    Ok(ack)
}

/// Does `ty` mention the named struct/sum (by its BARE name — binding types
/// carry resolved `module.Name` forms)?
fn type_mentions(ty: &Type, bare: &str) -> bool {
    match ty {
        Type::Struct(n) => display_name(n) == bare,
        Type::App(n, args) => {
            display_name(n) == bare || args.iter().any(|a| type_mentions(a, bare))
        }
        Type::Ptr(p) | Type::Ref(_, p) | Type::Slice(p) | Type::Array(p, _) | Type::Vec(p, _) => {
            type_mentions(p, bare)
        }
        Type::Fn(_, params, ret) => {
            params.iter().any(|p| type_mentions(p, bare)) || type_mentions(ret, bare)
        }
        _ => false,
    }
}

/// The `let` binding pairs that bring the slot table and every live binding
/// into scope, typed from the driver's records:
///
/// ```text
/// coil-repl--slots (cast (ptr (array (ptr i8) N)) env)
/// x (load (cast (ptr i64) (load (index coil-repl--slots 0))))
/// …
/// ```
fn prologue_pairs(host: &Host, n_slots: usize) -> String {
    let mut s = format!("coil-repl--slots (cast (ptr (array (ptr i8) {n_slots})) env)");
    for b in &host.bindings {
        s.push_str(&format!(
            "\n        {} (load (cast (ptr {}) (load (index coil-repl--slots {}))))",
            b.name,
            type_syntax(&b.ty),
            b.slot
        ));
    }
    s
}

/// Infer the type of `expr` in the session (nothing runs). Returns the type and
/// the resolved probe program (whose struct/sum tables drive value display).
/// Live bindings are in scope via the same prologue the entry function gets.
fn infer(session: &Session, expr: &str) -> Result<(Type, Program), String> {
    let (pro_open, pro_close) = match &session.host {
        Some(h) if !h.bindings.is_empty() => {
            (format!("(let [{}]\n", prologue_pairs(h, h.bindings.len())), ")".to_string())
        }
        _ => (String::new(), String::new()),
    };
    // The probe is checked with the same bare stdlib the eval programs get, so
    // inference of `(slice-len …)`/`(str-len …)`/`(printf …)` resolves here too
    // (this is where "undefined function" errors would otherwise surface).
    // Always INCLUDE the session's `main`: it's an ordinary `repl.main` function
    // now, so `(main)` type-checks. The probe defines `PROBE`, not `main`, so
    // there is no entry-point collision in either mode.
    let defs = session.source(false);
    let with_imports =
        format!("(module repl)\n{REPL_IMPORTS}{}", &defs["(module repl)\n".len()..]);
    let probe_src = format!(
        "{with_imports}(defn {PROBE} [(env (ptr i8))] (-> void)\n{pro_open}{expr}\n{pro_close})\n",
    );
    // The resolver namespaces definitions to their module: the probe's checked
    // name is `repl.<probe>`.
    crate::repl_infer_tail(&probe_src, &format!("repl.{PROBE}")).map_err(|e| {
        e.replace("<source>", "repl")
            .replace(&format!("function 'repl.{PROBE}': "), "")
            .replace(&format!("in 'repl.{PROBE}': "), "")
            .replace(&format!("repl.{PROBE}"), "expression")
    })
}

/// Live mode: compile one expression (or `(def name EXPR)`) into a dylib and
/// run it INSIDE this process. The generated entry reads every live binding out
/// of the slot table (typed prologue), runs the expression, prints the value,
/// and — for a def — writes the new value cell's pointer back into its slot.
fn eval_live<W: Write>(
    expr: &str,
    def_name: Option<&str>,
    session: &mut Session,
    out: &mut W,
) -> Result<(), String> {
    let (ty, program) = infer(session, expr)?;
    let shown = strip_refs(&ty);
    if shown == Type::Code {
        return Err(
            "this expression is compile-time Code (a macro value); the REPL runs runtime \
             programs — call the macro in a runtime position instead"
                .to_string(),
        );
    }
    if let Some(name) = def_name {
        if matches!(shown, Type::Void | Type::Never) {
            return Err(format!(
                "(def {name} …): the expression has type {} — there is no value to bind",
                ty_str(&shown)
            ));
        }
        // A binding shadows a same-named session definition in every later
        // eval's prologue — confusing enough to reject outright.
        let clash = ["defn", "const", "defstruct", "defsum"]
            .iter()
            .any(|k| session.defs.iter().any(|d| d.key.as_deref() == Some(&format!("{k} {name}"))));
        if clash {
            return Err(format!(
                "'{name}' is already a session definition; pick another name"
            ));
        }
    }
    let host = session.host.as_ref().expect("eval_live requires live mode");
    let is_def = def_name.is_some();
    // A def's new slot is the next UNUSED slot — env.len(), NOT bindings.len():
    // a rebound name leaves its old slot occupied, so the two counts diverge
    // (using bindings.len() here once stomped a live binding's slot).
    let new_slot = host.env.len();
    let n_slots = new_slot + usize::from(is_def);

    let mut src = String::from("(module repl)\n");
    src.push_str(REPL_IMPORTS);
    src.push_str(&session.source(false)["(module repl)\n".len()..]);

    // A Ref-typed expression (a place) is loaded explicitly when its target is
    // a scalar; aggregates auto-load at call/store boundaries.
    let loaded = if matches!(ty, Type::Ref(..)) && is_scalar(&shown) {
        format!("(load {expr})")
    } else {
        expr.to_string()
    };
    let mut gen = ShowGen::new(&program);
    let mut body = String::new();
    match (def_name, &shown) {
        (None, Type::Void | Type::Never) => {
            body.push_str(&format!("    {expr}\n"));
        }
        (None, _) => {
            let show = gen.show_fn(&shown);
            body.push_str(&format!("    ({show}\n{loaded}\n)\n    (print \"{{c}}\" 10)\n"));
        }
        (Some(name), _) => {
            // Bind: evaluate, copy the value into a heap cell (process-lifetime
            // — malloc'd memory outlives this dylib call), publish the cell's
            // pointer into the def's slot, and show what was stored.
            let show = gen.show_fn(&shown);
            let ack = print_literal(&format!("{name} = "));
            body.push_str(&format!(
                "    (let [coil-repl--v\n{loaded}\n]\n      \
                 (let [coil-repl--slot (alloc-heap {tsyn})]\n        \
                 (store! coil-repl--slot coil-repl--v)\n        \
                 (store! (index coil-repl--slots {new_slot}) (cast (ptr i8) coil-repl--slot))\n        \
                 {ack}\n        \
                 ({show} (load coil-repl--slot))\n        \
                 (print \"{{c}}\" 10)))\n",
                tsyn = type_syntax(&shown)
            ));
        }
    }
    for h in &gen.helpers {
        src.push_str(h);
        src.push('\n');
    }
    if is_def || !host.bindings.is_empty() {
        let pairs = prologue_pairs(host, n_slots);
        src.push_str(&format!(
            "(defn coil-repl--entry [(env (ptr i8))] (-> i64)\n  (let [{pairs}]\n  (do\n{body}    0)))\n"
        ));
    } else {
        src.push_str(&format!(
            "(defn coil-repl--entry [(env (ptr i8))] (-> i64)\n  (do\n{body}    0))\n"
        ));
    }
    session.evals += 1;
    let sym = format!("coil_repl_entry_{}", session.evals);
    src.push_str(&format!("(export-c [coil-repl--entry :as \"{sym}\"])\n"));

    let dylib = std::env::temp_dir().join(format!(
        "coil_repl_{}_{}.dylib",
        std::process::id(),
        session.evals
    ));
    let triple = crate::codegen::target_triple();
    crate::build_shared_lib(&src, &dylib, triple, &session.link_flags, None)
        .map_err(|e| e.replace("<source>", "repl"))?;
    let _ = out.flush();
    let host = session.host.as_mut().expect("eval_live requires live mode");
    host.load_and_run(&dylib, &sym, n_slots)?;
    if let Some(name) = def_name {
        if host.env[new_slot].is_null() {
            return Err(format!(
                "(def {name} …): internal — the eval didn't publish a value cell"
            ));
        }
        // Rebinding replaces the record; the old cell and slot stay allocated
        // (REPL leak policy — something may still point at them).
        host.bindings.retain(|b| b.name != name);
        host.bindings.push(Binding { name: name.to_string(), ty: shown, slot: new_slot });
    }
    Ok(())
}

/// Isolate mode (`--isolate`): compile and run one expression as a fresh
/// PROCESS — infer its type, generate a `main` that prints its value, build,
/// run, report a nonzero exit.
fn handle_expr<W: Write>(expr: &str, session: &mut Session, out: &mut W) -> Result<(), String> {
    let (ty, program) = infer(session, expr)?;
    let shown = strip_refs(&ty);
    if shown == Type::Code {
        return Err(
            "this expression is compile-time Code (a macro value); the REPL runs runtime \
             programs — call the macro in a runtime position instead"
                .to_string(),
        );
    }

    let mut src = String::from("(module repl)\n");
    src.push_str(REPL_IMPORTS);
    // INCLUDE the session's `main` (an ordinary `repl.main` now). The generated
    // entry is named `coil-repl--main` and claims the C `main` symbol via
    // `export-c`, so it doesn't collide with the user's `repl.main` — and the
    // expression can call `(main)`.
    src.push_str(&session.source(false)[ "(module repl)\n".len() ..]);

    let mut main_body = String::new();
    match shown {
        Type::Void | Type::Never => {
            main_body.push_str(&format!("    {expr}\n"));
        }
        _ => {
            let mut gen = ShowGen::new(&program);
            let show = gen.show_fn(&shown);
            for h in &gen.helpers {
                src.push_str(h);
                src.push('\n');
            }
            // A Ref-typed expression (a place, e.g. `(field p x)`) is loaded
            // explicitly when its target is a scalar; aggregates auto-load at
            // the call boundary.
            let arg = if matches!(ty, Type::Ref(..)) && is_scalar(&shown) {
                format!("(load {expr})")
            } else {
                expr.to_string()
            };
            main_body.push_str(&format!("    ({show}\n{arg}\n)\n    (print \"{{c}}\" 10)\n"));
        }
    }
    src.push_str(&format!("(defn coil-repl--main [] (-> i64)\n  (do\n{main_body}    0))\n"));
    src.push_str("(export-c [coil-repl--main :as \"main\"])\n");

    session.evals += 1;
    let exe: PathBuf = std::env::temp_dir().join(format!(
        "coil_repl_{}_{}",
        std::process::id(),
        session.evals
    ));
    let triple = crate::codegen::target_triple();
    crate::build_executable_linked_dbg(&src, &exe, triple, &session.link_flags, None)
        .map_err(|e| e.replace("<source>", "repl"))?;
    // The REPL's own chatter is flushed before the child writes to the same fd.
    let _ = out.flush();
    let status = std::process::Command::new(&exe).status();
    let _ = std::fs::remove_file(&exe);
    match status {
        Ok(s) => {
            if let Some(code) = s.code() {
                if code != 0 {
                    let _ = writeln!(out, "; process exited with code {code}");
                }
            } else {
                let _ = writeln!(out, "; process killed by a signal");
            }
            Ok(())
        }
        Err(e) => Err(format!("running compiled expression: {e}")),
    }
}

/// `:run` — build the WHOLE session (including the user's `main`) as a real
/// executable and run it, exactly like `coil run file`. This is how you drive
/// an example whose entry point is `main`: `main` is otherwise uncallable from
/// the REPL (the resolver keeps it bare, and expression programs generate their
/// own entry point that excludes it). The process's exit code is reported —
/// many examples use it as the program's answer.
fn run_session_main<W: Write>(session: &mut Session, out: &mut W) {
    let has_main = session.defs.iter().any(|d| d.key.as_deref() == Some("defn main"));
    if !has_main {
        let _ = writeln!(
            out,
            "no `main` in the session — :load a file that defines one, or just \
             evaluate an expression directly"
        );
        return;
    }
    let src = session.source(false);
    session.evals += 1;
    let exe: PathBuf = std::env::temp_dir().join(format!(
        "coil_repl_main_{}_{}",
        std::process::id(),
        session.evals
    ));
    let triple = crate::codegen::target_triple();
    if let Err(e) = crate::build_executable_linked_dbg(&src, &exe, triple, &session.link_flags, None)
    {
        print_err(out, &e.replace("<source>", "repl"));
        return;
    }
    // Flush the REPL's own chatter before the child writes to the same fd.
    let _ = out.flush();
    let status = std::process::Command::new(&exe).status();
    let _ = std::fs::remove_file(&exe);
    match status {
        Ok(s) => match s.code() {
            Some(code) => {
                let _ = writeln!(out, "; main exited with code {code}");
            }
            None => {
                let _ = writeln!(out, "; main killed by a signal");
            }
        },
        Err(e) => {
            let _ = writeln!(out, "; failed to run: {e}");
        }
    }
}

fn print_err<W: Write>(out: &mut W, e: &str) {
    let _ = writeln!(out, "error: {e}");
}

fn strip_refs(t: &Type) -> Type {
    match t {
        Type::Ref(_, inner) => strip_refs(inner),
        other => other.clone(),
    }
}

/// Scalars pass by value and print via a `load` when read out of a place;
/// aggregates pass as places and auto-load at call boundaries.
fn is_scalar(t: &Type) -> bool {
    matches!(
        t,
        Type::Int(..) | Type::Float(_) | Type::Bool | Type::Ptr(_) | Type::Fn(..) | Type::Vec(..)
    )
}

// ---------------------------------------------------------------------------
// Value display: generate one `(defn coil-repl--showN [(v T)] (-> i64) …)` per
// concrete type, memoized by the type's printed form so recursive types
// terminate (the helper simply calls itself).
// ---------------------------------------------------------------------------

struct ShowGen<'a> {
    program: &'a Program,
    helpers: Vec<String>,
    by_key: HashMap<String, String>,
}

impl<'a> ShowGen<'a> {
    fn new(program: &'a Program) -> ShowGen<'a> {
        ShowGen { program, helpers: Vec::new(), by_key: HashMap::new() }
    }

    /// The name of the show helper for `ty`, generating it (and its
    /// dependencies) on first use.
    fn show_fn(&mut self, ty: &Type) -> String {
        let key = ty_str(ty);
        if let Some(name) = self.by_key.get(&key) {
            return name.clone();
        }
        let name = format!("coil-repl--show{}", self.by_key.len());
        // Register BEFORE generating the body: a recursive type's body calls
        // its own helper.
        self.by_key.insert(key, name.clone());
        let mut stmts = Vec::new();
        self.stmts_for("v", ty, true, &mut stmts);
        let body = stmts
            .iter()
            .map(|s| format!("    {s}\n"))
            .collect::<String>();
        self.helpers.push(format!(
            "(defn {name} [(v {})] (-> i64)\n  (do\n{body}    0))\n",
            type_syntax(ty)
        ));
        name
    }

    /// Statements that print the value at `place` of type `ty`. `is_value`:
    /// `place` is already a loaded value (a parameter or pattern binding), not a
    /// pointer-like place that scalars must be `load`ed from.
    fn stmts_for(&mut self, place: &str, ty: &Type, is_value: bool, out: &mut Vec<String>) {
        let val = |scalar_place: &str| {
            if is_value {
                scalar_place.to_string()
            } else {
                format!("(load {scalar_place})")
            }
        };
        match ty {
            Type::Int(64, true) => out.push(format!("(print \"{{d}}\" {})", val(place))),
            Type::Int(bits, true) if *bits < 64 => {
                out.push(format!("(print \"{{d}}\" (cast :i64 {}))", val(place)))
            }
            // print-u takes an i64 whose BITS it reads unsigned, so every
            // unsigned width goes through a cast (same-width reinterpret for
            // u64, zero-extension below it).
            Type::Int(bits, false) if *bits <= 64 => {
                out.push(format!("(print \"{{u}}\" (cast :i64 {}))", val(place)))
            }
            Type::Int(..) | Type::Vec(..) | Type::Fn(..) => {
                // No honest 64-bit rendering — show the type, visibly.
                out.push(print_literal(&format!("#<{}>", ty_str(ty))));
            }
            Type::Bool => out.push(format!("(print \"{{b}}\" {})", val(place))),
            Type::Float(64) => out.push(format!("(print \"{{f}}\" {})", val(place))),
            Type::Float(_) => out.push(format!("(print \"{{f}}\" (cast :f64 {}))", val(place))),
            Type::Ptr(_) => {
                out.push(format!("(print \"#<ptr 0x{{x}}>\" (cast :i64 {}))", val(place)))
            }
            Type::Slice(elem) if **elem == Type::Int(8, false) => {
                // A string: print quoted. `{c}` writes the quote bytes so the
                // format string needs no escape support.
                out.push(format!("(print \"{{c}}{{s}}{{c}}\" 34 {} 34)", val(place)));
            }
            Type::Slice(elem) => {
                let elem_show = self.show_fn(elem);
                out.push("(print \"{c}\" 91)".to_string()); // '['
                out.push(format!(
                    "(for [coil-repl--i 0 (coil-repl-slice/slice-len {v})]\n      \
                     (do (if (icmp-gt coil-repl--i 0) (do (print \"{{c}}\" 32) 0) 0)\n          \
                     ({elem_show} (coil-repl-slice/slice-get {v} coil-repl--i))))",
                    v = val(place)
                ));
                out.push("(print \"{c}\" 93)".to_string()); // ']'
            }
            Type::Array(elem, n) => {
                let elem_show = self.show_fn(elem);
                let elem_arg = if is_scalar(elem) {
                    format!("(load (index {place} coil-repl--i))")
                } else {
                    format!("(index {place} coil-repl--i)")
                };
                out.push("(print \"{c}\" 91)".to_string());
                out.push(format!(
                    "(for [coil-repl--i 0 {n}]\n      \
                     (do (if (icmp-gt coil-repl--i 0) (do (print \"{{c}}\" 32) 0) 0)\n          \
                     ({elem_show} {elem_arg})))"
                ));
                out.push("(print \"{c}\" 93)".to_string());
            }
            Type::Struct(name) => self.named_stmts(place, name, &[], out),
            Type::App(name, args) => self.named_stmts(place, name, args, out),
            Type::Ref(_, inner) => {
                // A ref-typed field/element: the pointee is what to show.
                self.stmts_for(place, inner, false, out)
            }
            Type::Void | Type::Never | Type::Code => {
                // Handled before display generation; nested occurrences are
                // impossible in a checked value type. Render visibly anyway.
                out.push(print_literal(&format!("#<{}>", ty_str(ty))));
            }
        }
    }

    /// Display for a named type (struct or sum), `args` = generic arguments.
    fn named_stmts(&mut self, place: &str, name: &str, args: &[Type], out: &mut Vec<String>) {
        // Clone the definition data out of `self.program` up front: generating
        // nested helpers below needs `&mut self`.
        let struct_def = self
            .program
            .structs
            .iter()
            .find(|s| s.name == name)
            .map(|s| (matches!(s.layout, Layout::Bits(_)), s.type_params.clone(), s.fields.clone()));
        if let Some((is_bits, type_params, fields)) = struct_def {
            if is_bits {
                out.push(print_literal(&format!("#<bits {}>", display_name(name))));
                return;
            }
            let subst: HashMap<&str, &Type> =
                type_params.iter().map(String::as_str).zip(args.iter()).collect();
            out.push(print_literal(&format!("({}", display_name(name))));
            for (fname, fty) in &fields {
                let fty = subst_type(fty, &subst);
                out.push(print_literal(&format!(" :{fname} ")));
                let fplace = format!("(field {place} {fname})");
                if is_scalar(&fty) || matches!(fty, Type::Slice(_)) {
                    self.stmts_for(&fplace, &fty, false, out);
                } else {
                    let show = self.show_fn(&fty);
                    out.push(format!("({show} {fplace})"));
                }
            }
            out.push("(print \"{c}\" 41)".to_string()); // ')'
            return;
        }
        let sum_def = self
            .program
            .sums
            .iter()
            .find(|s| s.name == name)
            .map(|s| (s.type_params.clone(), s.variants.clone()));
        if let Some((type_params, variants)) = sum_def {
            let subst: HashMap<&str, &Type> =
                type_params.iter().map(String::as_str).zip(args.iter()).collect();
            let mut arms = String::new();
            for v in variants {
                let binds: Vec<String> =
                    (0..v.fields.len()).map(|i| format!("coil-repl--f{i}")).collect();
                let mut vstmts = Vec::new();
                if v.fields.is_empty() {
                    vstmts.push(print_literal(&format!("({})", display_name(&v.name))));
                } else {
                    vstmts.push(print_literal(&format!("({}", display_name(&v.name))));
                    for (i, (_, fty)) in v.fields.iter().enumerate() {
                        let fty = subst_type(fty, &subst);
                        vstmts.push("(print \"{c}\" 32)".to_string()); // ' '
                        // Pattern bindings are values (scalars) or places
                        // (aggregates) — both usable directly.
                        self.stmts_for(&binds[i], &fty, true, &mut vstmts);
                    }
                    vstmts.push("(print \"{c}\" 41)".to_string());
                }
                let body = vstmts.join("\n           ");
                arms.push_str(&format!(
                    "      ({} [{}]\n        (do {body}\n            0))\n",
                    v.name,
                    binds.join(" ")
                ));
            }
            out.push(format!("(match {place}\n{arms}    )"));
            return;
        }
        out.push(print_literal(&format!("#<{}>", display_name(name))));
    }
}

/// A `(print "…")` of literal text, brace-escaped for the fmt spec syntax.
fn print_literal(text: &str) -> String {
    let escaped = text.replace('{', "{{").replace('}', "}}");
    format!("(print \"{escaped}\")")
}

/// The display name of a resolved (module-qualified) definition name:
/// `coil.core.Some` shows as `Some`, `repl.Point` as `Point`. Only DISPLAY text
/// is stripped — match arms and type annotations keep the qualified name, which
/// is what resolves.
fn display_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// Substitute a generic definition's type parameters with concrete arguments.
fn subst_type(t: &Type, subst: &HashMap<&str, &Type>) -> Type {
    match t {
        Type::Struct(n) => match subst.get(n.as_str()) {
            Some(rep) => (*rep).clone(),
            None => t.clone(),
        },
        Type::Ptr(p) => Type::Ptr(Box::new(subst_type(p, subst))),
        Type::Ref(m, p) => Type::Ref(*m, Box::new(subst_type(p, subst))),
        Type::Slice(e) => Type::Slice(Box::new(subst_type(e, subst))),
        Type::Array(e, n) => Type::Array(Box::new(subst_type(e, subst)), *n),
        Type::Vec(e, n) => Type::Vec(Box::new(subst_type(e, subst)), *n),
        Type::App(n, args) => {
            Type::App(n.clone(), args.iter().map(|a| subst_type(a, subst)).collect())
        }
        Type::Fn(cc, params, ret) => Type::Fn(
            cc.clone(),
            params.iter().map(|p| subst_type(p, subst)).collect(),
            Box::new(subst_type(ret, subst)),
        ),
        _ => t.clone(),
    }
}

/// Surface syntax for a type, as it appears in a parameter annotation. This is
/// `ty_str` — its output IS the surface syntax for every type a show helper
/// takes (helpers are never generated for `fnptr`/`vec`/refs).
fn type_syntax(t: &Type) -> String {
    ty_str(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(src: &str) -> Sexp {
        reader::read_all(src, 0).unwrap().remove(0)
    }

    #[test]
    fn incomplete_detection() {
        assert!(input_incomplete("(defn f [] (-> i64)"));
        assert!(input_incomplete("(print \"unterminated"));
        assert!(!input_incomplete("(iadd 1 2)"));
        assert!(!input_incomplete("(iadd 1 2))")); // malformed but complete
        assert!(!input_incomplete("42"));
    }

    #[test]
    fn def_keys() {
        assert_eq!(def_key(&form("(defn f [] (-> i64) 1)")).as_deref(), Some("defn f"));
        assert_eq!(def_key(&form("(defstruct P [(x i64)])")).as_deref(), Some("defstruct P"));
        assert_eq!(def_key(&form("(impl Eq P (= [(a P) (b P)] (-> bool) true))")).as_deref(), Some("impl Eq P"));
        assert_eq!(def_key(&form("(derive Eq P)")).as_deref(), Some("impl Eq P"));
        assert_eq!(def_key(&form("(import \"io.coil\" :use *)")).as_deref(), Some("import io.coil"));
        assert_eq!(def_key(&form("(iadd 1 2)")), None);
    }

    #[test]
    fn redefinition_replaces_in_place() {
        let mut s = Session::new(Vec::new(), false);
        s.commit(Some("defn f".into()), "(defn f [] (-> i64) 1)".into(), None);
        s.commit(Some("defn g".into()), "(defn g [] (-> i64) 2)".into(), None);
        let (cand, at) = s.candidate(&Some("defn f".into()), "(defn f [] (-> i64) 9)");
        assert_eq!(at, Some(0));
        assert!(cand.contains("(-> i64) 9)"));
        assert!(!cand.contains("(-> i64) 1)"));
        assert!(cand.contains("defn g"));
    }
}
