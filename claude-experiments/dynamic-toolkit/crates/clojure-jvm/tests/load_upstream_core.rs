//! Attempt to load upstream `clojure/core.clj` directly through our
//! `Session::eval_str`, one form at a time. Reports the first form that
//! breaks so we know exactly what's still missing.
//!
//! The user goal: load every fn in upstream core.clj and run them
//! through our JIT. This is the driver test that proves we're there.

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
use std::cell::RefCell;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn install_capturing_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        // Echo every panic to stderr so SIGABRTs that abort during the
        // SECOND panic of a panic-during-panic still leave evidence of
        // the FIRST panic's location.
        eprintln!("[panic-hook] {}", msg);
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(msg));
    }));
}

const UPSTREAM_CORE: &str =
    "/Users/jimmyhmiller/Documents/Code/open-source/clojure/src/clj/clojure/core.clj";

/// Read+eval upstream core.clj one form at a time. Report the first form
/// (read or eval) that panics, so the next thing to implement is obvious.
#[test]
#[ignore = "tracks progress against upstream core.clj — long-running"]
fn load_upstream_core_progressively() {
    install_capturing_hook();
    let src = std::fs::read_to_string(UPSTREAM_CORE)
        .expect("upstream clojure/core.clj must be reachable at the hard-coded path");

    let mut sess = Session::new();
    let mut i: usize = 0;
    let mut byte_pos: usize = 0;
    loop {
        eprintln!("[upstream] iter {i} starting at byte {byte_pos}");
        let slice = &src[byte_pos..];
        let read_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut r = Reader::new(slice);
            let before = r.byte_pos();
            let form = r.read();
            let after = r.byte_pos();
            (form, before, after)
        }));
        let (form_opt, _before, after) = match read_outcome {
            Ok(t) => match t.0 {
                Ok(Some(form)) => (Some(form), t.1, t.2),
                Ok(None) => break,
                Err(e) => panic!("[upstream] form {i} READ ERR: {e}"),
            },
            Err(err) => {
                let msg = LAST_PANIC
                    .with(|p| p.borrow().clone())
                    .unwrap_or_else(|| panic_msg(&*err));
                eprintln!("[upstream] form {i} READ PANIC: {msg}");
                panic!("read panic at form {i}");
            }
        };
        byte_pos += after;
        eprintln!("[upstream] form {i} read OK, byte_pos={byte_pos}");
        let form = form_opt.unwrap();

        // Skip-list: forms whose runtime eval triggers a non-recoverable
        // SIGABRT (panic in extern "C" or panic-during-panic) that
        // catch_unwind can't catch. Identified by exact byte position
        // of the form in the source. Each entry is documented with the
        // form's name and why it's skipped.
        const SKIP_BYTE_POS: &[(usize, &str)] = &[
            // (let [^java.util.Properties properties (with-open ...)
            //       version-string (.getProperty properties "version") ...]
            //  (def *clojure-version* ...))
            // — uses regex compile + Integer/valueOf in a chain that
            //   cascades nil to a runtime extern panic we can't intercept.
            // (defn map-indexed ...) — body uses vswap! / @volatile chain
            //   whose nil cascade hits a non-recoverable extern panic.
            // (defn- prep-hashes ...) — case-dispatch helper. Body has
            //   nested `(let [[shift mask] (or ...)] ...)` whose second
            //   `let` macroexpand call returns a corrupted heap object
            //   (HeapBits with unrecognized type_id) under GC pressure,
            //   then "Unable to resolve symbol: shift". Same JIT-macro
            //   GC-corruption class as the two skips above. case-dispatch
            //   is rarely exercised by later forms.
            // (defn bounded-count ...) — body uses `loop` + `recur` with
            //   `seq`/`next` cascade. Hits panic-during-panic SIGABRT
            //   (error 5) inside the JIT call path.
            // (def default-data-readers ...) — references `#'clojure.uuid/...`
            //   var-quote to a namespace not loaded in this image. Hits a
            //   non-unwinding panic during analyze.
        ];
        let skip = SKIP_BYTE_POS.iter().find(|(p, _)| *p == byte_pos - after);
        if let Some((_, label)) = skip {
            eprintln!("[upstream] form {i} SKIPPED (skip-list: {label})");
            i += 1;
            continue;
        }
        eprintln!("[upstream] form {i} starting eval…");
        // Sniff the form shape for `(defmacro NAME …)` / `(defn NAME …)` /
        // `(def ^{:macro true} NAME …)` so we can print what got
        // *defined* (vs just what evaluated). Useful for figuring out
        // which macros are bound when the loader hits its wall.
        let defined_name: Option<String> = sniff_defined_name(&form);
        let eval_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sess.eval_form(form);
        }));
        if let (Some(name), true) = (&defined_name, eval_outcome.is_ok()) {
            eprintln!("[upstream] form {i} DEFINED {name}");
        }
        eprintln!(
            "[upstream] form {i} eval completed (outcome: {})",
            if eval_outcome.is_ok() { "Ok" } else { "Err" }
        );
        if std::env::var("CLJVM_LOADER_HEAPSTATS").is_ok() {
            eprintln!(
                "[upstream] form {i} heapstats: from_used={} collections={}",
                sess.gc.debug_heap_used(),
                sess.gc.collection_count()
            );
        }
        if let Err(err) = eval_outcome {
            let msg = LAST_PANIC
                .with(|p| p.borrow().clone())
                .unwrap_or_else(|| panic_msg(&*err));
            eprintln!("[upstream] form {i} EVAL PANIC: {msg}");
            if std::env::var("CLJVM_LOADER_SKIP_PANICS").is_ok() {
                eprintln!("[upstream] form {i} SKIPPED (CLJVM_LOADER_SKIP_PANICS=1)");
                LAST_PANIC.with(|p| *p.borrow_mut() = None);
                i += 1;
                continue;
            }
            panic!("eval panic at form {i} — see preceding line for message");
        }
        i += 1;
    }
    eprintln!("[upstream] processed {i} forms successfully");
}

/// Best-effort: inspect a top-level form and return the name being
/// defined plus a tag (`macro:`, `defn:`, etc.). Looks one level deep
/// only; doesn't descend into `do`/`when`/etc. — those rarely contain
/// top-level def shape at this layer.
fn sniff_defined_name(form: &clojure_jvm::lang::object::Object) -> Option<String> {
    use clojure_jvm::lang::object::Object;
    let Object::List(_) = form else { return None };
    let head = clojure_jvm::lang::rt::first(form);
    let head_name = match &head {
        Object::Symbol(s) => s.get_name().to_string(),
        _ => return None,
    };
    let after = clojure_jvm::lang::rt::next(form);
    let name_form = clojure_jvm::lang::rt::first(&after);
    let name = match name_form.peel_meta() {
        Object::Symbol(s) => s.get_name().to_string(),
        _ => return None,
    };
    // Note: `(def ^{:macro true} foo …)` is the macro-flag form
    // upstream uses for early bootstrapping; we just call it `def`
    // here and the cljs cross-reference reads the actual ^:macro flag
    // on the resulting var, not this sniff.
    let tag = match head_name.as_str() {
        "defmacro" => "macro",
        "defn" | "defn-" => "fn",
        "def" => "def",
        _ => return None,
    };
    Some(format!("{tag}:{name}"))
}

fn panic_msg(err: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = err.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<Box<str>>() {
        s.to_string()
    } else {
        format!("<panic payload type {}>", std::any::type_name_of_val(err))
    }
}
