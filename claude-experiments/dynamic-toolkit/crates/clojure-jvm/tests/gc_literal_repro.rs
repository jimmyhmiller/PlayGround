//! Minimal reproducer for the gc_literal corruption that blocks form
//! 202 in `load_upstream_core`. The hypothesis: a value loaded by
//! `gc_literal(idx)` early in a JIT-compiled fn is held in a register
//! across an intervening fn call; the GC moves the underlying heap
//! cell and updates the literal pool slot in place, but the register
//! isn't refreshed, so subsequent uses see stale bits that decode to
//! a different symbol after the slot's old memory got reused.
//!
//! This test attempts to reproduce that pattern with the smallest
//! possible Clojure-shaped fn and many iterations to surface the race.

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::read_str;

#[test]
#[ignore = "diagnostic — drives the gc_literal corruption seen at form 202"]
fn many_quoted_calls_under_gc_pressure() {
    let mut sess = Session::new();

    // Bind cons to its host implementation, mirroring the def at line 29
    // of upstream core.clj. Done before make-pair so make-pair's body's
    // `(cons ...)` resolves to a real fn handle.
    let cons_def = read_str(
        "(def cons (fn* ^:static cons [x seq] (. clojure.lang.RT (cons x seq))))",
    )
    .expect("read cons def");
    sess.eval_form(cons_def);

    // Define a fn that loads two quoted symbols and threads them
    // through `cons` to build (a b). The outer `cons` call sits
    // between the two `gc_literal` loads — perfect spot for the
    // bug to surface if it relates to register state across calls.
    let setup = read_str(
        "(def make-pair (fn* [] (cons (quote a) (cons (quote b) nil))))",
    )
    .expect("read setup");
    sess.eval_form(setup);

    // Drive the fn many times. Each invocation triggers EveryPoint
    // GC. If the bug fires, one of these calls returns a cons whose
    // head bits decode to something other than the symbol `a`.
    for i in 0..200 {
        let invoke = read_str("(make-pair)").expect("read invoke");
        let bits = sess.eval_form(invoke);
        // Decode the result back to inspect. If the head is wrong, we
        // want to know at WHICH iteration the corruption first hits.
        let head_ok = check_head_is_symbol(bits, "a");
        assert!(
            head_ok,
            "iteration {i}: head bits 0x{bits:x} did not decode to symbol `a`"
        );
    }
}

/// Nested macroexpansion: macro A's body uses macro B. When A is
/// expanded at a call site, the analysis of A's expansion result
/// triggers B's expansion. This nests two `run_jit` calls inside
/// one `eval_form`, which is the shape that surfaces the bug at
/// form 57 (`defmacro cond` whose body uses `when`).
///
/// Requires `CLJVM_HOST_DEFN=1` in env to define macros without
/// upstream defmacro running.
#[test]
#[ignore = "diagnostic — nested macroexpand reproducer"]
fn nested_macroexpand_under_gc_pressure() {
    if std::env::var("CLJVM_HOST_DEFN").is_err() {
        unsafe { std::env::set_var("CLJVM_HOST_DEFN", "1") };
    }
    let mut sess = Session::new();
    for stmt in [
        "(def cons (fn* ^:static cons [x seq] (. clojure.lang.RT (cons x seq))))",
        "(def list (. clojure.lang.PersistentList creator))",
    ] {
        eprintln!("[setup] {stmt}");
        let f = read_str(stmt).unwrap_or_else(|e| panic!("setup `{stmt}`: {e}"));
        sess.eval_form(f);
    }
    // when: returns (if test (do . body)). Define via host-defn defmacro path.
    eprintln!("[setup] defmacro when");
    let when_def = read_str(
        "(defmacro when [test & body] (list (quote if) test (cons (quote do) body)))",
    )
    .expect("read when");
    sess.eval_form(when_def);
    eprintln!("[setup] when defined");

    // Now define a macro that USES `when` in its body (like cond does).
    // Each call to outer-mac triggers when's macroexpansion as part of
    // analyzing outer-mac's result.
    eprintln!("[setup] defmacro outer-mac");
    let outer = read_str(
        "(defmacro outer-mac [x] (when x (list (quote quote) x)))",
    )
    .expect("read outer");
    sess.eval_form(outer);
    eprintln!("[setup] outer-mac defined");

    let invoke = read_str("(outer-mac 42)").expect("read invoke");
    let _ = sess.eval_form(invoke);
}

/// Invoke `when` indirectly via a form-analysis that has a (when …)
/// embedded. This is the simpler path — no nested defmacro needed.
#[test]
#[ignore = "diagnostic — single-level macroexpand of when"]
fn macroexpand_when_corruption() {
    if std::env::var("CLJVM_HOST_DEFN").is_err() {
        unsafe { std::env::set_var("CLJVM_HOST_DEFN", "1") };
    }
    let mut sess = Session::new();
    for stmt in [
        "(def cons (fn* ^:static cons [x seq] (. clojure.lang.RT (cons x seq))))",
        "(def list (. clojure.lang.PersistentList creator))",
        "(defmacro when [test & body] (list (quote if) test (cons (quote do) body)))",
    ] {
        let f = read_str(stmt).unwrap_or_else(|e| panic!("setup `{stmt}`: {e}"));
        sess.eval_form(f);
    }
    // Now use when in a form. The analyze of this form macroexpands
    // when. when's body is a JIT-compiled fn invoked via run_jit.
    let usage = read_str("(when true 1)").expect("read usage");
    let _ = sess.eval_form(usage);
}

/// Mimic `when`'s upstream body exactly:
/// `(list 'if test (cons 'do body))`. This is the body that, when
/// invoked many times during macroexpansion of cond at form 57, was
/// observed to return `(when ...)` instead of `(if ...)`.
#[test]
#[ignore = "diagnostic — exact repro of when's macro body shape"]
fn when_macro_body_under_gc_pressure() {
    let mut sess = Session::new();
    for stmt in [
        "(def cons (fn* ^:static cons [x seq] (. clojure.lang.RT (cons x seq))))",
        "(def list (. clojure.lang.PersistentList creator))",
    ] {
        let f = read_str(stmt).unwrap_or_else(|e| panic!("setup `{stmt}`: {e}"));
        sess.eval_form(f);
    }

    // when's body literally: (list 'if test (cons 'do body))
    let setup = read_str(
        "(def when-body (fn* [test body] (list (quote if) test (cons (quote do) body))))",
    )
    .expect("read when-body");
    sess.eval_form(setup);

    // Drive many invocations. test arg is a constant, body is nil so cons
    // returns (do). Result should be (if test (do)).
    for i in 0..1000 {
        let invoke = read_str("(when-body 42 nil)").expect("read invoke");
        let bits = sess.eval_form(invoke);
        let head_ok = check_head_is_symbol(bits, "if");
        if !head_ok {
            panic!(
                "iteration {i}: when-body return head bits 0x{bits:x} is \
                 not symbol `if` — corruption reproduced!"
            );
        }
    }
}

/// More complex reproducer that mimics defn's body shape: builds a
/// list of multiple symbols + a nested fn-form, with many calls
/// (cons / list / with-meta) interleaved. Closer to what defn
/// actually does and more likely to surface the corruption.
#[test]
#[ignore = "diagnostic — closer mimic of upstream defn body shape"]
fn defn_body_shape_under_gc_pressure() {
    let mut sess = Session::new();
    for stmt in [
        "(def cons (fn* ^:static cons [x seq] (. clojure.lang.RT (cons x seq))))",
        "(def list (. clojure.lang.PersistentList creator))",
        "(def first (fn* ^:static first [coll] (. clojure.lang.RT (first coll))))",
        "(def next (fn* ^:static next [x] (. clojure.lang.RT (next x))))",
    ] {
        let f = read_str(stmt).unwrap_or_else(|e| panic!("setup `{stmt}`: {e}"));
        sess.eval_form(f);
    }

    // Define a fn whose body uses MANY quoted-symbol literals
    // separated by calls. Mirrors the shape of defn / defmacro
    // expansion logic in upstream core.clj.
    let setup = read_str(
        "(def make-form
           (fn* []
             (cons (quote def)
               (cons (quote my-name)
                 (cons (cons (quote fn)
                         (cons (quote my-name)
                           (cons (cons (quote x) nil)
                             (cons (cons (quote +) (cons (quote x) (cons (quote 1) nil)))
                               nil))))
                   nil)))))",
    )
    .expect("read make-form");
    sess.eval_form(setup);

    // The expected return is `(def my-name (fn my-name (x) (+ x 1)))`.
    // Drive many iterations under GC pressure to surface stale-pointer
    // corruption if it exists. Check head is `def`, second is `my-name`.
    for i in 0..500 {
        let invoke = read_str("(make-form)").expect("read invoke");
        let bits = sess.eval_form(invoke);
        let head_ok = check_head_is_symbol(bits, "def");
        if !head_ok {
            panic!(
                "iteration {i}: head bits 0x{bits:x} did not decode to \
                 symbol `def` — corruption reproduced!"
            );
        }
    }
}

/// Inspect the JIT-returned NanBox. We expect a Cons whose `first`
/// field (offset 8) holds a NanBox-PTR pointing at a Symbol cell
/// whose Arc<Symbol> name matches `expected`.
fn check_head_is_symbol(bits: u64, expected: &str) -> bool {
    use clojure_jvm::runtime::{nanbox_payload, nanbox_tag};
    const TAG_PTR: u32 = 2;
    if !matches!(nanbox_tag(bits), Some(TAG_PTR)) {
        eprintln!(
            "  not TAG_PTR (bits 0x{bits:x}, tag {:?})",
            nanbox_tag(bits)
        );
        return false;
    }
    let p = nanbox_payload(bits) as *const u8;
    if p.is_null() {
        eprintln!("  null payload");
        return false;
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    // Cons type_id from `Session::new`'s ObjType registration. We
    // don't need to look it up — we just check that the head field
    // (offset 8) holds a TAG_PTR to a Symbol whose name matches.
    let head_bits = unsafe { p.add(8).cast::<u64>().read_unaligned() };
    if !matches!(nanbox_tag(head_bits), Some(TAG_PTR)) {
        eprintln!(
            "  head field is not TAG_PTR (bits 0x{head_bits:x}); cons tid={tid}"
        );
        return false;
    }
    let head_p = nanbox_payload(head_bits) as *const u8;
    if head_p.is_null() {
        eprintln!("  head field has null payload");
        return false;
    }
    let head_tid = unsafe { head_p.cast::<u16>().read_unaligned() } as usize;
    // Sanity: head_tid should equal symbol type_id (small int < 100).
    if head_tid > 100 {
        eprintln!(
            "  head field type_id={head_tid} (out of range — corruption!)"
        );
        return false;
    }
    // Read the Arc<Symbol> pointer at offset 8 and pull the name.
    let arc_ptr = unsafe { head_p.add(8).cast::<u64>().read_unaligned() }
        as *const clojure_jvm::lang::symbol::Symbol;
    if arc_ptr.is_null() {
        eprintln!("  symbol arc pointer null");
        return false;
    }
    unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
    let s = unsafe { std::sync::Arc::from_raw(arc_ptr) };
    if s.get_name() != expected {
        eprintln!(
            "  symbol name {:?} != expected {expected}",
            s.get_name()
        );
        return false;
    }
    true
}
