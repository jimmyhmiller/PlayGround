//! Minimal reproducer for the pool-slot corruption observed at form 347
//! (`(def system-newline (System/getProperty ...))`) of forked core.clj.
//!
//! Hypothesis: a `(def NAME (throw "..."))` form somehow corrupts
//! earlier-interned literal-pool slots (specifically the Symbol cells
//! for `let*` and `loop*` interned at session init).

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::read_str;

/// Assert that every populated pool slot decodes as a valid NanBox
/// (either non-pointer or an 8-byte-aligned pointer payload). A
/// misaligned ptr payload means corruption.
fn check_pool_aligned(sess: &Session, label: &str) {
    for i in 0..sess.literal_pool_len() {
        let bits = sess.literal_pool_get(i);
        let high = bits & 0xFFFF_0000_0000_0000;
        let payload = bits & 0x0000_FFFF_FFFF_FFFF;
        if high == 0x7FFE_0000_0000_0000 && payload != 0 && (payload & 0x7) != 0 {
            panic!(
                "{label}: pool[{i}] has misaligned ptr payload \
                 bits=0x{bits:016x} payload=0x{payload:016x}"
            );
        }
    }
}

#[test]
fn def_with_throwing_init_does_not_corrupt_pool_slots() {
    let mut sess = Session::new();
    check_pool_aligned(&sess, "after init");
    // Drive several throwing defs to expose the corruption.
    for i in 0..20 {
        let src = format!("(def x{i} (throw \"boom {i}\"))");
        let form = read_str(&src).expect("read");
        sess.eval_form(form);
        check_pool_aligned(&sess, &format!("after def x{i}"));
    }
}

#[test]
fn top_level_throw_does_not_corrupt_pool() {
    let mut sess = Session::new();
    check_pool_aligned(&sess, "init");
    eprintln!("[ranges before] {}", sess.dump_memory_ranges());
    eprintln!("[pool before] len={} [0]=0x{:016x} [1]=0x{:016x} [2]=0x{:016x} [3]=0x{:016x}",
        sess.literal_pool_len(),
        sess.literal_pool_get(0), sess.literal_pool_get(1),
        sess.literal_pool_get(2), sess.literal_pool_get(3));
    let form = read_str("(throw \"boom\")").expect("read");
    sess.eval_form(form);
    eprintln!("[ranges after]  {}", sess.dump_memory_ranges());
    eprintln!("[pool after]  len={} [0]=0x{:016x} [1]=0x{:016x} [2]=0x{:016x} [3]=0x{:016x}",
        sess.literal_pool_len(),
        sess.literal_pool_get(0), sess.literal_pool_get(1),
        sess.literal_pool_get(2), sess.literal_pool_get(3));
    check_pool_aligned(&sess, "after top-level throw");
}

#[test]
fn throw_nil_corrupts_pool() {
    let mut sess = Session::new();
    let form = read_str("(throw nil)").expect("read");
    sess.eval_form(form);
    check_pool_aligned(&sess, "after (throw nil)");
}

#[test]
fn throw_42_corrupts_pool() {
    let mut sess = Session::new();
    let form = read_str("(throw 42)").expect("read");
    sess.eval_form(form);
    check_pool_aligned(&sess, "after (throw 42)");
}

#[test]
fn top_level_throw_repeated_does_not_corrupt_pool() {
    let mut sess = Session::new();
    check_pool_aligned(&sess, "init");
    for i in 0..50 {
        let form = read_str("(throw \"boom\")").expect("read");
        sess.eval_form(form);
        check_pool_aligned(&sess, &format!("after iter {i}"));
    }
}

#[test]
fn def_with_nil_init_does_not_corrupt_pool() {
    let mut sess = Session::new();
    check_pool_aligned(&sess, "init");
    let form = read_str("(def z0 nil)").expect("read");
    sess.eval_form(form);
    check_pool_aligned(&sess, "after (def z0 nil)");
}

#[test]
fn def_with_unregistered_static_method_does_not_corrupt_pool_slots() {
    let mut sess = Session::new();
    check_pool_aligned(&sess, "after init");
    // This mirrors the form 347 case exactly: a `(def NAME body)`
    // whose body is an unregistered static method call (now rewritten
    // to throw at runtime).
    for i in 0..5 {
        let src = format!(
            "(def y{i} (System/getProperty \"some.prop\"))"
        );
        let form = read_str(&src).expect("read");
        sess.eval_form(form);
        check_pool_aligned(&sess, &format!("after def y{i}"));
    }
}
