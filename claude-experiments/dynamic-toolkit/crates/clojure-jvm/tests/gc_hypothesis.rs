//! Confirm the nested-syntax-quote crash is a GC-rooting (unrooted argument
//! across an allocating sub-call) bug. The "passing" direct-splice variant
//! should pass under default GC but crash under CLJVM_GC=every if the bug is
//! GC-timing-dependent.

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
use clojure_jvm::lang::object::Object;

fn is_top_level_load(form: &Object) -> bool {
    if let Object::List(l) = form {
        if let Some(Object::Symbol(s)) = l.iter().next() {
            return s.get_name() == "load";
        }
    }
    false
}

fn load_core_prefix() -> Session {
    const CORE: &str = include_str!("../clojure/core.clj");
    let mut sess = Session::new();
    let mut byte_pos = 0usize;
    loop {
        let slice = &CORE[byte_pos..];
        let mut r = Reader::new(slice);
        let before = r.byte_pos();
        let read = r.read();
        let after = r.byte_pos();
        byte_pos += after - before;
        match read {
            Ok(Some(form)) => {
                if is_top_level_load(&form) { break; }
                sess.eval_form(form);
            }
            Ok(None) => break,
            Err(e) => panic!("read err: {e}"),
        }
    }
    sess
}

#[test]
#[ignore = "regalloc pressure: many live values across nested calls"]
fn regalloc_pressure() {
    let mut sess = load_core_prefix();
    // A variadic call whose args are each allocating calls. Every earlier
    // arg's value must stay live across the later arg calls + the cons-fold.
    // If a value is clobbered across a call, the result list is wrong.
    eprintln!("(count (list 8 alloc-args))");
    let bits = sess.eval_str(
        "(count (list (+ 1 1) (+ 2 2) (+ 3 3) (+ 4 4) \
                      (+ 5 5) (+ 6 6) (+ 7 7) (+ 8 8)))",
    );
    let n = clojure_jvm::runtime::arg_to_i64(bits);
    eprintln!("count = {n} (want 8)");
    // And verify the actual values round-trip (sum should be 2+4+6+...+16=72).
    let sb = sess.eval_str(
        "(reduce1 + (list (+ 1 1) (+ 2 2) (+ 3 3) (+ 4 4) \
                          (+ 5 5) (+ 6 6) (+ 7 7) (+ 8 8)))",
    );
    eprintln!("sum = {} (want 72)", clojure_jvm::runtime::arg_to_i64(sb));
}

#[test]
#[ignore = "GC hypothesis: passing variant under stress GC"]
fn passing_variant_under_stress() {
    let mut sess = load_core_prefix();
    // Use a DEFINED head (`list`) so correct expansion is observable:
    // `(do (list (list 1)) (list 2))` → returns (list 2)'s value, no crash.
    // Drift the literal pool first with many syntax-quote-heavy defmacros,
    // then define+call mc. If the bug is literal-pool/gc_literal drift (not
    // GC), mc crashes (sqConcat 0xfc) regardless of GC policy.
    for i in 0..40 {
        sess.eval_str(&format!(
            "(defmacro drift{i} [a b] `(do (vector ~a ~b) (list ~a ~b) [~a ~b]))"
        ));
    }
    sess.eval_str("(defmacro mc [] `(do (list ~@(when true `((list 1)))) (list 2)))");
    eprintln!("CALLING mc (CLJVM_GC={:?})", std::env::var("CLJVM_GC").ok());
    let bits = sess.eval_str("(first (mc))");
    eprintln!("mc first => 0x{bits:016x} (want boxed 1)");
}

#[test]
#[ignore = "find regalloc arg-count threshold"]
fn regalloc_threshold() {
    let mut sess = load_core_prefix();
    for n in [2usize, 3, 4, 5, 6, 7, 8] {
        let args: String = (1..=n).map(|i| format!("(+ {i} {i})")).collect::<Vec<_>>().join(" ");
        let want: i64 = (1..=n as i64).map(|i| 2 * i).sum();
        let src = format!("(reduce1 + (list {args}))");
        let got = clojure_jvm::runtime::arg_to_i64(sess.eval_str(&src));
        eprintln!("n={n}: got={got} want={want} {}", if got == want { "ok" } else { "CORRUPT" });
    }
}

#[test]
#[ignore = "dump 8-list elements to find corrupted position"]
fn dump_list_elements() {
    let mut sess = load_core_prefix();
    let list = "(list (+ 1 1) (+ 2 2) (+ 3 3) (+ 4 4) (+ 5 5) (+ 6 6) (+ 7 7) (+ 8 8))";
    for i in 0..8 {
        // (first (nth-rest ...))  via nested rest
        let rests = "(rest ".repeat(i);
        let close = ")".repeat(i);
        let src = format!("(first {rests}{list}{close})");
        let got = clojure_jvm::runtime::arg_to_i64(sess.eval_str(&src));
        let want = 2 * (i as i64 + 1);
        eprintln!("elem[{i}] = {got} (want {want}) {}", if got == want { "ok" } else { "WRONG" });
    }
}

#[test]
#[ignore = "dump IR for the failing list form"]
fn dump_list_ir() {
    let mut sess = load_core_prefix();
    // Single form that builds the 8-element list (the corrupting one).
    let _ = sess.eval_str("(count (list (+ 1 1) (+ 2 2) (+ 3 3) (+ 4 4) (+ 5 5) (+ 6 6) (+ 7 7) (+ 8 8)))");
}

#[test]
#[ignore = "dump mmin macro fn IR"]
fn dump_mmin_ir() {
    let mut sess = load_core_prefix();
    // Define some drift macros first (mirrors replicate_ns state).
    for i in 0..3 {
        sess.eval_str(&format!("(defmacro d{i} [a] `(do (list ~a) (vector ~a) [~a]))"));
    }
    // The failing macro.
    sess.eval_str("(defmacro mmin [] `(do (foo ~@(when true `((list 1)))) (list 2)))");
}

#[test]
#[ignore = "verify call_with_packed 9-element path (variadic 8-fixed dynamic invoke)"]
fn variadic_eight_fixed_dynamic_invoke() {
    let mut sess = load_core_prefix();
    // 8 fixed params + variadic. Bound to a var and called dynamically with
    // 8 args → invoke_8 → pack_variadic_args → 9-element call_with_packed.
    sess.eval_str("(def vf (fn [a b c d e f g h & r] (+ a (+ b (+ c (+ d (+ e (+ f (+ g h)))))))))");
    let bits = sess.eval_str("(vf 1 2 3 4 5 6 7 8)");
    let got = clojure_jvm::runtime::arg_to_i64(bits);
    eprintln!("vf sum = {got} (want 36)");
    assert_eq!(got, 36);
}

#[test]
#[ignore = "test with-loading-context + map syntax-quote"]
fn test_wlc_and_map_sq() {
    let mut sess = load_core_prefix();
    // Map in syntax-quote should become (hash-map ...) with quoted keys.
    sess.eval_str("(defmacro mm [] `(identity {:a 1 :b 2}))");
    eprintln!("mm count:");
    let bits = sess.eval_str("(count (mm))");
    eprintln!("(count (mm)) = {} (want 2)", clojure_jvm::runtime::arg_to_i64(bits));
    eprintln!("calling with-loading-context:");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sess.eval_str("(with-loading-context (+ 1 2))")));
    eprintln!("wlc result: {:?}", r.map(|b| clojure_jvm::runtime::arg_to_i64(b)));
}

#[test]
#[ignore = "bisect wlc map"]
fn bisect_wlc_map() {
    let mut sess = load_core_prefix();
    for (name, src) in [
        ("m_symkey", "(defmacro m_symkey [] `(identity {clojure.lang.Compiler/LOADER 1}))"),
        ("m_methodval", "(defmacro m_methodval [] `(identity {:k (.getClassLoader (.getClass ^Object x#))}))"),
        ("m_gensymkey", "(defmacro m_gensymkey [] `(fn foo# [] {:k foo#}))"),
    ] {
        sess.eval_str(src);
        let call = format!("({})", name.trim_start_matches("m_").to_string());
        // just expand+analyze by calling; catch panic
        let mname = src.split_whitespace().nth(1).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sess.eval_str(&format!("({mname})"))));
        eprintln!("{mname}: {}", if r.is_ok() { "OK" } else { "CRASH" });
        let _ = call;
    }
}

#[test]
#[ignore = "direct hash-map with quoted symbol key"]
fn direct_hashmap_symkey() {
    let mut sess = load_core_prefix();
    for src in [
        "(count (hash-map :a 1))",
        "(count (hash-map (quote clojure.lang.Compiler/LOADER) 1))",
        "(count (hash-map clojure.lang.Compiler/LOADER 1))",
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sess.eval_str(src)));
        eprintln!("{src} => {}", match r { Ok(b) => format!("{}", clojure_jvm::runtime::arg_to_i64(b)), Err(_) => "CRASH".into() });
    }
}

#[test]
#[ignore = "exact wlc body"]
fn exact_wlc_body() {
    let mut sess = load_core_prefix();
    eprintln!("defining wlc2 (exact body)");
    sess.eval_str(r#"(defmacro wlc2 [& body]
  `((fn loading# []
        (. clojure.lang.Var (pushThreadBindings {clojure.lang.Compiler/LOADER
                                                 (.getClassLoader (.getClass ^Object loading#))}))
        (try
         ~@body
         (finally
          (. clojure.lang.Var (popThreadBindings)))))))"#);
    eprintln!("calling wlc2");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sess.eval_str("(wlc2 (+ 1 2))")));
    eprintln!("wlc2 => {}", if r.is_ok() { "OK" } else { "CRASH" });
}

#[test]
#[ignore = "diagnose reduce pieces"]
fn diagnose_reduce() {
    let mut sess = load_core_prefix();
    sess.eval_str("(clojure.lang.RT/load \"core/protocols\")");
    let report = |s: &mut Session, src: &str| {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.eval_str(src)));
        match r {
            Ok(b) => eprintln!("  {src} => {} (bits 0x{b:016x})", clojure_jvm::runtime::arg_to_i64(b)),
            Err(_) => eprintln!("  {src} => PANIC"),
        }
    };
    report(&mut sess, "(clojure.core.protocols/internal-reduce (list 2 3 4) + 1)");
    report(&mut sess, "(clojure.core.protocols/coll-reduce (list 1 2 3 4) + 100)");
    report(&mut sess, "(clojure.core.protocols/coll-reduce (list 1 2 3 4) +)");
    report(&mut sess, "(clojure.core.protocols/coll-reduce nil + 42)");
    report(&mut sess, "(satisfies? clojure.core.protocols/CollReduce (list 1))");
}

#[test]
#[ignore = "probe protocol registration post-load"]
fn probe_protocol_registration() {
    let mut sess = load_core_prefix();
    sess.eval_str("(clojure.lang.RT/load \"core/protocols\")");
    let report = |s: &mut Session, label: &str, src: &str| {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.eval_str(src)));
        match r {
            Ok(b) => eprintln!("  {label}: {src} => 0x{b:016x} (int {})", clojure_jvm::runtime::arg_to_i64(b)),
            Err(_) => eprintln!("  {label}: {src} => PANIC"),
        }
    };
    report(&mut sess, "bound?", "(if clojure.core.protocols/coll-reduce 1 0)");
    report(&mut sess, "sat-nil", "(satisfies? clojure.core.protocols/CollReduce nil)");
    report(&mut sess, "sat-obj", "(satisfies? clojure.core.protocols/CollReduce (list 1))");
    // Define a fresh multi-arity protocol + extend in current ns as a control.
    sess.eval_str("(defprotocol PCtl (pc [c f] [c f v]))");
    sess.eval_str("(extend-protocol PCtl nil (pc ([c f] :two) ([c f v] v)) java.lang.Object (pc ([c f] :obj2) ([c f v] :obj3)))");
    report(&mut sess, "ctl-nil3", "(pc nil + 77)");
    report(&mut sess, "ctl-obj2", "(pc (list 1) +)");
}

#[test]
#[ignore = "probe coll-reduce binding inside its ns"]
fn probe_coll_reduce_in_ns() {
    let mut sess = load_core_prefix();
    sess.eval_str("(clojure.lang.RT/load \"core/protocols\")");
    let report = |s: &mut Session, label: &str, src: &str| {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.eval_str(src)));
        eprintln!("  {label}: {}", match r { Ok(b) => format!("0x{b:016x} int {}", clojure_jvm::runtime::arg_to_i64(b)), Err(_) => "PANIC".into() });
    };
    // Switch into the protocols ns and check the unqualified var.
    sess.eval_str("(in-ns 'clojure.core.protocols)");
    report(&mut sess, "in-ns bound?", "(if coll-reduce 1 0)");
    report(&mut sess, "in-ns nil-reduce", "(coll-reduce nil (fn [& a] 0) 42)");
}

#[test]
#[ignore = "qualified in-ns switch"]
fn qualified_in_ns() {
    let mut sess = load_core_prefix();
    let report = |s: &mut Session, label: &str, src: &str| {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.eval_str(src)));
        eprintln!("  {label}: {}", match r { Ok(b) => format!("0x{b:016x}"), Err(_) => "PANIC".into() });
    };
    // unqualified in-ns then define + read back in that ns
    report(&mut sess, "unqual in-ns", "(in-ns 'aa.bb)");
    report(&mut sess, "def in aa.bb", "(def marker-a 1)");
    report(&mut sess, "read aa.bb/marker-a", "aa.bb/marker-a");
    sess.eval_str("(in-ns 'clojure.core)");
    // qualified in-ns (as the ns macro generates)
    report(&mut sess, "qual in-ns", "(clojure.core/in-ns 'cc.dd)");
    report(&mut sess, "def in cc.dd?", "(def marker-c 2)");
    report(&mut sess, "read cc.dd/marker-c", "cc.dd/marker-c");
}
