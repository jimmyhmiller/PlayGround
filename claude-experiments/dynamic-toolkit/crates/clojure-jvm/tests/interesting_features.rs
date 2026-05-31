//! Empirically probe what high-value clojure.core features actually work:
//! threading macros (`->`, `->>`), `map`, `filter`, `reduce1`, etc.
//!
//! Loads our forked core.clj form-by-form, stopping at the first top-level
//! `(load "…")` (line ~6684 — the sub-file protocol section we don't load
//! yet), which leaves a Session with all the early defs available. Then it
//! evals each probe expression and prints the raw result bits so we can see
//! ground truth without guessing.

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
use clojure_jvm::lang::object::Object;

/// Is this top-level form `(load "…")`? That's our stop boundary.
fn is_top_level_load(form: &Object) -> bool {
    if let Object::List(l) = form {
        if let Some(Object::Symbol(s)) = l.iter().next() {
            return s.get_name() == "load";
        }
    }
    false
}

/// Load core up to (not including) the first top-level `(load …)`.
fn load_core_prefix() -> Session {
    const CORE: &str = include_str!("../clojure/core.clj");
    let mut sess = Session::new();
    let mut byte_pos = 0usize;
    let mut n = 0usize;
    loop {
        let slice = &CORE[byte_pos..];
        let mut r = Reader::new(slice);
        let before = r.byte_pos();
        let read = r.read();
        let after = r.byte_pos();
        byte_pos += after - before;
        match read {
            Ok(Some(form)) => {
                if is_top_level_load(&form) {
                    eprintln!("[prefix] stopping at first (load …) after {n} forms");
                    break;
                }
                sess.eval_form(form);
                n += 1;
            }
            Ok(None) => break,
            Err(e) => panic!("read error at form {n} (byte {byte_pos}): {e}"),
        }
    }
    sess
}

/// Eval `src`, decode the result as an integer, and assert it equals
/// `expect`. Integers are boxed Longs; `arg_to_i64` reads either a boxed
/// Long or a native NanBox double.
fn check_int(sess: &mut Session, src: &str, expect: i64) {
    let bits = sess.eval_str(src);
    let got = clojure_jvm::runtime::arg_to_i64(bits);
    let ok = if got == expect { "ok" } else { "MISMATCH" };
    eprintln!("  [{ok}] {src:<46} => {got} (want {expect})  bits=0x{bits:016x}");
    assert_eq!(got, expect, "{src}");
}

/// Eval `src`, catching panics, and just report whether it ran and the raw
/// bits. Used to probe which primitives are available without asserting.
fn probe(sess: &mut Session, src: &str) {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sess.eval_str(src)));
    match r {
        Ok(bits) => eprintln!("  [ran ] {src:<46} => bits=0x{bits:016x}"),
        Err(_) => eprintln!("  [FAIL] {src:<46} => panicked"),
    }
}

#[test]
#[ignore = "probe refer chain"]
fn probe_refer_chain() {
    // The `refer` chain works bottom-up: find-ns / the-ns / ns-map /
    // ns-publics all return 652 entries (after wiring the Namespace heap
    // representation, structural `=` via equiv, and seq-over-map). The
    // remaining blocker is a separate codegen bug: an instance-method call
    // WITH ARGS used as a `doseq`/loop body has its receiver emit None and
    // the call is dropped, so `refer`'s `(. *ns* (refer sym v))` never runs.
    // (An arity-1 instance method, e.g. `.getMappings`, in the same loop body
    // DOES run — verified.)
    let mut sess = load_core_prefix();
    let np = clojure_jvm::runtime::arg_to_i64(
        sess.eval_str("(count (ns-publics 'clojure.core))"),
    );
    eprintln!("ns-publics count = {np}");
    assert!(np > 0, "ns-publics non-empty (Namespace heap repr + structural =)");
}

#[test]
#[ignore = "minimal instance-method-with-args repro"]
fn probe_inst_args() {
    // Regression guard for the prelude mock-var shadowing bug: bare class
    // names (`String`, `Exception`, …) must resolve to the host class, not to
    // the `(def String "#<MOCKED…>")` mock var. Before the fix `(instance?
    // String "a")` was false (the mock string's cell header read as a
    // non-Class) and `(symbol "s")` crashed in `(new …)` (its arity-1 body
    // does `(instance? String name)` then `(IllegalArgumentException. …)`).
    let mut sess = load_core_prefix();
    let is_str = sess.eval_str("(instance? String \"a\")");
    assert_eq!(
        is_str, 0x7ffd_0000_0000_0001,
        "(instance? String \"a\") must be true (bare class resolves to host class)"
    );
    // Must not crash:
    let _ = sess.eval_str("(symbol \"a\")");
}

#[test]
#[ignore = "reentrant refer-chain isolation"]
fn probe_reentrant_refer() {
    // Isolation of the reentrant-load blocker behind `reduce`. The
    // "reentrant-refer-probe" resource is loaded via `RT/load` (nested
    // run_jit, exactly like protocols.clj). Findings (read each marker via a
    // direct VarExpr, NOT `(deref (var …))` which doesn't read the root):
    //   rr-lit   (literal def)             → 4242  ✓
    //   rr-fnval (var deref of a fn)        → fn handle ✓
    //   rr-call  ((inc 41), a fn call)      → nil   ✗  return value lost
    //   rr-findns((find-ns …), a fn call)   → nil   ✗  (callee runs: cljvm_ns_find
    //                                                   gets 'clojure.core and finds it,
    //                                                   but the DIRECT JIT→JIT call return
    //                                                   is nil in the nested run_jit)
    //   rr-done  (literal def)              → 1     ✓
    // So reentrant (nested run_jit) direct call RETURN values are lost
    // (deterministic, not GC-timing); literals + var derefs survive. This is
    // the executor bug blocking `refer` (hence `reduce`) under reentrant load.
    let mut sess = load_core_prefix();
    sess.eval_str("(clojure.lang.RT/load \"reentrant-refer-probe\")");
    let lit = sess.eval_str("clojure.core/rr-lit");
    let done = sess.eval_str("clojure.core/rr-done");
    eprintln!("  reentrant rr-lit=0x{lit:x} rr-done=0x{done:x}");
    // Literal reentrant defs must persist (the parts that DO work).
    assert_ne!(lit, 0x7ffc_0000_0000_0000, "reentrant literal def must persist");
}

#[test]
#[ignore = "full ns body replica"]
fn replicate_ns_full() {
    let mut sess = load_core_prefix();
    // Exact copy of core's ns macro body, renamed myns.
    sess.eval_str(
        r#"(defmacro myns [name & references]
  (let [process-reference
        (fn [[kname & args]]
          `(~(symbol "clojure.core" (clojure.core/name kname))
             ~@(map #(list 'quote %) args)))
        docstring  (when (string? (first references)) (first references))
        references (if docstring (next references) references)
        name (if docstring
               (vary-meta name assoc :doc docstring)
               name)
        metadata   (when (map? (first references)) (first references))
        references (if metadata (next references) references)
        name (if metadata
               (vary-meta name merge metadata)
               name)
        gen-class-clause (first (filter #(= :gen-class (first %)) references))
        gen-class-call
          (when gen-class-clause
            (list* `gen-class :name (.replace (str name) \- \_) :impl-ns name :main true (next gen-class-clause)))
        references (remove #(= :gen-class (first %)) references)
        name-metadata (meta name)]
    `(do
       (clojure.core/in-ns '~name)
       ~@(when name-metadata
           `((.resetMeta (clojure.lang.Namespace/find '~name) ~name-metadata)))
       (with-loading-context
        ~@(when gen-class-call (list gen-class-call))
        ~@(when (and (not= name 'clojure.core) (not-any? #(= :refer-clojure (first %)) references))
            `((clojure.core/refer '~'clojure.core)))
        ~@(map process-reference references))
        (if (.equals '~name 'clojure.core)
          nil
          (do (dosync (commute @#'*loaded-libs* conj '~name)) nil)))))"#,
    );
    let _ = "skip myns call (aborts process); test myns2 directly";

    // Stripped: no docstring/metadata/gen-class let bindings; keep the SQ body.
    sess.eval_str(
        r#"(defmacro myns2 [name & references]
    `(do
       (clojure.core/in-ns '~name)
       (with-loading-context
        ~@(when (and (not= name 'clojure.core) (not-any? #(= :refer-clojure (first %)) references))
            `((clojure.core/refer '~'clojure.core)))
        ~@(map (fn [r] r) references))
        (if (.equals '~name 'clojure.core)
          nil
          (do (dosync (commute @#'*loaded-libs* conj '~name)) nil))))"#,
    );
    let _ = &sess; // skip myns2 call (aborts)

    // myns3: condition = true (no `and`/`not-any?`), no map splice.
    sess.eval_str(
        r#"(defmacro myns3 [name & references]
    `(do
       (clojure.core/in-ns '~name)
       (with-loading-context
        ~@(when true `((clojure.core/refer '~'clojure.core))))
        (if true
          nil
          nil)))"#,
    );
    // (myns3 call skipped — aborts process)

    // Minimal repro hypothesis: splice of a nested syntax-quote, then a
    // following nested-list form.
    sess.eval_str("(defmacro mmin [] `(do (foo ~@(when true `((list 1)))) (list 2)))");
    eprintln!("CALLING mmin (splice nested one level + sibling)");
    probe(&mut sess, "(mmin)");
    eprintln!("DONE mmin");
}

#[test]
#[ignore = "replicate ns body and bisect"]
fn replicate_ns_body() {
    let mut sess = load_core_prefix();
    // Clause A: just in-ns + the if/dosync tail (no with-loading-context).
    eprintln!("A: in-ns + tail");
    sess.eval_str(
        "(defmacro myns-a [name] \
           `(do (clojure.core/in-ns '~name) \
                (if (.equals '~name 'clojure.core) nil \
                  (do (dosync (commute @#'*loaded-libs* conj '~name)) nil))))",
    );
    probe(&mut sess, "(macroexpand-1 '(myns-a foo.bar))");
    // Clause B: with-loading-context + refer.
    eprintln!("B: with-loading-context + refer");
    sess.eval_str(
        "(defmacro myns-b [name] \
           `(do (clojure.core/in-ns '~name) \
                (with-loading-context \
                  ~@(when true `((clojure.core/refer '~'clojure.core)))) \
                nil))",
    );
    probe(&mut sess, "(macroexpand-1 '(myns-b foo.bar))");
    // Clause C: the (map process-reference references) splice with empty refs.
    eprintln!("C: map splice empty");
    sess.eval_str(
        "(defmacro myns-c [name & references] \
           (let [pr (fn [[kname & args]] \
                      `(~(symbol \"clojure.core\" (clojure.core/name kname)) \
                         ~@(map (fn [a] (list 'quote a)) args)))] \
             `(do (clojure.core/in-ns '~name) ~@(map pr references) nil)))",
    );
    probe(&mut sess, "(macroexpand-1 '(myns-c foo.bar))");
}

#[test]
#[ignore = "probe syntax-quote splice patterns"]
fn probe_splice_patterns() {
    let mut sess = load_core_prefix();
    eprintln!("=== splice patterns ===");
    // splice of (when false ...) → nil
    sess.eval_str("(defmacro m1 [] `(do ~@(when false [1]) 2))");
    probe(&mut sess, "(m1)");
    // splice of (map f '()) → empty lazy seq
    sess.eval_str("(defmacro m2 [] `(do ~@(map (fn [x] x) '()) 2))");
    probe(&mut sess, "(m2)");
    // splice of (remove f '())
    sess.eval_str("(defmacro m3 [] `(do ~@(remove (fn [x] false) '()) 2))");
    probe(&mut sess, "(m3)");
    // nested syntax-quote splice like ns's refer clause
    sess.eval_str("(defmacro m4 [] `(do ~@(when true `((foo '~'bar))) 2))");
    probe(&mut sess, "(m4)");
    // the deref-var form @#'x as data in expansion
    sess.eval_str("(def x 5)");
    sess.eval_str("(defmacro m5 [] `(do (commute @#'x conj 1) 2))");
    probe(&mut sess, "(m5)");
}

#[test]
#[ignore = "probe ns macro sub-expressions"]
fn probe_ns_subexprs() {
    let mut sess = load_core_prefix();
    eprintln!("=== ns macro sub-expressions ===");
    probe(&mut sess, "(meta 'clojure.core.protocols)");
    probe(&mut sess, "(map? nil)");
    probe(&mut sess, "(string? nil)");
    probe(&mut sess, "(first (remove (fn [x] false) '()))");
    probe(&mut sess, "(not-any? (fn [x] false) '())");
    probe(&mut sess, "(not= 'a 'b)");
    probe(&mut sess, "(vary-meta 'foo assoc :doc \"d\")");
    probe(&mut sess, "(symbol \"clojure.core\" \"refer\")");
    // The with-loading-context macro on its own.
    probe(&mut sess, "(macroexpand-1 '(with-loading-context 1 2))");
}

#[test]
#[ignore = "minimal ns switch in prefix session"]
fn minimal_ns_switch() {
    let mut sess = load_core_prefix();
    eprintln!("ns switch");
    sess.eval_str("(in-ns 'clojure.core.protocols)");
    eprintln!("ns switched OK; def marker");
    sess.eval_str("(def marker 5)");
    eprintln!("def OK");
    let bits = sess.eval_str("marker");
    eprintln!("marker bits=0x{bits:016x}");
}

#[test]
#[ignore = "bisect protocols.clj form-by-form"]
fn bisect_protocols() {
    use clojure_jvm::lang::lisp_reader::Reader;
    const SRC: &str = include_str!("../clojure/core_protocols.clj");
    let mut sess = load_core_prefix();
    let mut byte_pos = 0usize;
    let mut n = 0usize;
    loop {
        let slice = &SRC[byte_pos..];
        let mut r = Reader::new(slice);
        let before = r.byte_pos();
        let read = r.read();
        let after = r.byte_pos();
        byte_pos += after - before;
        match read {
            Ok(Some(form)) => {
                let head = format!("{form:?}");
                let short: String = head.chars().take(70).collect();
                eprintln!("--- form {n}: {short}");
                sess.eval_form(form);
                eprintln!("    OK");
                n += 1;
            }
            Ok(None) => break,
            Err(e) => panic!("read err: {e}"),
        }
    }
    eprintln!("ALL {n} FORMS LOADED");
}

#[test]
#[ignore = "isolate multi-arity protocol support"]
fn multi_arity_protocol() {
    let mut sess = load_core_prefix();
    eprintln!("defprotocol multi-arity");
    sess.eval_str("(defprotocol CR (cr [c f] [c f v]))");
    eprintln!("extend-protocol multi-arity");
    sess.eval_str("(extend-protocol CR java.lang.Object (cr ([c f] 11) ([c f v] 22)))");
    eprintln!("call arity 2");
    check_int(&mut sess, "(cr (list 1) +)", 11);
    eprintln!("call arity 3");
    check_int(&mut sess, "(cr (list 1) + 99)", 22);
}

#[test]
#[ignore = "just load protocols on top of prefix"]
fn load_protocols_on_prefix() {
    let mut sess = load_core_prefix();
    eprintln!("PREFIX LOADED; loading protocols");
    sess.eval_str("(clojure.lang.RT/load \"core/protocols\")");
    eprintln!("PROTOCOLS LOADED OK");
    // Sanity: the protocol var exists.
    let bits = sess.eval_str("(if clojure.core.protocols/coll-reduce 1 0)");
    eprintln!("coll-reduce var bits=0x{bits:016x}");
}

#[test]
#[ignore = "full reduce via loaded protocols"]
fn reduce_via_protocols() {
    let mut sess = load_core_prefix();
    // Load the real protocols file the Clojure way (reentrant RT/load).
    sess.eval_str("(clojure.lang.RT/load \"core/protocols\")");
    eprintln!("=== coll-reduce (protocol dispatch) ===");
    // Arity 2 (no init): (f) seeded reduce over a cons list.
    check_int(&mut sess, "(clojure.core.protocols/coll-reduce (list 1 2 3 4) +)", 10);
    // Arity 3 (with init).
    check_int(&mut sess, "(clojure.core.protocols/coll-reduce (list 1 2 3 4) + 100)", 110);
    // nil collection.
    check_int(&mut sess, "(clojure.core.protocols/coll-reduce nil + 42)", 42);
    // Over a lazy seq (map result).
    check_int(&mut sess, "(clojure.core.protocols/coll-reduce (map inc (list 1 2 3)) + 0)", 9);
    // Early termination via reduced.
    check_int(
        &mut sess,
        "(clojure.core.protocols/coll-reduce (list 1 2 3 4 5 6 7 8) \
           (fn [acc x] (if (>= acc 6) (reduced acc) (+ acc x))) 0)",
        10,
    );
}

#[test]
#[ignore = "probe reduce dependencies"]
fn probe_reduce_deps() {
    let mut sess = load_core_prefix();
    eprintln!("=== reduce dependency primitives ===");
    probe(&mut sess, "(reduced 5)");
    probe(&mut sess, "(reduced? (reduced 5))");
    probe(&mut sess, "(reduced? 5)");
    probe(&mut sess, "@(reduced 5)");
    probe(&mut sess, "(deref (reduced 5))");
    probe(&mut sess, "(class (list 1 2))");
    probe(&mut sess, "(instance? clojure.lang.ISeq (list 1 2))");
    probe(&mut sess, "(if-let [x (seq (list 1 2))] (first x) :none)");
    probe(&mut sess, "(loop [i 0 acc 0] (if (< i 3) (recur (inc i) (+ acc i)) acc))");
}

#[test]
#[ignore = "diagnostic probe of core feature support"]
fn probe_interesting_features() {
    let mut sess = load_core_prefix();
    eprintln!("=== threading macros ===");
    check_int(&mut sess, "(-> 5 inc inc)", 7);
    check_int(&mut sess, "(->> 5 inc inc)", 7);
    check_int(&mut sess, "(-> 10 (- 3) (- 2))", 5);
    check_int(&mut sess, "(->> (list 1 2 3) (cons 0) first)", 0);
    eprintln!("=== seq fns ===");
    check_int(&mut sess, "(first (map inc (list 1 2 3)))", 2);
    check_int(&mut sess, "(first (rest (map inc (list 1 2 3))))", 3);
    check_int(&mut sess, "(first (filter odd? (list 2 4 5 6)))", 5);
    check_int(&mut sess, "(reduce1 + (list 1 2 3 4))", 10);
    check_int(&mut sess, "(count (map inc (list 1 2 3)))", 3);
    check_int(&mut sess, "(count (filter odd? (list 1 2 3 4 5)))", 3);
    eprintln!("=== composed ===");
    check_int(&mut sess, "(reduce1 + (map inc (filter odd? (list 1 2 3 4 5))))", 12);
    check_int(&mut sess, "(->> (list 1 2 3 4 5) (filter odd?) (map inc) (reduce1 +))", 12);
}
