//! try/catch/finally semantics vs real Clojure (oracle-verified expectations).
//!
//! ONE #[test] fn on purpose: each test creates a Session, and multiple
//! Session-creating tests in one test process hit the known pre-existing
//! multi-session crash (NAMESPACES/VAR_ROOTS are process-global while each
//! Session owns its JIT CallTable + heap).
//!
//! Every expected value below was produced by /opt/homebrew/bin/clojure.

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

#[test]
#[ignore = "try/catch/finally semantics (single-session; run individually)"]
fn try_catch_semantics() {
    let mut sess = load_core_prefix();
    let mut eval = |src: &str| clojure_jvm::runtime::pr_str_bits(sess.eval_str(src));

    // Body value passes through finally.
    assert_eq!(eval("(try 1 (finally nil))"), "1");
    // catch + finally together.
    assert_eq!(
        eval("(try (throw (ex-info \"x\" {})) (catch Exception e :caught) (finally nil))"),
        ":caught"
    );
    // finally runs after catch, in order.
    assert_eq!(
        eval(
            "(let [a (atom [])] \
               (try (swap! a conj :body) \
                    (throw (ex-info \"x\" {})) \
                    (catch Exception e (swap! a conj :catch)) \
                    (finally (swap! a conj :finally))) \
               @a)"
        ),
        "[:body :catch :finally]"
    );
    // Catch-class selection: non-matching inner catch re-aborts to outer try.
    assert_eq!(
        eval(
            "(try (try (throw (ex-info \"x\" {})) (catch ArithmeticException e :wrong)) \
                  (catch Exception e :outer))"
        ),
        ":outer"
    );
    // Throw across a fn-call boundary is caught by the enclosing try.
    assert_eq!(
        eval("(try ((fn [] (throw (ex-info \"y\" {})))) (catch Exception e :inner-fn))"),
        ":inner-fn"
    );
    // Re-throw from a catch is caught by the outer try.
    assert_eq!(
        eval(
            "(try (try (throw (ex-info \"a\" {:x 1})) \
                       (catch Exception e (throw (ex-info \"b\" {:x 2})))) \
                  (catch Exception e (:x (ex-data e))))"
        ),
        "2"
    );
    // Multiple catch arms: first assignable arm wins, finally still runs.
    assert_eq!(
        eval(
            "(try (throw (ex-info \"x\" {})) \
                  (catch ArithmeticException e :a) \
                  (catch Exception e :e) \
                  (finally nil))"
        ),
        ":e"
    );
}
