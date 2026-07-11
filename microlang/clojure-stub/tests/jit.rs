//! The mini-Clojure frontend running WHOLLY on the native JIT tier (no fallback).
//! Exercises exactly the machinery the JIT used to bail out on: protocol dispatch,
//! `deftype` methods, `.-field` access, `apply`, and try/catch/finally — plus the
//! ported persistent collections that are built out of all of them.
//!
//! Run with: `cargo test -p clojure-stub --features jit --test jit`.
#![cfg(feature = "jit")]

use microlang::jit_cranelift::JitCranelift;
use microlang::{LowBitModel, Runtime};

/// Run a program on the JIT backend and format the result.
fn jit(src: &str) -> String {
    let mut rt = Runtime::<LowBitModel>::new();
    let backend = JitCranelift::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &backend, src);
    clojure_stub::clj_str(&rt, r)
}

#[test]
fn arithmetic_and_fns() {
    assert_eq!(jit("(defn fact [n] (if (< n 2) 1 (* n (fact (- n 1))))) (fact 10)"), "3628800");
    assert_eq!(jit("(reduce + 0 (range 100))"), "4950");
    assert_eq!(jit("(map inc [1 2 3])"), "(2 3 4)");
}

#[test]
fn persistent_vector_on_jit() {
    assert_eq!(jit("(conj [1 2] 3)"), "[1 2 3]");
    assert_eq!(jit("(assoc [1 2 3] 1 :x)"), "[1 :x 3]");
    assert_eq!(jit("(nth (vec (range 2000)) 1999)"), "1999");
    assert_eq!(jit("(count (vec (range 1000)))"), "1000");
    assert_eq!(jit("(peek (pop [1 2 3]))"), "2");
    assert_eq!(jit("(reduce + (vec (range 100)))"), "4950");
}

#[test]
fn maps_and_sets_on_jit() {
    assert_eq!(jit("(get {:a 1 :b 2} :b)"), "2");
    assert_eq!(jit("(type-of (into {} (map (fn [i] [i i]) (range 20))))"), "PersistentHashMap");
    assert_eq!(jit("(count (into {} (map (fn [i] [i (* i i)]) (range 200))))"), "200");
    assert_eq!(jit("(get (into {} (map (fn [i] [i (* i i)]) (range 200))) 137)"), "18769");
    assert_eq!(jit("(contains? #{1 2 3} 2)"), "true");
    assert_eq!(jit("(count (into #{} [1 1 2 2 3]))"), "3");
    assert_eq!(jit("(count (set (range 100)))"), "100");
}

#[test]
fn protocols_and_deftype_on_jit() {
    // deftype + inline protocol methods + .-field, all JIT-native.
    let pair = "(deftype Pair [a b]) \
                (extend-type Pair ISeqable (-seq [p] (list (field p 0) (field p 1))) \
                                  ICounted (-count [p] 2))";
    assert_eq!(jit(&format!("{pair} (seq (->Pair 10 20))")), "(10 20)");
    assert_eq!(jit(&format!("{pair} (map inc (->Pair 3 4))")), "(4 5)");
    // a user protocol dispatched on the JIT
    assert_eq!(
        jit("(defprotocol Shape (area [s])) (deftype Sq [side]) \
             (extend-type Sq Shape (area [s] (* (field s 0) (field s 0)))) (area (->Sq 5))"),
        "25"
    );
}

#[test]
fn multimethods_on_jit() {
    assert_eq!(
        jit("(defmulti f (fn [x] (mod x 2))) (defmethod f 0 [x] :even) (defmethod f 1 [x] :odd) [(f 4) (f 7)]"),
        "[:even :odd]"
    );
}

#[test]
fn apply_on_jit() {
    assert_eq!(jit("(apply + [1 2 3 4])"), "10");
    assert_eq!(jit("(apply + 1 2 [3 4 5])"), "15");
    assert_eq!(jit("(apply max [3 7 2 9 4])"), "9");
    assert_eq!(jit("(apply str (interpose \",\" [1 2 3]))"), "\"1,2,3\"");
}

#[test]
fn try_catch_on_jit() {
    assert_eq!(jit("(try (throw \"boom\") (catch :default e e))"), "\"boom\"");
    assert_eq!(jit("(try (+ 1 2) (catch :default e :caught))"), "3");
    assert_eq!(jit("(try (nth [1 2] 10) (catch :default e :oob))"), ":oob");
    assert_eq!(
        jit("(let [a (atom 0)] (try (throw \"x\") (catch :default e (reset! a 99)) (finally (swap! a inc))) @a)"),
        "100"
    );
    // a throw raised and caught across a JIT-compiled higher-order call
    assert_eq!(
        jit("(reduce + (map (fn [x] (try (if (even? x) x (throw \"odd\")) (catch :default e 0))) (range 10)))"),
        "20"
    );
}

#[test]
fn dynamic_vars_on_jit() {
    // `^:dynamic` + `binding` compiled: override, dynamic scope through a call,
    // set!, unwind-on-throw, nesting — all via the %dyn-* prims on the JIT.
    assert_eq!(jit("(def ^:dynamic *x* 10) (binding [*x* 20] *x*)"), "20");
    assert_eq!(jit("(def ^:dynamic *x* 1) (defn gx [] *x*) (binding [*x* 99] (gx))"), "99");
    assert_eq!(jit("(def ^:dynamic *x* 0) (binding [*x* 5] (set! *x* 7) *x*)"), "7");
    assert_eq!(jit("(def ^:dynamic *x* 1) (try (binding [*x* 9] (throw \"b\")) (catch :default e *x*))"), "1");
    assert_eq!(jit("(def ^:dynamic *x* 1) [(binding [*x* 2] (binding [*x* 3] *x*)) *x*]"), "[3 1]");
    // thread-local across a JIT worker thread (its binding stack starts empty).
    assert_eq!(jit("(def ^:dynamic *x* 0) (binding [*x* 5] [*x* @(future *x*)])"), "[5 0]");
}

#[test]
fn threads_on_jit() {
    // `future` spawns an OS thread that runs its closure on ITS OWN native JIT,
    // sharing the heap/atoms; `deref`/`@` joins it (parking during the wait).
    assert_eq!(jit("@(future (+ 1 2))"), "3");
    // atomic swap! across JIT worker threads over the shared atom
    assert_eq!(jit("(let [a (atom 0)] (dotimes [_ 5] @(future (swap! a inc))) @a)"), "5");
    // fan out N futures and combine — all workers native
    assert_eq!(jit("(reduce + (map deref (mapv (fn [i] (future (* i i))) (range 8))))"), "140");
    assert_eq!(
        jit("(reduce + (map deref (mapv (fn [i] (future (reduce + (range i)))) (range 10))))"),
        "120"
    );
    // a future whose body itself dispatches through the collection protocols
    assert_eq!(jit("@(future (count (into {} (map (fn [i] [i i]) (range 30)))))"), "30");
}

#[test]
fn lazy_seqs_on_jit() {
    assert_eq!(jit("(take 5 (map (fn [x] (* x x)) (range)))"), "(0 1 4 9 16)");
    assert_eq!(jit("(reduce + (take 10 (iterate inc 1)))"), "55");
    assert_eq!(jit("(for [x (range 4) :when (even? x)] (* x 10))"), "(0 20)");
}
