//! Oracle-verified probes for the Repeat seq type, lazy-aware `vec`/
//! `toArray`, and regex-backed `clojure.string/split`. Expected values
//! were produced by the real `clojure` CLI.
//!
//! Run: `cargo test -p clojure-jvm --release --test probe_seq_fixes -- --nocapture`

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;

const UPSTREAM_CORE: &str =
    "/Users/jimmyhmiller/Documents/Code/open-source/clojure/src/clj/clojure/core.clj";

fn load_full_core() -> Session {
    let src = std::fs::read_to_string(UPSTREAM_CORE).expect("upstream core.clj");
    let mut sess = Session::new();
    let mut byte_pos = 0usize;
    loop {
        let slice = &src[byte_pos..];
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut r = Reader::new(slice);
            let f = r.read();
            (f, r.byte_pos())
        }));
        let (form, after) = match read {
            Ok((Ok(Some(f)), a)) => (f, a),
            _ => break,
        };
        byte_pos += after;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sess.eval_form(form);
        }));
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sess.eval_str("(clojure.lang.RT/load \"clojure/string\")");
    }));
    sess
}

#[test]
fn probe_seq_fixes() {
    let mut sess = load_full_core();
    let cases: &[(&str, &str)] = &[
        // Bounded repeat: seq ops, count, realization.
        ("(repeat 3 :x)", "(:x :x :x)"),
        ("(count (repeat 3 :x))", "3"),
        ("(vec (repeat 2 7))", "[7 7]"),
        // Infinite repeat consumed lazily.
        ("(into [] (take 4 (repeat 9)))", "[9 9 9 9]"),
        ("(first (repeat :a))", ":a"),
        // The two interpose conformance exprs.
        ("(into [] (interpose 0 [1 2 3]))", "[1 0 2 0 3]"),
        ("(apply str (interpose \"-\" [\"a\" \"b\" \"c\"]))", "\"a-b-c\""),
        // partition-all (vec over lazy segments).
        ("(into [] (map vec (partition-all 2 [1 2 3])))", "[[1 2] [3]]"),
        // vec over plain lazy seqs.
        ("(vec (map inc [1 2 3]))", "[2 3 4]"),
        ("(vec (filter odd? [1 2 3 4 5]))", "[1 3 5]"),
        // string/split.
        ("(clojure.string/split \"a,b,c\" #\",\")", "[\"a\" \"b\" \"c\"]"),
        ("(clojure.string/split \"a1b22c\" #\"\\d+\")", "[\"a\" \"b\" \"c\"]"),
        ("(clojure.string/split \"a,b,,\" #\",\")", "[\"a\" \"b\"]"),
        // interleave/take over infinite repeat (interpose's building blocks).
        ("(into [] (interleave (repeat :s) [1 2]))", "[:s 1 :s 2]"),
        // Atoms / volatiles.
        ("(let [a (atom 0)] (swap! a inc) @a)", "1"),
        ("(let [a (atom {})] (swap! a assoc :k 1) @a)", "{:k 1}"),
        ("(let [a (atom 0)] (reset! a 42) @a)", "42"),
        ("(let [a (atom 0)] (compare-and-set! a 0 9) @a)", "9"),
        ("(let [a (atom 0)] (compare-and-set! a 5 9) @a)", "0"),
        ("(let [a (atom [])] (dotimes [i 3] (swap! a conj i)) @a)", "[0 1 2]"),
        ("(let [v (volatile! 1)] (vswap! v inc) @v)", "2"),
        ("(let [a (atom 0)] (while (< @a 3) (swap! a inc)) @a)", "3"),
        ("(let [a (atom 0)] (swap! a + 1 2 3 4) @a)", "10"),
        ("(let [s (atom [])] (doseq [x [1 2 3]] (swap! s conj x)) @s)", "[1 2 3]"),
        // Statics.
        ("(Math/abs -3)", "3"),
        ("(Math/floor 3.7)", "3.0"),
        ("(Math/ceil 3.2)", "4.0"),
        ("(Integer/parseInt \"42\")", "42"),
        // Predicates / identity.
        ("(fn? inc)", "true"),
        ("(fn? :kw)", "false"),
        ("(ifn? :kw)", "true"),
        ("(identical? :a :a)", "true"),
        ("(identical? 'a 'a)", "false"),
        ("(counted? [1])", "true"),
        ("(associative? {})", "true"),
        ("(associative? #{})", "false"),
        ("(sorted? (sorted-map))", "true"),
        ("(sorted? {})", "false"),
        // Meta.
        ("(meta (with-meta [1] {:m true}))", "{:m true}"),
        ("(:m (meta (with-meta [1] {:m 1})))", "1"),
        // defonce.
        ("(do (defonce probe-og 5) probe-og)", "5"),
        // empty.
        ("(empty [1 2])", "[]"),
        ("(empty {:a 1})", "{}"),
        // string round 2.
        ("(clojure.string/capitalize \"hELLO\")", "\"Hello\""),
        ("(clojure.string/triml \"  a\")", "\"a\""),
        ("(clojure.string/trimr \"a  \")", "\"a\""),
        ("(into [] (clojure.string/split-lines \"a\\nb\"))", "[\"a\" \"b\"]"),
        ("(clojure.string/index-of \"hello\" \"l\")", "2"),
        ("(clojure.string/last-index-of \"hello\" \"l\")", "3"),
        ("(clojure.string/index-of \"hello\" \"z\")", "nil"),
        // for — multi-binding, :when, :while, :let, laziness.
        ("(into [] (for [x [1 2] y [10 20]] (+ x y)))", "[11 21 12 22]"),
        ("(into [] (for [x (range 5) :when (odd? x)] x))", "[1 3]"),
        ("(into [] (for [x (range 6) :while (< x 3)] x))", "[0 1 2]"),
        ("(into [] (for [x [1 2] :let [y (* x 10)] :when (> y 10)] y))", "[20]"),
        ("(into [] (take 3 (for [x (iterate inc 0)] (* x x))))", "[0 1 4]"),
        // letfn — forward + mutual recursion.
        ("(letfn [(f [x] (g x)) (g [x] (* 2 x))] (f 5))", "10"),
        ("(letfn [(even2? [n] (if (zero? n) true (odd2? (dec n)))) (odd2? [n] (if (zero? n) false (even2? (dec n))))] (even2? 10))", "true"),
        // when-some / if-some (nil-test, not truthiness).
        ("(when-some [x 3] (inc x))", "4"),
        ("(when-some [x false] :ran)", ":ran"),
        ("(if-some [x nil] x :no)", ":no"),
        // macroexpand-1 / macroexpand the fns.
        ("(macroexpand-1 '(when true 1))", "(if true (do 1))"),
        // iterate / cycle.
        ("(into [] (take 5 (iterate inc 0)))", "[0 1 2 3 4]"),
        ("(into [] (take 4 (cycle [1 2])))", "[1 2 1 2]"),
        ("(first (iterate inc 7))", "7"),
        ("(into [] (take 5 (cycle [1 2 3])))", "[1 2 3 1 2]"),
        ("(into [] (repeatedly 2 (constantly 7)))", "[7 7]"),
        // select-keys cluster.
        ("(into (sorted-map) (select-keys {:a 1 :b 2 :c 3} [:a :c]))", "{:a 1, :c 3}"),
        ("(into (sorted-map) (update-vals {:a 1 :b 2} inc))", "{:a 2, :b 3}"),
        ("(into (sorted-map) (update-keys {\"a\" 1} keyword))", "{:a 1}"),
        // regex fns.
        ("(re-find #\"\\d+\" \"abc123def\")", "\"123\""),
        ("(into [] (re-seq #\"\\d\" \"a1b2\"))", "[\"1\" \"2\"]"),
        ("(re-matches #\"a.c\" \"abc\")", "\"abc\""),
        ("(re-matches #\"a.c\" \"abcd\")", "nil"),
        ("(re-find #\"(a+)(b+)\" \"xxaabbyy\")", "[\"aabb\" \"aa\" \"bb\"]"),
        // format / hash.
        ("(format \"%s=%d\" \"x\" 5)", "\"x=5\""),
        ("(hash 42)", "1871679806"),
        ("(hash \"abc\")", "74834163"),
        // Transducers via sequence.
        ("(into [] (sequence (map inc) [1 2 3]))", "[2 3 4]"),
        ("(into [] (dedupe [1 1 2 2 1]))", "[1 2 1]"),
        ("(into [] (sequence (comp (filter odd?) (map inc)) (range 6)))", "[2 4 6]"),
        ("(transduce (take 2) + [1 2 3 4])", "3"),
        // Multimethods.
        (
            "(do (defmulti pm-area :shape) (defmethod pm-area :circle [_] :c) (pm-area {:shape :circle}))",
            ":c",
        ),
        (
            "(do (defmulti pm-kind (fn [x] x)) (defmethod pm-kind :a [_] 1) (defmethod pm-kind :default [_] :dflt) [(pm-kind :a) (pm-kind :zz)])",
            "[1 :dflt]",
        ),
        // eval.
        ("(eval (quote (+ 1 2)))", "3"),
        ("(into [] (eval (quote (map inc [1 2 3]))))", "[2 3 4]"),
        ("(let [f (eval (quote (fn [x] (* x x))))] (f 7))", "49"),
    ];
    let mut failures = Vec::new();
    for (expr, want) in cases {
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            clojure_jvm::runtime::pr_str_bits(sess.eval_str(expr))
        }))
        .unwrap_or_else(|_| "PANIC".to_string());
        let ok = got == *want;
        println!("{} {expr}\n    got  {got}\n    want {want}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures.push(expr.to_string());
        }
    }
    assert!(failures.is_empty(), "failed probes: {failures:?}");
}

/// The hand-written HeapTypeIds fixtures in compiler.rs tests hardcode
/// `repeat_seq: 24`; pin the real assignment so a drift fails loudly.
#[test]
fn repeat_type_id_matches_fixtures() {
    let _sess = Session::new();
    assert_eq!(
        clojure_jvm::runtime::heap_type_ids().repeat_seq,
        24,
        "clojure.lang.Repeat ObjTypeId moved — update the HeapTypeIds \
         fixtures in compiler.rs tests"
    );
}
