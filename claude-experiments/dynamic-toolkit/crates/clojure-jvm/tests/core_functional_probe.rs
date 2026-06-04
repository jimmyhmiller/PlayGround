//! Honest functional assessment: load the FULL upstream core.clj (skipping
//! forms that fail), then evaluate a battery of representative core
//! expressions and report how many actually produce the right value. This
//! measures "how far from a working clojure core" — distinct from "does it
//! load."

use clojure_jvm::lang::compiler::Session;
use clojure_jvm::lang::lisp_reader::Reader;
use std::cell::RefCell;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

const UPSTREAM_CORE: &str =
    "/Users/jimmyhmiller/Documents/Code/open-source/clojure/src/clj/clojure/core.clj";

fn is_top_load(form: &clojure_jvm::lang::object::Object) -> bool {
    use clojure_jvm::lang::object::Object;
    if let Object::List(l) = form {
        if let Some(Object::Symbol(s)) = l.iter().next() {
            return s.get_name() == "load";
        }
    }
    false
}

fn load_core(stop_at_first_load: bool) -> Session {
    std::panic::set_hook(Box::new(|info| {
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(info.to_string()));
    }));
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
        if stop_at_first_load && is_top_load(&form) {
            break;
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sess.eval_form(form);
        }));
    }
    sess
}

#[test]
#[ignore = "functional assessment — long-running"]
fn core_functional_battery() {
    let prefix = std::env::var("PROBE_PREFIX").is_ok();
    eprintln!("=== loading core ({}) ===", if prefix { "PREFIX only" } else { "FULL" });
    let mut sess = load_core(prefix);

    // (expr, expected pr-str).
    //
    // LAZY SEQUENCES WORK: `map`/`filter`/`take`/composition all produce
    // correct results when realized via `reduce`/`into`/`count`/`last`/manual
    // `loop`+`next` — see the "lazy" group below. (A *bare* unrealized lazy
    // seq can't be PRINTED by this harness because `pr_str_bits` runs after
    // `eval_str` tears down the JIT run context, so forcing it then has no
    // safepoint session; inside real eval, forcing works. So we realize lazy
    // results to a concrete vector/number before printing.)
    let cases: &[(&str, &str)] = &[
        // ---- lazy sequences (realized to a concrete value before print) ----
        ("(into [] (map inc [1 2 3]))", "[2 3 4]"),
        ("(into [] (filter even? [1 2 3 4 5 6]))", "[2 4 6]"),
        ("(into [] (map inc (filter even? [1 2 3 4])))", "[3 5]"),
        ("(into [] (take 3 (map inc [10 20 30 40])))", "[11 21 31]"),
        ("(reduce + (map inc [1 2 3 4]))", "14"),
        ("(reduce + (map (fn [x] (* x x)) [1 2 3]))", "14"),
        ("(count (map inc [1 2 3 4 5]))", "5"),
        ("(last (map inc [1 2 3]))", "4"),
        ("(count (doall (map inc [1 2 3])))", "3"),
        ("(first (next (next (map inc [1 2 3]))))", "4"),
        ("(loop [s (seq (map inc [1 2 3])) acc 0] (if s (recur (next s) (+ acc (first s))) acc))", "9"),
        // ---- reduce (the fix: GC-rooted protocol dispatch through the
        //      multi-arity coll-reduce path). Must work with primops, inline
        //      fns, AND user fns — not just `+`. ----
        // ---- reduce (the fix: GC-rooted protocol dispatch through the
        //      multi-arity coll-reduce path). Must work with primops, inline
        //      fns, AND user fns — not just `+`. ----
        ("(reduce + [1 2 3 4])", "10"),
        ("(reduce + 100 [1 2 3])", "106"),
        ("(reduce (fn [a b] (+ a b)) [1 2 3 4])", "10"),
        ("(do (defn add2 [a b] (+ a b)) (reduce add2 [1 2 3 4]))", "10"),
        ("(reduce + (list 5 5 5))", "15"),
        ("(seq [1 2 3])", "(1 2 3)"),
        ("(apply + [1 2 3 4])", "10"),
        // ---- arithmetic / numbers ----
        ("(+ 1 2 3)", "6"),
        ("(* 2 3 4)", "24"),
        ("(- 10 3 2)", "5"),
        ("(inc 41)", "42"),
        ("(max 3 7 2)", "7"),
        ("(< 1 2 3)", "true"),
        ("(= 2 2)", "true"),
        // ---- collections: vectors ----
        ("(vector 1 2 3)", "[1 2 3]"),
        ("(conj [1 2] 3)", "[1 2 3]"),
        ("(count [1 2 3 4])", "4"),
        ("(nth [10 20 30] 1)", "20"),
        ("(peek [1 2 3])", "3"),
        ("(subvec [1 2 3 4] 1 3)", "[2 3]"),
        // ---- collections: maps ----
        ("(get {:a 1 :b 2} :b)", "2"),
        ("(count {:a 1 :b 2})", "2"),
        ("(keys {:a 1})", "(:a)"),
        ("(vals {:a 1})", "(1)"),
        ("(contains? {:a 1} :a)", "true"),
        // ---- lists / seqs ----
        ("(first '(1 2 3))", "1"),
        ("(rest '(1 2 3))", "(2 3)"),
        ("(cons 0 '(1 2))", "(0 1 2)"),
        ("(list 1 2 3)", "(1 2 3)"),
        ("(reverse [1 2 3])", "(3 2 1)"),
        // ---- strings ----
        ("(str \"a\" \"b\" \"c\")", "\"abc\""),
        ("(str 1 2 3)", "\"123\""),
        ("(subs \"hello\" 1 3)", "\"el\""),
        // ---- control / macros ----
        ("(if (> 2 1) :yes :no)", ":yes"),
        ("(let [x 5] (* x x))", "25"),
        ("(cond (= 1 2) :a (= 1 1) :b)", ":b"),
        ("(when true 42)", "42"),
        ("(-> 5 inc inc)", "7"),
        ("(loop [i 0 acc 0] (if (= i 5) acc (recur (inc i) (+ acc i))))", "10"),
    ];

    use std::io::Write;
    let mut pass = 0usize;
    let mut failed: Vec<&str> = Vec::new();
    for (src, expected) in cases {
        // Flush BEFORE eval so a non-unwinding abort still leaves a marker of
        // exactly which expression killed the process.
        eprint!("RUN  {src}  ... ");
        let _ = std::io::stderr().flush();
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            clojure_jvm::runtime::pr_str_bits(sess.eval_str(src))
        }));
        match got {
            Ok(s) if s == *expected => {
                pass += 1;
                eprintln!("ok");
            }
            Ok(s) => {
                eprintln!("WRONG got {s:?} want {expected:?}");
                failed.push(src);
            }
            Err(_) => {
                let m = LAST_PANIC
                    .with(|p| p.borrow_mut().take())
                    .unwrap_or_else(|| "<panic>".into());
                eprintln!("PANIC {}", m.chars().take(80).collect::<String>());
                failed.push(src);
            }
        }
        let _ = std::io::stderr().flush();
    }
    eprintln!("\n===== PASS {pass}/{} =====", cases.len());

    // Regression guard for the GC-rooting fix: `reduce` through the
    // multi-arity `coll-reduce` protocol path must work end-to-end (it
    // failed before the dispatch table was made a GC root source — the
    // multi-arity impl handle dangled after a GC during reentrant load).
    for r in [
        "(reduce + [1 2 3 4])",
        "(reduce (fn [a b] (+ a b)) [1 2 3 4])",
        "(do (defn add2 [a b] (+ a b)) (reduce add2 [1 2 3 4]))",
    ] {
        assert!(
            !failed.contains(&r),
            "reduce regression: `{r}` must work (GC-rooted protocol dispatch)"
        );
    }

    // Lazy sequences must produce correct results when realized.
    for r in [
        "(into [] (map inc [1 2 3]))",
        "(into [] (filter even? [1 2 3 4 5 6]))",
        "(into [] (take 3 (map inc [10 20 30 40])))",
        "(reduce + (map inc [1 2 3 4]))",
        "(last (map inc [1 2 3]))",
        "(loop [s (seq (map inc [1 2 3])) acc 0] (if s (recur (next s) (+ acc (first s))) acc))",
    ] {
        assert!(!failed.contains(&r), "lazy-seq regression: `{r}` must work");
    }
}
