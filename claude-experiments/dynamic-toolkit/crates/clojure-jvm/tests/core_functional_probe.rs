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

    // (expr, expected pr-str). Grouped by subsystem.
    let cases: &[(&str, &str)] = &[
        // ---- sequence library (front-loaded; the suspected-broken area) ----
        ("(seq [1 2 3])", "(1 2 3)"),
        ("(reduce + [1 2 3 4])", "10"),
        ("(reduce + 100 [1 2 3])", "106"),
        ("(apply + [1 2 3 4])", "10"),
        ("(range 4)", "(0 1 2 3)"),
        ("(doall (map inc [1 2 3]))", "(2 3 4)"),
        ("(map inc [1 2 3])", "(2 3 4)"),
        ("(filter even? [1 2 3 4])", "(2 4)"),
        ("(take 3 (range 10))", "(0 1 2)"),
        ("(into [] '(1 2 3))", "[1 2 3]"),
        // ---- arithmetic / numbers ----
        ("(+ 1 2 3)", "6"),
        ("(* 2 3 4)", "24"),
        ("(- 10 3 2)", "5"),
        ("(quot 17 5)", "3"),
        ("(rem 17 5)", "2"),
        ("(inc 41)", "42"),
        ("(max 3 7 2)", "7"),
        ("(< 1 2 3)", "true"),
        ("(= 2 2)", "true"),
        // ---- collections: vectors ----
        ("(vector 1 2 3)", "[1 2 3]"),
        ("(conj [1 2] 3)", "[1 2 3]"),
        ("(count [1 2 3 4])", "4"),
        ("(nth [10 20 30] 1)", "20"),
        ("(get [10 20 30] 2)", "30"),
        ("(peek [1 2 3])", "3"),
        ("(subvec [1 2 3 4] 1 3)", "[2 3]"),
        // ---- collections: maps ----
        ("(get {:a 1 :b 2} :b)", "2"),
        ("(assoc {:a 1} :b 2)", "{:a 1, :b 2}"),
        ("(count {:a 1 :b 2})", "2"),
        ("(keys {:a 1})", "(:a)"),
        ("(vals {:a 1})", "(1)"),
        ("(contains? {:a 1} :a)", "true"),
        // ---- collections: sets ----
        ("(conj #{1 2} 3)", "#{1 3 2}"),
        ("(contains? #{1 2 3} 2)", "true"),
        // ---- lists / seqs ----
        ("(first '(1 2 3))", "1"),
        ("(rest '(1 2 3))", "(2 3)"),
        ("(cons 0 '(1 2))", "(0 1 2)"),
        ("(list 1 2 3)", "(1 2 3)"),
        ("(reverse [1 2 3])", "(3 2 1)"),
        // ---- higher-order / sequence library ----
        ("(map inc [1 2 3])", "(2 3 4)"),
        ("(filter even? [1 2 3 4])", "(2 4)"),
        ("(reduce + [1 2 3 4])", "10"),
        ("(reduce + 100 [1 2 3])", "106"),
        ("(into [] '(1 2 3))", "[1 2 3]"),
        ("(take 3 (range 10))", "(0 1 2)"),
        ("(drop 2 [1 2 3 4])", "(3 4)"),
        ("(range 4)", "(0 1 2 3)"),
        ("(apply + [1 2 3 4])", "10"),
        ("(remove odd? [1 2 3 4])", "(2 4)"),
        ("(mapcat (fn [x] [x x]) [1 2])", "(1 1 2 2)"),
        ("(some even? [1 3 4])", "true"),
        ("(every? pos? [1 2 3])", "true"),
        ("(partition 2 [1 2 3 4])", "((1 2) (3 4))"),
        ("(sort [3 1 2])", "(1 2 3)"),
        ("(distinct [1 1 2 2 3])", "(1 2 3)"),
        ("(frequencies [1 1 2])", "{1 2, 2 1}"),
        ("(group-by even? [1 2 3 4])", "{false [1 3], true [2 4]}"),
        // ---- strings ----
        ("(str \"a\" \"b\" \"c\")", "\"abc\""),
        ("(str 1 2 3)", "\"123\""),
        ("(clojure.string/upper-case \"abc\")", "\"ABC\""),
        ("(subs \"hello\" 1 3)", "\"el\""),
        // ---- control / macros ----
        ("(if (> 2 1) :yes :no)", ":yes"),
        ("(let [x 5] (* x x))", "25"),
        ("(cond (= 1 2) :a (= 1 1) :b)", ":b"),
        ("(when true 42)", "42"),
        ("(-> 5 inc inc)", "7"),
        ("(->> [1 2 3] (map inc) (reduce +))", "9"),
        ("(loop [i 0 acc 0] (if (= i 5) acc (recur (inc i) (+ acc i))))", "10"),
        // ---- printing (multimethod subsystem) ----
        ("(pr-str [1 2 3])", "\"[1 2 3]\""),
        ("(pr-str {:a 1})", "\"{:a 1}\""),
    ];

    use std::io::Write;
    let mut pass = 0usize;
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
            Ok(s) => eprintln!("WRONG got {s:?} want {expected:?}"),
            Err(_) => {
                let m = LAST_PANIC
                    .with(|p| p.borrow_mut().take())
                    .unwrap_or_else(|| "<panic>".into());
                eprintln!("PANIC {}", m.chars().take(80).collect::<String>());
            }
        }
        let _ = std::io::stderr().flush();
    }
    eprintln!("\n===== PASS {pass}/{} =====", cases.len());
}
