//! Drive a fresh Engine through `core.clj`, one form at a time, and
//! see how far it gets before something blows up. As we add the
//! missing pieces we expand this test.

use clojure::Engine;

const CORE_PATH: &str = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/quick-clojure-poc/src/clojure/core.clj";

#[test]
fn load_core_first_n_lines() {
    // Read the file and feed lines until the first uncompilable form
    // — the test itself succeeds if we can load AT LEAST the protocol
    // declarations through line 446.
    let src = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    // Truncate at the first line we know we can't yet handle.
    let lines: Vec<&str> = src.lines().take(1).collect(); // start tiny
    let prefix = lines.join("\n");
    let e = Engine::new();
    e.eval(&prefix);
}

#[test]
fn load_core_protocols() {
    // Lines 1–456: ns + all defprotocol declarations.
    let src = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    let lines: Vec<&str> = src.lines().take(456).collect();
    let prefix = lines.join("\n");
    let e = Engine::new();
    e.eval(&prefix);
}

#[test]
fn load_core_through_reduced() {
    // Lines 1–476: protocols + (deftype* Reduced) + extend-type +
    // (def reduced ...).
    let src = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    let lines: Vec<&str> = src.lines().take(476).collect();
    let prefix = lines.join("\n");
    let e = Engine::new();
    e.eval(&prefix);
}

// (Removed: lines-1..605 partial-load test. The line count stopped
//  matching the source after the bootstrap macro section was added,
//  and `load_core_full` covers the same ground end-to-end.)

/// Load a specific prefix of core.clj. Helper for bisecting the
/// next failure point without leaving a half-form mid-load.
fn load_first_n_top_level_forms(n: usize) {
    let src = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    // Walk the source to find the end of the n-th top-level form by
    // tracking paren depth. Comments + strings handled.
    let bytes = src.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut in_line_comment = false;
    let mut form_count = 0usize;
    let mut form_started = false;
    let mut last_form_end: usize = 0;
    let mut i = 0;
    while i < bytes.len() && form_count < n {
        let c = bytes[i];
        if in_line_comment {
            if c == b'\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }
        if in_string {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b';' => in_line_comment = true,
            b'"' => {
                in_string = true;
                form_started = true;
            }
            b'(' | b'[' | b'{' => {
                depth += 1;
                form_started = true;
            }
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 && form_started {
                    form_count += 1;
                    last_form_end = i + 1;
                    form_started = false;
                }
            }
            _ => {
                if !c.is_ascii_whitespace() {
                    form_started = true;
                    if depth == 0 {
                        // top-level atom (rare); count once it ends
                        // at next whitespace/paren.
                    }
                }
            }
        }
        i += 1;
    }
    let prefix = std::str::from_utf8(&bytes[..last_form_end]).unwrap();
    let e = Engine::new();
    e.eval(prefix);
}

/// Regression: core.clj redefines `cons` as a seq-aware `defn`, which
/// would shadow the raw extern in the compiler's `func_refs`. Static
/// def-fn call sites use the (def-fn ABI) packed-list form, and the
/// arg-list builder needs the raw `cons` extern — not the user's
/// `(defn cons …)` — to avoid being routed through the closure path
/// with malformed args.
#[test]
fn macros_expand_in_a_defn_body() {
    // Smoke: a defn whose body uses our bootstrap cond+when. Mirrors
    // what spread does early in core.clj.
    let e = Engine::new();
    let core = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    // Take just the prelude up through the macros.
    let bytes = core.as_bytes();
    // Hard-code a slice — bootstrap ends right before "Variadic
    // Arithmetic Functions". Find that marker.
    let marker = b";; Variadic Arithmetic Functions";
    let end = core.find(std::str::from_utf8(marker).unwrap()).unwrap();
    let prelude = std::str::from_utf8(&bytes[..end]).unwrap();
    e.eval(prelude);
    // Now try a minimal user defn that uses cond.
    let v = e.eval("(defn t [x] (cond (nil? x) :nil :else :other)) (t 5)");
    assert_eq!(e.print(v), ":other");
}

#[test]
fn cons_redef_does_not_break_static_calls() {
    let e = Engine::new();
    e.eval("(deftype* Pair [a b])
            (defn cons [x coll] coll)
            (def P (Pair. 1 2))");
}

// Coarse milestones along the load. The protocol/reduced/reader
// tests above pin specific structural milestones; these confirm we
// keep up the pace as new constructs (deftype, multi-arity fn,
// recur-from-let, etc.) come online.
#[test]
fn load_core_50_forms() { load_first_n_top_level_forms(50); }
#[test]
fn load_core_55_forms() { load_first_n_top_level_forms(55); }
#[test]
fn load_core_56_forms() { load_first_n_top_level_forms(56); }
#[test]
fn load_core_57_forms() { load_first_n_top_level_forms(57); }
#[test]
fn load_core_58_forms() { load_first_n_top_level_forms(58); }
#[test]
fn load_core_60_forms() { load_first_n_top_level_forms(60); }
#[test]
fn load_core_65_forms() { load_first_n_top_level_forms(65); }
#[test]
fn load_core_66_forms() { load_first_n_top_level_forms(66); }
#[test]
fn load_core_67_forms() { load_first_n_top_level_forms(67); }
#[test]
fn load_core_68_forms() { load_first_n_top_level_forms(68); }
#[test]
fn load_core_69_forms() { load_first_n_top_level_forms(69); }
#[test]
fn load_core_70_forms() { load_first_n_top_level_forms(70); }
#[test]
fn load_core_75_forms() { load_first_n_top_level_forms(75); }
#[test]
fn load_core_76_forms() { load_first_n_top_level_forms(76); }
#[test]
fn load_core_77_forms() { load_first_n_top_level_forms(77); }
#[test]
fn load_core_78_forms() { load_first_n_top_level_forms(78); }
#[test]
fn load_core_79_forms() { load_first_n_top_level_forms(79); }
#[test]
fn load_core_80_forms() { load_first_n_top_level_forms(80); }
#[test]
fn load_core_85_forms() { load_first_n_top_level_forms(85); }
#[test]
fn load_core_90_forms() { load_first_n_top_level_forms(90); }
#[test]
fn load_core_95_forms() { load_first_n_top_level_forms(95); }
#[test]
fn load_core_100_forms() { load_first_n_top_level_forms(100); }
#[test]
fn load_core_200_forms() { load_first_n_top_level_forms(200); }
#[test]
fn load_core_300_forms() { load_first_n_top_level_forms(300); }

/// Smoke test: after loading the full corpus, can we actually CALL
/// the functions defined in it? Loading proves we compiled them;
/// this proves the compiled bodies don't immediately blow up at
/// call time. Each line is one feature exercised, terse on purpose.
fn smoke(src: &str) -> String {
    let e = Engine::new();
    let core = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    e.eval(&core);
    let v = e.eval(src);
    e.print(v)
}

#[test] fn smoke_inc()              { assert_eq!(smoke("(inc 41)"), "42"); }
#[test] fn smoke_mod_1_2()           { assert_eq!(smoke("(mod 1 2)"), "1"); }
#[test] fn smoke_quot_1_2()          { assert_eq!(smoke("(quot 1 2)"), "0"); }
#[test] fn smoke_rem_1_2()           { assert_eq!(smoke("(rem 1 2)"), "1"); }
#[test] fn smoke_div_1_2()           {
    // We don't have rationals — `/` is float division. (Real Clojure
    // would return `1/2`; we return 0.5. Document the deviation.)
    assert_eq!(smoke("(/ 1 2)"), "0.5");
}
#[test] fn smoke_zero_p_1()          { assert_eq!(smoke("(zero? 1)"), "false"); }
#[test] fn smoke_zero_p_0()          { assert_eq!(smoke("(zero? 0)"), "true"); }
#[test] fn smoke_int_p_1()           { assert_eq!(smoke("(integer? 1)"), "true"); }
#[test] fn smoke_even_p_2()          { assert_eq!(smoke("(even? 2)"), "true"); }
#[test] fn smoke_even_p_1()          { assert_eq!(smoke("(even? 1)"), "false"); }
#[test] fn smoke_filter_one()        { assert_eq!(smoke("(filter even? (list 2))"), "(2)"); }
#[test] fn smoke_filter_skip()       { assert_eq!(smoke("(filter even? (list 1 2))"), "(2)"); }
#[test] fn smoke_raw_is_string()     { assert_eq!(smoke("(__is_string \"hi\")"), "true"); }
#[test] fn smoke_just_list()         { assert_eq!(smoke("(list 1 2)"), "(1 2)"); }
#[test] fn smoke_just_count()        { assert_eq!(smoke("(count (list 1 2))"), "2"); }
#[test] fn smoke_count_string()      { assert_eq!(smoke("(count \"abc\")"), "3"); }
#[test] fn smoke_count_nil()         { assert_eq!(smoke("(count nil)"), "0"); }
#[test] fn smoke_count_empty_list()  { assert_eq!(smoke("(count EMPTY-LIST)"), "0"); }
#[test] fn smoke_cons_nil()          { assert_eq!(smoke("(cons 1 nil)"), "(1)"); }
#[test] fn smoke_count_cons()        { assert_eq!(smoke("(count (cons 1 nil))"), "1"); }
#[test] fn smoke_is_string_truth()   { assert_eq!(smoke("(string? \"hi\")"), "true"); }
#[test] fn smoke_is_string_false()   { assert_eq!(smoke("(string? 42)"), "false"); }
#[test] fn smoke_is_string_list()    { assert_eq!(smoke("(string? (list 1 2))"), "false"); }
#[test] fn smoke_count_list()       { assert_eq!(smoke("(count (list 1 2 3))"), "3"); }
#[test] fn smoke_count_vec()        { assert_eq!(smoke("(count [1 2 3 4])"), "4"); }
#[test] fn smoke_first_rest()       { assert_eq!(smoke("(first (rest (list 1 2 3)))"), "2"); }
#[test] fn smoke_map_inc()          { assert_eq!(smoke("(first (map inc (list 1 2 3)))"), "2"); }
#[test] fn smoke_filter()           { assert_eq!(smoke("(first (filter even? (list 1 2 3 4)))"), "2"); }
#[test] fn smoke_reduce_sum()       { assert_eq!(smoke("(reduce + 0 (list 1 2 3 4 5))"), "15"); }
#[test] fn smoke_str()              { assert_eq!(smoke("(str \"a\" 1 :b)"), "\"a1:b\""); }
#[test] fn smoke_assoc_get()        { assert_eq!(smoke("(get (assoc {} :x 1) :x)"), "1"); }
#[test] fn smoke_conj_vec()         { assert_eq!(smoke("(count (conj [1 2] 3))"), "3"); }
#[test] fn smoke_nth_vec()          { assert_eq!(smoke("(nth [10 20 30] 1)"), "20"); }
#[test] fn smoke_apply()            { assert_eq!(smoke("(apply + (list 1 2 3))"), "6"); }
#[test] fn smoke_into()             { assert_eq!(smoke("(count (into [] (list 1 2 3 4)))"), "4"); }
#[test] fn smoke_seq_reduce_2()      { assert_eq!(smoke("(seq-reduce + (list 1 2 3))"), "6"); }
#[test] fn smoke_seq_reduce_3()      { assert_eq!(smoke("(seq-reduce + 0 (list 1 2 3))"), "6"); }
#[test] fn smoke_reduce_2()          { assert_eq!(smoke("(reduce + (list 1 2 3))"), "6"); }
#[test] fn smoke_reduce_3()          { assert_eq!(smoke("(reduce + 0 (list 1 2 3))"), "6"); }
#[test] fn smoke_vector_basic()     { assert_eq!(smoke("(vector 1 2 3)"), "[1 2 3]"); }
#[test] fn smoke_vector_syms()       { assert_eq!(smoke("(count (vector :a :b))"), "2"); }
#[test] fn smoke_first_vec()         { assert_eq!(smoke("(first [:a :b])"), ":a"); }
#[test] fn smoke_let_basic()         { assert_eq!(smoke("(let [x 5] x)"), "5"); }
#[test] fn smoke_let_nested()        { assert_eq!(smoke("(let [t 5] (if t (let [x t] (inc x)) :no))"), "6"); }
#[test] fn smoke_if_let_basic()      { assert_eq!(smoke("(if-let [x 5] x :no)"), "5"); }
#[test] fn smoke_let_pv_bindings()   { assert_eq!(smoke("(let [v (vector :x 5)] (first v))"), ":x"); }
#[test] fn smoke_seq_iter_pv()       { assert_eq!(smoke("(first (vector :a :b :c))"), ":a"); }
#[test] fn smoke_macro_calls_vector() {
    // Macro returns a form that contains a PersistentVector built via
    // `(vector …)`. Tests whether expand can normalize/walk that.
    assert_eq!(
        smoke("(defmacro mv [x] (list (quote first) (vector x x))) (mv 7)"),
        "7"
    );
}
#[test] fn smoke_macro_with_gensym() {
    assert_eq!(
        smoke("(defmacro mg [] (let [g (gensym \"t__\")] (list (quote let) (vector g 5) g))) (mg)"),
        "5"
    );
}
#[test] fn smoke_macro_two_lets() {
    assert_eq!(
        smoke("(defmacro mt [bindings] (list (quote let) (vector (first bindings) (second bindings)) (cons (quote let) (cons (vector (quote y) (first bindings)) (cons (quote y) nil))))) (mt [x 7])"),
        "7"
    );
}
#[test] fn smoke_inline_whenlet() {
    // Same shape as when-let but defined locally so we can verify
    // the macro mechanism works on this exact pattern.
    assert_eq!(
        smoke("(defmacro wl [bindings & body] (let [b (first bindings) t (second bindings) g (gensym \"t__\")] (list (quote let) (vector g t) (list (quote if) g (cons (quote let) (cons (vector b g) body)) nil)))) (wl [x 5] (inc x))"),
        "6"
    );
}
#[test] fn smoke_count_pv()          { assert_eq!(smoke("(count (vector :a :b :c))"), "3"); }
// (Removed smoke_eval_macro_form — it tested `eval` which we don't
//  expose. Use `e.eval(...)` from Rust instead.)
#[test] fn smoke_simple_macro_use() {
    assert_eq!(smoke("(defmacro answer [] 42) (answer)"), "42");
}
#[test] fn smoke_user_cons_print() {
    assert_eq!(smoke("(cons 1 (cons 2 nil))"), "(1 2)");
}
#[test] fn smoke_macro_consonly() {
    assert_eq!(smoke("(defmacro plus () (cons (quote +) (cons 1 (cons 2 nil)))) (plus)"), "3");
}
#[test] fn smoke_macro_one_cons() {
    assert_eq!(smoke("(defmacro one [] (cons (quote inc) (cons 41 nil))) (one)"), "42");
}
#[test] fn smoke_macro_call_cons_only() {
    // Don't actually invoke the macro — just defining it shouldn't hang.
    assert_eq!(smoke("(defmacro one [] (cons (quote inc) (cons 41 nil))) :ok"), ":ok");
}
#[test] fn smoke_macro_returns_int_via_fn() {
    // Macro body calls a defn that uses cons. Probes whether JIT-calling
    // user cons during macroexpansion works at all.
    assert_eq!(smoke("(defmacro mk [] (count (cons 1 (cons 2 nil)))) (mk)"), "2");
}
#[test] fn smoke_user_cons_first_rest() {
    assert_eq!(smoke("(first (cons 1 (cons 2 nil)))"), "1");
}
#[test] fn smoke_macro_via_quote() {
    // Macro returns a quoted list — different code path than building
    // with cons. If this works but cons-built doesn't, the cons path
    // is producing a non-is_list value.
    assert_eq!(
        smoke("(defmacro qm [] (quote (+ 1 2))) (qm)"),
        "3"
    );
}
#[test] fn smoke_macro_returns_form() {
    // A macro that returns a list which should evaluate as code.
    assert_eq!(smoke("(defmacro one [] (cons (quote +) (cons 1 nil))) (one)"), "1");
}
#[test] fn smoke_local_macro_let() {
    assert_eq!(
        smoke("(defmacro my-let [bindings & body] (list (quote let) (vector (first bindings) (second bindings)) (cons (quote do) body))) (my-let [x 5] x)"),
        "5"
    );
}
#[test] fn smoke_second_vec()        { assert_eq!(smoke("(second [:a :b])"), ":b"); }
#[test] fn smoke_when_let()         { assert_eq!(smoke("(when-let [x 5] (inc x))"), "6"); }
#[test] fn smoke_threading()        { assert_eq!(smoke("(-> 1 inc inc inc)"), "4"); }
#[test] fn smoke_threading_last()   { assert_eq!(smoke("(->> (list 1 2 3) (map inc) (reduce +))"), "9"); }

// Whole file. Covers everything in core.clj end-to-end.
#[test]
fn load_core_full() {
    let src = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    let e = Engine::new();
    e.eval(&src);
}
