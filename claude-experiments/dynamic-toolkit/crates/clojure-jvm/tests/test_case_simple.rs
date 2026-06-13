//! `case` / `case*` conformance. Every expected value below is the
//! pr-str output of real Clojure (`/opt/homebrew/bin/clojure`) for the
//! same form — covering all three test types (`:int`, `:hash-equiv`,
//! `:hash-identity`), grouped constants, the post-switch equivalence
//! check (`2.0` must NOT match the int constant `2`), composite
//! constants, sparse int dispatch, and case-inside-fn.
//!
//! `case` is a clojure.core macro, so the session must load core.

#[test]
fn test_case_simple() {
    use clojure_jvm::lang::compiler::Session;
    let mut sess = Session::new_with_clojure_core();
    for (src, expect) in [
        // :int test type (compact)
        ("(case 2 1 :one 2 :two :default)", ":two"),
        ("(case 99 1 :one 2 :two :default)", ":default"),
        // :hash-identity (all-keyword constants)
        ("(case :b :a 1 :b 2 :c 3 0)", "2"),
        ("(case :zz :a 1 :b 2 :c 3 0)", "0"),
        // :hash-equiv (strings / symbols / composites)
        ("(case \"hello\" \"hi\" :hi \"hello\" :hello :none)", ":hello"),
        ("(case 'sym foo :foo sym :sym :none)", ":sym"),
        ("(case [1 2] [1 2] :v12 [3 4] :v34 :none)", ":v12"),
        ("(case \\b \\a :a \\b :b :none)", ":b"),
        // grouped constants `(1 2 3)` map to one result
        ("(case 5 (1 2 3) :small (4 5 6) :mid :big)", ":mid"),
        // post-switch equivalence: 2.0 hits int bucket 2 but is not = 2
        ("(case 2.0 2 :int-two :other)", ":other"),
        // sparse int dispatch
        ("(case 1000000 1000000 :million :other)", ":million"),
        // hash COLLISION: "Aa" and "BB" share Java String.hashCode (2112),
        // so the macro merges them into one skip-check bucket whose then is
        // a condp — exercises the skip-check path end to end.
        (
            "[(case \"Aa\" \"Aa\" :aa \"BB\" :bb :none) \
              (case \"BB\" \"Aa\" :aa \"BB\" :bb :none) \
              (case \"Ca\" \"Aa\" :aa \"BB\" :bb :none)]",
            "[:aa :bb :none]",
        ),
        // case compiled inside a fn body (Expression/Return contexts)
        (
            "(let [f (fn [x] (case x :up 1 :down -1 0))] [(f :up) (f :down) (f :left)])",
            "[1 -1 0]",
        ),
    ] {
        let result = sess.eval_str(src);
        let pr = clojure_jvm::runtime::pr_str_bits(result);
        assert_eq!(pr, expect, "form: {src}");
    }
}
