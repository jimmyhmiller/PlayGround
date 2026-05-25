//! Comprehensive smoke tests for core.clj. Each test is one short
//! Clojure expression compared against an expected printed result.
//!
//! Failures here are documentation, not regressions — they map the
//! current state of the runtime, so we know what works and what
//! doesn't.

use clojure::Engine;

const CORE_PATH: &str =
    "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/quick-clojure-poc/src/clojure/core.clj";

fn smoke(src: &str) -> String {
    let e = Engine::new();
    let core = std::fs::read_to_string(CORE_PATH).expect("can't read core.clj");
    e.eval(&core);
    let v = e.eval(src);
    e.print(v)
}

macro_rules! smoke_eq {
    ($name:ident, $src:expr, $expected:expr) => {
        #[test]
        fn $name() {
            assert_eq!(smoke($src), $expected);
        }
    };
}

// ============================================================================
// Arithmetic
// ============================================================================
smoke_eq!(arith_add_2, "(+ 1 2)", "3");
smoke_eq!(arith_add_3, "(+ 1 2 3)", "6");
smoke_eq!(arith_add_0, "(+)", "0");
smoke_eq!(arith_add_1, "(+ 5)", "5");
smoke_eq!(arith_sub_2, "(- 10 3)", "7");
smoke_eq!(arith_sub_3, "(- 10 3 2)", "5");
smoke_eq!(arith_sub_1, "(- 5)", "-5");
smoke_eq!(arith_mul_2, "(* 3 4)", "12");
smoke_eq!(arith_mul_3, "(* 2 3 4)", "24");
smoke_eq!(arith_mul_0, "(*)", "1");
smoke_eq!(arith_div_int, "(/ 10 2)", "5");
smoke_eq!(arith_quot, "(quot 7 2)", "3");
smoke_eq!(arith_rem, "(rem 7 2)", "1");
smoke_eq!(arith_mod_pos, "(mod 7 3)", "1");
smoke_eq!(arith_mod_neg, "(mod -1 3)", "2"); // mod truncates toward neg-inf
smoke_eq!(arith_inc, "(inc 41)", "42");
smoke_eq!(arith_dec, "(dec 5)", "4");
smoke_eq!(arith_abs_pos, "(abs 5)", "5");
smoke_eq!(arith_abs_neg, "(abs -5)", "5");
smoke_eq!(arith_min_2, "(min 3 7)", "3");
smoke_eq!(arith_min_3, "(min 5 2 8)", "2");
smoke_eq!(arith_max_2, "(max 3 7)", "7");
smoke_eq!(arith_max_3, "(max 5 2 8)", "8");
smoke_eq!(arith_zero_p_t, "(zero? 0)", "true");
smoke_eq!(arith_zero_p_f, "(zero? 1)", "false");
smoke_eq!(arith_pos_p_t, "(pos? 5)", "true");
smoke_eq!(arith_pos_p_f, "(pos? -5)", "false");
smoke_eq!(arith_neg_p_t, "(neg? -5)", "true");
smoke_eq!(arith_neg_p_f, "(neg? 5)", "false");
smoke_eq!(arith_even_p_t, "(even? 4)", "true");
smoke_eq!(arith_even_p_f, "(even? 5)", "false");
smoke_eq!(arith_odd_p_t, "(odd? 5)", "true");
smoke_eq!(arith_odd_p_f, "(odd? 4)", "false");

// ============================================================================
// Comparison
// ============================================================================
smoke_eq!(cmp_eq_2_t, "(= 1 1)", "true");
smoke_eq!(cmp_eq_2_f, "(= 1 2)", "false");
smoke_eq!(cmp_eq_3_t, "(= 1 1 1)", "true");
smoke_eq!(cmp_eq_3_f, "(= 1 1 2)", "false");
smoke_eq!(cmp_eq_strs, "(= \"a\" \"a\")", "true");
smoke_eq!(cmp_eq_kws, "(= :a :a)", "true");
smoke_eq!(cmp_lt, "(< 1 2)", "true");
smoke_eq!(cmp_lt_3, "(< 1 2 3)", "true");
smoke_eq!(cmp_lt_3_f, "(< 1 3 2)", "false");
smoke_eq!(cmp_gt, "(> 3 2)", "true");
smoke_eq!(cmp_le, "(<= 1 1 2)", "true");
smoke_eq!(cmp_ge, "(>= 3 3 2)", "true");
smoke_eq!(cmp_not_eq_t, "(not= 1 2)", "true");
smoke_eq!(cmp_not_eq_f, "(not= 1 1)", "false");

// ============================================================================
// Predicates
// ============================================================================
smoke_eq!(pred_nil_p_t, "(nil? nil)", "true");
smoke_eq!(pred_nil_p_f, "(nil? 0)", "false");
smoke_eq!(pred_some_p_t, "(some? 0)", "true");
smoke_eq!(pred_some_p_f, "(some? nil)", "false");
smoke_eq!(pred_true_p_t, "(true? true)", "true");
smoke_eq!(pred_true_p_f, "(true? 1)", "false");
smoke_eq!(pred_false_p_t, "(false? false)", "true");
smoke_eq!(pred_false_p_f, "(false? nil)", "false");
smoke_eq!(pred_int_t, "(integer? 5)", "true");
smoke_eq!(pred_int_f, "(integer? 1.5)", "false");
smoke_eq!(pred_str_t, "(string? \"a\")", "true");
smoke_eq!(pred_str_f, "(string? 1)", "false");
smoke_eq!(pred_kw_t, "(keyword? :a)", "true");
smoke_eq!(pred_kw_f, "(keyword? \"a\")", "false");
smoke_eq!(pred_sym_t, "(symbol? (quote a))", "true");
smoke_eq!(pred_sym_f, "(symbol? :a)", "false");
smoke_eq!(pred_num_t, "(number? 5)", "true");
smoke_eq!(pred_num_f, "(number? :a)", "false");
smoke_eq!(pred_fn_t, "(fn? inc)", "true");
smoke_eq!(pred_fn_f, "(fn? 5)", "false");

// ============================================================================
// Lists
// ============================================================================
smoke_eq!(list_empty, "(list)", "()");
smoke_eq!(list_one, "(list 1)", "(1)");
smoke_eq!(list_three, "(list 1 2 3)", "(1 2 3)");
smoke_eq!(list_first, "(first (list 1 2 3))", "1");
smoke_eq!(list_rest, "(first (rest (list 1 2 3)))", "2");
smoke_eq!(list_next, "(first (next (list 1 2 3)))", "2");
smoke_eq!(list_count, "(count (list 1 2 3 4))", "4");
smoke_eq!(list_count_empty, "(count (list))", "0");
smoke_eq!(list_nth_0, "(nth (list :a :b :c) 0)", ":a");
smoke_eq!(list_nth_1, "(nth (list :a :b :c) 1)", ":b");
smoke_eq!(list_cons_to_list, "(cons 0 (list 1 2))", "(0 1 2)");
smoke_eq!(list_cons_to_nil, "(cons 1 nil)", "(1)");
smoke_eq!(list_p_t, "(list? (list 1))", "true");
smoke_eq!(list_p_f_vec, "(list? [1])", "false");
smoke_eq!(list_seq, "(seq (list 1 2 3))", "(1 2 3)");
smoke_eq!(list_seq_empty, "(seq (list))", "nil");
smoke_eq!(list_conj_to_front, "(conj (list 2 3) 1)", "(1 2 3)");

// ============================================================================
// Vectors
// ============================================================================
smoke_eq!(vec_literal, "[1 2 3]", "[1 2 3]");
smoke_eq!(vec_empty, "[]", "[]");
smoke_eq!(vec_count, "(count [1 2 3 4 5])", "5");
smoke_eq!(vec_first, "(first [:a :b :c])", ":a");
smoke_eq!(vec_rest, "(first (rest [:a :b :c]))", ":b");
smoke_eq!(vec_nth, "(nth [10 20 30] 1)", "20");
smoke_eq!(vec_nth_default, "(nth [1 2 3] 99 :missing)", ":missing");
smoke_eq!(vec_get, "(get [10 20 30] 1)", "20");
smoke_eq!(vec_seq, "(first (seq [1 2 3]))", "1");
smoke_eq!(vec_conj, "(conj [1 2] 3)", "[1 2 3]");
smoke_eq!(vec_conj_multi, "(conj [1] 2 3 4)", "[1 2 3 4]");
smoke_eq!(vec_vec_fn, "(vec (list 1 2 3))", "[1 2 3]");
smoke_eq!(vec_vector_fn, "(vector :a :b :c)", "[:a :b :c]");
smoke_eq!(vec_vector_p_t, "(vector? [1])", "true");
smoke_eq!(vec_vector_p_f, "(vector? (list 1))", "false");
smoke_eq!(vec_assoc, "(assoc [10 20 30] 1 99)", "[10 99 30]");
smoke_eq!(vec_index_call, "([10 20 30] 1)", "20"); // vector as fn

// ============================================================================
// Maps
// ============================================================================
smoke_eq!(map_literal_empty, "{}", "{}");
smoke_eq!(map_get, "(get {:a 1 :b 2} :a)", "1");
smoke_eq!(map_get_missing, "(get {:a 1} :b)", "nil");
smoke_eq!(map_get_default, "(get {:a 1} :b :missing)", ":missing");
smoke_eq!(map_assoc, "(get (assoc {} :x 1) :x)", "1");
smoke_eq!(map_assoc_pair, "(get (assoc {} :x 1 :y 2) :y)", "2");
smoke_eq!(map_count, "(count {:a 1 :b 2 :c 3})", "3");
smoke_eq!(map_dissoc, "(get (dissoc {:a 1 :b 2} :a) :a)", "nil");
smoke_eq!(map_dissoc_keep, "(get (dissoc {:a 1 :b 2} :a) :b)", "2");
smoke_eq!(map_contains_t, "(contains? {:a 1} :a)", "true");
smoke_eq!(map_contains_f, "(contains? {:a 1} :b)", "false");
smoke_eq!(map_keys_count, "(count (keys {:a 1 :b 2 :c 3}))", "3");
smoke_eq!(map_vals_count, "(count (vals {:a 1 :b 2}))", "2");
smoke_eq!(map_p_t, "(map? {})", "true");
smoke_eq!(map_p_f, "(map? [])", "false");
smoke_eq!(map_kw_call, "(:a {:a 1 :b 2})", "1"); // keyword as fn
smoke_eq!(map_call_kw, "({:a 1 :b 2} :b)", "2"); // map as fn
smoke_eq!(map_merge, "(get (merge {:a 1} {:b 2}) :b)", "2");
smoke_eq!(map_select_keys, "(count (select-keys {:a 1 :b 2 :c 3} [:a :b]))", "2");
smoke_eq!(map_update, "(get (update {:a 1} :a inc) :a)", "2");

// ============================================================================
// Sets
// ============================================================================
smoke_eq!(set_literal, "(count #{1 2 3})", "3");
smoke_eq!(set_contains_t, "(contains? #{1 2 3} 2)", "true");
smoke_eq!(set_contains_f, "(contains? #{1 2 3} 5)", "false");
smoke_eq!(set_call, "(#{1 2 3} 2)", "2");
smoke_eq!(set_call_missing, "(#{1 2 3} 5)", "nil");
smoke_eq!(set_p_t, "(set? #{1})", "true");
smoke_eq!(set_p_f, "(set? [1])", "false");
smoke_eq!(set_conj, "(count (conj #{1 2} 3))", "3");
smoke_eq!(set_disj, "(count (disj #{1 2 3} 2))", "2");

// ============================================================================
// Sequence operations
// ============================================================================
smoke_eq!(seq_map, "(first (map inc [1 2 3]))", "2");
smoke_eq!(seq_map_count, "(count (map inc [1 2 3 4]))", "4");
smoke_eq!(seq_map_2coll, "(first (map + [1 2 3] [10 20 30]))", "11");
smoke_eq!(seq_filter, "(first (filter even? [1 2 3 4]))", "2");
smoke_eq!(seq_filter_count, "(count (filter even? [1 2 3 4 5 6]))", "3");
smoke_eq!(seq_remove, "(first (remove even? [1 2 3 4]))", "1");
smoke_eq!(seq_reduce, "(reduce + 0 [1 2 3 4 5])", "15");
smoke_eq!(seq_reduce_no_init, "(reduce + [1 2 3 4 5])", "15");
smoke_eq!(seq_reduce_max, "(reduce max [3 1 4 1 5 9 2 6])", "9");
smoke_eq!(seq_take, "(count (take 3 [1 2 3 4 5]))", "3");
smoke_eq!(seq_take_overflow, "(count (take 10 [1 2 3]))", "3");
smoke_eq!(seq_drop, "(first (drop 2 [1 2 3 4 5]))", "3");
smoke_eq!(seq_drop_count, "(count (drop 2 [1 2 3 4 5]))", "3");
smoke_eq!(seq_take_while, "(count (take-while pos? [1 2 3 -1 4]))", "3");
smoke_eq!(seq_drop_while, "(first (drop-while pos? [1 2 -3 4]))", "-3");
smoke_eq!(seq_concat, "(count (concat [1 2] [3 4 5]))", "5");
smoke_eq!(seq_concat_three, "(count (concat [1] [2 3] [4]))", "4");
smoke_eq!(seq_reverse, "(first (reverse [1 2 3]))", "3");
smoke_eq!(seq_into_vec, "(count (into [] [1 2 3]))", "3");
smoke_eq!(seq_into_list, "(count (into (list) [1 2 3]))", "3");
smoke_eq!(seq_repeat, "(count (take 5 (repeat :x)))", "5");
smoke_eq!(seq_repeat_n, "(count (repeat 4 :x))", "4");
smoke_eq!(seq_range, "(count (range 5))", "5");
smoke_eq!(seq_range_2, "(count (range 2 7))", "5");
smoke_eq!(seq_range_step, "(count (range 0 10 2))", "5");
smoke_eq!(seq_iterate, "(first (drop 3 (iterate inc 0)))", "3");
smoke_eq!(seq_partition, "(count (partition 2 [1 2 3 4 5 6]))", "3");
smoke_eq!(seq_partition_step, "(count (partition 2 1 [1 2 3 4]))", "3");
smoke_eq!(seq_interleave, "(count (interleave [1 2 3] [:a :b :c]))", "6");
smoke_eq!(seq_interpose, "(count (interpose 0 [1 2 3]))", "5");
smoke_eq!(seq_mapcat, "(count (mapcat reverse [[3 2 1] [6 5 4]]))", "6");
smoke_eq!(seq_apply_str, "(apply str [\"a\" \"b\" \"c\"])", "\"abc\"");
smoke_eq!(seq_apply_plus, "(apply + [1 2 3 4 5])", "15");
smoke_eq!(seq_some, "(some even? [1 3 5 4])", "true");
smoke_eq!(seq_every_t, "(every? pos? [1 2 3])", "true");
smoke_eq!(seq_every_f, "(every? pos? [1 -2 3])", "false");
smoke_eq!(seq_distinct, "(count (distinct [1 1 2 2 3 3]))", "3");
smoke_eq!(seq_sort, "(first (sort [3 1 2]))", "1");
smoke_eq!(seq_group_by, "(count (group-by even? [1 2 3 4 5]))", "2");
smoke_eq!(seq_frequencies, "(get (frequencies [1 1 2 3 3 3]) 3)", "3");
smoke_eq!(seq_zipmap_count, "(count (zipmap [:a :b :c] [1 2 3]))", "3");
smoke_eq!(seq_last, "(last [1 2 3])", "3");
smoke_eq!(seq_butlast, "(count (butlast [1 2 3 4]))", "3");
smoke_eq!(seq_empty_p_t, "(empty? [])", "true");
smoke_eq!(seq_empty_p_f, "(empty? [1])", "false");
smoke_eq!(seq_not_empty, "(not-empty [1 2])", "[1 2]");
smoke_eq!(seq_not_empty_nil, "(not-empty [])", "nil");

// ============================================================================
// Strings
// ============================================================================
smoke_eq!(str_empty, "(str)", "\"\"");
smoke_eq!(str_one, "(str \"hi\")", "\"hi\"");
smoke_eq!(str_concat, "(str \"a\" \"b\" \"c\")", "\"abc\"");
smoke_eq!(str_int, "(str 42)", "\"42\"");
smoke_eq!(str_mixed, "(str \"a\" 1 :b)", "\"a1:b\"");
smoke_eq!(str_nil, "(str nil)", "\"\"");
smoke_eq!(str_count, "(count \"hello\")", "5");
smoke_eq!(str_subs_2, "(subs \"hello\" 2)", "\"llo\"");
smoke_eq!(str_subs_3, "(subs \"hello\" 1 4)", "\"ell\"");
smoke_eq!(str_upper, "(clojure.string/upper-case \"hi\")", "\"HI\"");
smoke_eq!(str_lower, "(clojure.string/lower-case \"HI\")", "\"hi\"");
smoke_eq!(str_includes, "(clojure.string/includes? \"hello\" \"ell\")", "true");

// ============================================================================
// Keywords / Symbols
// ============================================================================
smoke_eq!(kw_literal, ":foo", ":foo");
smoke_eq!(kw_name, "(name :foo)", "\"foo\"");
smoke_eq!(kw_keyword_fn, "(keyword \"bar\")", ":bar");
smoke_eq!(sym_literal, "(quote x)", "x");
smoke_eq!(sym_name, "(name (quote x))", "\"x\"");
smoke_eq!(sym_symbol_fn, "(symbol \"x\")", "x");

// ============================================================================
// Atoms
// ============================================================================
smoke_eq!(atom_create_deref, "(deref (atom 5))", "5");
smoke_eq!(atom_at_deref, "@(atom 5)", "5");
smoke_eq!(atom_reset, "(let [a (atom 1)] (reset! a 99) @a)", "99");
smoke_eq!(atom_swap, "(let [a (atom 1)] (swap! a inc) @a)", "2");
smoke_eq!(atom_swap_args, "(let [a (atom 10)] (swap! a + 5) @a)", "15");

// ============================================================================
// Loop / recur
// ============================================================================
smoke_eq!(loop_simple, "(loop [i 0] (if (< i 5) (recur (inc i)) i))", "5");
smoke_eq!(loop_acc, "(loop [i 0 acc 0] (if (< i 10) (recur (inc i) (+ acc i)) acc))", "45");
smoke_eq!(loop_dotimes, "(let [a (atom 0)] (dotimes [i 5] (swap! a + i)) @a)", "10");
smoke_eq!(loop_while, "(let [a (atom 0)] (while (< @a 5) (swap! a inc)) @a)", "5");

// ============================================================================
// Try / catch / throw
// ============================================================================
smoke_eq!(try_no_throw, "(try 1 (catch :x e e))", "1");
smoke_eq!(try_catch, "(try (throw :boom) (catch :x e :handled))", ":handled");
smoke_eq!(try_finally, "(let [a (atom 0)] (try 1 (finally (reset! a 9))) @a)", "9");

// ============================================================================
// fn / multi-arity / variadic / closure
// ============================================================================
smoke_eq!(fn_anon, "((fn [x] (* x x)) 5)", "25");
smoke_eq!(fn_anon_short, "(#(* % %) 5)", "25");
smoke_eq!(fn_multi_arity_a, "((fn ([x] x) ([x y] (+ x y))) 7)", "7");
smoke_eq!(fn_multi_arity_b, "((fn ([x] x) ([x y] (+ x y))) 7 8)", "15");
smoke_eq!(fn_variadic, "((fn [& args] (count args)) 1 2 3)", "3");
smoke_eq!(fn_closure, "(let [f (let [x 5] (fn [] x))] (f))", "5");
smoke_eq!(fn_recur_self, "(letfn [(f [n] (if (zero? n) 0 (+ n (f (dec n)))))] (f 5))", "15");

// ============================================================================
// let / when / cond / case / condp
// ============================================================================
smoke_eq!(let_basic, "(let [x 5] x)", "5");
smoke_eq!(let_seq, "(let [x 5 y 7] (+ x y))", "12");
smoke_eq!(let_nested, "(let [x 5] (let [y (* x 2)] y))", "10");
smoke_eq!(let_destructure_vec, "(let [[a b] [1 2]] (+ a b))", "3");
smoke_eq!(let_destructure_map, "(let [{:keys [a b]} {:a 1 :b 2}] (+ a b))", "3");
smoke_eq!(let_destructure_amp, "(let [[a & rs] [1 2 3 4]] (count rs))", "3");
smoke_eq!(when_t, "(when true 42)", "42");
smoke_eq!(when_f, "(when false 42)", "nil");
smoke_eq!(when_not_t, "(when-not false 42)", "42");
smoke_eq!(if_not_t, "(if-not false :a :b)", ":a");
smoke_eq!(cond_first, "(cond true :a :else :b)", ":a");
smoke_eq!(cond_else, "(cond false :a :else :b)", ":b");
smoke_eq!(cond_three, "(cond false :a true :b :else :c)", ":b");
smoke_eq!(case_match, "(case 2 1 :one 2 :two :other)", ":two");
smoke_eq!(case_default, "(case 99 1 :one 2 :two :other)", ":other");
smoke_eq!(condp_match, "(condp = 2 1 :one 2 :two :other)", ":two");
smoke_eq!(if_let_t, "(if-let [x 5] x :no)", "5");
smoke_eq!(if_let_f, "(if-let [x nil] x :no)", ":no");
smoke_eq!(when_let_t, "(when-let [x 5] (inc x))", "6");
smoke_eq!(when_let_f, "(when-let [x nil] (inc x))", "nil");

// ============================================================================
// Logical
// ============================================================================
smoke_eq!(and_all_true, "(and 1 2 3)", "3");
smoke_eq!(and_short, "(and 1 nil 3)", "nil");
smoke_eq!(and_empty, "(and)", "true");
smoke_eq!(or_first_true, "(or 1 2 3)", "1");
smoke_eq!(or_skip_nil, "(or nil nil 3)", "3");
smoke_eq!(or_all_nil, "(or nil nil)", "nil");
smoke_eq!(or_empty, "(or)", "nil");
smoke_eq!(not_t, "(not false)", "true");
smoke_eq!(not_f, "(not 1)", "false");

// ============================================================================
// Threading
// ============================================================================
smoke_eq!(thread_first, "(-> 1 inc inc inc)", "4");
smoke_eq!(thread_first_form, "(-> 5 (- 1) (* 2))", "8");
smoke_eq!(thread_last, "(->> [1 2 3 4 5] (filter even?) (reduce +))", "6");
smoke_eq!(thread_some, "(some-> 1 inc inc)", "3");
smoke_eq!(thread_some_nil, "(some-> nil inc)", "nil");
smoke_eq!(thread_cond, "(cond-> 1 true inc false dec)", "2");
smoke_eq!(thread_as, "(as-> 1 x (+ x 1) (* x 2))", "4");

// ============================================================================
// Apply / partial / comp / juxt / constantly / identity
// ============================================================================
smoke_eq!(apply_fn, "(apply + (list 1 2 3 4))", "10");
smoke_eq!(apply_args, "(apply + 1 2 (list 3 4))", "10");
smoke_eq!(partial_basic, "((partial + 10) 5)", "15");
smoke_eq!(partial_two, "((partial + 1 2) 3 4)", "10");
smoke_eq!(comp_two, "((comp inc inc) 5)", "7");
smoke_eq!(comp_three, "((comp str inc inc) 5)", "\"7\"");
smoke_eq!(juxt_fns, "(count ((juxt inc dec) 5))", "2");
smoke_eq!(constantly_fn, "((constantly 99) 1 2 3)", "99");
smoke_eq!(identity_fn, "(identity 42)", "42");
smoke_eq!(complement_fn, "((complement even?) 3)", "true");

// ============================================================================
// defprotocol / deftype / defrecord
// ============================================================================
smoke_eq!(
    proto_basic,
    "(defprotocol IGreet (greet [this])) \
     (deftype Hi [name] IGreet (greet [this] (str \"hi \" (.-name this)))) \
     (greet (Hi. \"world\"))",
    "\"hi world\""
);
smoke_eq!(
    proto_field,
    "(deftype P [x y]) (.-x (P. 3 4))",
    "3"
);
smoke_eq!(
    proto_extend_existing,
    "(defprotocol IFoo (foo [this])) \
     (extend-type Number IFoo (foo [this] (+ this 1))) \
     (foo 10)",
    "11"
);
smoke_eq!(
    instance_check,
    "(deftype Box [v]) (instance? Box (Box. 1))",
    "true"
);
smoke_eq!(
    satisfies_check,
    "(defprotocol IBar (bar [this])) \
     (deftype B [] IBar (bar [this] :ok)) \
     (satisfies? IBar (B.))",
    "true"
);

// ============================================================================
// defmulti / defmethod
// ============================================================================
smoke_eq!(
    multimethod_basic,
    "(defmulti shape :kind) \
     (defmethod shape :circle [s] :round) \
     (defmethod shape :square [s] :corners) \
     (shape {:kind :circle})",
    ":round"
);

// ============================================================================
// Quasi-quote
// ============================================================================
smoke_eq!(qq_simple, "`(1 2 3)", "(1 2 3)");
smoke_eq!(qq_unquote, "(let [x 5] `(a ~x b))", "(a 5 b)");
smoke_eq!(qq_splice, "(let [xs [1 2 3]] `(a ~@xs b))", "(a 1 2 3 b)");

// ============================================================================
// Meta (we strip metadata, so these should... ?)
// ============================================================================
smoke_eq!(meta_nil, "(meta 5)", "nil");
smoke_eq!(meta_with_meta, "(meta (with-meta [1 2] {:a 1}))", "{:a 1}");

// ============================================================================
// Lazy seqs
// ============================================================================
smoke_eq!(lazy_take, "(count (take 3 (iterate inc 0)))", "3");
smoke_eq!(lazy_first, "(first (iterate inc 100))", "100");
smoke_eq!(lazy_realized_three, "(reduce + (take 5 (iterate inc 1)))", "15");
smoke_eq!(lazy_seq_macro, "(first (lazy-seq [42]))", "42");

// ============================================================================
// Records (defrecord — we use deftype with positional fields)
// ============================================================================
smoke_eq!(record_basic, "(deftype R [a b]) (.-a (R. 10 20))", "10");
smoke_eq!(
    record_method_uses_field,
    "(defprotocol IPair (sum [this])) \
     (deftype Pair [a b] IPair (sum [this] (+ a b))) \
     (sum (Pair. 3 4))",
    "7"
);

// ============================================================================
// Misc: get-in, update-in, assoc-in
// ============================================================================
smoke_eq!(getin_2, "(get-in {:a {:b 5}} [:a :b])", "5");
smoke_eq!(getin_missing, "(get-in {:a {:b 5}} [:a :c])", "nil");
smoke_eq!(getin_default, "(get-in {:a {:b 5}} [:a :c] :no)", ":no");
smoke_eq!(associn, "(get-in (assoc-in {} [:a :b] 99) [:a :b])", "99");
smoke_eq!(updatein, "(get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b])", "2");

// ============================================================================
// First-class fns / higher order
// ============================================================================
smoke_eq!(hof_pass_inc, "(let [f inc] (f 5))", "6");
smoke_eq!(hof_returned_fn, "((let [f (fn [n] (fn [x] (+ x n)))] (f 10)) 5)", "15");
smoke_eq!(hof_map_anon, "(first (map (fn [x] (* x x)) [3 4 5]))", "9");
