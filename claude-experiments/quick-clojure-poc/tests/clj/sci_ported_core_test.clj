(load-file "tests/clj/test_helper.clj")

;; Tests ported from SCI (Small Clojure Interpreter) core_test.cljc
;; Source: https://github.com/babashka/sci

;; ============================================================================
;; core-test: do (SCI line 48)
;; ============================================================================

(testing "do")

(is= (do 1 2 nil) nil "sci_core_do_returns_nil")
(is= (do 1 2 3) 3 "sci_core_do_returns_last")

;; ============================================================================
;; core-test: if and when (SCI line 58)
;; ============================================================================

(testing "if and when")

(is= (if true 10 20) 10 "sci_core_if_true")
(is= (if false 10 20) 20 "sci_core_if_false")
(is= (when true 0 1 2) 2 "sci_core_when_true")
(is= (when false 1) nil "sci_core_when_false_is_nil")

;; ============================================================================
;; core-test: and and or (SCI line 80)
;; ============================================================================

(testing "and and or")

(is= (and false true 0) false "sci_core_and_short_circuit")
(is= (and true true 0) 0 "sci_core_and_returns_last")
(is= (or false false 1) 1 "sci_core_or_returns_first_truthy")
(is= (or false false false) false "sci_core_or_all_false")
(is= (or nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil true) true "sci_core_or_many_nils_then_true")

;; ============================================================================
;; core-test: fn literals (SCI line 87)
;; ============================================================================

(testing "fn literals")

(is= (#(+ 1 %) 1) 2 "sci_core_fn_literal_basic")
(is= (str (map #(+ 1 %) [0 1 2])) "(1 2 3)" "sci_core_fn_literal_with_map")

;; ============================================================================
;; core-test: map, keep (SCI line 93)
;; ============================================================================

(testing "map, keep")

(is= (str (map inc [0 1 2])) "(1 2 3)" "sci_core_map_inc")
(is= (str (keep odd? [0 1 2])) "(false true false)" "sci_core_keep")

;; ============================================================================
;; core-test: calling IFns (SCI line 122)
;; ============================================================================

(testing "calling IFns")

(is= ({:a 1} 2 3) 3 "sci_core_map_as_fn_default")
(is= ({:a 1} :a 3) 1 "sci_core_map_as_fn_found")
(is= ((hash-map :a 1) :a 3) 1 "sci_core_hashmap_as_fn")
(is= (#{:a :b :c} :a) :a "sci_core_set_as_fn")
(is= ((get {:foo identity} :foo) 1) 1 "sci_core_eval_fn_from_map")

;; ============================================================================
;; destructure-test (SCI line 131) -- all ignored
;; ============================================================================

;; ============================================================================
;; let-test (SCI line 162)
;; ============================================================================

(testing "let")

(is= (let [x 1 y (+ x x)] (str "[" x " " y "]")) "[1 2]" "sci_let_basic")
(is= (let [x 1 y (+ x x)] y) 2 "sci_let_basic_values")
(is= (let [x 2] 1 2 3 x) 2 "sci_let_multiple_body")

;; sci_let_nested_shadow: two separate printlns with expected "2\n1"
(let [x 1]
  (is= (let [x 2] x) 2 "sci_let_nested_shadow_inner")
  (is= x 1 "sci_let_nested_shadow_outer"))

;; ============================================================================
;; closure-test (SCI line 185) -- sci_closure_defn_in_let is ignored
;; ============================================================================

(testing "closures")

(is= (let [x 1 y 2] ((fn [] (let [g (fn [] y)] (+ x (g)))))) 3 "sci_closure_nested")

;; ============================================================================
;; fn-literal-test (SCI line 191)
;; ============================================================================

(testing "fn literals")

(is= (str (map #(do %) [1 2 3])) "(1 2 3)" "sci_fn_literal_identity")
(is= (str (map-indexed #(do [%1 %2]) [1 2 3])) "([0 1] [1 2] [2 3])" "sci_fn_literal_map_indexed")
(is= (str (apply #(do %&) [1 2 3])) "(1 2 3)" "sci_fn_literal_rest_args")

;; ============================================================================
;; fn-test (SCI line 199)
;; ============================================================================

(testing "fn")

;; sci_fn_named_recursive: ignored
;; sci_fn_seq_destructure_rest: ignored

(is= (str ((fn [x & xs] xs) 1 2 3)) "(2 3)" "sci_fn_rest_params")

;; sci_fn_rest_destructure_first: ignored

(is= ((fn ([x] x) ([x y] y)) 1) 1 "sci_fn_multi_arity_single")
(is= ((fn ([x] x) ([x y] y)) 1 2) 2 "sci_fn_multi_arity_double")
(is= ((fn ([x & xs] "variadic") ([x] "otherwise")) 1) "otherwise" "sci_fn_variadic_vs_fixed_1arg")
(is= ((fn ([x] "otherwise") ([x & xs] "variadic")) 1 2) "variadic" "sci_fn_variadic_vs_fixed_2arg")
(is= (str (apply (fn [x & xs] xs) 1 2 [3 4])) "(2 3 4)" "sci_fn_apply_with_rest")

;; ============================================================================
;; def-test (SCI line 229)
;; ============================================================================

(testing "def")

(def foo "nice val")
(is= foo "nice val" "sci_def_basic")

(def foo)
(def foo "docstring" 2)
(is= foo 2 "sci_def_with_docstring")

(is= (try (def x 1) x) 1 "sci_def_in_try")
(is= (try (defn x [] 1) (x)) 1 "sci_defn_in_try")

;; ============================================================================
;; defn-test (SCI line 260)
;; ============================================================================

(testing "defn")

(defn foo "increment" [x] (inc x))
(is= (foo 1) 2 "sci_defn_basic")

(defn foo ([x] (inc x)) ([x y] (+ x y)))
(is= (foo 1 2) 3 "sci_defn_multi_arity")

(defn foo [x] (inc x))
(defn foo "dec" [x] (dec x))
(is= (foo 1) 0 "sci_defn_redefine")

;; ============================================================================
;; threading macros (SCI line 448, 1397)
;; ============================================================================

(testing "threading macros")

(is= (-> 3 inc inc inc) 6 "sci_thread_first_basic")
(is= (-> 1 inc inc inc) 4 "sci_thread_first_complex")
(is= (->> ["foo" "baaar" "baaaaaz"] (map count) (apply max)) 7 "sci_thread_last_basic")

;; ============================================================================
;; comment-test (SCI line 628)
;; ============================================================================

(testing "comment")

(is= (comment "anything") nil "sci_comment_returns_nil_1")
(is= (comment anything) nil "sci_comment_returns_nil_2")
(is= (comment 1) nil "sci_comment_returns_nil_3")
(is= (comment (+ 1 2 (* 3 4))) nil "sci_comment_returns_nil_4")

;; ============================================================================
;; recur-test (SCI line 663)
;; ============================================================================

(testing "recur")

(defn hello [x] (if (< x 10000) (recur (inc x)) x))
(is= (hello 0) 10000 "sci_recur_basic")

(is= (str ((fn [& args] (if-let [x (next args)] (recur x) args)) 1 2 3 4)) "(4)" "sci_recur_variadic")
(is= (str ((fn [x & args] (if-let [x (next args)] (recur x x) x)) nil 2 3 4)) "(4)" "sci_recur_variadic_with_fixed")

(defn foo [x & xs] (if (pos? x) (recur (dec x) (rest xs)) xs))
(is= (str (apply foo 10 (range 11))) "(10)" "sci_recur_defn_with_apply")

;; sci_recursion_depth: ignored (named fn self-reference)

;; ============================================================================
;; loop-test (SCI line 736)
;; ============================================================================

(testing "loop")

;; sci_loop_destructure: ignored

(is= (str (loop [l (list 2 1) c (count l)] (if (> c 4) l (recur (conj l (inc c)) (inc c))))) "(5 4 3 2 1)" "sci_loop_conj_list")
(is= (let [x 1] (loop [x (inc x)] x)) 2 "sci_loop_let_shadow")

;; ============================================================================
;; for-test (SCI line 768) -- all ignored
;; ============================================================================

;; ============================================================================
;; cond-test (SCI line 800)
;; ============================================================================

(testing "cond")

(is= (let [x 2] (cond (string? x) 1 (int? x) 2)) 2 "sci_cond_match_int")
(is= (let [x 2] (cond (string? x) 1 :else 2)) 2 "sci_cond_else")

;; ============================================================================
;; condp-test (SCI line 810) -- ignored
;; ============================================================================

;; ============================================================================
;; case-test (SCI line 821) -- all ignored
;; ============================================================================

;; ============================================================================
;; variable-can-have-macro-or-var-name (SCI line 868)
;; ============================================================================

(testing "variable can have macro or var name")

(defn foo [merge] merge)
(is= (foo true) true "sci_var_named_merge")

(defn foo [comment] comment)
(is= (foo true) true "sci_var_named_comment")

;; sci_var_named_fn: ignored

;; ============================================================================
;; try-catch (SCI line 879)
;; ============================================================================

(testing "try-catch")

(is= (try 1 2 3) 3 "sci_try_returns_body")
(is= (try 'hello) 'hello "sci_try_returns_quoted")
(is= (try 1 2 nil) nil "sci_try_nil_in_body")
(is= (try 1 2 nil 1) 1 "sci_try_nil_then_value")

;; ============================================================================
;; letfn-test (SCI line 1073) -- all ignored
;; ============================================================================

;; ============================================================================
;; defn--test (SCI line 1089)
;; ============================================================================

(testing "defn-")

(defn- foo [] 1)
(is= (foo) 1 "sci_defn_private")

;; ============================================================================
;; defonce-test (SCI line 1105) -- ignored
;; ============================================================================

;; ============================================================================
;; ifs-test (SCI line 1250)
;; ============================================================================

(testing "if-let")

(is= (if-let [foo nil] 1 2) 2 "sci_if_let_nil")
(is= (if-let [foo false] 1 2) 2 "sci_if_let_false")
(is= (if-let [foo 42] foo 0) 42 "sci_if_let_truthy")

;; sci_if_some_nil: ignored
;; sci_if_some_false: ignored

;; ============================================================================
;; whens-test (SCI line 1256)
;; ============================================================================

(testing "when-let")

(is= (when-let [foo nil] 1) nil "sci_when_let_nil")
(is= (when-let [foo false] 1) nil "sci_when_let_false")
(is= (when-let [foo 42] foo) 42 "sci_when_let_truthy")

;; sci_when_some_nil: ignored

;; ============================================================================
;; self-ref-test (SCI line 1507) -- all ignored
;; ============================================================================

;; ============================================================================
;; Arithmetic operations
;; ============================================================================

(testing "arithmetic")

(is= (+ 1 2 3) 6 "sci_add_variadic_1")
(is= (+ 1 2 3 4 5) 15 "sci_add_variadic_2")
(is= (* 1 2 3 4) 24 "sci_mul_variadic")
(is= (- 10 3 2) 5 "sci_sub_variadic")
(is= (+ 5) 5 "sci_unary_plus")
(is= (- 5) -5 "sci_unary_minus")
(is= (* 5) 5 "sci_unary_mul")
(is= (+) 0 "sci_zero_arity_plus")
(is= (*) 1 "sci_zero_arity_mul")

(testing "comparisons")

(is= (< 1 2) true "sci_lt")
(is= (< 2 1) false "sci_lt_false")
(is= (<= 1 1) true "sci_lte")
(is= (> 2 1) true "sci_gt")
(is= (>= 2 2) true "sci_gte")
(is= (= 1 1) true "sci_eq")
(is= (not= 1 2) true "sci_not_eq")

(is= (min 1 2 3) 1 "sci_min")
(is= (max 1 2 3) 3 "sci_max")

(is= (inc 0) 1 "sci_inc")
(is= (dec 1) 0 "sci_dec")

(testing "numeric predicates")

(is= (zero? 0) true "sci_zero")
(is= (pos? 1) true "sci_pos")
(is= (neg? -1) true "sci_neg")
(is= (even? 2) true "sci_even")
(is= (odd? 3) true "sci_odd")

(is= (mod 10 3) 1 "sci_mod")
(is= (rem 10 3) 1 "sci_rem")

(is= (abs -5) 5 "sci_abs_neg")
(is= (abs 5) 5 "sci_abs_pos")

;; ============================================================================
;; Bit operations
;; ============================================================================

(testing "bit operations")

(is= (bit-and 255 15) 15 "sci_bit_and")
(is= (bit-or 15 240) 255 "sci_bit_or")
(is= (bit-xor 255 15) 240 "sci_bit_xor")
(is= (bit-not 0) -1 "sci_bit_not")
(is= (bit-shift-left 1 4) 16 "sci_bit_shift_left")
(is= (bit-shift-right 16 4) 1 "sci_bit_shift_right")

;; ============================================================================
;; Logic
;; ============================================================================

(testing "logic")

(is= (not true) false "sci_not_true")
(is= (not false) true "sci_not_false")
(is= (not nil) true "sci_not_nil")
(is= (not 1) false "sci_not_1")

;; ============================================================================
;; Predicates
;; ============================================================================

(testing "predicates")

(is= (nil? nil) true "sci_nil_pred_true")
(is= (nil? 1) false "sci_nil_pred_false")

(is= (true? true) true "sci_true_pred")
(is= (true? 1) false "sci_true_pred_false")
(is= (false? false) true "sci_false_pred")
(is= (false? nil) false "sci_false_pred_nil")

(is= (number? 42) true "sci_number_pred")
(is= (number? :a) false "sci_number_pred_false")
(is= (string? "hi") true "sci_string_pred")
(is= (string? 1) false "sci_string_pred_false")
(is= (keyword? :a) true "sci_keyword_pred")
(is= (symbol? 'a) true "sci_symbol_pred")
(is= (vector? [1 2]) true "sci_vector_pred")
(is= (map? {:a 1}) true "sci_map_pred")
(is= (fn? inc) true "sci_fn_pred")

(is= (integer? 1) true "sci_integer_pred_true")
(is= (integer? 1.0) false "sci_integer_pred_false")
(is= (float? 1.0) true "sci_float_pred_true")
(is= (float? 1) false "sci_float_pred_false")

(is= (coll? [1]) true "sci_coll_pred_vec")
(is= (coll? {:a 1}) true "sci_coll_pred_map")
(is= (coll? 1) false "sci_coll_pred_int")

;; ============================================================================
;; Sequence operations
;; ============================================================================

(testing "sequence operations")

(is= (first [1 2 3]) 1 "sci_first")
(is= (str (rest [1 2 3])) "(2 3)" "sci_rest")
(is= (str (next [1 2 3])) "(2 3)" "sci_next")
(is= (next [1]) nil "sci_next_single")

(is= (nth [10 20 30] 0) 10 "sci_nth_0")
(is= (nth [10 20 30] 2) 30 "sci_nth_2")

(is= (count [1 2 3]) 3 "sci_count_vec")
(is= (count []) 0 "sci_count_empty")
(is= (count {:a 1 :b 2}) 2 "sci_count_map")

(is= (str (conj [1 2] 3)) "[1 2 3]" "sci_conj_vector")
(is= (str (conj '(1 2) 3)) "(3 1 2)" "sci_conj_list")
(is= (str (cons 0 [1 2 3])) "(0 1 2 3)" "sci_cons")
(is= (str (concat [1 2] [3 4] [5])) "(1 2 3 4 5)" "sci_concat")
(is= (str (into [] '(1 2 3))) "[1 2 3]" "sci_into_vector")
(is= (str (reverse [1 2 3])) "(3 2 1)" "sci_reverse")
(is= (str (sort [3 1 2])) "(1 2 3)" "sci_sort")
(is= (str (distinct [1 2 1 3 2 4])) "(1 2 3 4)" "sci_distinct")
(is= (str (interleave [1 2 3] [:a :b :c])) "(1 :a 2 :b 3 :c)" "sci_interleave")
(is= (str (partition 2 [1 2 3 4 5])) "((1 2) (3 4))" "sci_partition")
(is= (last [1 2 3]) 3 "sci_last")
(is= (str (butlast [1 2 3])) "(1 2)" "sci_butlast")
(is= (str (flatten [1 [2 [3 4]] 5])) "(1 2 3 4 5)" "sci_flatten")
(is= (str (vec '(1 2 3))) "[1 2 3]" "sci_vec_from_list")

;; ============================================================================
;; Higher-order functions
;; ============================================================================

(testing "higher-order functions")

(is= (str (map inc [0 1 2])) "(1 2 3)" "sci_map_basic")
(is= (str (map + [1 2 3] [10 20 30])) "(11 22 33)" "sci_map_multiple_colls")
(is= (str (filter even? [1 2 3 4 5 6])) "(2 4 6)" "sci_filter")
(is= (str (remove even? [1 2 3 4 5 6])) "(1 3 5)" "sci_remove")

(is= (reduce + [1 2 3 4 5]) 15 "sci_reduce_no_init")
(is= (reduce + 10 [1 2 3]) 16 "sci_reduce_with_init")

(is= (reduce + 0 (filter odd? (map inc [0 1 2 3 4]))) 9 "sci_reduce_with_map_filter")

(is= (apply + [1 2 3]) 6 "sci_apply_vec")
(is= (apply + 1 2 [3 4]) 10 "sci_apply_mixed")
(is= (apply str ["a" "b" "c"]) "abc" "sci_apply_str")

(is= (some even? [1 3 5 6]) true "sci_some_true")
(is= (some even? [1 3 5]) nil "sci_some_nil")

(is= (every? even? [2 4 6]) true "sci_every_true")
(is= (every? even? [2 3 6]) false "sci_every_false")

(is= (not-every? even? [2 4 6]) false "sci_not_every_false")
(is= (not-every? even? [2 3 6]) true "sci_not_every_true")

(is= (not-any? even? [1 3 5]) true "sci_not_any_true")
(is= (not-any? even? [1 2 5]) false "sci_not_any_false")

(is= (str (mapcat #(vector % (* % %)) [1 2 3])) "(1 1 2 4 3 9)" "sci_mapcat")

(is= (str (take 3 [1 2 3 4 5])) "(1 2 3)" "sci_take")
(is= (str (drop 3 [1 2 3 4 5])) "(4 5)" "sci_drop")

(is= (str (take-while #(< % 4) [1 2 3 4 5])) "(1 2 3)" "sci_take_while")
(is= (str (drop-while #(< % 4) [1 2 3 4 5])) "(4 5)" "sci_drop_while")

;; sci_repeat_infinite: ignored

(is= (str (repeat 3 :a)) "(:a :a :a)" "sci_repeat_bounded")

;; sci_iterate: ignored

(is= (str (range 5)) "(0 1 2 3 4)" "sci_range_5")
(is= (str (range 2 5)) "(2 3 4)" "sci_range_2_5")
(is= (str (range 0 10 3)) "(0 3 6 9)" "sci_range_step")

;; ============================================================================
;; Collection operations (maps, sets)
;; ============================================================================

(testing "collection operations")

(is= (get {:a 1 :b 2} :a) 1 "sci_get_found")
(is= (get {:a 1} :b) nil "sci_get_not_found")
(is= (get {:a 1} :b :default) :default "sci_get_default")

(is= (contains? {:a 1} :a) true "sci_contains_true")
(is= (contains? {:a 1} :b) false "sci_contains_false")

(is= (get (assoc {:a 1} :b 2) :b) 2 "sci_assoc_get")
(is= (count (assoc {:a 1} :b 2)) 2 "sci_assoc_count")

(is= (contains? (dissoc {:a 1 :b 2} :b) :b) false "sci_dissoc_contains")
(is= (count (dissoc {:a 1 :b 2} :b)) 1 "sci_dissoc_count")

(is= (get (merge {:a 1} {:b 2} {:c 3}) :c) 3 "sci_merge_get")
(is= (count (merge {:a 1} {:b 2} {:c 3})) 3 "sci_merge_count")

(is= (get (update {:a 1} :a inc) :a) 2 "sci_update")

(is= (get-in {:a {:b {:c 42}}} [:a :b :c]) 42 "sci_get_in")
(is= (get-in (assoc-in {} [:a :b :c] 42) [:a :b :c]) 42 "sci_assoc_in")
(is= (get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b]) 2 "sci_update_in")

(is= (count (select-keys {:a 1 :b 2 :c 3} [:a :c])) 2 "sci_select_keys_count")
(is= (get (select-keys {:a 1 :b 2 :c 3} [:a :c]) :a) 1 "sci_select_keys_get")

(is= (count (keys {:a 1 :b 2})) 2 "sci_keys_count")
(is= (count (vals {:a 1 :b 2})) 2 "sci_vals_count")

(is= (get (zipmap [:a :b :c] [1 2 3]) :b) 2 "sci_zipmap_get")
(is= (count (zipmap [:a :b :c] [1 2 3])) 3 "sci_zipmap_count")

(is= (count (set [1 2 2 3 3 3])) 3 "sci_set_from_vector_count")
(is= (contains? (set [1 2 3]) 2) true "sci_set_from_vector_contains")

(is= (count (disj #{1 2 3} 2)) 2 "sci_disj_count")
(is= (contains? (disj #{1 2 3} 2) 2) false "sci_disj_contains")

;; ============================================================================
;; Keyword operations
;; ============================================================================

(testing "keyword as fn")

(is= (:a {:a 42 :b 99}) 42 "sci_keyword_as_fn_found")
(is= (:c {:a 1}) nil "sci_keyword_as_fn_nil")
(is= (:c {:a 1} :default) :default "sci_keyword_as_fn_default")

;; ============================================================================
;; String operations
;; ============================================================================

(testing "string operations")

(is= (str "hello" " " "world") "hello world" "sci_str_concat")
(is= (str 1 2 3) "123" "sci_str_numbers")
(is= (str nil) "" "sci_str_nil")

(is= (name :foo) "foo" "sci_name_keyword")
(is= (name 'bar) "bar" "sci_name_symbol")

;; sci_str_upper_lower: ignored
;; sci_str_join: ignored
;; sci_str_includes: ignored

;; ============================================================================
;; Empty / seq
;; ============================================================================

(testing "empty / seq")

(is= (empty? []) true "sci_empty_vec")
(is= (empty? [1]) false "sci_empty_vec_nonempty")
(is= (empty? nil) true "sci_empty_nil")

(is= (seq []) nil "sci_seq_empty_vec")
(is= (seq nil) nil "sci_seq_nil")

;; ============================================================================
;; Identity, constantly, comp, partial, complement, juxt
;; ============================================================================

(testing "higher-order helpers")

(is= (identity 42) 42 "sci_identity")
(is= ((constantly 5) 1 2 3) 5 "sci_constantly")
(is= ((comp inc inc inc) 0) 3 "sci_comp")
(is= ((partial + 10) 5) 15 "sci_partial")
(is= ((complement nil?) 1) true "sci_complement_truthy")
(is= ((complement nil?) nil) false "sci_complement_nil")

(is= (nth ((juxt inc dec) 1) 0) 2 "sci_juxt_inc")
(is= (nth ((juxt inc dec) 1) 1) 0 "sci_juxt_dec")

;; ============================================================================
;; Do form
;; ============================================================================

(testing "do form")

(is= (do 1 2 3) 3 "sci_do_returns_last")

;; sci_do_side_effects: tests println output, converted to underlying values
(is= (+ 1 0) 1 "sci_do_side_effects_1")
(is= (+ 1 1) 2 "sci_do_side_effects_2")

;; ============================================================================
;; Frequencies, group-by
;; ============================================================================

(testing "frequencies and group-by")

(is= (get (frequencies [:a :b :a :c :b :a]) :a) 3 "sci_frequencies_a")
(is= (get (frequencies [:a :b :a :c :b :a]) :b) 2 "sci_frequencies_b")
(is= (get (frequencies [:a :b :a :c :b :a]) :c) 1 "sci_frequencies_c")

(is= (count (get (group-by odd? [1 2 3 4 5]) true)) 3 "sci_group_by_odd_true")
(is= (count (get (group-by odd? [1 2 3 4 5]) false)) 2 "sci_group_by_odd_false")

;; ============================================================================
;; While, atom (SCI line 1273) -- ignored
;; ============================================================================

;; ============================================================================
;; Delay (SCI line 1086) -- ignored
;; ============================================================================

;; ============================================================================
;; Trampoline (SCI line 645) -- ignored
;; ============================================================================

(test-summary)
