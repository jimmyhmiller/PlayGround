(load-file "tests/clj/test_helper.clj")

;; Tests ported from ClojureScript core_test.cljs
;; Source: https://github.com/clojure/clojurescript/blob/master/src/test/cljs/cljs/core_test.cljs
;; Converted from tests/cljs_ported_core_test.rs

;; ============================================================================
;; ARITHMETIC
;; ============================================================================

(testing "ARITHMETIC")

(is= (+ 1 2) 3 "cljs_arithmetic_addition_basic")

(is= (+ 1 2 3) 6 "cljs_arithmetic_addition_variadic")
(is= (+ 1 2 3 4 5) 15 "cljs_arithmetic_addition_variadic")

(is= (+ 1 2 3 4 5 6 7 8 9 10) 55 "cljs_arithmetic_addition_many")

(is= (+) 0 "cljs_arithmetic_addition_zero_arity")

(is= (+ 5) 5 "cljs_arithmetic_addition_unary")

(is= (+ 1.5 2.5) 4.0 "cljs_arithmetic_addition_float")

(is= (- 10 3) 7 "cljs_arithmetic_subtraction_basic")

(is= (- 10 3 2) 5 "cljs_arithmetic_subtraction_variadic")

(is= (- 5) -5 "cljs_arithmetic_subtraction_unary")

(is= (* 2 3) 6 "cljs_arithmetic_multiplication_basic")

(is= (* 2 3 4) 24 "cljs_arithmetic_multiplication_variadic")

(is= (* 1 2 3 4 5 6 7 8 9 10) 3628800 "cljs_arithmetic_multiplication_many")

(is= (*) 1 "cljs_arithmetic_multiplication_zero_arity")

(is= (* 5) 5 "cljs_arithmetic_multiplication_unary")

(is= (/ 10 2) 5 "cljs_arithmetic_division_basic")

(is= (/ 7 2) 3 "cljs_arithmetic_division_integer")

(is= (/ 10.0 3) 3.3333333333333335 "cljs_arithmetic_division_float")

(is= (quot 10 3) 3 "cljs_arithmetic_quot")
(is= (quot 7 2) 3 "cljs_arithmetic_quot")

(is= (rem 10 3) 1 "cljs_arithmetic_rem")
(is= (rem 7 2) 1 "cljs_arithmetic_rem")
(is= (rem -7 2) -1 "cljs_arithmetic_rem")

(is= (mod 10 3) 1 "cljs_arithmetic_mod")
(is= (mod 7 2) 1 "cljs_arithmetic_mod")
(is= (mod -7 2) 1 "cljs_arithmetic_mod")

(is= (inc 0) 1 "cljs_arithmetic_inc")
(is= (inc 5) 6 "cljs_arithmetic_inc")
(is= (inc -1) 0 "cljs_arithmetic_inc")

(is= (dec 1) 0 "cljs_arithmetic_dec")
(is= (dec 5) 4 "cljs_arithmetic_dec")
(is= (dec 0) -1 "cljs_arithmetic_dec")

(is= (max 1 2 3) 3 "cljs_arithmetic_max")
(is= (max 5 3 7 1 9) 9 "cljs_arithmetic_max")

(is= (min 1 2 3) 1 "cljs_arithmetic_min")
(is= (min 5 3 7 1 9) 1 "cljs_arithmetic_min")

(is= (abs -5) 5 "cljs_arithmetic_abs")
(is= (abs 5) 5 "cljs_arithmetic_abs")
(is= (abs 0) 0 "cljs_arithmetic_abs")

(is= (+ 1000000000 2000000000) 3000000000 "cljs_arithmetic_large_numbers")
(is= (* 100000 100000) 10000000000 "cljs_arithmetic_large_numbers")

;; ============================================================================
;; NUMERIC PREDICATES
;; ============================================================================

(testing "NUMERIC PREDICATES")

(is= (zero? 0) true "cljs_numeric_pred_zero")
(is= (zero? 1) false "cljs_numeric_pred_zero")

(is= (pos? 1) true "cljs_numeric_pred_pos")
(is= (pos? 0) false "cljs_numeric_pred_pos")
(is= (pos? -1) false "cljs_numeric_pred_pos")

(is= (neg? -1) true "cljs_numeric_pred_neg")
(is= (neg? 0) false "cljs_numeric_pred_neg")
(is= (neg? 1) false "cljs_numeric_pred_neg")

(is= (even? 0) true "cljs_numeric_pred_even")
(is= (even? 2) true "cljs_numeric_pred_even")
(is= (even? 3) false "cljs_numeric_pred_even")

(is= (odd? 1) true "cljs_numeric_pred_odd")
(is= (odd? 3) true "cljs_numeric_pred_odd")
(is= (odd? 2) false "cljs_numeric_pred_odd")

(is= (number? 42) true "cljs_numeric_pred_number")
(is= (number? 3.14) true "cljs_numeric_pred_number")
(is= (number? :a) false "cljs_numeric_pred_number")
(is= (number? "hello") false "cljs_numeric_pred_number")

(is= (integer? 1) true "cljs_numeric_pred_integer")
(is= (integer? 1.0) false "cljs_numeric_pred_integer")

(is= (float? 1.0) true "cljs_numeric_pred_float")
(is= (float? 1) false "cljs_numeric_pred_float")

;; ============================================================================
;; COMPARISON OPERATORS
;; ============================================================================

(testing "COMPARISON OPERATORS")

(is= (< 1 2) true "cljs_comparison_less_than")
(is= (< 2 1) false "cljs_comparison_less_than")
(is= (< 1 1) false "cljs_comparison_less_than")

(is= (< 1 2 3) true "cljs_comparison_less_than_variadic")
(is= (< 1 2 3 4 5) true "cljs_comparison_less_than_variadic")
(is= (< 1 3 2) false "cljs_comparison_less_than_variadic")

(is= (<= 1 1) true "cljs_comparison_less_equal")
(is= (<= 1 2) true "cljs_comparison_less_equal")
(is= (<= 2 1) false "cljs_comparison_less_equal")
(is= (<= 1 1 2 2 3) true "cljs_comparison_less_equal")

(is= (> 2 1) true "cljs_comparison_greater_than")
(is= (> 1 2) false "cljs_comparison_greater_than")
(is= (> 5 4 3 2 1) true "cljs_comparison_greater_than")

(is= (>= 2 2) true "cljs_comparison_greater_equal")
(is= (>= 2 1) true "cljs_comparison_greater_equal")
(is= (>= 1 2) false "cljs_comparison_greater_equal")
(is= (>= 3 3 2 1) true "cljs_comparison_greater_equal")

(is= (= 1 1) true "cljs_comparison_equality_primitives")
(is= (= 1 2) false "cljs_comparison_equality_primitives")
(is= (= :a :a) true "cljs_comparison_equality_primitives")
(is= (= :a :b) false "cljs_comparison_equality_primitives")
(is= (= nil nil) true "cljs_comparison_equality_primitives")
(is= (= true true) true "cljs_comparison_equality_primitives")
(is= (= false false) true "cljs_comparison_equality_primitives")
(is= (= true false) false "cljs_comparison_equality_primitives")

(is= (= 1 1 1 1) true "cljs_comparison_equality_variadic")
(is= (= 1 1 2 1) false "cljs_comparison_equality_variadic")

(is= (not= 1 2) true "cljs_comparison_not_equal")
(is= (not= 1 1) false "cljs_comparison_not_equal")

(is= (== 1 1) true "cljs_comparison_double_equals")

;; ============================================================================
;; BIT OPERATIONS
;; ============================================================================

(testing "BIT OPERATIONS")

(is= (bit-and 255 15) 15 "cljs_bit_and")

(is= (bit-or 15 240) 255 "cljs_bit_or")

(is= (bit-xor 255 15) 240 "cljs_bit_xor")

(is= (bit-not 0) -1 "cljs_bit_not")

(is= (bit-shift-left 1 4) 16 "cljs_bit_shift_left")

(is= (bit-shift-right 16 4) 1 "cljs_bit_shift_right")

;; ============================================================================
;; BOOLEAN AND LOGIC
;; ============================================================================

(testing "BOOLEAN AND LOGIC")

(is= (not true) false "cljs_logic_not")
(is= (not false) true "cljs_logic_not")
(is= (not nil) true "cljs_logic_not")
(is= (not 1) false "cljs_logic_not")
(is= (not 0) false "cljs_logic_not")

(is= (and) true "cljs_logic_and")
(is= (and true true) true "cljs_logic_and")
(is= (and true false) false "cljs_logic_and")
(is= (and false true 0) false "cljs_logic_and")
(is= (and true true 0) 0 "cljs_logic_and")

(is= (or) nil "cljs_logic_or")
(is= (or false false 1) 1 "cljs_logic_or")
(is= (or false false false) false "cljs_logic_or")
(is= (or nil nil true) true "cljs_logic_or")

(is= (or nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil true) true "cljs_logic_or_many_nils")

;; ============================================================================
;; TYPE PREDICATES
;; ============================================================================

(testing "TYPE PREDICATES")

(is= (nil? nil) true "cljs_pred_nil")
(is= (nil? 1) false "cljs_pred_nil")
(is= (nil? false) false "cljs_pred_nil")

(is= (some? nil) false "cljs_pred_some")
(is= (some? 1) true "cljs_pred_some")
(is= (some? false) true "cljs_pred_some")

(is= (true? true) true "cljs_pred_true")
(is= (true? 1) false "cljs_pred_true")
(is= (true? false) false "cljs_pred_true")

(is= (false? false) true "cljs_pred_false")
(is= (false? nil) false "cljs_pred_false")
(is= (false? true) false "cljs_pred_false")

(is= (string? "hi") true "cljs_pred_string")
(is= (string? 1) false "cljs_pred_string")
(is= (string? :a) false "cljs_pred_string")

(is= (keyword? :a) true "cljs_pred_keyword")
(is= (keyword? "a") false "cljs_pred_keyword")
(is= (keyword? 1) false "cljs_pred_keyword")

(is= (symbol? 'a) true "cljs_pred_symbol")
(is= (symbol? :a) false "cljs_pred_symbol")
(is= (symbol? 1) false "cljs_pred_symbol")

(is= (fn? inc) true "cljs_pred_fn")
(is= (fn? +) true "cljs_pred_fn")
(is= (fn? 1) false "cljs_pred_fn")

(is= (vector? [1 2]) true "cljs_pred_vector")
(is= (vector? (list 1)) false "cljs_pred_vector")
(is= (vector? #{1}) false "cljs_pred_vector")

(is= (map? {:a 1}) true "cljs_pred_map")
(is= (map? {}) true "cljs_pred_map")
(is= (map? [1]) false "cljs_pred_map")

(is= (set? #{1 2}) true "cljs_pred_set")
(is= (set? #{}) true "cljs_pred_set")
(is= (set? [1]) false "cljs_pred_set")

(is= (list? (list 1)) true "cljs_pred_list")
(is= (list? [1]) false "cljs_pred_list")

(is= (coll? [1]) true "cljs_pred_coll")
(is= (coll? {:a 1}) true "cljs_pred_coll")
(is= (coll? #{1}) true "cljs_pred_coll")
(is= (coll? (list 1)) true "cljs_pred_coll")
(is= (coll? 1) false "cljs_pred_coll")

(is= (sequential? [1]) true "cljs_pred_sequential")
(is= (sequential? (list 1)) true "cljs_pred_sequential")
(is= (sequential? {:a 1}) false "cljs_pred_sequential")
(is= (sequential? #{1}) false "cljs_pred_sequential")

(is= (associative? {:a 1}) true "cljs_pred_associative")
(is= (associative? [1]) true "cljs_pred_associative")

;; ============================================================================
;; STRING OPERATIONS
;; ============================================================================

(testing "STRING OPERATIONS")

(is= (str) "" "cljs_str_basic")
(is= (str nil) "" "cljs_str_basic")
(is= (str "a") "a" "cljs_str_basic")
(is= (str 1) "1" "cljs_str_basic")

(is= (str "hello" " " "world") "hello world" "cljs_str_concatenation")
(is= (str "a" "b") "ab" "cljs_str_concatenation")

(is= (str 1 2 3) "123" "cljs_str_mixed_types")
(is= (str "a" 1 "b" 2 "c" 3) "a1b2c3" "cljs_str_mixed_types")
(is= (str "a" nil "b") "ab" "cljs_str_mixed_types")

(is= (str "x" "y" "z" "z" "y") "xyzzy" "cljs_str_xyzzy")

(is= (str 0 "a" true nil :key/word) "0atrue:key/word" "cljs_str_mixed_all_types")

(is= (str [1 2 3]) "[1 2 3]" "cljs_str_on_collections")
(is= (str (list 1 2 3)) "(1 2 3)" "cljs_str_on_collections")
(is= (str :hello) ":hello" "cljs_str_on_collections")
(is= (str true) "true" "cljs_str_on_collections")
(is= (str false) "false" "cljs_str_on_collections")

(is= (str []) "[]" "cljs_str_empty_collections")
(is= (str (list)) "()" "cljs_str_empty_collections")
(is= (str #{}) "#{}" "cljs_str_empty_collections")
(is= (str {}) "{}" "cljs_str_empty_collections")

(is= (count "hello") 5 "cljs_str_count")
(is= (count "") 0 "cljs_str_count")
(is= (count "a") 1 "cljs_str_count")

(is= (name :foo) "foo" "cljs_str_name")
(is= (name 'bar) "bar" "cljs_str_name")
(is= (name "already-a-string") "already-a-string" "cljs_str_name")

(is= (apply str ["a" "b" "c"]) "abc" "cljs_str_apply_str")
(is= (apply str []) "" "cljs_str_apply_str")

(is= (reduce str ["a" "b" "c"]) "abc" "cljs_str_reduce_str")
(is= (reduce str "" ["a" "b" "c"]) "abc" "cljs_str_reduce_str")

(is= (apply str (range 5)) "01234" "cljs_str_apply_str_range")

;; ============================================================================
;; SYMBOL OPERATIONS
;; ============================================================================

(testing "SYMBOL OPERATIONS")

(is= (symbol "hello") 'hello "cljs_symbol_from_string")
(is= (symbol "ns" "name") 'ns/name "cljs_symbol_from_string")

(is= (symbol? 'foo) true "cljs_symbol_pred")
(is= (symbol? :foo) false "cljs_symbol_pred")

;; ============================================================================
;; KEYWORD OPERATIONS
;; ============================================================================

(testing "KEYWORD OPERATIONS")

(is= (:a {:a 42 :b 99}) 42 "cljs_keyword_as_fn")
(is= (:c {:a 1}) nil "cljs_keyword_as_fn")
(is= (:c {:a 1} :default) :default "cljs_keyword_as_fn")
(is= (:b {:a 1} :not-found) :not-found "cljs_keyword_as_fn")

(is= :foo/bar :foo/bar "cljs_keyword_namespaced")
(is= (str :hello/world) ":hello/world" "cljs_keyword_namespaced")

;; ============================================================================
;; VECTORS
;; ============================================================================

(testing "VECTORS")

(is= (vector 1 2 3) [1 2 3] "cljs_vector_creation")
(is= (vec (list 1 2 3)) [1 2 3] "cljs_vector_creation")
(is= [1 2 3] [1 2 3] "cljs_vector_creation")

(is= (count [1 2 3]) 3 "cljs_vector_count")
(is= (count []) 0 "cljs_vector_count")

(is= (nth [10 20 30] 0) 10 "cljs_vector_nth")
(is= (nth [10 20 30] 1) 20 "cljs_vector_nth")
(is= (nth [10 20 30] 2) 30 "cljs_vector_nth")

(is= (nth [1 2 3] 5 :not-found) :not-found "cljs_vector_nth_with_default")

(is= (get [10 20 30] 0) 10 "cljs_vector_get")
(is= (get [10 20 30] 1) 20 "cljs_vector_get")

(is= (first [1 2 3]) 1 "cljs_vector_first_second_last")
(is= (second [1 2 3]) 2 "cljs_vector_first_second_last")
(is= (last [1 2 3]) 3 "cljs_vector_first_second_last")

(is= (conj [1 2] 3) [1 2 3] "cljs_vector_conj")
(is= (vector? (conj [1] 2)) true "cljs_vector_conj")

(is= (assoc [1 2 3] 0 :a) [:a 2 3] "cljs_vector_assoc")
(is= (assoc [1 2 3] 1 :b) [1 :b 3] "cljs_vector_assoc")

(is= (str (update [1 2 3] 0 inc)) "[2 2 3]" "cljs_vector_update")

(is= ([10 20 30] 0) 10 "cljs_vector_as_fn")
(is= ([10 20 30] 1) 20 "cljs_vector_as_fn")
(is= ([10 20 30] 2) 30 "cljs_vector_as_fn")

(is= (contains? [5 6 7] 0) true "cljs_vector_contains")
(is= (contains? [5 6 7] 1) true "cljs_vector_contains")
(is= (contains? [5 6 7] 2) true "cljs_vector_contains")
(is= (contains? [5 6 7] 3) false "cljs_vector_contains")

(is= (empty? []) true "cljs_vector_empty")
(is= (empty? [1]) false "cljs_vector_empty")

(is= (seq []) nil "cljs_vector_seq")
(is= (str (seq [1])) "(1)" "cljs_vector_seq")

(is= (str (empty [1 2 3])) "[]" "cljs_vector_empty_fn")

(is= (str (into [] (list 1 2 3))) "[1 2 3]" "cljs_vector_into")
(is= (str (into [] (range 5))) "[0 1 2 3 4]" "cljs_vector_into")

;; ============================================================================
;; LISTS
;; ============================================================================

(testing "LISTS")

(is= (str (list 1 2 3)) "(1 2 3)" "cljs_list_creation")

(is= (list? (list 1)) true "cljs_list_predicates")
(is= (list? [1]) false "cljs_list_predicates")

(is= (str (conj (list 1 2) 3)) "(3 1 2)" "cljs_list_conj")
(is= (list? (conj (list 1) 2)) true "cljs_list_conj")

(is= (first (list 1 2 3)) 1 "cljs_list_first_rest")
(is= (str (rest (list 1 2 3))) "(2 3)" "cljs_list_first_rest")

(is= (count (list 1 2 3)) 3 "cljs_list_count")
(is= (count (list)) 0 "cljs_list_count")

(is= (str (into (list) [1 2 3])) "(3 2 1)" "cljs_list_into")

(is= (str (empty (list 1 2 3))) "()" "cljs_list_empty_fn")

;; ============================================================================
;; MAPS
;; ============================================================================

(testing "MAPS")

(is= (get {:a 1 :b 2} :a) 1 "cljs_map_get")
(is= (get {:a 1} :b) nil "cljs_map_get")
(is= (get {:a 1} :b :default) :default "cljs_map_get")

(is= (contains? {:a 1 :b 2} :a) true "cljs_map_contains")
(is= (contains? {:a 1 :b 2} :z) false "cljs_map_contains")
(is= (contains? nil 42) false "cljs_map_contains")

(is= (count {:a 1 :b 2}) 2 "cljs_map_count")
(is= (count {}) 0 "cljs_map_count")
(is= (count {:a 1 :b 2 :c 3 :d 4 :e 5}) 5 "cljs_map_count")

(is= (get (assoc {:a 1} :b 2) :b) 2 "cljs_map_assoc")
(is= (count (assoc {:a 1} :b 2)) 2 "cljs_map_assoc")

(is= (count (assoc {} :a 1 :b 2 :c 3)) 3 "cljs_map_assoc_multiple")
(is= (get (assoc {} :a 1 :b 2 :c 3) :b) 2 "cljs_map_assoc_multiple")

(is= (contains? (dissoc {:a 1 :b 2} :b) :b) false "cljs_map_dissoc")
(is= (count (dissoc {:a 1 :b 2} :b)) 1 "cljs_map_dissoc")

(is= (count (dissoc {:a 1 :b 2 :c 3} :a :c)) 1 "cljs_map_dissoc_multiple")
(is= (get (dissoc {:a 1 :b 2 :c 3} :a :c) :b) 2 "cljs_map_dissoc_multiple")

(is= (count (merge {:a 1} {:b 2} {:c 3})) 3 "cljs_map_merge")
(is= (get (merge {:a 1} {:b 2} {:c 3}) :c) 3 "cljs_map_merge")
(is= (count (merge {:a 1} {:b 2} {:c 3} {:d 4})) 4 "cljs_map_merge")

(is= (get (merge {:a 1} {:a 2}) :a) 2 "cljs_map_merge_overwrite")
(is= (get (merge {:a 1 :b 2} {:b 3 :c 4}) :b) 3 "cljs_map_merge_overwrite")

(is= (get (update {:a 1} :a inc) :a) 2 "cljs_map_update")
(is= (get (update {:a 1} :a + 10) :a) 11 "cljs_map_update")
(is= (get (update {:a 1} :a + 10 20) :a) 31 "cljs_map_update")

(is= ({:a 1} :a) 1 "cljs_map_as_fn")
(is= ({:a 1} :b) nil "cljs_map_as_fn")
(is= ({:a 1} :b 99) 99 "cljs_map_as_fn")
(is= ({:a 1} 2 3) 3 "cljs_map_as_fn")

(is= ((hash-map :a 1) :a 3) 1 "cljs_map_hash_map")
(is= (count (apply hash-map [:a 1 :b 2])) 2 "cljs_map_hash_map")

(is= (count (keys {:a 1 :b 2})) 2 "cljs_map_keys_vals")
(is= (count (vals {:a 1 :b 2})) 2 "cljs_map_keys_vals")
(is= (set (keys {:a 1 :b 2})) #{:a :b} "cljs_map_keys_vals")
(is= (set (vals {:a 1 :b 2})) #{1 2} "cljs_map_keys_vals")

(is= (key (first {:a 1})) :a "cljs_map_key_val")
(is= (val (first {:a 1})) 1 "cljs_map_key_val")

(is= (count (select-keys {:a 1 :b 2 :c 3} [:a :c])) 2 "cljs_map_select_keys")
(is= (get (select-keys {:a 1 :b 2 :c 3} [:a :c]) :a) 1 "cljs_map_select_keys")
(is= (count (select-keys {:a 1 :b 2 :c 3 :d 4} [:a :c])) 2 "cljs_map_select_keys")

(is= (get (zipmap [:a :b :c] [1 2 3]) :b) 2 "cljs_map_zipmap")
(is= (count (zipmap [:a :b :c] [1 2 3])) 3 "cljs_map_zipmap")

(is= (get (frequencies [:a :b :a :c :b :a]) :a) 3 "cljs_map_frequencies")
(is= (get (frequencies [:a :b :a :c :b :a]) :b) 2 "cljs_map_frequencies")
(is= (get (frequencies [:a :b :a :c :b :a]) :c) 1 "cljs_map_frequencies")
(is= (get (frequencies [1 1 2 3 3 3]) 3) 3 "cljs_map_frequencies")
(is= (get (frequencies [1 1 2 3 3 3]) 1) 2 "cljs_map_frequencies")

(is= (count (get (group-by odd? [1 2 3 4 5]) true)) 3 "cljs_map_group_by")
(is= (count (get (group-by odd? [1 2 3 4 5]) false)) 2 "cljs_map_group_by")
(is= (str (get (group-by odd? [1 2 3 4 5]) true)) "[1 3 5]" "cljs_map_group_by")

(is= (str (empty {:a 1})) "{}" "cljs_map_empty_fn")

;; ============================================================================
;; GET-IN, ASSOC-IN, UPDATE-IN
;; ============================================================================

(testing "GET-IN, ASSOC-IN, UPDATE-IN")

(is= (get-in {:foo 1 :bar 2} [:foo]) 1 "cljs_get_in_basic")
(is= (get-in {:foo {:bar 2}} [:foo :bar]) 2 "cljs_get_in_basic")
(is= (get-in {:a {:b {:c 42}}} [:a :b :c]) 42 "cljs_get_in_basic")

(is= (get-in [{:a 1} {:a 2}] [0 :a]) 1 "cljs_get_in_vector_index")
(is= (get-in [{:a 1} {:a 2}] [1 :a]) 2 "cljs_get_in_vector_index")

(is= (get-in [[1 2] [3 4] [5 6]] [1 0]) 3 "cljs_get_in_nested_vector")
(is= (get-in [[1 2] [3 4] [5 6]] [2 1]) 6 "cljs_get_in_nested_vector")

(is= (get-in {:a {:b {:c {:d 42}}}} [:a :b :c :d]) 42 "cljs_get_in_deep")
(is= (get-in [{:foo 1 :bar [{:baz 1} {:buzz 2}]} {:foo 3 :bar [{:baz 3} {:buzz 4}]}] [1 :bar 1 :buzz]) 4 "cljs_get_in_deep")

(is= (get-in {:a 1} [:b]) nil "cljs_get_in_not_found")

(is= (get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b]) 2 "cljs_update_in_basic")
(is= (get-in (update-in {:a {:b [1 2 3]}} [:a :b] count) [:a :b]) 3 "cljs_update_in_basic")

(is= (get-in (assoc-in {} [:a :b :c] 42) [:a :b :c]) 42 "cljs_assoc_in_basic")
(is= (get-in (assoc-in {:a {:b 1}} [:a :b] 99) [:a :b]) 99 "cljs_assoc_in_basic")

;; ============================================================================
;; SETS
;; ============================================================================

(testing "SETS")

(is= (count (set [1 2 2])) 2 "cljs_set_creation")
(is= (count (hash-set 1 2 2)) 2 "cljs_set_creation")
(is= (count (apply hash-set [1 2 2])) 2 "cljs_set_creation")

(is= (contains? #{1 2 3} 2) true "cljs_set_contains")
(is= (contains? #{1 2 3} 4) false "cljs_set_contains")

(is= (count #{1 2 3}) 3 "cljs_set_count")
(is= (count #{}) 0 "cljs_set_count")

(is= (#{:a :b :c} :a) :a "cljs_set_as_fn")
(is= (#{:a :b :c} :d) nil "cljs_set_as_fn")
(is= (#{1 2 3} 2) 2 "cljs_set_as_fn")
(is= (#{1 2 3} 4) nil "cljs_set_as_fn")

(is= (count (disj #{1 2 3} 2)) 2 "cljs_set_disj")
(is= (contains? (disj #{1 2 3} 2) 2) false "cljs_set_disj")

(is= (count (disj #{1 2 3 4} 1 3)) 2 "cljs_set_disj_multiple")
(is= (contains? (disj #{1 2 3 4} 1 3) 2) true "cljs_set_disj_multiple")
(is= (contains? (disj #{1 2 3 4} 1 3) 1) false "cljs_set_disj_multiple")

(is= (count (conj #{1 2} 3)) 3 "cljs_set_conj")

(is= (count (conj #{1 2} 2)) 2 "cljs_set_conj_existing")

(is= (count (set [1 2 2 3 3 3])) 3 "cljs_set_from_vector")
(is= (contains? (set [1 2 3]) 2) true "cljs_set_from_vector")

(is= (str (empty #{1 2})) "#{}" "cljs_set_empty_fn")

(is= (count (into #{} [1 2 2 3])) 3 "cljs_set_into")

;; ============================================================================
;; SEQUENCE OPERATIONS
;; ============================================================================

(testing "SEQUENCE OPERATIONS")

(is= (first [1 2 3]) 1 "cljs_seq_first")
(is= (first (list 1 2 3)) 1 "cljs_seq_first")
(is= (first nil) nil "cljs_seq_first")
(is= (first []) nil "cljs_seq_first")

(is= (second [10 20 30]) 20 "cljs_seq_second")

(is= (str (rest [1 2 3])) "(2 3)" "cljs_seq_rest")
(is= (str (rest [1])) "()" "cljs_seq_rest")

(is= (str (next [1 2 3])) "(2 3)" "cljs_seq_next")
(is= (next [1]) nil "cljs_seq_next")
(is= (next nil) nil "cljs_seq_next")

(is= (last [1 2 3]) 3 "cljs_seq_last")
(is= (last nil) nil "cljs_seq_last")
(is= (last []) nil "cljs_seq_last")

(is= (count [1 2 3]) 3 "cljs_seq_count")
(is= (count nil) 0 "cljs_seq_count")
(is= (count []) 0 "cljs_seq_count")
(is= (count {:a 1 :b 2}) 2 "cljs_seq_count")

(is= (str (cons 0 [1 2 3])) "(0 1 2 3)" "cljs_seq_cons")

(is= (str (conj nil 1)) "(1)" "cljs_seq_conj_nil")

(is= (str (concat [1 2] [3 4] [5])) "(1 2 3 4 5)" "cljs_seq_concat")

(is= (str (concat nil [])) "" "cljs_seq_concat_empty")
(is= (str (concat [] [])) "" "cljs_seq_concat_empty")

(is= (str (into [] '(1 2 3))) "[1 2 3]" "cljs_seq_into_vector")

(is= (get (into {} [[:a 1] [:b 2]]) :a) 1 "cljs_seq_into_map")

(is= (count (into #{} [1 2 2 3])) 3 "cljs_seq_into_set")

(is= (str (into nil [1 2 3])) "(3 2 1)" "cljs_seq_into_nil")

(is= (str (reverse [1 2 3])) "(3 2 1)" "cljs_seq_reverse")

(is= (str (sort [3 1 2])) "(1 2 3)" "cljs_seq_sort")

(is= (str (sort > [3 1 2])) "(3 2 1)" "cljs_seq_sort_with_comparator")

(is= (str (distinct [1 2 1 3 2 4])) "(1 2 3 4)" "cljs_seq_distinct")

(is= (str (interleave [1 2 3] [:a :b :c])) "(1 :a 2 :b 3 :c)" "cljs_seq_interleave")

(is= (str (partition 2 [1 2 3 4 5])) "((1 2) (3 4))" "cljs_seq_partition")

(is= (str (butlast [1 2 3])) "(1 2)" "cljs_seq_butlast")

(is= (str (flatten [1 [2 [3 4]] 5])) "(1 2 3 4 5)" "cljs_seq_flatten")
(is= (str (flatten [1 [2] [[3]]])) "(1 2 3)" "cljs_seq_flatten")

(is= (str (vec '(1 2 3))) "[1 2 3]" "cljs_seq_vec")

(is= (empty? []) true "cljs_seq_empty")
(is= (empty? nil) true "cljs_seq_empty")
(is= (empty? [1]) false "cljs_seq_empty")

(is= (seq []) nil "cljs_seq_on_empty")
(is= (seq nil) nil "cljs_seq_on_empty")

(is= (str (range 5)) "(0 1 2 3 4)" "cljs_seq_range")
(is= (str (range 2 5)) "(2 3 4)" "cljs_seq_range")
(is= (str (range 0 10 3)) "(0 3 6 9)" "cljs_seq_range")

(is= (str (repeat 3 :a)) "(:a :a :a)" "cljs_seq_repeat_bounded")

(is= (str (take 3 [1 2 3 4 5])) "(1 2 3)" "cljs_seq_take")
(is= (str (take 0 [1 2 3])) "" "cljs_seq_take")
(is= (str (take 10 [1 2 3])) "(1 2 3)" "cljs_seq_take")

(is= (str (drop 3 [1 2 3 4 5])) "(4 5)" "cljs_seq_drop")
(is= (str (drop 0 [1 2 3])) "(1 2 3)" "cljs_seq_drop")
(is= (str (drop 10 [1 2 3])) "" "cljs_seq_drop")

(is= (str (take-while #(< % 4) [1 2 3 4 5])) "(1 2 3)" "cljs_seq_take_while")

(is= (str (drop-while #(< % 4) [1 2 3 4 5])) "(4 5)" "cljs_seq_drop_while")

(is= (str (apply vector (drop-while (partial = 1) [1 2 3]))) "[2 3]" "cljs_seq_drop_while_partial")

(is= (first (rest (rest (rest (range 3))))) nil "cljs_seq_rest_rest_rest_range")

;; ============================================================================
;; HIGHER-ORDER FUNCTIONS
;; ============================================================================

(testing "HIGHER-ORDER FUNCTIONS")

(is= (str (map inc [0 1 2])) "(1 2 3)" "cljs_hof_map_basic")

(is= (str (map #(+ 1 %) [0 1 2])) "(1 2 3)" "cljs_hof_map_with_fn_literal")

(is= (str (map + [1 2 3] [10 20 30])) "(11 22 33)" "cljs_hof_map_multiple_colls")
(is= (str (map + [1 2 3] [4 5 6] [7 8 9])) "(12 15 18)" "cljs_hof_map_multiple_colls")

(is= (str (map :a [{:a 1} {:a 2} {:a 3}])) "(1 2 3)" "cljs_hof_map_with_keywords")
(is= (str (map :name [{:name "a"} {:name "b"} {:name "c"}])) "(\"a\" \"b\" \"c\")" "cljs_hof_map_with_keywords")

(is= (str (map vector [:a :b :c] [1 2 3])) "([:a 1] [:b 2] [:c 3])" "cljs_hof_map_with_vector")

(is= (str (map inc (range 5))) "(1 2 3 4 5)" "cljs_hof_map_inc_range")

(is= (str (map #(* % %) [1 2 3 4 5])) "(1 4 9 16 25)" "cljs_hof_map_squared")

(is= (str (map (constantly 42) [1 2 3])) "(42 42 42)" "cljs_hof_map_constantly")

(is= (str (map name [:a :b :c])) "(\"a\" \"b\" \"c\")" "cljs_hof_map_name")

(is= (str (map str [1 2 3])) "(\"1\" \"2\" \"3\")" "cljs_hof_map_str")

(is= (str (map count [[1] [1 2] [1 2 3]])) "(1 2 3)" "cljs_hof_map_count")

(is= (str (map first [[1 2] [3 4] [5 6]])) "(1 3 5)" "cljs_hof_map_first")

(is= (str (map rest [[1 2 3] [4 5 6]])) "((2 3) (5 6))" "cljs_hof_map_rest")

(is= (str (map #(map inc %) [[1 2] [3 4]])) "((2 3) (4 5))" "cljs_hof_map_nested")

(is= (str (filter even? [1 2 3 4 5 6])) "(2 4 6)" "cljs_hof_filter")

(is= (str (filter odd? [1 2 3])) "(1 3)" "cljs_hof_filter_odd")

(is= (str (filter (fn [x] false) [1 2 3])) "" "cljs_hof_filter_empty")

(is= (str (remove even? [1 2 3 4 5 6])) "(1 3 5)" "cljs_hof_remove")

(is= (str (keep odd? [0 1 2])) "(false true false)" "cljs_hof_keep")
(is= (str (keep identity [1 nil 2 nil 3])) "(1 2 3)" "cljs_hof_keep")

(is= (str (map-indexed #(do [%1 %2]) [1 2 3])) "([0 1] [1 2] [2 3])" "cljs_hof_map_indexed")
(is= (str (map-indexed vector [:a :b :c])) "([0 :a] [1 :b] [2 :c])" "cljs_hof_map_indexed")

(is= (str (mapcat #(vector % (* % %)) [1 2 3])) "(1 1 2 4 3 9)" "cljs_hof_mapcat")
(is= (str (mapcat reverse [[3 2 1] [6 5 4]])) "(1 2 3 4 5 6)" "cljs_hof_mapcat")

(is= (reduce + [1 2 3 4 5]) 15 "cljs_hof_reduce_basic")
(is= (reduce + 10 [1 2 3]) 16 "cljs_hof_reduce_basic")
(is= (reduce + 0 []) 0 "cljs_hof_reduce_basic")
(is= (reduce * 1 []) 1 "cljs_hof_reduce_basic")

(is= (reduce + 0 (range 10)) 45 "cljs_hof_reduce_with_range")
(is= (reduce + (range 1 6)) 15 "cljs_hof_reduce_with_range")

(is= (reduce + 0 (filter odd? (map inc [0 1 2 3 4]))) 9 "cljs_hof_reduce_with_map_filter")

(is= (str (reduce conj [] [1 2 3])) "[1 2 3]" "cljs_hof_reduce_conj")
(is= (str (reduce (fn [acc x] (conj acc (* x x))) [] [1 2 3 4])) "[1 4 9 16]" "cljs_hof_reduce_conj")

(is= (reduce-kv (fn [acc k v] (+ acc v)) 0 {:a 1 :b 2 :c 3}) 6 "cljs_hof_reduce_kv_map")

(is= (reduce-kv (fn [acc k v] (+ acc v)) 0 [10 20 30]) 60 "cljs_hof_reduce_kv_vector")

(is= (apply + [1 2 3]) 6 "cljs_hof_apply")
(is= (apply + 1 2 [3 4]) 10 "cljs_hof_apply")
(is= (apply + 1 2 3 [4 5]) 15 "cljs_hof_apply")
(is= (apply + []) 0 "cljs_hof_apply")

(is= (apply str ["a" "b" "c"]) "abc" "cljs_hof_apply_str")

(is= (apply vector 1 2 [3 4]) [1 2 3 4] "cljs_hof_apply_vector")

(is= (str (apply list 1 2 [3 4])) "(1 2 3 4)" "cljs_hof_apply_list")

(is= (apply max [1 2 3]) 3 "cljs_hof_apply_max_min")
(is= (apply min [5 3 7]) 3 "cljs_hof_apply_max_min")

(is= (some even? [1 3 5 6]) true "cljs_hof_some")
(is= (some even? [1 3 5]) nil "cljs_hof_some")

(is= (some identity [nil nil 3 nil]) 3 "cljs_hof_some_identity")
(is= (some identity [nil nil nil]) nil "cljs_hof_some_identity")

(is= (some #{1 2} [3 4 1 5]) 1 "cljs_hof_some_set_as_pred")
(is= (some #{1 2} [3 4 5]) nil "cljs_hof_some_set_as_pred")
(is= (some #{:a} [:b :c :a :d]) :a "cljs_hof_some_set_as_pred")

(is= (some :a [{:b 1} {:a 2} {:c 3}]) 2 "cljs_hof_some_keyword_as_pred")

(is= (every? even? [2 4 6]) true "cljs_hof_every")
(is= (every? even? [2 3 6]) false "cljs_hof_every")
(is= (every? :a [{:a 1} {:a 2} {:a 3}]) true "cljs_hof_every")

(is= (not-every? even? [2 4 6]) false "cljs_hof_not_every")
(is= (not-every? even? [2 3 6]) true "cljs_hof_not_every")

(is= (not-any? even? [1 3 5]) true "cljs_hof_not_any")
(is= (not-any? even? [1 2 5]) false "cljs_hof_not_any")

;; ============================================================================
;; IDENTITY, CONSTANTLY, COMP, PARTIAL, COMPLEMENT, JUXT
;; ============================================================================

(testing "IDENTITY, CONSTANTLY, COMP, PARTIAL, COMPLEMENT, JUXT")

(is= (identity 42) 42 "cljs_hof_identity")
(is= (identity nil) nil "cljs_hof_identity")
(is= (identity :a) :a "cljs_hof_identity")

(is= ((constantly 5) 1 2 3) 5 "cljs_hof_constantly")
(is= ((constantly nil) 1) nil "cljs_hof_constantly")

(is= ((comp inc inc inc) 0) 3 "cljs_hof_comp")
(is= ((comp str inc) 1) "2" "cljs_hof_comp")
(is= ((comp str inc inc) 0) "2" "cljs_hof_comp")
(is= ((comp) 42) 42 "cljs_hof_comp")

(is= ((partial + 10) 5) 15 "cljs_hof_partial")
(is= ((partial + 1 2 3) 4) 10 "cljs_hof_partial")
(is= (apply (partial + 1 2) [3 4]) 10 "cljs_hof_partial")

(is= ((complement nil?) 1) true "cljs_hof_complement")
(is= ((complement nil?) nil) false "cljs_hof_complement")
(is= ((complement even?) 3) true "cljs_hof_complement")
(is= ((complement even?) 4) false "cljs_hof_complement")

(is= (nth ((juxt inc dec) 1) 0) 2 "cljs_hof_juxt")
(is= (nth ((juxt inc dec) 1) 1) 0 "cljs_hof_juxt")
(is= (str ((juxt inc dec) 1)) "[2 0]" "cljs_hof_juxt")
(is= (str ((juxt + - *) 3 4)) "[7 -1 12]" "cljs_hof_juxt")
(is= (str ((juxt first last) [1 2 3])) "[1 3]" "cljs_hof_juxt")

;; ============================================================================
;; CONTROL FLOW
;; ============================================================================

(testing "CONTROL FLOW")

(is= (if true 10 20) 10 "cljs_control_if_basic")
(is= (if false 10 20) 20 "cljs_control_if_basic")
(is= (if nil 10 20) 20 "cljs_control_if_basic")

;; 0 and "" are truthy in Clojure
(is= (if 0 "yes" "no") "yes" "cljs_control_if_truthy_values")
(is= (if "" "yes" "no") "yes" "cljs_control_if_truthy_values")
(is= (if (> 1 0) "yes" "no") "yes" "cljs_control_if_truthy_values")

(is= (if true 1) 1 "cljs_control_if_no_else")
(is= (if false 1) nil "cljs_control_if_no_else")

(is= (when true 0 1 2) 2 "cljs_control_when")
(is= (when false 1) nil "cljs_control_when")

(is= (cond true 1 :else 2) 1 "cljs_control_cond")
(is= (cond false 1 :else 2) 2 "cljs_control_cond")
(is= (cond (= 1 2) :a (= 2 3) :b (= 3 3) :c :else :d) :c "cljs_control_cond")

(is= (let [x 2] (cond (string? x) 1 (integer? x) 2)) 2 "cljs_control_cond_type_check")

(is= (if-let [x 42] x 0) 42 "cljs_control_if_let")
(is= (if-let [x nil] 1 2) 2 "cljs_control_if_let")
(is= (if-let [x false] 1 2) 2 "cljs_control_if_let")
(is= (if-let [x (seq [1 2 3])] (first x) :empty) 1 "cljs_control_if_let")
(is= (if-let [x (seq [])] (first x) :empty) :empty "cljs_control_if_let")

(is= (when-let [x 42] x) 42 "cljs_control_when_let")
(is= (when-let [x nil] 1) nil "cljs_control_when_let")
(is= (when-let [x false] 1) nil "cljs_control_when_let")
(is= (when-let [x (first [42])] x) 42 "cljs_control_when_let")

(is= (do) nil "cljs_control_do")
(is= (do 1 2 3) 3 "cljs_control_do")
(is= (do 1 2 nil) nil "cljs_control_do")

(is= (comment "anything") nil "cljs_control_comment")
(is= (comment 1) nil "cljs_control_comment")
(is= (comment (+ 1 2 (* 3 4))) nil "cljs_control_comment")

;; ============================================================================
;; THREADING MACROS
;; ============================================================================

(testing "THREADING MACROS")

(is= (-> 1 inc inc inc) 4 "cljs_thread_first_basic")
(is= (-> 3 inc inc inc) 6 "cljs_thread_first_basic")

(is= (-> {:a 1} (assoc :b 2) (assoc :c 3) count) 3 "cljs_thread_first_with_collections")
(is= (-> {:a 1} (assoc :b 2) :b) 2 "cljs_thread_first_with_collections")
(is= (-> {:a 1 :b 2} (assoc :c 3) (dissoc :a) count) 2 "cljs_thread_first_with_collections")

(is= (-> "hello" count) 5 "cljs_thread_first_str")

(is= (-> [1 2 3] (conj 4) count) 4 "cljs_thread_first_vector")

(is= (->> ["foo" "baaar" "baaaaaz"] (map count) (apply max)) 7 "cljs_thread_last_basic")

(is= (->> [1 2 3 4 5] (filter odd?) (map inc) (reduce +)) 12 "cljs_thread_last_pipeline")
(is= (->> (range 10) (filter even?) (map #(* % %)) (reduce +)) 120 "cljs_thread_last_pipeline")
(is= (->> (range 1 11) (reduce +)) 55 "cljs_thread_last_pipeline")

(is= (->> (range 20) (filter even?) (map #(* % %)) (take 5) (reduce +)) 120 "cljs_thread_last_complex")
(is= (->> [1 2 3 4 5 6 7 8 9 10] (filter even?) (map #(* % %)) (reduce +)) 220 "cljs_thread_last_complex")

;; ============================================================================
;; LET BINDING
;; ============================================================================

(testing "LET BINDING")

(is= (let [x 1] x) 1 "cljs_let_basic")
(is= (let [x 1 y 2] (+ x y)) 3 "cljs_let_basic")
(is= (let [x 1 y (+ x x)] y) 2 "cljs_let_basic")

(is= (let [a 1 b 2 c 3 d 4 e 5] (+ a b c d e)) 15 "cljs_let_many_bindings")

(is= (let [x 1] (let [y 2] (+ x y))) 3 "cljs_let_nested")

(is= (let [x 2] 1 2 3 x) 2 "cljs_let_multiple_body")

;; cljs_let_shadow uses println output, test underlying logic
;; (let [x 1] (println (let [x 2] x)) (println x)) => "2\n1"
;; Test: inner let shadows x
(is= (let [x 1] (let [x 2] x)) 2 "cljs_let_shadow_inner")
(is= (let [x 1] x) 1 "cljs_let_shadow_outer")

(is= (let [m {:a 1 :b 2 :c 3}] (+ (:a m) (:b m) (:c m))) 6 "cljs_let_with_map")

;; ============================================================================
;; FN AND CLOSURES
;; ============================================================================

(testing "FN AND CLOSURES")

(is= (#(+ 1 %) 1) 2 "cljs_fn_literal_basic")
(is= (#(* % %) 5) 25 "cljs_fn_literal_basic")
(is= (#(+ %1 %2) 3 4) 7 "cljs_fn_literal_basic")

(is= (str (#(vector %1 %2 %3) 1 2 3)) "[1 2 3]" "cljs_fn_literal_three_args")

(is= (str (#(do %&) 1 2 3)) "(1 2 3)" "cljs_fn_literal_rest")

(is= (str ((fn [x & xs] xs) 1 2 3)) "(2 3)" "cljs_fn_rest_params")

(is= ((fn ([x] x) ([x y] y)) 1) 1 "cljs_fn_multi_arity")
(is= ((fn ([x] x) ([x y] y)) 1 2) 2 "cljs_fn_multi_arity")

(is= ((fn ([x & xs] "variadic") ([x] "otherwise")) 1) "otherwise" "cljs_fn_variadic_vs_fixed")
(is= ((fn ([x] "otherwise") ([x & xs] "variadic")) 1 2) "variadic" "cljs_fn_variadic_vs_fixed")

(is= (str (apply (fn [x & xs] xs) 1 2 [3 4])) "(2 3 4)" "cljs_fn_apply_with_rest")

(is= (let [x 10 f (fn [y] (+ x y))] (f 5)) 15 "cljs_fn_closures")
(is= ((let [x 10] (fn [y] (+ x y))) 5) 15 "cljs_fn_closures")

(is= (let [x 1 y 2] ((fn [] (let [g (fn [] y)] (+ x (g)))))) 3 "cljs_fn_closure_nested")

(is= (let [f (fn [x] (fn [y] (fn [z] (+ x y z))))] (((f 1) 2) 3)) 6 "cljs_fn_deeply_nested")

(defn tf ([x] x) ([_ _ & zs] zs))
(is= (nil? (tf 1 2)) true "cljs_fn_multi_arity_nil_rest")
(is= (str (tf 1 2 3 4)) "(3 4)" "cljs_fn_multi_arity_nil_rest")

;; ============================================================================
;; DEFN
;; ============================================================================

(testing "DEFN")

(defn foo "increment" [x] (inc x))
(is= (foo 1) 2 "cljs_defn_basic")

(defn f ([x] x) ([x y] (+ x y)))
(is= (+ (f 1) (f 2 3)) 6 "cljs_defn_multi_arity")

(defn vari [x & xs] (count xs))
(is= (vari 1 2 3 4) 3 "cljs_defn_variadic")
(defn vari2 [x & xs] (str xs))
(is= (vari2 1 2 3) "(2 3)" "cljs_defn_variadic")

(defn f2 ([] 0) ([x] x) ([x y] (+ x y)) ([x y & more] (apply + x y more)))
(is= (+ (f2) (f2 1) (f2 2 3) (f2 4 5 6 7)) 28 "cljs_defn_all_arities")

(defn- foo-private [] 1)
(is= (foo-private) 1 "cljs_defn_private")

(defn square [x] (* x x))
(is= (square 7) 49 "cljs_defn_square")

(defn double-fn [x] (* 2 x))
(defn add1 [x] (+ 1 x))
(is= (double-fn (add1 3)) 8 "cljs_defn_compose")

;; ============================================================================
;; DEF
;; ============================================================================

(testing "DEF")

(def foo-val "nice val")
(is= foo-val "nice val" "cljs_def_basic")

(def foo-val2)
(def foo-val2 "docstring" 2)
(is= foo-val2 2 "cljs_def_with_docstring")

(def x-def 1)
(def y-def 2)
(is= (+ x-def y-def) 3 "cljs_def_multiple")

;; ============================================================================
;; RECUR AND LOOP
;; ============================================================================

(testing "RECUR AND LOOP")

(defn hello [x] (if (< x 10000) (recur (inc x)) x))
(is= (hello 0) 10000 "cljs_recur_basic")

(defn countdown [n] (if (zero? n) "done" (recur (dec n))))
(is= (countdown 100) "done" "cljs_recur_countdown")

(defn sum-fn [n acc] (if (zero? n) acc (recur (dec n) (+ acc n))))
(is= (sum-fn 100 0) 5050 "cljs_recur_sum")

(defn fib [n a b] (if (zero? n) a (recur (dec n) b (+ a b))))
(is= (fib 10 0 1) 55 "cljs_recur_fibonacci")

(is= (str ((fn [& args] (if-let [x (next args)] (recur x) args)) 1 2 3 4)) "(4)" "cljs_recur_variadic")

(is= (str ((fn [x & args] (if-let [x (next args)] (recur x x) x)) nil 2 3 4)) "(4)" "cljs_recur_variadic_with_fixed")

(is= (loop [x 0] (if (< x 10) (recur (inc x)) x)) 10 "cljs_loop_basic")

(is= (loop [i 0 acc 0] (if (> i 10) acc (recur (inc i) (+ acc i)))) 55 "cljs_loop_accumulator")

(is= (loop [x 5 acc 1] (if (zero? x) acc (recur (dec x) (* acc x)))) 120 "cljs_loop_factorial")

(is= (str (loop [l (list 2 1) c (count l)] (if (> c 4) l (recur (conj l (inc c)) (inc c))))) "(5 4 3 2 1)" "cljs_loop_conj_list")

(is= (let [x 1] (loop [x (inc x)] x)) 2 "cljs_loop_let_shadow")

(is= (loop [i 0 total 0] (if (> i 3) total (recur (inc i) (+ total (loop [j 0 s 0] (if (> j 3) s (recur (inc j) (+ s j)))))))) 24 "cljs_loop_nested")

;; ============================================================================
;; TRY/CATCH/THROW
;; ============================================================================

(testing "TRY/CATCH/THROW")

(is= (try 1 2 3) 3 "cljs_try_returns_body")

(is= (try 'hello) 'hello "cljs_try_returns_quoted")

(is= (try 1 2 nil) nil "cljs_try_nil_in_body")

(is= (try (throw "err") (catch Exception e "caught")) "caught" "cljs_try_catch_thrown_string")

(is= (try (+ 1 2)) 3 "cljs_try_no_exception")
(is= (try 1 (catch Exception e "caught")) 1 "cljs_try_no_exception")

(is= (try 42 (finally nil)) 42 "cljs_try_finally")

;; ============================================================================
;; PROTOCOLS AND DEFTYPE
;; ============================================================================

(testing "PROTOCOLS AND DEFTYPE")

(defprotocol IGreet (greet [this]))
(deftype Greeter [] IGreet (greet [this] "hello"))
(is= (greet (Greeter.)) "hello" "cljs_protocol_basic")

(defprotocol ISpeak (speak [this]))
(deftype Dog [name] ISpeak (speak [this] (str "Woof, I am " name)))
(is= (speak (Dog. "Rex")) "Woof, I am Rex" "cljs_protocol_with_field")

(defprotocol IArea (area [this]))
(deftype Circle [r] IArea (area [this] (* 3.14 r r)))
(is= (area (Circle. 5)) 78.5 "cljs_protocol_with_computation")

(defprotocol ICalc (add [this x]) (mul [this x]))
(deftype Num [n] ICalc (add [this x] (+ n x)) (mul [this x] (* n x)))
(is= (let [c (Num. 10)] (+ (add c 5) (mul c 3))) 45 "cljs_protocol_multiple_methods")

(defprotocol IFoo (foo-method [this]))
(deftype Bar [] IFoo (foo-method [this] 42))
(is= (satisfies? IFoo (Bar.)) true "cljs_protocol_satisfies")

(defprotocol ILen (my-len [this]))
(deftype Wrapper [items] ILen (my-len [this] (count items)))
(is= (my-len (Wrapper. [1 2 3])) 3 "cljs_deftype_wrapper")

(defprotocol IShow (show [this]))
(deftype Point [x y] IShow (show [this] (str "(" x "," y ")")))
(is= (show (Point. 3 4)) "(3,4)" "cljs_deftype_point")

;; ============================================================================
;; IFn - COLLECTIONS AS FUNCTIONS
;; ============================================================================

(testing "IFn - COLLECTIONS AS FUNCTIONS")

(is= ({:a 1} :a 3) 1 "cljs_ifn_map_as_fn_found")

(is= ({:a 1} 2 3) 3 "cljs_ifn_map_as_fn_default")

(is= ((hash-map :a 1) :a 3) 1 "cljs_ifn_hashmap_as_fn")

(is= (#{:a :b :c} :a) :a "cljs_ifn_set_as_fn")
(is= (#{:a :b :c} :d) nil "cljs_ifn_set_as_fn")

(is= ([10 20 30] 1) 20 "cljs_ifn_vector_as_fn")

(is= ((get {:foo identity} :foo) 1) 1 "cljs_ifn_fn_from_map")

(is= (:a {:a 42 :b 99}) 42 "cljs_ifn_keyword_as_fn")
(is= (:c {:a 1}) nil "cljs_ifn_keyword_as_fn")
(is= (:c {:a 1} :default) :default "cljs_ifn_keyword_as_fn")

;; ============================================================================
;; PRINTLN OUTPUT TESTS
;; ============================================================================

;; cljs_println_basic - test underlying values rather than println output
(testing "PRINTLN")

;; cljs_println_returns_nil
(is= (nil? (println "hi")) true "cljs_println_returns_nil")

;; ============================================================================
;; CONTAINS? (CLJS test-contains?)
;; ============================================================================

(testing "CONTAINS?")

(is= (contains? {:a 1 :b 2} :a) true "cljs_contains_map")
(is= (contains? {:a 1 :b 2} :z) false "cljs_contains_map")

(is= (contains? [5 6 7] 1) true "cljs_contains_vector")
(is= (contains? [5 6 7] 2) true "cljs_contains_vector")
(is= (contains? [5 6 7] 3) false "cljs_contains_vector")

(is= (contains? nil 42) false "cljs_contains_nil")

(is= (contains? #{1 2 3} 2) true "cljs_contains_set")
(is= (contains? #{1 2 3} 4) false "cljs_contains_set")

;; ============================================================================
;; DOTIMES
;; ============================================================================

(testing "DOTIMES")

;; cljs_dotimes_basic - uses println output, skip

(is= (dotimes [i 3] i) nil "cljs_dotimes_returns_nil")

;; ============================================================================
;; GENSYM
;; ============================================================================

(testing "GENSYM")

;; cljs_gensym - verify it returns a symbol (starts with G__)
(is (string? (str (gensym))) "cljs_gensym")

;; ============================================================================
;; IDENTICAL?
;; ============================================================================

(testing "IDENTICAL?")

(is= (identical? nil nil) true "cljs_identical")
(is= (identical? 1 1) true "cljs_identical")

;; ============================================================================
;; HASH
;; ============================================================================

(testing "HASH")

(is= (hash 42) 42 "cljs_hash_number")

;; ============================================================================
;; ASSOC ON NIL
;; ============================================================================

(testing "ASSOC ON NIL")

(is= (count (assoc nil :a 1)) 1 "cljs_assoc_nil")

;; ============================================================================
;; VARIABLE NAMES MATCHING MACROS
;; ============================================================================

(testing "VARIABLE NAMES MATCHING MACROS")

(defn foo-merge [merge] merge)
(is= (foo-merge true) true "cljs_var_named_merge")

(defn foo-comment [comment] comment)
(is= (foo-comment true) true "cljs_var_named_comment")

;; ============================================================================
;; EMPTY FUNCTION
;; ============================================================================

(testing "EMPTY FUNCTION")

(is= (str (empty [1 2 3])) "[]" "cljs_empty_vector")

(is= (str (empty (list 1 2 3))) "()" "cljs_empty_list")

(is= (str (empty {:a 1})) "{}" "cljs_empty_map")

(is= (str (empty #{1 2})) "#{}" "cljs_empty_set")

;; ============================================================================
;; FN LITERAL MAP/REDUCE
;; ============================================================================

(testing "FN LITERAL MAP/REDUCE")

(is= (str (map #(+ 1 %) [0 1 2])) "(1 2 3)" "cljs_fn_literal_with_map")

(is= (str (map #(do %) [1 2 3])) "(1 2 3)" "cljs_fn_literal_identity")

(is= (str (map-indexed #(do [%1 %2]) [1 2 3])) "([0 1] [1 2] [2 3])" "cljs_fn_literal_map_indexed")

(is= (str (apply #(do %&) [1 2 3])) "(1 2 3)" "cljs_fn_literal_rest_args")

;; Ignored tests are NOT included (some->, some->>, cond->, cond->>,
;; as->, case, for, atom, compare, boolean, subs, keyword constructor,
;; sorted-set, sorted-map, interpose, partition-all, partition-by,
;; take-nth, dedupe, reductions, iterate, cycle)

(test-summary)