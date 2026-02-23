(load-file "tests/clj/test_helper.clj")

;; ============================================================================
;; SEQUENCE TESTS (from sequences.clj)
;; ============================================================================

;; -- first --

(testing "first")

(is= (first nil) nil "clj-seq-first-nil")
(is= (first []) nil "clj-seq-first-empty-vec")
(is= (first [1]) 1 "clj-seq-first-single-vec")
(is= (first [1 2 3]) 1 "clj-seq-first-multi-vec")
(is= (first [nil]) nil "clj-seq-first-nil-in-vec")
(is= (first [[1 2] [3 4]]) [1 2] "clj-seq-first-nested-vec")
(is= (first (list 1 2 3)) 1 "clj-seq-first-list")
(is= (first (list)) nil "clj-seq-first-empty-list")

;; -- next --

(testing "next")

(is= (next nil) nil "clj-seq-next-nil")
(is= (next []) nil "clj-seq-next-empty-vec")
(is= (next [1]) nil "clj-seq-next-single-vec")
(is= (str (next [1 2 3])) "(2 3)" "clj-seq-next-multi-vec")
(is= (str (next (list 1 2 3))) "(2 3)" "clj-seq-next-list")
(is= (next (list 1)) nil "clj-seq-next-single-list")

;; -- rest --

(testing "rest")

(is= (rest nil) nil "clj-seq-rest-nil")
(is= (rest []) nil "clj-seq-rest-empty-vec")
(is= (str (rest [1 2 3])) "(2 3)" "clj-seq-rest-multi-vec")
(is= (str (rest [1])) "()" "clj-seq-rest-single-vec")

;; -- last --

(testing "last")

(is= (last nil) nil "clj-seq-last-nil")
(is= (last []) nil "clj-seq-last-empty-vec")
(is= (last [1]) 1 "clj-seq-last-single")
(is= (last [1 2 3]) 3 "clj-seq-last-multi")
(is= (last [1 nil]) nil "clj-seq-last-nil-in-vec")
(is= (last [[] nil]) nil "clj-seq-last-nested")
(is= (last (list 1 2 3)) 3 "clj-seq-last-list")

;; -- second / ffirst --

(testing "second / ffirst")

(is= (second [1 2 3]) 2 "clj-seq-second-vec")
(is= (second nil) nil "clj-seq-second-nil")
(is= (second [1]) nil "clj-seq-second-single")
(is= (first (first [[1 2] [3 4]])) 1 "clj-seq-ffirst-nested")
(is= (second (first [[1 2] [3 4]])) 2 "clj-seq-second-of-first")

;; -- cons --

(testing "cons")

(is= (str (cons 1 nil)) "(1)" "clj-seq-cons-nil")
(is= (str (cons 1 [2 3])) "(1 2 3)" "clj-seq-cons-vec")
(is= (str (cons 1 (list 2 3))) "(1 2 3)" "clj-seq-cons-list")
(is= (str (cons 1 [])) "(1)" "clj-seq-cons-empty-vec")
(is= (str (cons nil nil)) "(nil)" "clj-seq-cons-nil-to-nil")

;; -- conj --

(testing "conj")

(is= (str (conj nil 1)) "(1)" "clj-seq-conj-nil")
(is= (conj [1 2] 3) [1 2 3] "clj-seq-conj-vec")
(is= (str (conj (list 2 3) 1)) "(1 2 3)" "clj-seq-conj-list")
(is= (conj [] 1) [1] "clj-seq-conj-empty-vec")
(is= (contains? (conj #{1 2} 3) 3) true "clj-seq-conj-set")
(is= (get (conj {:a 1} [:b 2]) :b) 2 "clj-seq-conj-map-entry")

;; -- empty --

(testing "empty")

(is= (empty nil) nil "clj-seq-empty-nil")
(is= (empty [1 2]) [] "clj-seq-empty-vec")
(is= (count (empty {:a 1})) 0 "clj-seq-empty-map")

;; -- count --

(testing "count")

(is= (count nil) 0 "clj-seq-count-nil")
(is= (count []) 0 "clj-seq-count-empty-vec")
(is= (count [1 2 3]) 3 "clj-seq-count-vec")
(is= (count (list 1 2 3)) 3 "clj-seq-count-list")
(is= (count {:a 1 :b 2}) 2 "clj-seq-count-map")
(is= (count #{1 2 3}) 3 "clj-seq-count-set")
(is= (count "abc") 3 "clj-seq-count-string")

;; -- nth --

(testing "nth")

(is= (nth [1 2 3] 0) 1 "clj-seq-nth-vec")
(is= (nth [1 2 3] 1) 2 "clj-seq-nth-vec-middle")
(is= (nth [1 2 3] 2) 3 "clj-seq-nth-vec-last")
;; clj_seq_nth_list - SKIPPED (ignored)
(is= (nth [1 2 3] 5 :not-found) :not-found "clj-seq-nth-with-default")

;; -- map --

(testing "map")

(is= (str (map inc [1 2 3])) "(2 3 4)" "clj-seq-map-inc")
(is= (map inc nil) nil "clj-seq-map-nil")
(is= (str (map identity [1 2 3])) "(1 2 3)" "clj-seq-map-identity")
(is= (str (map + [1 2 3] [10 20 30])) "(11 22 33)" "clj-seq-map-add-two-colls")
(is= (vec (map inc [1 2 3])) [2 3 4] "clj-seq-map-to-vec")
(is= (into [] (map inc [1 2 3])) [2 3 4] "clj-seq-map-into-vec")

;; -- filter --

(testing "filter")

(is= (str (filter even? [1 2 3 4 5])) "(2 4)" "clj-seq-filter-even")
(is= (str (filter odd? [1 2 3 4 5])) "(1 3 5)" "clj-seq-filter-odd")
(is= (filter even? nil) nil "clj-seq-filter-nil")
(is= (filter even? [1 3 5]) nil "clj-seq-filter-none-match")
(is= (str (filter even? [2 4 6])) "(2 4 6)" "clj-seq-filter-all-match")

;; -- remove --

(testing "remove")

(is= (str (remove even? [1 2 3 4 5])) "(1 3 5)" "clj-seq-remove-even")

;; -- keep --

(testing "keep")

(is= (str (keep identity [1 nil 2 nil 3])) "(1 2 3)" "clj-seq-keep-identity")

;; -- reduce --

(testing "reduce")

(is= (reduce + [1 2 3 4 5]) 15 "clj-seq-reduce-plus")
(is= (reduce + 10 [1 2 3 4 5]) 25 "clj-seq-reduce-plus-init")
(is= (reduce + 100 [1 2 3]) 106 "clj-seq-reduce-plus-init-100")
(is= (reduce conj [] [1 2 3]) [1 2 3] "clj-seq-reduce-conj-vec")
(is= (reduce (fn [acc x] (+ acc x)) 0 [1 2 3 4 5]) 15 "clj-seq-reduce-fn")
(is= (reduce + (range 10)) 45 "clj-seq-reduce-range")
(is= (reduce + (range 100)) 4950 "clj-seq-reduce-range-100")
(is= (reduce + 0 (range 100)) 4950 "clj-seq-reduce-range-init")
(is= (reduce + (map inc (range 10))) 55 "clj-seq-reduce-map-range")
(is= (reduce + (filter even? (range 10))) 20 "clj-seq-reduce-filter-range")
(is= (reduce + (filter odd? (range 10))) 25 "clj-seq-reduce-filter-odd-range")

;; -- take / drop --

(testing "take / drop")

(is= (str (take 3 [1 2 3 4 5])) "(1 2 3)" "clj-seq-take-3")
(is= (str (take 1 [1 2 3 4 5])) "(1)" "clj-seq-take-1")
(is= (str (take 5 [1 2 3 4 5])) "(1 2 3 4 5)" "clj-seq-take-5")
(is= (str (take 9 [1 2 3 4 5])) "(1 2 3 4 5)" "clj-seq-take-9-from-5")
(is= (take 0 [1 2 3 4 5]) nil "clj-seq-take-0")
(is= (str (drop 1 [1 2 3 4 5])) "(2 3 4 5)" "clj-seq-drop-1")
(is= (str (drop 3 [1 2 3 4 5])) "(4 5)" "clj-seq-drop-3")
(is= (drop 5 [1 2 3 4 5]) nil "clj-seq-drop-5")
(is= (drop 9 [1 2 3 4 5]) nil "clj-seq-drop-9")
(is= (str (drop 0 [1 2 3 4 5])) "(1 2 3 4 5)" "clj-seq-drop-0")

;; -- take-while / drop-while --

(testing "take-while / drop-while")

(is= (str (take-while pos? [1 2 3 -1 4])) "(1 2 3)" "clj-seq-take-while-pos")
(is= (take-while pos? [-1 1 2]) nil "clj-seq-take-while-none")
(is= (str (take-while pos? [1 2 3 4])) "(1 2 3 4)" "clj-seq-take-while-all")
(is= (str (drop-while pos? [1 2 3 -1 4])) "(-1 4)" "clj-seq-drop-while-pos")
(is= (str (drop-while pos? [-1 1 2 3])) "(-1 1 2 3)" "clj-seq-drop-while-none")
(is= (drop-while pos? [1 2 3 4]) nil "clj-seq-drop-while-all")

;; -- concat --

(testing "concat")

(is= (str (concat [1 2] [3 4])) "(1 2 3 4)" "clj-seq-concat-two")
(is= (str (concat [1 2] [3 4] [5 6])) "(1 2 3 4 5 6)" "clj-seq-concat-three")
(is= (str (concat [1 2])) "(1 2)" "clj-seq-concat-single")
(is= (count (concat [1 2] [3 4])) 4 "clj-seq-concat-count")
(is= (into [] (concat [1 2] [3 4])) [1 2 3 4] "clj-seq-concat-into-vec")

;; -- interleave --

(testing "interleave")

(is= (str (interleave [1 2 3] [4 5 6])) "(1 4 2 5 3 6)" "clj-seq-interleave-two")
(is= (str (interleave [1] [3 4])) "(1 3)" "clj-seq-interleave-unequal-first-shorter")
(is= (str (interleave [1 2] [3])) "(1 3)" "clj-seq-interleave-unequal-second-shorter")
(is= (apply str (interleave [1 2 3] [:a :b :c])) "1:a2:b3:c" "clj-seq-interleave-apply-str")

;; -- distinct --

(testing "distinct")

(is= (str (distinct [1 2 1 3 2])) "(1 2 3)" "clj-seq-distinct-basic")
(is= (str (distinct [1 1 1])) "(1)" "clj-seq-distinct-all-same")
(is= (str (distinct [1 2 3])) "(1 2 3)" "clj-seq-distinct-already-unique")
(is= (count (distinct [1 1 2 2 3])) 3 "clj-seq-distinct-count")
(is= (first (distinct [3 1 2 1])) 3 "clj-seq-distinct-preserves-order")

;; -- reverse --

(testing "reverse")

(is= (str (reverse [1 2 3])) "(3 2 1)" "clj-seq-reverse-vec")
(is= (str (reverse [1])) "(1)" "clj-seq-reverse-single")
(is= (str (reverse [])) "()" "clj-seq-reverse-empty")
(is= (last (reverse [1 2 3])) 1 "clj-seq-reverse-last")

;; -- sort --

(testing "sort")

(is= (str (sort [3 1 2])) "(1 2 3)" "clj-seq-sort-vec")
(is= (str (sort [1 2 3])) "(1 2 3)" "clj-seq-sort-already-sorted")
(is= (str (sort [3 2 1])) "(1 2 3)" "clj-seq-sort-reverse")
(is= (str (sort [3 1 4 1 5 9 2 6])) "(1 1 2 3 4 5 6 9)" "clj-seq-sort-with-dups")
(is= (first (sort [3 1 2])) 1 "clj-seq-sort-first")
(is= (last (sort [3 1 2])) 3 "clj-seq-sort-last")
(is= (str (sort (distinct [3 1 2 1 3]))) "(1 2 3)" "clj-seq-sort-distinct")

;; -- butlast --

(testing "butlast")

(is= (str (butlast [1 2 3])) "(1 2)" "clj-seq-butlast-vec")
(is= (butlast [1]) nil "clj-seq-butlast-single")
(is= (butlast []) nil "clj-seq-butlast-empty")

;; -- partition --

(testing "partition")

(is= (str (partition 2 [1 2 3 4])) "((1 2) (3 4))" "clj-seq-partition-2")
(is= (str (partition 2 [1 2 3])) "((1 2))" "clj-seq-partition-2-odd")
(is= (str (partition 3 [1 2 3 4 5 6 7 8])) "((1 2 3) (4 5 6))" "clj-seq-partition-3")
(is= (str (partition 2 3 [1 2 3 4 5 6 7])) "((1 2) (4 5))" "clj-seq-partition-with-step")
(is= (str (partition 2 3 [1 2 3 4 5 6 7 8])) "((1 2) (4 5) (7 8))" "clj-seq-partition-step-equals-size")
(is= (str (partition 1 [1 2 3])) "((1) (2) (3))" "clj-seq-partition-1")
(is= (partition 5 [1 2 3]) nil "clj-seq-partition-larger-than-coll")
(is= (count (partition 2 [1 2 3 4 5])) 2 "clj-seq-partition-count")
(is= (str (first (partition 2 [1 2 3 4]))) "(1 2)" "clj-seq-partition-first")

;; -- flatten --

(testing "flatten")

(is= (str (flatten [[1 2] [3 [4 5]]])) "(1 2 3 4 5)" "clj-seq-flatten-nested")
(is= (str (flatten [1 [2 [3 [4]]]])) "(1 2 3 4)" "clj-seq-flatten-deeply-nested")
(is= (str (flatten [1 2 3])) "(1 2 3)" "clj-seq-flatten-already-flat")

;; -- mapcat --

(testing "mapcat")

(is= (str (mapcat list [1 2 3])) "(1 2 3)" "clj-seq-mapcat-list")
(is= (str (mapcat reverse [[1 2] [3 4]])) "(2 1 4 3)" "clj-seq-mapcat-reverse")

;; -- apply --

(testing "apply")

(is= (apply + [1 2 3]) 6 "clj-seq-apply-plus")
(is= (apply str [1 2 3]) "123" "clj-seq-apply-str")
(is= (str (apply list [1 2 3])) "(1 2 3)" "clj-seq-apply-list")
(is= (apply vector [1 2 3]) [1 2 3] "clj-seq-apply-vector")
(is= (str (apply concat [[1 2] [3 4] [5 6]])) "(1 2 3 4 5 6)" "clj-seq-apply-concat")

;; -- map-indexed --

(testing "map-indexed")

(is= (str (map-indexed (fn [i v] (+ i v)) [10 20 30])) "(10 21 32)" "clj-seq-map-indexed")

;; -- empty? --

(testing "empty?")

(is= (empty? nil) true "clj-seq-empty-check-nil")
(is= (empty? []) true "clj-seq-empty-check-empty-vec")
(is= (empty? [1]) false "clj-seq-empty-check-nonempty-vec")
(is= (empty? {}) true "clj-seq-empty-check-empty-map")
(is= (empty? {:a 1}) false "clj-seq-empty-check-nonempty-map")

;; -- every? / some / not-every? / not-any? --

(testing "every? / some / not-every? / not-any?")

(is= (every? pos? [1 2 3]) true "clj-seq-every-true")
(is= (every? pos? [1 -2 3]) false "clj-seq-every-false")
(is= (every? pos? []) true "clj-seq-every-empty")
(is= (every? pos? nil) true "clj-seq-every-nil")
(is= (every? pos? [-1 -2]) false "clj-seq-every-all-neg")
(is= (some pos? [1 2 3]) true "clj-seq-some-found")
(is= (some pos? [-1 -2]) nil "clj-seq-some-not-found")
(is= (some pos? nil) nil "clj-seq-some-nil-input")
(is= (some #{:a} [:b :a :c]) :a "clj-seq-some-set-as-fn")
(is= (some #{:a} [:b :c]) nil "clj-seq-some-set-not-found")
(is= (some even? [1 2 3]) true "clj-seq-some-even")
(is= (some even? [1 3 5]) nil "clj-seq-some-even-not-found")
(is= (not-every? pos? [1 -2 3]) true "clj-seq-not-every-true")
(is= (not-every? pos? [1 2 3]) false "clj-seq-not-every-false")
(is= (not-every? pos? []) false "clj-seq-not-every-empty")
(is= (not-any? pos? [-1 -2]) true "clj-seq-not-any-true")
(is= (not-any? pos? [1 2 3]) false "clj-seq-not-any-false")
(is= (not-any? pos? [-1 -2 3]) false "clj-seq-not-any-mixed")
(is= (not-any? pos? []) true "clj-seq-not-any-empty")

;; -- range --

(testing "range")

(is= (str (range 5)) "(0 1 2 3 4)" "clj-seq-range-5")
(is= (range 0) nil "clj-seq-range-0")
(is= (str (range 1)) "(0)" "clj-seq-range-1")
(is= (str (range 2 5)) "(2 3 4)" "clj-seq-range-start-end")
(is= (str (range 0 10 2)) "(0 2 4 6 8)" "clj-seq-range-start-end-step")
(is= (str (range 0 10 3)) "(0 3 6 9)" "clj-seq-range-start-end-step-3")
(is= (str (range 3 6)) "(3 4 5)" "clj-seq-range-neg")
(is= (str (range -2 3)) "(-2 -1 0 1 2)" "clj-seq-range-neg-values")
(is= (into [] (range 5)) [0 1 2 3 4] "clj-seq-range-into-vec")
(is= (count (range 10)) 10 "clj-seq-range-count")
(is= (last (range 5)) 4 "clj-seq-range-last")
(is= (reduce + (range 1 101)) 5050 "clj-seq-range-sum")

;; -- repeat --

(testing "repeat")

(is= (str (repeat 3 :x)) "(:x :x :x)" "clj-seq-repeat-3")
(is= (repeat 0 :x) nil "clj-seq-repeat-0")
(is= (str (repeat 1 :x)) "(:x)" "clj-seq-repeat-1")
(is= (str (repeat 5 7)) "(7 7 7 7 7)" "clj-seq-repeat-5")
(is= (repeat -1 7) nil "clj-seq-repeat-neg")

;; -- into --

(testing "into")

(is= (into [] (list 1 2 3)) [1 2 3] "clj-seq-into-vec-from-list")
(is= (into [] nil) [] "clj-seq-into-vec-from-nil")
(is= (into [] (range 5)) [0 1 2 3 4] "clj-seq-into-vec-from-range")
(is= (into [] (filter odd? (range 10))) [1 3 5 7 9] "clj-seq-into-vec-filter")
(is= (into [] (take 5 (range 100))) [0 1 2 3 4] "clj-seq-into-vec-take")
(is= (into [] (drop 5 (range 10))) [5 6 7 8 9] "clj-seq-into-vec-drop")
(is= (count (into #{} [1 2 3 2 1])) 3 "clj-seq-into-set-count")

;; -- group-by / frequencies --

(testing "group-by / frequencies")

(is= (get (group-by even? [1 2 3 4 5]) true) [2 4] "clj-seq-group-by-even")
(is= (get (group-by even? [1 2 3 4 5]) false) [1 3 5] "clj-seq-group-by-odd")
(is= (get (frequencies [1 1 2 2 2 3]) 2) 3 "clj-seq-frequencies-count")
(is= (get (frequencies [1 1 2 2 2 3]) 1) 2 "clj-seq-frequencies-single")
(is= (get (frequencies [1 1 2 2 2 3]) 3) 1 "clj-seq-frequencies-unique")

;; -- thread macros --

(testing "thread macros")

(is= (-> 1 inc inc inc) 4 "clj-seq-thread-first")
(is= (str (->> [1 2 3] (map inc) (filter even?))) "(2 4)" "clj-seq-thread-last")
(is= (->> (range 10) (map inc) (reduce +)) 55 "clj-seq-thread-last-reduce")

;; -- higher order fns --

(testing "higher order fns")

(is= ((comp inc inc) 0) 2 "clj-seq-comp")
(is= ((partial + 10) 5) 15 "clj-seq-partial")
(is= ((juxt inc dec) 5) [6 4] "clj-seq-juxt")

;; ============================================================================
;; DATA STRUCTURE TESTS (from data_structures.clj)
;; ============================================================================

;; -- maps --

(testing "maps")

(is= (get {:a 1 :b 2} :a) 1 "clj-ds-map-get")
(is= (get {:a 1 :b 2} :c) nil "clj-ds-map-get-missing")
(is= (get {:a 1 :b 2} :c :default) :default "clj-ds-map-get-default")
(is= (:a {:a 1 :b 2}) 1 "clj-ds-map-keyword-lookup")
(is= ({:a 1 :b 2} :a) 1 "clj-ds-map-as-fn")
(is= (get (assoc {:a 1} :b 2) :b) 2 "clj-ds-map-assoc")
(is= (get (assoc {:a 1} :a 2) :a) 2 "clj-ds-map-assoc-overwrite")
(is= (count (assoc {:a 1 :b 2} :c 3 :d 4)) 4 "clj-ds-map-assoc-multi")
(is= (get (dissoc {:a 1 :b 2} :a) :a) nil "clj-ds-map-dissoc")
(is= (get (dissoc {:a 1 :b 2} :a) :b) 2 "clj-ds-map-dissoc-retains")
(is= (count (dissoc {:a 1 :b 2 :c 3} :a :c)) 1 "clj-ds-map-dissoc-multi")
(is= (contains? {:a 1} :a) true "clj-ds-map-contains")
(is= (contains? {:a 1} :b) false "clj-ds-map-contains-missing")
(is= (contains? {:a 1} nil) false "clj-ds-map-contains-nil-key")
(is= (str (keys {:a 1})) "(:a)" "clj-ds-map-keys")
(is= (count (keys {:a 1 :b 2})) 2 "clj-ds-map-keys-count")
(is= (str (vals {:a 1})) "(1)" "clj-ds-map-vals")
(is= (count (vals {:a 1 :b 2})) 2 "clj-ds-map-vals-count")
(is= (key (first {:a 1})) :a "clj-ds-map-key-fn")
(is= (val (first {:a 1})) 1 "clj-ds-map-val-fn")
(is= (count (merge {:a 1} {:b 2} {:c 3})) 3 "clj-ds-map-merge-count")
(is= (get (merge {:a 1} {:a 2}) :a) 2 "clj-ds-map-merge-overwrite")
(is= (get (update {:a 1} :a inc) :a) 2 "clj-ds-map-update")
(is= (get-in {:a {:b 2}} [:a :b]) 2 "clj-ds-map-get-in")
(is= (get-in {:a {:b 2}} [:a :c]) nil "clj-ds-map-get-in-missing")
(is= (get-in {:a {:b 2}} [:a :c] 0) 0 "clj-ds-map-get-in-default")
(is= (get-in (assoc-in {} [:a :b :c] 42) [:a :b :c]) 42 "clj-ds-map-assoc-in")
(is= (get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b]) 2 "clj-ds-map-update-in")
(is= (get (select-keys {:a 1 :b 2 :c 3} [:a :b]) :a) 1 "clj-ds-map-select-keys")
(is= (count (select-keys {:a 1 :b 2 :c 3} [:a :b])) 2 "clj-ds-map-select-keys-count")
(is= (get (select-keys {:a 1 :b 2 :c 3} [:a :b]) :c) nil "clj-ds-map-select-keys-missing")
(is= (get (zipmap [:a :b :c] [1 2 3]) :b) 2 "clj-ds-map-zipmap-get")
(is= (count (zipmap [:a :b :c] [1 2 3])) 3 "clj-ds-map-zipmap-count")
(is= (count (zipmap [:a] [1 2])) 1 "clj-ds-map-zipmap-shorter-keys")
(is= (count (zipmap [:a :b] [1])) 1 "clj-ds-map-zipmap-shorter-vals")
(is= (count (hash-map :a 1 :b 2)) 2 "clj-ds-map-hash-map-count")
(is= (get (hash-map :a 1 :b 2) :a) 1 "clj-ds-map-hash-map-get")
(is= (get (into {} [[:a 1] [:b 2]]) :b) 2 "clj-ds-map-into")
(is= (count (into {} [[:a 1] [:b 2]])) 2 "clj-ds-map-into-count")
(is= (count (reduce conj {} [[:a 1] [:b 2]])) 2 "clj-ds-map-reduce-conj")
(is= (get (conj {:a 1} [:b 2]) :b) 2 "clj-ds-map-conj-entry")
(is= (count {}) 0 "clj-ds-map-count-empty")
(is= (count (empty {:a 1})) 0 "clj-ds-map-empty")

;; -- sets --

(testing "sets")

(is= (contains? #{1 2 3} 2) true "clj-ds-set-contains")
(is= (contains? #{1 2 3} 4) false "clj-ds-set-contains-missing")
(is= (count #{1 2 3}) 3 "clj-ds-set-count")
(is= (contains? (conj #{1 2} 3) 3) true "clj-ds-set-conj")
(is= (contains? (disj #{1 2 3} 2) 2) false "clj-ds-set-disj")
(is= (count (disj #{1 2 3} 2)) 2 "clj-ds-set-disj-count")
(is= (count (set [1 2 3 2 1])) 3 "clj-ds-set-from-vec")
(is= (contains? (set [1 2 3]) 2) true "clj-ds-set-from-vec-contains")
(is= (count (hash-set 1 2 3)) 3 "clj-ds-set-hash-set-count")
(is= (contains? (hash-set 1 2 3) 2) true "clj-ds-set-hash-set-contains")
(is= (count (into #{} [1 2 3 2 1])) 3 "clj-ds-set-into-count")
(is= (contains? (into #{} [1 2 3]) 3) true "clj-ds-set-into-contains")
(is= (count #{}) 0 "clj-ds-set-empty")
(is= (#{1 2 3} 2) 2 "clj-ds-set-as-fn")
(is= (#{1 2 3} 4) nil "clj-ds-set-as-fn-missing")

;; -- contains? on vectors --

(testing "contains? on vectors")

(is= (contains? [1 2 3] 0) true "clj-ds-vec-contains-index")
(is= (contains? [1 2 3] 2) true "clj-ds-vec-contains-last-index")
(is= (contains? [1 2 3] 3) false "clj-ds-vec-contains-out-of-bounds")
(is= (contains? [1 2 3] -1) false "clj-ds-vec-contains-neg")
(is= (contains? [] 0) false "clj-ds-vec-contains-empty")

;; -- list operations --

(testing "list operations")

(is= (str (list 1 2 3)) "(1 2 3)" "clj-ds-list-create")
(is= (str (list)) "()" "clj-ds-list-create-empty")
(is= (list? (list 1 2)) true "clj-ds-list-is-list")
(is= (list? [1 2]) false "clj-ds-list-vec-is-not-list")
(is= (str (conj (list 2 3) 1)) "(1 2 3)" "clj-ds-list-conj-front")

;; ============================================================================
;; VECTOR TESTS (from vectors.clj)
;; ============================================================================

(testing "vectors")

(is= [1 2 3] [1 2 3] "clj-vec-create")
(is= [] [] "clj-vec-create-empty")
(is= (conj [1 2] 3) [1 2 3] "clj-vec-conj")
(is= (assoc [1 2 3] 1 99) [1 99 3] "clj-vec-assoc")
(is= (assoc [1 2 3] 0 99) [99 2 3] "clj-vec-assoc-first")
(is= (assoc [1 2 3] 2 99) [1 2 99] "clj-vec-assoc-last")
(is= (nth [10 20 30] 1) 20 "clj-vec-nth")
(is= (get [10 20 30] 1) 20 "clj-vec-get")
(is= (get [10 20 30] 5) nil "clj-vec-get-out-of-bounds")
(is= (get [10 20 30] 5 :not-found) :not-found "clj-vec-get-default")
(is= ([10 20 30] 1) 20 "clj-vec-as-fn")
(is= (count [1 2 3]) 3 "clj-vec-count")
(is= (count []) 0 "clj-vec-count-empty")
(is= (vector? [1 2]) true "clj-vec-is-vector")
(is= (vector? (list 1 2)) false "clj-vec-list-is-not-vector")
(is= (vec (list 1 2 3)) [1 2 3] "clj-vec-from-list")
(is= (vec nil) [] "clj-vec-from-nil")
(is= (vec (range 4)) [0 1 2 3] "clj-vec-from-range")
(is= [[1 2] [3 4]] [[1 2] [3 4]] "clj-vec-nested")
(is= (first [1 2 3]) 1 "clj-vec-first")
(is= (last [1 2 3]) 3 "clj-vec-last")
(is= (str [1 2 3]) "[1 2 3]" "clj-vec-str")
(is= (str []) "[]" "clj-vec-str-empty")
(is= (into [] [1 2 3]) [1 2 3] "clj-vec-into")
(is= (reduce + 10 [2 4 6]) 22 "clj-vec-reduce-kv")

;; ============================================================================
;; LOGIC TESTS (from logic.clj)
;; ============================================================================

;; -- if --

(testing "if")

(is= (if true :t :f) :t "clj-logic-if-true")
(is= (if false :t :f) :f "clj-logic-if-false")
(is= (if nil :t :f) :f "clj-logic-if-nil")
(is= (if 0 :t :f) :t "clj-logic-if-zero-is-truthy")
(is= (if "" :t :f) :t "clj-logic-if-empty-string-is-truthy")
(is= (if [] :t :f) :t "clj-logic-if-empty-vec-is-truthy")
(is= (if {} :t :f) :t "clj-logic-if-empty-map-is-truthy")
(is= (if :kw :t :f) :t "clj-logic-if-keyword-is-truthy")
(is= (if 42 :t :f) :t "clj-logic-if-number-is-truthy")
(is= (if true :t) :t "clj-logic-if-true-no-else")
(is= (if false :t) nil "clj-logic-if-false-no-else")
(is= (if nil :t) nil "clj-logic-if-nil-no-else")

;; -- nil punning --

(testing "nil punning")

(is= (if (first []) :no :yes) :yes "clj-logic-nil-punning-first-empty")
(is= (if (next [1]) :no :yes) :yes "clj-logic-nil-punning-next-single")
(is= (if (seq nil) :no :yes) :yes "clj-logic-nil-punning-seq-nil")
(is= (if (seq []) :no :yes) :yes "clj-logic-nil-punning-seq-empty")

;; -- and --

(testing "and")

(is= (and) true "clj-logic-and-empty")
(is= (and true) true "clj-logic-and-true")
(is= (and nil) nil "clj-logic-and-nil")
(is= (and false) false "clj-logic-and-false")
(is= (and true nil) nil "clj-logic-and-true-nil")
(is= (and true false) false "clj-logic-and-true-false")
(is= (and 1 true :kw) :kw "clj-logic-and-returns-last-truthy")
(is= (and 1 true :kw nil) nil "clj-logic-and-short-circuits-on-nil")
(is= (and 1 true :kw false) false "clj-logic-and-short-circuits-on-false")

;; -- or --

(testing "or")

(is= (or) nil "clj-logic-or-empty")
(is= (or true) true "clj-logic-or-true")
(is= (or nil) nil "clj-logic-or-nil")
(is= (or false) false "clj-logic-or-false")
(is= (or nil false true) true "clj-logic-or-nil-false-true")
(is= (or nil false 1 2) 1 "clj-logic-or-nil-false-1")
(is= (or nil false "abc" :kw) "abc" "clj-logic-or-nil-false-str")
(is= (or false nil) nil "clj-logic-or-false-nil")
(is= (or nil false) false "clj-logic-or-nil-false")
(is= (or nil nil nil false) false "clj-logic-or-nil-nil-nil-false")

;; -- not --

(testing "not")

(is= (not nil) true "clj-logic-not-nil")
(is= (not false) true "clj-logic-not-false")
(is= (not true) false "clj-logic-not-true")
(is= (not 0) false "clj-logic-not-zero")
(is= (not 42) false "clj-logic-not-number")
(is= (not "") false "clj-logic-not-empty-string")
(is= (not "abc") false "clj-logic-not-string")
(is= (not :kw) false "clj-logic-not-keyword")
(is= (not []) false "clj-logic-not-empty-vec")
(is= (not [1 2]) false "clj-logic-not-vec")
(is= (not {}) false "clj-logic-not-empty-map")
(is= (not {:a 1}) false "clj-logic-not-map")

;; -- some? --

(testing "some?")

(is= (some? nil) false "clj-logic-some-nil")
(is= (some? false) true "clj-logic-some-false")
(is= (some? 0) true "clj-logic-some-zero")
(is= (some? "abc") true "clj-logic-some-string")
(is= (some? []) true "clj-logic-some-vec")

;; -- cond --

(testing "cond")

(is= (cond true :a false :b) :a "clj-logic-cond-first-true")
(is= (cond false :a true :b) :b "clj-logic-cond-second-true")
(is= (cond false :a false :b :else :c) :c "clj-logic-cond-else")

;; -- when / when-not / if-not --

(testing "when / when-not / if-not")

(is= (when true 42) 42 "clj-logic-when-true")
(is= (when false 42) nil "clj-logic-when-false")
(is= (when-not true 42) nil "clj-logic-when-not-true")
(is= (when-not false 42) 42 "clj-logic-when-not-false")
(is= (if-not true :a :b) :b "clj-logic-if-not-true")
(is= (if-not false :a :b) :a "clj-logic-if-not-false")

;; -- if-let / when-let --

(testing "if-let / when-let")

(is= (if-let [x 42] x :nope) 42 "clj-logic-if-let-truthy")
(is= (if-let [x nil] x :nope) :nope "clj-logic-if-let-nil")
(is= (when-let [x 42] x) 42 "clj-logic-when-let-truthy")
(is= (when-let [x nil] x) nil "clj-logic-when-let-nil")

;; ============================================================================
;; TYPE PREDICATE TESTS
;; ============================================================================

(testing "type predicates")

(is= (nil? nil) true "clj-ds-type-nil-check")
(is= (nil? false) false "clj-ds-type-nil-check-false")
(is= (true? true) true "clj-ds-type-true-check")
(is= (true? 1) false "clj-ds-type-true-check-false")
(is= (false? false) true "clj-ds-type-false-check")
(is= (false? nil) false "clj-ds-type-false-check-nil")
(is= (number? 42) true "clj-ds-type-number-check")
(is= (number? "42") false "clj-ds-type-number-check-false")
(is= (string? "abc") true "clj-ds-type-string-check")
(is= (string? 42) false "clj-ds-type-string-check-false")
(is= (keyword? :a) true "clj-ds-type-keyword-check")
(is= (keyword? "a") false "clj-ds-type-keyword-check-false")
(is= (vector? [1 2]) true "clj-ds-type-vector-check")
(is= (vector? (list 1 2)) false "clj-ds-type-vector-check-list")
(is= (map? {:a 1}) true "clj-ds-type-map-check")
(is= (map? [1 2]) false "clj-ds-type-map-check-false")
(is= (set? #{1 2}) true "clj-ds-type-set-check")
(is= (set? [1 2]) false "clj-ds-type-set-check-false")
(is= (coll? [1 2]) true "clj-ds-type-coll-check-vec")
(is= (coll? (list 1 2)) true "clj-ds-type-coll-check-list")
(is= (coll? {:a 1}) true "clj-ds-type-coll-check-map")
(is= (coll? #{1}) true "clj-ds-type-coll-check-set")
(is= (coll? 42) false "clj-ds-type-coll-check-number")
(is= (sequential? [1 2]) true "clj-ds-type-sequential-vec")
(is= (sequential? (list 1 2)) true "clj-ds-type-sequential-list")
(is= (associative? {:a 1}) true "clj-ds-type-associative-map")
(is= (associative? [1 2]) true "clj-ds-type-associative-vec")

;; ============================================================================
;; NUMERIC TESTS
;; ============================================================================

(testing "numeric")

(is= (pos? 1) true "clj-ds-num-pos")
(is= (pos? 0) false "clj-ds-num-pos-zero")
(is= (pos? -1) false "clj-ds-num-pos-neg")
(is= (neg? -1) true "clj-ds-num-neg")
(is= (neg? 0) false "clj-ds-num-neg-zero")
(is= (neg? 1) false "clj-ds-num-neg-pos")
(is= (zero? 0) true "clj-ds-num-zero")
(is= (zero? 1) false "clj-ds-num-zero-false")
(is= (even? 2) true "clj-ds-num-even")
(is= (even? 3) false "clj-ds-num-even-false")
(is= (odd? 3) true "clj-ds-num-odd")
(is= (odd? 2) false "clj-ds-num-odd-false")
(is= (inc 41) 42 "clj-ds-num-inc")
(is= (dec 43) 42 "clj-ds-num-dec")
(is= (min 1 2 3) 1 "clj-ds-num-min")
(is= (max 1 2 3) 3 "clj-ds-num-max")
(is= (abs -5) 5 "clj-ds-num-abs")
(is= (abs 5) 5 "clj-ds-num-abs-pos")
(is= (rem 10 3) 1 "clj-ds-num-rem")
(is= (mod 10 3) 1 "clj-ds-num-mod")
(is= (quot 10 3) 3 "clj-ds-num-quot")

;; ============================================================================
;; MISC / COMBINED TESTS
;; ============================================================================

(testing "misc / combined")

(is= (identity 42) 42 "clj-ds-identity")
(is= (identity nil) nil "clj-ds-identity-nil")
(is= (str nil) "" "clj-ds-str-nil")
(is= (str 42) "42" "clj-ds-str-number")
(is= (str :kw) ":kw" "clj-ds-str-keyword")
(is= (str true) "true" "clj-ds-str-bool-true")
(is= (str false) "false" "clj-ds-str-bool-false")
(is= (str [1 2 3]) "[1 2 3]" "clj-ds-str-vec")
(is= (str (list 1 2 3)) "(1 2 3)" "clj-ds-str-list")
(is= (str 1 2 3) "123" "clj-ds-str-concat")
(is= (let [x 1 y 2] (+ x y)) 3 "clj-ds-let-basic")
(is= (let [x (+ 1 2)] (* x x)) 9 "clj-ds-let-nested")
(is= (loop [i 0 sum 0] (if (< i 10) (recur (inc i) (+ sum i)) sum)) 45 "clj-ds-loop-recur-sum")
(is= ((fn [x] (+ x 1)) 41) 42 "clj-ds-fn-basic")
(is= ((fn [a b c] (+ a b c)) 1 2 3) 6 "clj-ds-fn-multi-arg")
(is= (do 1 2 3) 3 "clj-ds-do-returns-last")
(is= (do 1 2 nil) nil "clj-ds-do-returns-nil")

(test-summary)
