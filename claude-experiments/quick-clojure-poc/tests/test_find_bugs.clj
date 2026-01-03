;; ============================================================
;; Bug Finding Test Suite
;; Tests designed to find bugs in existing implementations
;; ============================================================

;; ============= ARITHMETIC EDGE CASES =============

;; Test 1: Large number arithmetic
(println "Test: Large numbers")
(println (+ 1000000000 1000000000))  ;; Should be 2000000000

;; Test 2: Negative number arithmetic
(println "Test: Negative numbers")
(println (- 0 10))    ;; Should be -10
(println (+ -5 3))    ;; Should be -2
(println (* -3 -4))   ;; Should be 12
(println (/ -20 4))   ;; Should be -5

;; Test 3: Division edge cases
(println "Test: Division")
(println (/ 10 3))    ;; Should be 3 (integer division)
(println (/ 1 2))     ;; Should be 0 (integer division)

;; ============= COMPARISON EDGE CASES =============

;; Test 4: Comparing negative numbers
(println "Test: Negative comparisons")
(println (< -5 -3))   ;; Should be true
(println (> -3 -5))   ;; Should be true
(println (< -5 0))    ;; Should be true
(println (> 0 -5))    ;; Should be true

;; Test 5: Equality edge cases
(println "Test: Equality")
(println (= nil nil))     ;; Should be true
(println (= true true))   ;; Should be true
(println (= false false)) ;; Should be true
(println (= 0 0))         ;; Should be true
(println (= -1 -1))       ;; Should be true

;; Test 6: Boolean comparisons with nil
(println "Test: Boolean vs nil")
(println (= nil false))   ;; Should be false
(println (= nil true))    ;; Should be false

;; ============= COLLECTION OPERATIONS =============

;; Test 7: Vector operations
(println "Test: Vector basic")
(def v [1 2 3])
(println (first v))       ;; Should be 1
(println (rest v))        ;; Should be (2 3)
(println (count v))       ;; Should be 3
(println (nth v 0))       ;; Should be 1
(println (nth v 2))       ;; Should be 3

;; Test 8: Empty collection edge cases
(println "Test: Empty collections")
(def empty-vec [])
(println (count empty-vec))  ;; Should be 0
(println (first empty-vec))  ;; Should be nil
(println (rest empty-vec))   ;; Should be () or nil

;; Test 9: Single element collections
(println "Test: Single element")
(def single [42])
(println (first single))     ;; Should be 42
(println (rest single))      ;; Should be () or empty
(println (count single))     ;; Should be 1

;; Test 10: Vector conj
(println "Test: Vector conj")
(println (conj [1 2] 3))     ;; Should be [1 2 3]
(println (conj [] 1))        ;; Should be [1]

;; Test 11: List operations
(println "Test: List operations")
(def lst (list 1 2 3))
(println (first lst))        ;; Should be 1
(println (count lst))        ;; Should be 3

;; Test 12: cons operations
(println "Test: cons")
(println (cons 0 [1 2 3]))   ;; Should be (0 1 2 3)
(println (cons 1 nil))       ;; Should be (1)
(println (cons 1 (list 2)))  ;; Should be (1 2)

;; ============= MAP OPERATIONS =============

;; Test 13: Map creation and lookup
(println "Test: Map operations")
(def m {:a 1 :b 2})
(println (get m :a))         ;; Should be 1
(println (get m :b))         ;; Should be 2
(println (get m :c))         ;; Should be nil
(println (get m :c "default")) ;; Should be "default"

;; Test 14: Map with integer keys
(println "Test: Map with int keys")
(def int-map {1 "one" 2 "two"})
(println (get int-map 1))    ;; Should be "one"
(println (get int-map 2))    ;; Should be "two"

;; Test 15: assoc and dissoc
(println "Test: assoc/dissoc")
(println (assoc {:a 1} :b 2))  ;; Should be {:a 1 :b 2}
(println (dissoc {:a 1 :b 2} :a))  ;; Should be {:b 2}

;; Test 16: Map count
(println "Test: Map count")
(println (count {:a 1 :b 2 :c 3}))  ;; Should be 3
(println (count {}))                ;; Should be 0

;; ============= FUNCTION EDGE CASES =============

;; Test 17: Zero-arity functions
(println "Test: Zero arity")
(def zero-arity (fn [] 42))
(println (zero-arity))       ;; Should be 42

;; Test 18: Functions returning nil
(println "Test: Nil return")
(def returns-nil (fn [] nil))
(println (returns-nil))      ;; Should be nil

;; Test 19: Functions returning false
(println "Test: False return")
(def returns-false (fn [] false))
(println (returns-false))    ;; Should be false

;; Test 20: Identity function
(println "Test: Identity")
(def identity-fn (fn [x] x))
(println (identity-fn 42))   ;; Should be 42
(println (identity-fn nil))  ;; Should be nil
(println (identity-fn false)) ;; Should be false

;; ============= CLOSURE EDGE CASES =============

;; Test 21: Closure capturing nil
(println "Test: Closure capturing nil")
(def capture-nil-val nil)
(def closure-nil (fn [] capture-nil-val))
(println (closure-nil))      ;; Should be nil

;; Test 22: Closure capturing false
(println "Test: Closure capturing false")
(def capture-false-val false)
(def closure-false (fn [] capture-false-val))
(println (closure-false))    ;; Should be false

;; Test 23: Nested closures
(println "Test: Nested closures")
(def make-adder (fn [x] (fn [y] (+ x y))))
(def add10 (make-adder 10))
(println (add10 5))          ;; Should be 15
(println (add10 0))          ;; Should be 10
(println (add10 -5))         ;; Should be 5

;; ============= MULTI-ARITY EDGE CASES =============

;; Test 24: Multi-arity with zero
(println "Test: Multi-arity")
(def multi
  (fn
    ([] 0)
    ([x] x)
    ([x y] (+ x y))))
(println (multi))            ;; Should be 0
(println (multi 5))          ;; Should be 5
(println (multi 3 4))        ;; Should be 7

;; ============= VARIADIC EDGE CASES =============

;; Test 25: Variadic with no extra args
(println "Test: Variadic no extras")
(def var-fn (fn [& args] (if args (first args) nil)))
(println (var-fn))           ;; Should be nil
(println (var-fn 1))         ;; Should be 1
(println (var-fn 1 2 3))     ;; Should be 1

;; Test 26: Variadic with fixed args
(println "Test: Variadic with fixed")
(def var-fixed (fn [a b & rest] (+ a b)))
(println (var-fixed 1 2))    ;; Should be 3
(println (var-fixed 1 2 3))  ;; Should be 3

;; ============= CONTROL FLOW EDGE CASES =============

;; Test 27: if with nil as condition
(println "Test: if with nil")
(println (if nil "truthy" "falsy"))  ;; Should be "falsy"

;; Test 28: if with false as condition
(println "Test: if with false")
(println (if false "truthy" "falsy")) ;; Should be "falsy"

;; Test 29: if with 0 as condition (truthy!)
(println "Test: if with 0")
(println (if 0 "truthy" "falsy"))    ;; Should be "truthy" (0 is truthy in Clojure)

;; Test 30: if with empty collection (truthy!)
(println "Test: if with empty vec")
(println (if [] "truthy" "falsy"))   ;; Should be "truthy" (empty vec is truthy)

;; Test 31: Nested if
(println "Test: Nested if")
(println (if true (if false 1 2) 3)) ;; Should be 2

;; ============= LET EDGE CASES =============

;; Test 32: Let with shadowing
(println "Test: Let shadowing")
(def x 100)
(println (let [x 1] x))      ;; Should be 1
(println x)                  ;; Should be 100 (not affected)

;; Test 33: Let binding order
(println "Test: Let binding order")
(println (let [a 1 b (+ a 1) c (+ b 1)] c))  ;; Should be 3

;; Test 34: Empty let
(println "Test: Empty let body")
(println (let [x 1] nil))    ;; Should be nil

;; ============= LOOP/RECUR EDGE CASES =============

;; Test 35: Loop with zero iterations
(println "Test: Loop zero iterations")
(println (loop [i 0] (if (< i 0) (recur (- i 1)) i)))  ;; Should be 0

;; Test 36: Tail recursion
(println "Test: Tail recursion")
(def factorial
  (fn [n]
    (loop [n n acc 1]
      (if (< n 2)
        acc
        (recur (- n 1) (* n acc))))))
(println (factorial 5))      ;; Should be 120
(println (factorial 0))      ;; Should be 1
(println (factorial 1))      ;; Should be 1

;; ============= EXCEPTION EDGE CASES =============

;; Test 37: try without exception
(println "Test: try no exception")
(println (try 42 (catch Exception e -1)))  ;; Should be 42

;; Test 38: try with exception from pre-condition
(println "Test: try with exception")
(def must-be-positive (fn [x] {:pre [(> x 0)]} x))
(println (try (must-be-positive 5) (catch Exception e "caught")))  ;; Should be 5
(println (try (must-be-positive -1) (catch Exception e "caught"))) ;; Should be "caught"

;; ============= KEYWORD EDGE CASES =============

;; Test 39: Keyword as function
(println "Test: Keyword as fn")
(println (:a {:a 1 :b 2}))   ;; Should be 1
(println (:c {:a 1 :b 2}))   ;; Should be nil
(println (:c {:a 1} "default"))  ;; Should be "default"

;; Test 40: Keyword equality
(println "Test: Keyword equality")
(println (= :foo :foo))      ;; Should be true
(println (= :foo :bar))      ;; Should be false

;; ============= STRING EDGE CASES =============

;; Test 41: Empty string
(println "Test: Empty string")
(def empty-str "")
(println (= empty-str ""))   ;; Should be true
(println (count empty-str))  ;; Should be 0 if implemented

;; Test 42: String with spaces
(println "Test: String with spaces")
(println "hello world")      ;; Should be "hello world"

;; ============= BOOLEAN OPERATIONS =============

;; Test 43: not operation
(println "Test: not")
(println (not true))         ;; Should be false
(println (not false))        ;; Should be true
(println (not nil))          ;; Should be true
(println (not 0))            ;; Should be false (0 is truthy)

;; Test 44: and/or with short circuit
(println "Test: and/or")
(println (and true true))    ;; Should be true
(println (and true false))   ;; Should be false
(println (and nil "never"))  ;; Should be nil (short circuit)
(println (or false true))    ;; Should be true
(println (or nil false))     ;; Should be false
(println (or nil "default")) ;; Should be "default"

;; ============= PREDICATE EDGE CASES =============

;; Test 45: Type predicates
(println "Test: Type predicates")
(println (nil? nil))         ;; Should be true
(println (nil? false))       ;; Should be false
(println (true? true))       ;; Should be true
(println (true? 1))          ;; Should be false
(println (false? false))     ;; Should be true
(println (false? nil))       ;; Should be false

;; Test 46: Collection predicates
(println "Test: Collection predicates")
(println (vector? []))       ;; Should be true
(println (vector? (list)))   ;; Should be false
(println (list? (list 1)))   ;; Should be true
(println (map? {}))          ;; Should be true
(println (map? []))          ;; Should be false

;; ============= NUMERIC PREDICATES =============

;; Test 47: zero? and number predicates
(println "Test: Numeric predicates")
(println (zero? 0))          ;; Should be true
(println (zero? 1))          ;; Should be false
(println (even? 2))          ;; Should be true
(println (even? 3))          ;; Should be false
(println (odd? 3))           ;; Should be true
(println (odd? 2))           ;; Should be false

;; ============= QUOTING =============

;; Test 48: Quote
(println "Test: Quote")
(println (quote foo))        ;; Should be foo (symbol)

;; ============= DO BLOCK EDGE CASES =============

;; Test 49: do returns last value
(println "Test: do block")
(println (do 1 2 3))         ;; Should be 3
(println (do))               ;; Should be nil

;; ============= FINAL MARKER =============
(println "=== All tests complete ===")
