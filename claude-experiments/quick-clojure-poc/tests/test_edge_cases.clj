;; ============================================================
;; Edge Cases and Integration Tests
;; ============================================================

;; Test 1: Deeply nested closures with multi-arity
(def outer
  (fn [a]
    (fn [b]
      (fn
        ([] (+ a b))
        ([c] (+ a (+ b c)))))))

(def middle ((outer 10) 20))
(middle)      ;; => 30
(middle 5)    ;; => 35

;; Test 2: Variadic within multi-arity closure
(def make-summer
  (fn [base]
    (fn [& nums]
      (if nums
        (+ base 100)  ;; Has args
        base))))      ;; No args

(def summer (make-summer 42))
(summer)        ;; => 42
(summer 1)      ;; => 142
(summer 1 2 3)  ;; => 142

;; Test 3: Pre-condition in nested closure
(def make-safe-div
  (fn [default]
    (fn [x y]
      {:pre [(> y 0)]}
      (if (> y 0)
        (/ x y)
        default))))

(def safe-div (make-safe-div 0))
(safe-div 10 2)  ;; => 5
(safe-div 20 4)  ;; => 5

;; Test 4: Post-condition with closure
(def make-bounded
  (fn [max-val]
    (fn [x]
      {:post [(< % max-val)]}
      x)))

(def under-100 (make-bounded 100))
(under-100 50)  ;; => 50

;; Test 5: Factorial with pre/post conditions
(def safe-factorial
  (fn
    ([n]
      {:pre [(> n -1) (< n 15)]}  ;; n must be non-negative and small
      (safe-factorial n 1))
    ([n acc]
      {:post [(> % 0)]}  ;; result must be positive
      (if (< n 2)
        acc
        (safe-factorial (- n 1) (* n acc))))))

(safe-factorial 0)   ;; => 1
(safe-factorial 1)   ;; => 1
(safe-factorial 5)   ;; => 120
(safe-factorial 10)  ;; => 3628800

;; Test 6: Mutual recursion style (simulate with higher-order functions)
(def even-or-odd
  (fn [n]
    (if (< n 2)
      (if (= n 0) 1 0)  ;; 1 for even, 0 for odd
      (even-or-odd (- n 2)))))

(even-or-odd 0)   ;; => 1 (even)
(even-or-odd 1)   ;; => 0 (odd)
(even-or-odd 10)  ;; => 1 (even)
(even-or-odd 11)  ;; => 0 (odd)

;; Test 7: Complex closure chain
(def add (fn [a] (fn [b] (+ a b))))
(def mul (fn [a] (fn [b] (* a b))))

(def add5 (add 5))
(def mul3 (mul 3))

(mul3 (add5 10))  ;; => 45 (3 * (5 + 10))

;; Test 8: Variadic with pre-condition (args check)
(def at-least-two
  (fn [a b & rest]
    {:pre [(> a 0) (> b 0)]}
    (+ a b)))

(at-least-two 1 2)       ;; => 3
(at-least-two 5 10)      ;; => 15
(at-least-two 5 10 20)   ;; => 15 (rest is ignored in body)

;; Test 9: Closure over multiple values
(def make-linear
  (fn [m b]
    (fn [x]
      (+ (* m x) b))))

(def f (make-linear 2 3))  ;; f(x) = 2x + 3
(f 0)   ;; => 3
(f 1)   ;; => 5
(f 5)   ;; => 13
(f 10)  ;; => 23

;; Test 10: Exception catching in complex scenario
(def guarded
  (fn [x]
    {:pre [(> x 0)]}
    x))

(def process
  (fn [x]
    (try
      (guarded x)
      (catch Exception e
        -999))))

(process 10)   ;; => 10
(process -5)   ;; => -999
