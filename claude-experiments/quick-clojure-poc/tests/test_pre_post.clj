;; ============================================================
;; Pre/Post Condition Tests
;; ============================================================

;; Test 1: Simple pre-condition - pass
(def positive-only
  (fn [x]
    {:pre [(> x 0)]}
    x))

(positive-only 5)    ;; => 5
(positive-only 100)  ;; => 100

;; Test 2: Simple post-condition - pass
(def double-it
  (fn [x]
    {:post [(> % x)]}
    (* x 2)))

(double-it 3)   ;; => 6
(double-it 10)  ;; => 20

;; Test 3: Both pre and post - pass
(def safe-increment
  (fn [x]
    {:pre [(> x 0)]
     :post [(> % x)]}
    (+ x 1)))

(safe-increment 5)   ;; => 6
(safe-increment 99)  ;; => 100

;; Test 4: Multiple pre-conditions - pass
(def in-range
  (fn [x]
    {:pre [(> x 0) (< x 100)]}
    x))

(in-range 50)  ;; => 50
(in-range 1)   ;; => 1
(in-range 99)  ;; => 99

;; Test 5: Multiple post-conditions - pass
(def compute
  (fn [x]
    {:post [(> % 0) (< % 1000)]}
    (+ x 10)))

(compute 5)   ;; => 15
(compute 100) ;; => 110

;; Test 6: Pre-condition with closure
(def make-validator
  (fn [min-val]
    (fn [x]
      {:pre [(> x min-val)]}
      x)))

(def validate-gt-10 (make-validator 10))
(validate-gt-10 20)  ;; => 20
(validate-gt-10 50)  ;; => 50

;; Test 7: Pre-condition failure - caught
(def must-be-positive
  (fn [x]
    {:pre [(> x 0)]}
    x))

(try
  (must-be-positive -5)
  (catch Exception e
    -1))  ;; => -1 (exception caught)

;; Test 8: Post-condition failure - caught  
(def must-return-positive
  (fn [x]
    {:post [(> % 0)]}
    x))

(try
  (must-return-positive -5)
  (catch Exception e
    -2))  ;; => -2 (exception caught)

;; Test 9: Pre-condition with nil check
(def not-nil
  (fn [x]
    {:pre [x]}  ;; x must be truthy (not nil or false)
    x))

(not-nil 42)   ;; => 42
(not-nil true) ;; => true

;; Test 10: Chained function calls with conditions
(def step1
  (fn [x]
    {:pre [(> x 0)]
     :post [(> % x)]}
    (+ x 1)))

(def step2
  (fn [x]
    {:pre [(> x 1)]
     :post [(> % x)]}
    (* x 2)))

(step2 (step1 5))  ;; => step1(5)=6, step2(6)=12
