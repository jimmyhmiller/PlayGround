;; ============================================================
;; Multi-Arity Function Tests
;; ============================================================

;; Test 1: Simple two-arity function
(def greet
  (fn
    ([] 0)
    ([x] x)))

(greet)      ;; => 0
(greet 42)   ;; => 42

;; Test 2: Three arities
(def arity-test
  (fn
    ([] 100)
    ([a] (+ a 1))
    ([a b] (+ a b))))

(arity-test)        ;; => 100
(arity-test 5)      ;; => 6
(arity-test 10 20)  ;; => 30

;; Test 3: Four arities
(def four-arity
  (fn
    ([] 0)
    ([a] a)
    ([a b] (+ a b))
    ([a b c] (+ a (+ b c)))))

(four-arity)           ;; => 0
(four-arity 1)         ;; => 1
(four-arity 1 2)       ;; => 3
(four-arity 1 2 3)     ;; => 6

;; Test 4: Multi-arity with closures
(def make-adder
  (fn [base]
    (fn
      ([] base)
      ([x] (+ base x))
      ([x y] (+ base (+ x y))))))

(def add10 (make-adder 10))
(add10)       ;; => 10
(add10 5)     ;; => 15
(add10 5 3)   ;; => 18

;; Test 5: Multi-arity calling multi-arity
(def dispatcher
  (fn
    ([] (arity-test))
    ([x] (arity-test x))
    ([x y] (arity-test x y))))

(dispatcher)       ;; => 100
(dispatcher 5)     ;; => 6
(dispatcher 10 20) ;; => 30

;; Test 6: Recursion with multi-arity
(def factorial
  (fn
    ([n] (factorial n 1))
    ([n acc]
      (if (< n 2)
        acc
        (factorial (- n 1) (* n acc))))))

(factorial 1)  ;; => 1
(factorial 5)  ;; => 120
(factorial 10) ;; => 3628800
