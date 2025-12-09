;; ============================================================
;; Variadic Function Tests
;; ============================================================

;; Test 1: Simple variadic - all args as rest
(def collect-all (fn [& args] args))
(collect-all)        ;; => nil
(collect-all 1)      ;; => (1)
(collect-all 1 2 3)  ;; => (1 2 3)

;; Test 2: Fixed + rest args
(def fixed-plus-rest
  (fn [a b & rest] rest))

(fixed-plus-rest 1 2)           ;; => nil
(fixed-plus-rest 1 2 3)         ;; => (3)
(fixed-plus-rest 1 2 3 4 5)     ;; => (3 4 5)

;; Test 3: Variadic with computation on rest
(def has-args?
  (fn [& args]
    (if args
      ;; args is truthy (not nil)
      42
      0)))

(has-args?)      ;; => 0 (args is nil)
(has-args? 1)    ;; => 42 (args is a list)
(has-args? 1 2)  ;; => 42 (args is a list)

;; Test 4: Multi-arity with variadic fallback
;; Note: In Clojure, variadic can't overlap with fixed arities
;; So [a b & more] catches 2+ args, but we also have [a b] for exactly 2
;; This should error, so let's use a different pattern
(def flexible
  (fn
    ([] 0)
    ([a] a)
    ([a b c & more] 999)))

(flexible)             ;; => 0
(flexible 10)          ;; => 10
(flexible 10 20 30)    ;; => 999
(flexible 10 20 30 40) ;; => 999

;; Test 5: Variadic closure
(def make-collector
  (fn [prefix]
    (fn [& items]
      (if items
        prefix
        0))))

(def collect (make-collector 42))
(collect)        ;; => 0
(collect 1)      ;; => 42
(collect 1 2 3)  ;; => 42

;; Test 6: Pass list to another function
(def identity-fn (fn [x] x))
(def pass-rest
  (fn [& args]
    (identity-fn args)))

(pass-rest)      ;; => nil
(pass-rest 1)    ;; => (1)
(pass-rest 1 2)  ;; => (1 2)

;; Test 7: First element access (using if to check)
(def first-or-zero
  (fn [& args]
    (if args
      100  ;; Can't access first yet, but at least check it's a list
      0)))

(first-or-zero)      ;; => 0
(first-or-zero 1)    ;; => 100
(first-or-zero 1 2)  ;; => 100
