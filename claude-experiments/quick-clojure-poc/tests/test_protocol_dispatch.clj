;; Protocol Dispatch Tests
;; Tests for ExternalCall/ExternalCallWithSaves mechanism

;; ============================================
;; Setup: Define protocol and type
;; ============================================

(defprotocol ISeq
  (-first [coll]))

(deftype MyList [head tail])

(extend-type MyList
  ISeq
  (-first [ml] (.-head ml)))

;; ============================================
;; Test 1: Direct protocol call
;; Expected: 99
;; ============================================
(def ml1 (MyList. 99 nil))
(-first ml1)

;; ============================================
;; Test 2: Protocol call through function (THE ORIGINAL BUG)
;; This was the bug: argument in x0-x7 got clobbered by trampoline call
;; Expected: 99
;; ============================================
(def get-first (fn [x] (-first x)))
(get-first ml1)

;; ============================================
;; Test 3: Protocol call with result in computation
;; Expected: 100
;; ============================================
(+ (-first ml1) 1)

;; ============================================
;; Test 4: Protocol call through function with computation
;; Expected: 100
;; ============================================
(def get-first-plus-1 (fn [x] (+ (-first x) 1)))
(get-first-plus-1 ml1)

;; ============================================
;; Test 5: Nested function calls with protocol
;; Expected: 99
;; ============================================
(def level1 (fn [x] (-first x)))
(def level2 (fn [x] (level1 x)))
(level2 ml1)

;; ============================================
;; Test 6: Protocol call in let binding
;; Expected: 198
;; ============================================
(def with-let (fn [lst]
  (let [x (-first lst)]
    (+ x x))))
(with-let ml1)

;; ============================================
;; Test 7: Protocol call on freshly constructed object
;; Expected: 777
;; ============================================
(-first (MyList. 777 nil))

;; ============================================
;; Test 8: Protocol call on object constructed in function
;; Expected: 555
;; ============================================
(def make-and-get (fn [n]
  (-first (MyList. n nil))))
(make-and-get 555)

;; ============================================
;; Test 9: Two-argument function with protocol call (FIXED BUG)
;; This was a bug: when multiple argument registers needed preservation
;; across calls, the saves list confused virtual/physical registers.
;; Expected: 142 (42 + 100)
;; ============================================
(def two-arg-fn (fn [a b] (+ (-first a) b)))
(two-arg-fn (MyList. 42 nil) 100)

;; ============================================
;; Test 10: Three-argument function with protocol call
;; Expected: 343 (42 + 100 + 201)
;; ============================================
(def three-arg-fn (fn [a b c] (+ (+ (-first a) b) c)))
(three-arg-fn (MyList. 42 nil) 100 201)

;; ============================================
;; Test 11: Multiple protocol calls in one function
;; Expected: 141 (42 + 99)
;; ============================================
(def sum-firsts (fn [a b] (+ (-first a) (-first b))))
(sum-firsts (MyList. 42 nil) ml1)

;; ============================================
;; Tests 12-18: Various argument counts with protocol calls
;; Testing that register saves work correctly for all arg counts
;; Each test: sum of (-first a) + b + c + ... (all args after first)
;; ============================================

;; Test 12: 4 arguments - Expected: 100 (10 + 20 + 30 + 40)
(def fn-4-args (fn [a b c d] (+ (+ (+ (-first a) b) c) d)))
(fn-4-args (MyList. 10 nil) 20 30 40)

;; Test 13: 5 arguments - Expected: 150 (10 + 20 + 30 + 40 + 50)
(def fn-5-args (fn [a b c d e] (+ (+ (+ (+ (-first a) b) c) d) e)))
(fn-5-args (MyList. 10 nil) 20 30 40 50)

;; Test 14: 6 arguments - Expected: 210 (10 + 20 + 30 + 40 + 50 + 60)
(def fn-6-args (fn [a b c d e f] (+ (+ (+ (+ (+ (-first a) b) c) d) e) f)))
(fn-6-args (MyList. 10 nil) 20 30 40 50 60)

;; Test 15: 7 arguments - Expected: 280 (10 + 20 + 30 + 40 + 50 + 60 + 70)
(def fn-7-args (fn [a b c d e f g] (+ (+ (+ (+ (+ (+ (-first a) b) c) d) e) f) g)))
(fn-7-args (MyList. 10 nil) 20 30 40 50 60 70)

;; Test 16: 8 arguments (max for ARM64 register calling convention)
;; Expected: 360 (10 + 20 + 30 + 40 + 50 + 60 + 70 + 80)
(def fn-8-args (fn [a b c d e f g h] (+ (+ (+ (+ (+ (+ (+ (-first a) b) c) d) e) f) g) h)))
(fn-8-args (MyList. 10 nil) 20 30 40 50 60 70 80)

;; Test 17: Protocol call using later argument (not first)
;; Expected: 130 (10 + 20 + 100)
(def use-third-arg (fn [a b c] (+ (+ a b) (-first c))))
(use-third-arg 10 20 (MyList. 100 nil))

;; Test 18: Protocol call on multiple arguments at different positions
;; Expected: 200 (100 + 20 + 30 + 40 + 10)
(def multi-protocol (fn [a b c d e] (+ (+ (+ (+ (-first a) b) c) d) (-first e))))
(multi-protocol (MyList. 100 nil) 20 30 40 (MyList. 10 nil))
