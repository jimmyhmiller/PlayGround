;; Example 7: Higher-Order Functions
;; Demonstrates: Functions as values, function parameters, function composition

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Apply a function twice
(def apply-twice (: (-> [(-> [I32] I32) I32] I32))
  (fn [f x]
    (f (f x))))

;; Apply a function n times
(def apply-n-times (: (-> [(-> [I32] I32) I32 I32] I32))
  (fn [f n x]
    (if (<= n 0)
        x
        (apply-n-times f (- n 1) (f x)))))

;; Some simple functions
(def add1 (: (-> [I32] I32))
  (fn [x] (+ x 1)))

(def double (: (-> [I32] I32))
  (fn [x] (* x 2)))

(def square (: (-> [I32] I32))
  (fn [x] (* x x)))

;; Compose two functions
(def compose (: (-> [(-> [I32] I32) (-> [I32] I32) I32] I32))
  (fn [f g x]
    (f (g x))))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "apply-twice(add1, 5) = %d\n")
            (apply-twice add1 5))

    (printf (c-str "apply-twice(double, 3) = %d\n")
            (apply-twice double 3))

    (printf (c-str "apply-n-times(add1, 10, 0) = %d\n")
            (apply-n-times add1 10 0))

    (printf (c-str "apply-n-times(double, 3, 1) = %d\n")
            (apply-n-times double 3 1))

    (printf (c-str "compose(square, double, 3) = %d\n")
            (compose square double 3))

    (printf (c-str "compose(double, square, 3) = %d\n")
            (compose double square 3))

    0))

(main-fn)
