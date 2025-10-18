;; Example 2: Fibonacci Sequence
;; Demonstrates: Recursion, conditionals, arithmetic operations, typed integers

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

(def fib (: (-> [U32] U32))
  (fn [n]
    (if (<= n 1)
        n
        (+ (fib (- n 1))
           (fib (- n 2))))))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "fib(0) = %u\n") (fib 0))
    (printf (c-str "fib(1) = %u\n") (fib 1))
    (printf (c-str "fib(5) = %u\n") (fib 5))
    (printf (c-str "fib(10) = %u\n") (fib 10))
    (printf (c-str "fib(15) = %u\n") (fib 15))
    0))

(main-fn)
