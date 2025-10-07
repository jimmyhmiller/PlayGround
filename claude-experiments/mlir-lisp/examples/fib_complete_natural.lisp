;; Complete recursive Fibonacci with natural expression syntax!
;; This is the syntax we wanted: recursion + expression-level if

(defn fib [n:i32] i32
  (if (< n 2)
    n
    (+ (fib (- n 1)) (fib (- n 2)))))

(defn main [] i32
  (fib 10))
