;; Recursive Fibonacci with natural expression syntax
;; NOTE: This example intentionally has no base case to demonstrate
;; that recursive calls work. It will overflow the stack at runtime.
;; For a working fibonacci, see fib_natural.lisp which uses block-based if.

(defn fib [n:i32] i32
  (+ (fib (- n 1)) (fib (- n 2))))

;; Just return 0 to avoid infinite recursion during testing
(defn main [] i32
  0)
