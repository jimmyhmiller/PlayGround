;; Recursive Fibonacci with high-level macros
;; fib(n) = n if n <= 1, else fib(n-1) + fib(n-2)

;; High-level syntax macros
(defmacro const [value]
  (op arith.constant :attrs {:value value} :results [i32]))

(defmacro + [a b]
  (op arith.addi :operands [a b] :results [i32]))

(defmacro - [a b]
  (op arith.subi :operands [a b] :results [i32]))

(defmacro <= [a b]
  (op arith.cmpi :attrs {:predicate "sle"} :operands [a b] :results [i1]))

(defmacro call [func_name args]
  (op func.call :attrs {:callee func_name} :operands args :results [i32]))

;; Recursive fibonacci function
(defn fib [n:i32] i32
  (block entry []
    (const 1 :as %one)
    (<= n %one :as %is_base)
    (op cf.cond_br :operands [%is_base] :true base_case :false recursive_case))

  (block base_case []
    (op cf.br :dest exit :args [n]))

  (block recursive_case []
    (const 1 :as %one_rc)
    (const 2 :as %two)
    (- n %one_rc :as %n_minus_1)
    (- n %two :as %n_minus_2)
    (call "fib" [%n_minus_1] :as %fib_n_1)
    (call "fib" [%n_minus_2] :as %fib_n_2)
    (+ %fib_n_1 %fib_n_2 :as %result)
    (op cf.br :dest exit :args [%result]))

  (block exit [i32]
    (op func.return :operands [^0])))

(defn main [] i32
  (const 10 :as %ten)
  (call "fib" [%ten] :as %result)
  (op func.return :operands [%result]))
