;; Natural syntax fibonacci using macros
;; Goal: make it as readable as possible

;; Define high-level macros
(defmacro const [value]
  (op arith.constant :attrs {:value value} :results [i32]))

(defmacro <= [a b]
  (op arith.cmpi :attrs {:predicate "sle"} :operands [a b] :results [i1]))

(defmacro - [a b]
  (op arith.subi :operands [a b] :results [i32]))

(defmacro + [a b]
  (op arith.addi :operands [a b] :results [i32]))

(defmacro call [func_name args]
  (op func.call :attrs {:callee func_name} :operands args :results [i32]))

(defmacro return [value]
  (op func.return :operands value))

;; Recursive fibonacci - natural syntax with block-based if
(defn fib [n]
  (block entry []
    (const 1 :as %one)
    (<= n %one :as %is_base)
    (op cf.cond_br :operands [%is_base] :true base :false recursive))

  (block base []
    (op cf.br :dest exit :args [n]))

  (block recursive []
    (const 1 :as %one_r)
    (const 2 :as %two)
    (- n %one_r :as %n1)
    (- n %two :as %n2)
    (call "fib" [%n1] :as %fib1)
    (call "fib" [%n2] :as %fib2)
    (+ %fib1 %fib2 :as %sum)
    (op cf.br :dest exit :args [%sum]))

  (block exit [i32]
    (return [^0])))

;; Main function
(defn main []
  (const 10 :as %n)
  (call "fib" [%n] :as %result)
  (return [%result]))
