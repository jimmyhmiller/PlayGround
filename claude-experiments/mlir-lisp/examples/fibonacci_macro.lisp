;; Fibonacci with macros to simplify the syntax

;; Macro for defining constants
(defmacro const [value result_name]
  (op arith.constant
    :attrs {:value value}
    :results [i32]
    :as result_name))

;; Macro for addition
(defmacro add [a b result_name]
  (op arith.addi
    :operands [a b]
    :results [i32]
    :as result_name))

;; Macro for comparison
(defmacro less_than [a b result_name]
  (op arith.cmpi
    :attrs {:predicate "slt"}
    :operands [a b]
    :results [i1]
    :as result_name))

(defmacro less_or_equal [a b result_name]
  (op arith.cmpi
    :attrs {:predicate "sle"}
    :operands [a b]
    :results [i1]
    :as result_name))

;; Macro for conditional branch
(defmacro cond_branch [condition true_dest false_dest]
  (op cf.cond_br
    :operands [condition]
    :true true_dest
    :false false_dest))

;; Macro for unconditional branch
(defmacro branch [dest args_list]
  (op cf.br
    :dest dest
    :args args_list))

;; Fibonacci function with cleaner syntax using macros
(defn fib [n:i32] i32
  (block entry []
    (const 0 %zero)
    (const 1 %one)
    (less_or_equal n %one %base_case)
    (cond_branch %base_case return_n loop_init))

  (block return_n []
    (branch exit [n]))

  (block loop_init []
    (const 0 %a_init)
    (const 1 %b_init)
    (const 1 %i_init)
    (branch loop_header [%a_init %b_init %i_init]))

  (block loop_header [i32 i32 i32]
    (less_than ^2 n %continue)
    (cond_branch %continue loop_body loop_exit))

  (block loop_body []
    (add ^0 ^1 %next_b)
    (const 1 %one_inc)
    (add ^2 %one_inc %next_i)
    (branch loop_header [^1 %next_b %next_i]))

  (block loop_exit []
    (branch exit [^1]))

  (block exit [i32]
    (op func.return :operands [^0])))

(defn main [] i32
  (const 10 %ten)
  (op func.call :attrs {:callee "fib"} :operands [%ten] :results [i32] :as %result)
  (op func.return :operands [%result]))
