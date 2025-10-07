;; Fibonacci using iteration with control flow
;; Computes fib(10) = 55 using a loop with blocks

(defn fib [n:i32] i32
  (block entry []
    ;; Initialize: a=0, b=1, i=0
    (op arith.constant :attrs {:value 0} :results [i32] :as %zero)
    (op arith.constant :attrs {:value 1} :results [i32] :as %one)

    ;; Check if n <= 1
    (op arith.cmpi :attrs {:predicate "sle"} :operands [n %one] :results [i1] :as %base_case)
    (op cf.cond_br :operands [%base_case] :true return_n :false loop_init))

  (block return_n []
    ;; Base case: return n
    (op cf.br :dest exit :args [n]))

  (block loop_init []
    ;; Start loop with a=0, b=1, i=1
    (op arith.constant :attrs {:value 0} :results [i32] :as %a_init)
    (op arith.constant :attrs {:value 1} :results [i32] :as %b_init)
    (op arith.constant :attrs {:value 1} :results [i32] :as %i_init)
    (op cf.br :dest loop_header :args [%a_init %b_init %i_init]))

  (block loop_header [i32 i32 i32]
    ;; Loop header: a=^0, b=^1, i=^2
    ;; Check if i < n
    (op arith.cmpi :attrs {:predicate "slt"} :operands [^2 n] :results [i1] :as %continue)
    (op cf.cond_br :operands [%continue] :true loop_body :false loop_exit))

  (block loop_body []
    ;; Get current values from block args
    ;; Compute: next_a = b, next_b = a + b, next_i = i + 1
    (op arith.addi :operands [^0 ^1] :results [i32] :as %next_b)
    (op arith.constant :attrs {:value 1} :results [i32] :as %one_inc)
    (op arith.addi :operands [^2 %one_inc] :results [i32] :as %next_i)

    ;; Loop back with new values: a=b, b=next_b, i=next_i
    (op cf.br :dest loop_header :args [^1 %next_b %next_i]))

  (block loop_exit []
    ;; Return b (which is ^1 from loop_header)
    (op cf.br :dest exit :args [^1]))

  (block exit [i32]
    (op func.return :operands [^0])))

(defn main [] i32
  ;; Compute fib(10)
  (op arith.constant :attrs {:value 10} :results [i32] :as %ten)
  (op func.call :attrs {:callee "fib"} :operands [%ten] :results [i32] :as %result)
  (op func.return :operands [%result]))
