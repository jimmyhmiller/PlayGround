;; Recursive Fibonacci Function - PARTIAL TERSE CONVERSION
;;
;; ✅ This file WORKS with current implementation!
;;
;; Uses currently supported terse features:
;; ✅ Terse operations: (op.name {attrs} operands)
;; ✅ Declare form: (declare name expr)
;; ✅ Type inference from attributes and operands
;;
;; Still uses verbose syntax for:
;; - func.func (not yet terse)
;; - scf.if (not yet terse)
;; - regions and blocks (not yet terse)
;; - scf.yield and func.return (not yet terse)
;;
;; Function signature: fibonacci(n: i32) -> i32

(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @fibonacci
      :function_type (!function (inputs i32) (results i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ (: %n i32) ])

          ;; Check if n <= 1 (base case)
          (declare c1 (arith.constant {:value (: 1 i32)}))

          ;; arith.cmpi needs explicit result type (no inference yet)
          (operation
            (name arith.cmpi)
            (result-bindings [%cond])
            (result-types i1)
            (operands %n %c1)
            (attributes { :predicate (: 3 i64) }))

          ;; scf.if still verbose (no terse syntax yet)
          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types i32)
            (operands %cond)
            (regions
              ;; Then region: base case, return n
              (region
                (block
                  (arguments [])
                  (operation
                    (name scf.yield)
                    (operands %n))))

              ;; Else region: recursive case
              (region
                (block
                  (arguments [])

                  ;; ✅ TERSE: Compute fib(n-1)
                  (declare c1_rec (arith.constant {:value (: 1 i32)}))
                  (declare n_minus_1 (arith.subi %n %c1_rec))

                  ;; func.call needs explicit result type (no inference yet)
                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_1])
                    (result-types i32)
                    (operands %n_minus_1)
                    (attributes { :callee @fibonacci }))

                  ;; ✅ TERSE: Compute fib(n-2)
                  (declare c2 (arith.constant {:value (: 2 i32)}))
                  (declare n_minus_2 (arith.subi %n %c2))

                  ;; func.call needs explicit result type (no inference yet)
                  (operation
                    (name func.call)
                    (result-bindings [%fib_n_minus_2])
                    (result-types i32)
                    (operands %n_minus_2)
                    (attributes { :callee @fibonacci }))

                  ;; ✅ TERSE: Add results
                  (declare sum (arith.addi %fib_n_minus_1 %fib_n_minus_2))

                  ;; Still verbose (no implicit yield)
                  (operation
                    (name scf.yield)
                    (operands %sum))))))

          ;; Still verbose (no implicit return)
          (operation
            (name func.return)
            (operands %result))))))

  ;; Main function calls fibonacci(10)
  (operation
    (name func.func)
    (attributes {
      :sym_name @main
      :function_type (!function (inputs) (results i64))
    })
    (regions
      (region
        (block [^entry]
          (arguments [])

          ;; ✅ TERSE: Create constant
          (declare n (arith.constant {:value (: 10 i32)}))

          ;; func.call needs explicit result type
          (operation
            (name func.call)
            (result-bindings [%fib_result])
            (result-types i32)
            (operands %n)
            (attributes { :callee @fibonacci }))

          ;; arith.extsi needs explicit result type
          (operation
            (name arith.extsi)
            (result-bindings [%result_i64])
            (result-types i64)
            (operands %fib_result))

          ;; Still verbose (no implicit return)
          (operation
            (name func.return)
            (operands %result_i64)))))))
