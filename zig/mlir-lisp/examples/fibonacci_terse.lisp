;; Recursive Fibonacci Function - TERSE SYNTAX VERSION
;;
;; ⚠️  NOTE: This file demonstrates what terse syntax WILL look like
;;     It DOES NOT currently parse - missing features are marked with [TODO]
;;
;; Features demonstrated (not yet implemented):
;; - [TODO] func.func with terse syntax
;; - [TODO] Let bindings with scoped variables
;; - [TODO] scf.if with then/else regions
;; - [TODO] Implicit scf.yield (last expr in region)
;; - [TODO] Implicit func.return (last expr in function)
;; - [TODO] Region parsing in terse operations
;; - ✅ Terse operations: (op.name {attrs} operands)
;; - ✅ Type inference from attributes and operands
;;
;; Function signature: fibonacci(n: i32) -> i32
;;
;; Algorithm:
;;   if n <= 1:
;;     return n
;;   else:
;;     return fibonacci(n-1) + fibonacci(n-2)

(mlir
  ;; [TODO: func.func terse syntax]
  ;; Fibonacci function with structured control flow
  (func.func {:sym_name fibonacci
              :function_type (-> (i32) (i32))}
    [(: n i32)]  ;; [TODO: function arguments]

    ;; [TODO: let bindings]
    (let [(: c1 (arith.constant {:value 1}))
          (: cond (arith.cmpi {:predicate sle} n c1))]

      ;; [TODO: scf.if with regions]
      (scf.if {} cond
        ;; Then region: base case - return n
        ;; [TODO: implicit scf.yield]
        (region n)

        ;; Else region: recursive case
        (region
          ;; [TODO: nested let in region]
          (let [;; Compute fib(n-1)
                (: c1-rec (arith.constant {:value 1}))
                (: n-minus-1 (arith.subi {} n c1-rec))
                (: fib-n-minus-1 (func.call {:callee @fibonacci} n-minus-1))

                ;; Compute fib(n-2)
                (: c2 (arith.constant {:value 2}))
                (: n-minus-2 (arith.subi {} n c2))
                (: fib-n-minus-2 (func.call {:callee @fibonacci} n-minus-2))

                ;; Add results
                (: sum (arith.addi {} fib-n-minus-1 fib-n-minus-2))]

            ;; [TODO: implicit scf.yield]
            sum)))))  ;; [TODO: implicit func.return]

  ;; Main function calls fibonacci(10)
  (func.func {:sym_name main
              :function_type (-> () (i64))}
    []  ;; No arguments

    (let [(: n (arith.constant {:value 10}))
          (: fib-result (func.call {:callee @fibonacci} n))
          (: result-i64 (arith.extsi {} fib-result))]
      result-i64)))  ;; [TODO: implicit func.return, auto-cast to i64]
