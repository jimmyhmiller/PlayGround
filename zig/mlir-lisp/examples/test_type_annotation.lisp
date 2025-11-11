;; Test explicit type annotations in declare

(mlir
  (operation
    (name func.func)
    (attributes {
      :sym_name @main
      :function_type (!function (inputs) (results i64))
    })
    (regions
      (region
        (block
          (arguments [])

          ;; Test with explicit type annotation
          (declare c10 (arith.constant {:value (: 10 i32)}))
          (declare c3 (arith.constant {:value (: 3 i32)}))

          ;; Test arith.cmpi with explicit type annotation
          (declare cond (: (arith.cmpi {:predicate (: 4 i64)} %c10 %c3) i1))

          ;; Test func.call with explicit type annotation (simulate a call)
          (declare result (: (arith.subi %c10 %c3) i32))

          ;; Convert and return
          (operation
            (name arith.extsi)
            (result-bindings [%result_i64])
            (result-types i64)
            (operands %result))

          (operation
            (name func.return)
            (operands %result_i64)))))))
