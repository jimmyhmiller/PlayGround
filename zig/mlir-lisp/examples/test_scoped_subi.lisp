;; Test if terse operations work inside scf.if regions

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

          ;; Create values in outer scope
          (declare c10 (arith.constant {:value (: 10 i32)}))
          (declare c1 (arith.constant {:value (: 1 i32)}))

          ;; Test comparison
          (operation
            (name arith.cmpi)
            (result-bindings [%cond])
            (result-types i1)
            (operands %c10 %c1)
            (attributes { :predicate (: 4 i64) }))  ;; sgt: >

          ;; Use scf.if with terse operations inside
          (operation
            (name scf.if)
            (result-bindings [%result])
            (result-types i32)
            (operands %cond)
            (regions
              ;; Then: return c10
              (region
                (block
                  (arguments [])
                  (operation
                    (name scf.yield)
                    (operands %c10))))

              ;; Else: compute c10 - c1 using TERSE
              (region
                (block
                  (arguments [])

                  ;; ✅ Try terse subi inside nested region
                  (declare diff (arith.subi %c10 %c1))

                  (operation
                    (name scf.yield)
                    (operands %diff))))))

          ;; Convert and return
          (operation
            (name arith.extsi)
            (result-bindings [%result_i64])
            (result-types i64)
            (operands %result))

          (operation
            (name func.return)
            (operands %result_i64)))))))
