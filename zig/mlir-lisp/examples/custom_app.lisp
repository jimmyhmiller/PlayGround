;; Application code with custom dialect

;; Define custom dialect
(operation
  (name irdl.dialect)
  (attributes {:sym_name @custom})
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name irdl.operation)
          (attributes {:sym_name @magic})
          (regions
            (region
              (block
                (arguments [])
                (operation
                  (name irdl.is)
                  (result-bindings [%i32])
                  (result-types !irdl.attribute)
                  (attributes {:expected i32}))
                (operation
                  (name irdl.results)
                  (operand-uses %i32)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>}))))))))))

;; Main function using custom.magic
(operation
  (name func.func)
  (attributes {:sym_name @main :function_type (!function (inputs) (results i64))})
  (regions
    (region
      (block [^entry]
        (arguments [])
        (operation
          (name custom.magic)
          (result-bindings [%magic])
          (result-types i32))
        (operation
          (name arith.extsi)
          (result-bindings [%result])
          (result-types i64)
          (operands %magic))
        (operation
          (name func.return)
          (operands %result))))))
