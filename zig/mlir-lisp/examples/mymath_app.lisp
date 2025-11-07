;; IRDL: Define mymath dialect
(operation
  (name irdl.dialect)
  (attributes {:sym_name @mymath})
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name irdl.operation)
          (attributes {:sym_name @add})
          (regions
            (region
              (block
                (arguments [])
                (operation
                  (name irdl.is)
                  (result-bindings [%0])
                  (result-types !irdl.attribute)
                  (attributes {:expected i8}))
                (operation
                  (name irdl.is)
                  (result-bindings [%1])
                  (result-types !irdl.attribute)
                  (attributes {:expected i16}))
                (operation
                  (name irdl.is)
                  (result-bindings [%2])
                  (result-types !irdl.attribute)
                  (attributes {:expected i32}))
                (operation
                  (name irdl.is)
                  (result-bindings [%3])
                  (result-types !irdl.attribute)
                  (attributes {:expected i64}))
                (operation
                  (name irdl.is)
                  (result-bindings [%4])
                  (result-types !irdl.attribute)
                  (attributes {:expected f32}))
                (operation
                  (name irdl.is)
                  (result-bindings [%5])
                  (result-types !irdl.attribute)
                  (attributes {:expected f64}))
                (operation
                  (name irdl.any_of)
                  (operand-uses %0 %1 %2 %3 %4 %5)
                  (result-bindings [%arith_type])
                  (result-types !irdl.attribute))
                (operation
                  (name irdl.operands)
                  (operand-uses %arith_type %arith_type)
                  (attributes {:names ["lhs" "rhs"] :variadicity #irdl<variadicity_array[ single,  single]>}))
                (operation
                  (name irdl.results)
                  (operand-uses %arith_type)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>}))))))))))

;; Application code
(operation
  (name func.func)
  (attributes {:sym_name @main :function_type (!function (inputs) (results i64))})
  (regions
    (region
      (block [^entry]
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%c10])
          (result-types i32)
          (attributes {:value (: 10 i32)}))
        (operation
          (name arith.constant)
          (result-bindings [%c32])
          (result-types i32)
          (attributes {:value (: 32 i32)}))
        (operation
          (name mymath.add)
          (operand-uses %c10 %c32)
          (result-bindings [%result])
          (result-types i32))
        (operation
          (name arith.extsi)
          (operand-uses %result)
          (result-bindings [%result64])
          (result-types i64))
        (operation
          (name func.return)
          (operands %result64))))))
