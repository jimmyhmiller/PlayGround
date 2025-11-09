(operation
  (name func.func)
  (attributes {}
    :sym_name @main
    :function_type (!function (inputs) (results i64)))
  
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%true])
          (result-types i1)
          (attributes { :value (: 1 i1)}))
        (operation
          (name scf.if)
          (result-bindings [%result])
          (result-types i64)
          (operands %true)
          (regions
            (region
              (block
                (arguments [])
                (operation
                  (name arith.constant)
                  (result-bindings [%then_val])
                  (result-types i64)
                  (attributes { :value (: 42 i64)}))
                (operation
                  (name scf.yield)
                  (operands %then_val))))
            (region
              (block
                (arguments [])
                (operation
                  (name arith.constant)
                  (result-bindings [%else_val])
                  (result-types i64)
                  (attributes { :value (: 0 i64)}))
                (operation
                  (name scf.yield)
                  (operands %else_val))))))
        (operation
          (name func.return)
          (operands %result))))))
