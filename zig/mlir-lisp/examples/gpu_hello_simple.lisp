(operation
  (name builtin.module)
  (attributes {:gpu.container_module true})
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name func.func)
          (attributes {:function_type (!function (inputs) (results i64)) :sym_name @main})
          (regions
            (region
              (block
                (arguments [])
                (operation
                  (name arith.constant)
                  (result-bindings [%c10])
                  (result-types index)
                  (attributes {:value (: 10 index)}))
                (operation
                  (name arith.constant)
                  (result-bindings [%c1])
                  (result-types index)
                  (attributes {:value (: 1 index)}))
                (operation
                  (name memref.alloc)
                  (result-bindings [%output])
                  (result-types memref<10xindex>)
                  (attributes {:operandSegmentSizes array<i32: 0, 0>}))
                (operation
                  (name gpu.launch)
                  (operand-uses %c10 %c1 %c1 %c1 %c1 %c1)
                  (attributes {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>})
                  (regions
                    (region
                      (block
                        (arguments [(: %bx index) (: %by index) (: %bz index) (: %tx index) (: %ty index) (: %tz index) (: %num_bx index) (: %num_by index) (: %num_bz index) (: %num_tx index) (: %num_ty index) (: %num_tz index)])
                        (operation
                          (name memref.store)
                          (operand-uses %bx %output %bx))
                        (operation
                          (name gpu.terminator))))))
                (operation
                  (name memref.dealloc)
                  (operand-uses %output))
                (operation
                  (name arith.constant)
                  (result-bindings [%zero])
                  (result-types i64)
                  (attributes {:value (: 0 i64)}))
                (operation
                  (name func.return)
                  (operand-uses %zero))))))))))
