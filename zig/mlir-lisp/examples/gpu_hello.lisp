(operation
  (name builtin.module)
  (attributes {:gpu.container_module true})
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name func.func)
          (attributes {:function_type (!function (inputs) (results)) :sym_name @main})
          (regions
            (region
              (block
                (arguments [])
                (operation
                  (name arith.constant)
                  (result-bindings [%c2])
                  (result-types index)
                  (attributes {:value (: 2 index)}))
                (operation
                  (name arith.constant)
                  (result-bindings [%c1])
                  (result-types index)
                  (attributes {:value (: 1 index)}))
                (operation
                  (name gpu.launch)
                  (operands %c1 %c1 %c1 %c2 %c1 %c1)
                  (attributes {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>})
                  (regions
                    (region
                      (block
                        (arguments [(: %bx index) (: %by index) (: %bz index) (: %tx index) (: %ty index) (: %tz index) (: %num_bx index) (: %num_by index) (: %num_bz index) (: %num_tx index) (: %num_ty index) (: %num_tz index)])
                        (operation
                          (name gpu.printf)
                          (operands %tx)
                          (attributes {:format "Hello from thread %lld\n"}))
                        (operation
                          (name gpu.terminator))))))
                (operation
                  (name func.return))))))))))
