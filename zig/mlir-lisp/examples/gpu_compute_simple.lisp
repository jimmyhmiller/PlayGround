;; Simple GPU computation that returns a value (no memory access)
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
                  (result-bindings [%c1])
                  (result-types index)
                  (attributes {:value (: 1 index)}))
                (operation
                  (name arith.constant)
                  (result-bindings [%c42])
                  (result-types i64)
                  (attributes {:value (: 42 i64)}))
                ;; Launch GPU kernel with 1 block, 1 thread - no memory operations
                (operation
                  (name gpu.launch)
                  (operands %c1 %c1 %c1 %c1 %c1 %c1)
                  (attributes {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>})
                  (regions
                    (region
                      (block
                        (arguments [(: %bx index) (: %by index) (: %bz index) (: %tx index) (: %ty index) (: %tz index) (: %num_bx index) (: %num_by index) (: %num_bz index) (: %num_tx index) (: %num_ty index) (: %num_tz index)])
                        ;; Just do some arithmetic - no memory access
                        (operation
                          (name gpu.terminator))))))
                ;; Return the value
                (operation
                  (name func.return)
                  (operands %c42))))))))))
