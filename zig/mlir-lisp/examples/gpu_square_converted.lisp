(operation
    (name builtin.module)
    (attributes {:gpu.container_module true})
    (regions
      (region
        (block
          (arguments [])
          (operation
            (name gpu.module)
            (attributes {:sym_name @kernel_module})
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name gpu.func)
                    (attributes {:gpu.kernel true :sym_name @square_kernel :workgroup_attributions (: 0 i64) :function_type (!function (inputs memref<10x10xf32> memref<10x10xf32>) (results))})
                    (regions
                      (region
                        (block [^bb0]
                          (arguments [(: %arg0 memref<10x10xf32>) (: %arg1 memref<10x10xf32>)])
                          (operation
                            (name gpu.block_id)
                            (result-bindings [%5])
                            (result-types index)
                            (attributes {:dimension #gpu<dim x>}))
                          (operation
                            (name gpu.thread_id)
                            (result-bindings [%6])
                            (result-types index)
                            (attributes {:dimension #gpu<dim x>}))
                          (operation
                            (name memref.load)
                            (result-bindings [%7])
                            (result-types f32)
                            (operands %arg0 %5 %6))
                          (operation
                            (name arith.mulf)
                            (result-bindings [%8])
                            (result-types f32)
                            (operands %7 %7)
                            (attributes {:fastmath #arith.fastmath<none>}))
                          (operation
                            (name memref.store)
                            (operands %8 %arg1 %5 %6))
                          (operation
                            (name gpu.return))))))))))
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
                    (name arith.constant)
                    (result-bindings [%c0])
                    (result-types index)
                    (attributes {:value (: 0 index)}))
                  (operation
                    (name arith.constant)
                    (result-bindings [%f5])
                    (result-types f32)
                    (attributes {:value (: 5.0 f32)}))
                  (operation
                    (name memref.alloc)
                    (result-bindings [%input])
                    (result-types memref<10x10xf32>)
                    (attributes {:operandSegmentSizes array<i32: 0, 0>}))
                  (operation
                    (name memref.alloc)
                    (result-bindings [%output])
                    (result-types memref<10x10xf32>)
                    (attributes {:operandSegmentSizes array<i32: 0, 0>}))
                  (operation
                    (name scf.for)
                    (operands %c0 %c10 %c1)
                    (regions
                      (region
                        (block [^loop1]
                          (arguments [(: %i index)])
                          (operation
                            (name scf.for)
                            (operands %c0 %c10 %c1)
                            (regions
                              (region
                                (block [^loop2]
                                  (arguments [(: %j index)])
                                  (operation
                                    (name memref.store)
                                    (operands %f5 %input %i %j))
                                  (operation
                                    (name scf.yield))))))
                          (operation
                            (name scf.yield))))))
                  (operation
                    (name gpu.launch_func)
                    (operands %c10 %c1 %c1 %c10 %c1 %c1 %input %output)
                    (attributes {:kernel @kernel_module::@square_kernel :operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 0>}))
                  (operation
                    (name memref.dealloc)
                    (operands %input))
                  (operation
                    (name memref.dealloc)
                    (operands %output))
                  (operation
                    (name arith.constant)
                    (result-bindings [%4])
                    (result-types i64)
                    (attributes {:value (: 0 i64)}))
                  (operation
                    (name func.return)
                    (operands %4))))))))))
