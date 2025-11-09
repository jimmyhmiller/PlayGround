(operation
    (name builtin.module)
    (attributes {:gpu.container_module true})
    (regions
      (region
        (block
          (arguments [])
          (operation
            (name gpu.module)
            (attributes {:sym_name @matmul_kernel})
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name gpu.func)
                    (attributes {:gpu.kernel true :sym_name @matmul :workgroup_attributions (: 0 i64) :function_type (!function (inputs memref<16x16xf32> memref<16x16xf32> memref<16x16xf32>) (results))})
                    (regions
                      (region
                        (block [^bb0]
                          (arguments [(: %arg0 memref<16x16xf32>) (: %arg1 memref<16x16xf32>) (: %arg2 memref<16x16xf32>)])
                          (operation
                            (name gpu.block_id)
                            (result-bindings [%6])
                            (result-types index)
                            (attributes {:dimension #gpu<dim x>}))
                          (operation
                            (name gpu.block_id)
                            (result-bindings [%7])
                            (result-types index)
                            (attributes {:dimension #gpu<dim y>}))
                          (operation
                            (name arith.constant)
                            (result-bindings [%8])
                            (result-types f32)
                            (attributes {:value (: 0.000000e+00 f32)}))
                          (operation
                            (name arith.constant)
                            (result-bindings [%9])
                            (result-types index)
                            (attributes {:value (: 0 index)}))
                          (operation
                            (name arith.constant)
                            (result-bindings [%10])
                            (result-types index)
                            (attributes {:value (: 16 index)}))
                          (operation
                            (name arith.constant)
                            (result-bindings [%11])
                            (result-types index)
                            (attributes {:value (: 1 index)}))
                          (operation
                            (name scf.for)
                            (result-bindings [%12])
                            (result-types f32)
                            (operands %9 %10 %11 %8)
                            (regions
                              (region
                                (block [^bb0]
                                  (arguments [(: %arg3 index) (: %arg4 f32)])
                                  (operation
                                    (name memref.load)
                                    (result-bindings [%13])
                                    (result-types f32)
                                    (operands %arg0 %6 %arg3))
                                  (operation
                                    (name memref.load)
                                    (result-bindings [%14])
                                    (result-types f32)
                                    (operands %arg1 %arg3 %7))
                                  (operation
                                    (name arith.mulf)
                                    (result-bindings [%15])
                                    (result-types f32)
                                    (operands %13 %14)
                                    (attributes {:fastmath #arith.fastmath<none>}))
                                  (operation
                                    (name arith.addf)
                                    (result-bindings [%16])
                                    (result-types f32)
                                    (operands %arg4 %15)
                                    (attributes {:fastmath #arith.fastmath<none>}))
                                  (operation
                                    (name scf.yield)
                                    (operands %16))))))
                          (operation
                            (name memref.store)
                            (operands %12 %arg2 %6 %7))
                          (operation
                            (name gpu.return))))))))))
          (operation
            (name func.func)
            (attributes {:function_type (!function (inputs) (results i32)) :sym_name @main})
            (regions
              (region
                (block
                  (arguments [])
                  (operation
                    (name arith.constant)
                    (result-bindings [%0])
                    (result-types index)
                    (attributes {:value (: 16 index)}))
                  (operation
                    (name arith.constant)
                    (result-bindings [%1])
                    (result-types index)
                    (attributes {:value (: 1 index)}))
                  (operation
                    (name memref.alloc)
                    (result-bindings [%2])
                    (result-types memref<16x16xf32>)
                    (attributes {:operandSegmentSizes array<i32: 0, 0>}))
                  (operation
                    (name memref.alloc)
                    (result-bindings [%3])
                    (result-types memref<16x16xf32>)
                    (attributes {:operandSegmentSizes array<i32: 0, 0>}))
                  (operation
                    (name memref.alloc)
                    (result-bindings [%4])
                    (result-types memref<16x16xf32>)
                    (attributes {:operandSegmentSizes array<i32: 0, 0>}))
                  (operation
                    (name gpu.launch_func)
                    (operands %0 %0 %1 %1 %1 %1 %2 %3 %4)
                    (attributes {:kernel @matmul_kernel::@matmul :operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 0>}))
                  (operation
                    (name arith.constant)
                    (result-bindings [%5])
                    (result-types i32)
                    (attributes {:value (: 0 i32)}))
                  (operation
                    (name func.return)
                    (operands %5))))))))))
