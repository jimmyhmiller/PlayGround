;; Test nested GPU structure
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
                          (name gpu.return))))))))))))))
