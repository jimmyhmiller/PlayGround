;; Test exact attributes from gpu.func
(operation
  (name gpu.func)
  (attributes {:gpu.kernel true :sym_name @square_kernel :workgroup_attributions (: 0 i64) :function_type (!function (inputs memref<10x10xf32> memref<10x10xf32>) (results))}))
