; GAP: Dense array attributes
; MLIR supports dense<[1, 2, 3, 4]> : tensor<4xi32> attribute syntax
; This file tests whether lispier can express dense element attributes

(require-dialect [func :as f] [arith :as a])

(module
  (do
    ; Test 1: arith.constant with dense tensor attribute
    ; In MLIR: %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    (f/func {:sym_name "constant_tensor"
             :function_type (-> [] [tensor<4xi32>])}
      (region
        (block []
          ; How do we express dense<[1, 2, 3, 4]> in lispier syntax?
          ; Option 1: As a special attribute syntax
          (def cst (arith.constant {:value "dense<[1, 2, 3, 4]> : tensor<4xi32>"
                                    :result tensor<4xi32>}))
          (f/return cst))))

    ; Test 2: Dense float tensor
    ; In MLIR: %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    (f/func {:sym_name "constant_float_tensor"
             :function_type (-> [] [tensor<4xf32>])}
      (region
        (block []
          (def cst (arith.constant {:value "dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>"
                                    :result tensor<4xf32>}))
          (f/return cst))))

    ; Test 3: Dense 2D tensor
    ; In MLIR: %cst = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
    (f/func {:sym_name "constant_2d_tensor"
             :function_type (-> [] [tensor<2x2xi32>])}
      (region
        (block []
          (def cst (arith.constant {:value "dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>"
                                    :result tensor<2x2xi32>}))
          (f/return cst))))

    ; Test 4: Splat constant (all same value)
    ; In MLIR: %cst = arith.constant dense<0.0> : tensor<4x4xf32>
    (f/func {:sym_name "splat_constant"
             :function_type (-> [] [tensor<4x4xf32>])}
      (region
        (block []
          (def cst (arith.constant {:value "dense<0.0> : tensor<4x4xf32>"
                                    :result tensor<4x4xf32>}))
          (f/return cst))))))
