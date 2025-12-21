; GAP: Tensor types
; MLIR supports tensor types like tensor<4x4xf32>, tensor<?xf32>, tensor<*xf32>
; This file tests whether the lispier syntax supports tensor types

(require-dialect [func :as f] [arith :as a] [tensor :as t])

(module
  (do
    ; Test 1: Static tensor type in function signature
    (f/func {:sym_name "tensor_arg"
             :function_type (-> [tensor<4x4xf32>] [tensor<4x4xf32>])}
      (region
        (block [(: input tensor<4x4xf32>)]
          (f/return input))))

    ; Test 2: Dynamic tensor type
    (f/func {:sym_name "dynamic_tensor"
             :function_type (-> [tensor<?xf32>] [tensor<?xf32>])}
      (region
        (block [(: input tensor<?xf32>)]
          (f/return input))))

    ; Test 3: tensor.empty operation to create a tensor
    (f/func {:sym_name "create_tensor"
             :function_type (-> [] [tensor<4xf32>])}
      (region
        (block []
          (def t (t/empty {:result tensor<4xf32>}))
          (f/return t))))

    ; Test 4: tensor.extract - extract scalar from tensor
    (f/func {:sym_name "extract_element"
             :function_type (-> [tensor<4xf32>] [f32])}
      (region
        (block [(: t tensor<4xf32>)]
          (def idx (: 0 index))
          (def elem (t/extract {:result f32} t idx))
          (f/return elem))))))
