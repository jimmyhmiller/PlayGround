; GAP: Tuple types
; MLIR supports tuple types like tuple<i32, f32, i64>
; This file tests tuple type support

; GAP: Tuple types
; Note: builtin dialect operations may need special handling
(require-dialect [func :as f] [builtin :as b])

(module
  (do
    ; Test 1: Tuple type in function signature
    (f/func {:sym_name "tuple_arg"
             :function_type (-> [tuple<i32, f32>] [tuple<i32, f32>])}
      (region
        (block [(: t tuple<i32, f32>)]
          (f/return t))))

    ; Test 2: Return multiple values as tuple
    (f/func {:sym_name "make_tuple"
             :function_type (-> [i32 f32] [tuple<i32, f32>])}
      (region
        (block [(: a i32) (: b f32)]
          ; How to create a tuple value?
          ; MLIR doesn't have a built-in "make tuple" op -
          ; tuples are usually used with unrealized_conversion_cast
          ; or dialect-specific operations
          (def t (builtin.unrealized_conversion_cast {:result tuple<i32, f32>} a b))
          (f/return t))))

    ; Test 3: Extract from tuple
    ; Getting elements from tuples requires cast or specific dialect ops
    (f/func {:sym_name "extract_from_tuple"
             :function_type (-> [tuple<i32, f32>] [i32])}
      (region
        (block [(: t tuple<i32, f32>)]
          ; How to extract element 0 from tuple?
          (def elem (builtin.unrealized_conversion_cast {:result i32} t))
          (f/return elem))))

    ; Test 4: Nested tuple
    (f/func {:sym_name "nested_tuple"
             :function_type (-> [tuple<tuple<i32, i32>, f32>] [tuple<tuple<i32, i32>, f32>])}
      (region
        (block [(: t tuple<tuple<i32, i32>, f32>)]
          (f/return t))))))
