; GAP: Arith division and remainder operations
; The arith dialect has many operations not yet tested
; This file tests division, remainder, negation, and casting ops

(require-dialect [func :as f] [arith :as a])

(module
  (do
    ; Test 1: Signed integer division
    (f/func {:sym_name "divsi"
             :function_type (-> [i64 i64] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x i64) (: y i64)]
          (def result (a/divsi x y))
          (f/return result))))

    ; Test 2: Unsigned integer division
    (f/func {:sym_name "divui"
             :function_type (-> [i64 i64] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x i64) (: y i64)]
          (def result (a/divui x y))
          (f/return result))))

    ; Test 3: Signed integer remainder
    (f/func {:sym_name "remsi"
             :function_type (-> [i64 i64] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x i64) (: y i64)]
          (def result (a/remsi x y))
          (f/return result))))

    ; Test 4: Float division
    (f/func {:sym_name "divf"
             :function_type (-> [f64 f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64) (: y f64)]
          (def result (a/divf x y))
          (f/return result))))

    ; Test 5: Float negation
    (f/func {:sym_name "negf"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (a/negf x))
          (f/return result))))

    ; Test 6: Integer extension (sign extend i32 to i64)
    (f/func {:sym_name "extsi"
             :function_type (-> [i32] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x i32)]
          (def result (a/extsi {:result i64} x))
          (f/return result))))

    ; Test 7: Integer truncation (i64 to i32)
    (f/func {:sym_name "trunci"
             :function_type (-> [i64] [i32])
             :llvm.emit_c_interface true}
      (region
        (block [(: x i64)]
          (def result (a/trunci {:result i32} x))
          (f/return result))))

    ; Test 8: Float extension (f32 to f64)
    (f/func {:sym_name "extf"
             :function_type (-> [f32] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f32)]
          (def result (a/extf {:result f64} x))
          (f/return result))))

    ; Test 9: Float to int
    (f/func {:sym_name "fptosi"
             :function_type (-> [f64] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (a/fptosi {:result i64} x))
          (f/return result))))

    ; Test 10: Int to float
    (f/func {:sym_name "sitofp"
             :function_type (-> [i64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x i64)]
          (def result (a/sitofp {:result f64} x))
          (f/return result))))))
