; WORKING: Math dialect operations
; The math dialect provides transcendental and other math operations
; Status: All math operations work correctly (sqrt, exp, log, trig, etc.)

(require-dialect [func :as f] [math :as m])

(module
  (do
    ; Test 1: Square root
    (f/func {:sym_name "sqrt"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/sqrt x))
          (f/return result))))

    ; Test 2: Exponential
    (f/func {:sym_name "exp"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/exp x))
          (f/return result))))

    ; Test 3: Natural logarithm
    (f/func {:sym_name "log"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/log x))
          (f/return result))))

    ; Test 4: Sine
    (f/func {:sym_name "sin"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/sin x))
          (f/return result))))

    ; Test 5: Cosine
    (f/func {:sym_name "cos"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/cos x))
          (f/return result))))

    ; Test 6: Tangent
    (f/func {:sym_name "tan"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/tan x))
          (f/return result))))

    ; Test 7: Power
    (f/func {:sym_name "powf"
             :function_type (-> [f64 f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: base f64) (: exp f64)]
          (def result (m/powf base exp))
          (f/return result))))

    ; Test 8: Absolute value
    (f/func {:sym_name "absf"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/absf x))
          (f/return result))))

    ; Test 9: Floor
    (f/func {:sym_name "floor"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/floor x))
          (f/return result))))

    ; Test 10: Ceil
    (f/func {:sym_name "ceil"
             :function_type (-> [f64] [f64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x f64)]
          (def result (m/ceil x))
          (f/return result))))))
