; GAP: Complex number types
; MLIR supports complex<f32>, complex<f64> types
; This file tests complex type support

(require-dialect [func :as f] [arith :as a] [complex :as c])

(module
  (do
    ; Test 1: Complex type in function signature
    (f/func {:sym_name "complex_arg"
             :function_type (-> [complex<f32>] [complex<f32>])}
      (region
        (block [(: z complex<f32>)]
          (f/return z))))

    ; Test 2: complex.create - create complex from real and imaginary parts
    (f/func {:sym_name "create_complex"
             :function_type (-> [f32 f32] [complex<f32>])}
      (region
        (block [(: real f32) (: imag f32)]
          (def z (c/create {:result complex<f32>} real imag))
          (f/return z))))

    ; Test 3: complex.re - extract real part
    (f/func {:sym_name "get_real"
             :function_type (-> [complex<f32>] [f32])}
      (region
        (block [(: z complex<f32>)]
          (def real (c/re {:result f32} z))
          (f/return real))))

    ; Test 4: complex.im - extract imaginary part
    (f/func {:sym_name "get_imag"
             :function_type (-> [complex<f32>] [f32])}
      (region
        (block [(: z complex<f32>)]
          (def imag (c/im {:result f32} z))
          (f/return imag))))

    ; Test 5: complex.add - add two complex numbers
    (f/func {:sym_name "complex_add"
             :function_type (-> [complex<f32> complex<f32>] [complex<f32>])}
      (region
        (block [(: a complex<f32>) (: b complex<f32>)]
          (def result (c/add a b))
          (f/return result))))

    ; Test 6: complex.mul - multiply two complex numbers
    (f/func {:sym_name "complex_mul"
             :function_type (-> [complex<f32> complex<f32>] [complex<f32>])}
      (region
        (block [(: a complex<f32>) (: b complex<f32>)]
          (def result (c/mul a b))
          (f/return result))))

    ; Test 7: complex.abs - absolute value (magnitude)
    (f/func {:sym_name "complex_abs"
             :function_type (-> [complex<f32>] [f32])}
      (region
        (block [(: z complex<f32>)]
          (def mag (c/abs {:result f32} z))
          (f/return mag))))

    ; Test 8: complex.conj - complex conjugate
    (f/func {:sym_name "complex_conj"
             :function_type (-> [complex<f32>] [complex<f32>])}
      (region
        (block [(: z complex<f32>)]
          (def conj (c/conj z))
          (f/return conj))))))
