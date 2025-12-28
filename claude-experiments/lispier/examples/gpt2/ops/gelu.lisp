;; GELU Activation for GPT-2
;; Reference: https://arxiv.org/abs/1606.08415
;;
;; GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
;;
;; This is the approximate GELU used in GPT-2/GPT-3
;; The exact GELU uses the error function, but the tanh approximation is faster

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)
(require-dialect math)

;; Compilation pipeline for ROCm
(compilation
  (target rocm
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl)
    (pass gpu-module-to-binary)
    (pass convert-scf-to-cf)
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-math-to-llvm)
    (pass convert-func-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass gpu-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; GELU forward pass (elementwise)
    ;; out: output tensor (same shape as input)
    ;; inp: input tensor
    ;; N: total number of elements
    ;;
    ;; Using 1D memrefs for simplicity - can be reshaped for any tensor
    (func.func {:sym_name "gelu_forward"
                :function_type (-> [memref<49152xf32> memref<49152xf32>] [])}  ; 1 * 64 * 768 = 49152
      (region
        (block [(: out memref<49152xf32>) (: inp memref<49152xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def N (: 49152 index))

          ;; Constants for GELU
          ;; GELU_SCALING_FACTOR = sqrt(2/pi) ≈ 0.7978845608
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))
          (def coeff (: 0.044715 f32))
          (def sqrt_2_pi (: 0.7978845608 f32))

          (scf.for c0 N c1
            (region
              (block [(: i index)]
                (def x (memref.load {:result f32} inp i))

                ;; x^3
                (def x_sq (arith.mulf x x))
                (def x_cube (arith.mulf x_sq x))

                ;; 0.044715 * x^3
                (def cube_term (arith.mulf coeff x_cube))

                ;; x + 0.044715 * x^3
                (def inner (arith.addf x cube_term))

                ;; sqrt(2/pi) * (x + 0.044715 * x^3)
                (def scaled (arith.mulf sqrt_2_pi inner))

                ;; tanh(...)
                (def tanh_val (math.tanh scaled))

                ;; 1 + tanh(...)
                (def one_plus_tanh (arith.addf one tanh_val))

                ;; 0.5 * x
                (def half_x (arith.mulf half x))

                ;; 0.5 * x * (1 + tanh(...))
                (def result (arith.mulf half_x one_plus_tanh))

                (memref.store result out i)
                (scf.yield))))

          (func.return))))

    ;; GPU version using gpu.launch
    (func.func {:sym_name "gelu_forward_gpu"
                :function_type (-> [memref<49152xf32> memref<49152xf32>] [])}
      (region
        (block [(: out memref<49152xf32>) (: inp memref<49152xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c192 (: 192 index))  ; 49152 / 256 = 192 blocks
          (def c256 (: 256 index))  ; 256 threads per block

          ;; Constants for GELU
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))
          (def coeff (: 0.044715 f32))
          (def sqrt_2_pi (: 0.7978845608 f32))

          ;; GPU launch: 192 blocks x 256 threads = 49152 threads
          (gpu.launch {:operandSegmentSizes array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}
            c192 c1 c1 c256 c1 c1
            (region
              (block [(: bx index) (: by index) (: bz index)
                      (: tx index) (: ty index) (: tz index)
                      (: gridDimX index) (: gridDimY index) (: gridDimZ index)
                      (: blockDimX index) (: blockDimY index) (: blockDimZ index)]

                ;; Global thread index
                (def i_part (arith.muli bx blockDimX))
                (def i (arith.addi i_part tx))

                (def x (memref.load {:result f32} inp i))

                ;; GELU computation
                (def x_sq (arith.mulf x x))
                (def x_cube (arith.mulf x_sq x))
                (def cube_term (arith.mulf coeff x_cube))
                (def inner (arith.addf x cube_term))
                (def scaled (arith.mulf sqrt_2_pi inner))
                (def tanh_val (math.tanh scaled))
                (def one_plus_tanh (arith.addf one tanh_val))
                (def half_x (arith.mulf half x))
                (def result (arith.mulf half_x one_plus_tanh))

                (memref.store result out i)
                (gpu.terminator))))

          (func.return))))

    ;; Test function
    (func.func {:sym_name "test_gelu"
                :function_type (-> [] [])}
      (region
        (block []
          (def out (memref.alloc {:result memref<49152xf32>}))
          (def inp (memref.alloc {:result memref<49152xf32>}))

          ;; Initialize with some values
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def N (: 49152 index))

          (scf.for c0 N c1
            (region
              (block [(: i index)]
                (def i_f (arith.index_cast {:result i64} i))
                (def val_f (arith.sitofp {:result f32} i_f))
                ;; Scale to [-3, 3] range
                (def scale (: 0.000122 f32))  ; 6 / 49152
                (def three (: 3.0 f32))
                (def val (arith.subf (arith.mulf val_f scale) three))
                (memref.store val inp i)
                (scf.yield))))

          ;; Run GELU
          (func.call "gelu_forward" out inp)

          ;; Cleanup
          (memref.dealloc out)
          (memref.dealloc inp)

          (func.return))))))
