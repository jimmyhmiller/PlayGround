;; Layer Normalization for GPT-2
;; Reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
;;
;; LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
;;
;; For GPT-2:
;; - Input: (B, T, C) - batch, sequence length, channels
;; - Normalizes over the last dimension (C)
;; - gamma (weight) and beta (bias) are (C,) vectors

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
    ;; LayerNorm forward pass
    ;; out: (B, T, C) - normalized output
    ;; mean: (B, T) - computed mean (cached for backward)
    ;; rstd: (B, T) - reciprocal std dev (cached for backward)
    ;; inp: (B, T, C) - input
    ;; weight: (C,) - gamma scale parameter
    ;; bias: (C,) - beta shift parameter
    ;; eps: scalar - epsilon for numerical stability
    ;;
    ;; For simplicity, using fixed sizes: B=1, T=64, C=768 (GPT-2 small)
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<1x64x768xf32>   ; out
                                    memref<1x64xf32>       ; mean
                                    memref<1x64xf32>       ; rstd
                                    memref<1x64x768xf32>   ; inp
                                    memref<768xf32>        ; weight
                                    memref<768xf32>        ; bias
                                    f32] [])}              ; eps
      (region
        (block [(: out memref<1x64x768xf32>)
                (: mean memref<1x64xf32>)
                (: rstd memref<1x64xf32>)
                (: inp memref<1x64x768xf32>)
                (: weight memref<768xf32>)
                (: bias memref<768xf32>)
                (: eps f32)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def C_f32 (: 768.0 f32))
          (def zero (: 0.0 f32))

          ;; Loop over batch and sequence positions
          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      ;; Step 1: Compute mean over C dimension
                      (def sum_val (scf.for {:result f32} c0 C c1 zero
                        (region
                          (block [(: c index) (: acc f32)]
                            (def x (memref.load {:result f32} inp b t c))
                            (def new_acc (arith.addf acc x))
                            (scf.yield new_acc)))))
                      (def m (arith.divf sum_val C_f32))
                      (memref.store m mean b t)

                      ;; Step 2: Compute variance over C dimension
                      (def var_val (scf.for {:result f32} c0 C c1 zero
                        (region
                          (block [(: c index) (: vacc f32)]
                            (def x (memref.load {:result f32} inp b t c))
                            (def diff (arith.subf x m))
                            (def diff_sq (arith.mulf diff diff))
                            (def new_vacc (arith.addf vacc diff_sq))
                            (scf.yield new_vacc)))))
                      (def variance (arith.divf var_val C_f32))

                      ;; Step 3: Compute reciprocal std dev
                      (def var_eps (arith.addf variance eps))
                      (def std (math.sqrt var_eps))
                      (def one (: 1.0 f32))
                      (def rs (arith.divf one std))
                      (memref.store rs rstd b t)

                      ;; Step 4: Normalize, scale and shift
                      (scf.for c0 C c1
                        (region
                          (block [(: c index)]
                            (def x (memref.load {:result f32} inp b t c))
                            (def x_norm (arith.mulf (arith.subf x m) rs))
                            (def gamma (memref.load {:result f32} weight c))
                            (def beta (memref.load {:result f32} bias c))
                            (def scaled (arith.mulf x_norm gamma))
                            (def result (arith.addf scaled beta))
                            (memref.store result out b t c)
                            (scf.yield))))

                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; Test function
    (func.func {:sym_name "test_layernorm"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate tensors
          (def out (memref.alloc {:result memref<1x64x768xf32>}))
          (def mean (memref.alloc {:result memref<1x64xf32>}))
          (def rstd (memref.alloc {:result memref<1x64xf32>}))
          (def inp (memref.alloc {:result memref<1x64x768xf32>}))
          (def weight (memref.alloc {:result memref<768xf32>}))
          (def bias (memref.alloc {:result memref<768xf32>}))

          ;; Initialize weight to 1.0, bias to 0.0
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def C (: 768 index))
          (def one (: 1.0 f32))
          (def zero (: 0.0 f32))

          (scf.for c0 C c1
            (region
              (block [(: c index)]
                (memref.store one weight c)
                (memref.store zero bias c)
                (scf.yield))))

          ;; Initialize inp with some values
          (def B (: 1 index))
          (def T (: 64 index))
          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 C c1
                        (region
                          (block [(: c index)]
                            ;; Simple pattern: value = (b + t + c) * 0.01
                            (def b_idx (arith.index_cast {:result i64} b))
                            (def t_idx (arith.index_cast {:result i64} t))
                            (def c_idx (arith.index_cast {:result i64} c))
                            (def sum1 (arith.addi b_idx t_idx))
                            (def sum2 (arith.addi sum1 c_idx))
                            (def val_f (arith.sitofp {:result f32} sum2))
                            (def val (arith.mulf val_f (: 0.01 f32)))
                            (memref.store val inp b t c)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          ;; Run layernorm
          (def eps (: 0.00001 f32))
          (func.call "layernorm_forward" out mean rstd inp weight bias eps)

          ;; Clean up
          (memref.dealloc out)
          (memref.dealloc mean)
          (memref.dealloc rstd)
          (memref.dealloc inp)
          (memref.dealloc weight)
          (memref.dealloc bias)

          (func.return))))))
