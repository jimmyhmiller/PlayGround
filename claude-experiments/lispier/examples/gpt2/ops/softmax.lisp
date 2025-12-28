;; Softmax for GPT-2
;; Reference: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
;;
;; softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
;;
;; For numerical stability, we subtract the max before exponentiating.
;; This is applied along the last dimension (vocabulary for logits).

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
    ;; Softmax forward pass
    ;; probs: (B, T, V) - output probabilities
    ;; logits: (B, T, V) - input logits (unnormalized log probabilities)
    ;; V: vocab size (actual, not padded)
    ;; Vp: padded vocab size (for memory alignment)
    ;;
    ;; For GPT-2 small: V=50257, Vp=50304 (padded to 128)
    ;; Using smaller sizes for testing: B=1, T=64, V=1024, Vp=1024
    (func.func {:sym_name "softmax_forward"
                :function_type (-> [memref<1x64x1024xf32>   ; probs
                                    memref<1x64x1024xf32>   ; logits
                                    index                    ; V
                                    index] [])}              ; Vp
      (region
        (block [(: probs memref<1x64x1024xf32>)
                (: logits memref<1x64x1024xf32>)
                (: V index)
                (: Vp index)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def neg_inf (: -10000.0 f32))
          (def zero (: 0.0 f32))

          ;; Loop over batch and sequence positions
          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      ;; Step 1: Find max for numerical stability
                      (def max_val (scf.for {:result f32} c0 V c1 neg_inf
                        (region
                          (block [(: v index) (: curr_max f32)]
                            (def logit (memref.load {:result f32} logits b t v))
                            ;; max(curr_max, logit)
                            (def is_greater (arith.cmpf {:predicate "ogt"} logit curr_max))
                            (def new_max (arith.select is_greater logit curr_max))
                            (scf.yield new_max)))))

                      ;; Step 2: Compute exp(x - max) and sum
                      (def exp_sum (scf.for {:result f32} c0 V c1 zero
                        (region
                          (block [(: v index) (: sum_acc f32)]
                            (def logit (memref.load {:result f32} logits b t v))
                            (def shifted (arith.subf logit max_val))
                            (def exp_val (math.exp shifted))
                            ;; Store exp value temporarily in probs
                            (memref.store exp_val probs b t v)
                            (def new_sum (arith.addf sum_acc exp_val))
                            (scf.yield new_sum)))))

                      ;; Step 3: Normalize by dividing by sum
                      (scf.for c0 V c1
                        (region
                          (block [(: v index)]
                            (def exp_val (memref.load {:result f32} probs b t v))
                            (def prob (arith.divf exp_val exp_sum))
                            (memref.store prob probs b t v)
                            (scf.yield))))

                      ;; Step 4: Zero out padded positions (V to Vp)
                      (scf.for V Vp c1
                        (region
                          (block [(: v index)]
                            (memref.store zero probs b t v)
                            (scf.yield))))

                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; Test function
    (func.func {:sym_name "test_softmax"
                :function_type (-> [] [])}
      (region
        (block []
          (def probs (memref.alloc {:result memref<1x64x1024xf32>}))
          (def logits (memref.alloc {:result memref<1x64x1024xf32>}))

          ;; Initialize with some values
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def V (: 1024 index))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 V c1
                        (region
                          (block [(: v index)]
                            ;; Random-ish logits based on position
                            (def v_f (arith.index_cast {:result i64} v))
                            (def val_f (arith.sitofp {:result f32} v_f))
                            (def scale (: 0.01 f32))
                            (def val (arith.mulf val_f scale))
                            (memref.store val logits b t v)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          ;; Run softmax
          (def V_idx (: 1024 index))
          (def Vp_idx (: 1024 index))
          (func.call "softmax_forward" probs logits V_idx Vp_idx)

          ;; Cleanup
          (memref.dealloc probs)
          (memref.dealloc logits)

          (func.return))))))
