;; Multi-Head Causal Self-Attention for GPT-2
;;
;; This file implements attention using CPU scf.for loops.
;; The algorithm follows the llm.c reference implementation.
;;
;; For B=1, T=64, C=768, NH=12 (GPT-2 Small):
;; - head_size (hs) = 768/12 = 64
;; - QKV input: (B, T, 3*C) where Q, K, V each have size C
;; - preatt: (B, NH, T, T) - pre-softmax attention scores
;; - att: (B, NH, T, T) - post-softmax attention weights
;; - out: (B, T, C) - attention output
;;
;; Computation per (batch, head, time):
;; 1. Q[t] dot K[t2] for all t2, scale by 1/sqrt(head_size)
;; 2. Causal mask: set -inf for t2 > t
;; 3. Softmax over t2 dimension
;; 4. Weighted sum of V vectors
;;
;; Memory layout in QKV tensor (B, T, 3*C):
;; - Q for head h: qkv[b, t, h*hs : (h+1)*hs]
;; - K for head h: qkv[b, t, C + h*hs : C + (h+1)*hs]
;; - V for head h: qkv[b, t, 2*C + h*hs : 2*C + (h+1)*hs]

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)
(require-dialect math)
(require-dialect vector)

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
    ;; Attention forward pass
    ;; out: (B, T, C) - attention output
    ;; preatt: (B, NH, T, T) - pre-softmax attention scores (scratch)
    ;; att: (B, NH, T, T) - post-softmax attention weights (scratch)
    ;; qkv: (B, T, 3*C) - input QKV projections
    ;;
    ;; For GPT-2 small: B=1, T=64, C=768, NH=12, hs=64
    (func.func {:sym_name "attention_forward"
                :function_type (-> [memref<1x64x768xf32>      ; out
                                    memref<1x12x64x64xf32>    ; preatt
                                    memref<1x12x64x64xf32>    ; att
                                    memref<1x64x2304xf32>] [])} ; qkv (3*768=2304)
      (region
        (block [(: out memref<1x64x768xf32>)
                (: preatt memref<1x12x64x64xf32>)
                (: att memref<1x12x64x64xf32>)
                (: qkv memref<1x64x2304xf32>)]

          ;; Constants
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def NH (: 12 index))
          (def hs (: 64 index))  ; head_size = C / NH = 768 / 12 = 64

          ;; Scalars
          (def zero (: 0.0 f32))
          (def neg_inf (: -10000.0 f32))
          (def scale (: 0.125 f32))  ; 1/sqrt(64) = 1/8 = 0.125

          ;; Loop over batch
          (scf.for c0 B c1
            (region
              (block [(: b index)]
                ;; Loop over heads
                (scf.for c0 NH c1
                  (region
                    (block [(: h index)]
                      ;; Compute base offset for this head: h * hs
                      (def h_offset (arith.muli h hs))

                      ;; Loop over query positions (t)
                      (scf.for c0 T c1
                        (region
                          (block [(: t index)]

                            ;; ========================================
                            ;; Pass 1: Compute attention scores Q·K^T
                            ;; ========================================
                            ;; For each key position t2, compute Q[t] · K[t2] / sqrt(hs)
                            ;; Apply causal mask: -inf for t2 > t

                            (def max_score (scf.for {:result f32} c0 T c1 neg_inf
                              (region
                                (block [(: t2 index) (: curr_max f32)]
                                  ;; Check if t2 <= t (causal mask)
                                  (def is_valid (arith.cmpi {:predicate "ule"} t2 t))

                                  (def score (scf.if {:result f32} is_valid
                                    (region
                                      (block []
                                        ;; Compute dot product Q[t] · K[t2]
                                        (def dot (scf.for {:result f32} c0 hs c1 zero
                                          (region
                                            (block [(: d index) (: acc f32)]
                                              ;; Q index: qkv[b, t, h*hs + d]
                                              (def q_idx (arith.addi h_offset d))
                                              (def q_val (memref.load {:result f32} qkv b t q_idx))

                                              ;; K index: qkv[b, t2, C + h*hs + d]
                                              (def k_base (arith.addi C h_offset))
                                              (def k_idx (arith.addi k_base d))
                                              (def k_val (memref.load {:result f32} qkv b t2 k_idx))

                                              ;; Accumulate
                                              (def prod (arith.mulf q_val k_val))
                                              (def new_acc (arith.addf acc prod))
                                              (scf.yield new_acc)))))

                                        ;; Scale by 1/sqrt(head_size)
                                        (def scaled_dot (arith.mulf dot scale))
                                        (scf.yield scaled_dot)))
                                    (region
                                      (block []
                                        ;; Causal mask: t2 > t, return -inf
                                        (scf.yield neg_inf)))))

                                  ;; Store score in preatt
                                  (memref.store score preatt b h t t2)

                                  ;; Track max for softmax stability
                                  (def is_greater (arith.cmpf {:predicate "ogt"} score curr_max))
                                  (def new_max (arith.select is_greater score curr_max))
                                  (scf.yield new_max)))))

                            ;; ========================================
                            ;; Pass 2: Softmax - exp and sum
                            ;; ========================================
                            (def exp_sum (scf.for {:result f32} c0 T c1 zero
                              (region
                                (block [(: t2 index) (: sum_acc f32)]
                                  (def score (memref.load {:result f32} preatt b h t t2))
                                  (def shifted (arith.subf score max_score))
                                  (def exp_val (math.exp shifted))
                                  ;; Store exp value in att
                                  (memref.store exp_val att b h t t2)
                                  (def new_sum (arith.addf sum_acc exp_val))
                                  (scf.yield new_sum)))))

                            ;; ========================================
                            ;; Pass 3: Softmax - normalize
                            ;; ========================================
                            (scf.for c0 T c1
                              (region
                                (block [(: t2 index)]
                                  (def exp_val (memref.load {:result f32} att b h t t2))
                                  (def normalized (arith.divf exp_val exp_sum))
                                  (memref.store normalized att b h t t2)
                                  (scf.yield))))

                            ;; ========================================
                            ;; Pass 4: Weighted sum of V vectors
                            ;; ========================================
                            ;; out[b, t, h*hs + d] = sum_t2(att[b, h, t, t2] * V[t2, d])
                            (scf.for c0 hs c1
                              (region
                                (block [(: d index)]
                                  (def weighted_sum (scf.for {:result f32} c0 T c1 zero
                                    (region
                                      (block [(: t2 index) (: acc f32)]
                                        ;; Get attention weight
                                        (def att_weight (memref.load {:result f32} att b h t t2))

                                        ;; V index: qkv[b, t2, 2*C + h*hs + d]
                                        (def v_base_c (arith.addi C C))  ; 2*C
                                        (def v_base (arith.addi v_base_c h_offset))
                                        (def v_idx (arith.addi v_base d))
                                        (def v_val (memref.load {:result f32} qkv b t2 v_idx))

                                        ;; Accumulate weighted value
                                        (def weighted (arith.mulf att_weight v_val))
                                        (def new_acc (arith.addf acc weighted))
                                        (scf.yield new_acc)))))

                                  ;; Store output: out[b, t, h*hs + d]
                                  (def out_idx (arith.addi h_offset d))
                                  (memref.store weighted_sum out b t out_idx)
                                  (scf.yield))))

                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; Test function for attention
    (func.func {:sym_name "test_attention"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate tensors
          (def out (memref.alloc {:result memref<1x64x768xf32>}))
          (def preatt (memref.alloc {:result memref<1x12x64x64xf32>}))
          (def att (memref.alloc {:result memref<1x12x64x64xf32>}))
          (def qkv (memref.alloc {:result memref<1x64x2304xf32>}))

          ;; Constants
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def QKV_C (: 2304 index))  ; 3 * 768

          ;; Initialize QKV with simple pattern
          ;; Q, K, V all get similar values for easy validation
          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 QKV_C c1
                        (region
                          (block [(: c index)]
                            ;; Value based on position: small values to avoid overflow
                            (def t_i64 (arith.index_cast {:result i64} t))
                            (def c_i64 (arith.index_cast {:result i64} c))
                            (def sum_i64 (arith.addi t_i64 c_i64))
                            (def val_f (arith.sitofp {:result f32} sum_i64))
                            (def scale (: 0.001 f32))
                            (def val (arith.mulf val_f scale))
                            (memref.store val qkv b t c)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          ;; Run attention
          (func.call "attention_forward" out preatt att qkv)

          ;; Print a few output values to verify
          ;; out[0, 0, 0] and out[0, 0, 1] should be non-zero
          (def val0 (memref.load {:result f32} out c0 c0 c0))
          (def val1 (memref.load {:result f32} out c0 c0 c1))
          (vector.print val0)
          (vector.print val1)

          ;; Print attention weights for first head, first query position
          ;; att[0, 0, 0, 0] should be 1.0 (only position 0 is valid for t=0)
          (def att_00 (memref.load {:result f32} att c0 c0 c0 c0))
          (vector.print att_00)

          ;; Clean up
          (memref.dealloc out)
          (memref.dealloc preatt)
          (memref.dealloc att)
          (memref.dealloc qkv)

          (func.return))))

    ;; Main entry point - calls test_attention
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          (func.call "test_attention")
          (func.return))))))