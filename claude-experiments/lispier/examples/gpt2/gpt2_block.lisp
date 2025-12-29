;; GPT-2 Single Transformer Block
;;
;; This file implements one transformer block using the existing ops.
;; For GPT-2 Small: B=1, T=64, C=768, NH=12, hs=64
;;
;; Block structure:
;; 1. LayerNorm1 -> QKV matmul -> Attention -> Attention proj -> Residual1
;; 2. LayerNorm2 -> FC matmul -> GELU -> FC proj -> Residual2

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)
(require-dialect math)
(require-dialect vector)

;; Compilation pipeline (ROCm backend)
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
    (pass convert-vector-to-llvm)
    (pass convert-func-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass gpu-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; =========================================================================
    ;; Matrix Multiplication: (B,T,C) @ (C,K) + bias -> (B,T,K)
    ;; Generic matmul for transformer projections
    ;; =========================================================================
    (func.func {:sym_name "matmul_bias"
                :function_type (-> [memref<1x64x2304xf32>    ; out (B,T,K) where K=3*C for QKV
                                    memref<1x64x768xf32>     ; inp (B,T,C)
                                    memref<768x2304xf32>     ; weight (C,K)
                                    memref<2304xf32>] [])}   ; bias (K)
      (region
        (block [(: out memref<1x64x2304xf32>)
                (: inp memref<1x64x768xf32>)
                (: weight memref<768x2304xf32>)
                (: bias memref<2304xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def K (: 2304 index))
          (def zero (: 0.0 f32))

          ;; out[b,t,k] = sum_c(inp[b,t,c] * weight[c,k]) + bias[k]
          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 K c1
                        (region
                          (block [(: k index)]
                            ;; Dot product over C dimension
                            (def dot (scf.for {:result f32} c0 C c1 zero
                              (region
                                (block [(: c index) (: acc f32)]
                                  (def inp_val (memref.load {:result f32} inp b t c))
                                  (def w_val (memref.load {:result f32} weight c k))
                                  (def prod (arith.mulf inp_val w_val))
                                  (def new_acc (arith.addf acc prod))
                                  (scf.yield new_acc)))))
                            ;; Add bias
                            (def b_val (memref.load {:result f32} bias k))
                            (def result (arith.addf dot b_val))
                            (memref.store result out b t k)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; =========================================================================
    ;; LayerNorm: Same as ops/layernorm.lisp but inlined here for completeness
    ;; =========================================================================
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<1x64x768xf32>   ; out
                                    memref<1x64xf32>       ; mean (scratch)
                                    memref<1x64xf32>       ; rstd (scratch)
                                    memref<1x64x768xf32>   ; inp
                                    memref<768xf32>        ; weight (gamma)
                                    memref<768xf32>        ; bias (beta)
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
          (def one (: 1.0 f32))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      ;; Mean
                      (def sum_val (scf.for {:result f32} c0 C c1 zero
                        (region
                          (block [(: c index) (: acc f32)]
                            (def x (memref.load {:result f32} inp b t c))
                            (def new_acc (arith.addf acc x))
                            (scf.yield new_acc)))))
                      (def m (arith.divf sum_val C_f32))
                      (memref.store m mean b t)

                      ;; Variance
                      (def var_val (scf.for {:result f32} c0 C c1 zero
                        (region
                          (block [(: c index) (: vacc f32)]
                            (def x (memref.load {:result f32} inp b t c))
                            (def diff (arith.subf x m))
                            (def diff_sq (arith.mulf diff diff))
                            (def new_vacc (arith.addf vacc diff_sq))
                            (scf.yield new_vacc)))))
                      (def variance (arith.divf var_val C_f32))

                      ;; Reciprocal std
                      (def var_eps (arith.addf variance eps))
                      (def std (math.sqrt var_eps))
                      (def rs (arith.divf one std))
                      (memref.store rs rstd b t)

                      ;; Normalize
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

    ;; =========================================================================
    ;; Attention Forward (from ops/attention.lisp)
    ;; =========================================================================
    (func.func {:sym_name "attention_forward"
                :function_type (-> [memref<1x64x768xf32>      ; out
                                    memref<1x12x64x64xf32>    ; preatt
                                    memref<1x12x64x64xf32>    ; att
                                    memref<1x64x2304xf32>] [])} ; qkv
      (region
        (block [(: out memref<1x64x768xf32>)
                (: preatt memref<1x12x64x64xf32>)
                (: att memref<1x12x64x64xf32>)
                (: qkv memref<1x64x2304xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def NH (: 12 index))
          (def hs (: 64 index))
          (def zero (: 0.0 f32))
          (def neg_inf (: -10000.0 f32))
          (def scale (: 0.125 f32))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 NH c1
                  (region
                    (block [(: h index)]
                      (def h_offset (arith.muli h hs))
                      (scf.for c0 T c1
                        (region
                          (block [(: t index)]
                            ;; Pass 1: Q·K^T scores with causal mask
                            (def max_score (scf.for {:result f32} c0 T c1 neg_inf
                              (region
                                (block [(: t2 index) (: curr_max f32)]
                                  (def is_valid (arith.cmpi {:predicate "ule"} t2 t))
                                  (def score (scf.if {:result f32} is_valid
                                    (region
                                      (block []
                                        (def dot (scf.for {:result f32} c0 hs c1 zero
                                          (region
                                            (block [(: d index) (: acc f32)]
                                              (def q_idx (arith.addi h_offset d))
                                              (def q_val (memref.load {:result f32} qkv b t q_idx))
                                              (def k_base (arith.addi C h_offset))
                                              (def k_idx (arith.addi k_base d))
                                              (def k_val (memref.load {:result f32} qkv b t2 k_idx))
                                              (def prod (arith.mulf q_val k_val))
                                              (def new_acc (arith.addf acc prod))
                                              (scf.yield new_acc)))))
                                        (def scaled_dot (arith.mulf dot scale))
                                        (scf.yield scaled_dot)))
                                    (region
                                      (block []
                                        (scf.yield neg_inf)))))
                                  (memref.store score preatt b h t t2)
                                  (def is_greater (arith.cmpf {:predicate "ogt"} score curr_max))
                                  (def new_max (arith.select is_greater score curr_max))
                                  (scf.yield new_max)))))

                            ;; Pass 2: Softmax exp and sum
                            (def exp_sum (scf.for {:result f32} c0 T c1 zero
                              (region
                                (block [(: t2 index) (: sum_acc f32)]
                                  (def score (memref.load {:result f32} preatt b h t t2))
                                  (def shifted (arith.subf score max_score))
                                  (def exp_val (math.exp shifted))
                                  (memref.store exp_val att b h t t2)
                                  (def new_sum (arith.addf sum_acc exp_val))
                                  (scf.yield new_sum)))))

                            ;; Pass 3: Normalize
                            (scf.for c0 T c1
                              (region
                                (block [(: t2 index)]
                                  (def exp_val (memref.load {:result f32} att b h t t2))
                                  (def normalized (arith.divf exp_val exp_sum))
                                  (memref.store normalized att b h t t2)
                                  (scf.yield))))

                            ;; Pass 4: Weighted sum of V
                            (scf.for c0 hs c1
                              (region
                                (block [(: d index)]
                                  (def weighted_sum (scf.for {:result f32} c0 T c1 zero
                                    (region
                                      (block [(: t2 index) (: acc f32)]
                                        (def att_weight (memref.load {:result f32} att b h t t2))
                                        (def v_base_c (arith.addi C C))
                                        (def v_base (arith.addi v_base_c h_offset))
                                        (def v_idx (arith.addi v_base d))
                                        (def v_val (memref.load {:result f32} qkv b t2 v_idx))
                                        (def weighted (arith.mulf att_weight v_val))
                                        (def new_acc (arith.addf acc weighted))
                                        (scf.yield new_acc)))))
                                  (def out_idx (arith.addi h_offset d))
                                  (memref.store weighted_sum out b t out_idx)
                                  (scf.yield))))
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Attention Output Projection: (B,T,C) @ (C,C) + bias -> (B,T,C)
    ;; =========================================================================
    (func.func {:sym_name "attn_proj_forward"
                :function_type (-> [memref<1x64x768xf32>    ; out
                                    memref<1x64x768xf32>    ; inp
                                    memref<768x768xf32>     ; weight
                                    memref<768xf32>] [])}   ; bias
      (region
        (block [(: out memref<1x64x768xf32>)
                (: inp memref<1x64x768xf32>)
                (: weight memref<768x768xf32>)
                (: bias memref<768xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def zero (: 0.0 f32))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 C c1
                        (region
                          (block [(: k index)]
                            (def dot (scf.for {:result f32} c0 C c1 zero
                              (region
                                (block [(: c index) (: acc f32)]
                                  (def inp_val (memref.load {:result f32} inp b t c))
                                  (def w_val (memref.load {:result f32} weight c k))
                                  (def prod (arith.mulf inp_val w_val))
                                  (def new_acc (arith.addf acc prod))
                                  (scf.yield new_acc)))))
                            (def b_val (memref.load {:result f32} bias k))
                            (def result (arith.addf dot b_val))
                            (memref.store result out b t k)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Residual Add: out = inp1 + inp2
    ;; =========================================================================
    (func.func {:sym_name "residual_forward"
                :function_type (-> [memref<1x64x768xf32>
                                    memref<1x64x768xf32>
                                    memref<1x64x768xf32>] [])}
      (region
        (block [(: out memref<1x64x768xf32>)
                (: inp1 memref<1x64x768xf32>)
                (: inp2 memref<1x64x768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 C c1
                        (region
                          (block [(: c index)]
                            (def v1 (memref.load {:result f32} inp1 b t c))
                            (def v2 (memref.load {:result f32} inp2 b t c))
                            (def sum (arith.addf v1 v2))
                            (memref.store sum out b t c)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; MLP FC1: (B,T,C) @ (C,4C) + bias -> (B,T,4C)
    ;; =========================================================================
    (func.func {:sym_name "mlp_fc_forward"
                :function_type (-> [memref<1x64x3072xf32>   ; out (B,T,4C)
                                    memref<1x64x768xf32>    ; inp (B,T,C)
                                    memref<768x3072xf32>    ; weight (C,4C)
                                    memref<3072xf32>] [])}  ; bias (4C)
      (region
        (block [(: out memref<1x64x3072xf32>)
                (: inp memref<1x64x768xf32>)
                (: weight memref<768x3072xf32>)
                (: bias memref<3072xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def C4 (: 3072 index))
          (def zero (: 0.0 f32))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 C4 c1
                        (region
                          (block [(: k index)]
                            (def dot (scf.for {:result f32} c0 C c1 zero
                              (region
                                (block [(: c index) (: acc f32)]
                                  (def inp_val (memref.load {:result f32} inp b t c))
                                  (def w_val (memref.load {:result f32} weight c k))
                                  (def prod (arith.mulf inp_val w_val))
                                  (def new_acc (arith.addf acc prod))
                                  (scf.yield new_acc)))))
                            (def b_val (memref.load {:result f32} bias k))
                            (def result (arith.addf dot b_val))
                            (memref.store result out b t k)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; GELU Activation (tanh approximation)
    ;; =========================================================================
    (func.func {:sym_name "gelu_forward"
                :function_type (-> [memref<1x64x3072xf32>
                                    memref<1x64x3072xf32>] [])}
      (region
        (block [(: out memref<1x64x3072xf32>)
                (: inp memref<1x64x3072xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C4 (: 3072 index))

          ;; Constants for GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))
          (def sqrt_2_over_pi (: 0.7978845608 f32))
          (def coeff (: 0.044715 f32))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 C4 c1
                        (region
                          (block [(: c index)]
                            (def x (memref.load {:result f32} inp b t c))
                            (def x2 (arith.mulf x x))
                            (def x3 (arith.mulf x2 x))
                            (def inner1 (arith.mulf coeff x3))
                            (def inner2 (arith.addf x inner1))
                            (def inner3 (arith.mulf sqrt_2_over_pi inner2))
                            (def tanh_val (math.tanh inner3))
                            (def one_plus_tanh (arith.addf one tanh_val))
                            (def half_x (arith.mulf half x))
                            (def result (arith.mulf half_x one_plus_tanh))
                            (memref.store result out b t c)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; MLP Projection: (B,T,4C) @ (4C,C) + bias -> (B,T,C)
    ;; =========================================================================
    (func.func {:sym_name "mlp_proj_forward"
                :function_type (-> [memref<1x64x768xf32>    ; out (B,T,C)
                                    memref<1x64x3072xf32>   ; inp (B,T,4C)
                                    memref<3072x768xf32>    ; weight (4C,C)
                                    memref<768xf32>] [])}   ; bias (C)
      (region
        (block [(: out memref<1x64x768xf32>)
                (: inp memref<1x64x3072xf32>)
                (: weight memref<3072x768xf32>)
                (: bias memref<768xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def B (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def C4 (: 3072 index))
          (def zero (: 0.0 f32))

          (scf.for c0 B c1
            (region
              (block [(: b index)]
                (scf.for c0 T c1
                  (region
                    (block [(: t index)]
                      (scf.for c0 C c1
                        (region
                          (block [(: k index)]
                            (def dot (scf.for {:result f32} c0 C4 c1 zero
                              (region
                                (block [(: c index) (: acc f32)]
                                  (def inp_val (memref.load {:result f32} inp b t c))
                                  (def w_val (memref.load {:result f32} weight c k))
                                  (def prod (arith.mulf inp_val w_val))
                                  (def new_acc (arith.addf acc prod))
                                  (scf.yield new_acc)))))
                            (def b_val (memref.load {:result f32} bias k))
                            (def result (arith.addf dot b_val))
                            (memref.store result out b t k)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Single Transformer Block
    ;; =========================================================================
    (func.func {:sym_name "transformer_block"
                :function_type (-> [memref<1x64x768xf32>    ; residual_out (B,T,C)
                                    memref<1x64x768xf32>    ; residual_in (B,T,C)
                                    ;; Layer parameters
                                    memref<768xf32>         ; ln1_weight
                                    memref<768xf32>         ; ln1_bias
                                    memref<768x2304xf32>    ; qkv_weight
                                    memref<2304xf32>        ; qkv_bias
                                    memref<768x768xf32>     ; attn_proj_weight
                                    memref<768xf32>         ; attn_proj_bias
                                    memref<768xf32>         ; ln2_weight
                                    memref<768xf32>         ; ln2_bias
                                    memref<768x3072xf32>    ; fc_weight
                                    memref<3072xf32>        ; fc_bias
                                    memref<3072x768xf32>    ; fc_proj_weight
                                    memref<768xf32>         ; fc_proj_bias
                                    ;; Scratch buffers
                                    memref<1x64xf32>        ; ln_mean
                                    memref<1x64xf32>        ; ln_rstd
                                    memref<1x64x768xf32>    ; ln1_out
                                    memref<1x64x2304xf32>   ; qkv
                                    memref<1x12x64x64xf32>  ; preatt
                                    memref<1x12x64x64xf32>  ; att
                                    memref<1x64x768xf32>    ; attn_out
                                    memref<1x64x768xf32>    ; attn_proj
                                    memref<1x64x768xf32>    ; residual2
                                    memref<1x64x768xf32>    ; ln2_out
                                    memref<1x64x3072xf32>   ; fch
                                    memref<1x64x3072xf32>   ; fch_gelu
                                    memref<1x64x768xf32>] [])}  ; fc_proj
      (region
        (block [(: residual_out memref<1x64x768xf32>)
                (: residual_in memref<1x64x768xf32>)
                (: ln1_weight memref<768xf32>)
                (: ln1_bias memref<768xf32>)
                (: qkv_weight memref<768x2304xf32>)
                (: qkv_bias memref<2304xf32>)
                (: attn_proj_weight memref<768x768xf32>)
                (: attn_proj_bias memref<768xf32>)
                (: ln2_weight memref<768xf32>)
                (: ln2_bias memref<768xf32>)
                (: fc_weight memref<768x3072xf32>)
                (: fc_bias memref<3072xf32>)
                (: fc_proj_weight memref<3072x768xf32>)
                (: fc_proj_bias memref<768xf32>)
                (: ln_mean memref<1x64xf32>)
                (: ln_rstd memref<1x64xf32>)
                (: ln1_out memref<1x64x768xf32>)
                (: qkv memref<1x64x2304xf32>)
                (: preatt memref<1x12x64x64xf32>)
                (: att memref<1x12x64x64xf32>)
                (: attn_out memref<1x64x768xf32>)
                (: attn_proj memref<1x64x768xf32>)
                (: residual2 memref<1x64x768xf32>)
                (: ln2_out memref<1x64x768xf32>)
                (: fch memref<1x64x3072xf32>)
                (: fch_gelu memref<1x64x3072xf32>)
                (: fc_proj memref<1x64x768xf32>)]

          (def eps (: 0.00001 f32))

          ;; 1. LayerNorm1
          (func.call "layernorm_forward" ln1_out ln_mean ln_rstd residual_in ln1_weight ln1_bias eps)

          ;; 2. QKV Projection
          (func.call "matmul_bias" qkv ln1_out qkv_weight qkv_bias)

          ;; 3. Attention
          (func.call "attention_forward" attn_out preatt att qkv)

          ;; 4. Attention Output Projection
          (func.call "attn_proj_forward" attn_proj attn_out attn_proj_weight attn_proj_bias)

          ;; 5. Residual 1
          (func.call "residual_forward" residual2 residual_in attn_proj)

          ;; 6. LayerNorm2
          (func.call "layernorm_forward" ln2_out ln_mean ln_rstd residual2 ln2_weight ln2_bias eps)

          ;; 7. MLP FC
          (func.call "mlp_fc_forward" fch ln2_out fc_weight fc_bias)

          ;; 8. GELU
          (func.call "gelu_forward" fch_gelu fch)

          ;; 9. MLP Projection
          (func.call "mlp_proj_forward" fc_proj fch_gelu fc_proj_weight fc_proj_bias)

          ;; 10. Residual 2
          (func.call "residual_forward" residual_out residual2 fc_proj)

          (func.return))))

    ;; =========================================================================
    ;; Test: Run one transformer block with random weights
    ;; =========================================================================
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate all tensors
          (def residual_in (memref.alloc {:result memref<1x64x768xf32>}))
          (def residual_out (memref.alloc {:result memref<1x64x768xf32>}))

          ;; Layer params (initialized to simple values)
          (def ln1_weight (memref.alloc {:result memref<768xf32>}))
          (def ln1_bias (memref.alloc {:result memref<768xf32>}))
          (def qkv_weight (memref.alloc {:result memref<768x2304xf32>}))
          (def qkv_bias (memref.alloc {:result memref<2304xf32>}))
          (def attn_proj_weight (memref.alloc {:result memref<768x768xf32>}))
          (def attn_proj_bias (memref.alloc {:result memref<768xf32>}))
          (def ln2_weight (memref.alloc {:result memref<768xf32>}))
          (def ln2_bias (memref.alloc {:result memref<768xf32>}))
          (def fc_weight (memref.alloc {:result memref<768x3072xf32>}))
          (def fc_bias (memref.alloc {:result memref<3072xf32>}))
          (def fc_proj_weight (memref.alloc {:result memref<3072x768xf32>}))
          (def fc_proj_bias (memref.alloc {:result memref<768xf32>}))

          ;; Scratch buffers
          (def ln_mean (memref.alloc {:result memref<1x64xf32>}))
          (def ln_rstd (memref.alloc {:result memref<1x64xf32>}))
          (def ln1_out (memref.alloc {:result memref<1x64x768xf32>}))
          (def qkv (memref.alloc {:result memref<1x64x2304xf32>}))
          (def preatt (memref.alloc {:result memref<1x12x64x64xf32>}))
          (def att (memref.alloc {:result memref<1x12x64x64xf32>}))
          (def attn_out (memref.alloc {:result memref<1x64x768xf32>}))
          (def attn_proj (memref.alloc {:result memref<1x64x768xf32>}))
          (def residual2 (memref.alloc {:result memref<1x64x768xf32>}))
          (def ln2_out (memref.alloc {:result memref<1x64x768xf32>}))
          (def fch (memref.alloc {:result memref<1x64x3072xf32>}))
          (def fch_gelu (memref.alloc {:result memref<1x64x3072xf32>}))
          (def fc_proj (memref.alloc {:result memref<1x64x768xf32>}))

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def one (: 1.0 f32))
          (def zero (: 0.0 f32))
          (def small (: 0.01 f32))

          ;; Initialize LayerNorm weights to 1, biases to 0
          (def C (: 768 index))
          (scf.for c0 C c1
            (region
              (block [(: c index)]
                (memref.store one ln1_weight c)
                (memref.store zero ln1_bias c)
                (memref.store one ln2_weight c)
                (memref.store zero ln2_bias c)
                (memref.store zero attn_proj_bias c)
                (memref.store zero fc_proj_bias c)
                (scf.yield))))

          ;; Initialize QKV bias to 0
          (def K3C (: 2304 index))
          (scf.for c0 K3C c1
            (region
              (block [(: k index)]
                (memref.store zero qkv_bias k)
                (scf.yield))))

          ;; Initialize FC bias to 0
          (def C4 (: 3072 index))
          (scf.for c0 C4 c1
            (region
              (block [(: k index)]
                (memref.store zero fc_bias k)
                (scf.yield))))

          ;; Initialize weights with small values (identity-like for projections)
          (scf.for c0 C c1
            (region
              (block [(: i index)]
                (scf.for c0 C c1
                  (region
                    (block [(: j index)]
                      (def is_diag (arith.cmpi {:predicate "eq"} i j))
                      (def w_val (arith.select is_diag small zero))
                      (memref.store w_val attn_proj_weight i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize QKV weight (small identity-like)
          (scf.for c0 C c1
            (region
              (block [(: i index)]
                (scf.for c0 K3C c1
                  (region
                    (block [(: j index)]
                      (memref.store small qkv_weight i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize FC weights (small values)
          (scf.for c0 C c1
            (region
              (block [(: i index)]
                (scf.for c0 C4 c1
                  (region
                    (block [(: j index)]
                      (memref.store small fc_weight i j)
                      (scf.yield))))
                (scf.yield))))

          (scf.for c0 C4 c1
            (region
              (block [(: i index)]
                (scf.for c0 C c1
                  (region
                    (block [(: j index)]
                      (memref.store small fc_proj_weight i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize input with simple pattern
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
                            (def t_i64 (arith.index_cast {:result i64} t))
                            (def c_i64 (arith.index_cast {:result i64} c))
                            (def sum_i64 (arith.addi t_i64 c_i64))
                            (def val_f (arith.sitofp {:result f32} sum_i64))
                            (def val (arith.mulf val_f small))
                            (memref.store val residual_in b t c)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          ;; Run transformer block
          (func.call "transformer_block"
            residual_out residual_in
            ln1_weight ln1_bias qkv_weight qkv_bias
            attn_proj_weight attn_proj_bias
            ln2_weight ln2_bias fc_weight fc_bias
            fc_proj_weight fc_proj_bias
            ln_mean ln_rstd ln1_out qkv preatt att
            attn_out attn_proj residual2 ln2_out fch fch_gelu fc_proj)

          ;; Print some output values
          (def out0 (memref.load {:result f32} residual_out c0 c0 c0))
          (def out1 (memref.load {:result f32} residual_out c0 c0 c1))
          (def out2 (memref.load {:result f32} residual_out c0 c1 c0))
          (vector.print out0)
          (vector.print out1)
          (vector.print out2)

          ;; Cleanup (skip for brevity in test)
          (func.return))))))