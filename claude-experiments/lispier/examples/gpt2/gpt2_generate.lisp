;; GPT-2 Text Generation
;; Complete inference pipeline: tokens → forward → logits → sample → decode
;;
;; This file implements everything llm.c does in C:
;; - Token embedding lookup
;; - 12-layer transformer forward pass
;; - Logits projection (hidden @ wte^T)
;; - Softmax over vocabulary
;; - Argmax / multinomial sampling
;; - Tokenizer decoder

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect llvm)
(require-dialect scf)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; Math functions implemented manually (avoiding math dialect lowering issues)

;; Model constants
;; T=64 (sequence length), C=768 (channels), V=50257 (vocab), L=12 (layers)

(module
  (do

    ;; Hardcoded paths - set LLM_C_PATH env var in shell before running
    ;; or modify these paths directly
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<55 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/Users/jimmyhmiller/Documents/Code/llm.c/gpt2_124M.bin\0" :result !llvm.array<55 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "tokenizer_path"
                       :linkage 0
                       :global_type !llvm.array<60 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/Users/jimmyhmiller/Documents/Code/llm.c/gpt2_tokenizer.bin\0" :result !llvm.array<60 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "read_mode"
                       :linkage 0
                       :global_type !llvm.array<3 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "rb\0" :result !llvm.array<3 x i8>}))
          (llvm.return s))))

    ;; Global params pointer
    (llvm.mlir.global {:sym_name "g_params"
                       :linkage 10
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null))))

    ;; Tokenizer: array of 50257 string pointers
    (llvm.mlir.global {:sym_name "token_table"
                       :linkage 10
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null))))

    (llvm.mlir.global {:sym_name "eot_token"
                       :linkage 10
                       :global_type i32
                       :constant false}
      (region
        (block []
          (def v (llvm.mlir.constant {:value 50256 :result i32}))
          (llvm.return v))))

    ;; =========================================================================
    ;; Math helper: sqrt via Newton-Raphson (10 iterations)
    ;; =========================================================================
    (func.func {:sym_name "my_sqrtf"
                :function_type (-> [f32] [f32])}
      (region
        (block [(: x f32)]
          (def half (: 0.5 f32))
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c10 (: 10 index))
          ;; Initial guess: x * 0.5
          (def init_guess (arith.mulf x half))
          ;; Newton-Raphson: guess = (guess + x/guess) * 0.5
          (def result (scf.for {:result f32} c0 c10 c1 init_guess
            (region
              (block [(: i index) (: guess f32)]
                (def new_guess (arith.mulf (arith.addf guess (arith.divf x guess)) half))
                (scf.yield new_guess)))))
          (func.return result))))

    ;; =========================================================================
    ;; Math helper: exp via range reduction
    ;; Strategy: exp(x) = exp(x/16)^16, Taylor series is accurate for small x
    ;; =========================================================================
    (func.func {:sym_name "my_expf_small"
                :function_type (-> [f32] [f32])}
      (region
        (block [(: x f32)]
          ;; Taylor series for small x: 1 + x + x²/2! + ... + x⁸/8!
          (def one (: 1.0 f32))
          (def term1 x)
          (def term2 (arith.divf (arith.mulf term1 x) (: 2.0 f32)))
          (def term3 (arith.divf (arith.mulf term2 x) (: 3.0 f32)))
          (def term4 (arith.divf (arith.mulf term3 x) (: 4.0 f32)))
          (def term5 (arith.divf (arith.mulf term4 x) (: 5.0 f32)))
          (def term6 (arith.divf (arith.mulf term5 x) (: 6.0 f32)))
          (def term7 (arith.divf (arith.mulf term6 x) (: 7.0 f32)))
          (def term8 (arith.divf (arith.mulf term7 x) (: 8.0 f32)))
          (def sum1 (arith.addf one term1))
          (def sum2 (arith.addf sum1 term2))
          (def sum3 (arith.addf sum2 term3))
          (def sum4 (arith.addf sum3 term4))
          (def sum5 (arith.addf sum4 term5))
          (def sum6 (arith.addf sum5 term6))
          (def sum7 (arith.addf sum6 term7))
          (def result (arith.addf sum7 term8))
          (func.return result))))

    (func.func {:sym_name "my_expf"
                :function_type (-> [f32] [f32])}
      (region
        (block [(: x f32)]
          ;; Clamp to prevent overflow/underflow
          (def x_clamped (arith.minimumf (arith.maximumf x (: -80.0 f32)) (: 80.0 f32)))
          ;; Divide by 16 to bring into Taylor series convergence range
          (def x_small (arith.divf x_clamped (: 16.0 f32)))
          ;; Compute exp(x/16)
          (def exp_small (func.call {:result f32} "my_expf_small" x_small))
          ;; Square 4 times: exp(x) = exp(x/16)^16
          (def exp2 (arith.mulf exp_small exp_small))
          (def exp4 (arith.mulf exp2 exp2))
          (def exp8 (arith.mulf exp4 exp4))
          (def exp16 (arith.mulf exp8 exp8))
          (func.return exp16))))

    ;; =========================================================================
    ;; Math helper: tanh via exp
    ;; For |x| > 10, tanh ≈ ±1 (clamp to avoid overflow issues)
    ;; tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    ;; =========================================================================
    (func.func {:sym_name "my_tanhf"
                :function_type (-> [f32] [f32])}
      (region
        (block [(: x f32)]
          (def one (: 1.0 f32))
          (def two (: 2.0 f32))
          (def threshold (: 10.0 f32))
          ;; Clamp x to [-10, 10] to avoid exp overflow (tanh saturates to ±1 anyway)
          (def x_clamped (arith.minimumf (arith.maximumf x (arith.negf threshold)) threshold))
          (def two_x (arith.mulf two x_clamped))
          (def exp_2x (func.call {:result f32} "my_expf" two_x))
          ;; tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
          (def result (arith.divf (arith.subf exp_2x one) (arith.addf exp_2x one)))
          (func.return result))))

    ;; =========================================================================
    ;; LayerNorm forward
    ;; =========================================================================
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<64x768xf32>   ; out
                                    memref<64x768xf32>   ; inp
                                    memref<768xf32>      ; weight (gamma)
                                    memref<768xf32>] [])} ; bias (beta)
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768xf32>)
                (: bias memref<768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c768_f32 (: 768.0 f32))
          (def zero (: 0.0 f32))
          (def eps (: 1e-5 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                ;; Mean
                (def sum_val (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def v (memref.load inp t c))
                      (scf.yield (arith.addf acc v))))))
                (def mean (arith.divf sum_val c768_f32))

                ;; Variance
                (def var_sum (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def v (memref.load inp t c))
                      (def diff (arith.subf v mean))
                      (def sq (arith.mulf diff diff))
                      (scf.yield (arith.addf acc sq))))))
                (def var (arith.divf var_sum c768_f32))
                (def std (func.call {:result f32} "my_sqrtf" (arith.addf var eps)))

                ;; Normalize
                (scf.for c0 c768 c1
                  (region
                    (block [(: c index)]
                      (def v (memref.load inp t c))
                      (def norm (arith.divf (arith.subf v mean) std))
                      (def g (memref.load weight c))
                      (def b (memref.load bias c))
                      (memref.store (arith.addf (arith.mulf norm g) b) out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Matmul QKV: (64,768) @ (768,2304) + bias -> (64,2304)
    ;; =========================================================================
    (func.func {:sym_name "matmul_qkv"
                :function_type (-> [memref<64x2304xf32>
                                    memref<64x768xf32>
                                    memref<768x2304xf32>
                                    memref<2304xf32>] [])}
      (region
        (block [(: out memref<64x2304xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768x2304xf32>)
                (: bias memref<2304xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c2304 (: 2304 index))
          (def zero (: 0.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c2304 c1
                  (region
                    (block [(: k index)]
                      (def sum (scf.for {:result f32} c0 c768 c1 zero
                        (region
                          (block [(: c index) (: acc f32)]
                            (def inp_val (memref.load inp t c))
                            (def w_val (memref.load weight c k))
                            (scf.yield (arith.addf acc (arith.mulf inp_val w_val)))))))
                      (def b (memref.load bias k))
                      (memref.store (arith.addf sum b) out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Attention forward (multi-head causal self-attention)
    ;; =========================================================================
    (func.func {:sym_name "attention_forward"
                :function_type (-> [memref<64x768xf32>      ; out (T,C)
                                    memref<64x2304xf32>] [])} ; qkv (T,3C)
      (region
        (block [(: out memref<64x768xf32>)
                (: qkv memref<64x2304xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c12 (: 12 index))
          (def hs (: 64 index))  ; head_size = 768/12 = 64
          (def zero (: 0.0 f32))
          (def neg_inf (: -1e9 f32))
          (def scale (: 0.125 f32))  ; 1/sqrt(64)

          ;; For each head
          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                (def h_offset (arith.muli h hs))
                (def k_offset (arith.addi h_offset c768))
                (def v_offset (arith.addi k_offset c768))

                ;; For each query position
                (scf.for c0 c64 c1
                  (region
                    (block [(: t index)]
                      ;; Compute attention scores for this position
                      ;; First pass: compute scores and find max (for softmax stability)
                      (def max_score (scf.for {:result f32} c0 (arith.addi t c1) c1 neg_inf
                        (region
                          (block [(: t2 index) (: max_acc f32)]
                            ;; Q[t] dot K[t2]
                            (def dot (scf.for {:result f32} c0 hs c1 zero
                              (region
                                (block [(: i index) (: dot_acc f32)]
                                  (def q_idx (arith.addi h_offset i))
                                  (def k_idx (arith.addi k_offset i))
                                  (def q_val (memref.load qkv t q_idx))
                                  (def k_val (memref.load qkv t2 k_idx))
                                  (scf.yield (arith.addf dot_acc (arith.mulf q_val k_val)))))))
                            (def score (arith.mulf dot scale))
                            (def new_max (arith.maximumf max_acc score))
                            (scf.yield new_max)))))

                      ;; Second pass: exp and sum
                      (def exp_sum (scf.for {:result f32} c0 (arith.addi t c1) c1 zero
                        (region
                          (block [(: t2 index) (: sum_acc f32)]
                            (def dot (scf.for {:result f32} c0 hs c1 zero
                              (region
                                (block [(: i index) (: dot_acc f32)]
                                  (def q_idx (arith.addi h_offset i))
                                  (def k_idx (arith.addi k_offset i))
                                  (def q_val (memref.load qkv t q_idx))
                                  (def k_val (memref.load qkv t2 k_idx))
                                  (scf.yield (arith.addf dot_acc (arith.mulf q_val k_val)))))))
                            (def score (arith.mulf dot scale))
                            (def exp_score (func.call {:result f32} "my_expf" (arith.subf score max_score)))
                            (scf.yield (arith.addf sum_acc exp_score))))))

                      ;; Third pass: weighted sum of V
                      (scf.for c0 hs c1
                        (region
                          (block [(: i index)]
                            (def weighted_v (scf.for {:result f32} c0 (arith.addi t c1) c1 zero
                              (region
                                (block [(: t2 index) (: v_acc f32)]
                                  (def dot (scf.for {:result f32} c0 hs c1 zero
                                    (region
                                      (block [(: j index) (: dot_acc f32)]
                                        (def q_idx (arith.addi h_offset j))
                                        (def k_idx (arith.addi k_offset j))
                                        (def q_val (memref.load qkv t q_idx))
                                        (def k_val (memref.load qkv t2 k_idx))
                                        (scf.yield (arith.addf dot_acc (arith.mulf q_val k_val)))))))
                                  (def score (arith.mulf dot scale))
                                  (def attn_weight (arith.divf (func.call {:result f32} "my_expf" (arith.subf score max_score)) exp_sum))
                                  (def v_idx (arith.addi v_offset i))
                                  (def v_val (memref.load qkv t2 v_idx))
                                  (scf.yield (arith.addf v_acc (arith.mulf attn_weight v_val)))))))
                            (def out_idx (arith.addi h_offset i))
                            (memref.store weighted_v out t out_idx)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Matmul Attention Proj: (64,768) @ (768,768) + bias -> (64,768)
    ;; =========================================================================
    (func.func {:sym_name "matmul_attn_proj"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>
                                    memref<768x768xf32>
                                    memref<768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768x768xf32>)
                (: bias memref<768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def zero (: 0.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: k index)]
                      (def sum (scf.for {:result f32} c0 c768 c1 zero
                        (region
                          (block [(: c index) (: acc f32)]
                            (def inp_val (memref.load inp t c))
                            (def w_val (memref.load weight c k))
                            (scf.yield (arith.addf acc (arith.mulf inp_val w_val)))))))
                      (def b (memref.load bias k))
                      (memref.store (arith.addf sum b) out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Residual add
    ;; =========================================================================
    (func.func {:sym_name "residual_add"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>
                                    memref<64x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: a memref<64x768xf32>)
                (: b memref<64x768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: c index)]
                      (def va (memref.load a t c))
                      (def vb (memref.load b t c))
                      (memref.store (arith.addf va vb) out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Matmul FC: (64,768) @ (768,3072) + bias -> (64,3072)
    ;; =========================================================================
    (func.func {:sym_name "matmul_fc"
                :function_type (-> [memref<64x3072xf32>
                                    memref<64x768xf32>
                                    memref<768x3072xf32>
                                    memref<3072xf32>] [])}
      (region
        (block [(: out memref<64x3072xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768x3072xf32>)
                (: bias memref<3072xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c3072 (: 3072 index))
          (def zero (: 0.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c3072 c1
                  (region
                    (block [(: k index)]
                      (def sum (scf.for {:result f32} c0 c768 c1 zero
                        (region
                          (block [(: c index) (: acc f32)]
                            (def inp_val (memref.load inp t c))
                            (def w_val (memref.load weight c k))
                            (scf.yield (arith.addf acc (arith.mulf inp_val w_val)))))))
                      (def b (memref.load bias k))
                      (memref.store (arith.addf sum b) out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; GELU activation
    ;; =========================================================================
    (func.func {:sym_name "gelu_forward"
                :function_type (-> [memref<64x3072xf32> memref<64x3072xf32>] [])}
      (region
        (block [(: out memref<64x3072xf32>) (: inp memref<64x3072xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c3072 (: 3072 index))
          (def sqrt_2_over_pi (: 0.7978845608 f32))
          (def coeff (: 0.044715 f32))
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c3072 c1
                  (region
                    (block [(: c index)]
                      (def x (memref.load inp t c))
                      (def x3 (arith.mulf x (arith.mulf x x)))
                      (def inner (arith.mulf sqrt_2_over_pi (arith.addf x (arith.mulf coeff x3))))
                      (def tanh_val (func.call {:result f32} "my_tanhf" inner))
                      (def result (arith.mulf (arith.mulf half x) (arith.addf one tanh_val)))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Matmul FC Proj: (64,3072) @ (3072,768) + bias -> (64,768)
    ;; =========================================================================
    (func.func {:sym_name "matmul_fc_proj"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x3072xf32>
                                    memref<3072x768xf32>
                                    memref<768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x3072xf32>)
                (: weight memref<3072x768xf32>)
                (: bias memref<768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c3072 (: 3072 index))
          (def zero (: 0.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: k index)]
                      (def sum (scf.for {:result f32} c0 c3072 c1 zero
                        (region
                          (block [(: c index) (: acc f32)]
                            (def inp_val (memref.load inp t c))
                            (def w_val (memref.load weight c k))
                            (scf.yield (arith.addf acc (arith.mulf inp_val w_val)))))))
                      (def b (memref.load bias k))
                      (memref.store (arith.addf sum b) out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Embedding lookup: token_ids -> embeddings (wte + wpe)
    ;; =========================================================================
    (func.func {:sym_name "embedding_lookup"
                :function_type (-> [memref<64x768xf32>      ; out
                                    !llvm.ptr               ; wte (50257*768 floats)
                                    !llvm.ptr               ; wpe (1024*768 floats)
                                    memref<64xi32>] [])}    ; token_ids
      (region
        (block [(: out memref<64x768xf32>)
                (: wte !llvm.ptr)
                (: wpe !llvm.ptr)
                (: token_ids memref<64xi32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c768_i64 (: 768 i64))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (def tok_id (memref.load token_ids t))
                (def tok_id_i64 (arith.extsi {:result i64} tok_id))
                (def t_i64 (arith.index_cast {:result i64} t))

                ;; wte offset = tok_id * 768
                (def wte_offset (arith.muli tok_id_i64 c768_i64))
                ;; wpe offset = t * 768
                (def wpe_offset (arith.muli t_i64 c768_i64))

                (scf.for c0 c768 c1
                  (region
                    (block [(: c index)]
                      (def c_i64 (arith.index_cast {:result i64} c))
                      ;; wte[tok_id, c]
                      (def wte_ptr (ptr-at f32 wte (arith.addi wte_offset c_i64)))
                      (def wte_val (llvm.load {:result f32} wte_ptr))
                      ;; wpe[t, c]
                      (def wpe_ptr (ptr-at f32 wpe (arith.addi wpe_offset c_i64)))
                      (def wpe_val (llvm.load {:result f32} wpe_ptr))
                      ;; out[t, c] = wte + wpe
                      (memref.store (arith.addf wte_val wpe_val) out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Logits projection: hidden (T,768) @ wte^T (768,50257) -> logits (T,50257)
    ;; Note: We compute for just one position (the last one) for generation
    ;; =========================================================================
    (func.func {:sym_name "logits_last_position"
                :function_type (-> [memref<50257xf32>       ; logits (for one position)
                                    memref<64x768xf32>      ; hidden
                                    !llvm.ptr               ; wte
                                    index] [])}             ; position
      (region
        (block [(: logits memref<50257xf32>)
                (: hidden memref<64x768xf32>)
                (: wte !llvm.ptr)
                (: pos index)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c768 (: 768 index))
          (def c50257 (: 50257 index))
          (def c768_i64 (: 768 i64))
          (def zero (: 0.0 f32))

          ;; logits[v] = sum_c hidden[pos, c] * wte[v, c]
          (scf.for c0 c50257 c1
            (region
              (block [(: v index)]
                (def v_i64 (arith.index_cast {:result i64} v))
                (def wte_row_offset (arith.muli v_i64 c768_i64))

                (def sum (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def c_i64 (arith.index_cast {:result i64} c))
                      (def h_val (memref.load hidden pos c))
                      (def wte_ptr (ptr-at f32 wte (arith.addi wte_row_offset c_i64)))
                      (def wte_val (llvm.load {:result f32} wte_ptr))
                      (scf.yield (arith.addf acc (arith.mulf h_val wte_val)))))))
                (memref.store sum logits v)
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Argmax over logits (simple version - find max then find its index)
    ;; =========================================================================
    (func.func {:sym_name "argmax"
                :function_type (-> [memref<50257xf32>] [i32])}
      (region
        (block [(: logits memref<50257xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c50257 (: 50257 index))
          (def neg_inf (: -1e38 f32))

          ;; First pass: find max value
          (def max_val (scf.for {:result f32} c0 c50257 c1 neg_inf
            (region
              (block [(: v index) (: best f32)]
                (def val (memref.load logits v))
                (def new_best (arith.maximumf best val))
                (scf.yield new_best)))))

          ;; Second pass: find index of max (first occurrence)
          ;; Use -1 as sentinel to indicate "not found yet"
          (def best_idx (scf.for {:result i32} c0 c50257 c1 (: -1 i32)
            (region
              (block [(: v index) (: found_idx i32)]
                (def val (memref.load logits v))
                (def is_max (arith.cmpf {:predicate 1} val max_val))  ; oeq
                (def v_i32 (arith.index_cast {:result i32} v))
                ;; Only update if not found yet (found_idx < 0) and this is the max
                (def not_found_yet (arith.cmpi {:predicate 2} found_idx (: 0 i32)))  ; slt
                (def should_update (arith.andi is_max not_found_yet))
                (def new_idx (arith.select should_update v_i32 found_idx))
                (scf.yield new_idx)))))

          (func.return best_idx))))

    ;; =========================================================================
    ;; Tokenizer init: load token table from binary file
    ;; =========================================================================
    (func.func {:sym_name "tokenizer_init"
                :function_type (-> [!llvm.ptr] [i32])}  ; path -> success
      (region
        (block [(: path !llvm.ptr)]
          (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
          (def file (call !llvm.ptr fopen path mode))

          ;; Read header (256 * 4 = 1024 bytes)
          (def header_size (: 1024 i64))
          (def header_ptr (call !llvm.ptr malloc header_size))
          (def _read_h (call i64 fread header_ptr (: 4 i64) (: 256 i64) file))

          ;; Check magic (should be 20240328)
          (def magic (llvm.load {:result i32} header_ptr))
          (print "Tokenizer magic: %d\n" magic)

          ;; Get vocab size (header[2])
          (def vocab_ptr (ptr-at i32 header_ptr (: 2 i64)))
          (def vocab_size (llvm.load {:result i32} vocab_ptr))
          (print "Vocab size: %d\n" vocab_size)

          ;; Get eot_token (header[3] for version 2, else 50256)
          (def version_ptr (ptr-at i32 header_ptr (: 1 i64)))
          (def version (llvm.load {:result i32} version_ptr))
          (def eot_ptr (ptr-at i32 header_ptr (: 3 i64)))
          (def eot_from_header (llvm.load {:result i32} eot_ptr))
          (def is_v2 (arith.cmpi {:predicate 2} version (: 2 i32)))  ; eq
          (def eot (arith.select is_v2 eot_from_header (: 50256 i32)))
          (def eot_global (llvm.mlir.addressof {:global_name @eot_token :result !llvm.ptr}))
          (llvm.store eot eot_global)
          (print "EOT token: %d\n" eot)

          ;; Allocate token table: 50257 pointers
          (def table_bytes (: 402056 i64))  ; 50257 * 8 bytes per pointer
          (def table_ptr (call !llvm.ptr malloc table_bytes))
          (def table_global (llvm.mlir.addressof {:global_name @token_table :result !llvm.ptr}))
          (llvm.store table_ptr table_global)

          ;; Read each token
          (def vocab_size_idx (arith.index_cast {:result index} vocab_size))
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def one_byte (: 1 i64))

          (scf.for c0 vocab_size_idx c1
            (region
              (block [(: i index)]
                ;; Read length (1 byte)
                (def len_buf (call !llvm.ptr malloc one_byte))
                (def _r1 (call i64 fread len_buf one_byte one_byte file))
                (def len_u8 (llvm.load {:result i8} len_buf))
                (def len (arith.extui {:result i64} len_u8))
                (call! free len_buf)

                ;; Allocate string buffer (len + 1 for null terminator)
                (def str_size (arith.addi len one_byte))
                (def str_ptr (call !llvm.ptr malloc str_size))

                ;; Read string
                (def _r2 (call i64 fread str_ptr one_byte len file))

                ;; Add null terminator
                (def null_ptr (ptr-at i8 str_ptr len))
                (def null_byte (: 0 i8))
                (llvm.store null_byte null_ptr)

                ;; Store in table
                (def i_i64 (arith.index_cast {:result i64} i))
                (def table_entry (ptr-at !llvm.ptr table_ptr i_i64))
                (llvm.store str_ptr table_entry)
                (scf.yield))))

          (def _close (call i32 fclose file))
          (call! free header_ptr)
          (print "Tokenizer loaded\n")
          (func.return (: 1 i32)))))

    ;; =========================================================================
    ;; Tokenizer decode: token_id -> string pointer
    ;; =========================================================================
    (func.func {:sym_name "tokenizer_decode"
                :function_type (-> [i32] [!llvm.ptr])}
      (region
        (block [(: token_id i32)]
          (def table_global (llvm.mlir.addressof {:global_name @token_table :result !llvm.ptr}))
          (def table_ptr (llvm.load {:result !llvm.ptr} table_global))
          (def id_i64 (arith.extsi {:result i64} token_id))
          (def entry_ptr (ptr-at !llvm.ptr table_ptr id_i64))
          (def str_ptr (llvm.load {:result !llvm.ptr} entry_ptr))
          (func.return str_ptr))))))

;; Weight offsets (same as gpt2_full_forward.lisp)
;; wte: 0
;; wpe: 38597376
;; ln1w_base: 39383808, stride: 768
;; etc.

(defn main [] -> i64
  ;; Load checkpoint (hardcoded path - modify checkpoint_path global if needed)
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; =========================================================================
  ;; Read checkpoint header (256 ints) to get model config
  ;; Header format:
  ;;   [0]: magic (20240326)
  ;;   [1]: version (3)
  ;;   [2]: max_seq_len (maxT)
  ;;   [3]: vocab_size (V)
  ;;   [4]: num_layers (L)
  ;;   [5]: num_heads (NH)
  ;;   [6]: channels (C)
  ;;   [7]: padded_vocab_size (Vp)
  ;; =========================================================================
  (def header_buf (call !llvm.ptr malloc (: 1024 i64)))  ; 256 * 4 bytes
  (def _read_header (call i64 fread header_buf (: 4 i64) (: 256 i64) file))

  ;; Extract model config from header
  (def magic (llvm.load {:result i32} header_buf))
  (def version (llvm.load {:result i32} (ptr-at i32 header_buf (: 1 i64))))
  (def maxT_i32 (llvm.load {:result i32} (ptr-at i32 header_buf (: 2 i64))))
  (def V_i32 (llvm.load {:result i32} (ptr-at i32 header_buf (: 3 i64))))
  (def L_i32 (llvm.load {:result i32} (ptr-at i32 header_buf (: 4 i64))))
  (def NH_i32 (llvm.load {:result i32} (ptr-at i32 header_buf (: 5 i64))))
  (def C_i32 (llvm.load {:result i32} (ptr-at i32 header_buf (: 6 i64))))
  (def Vp_i32 (llvm.load {:result i32} (ptr-at i32 header_buf (: 7 i64))))

  ;; Convert to i64 for offset calculations
  (def maxT (arith.extsi {:result i64} maxT_i32))
  (def V (arith.extsi {:result i64} V_i32))
  (def L (arith.extsi {:result i64} L_i32))
  (def NH (arith.extsi {:result i64} NH_i32))
  (def C (arith.extsi {:result i64} C_i32))
  (def Vp (arith.extsi {:result i64} Vp_i32))

  ;; Print config
  (print "[GPT-2 Config]\n")
  (print "  max_seq_len: %d\n" maxT_i32)
  (print "  vocab_size: %d\n" V_i32)
  (print "  padded_vocab_size: %d\n" Vp_i32)
  (print "  num_layers: %d\n" L_i32)
  (print "  num_heads: %d\n" NH_i32)
  (print "  channels: %d\n" C_i32)

  ;; Compute derived dimensions
  (def C3 (arith.muli C (: 3 i64)))      ; 3*C = 2304
  (def C4 (arith.muli C (: 4 i64)))      ; 4*C = 3072

  ;; =========================================================================
  ;; Compute parameter sizes (number of floats per tensor)
  ;; From llm.c fill_in_parameter_sizes:
  ;;   0. wte:      Vp * C
  ;;   1. wpe:      maxT * C
  ;;   2. ln1w:     L * C
  ;;   3. ln1b:     L * C
  ;;   4. qkvw:     L * 3*C * C
  ;;   5. qkvb:     L * 3*C
  ;;   6. attprojw: L * C * C
  ;;   7. attprojb: L * C
  ;;   8. ln2w:     L * C
  ;;   9. ln2b:     L * C
  ;;  10. fcw:      L * 4*C * C
  ;;  11. fcb:      L * 4*C
  ;;  12. fcprojw:  L * C * 4*C  (NOTE: same size as fcw, but different layout!)
  ;;  13. fcprojb:  L * C
  ;;  14. lnfw:     C
  ;;  15. lnfb:     C
  ;; =========================================================================
  (def size_wte (arith.muli Vp C))  ; llm.c uses Vp (padded) for checkpoint storage
  (def size_wpe (arith.muli maxT C))
  (def size_ln1w (arith.muli L C))
  (def size_ln1b (arith.muli L C))
  (def size_qkvw (arith.muli L (arith.muli C3 C)))
  (def size_qkvb (arith.muli L C3))
  (def size_attprojw (arith.muli L (arith.muli C C)))
  (def size_attprojb (arith.muli L C))
  (def size_ln2w (arith.muli L C))
  (def size_ln2b (arith.muli L C))
  (def size_fcw (arith.muli L (arith.muli C4 C)))
  (def size_fcb (arith.muli L C4))
  (def size_fcprojw (arith.muli L (arith.muli C C4)))
  (def size_fcprojb (arith.muli L C))
  (def size_lnfw C)
  (def size_lnfb C)

  ;; Compute total params
  (def total_params
    (arith.addi size_wte
      (arith.addi size_wpe
        (arith.addi size_ln1w
          (arith.addi size_ln1b
            (arith.addi size_qkvw
              (arith.addi size_qkvb
                (arith.addi size_attprojw
                  (arith.addi size_attprojb
                    (arith.addi size_ln2w
                      (arith.addi size_ln2b
                        (arith.addi size_fcw
                          (arith.addi size_fcb
                            (arith.addi size_fcprojw
                              (arith.addi size_fcprojb
                                (arith.addi size_lnfw size_lnfb))))))))))))))))

  (print "  total_params: %ld\n" total_params)

  ;; =========================================================================
  ;; Compute base offsets (cumulative)
  ;; =========================================================================
  (def wte_base (: 0 i64))
  (def wpe_base (arith.addi wte_base size_wte))
  (def ln1w_base (arith.addi wpe_base size_wpe))
  (def ln1b_base (arith.addi ln1w_base size_ln1w))
  (def qkvw_base (arith.addi ln1b_base size_ln1b))
  (def qkvb_base (arith.addi qkvw_base size_qkvw))
  (def attprojw_base (arith.addi qkvb_base size_qkvb))
  (def attprojb_base (arith.addi attprojw_base size_attprojw))
  (def ln2w_base (arith.addi attprojb_base size_attprojb))
  (def ln2b_base (arith.addi ln2w_base size_ln2w))
  (def fcw_base (arith.addi ln2b_base size_ln2b))
  (def fcb_base (arith.addi fcw_base size_fcw))
  (def fcprojw_base (arith.addi fcb_base size_fcb))
  (def fcprojb_base (arith.addi fcprojw_base size_fcprojw))
  (def lnfw_base (arith.addi fcprojb_base size_fcprojb))
  (def lnfb_base (arith.addi lnfw_base size_lnfw))

  ;; Debug: verify offsets (expected from hardcoded: fcw=67805184, fcprojw=96153600)
  (print "[Offsets] fcw_base=%ld (exp 67805184) fcprojw_base=%ld (exp 96153600)\n"
         fcw_base fcprojw_base)

  ;; =========================================================================
  ;; Compute per-layer strides
  ;; =========================================================================
  (def ln_stride C)                              ; C
  (def qkvw_stride (arith.muli C3 C))            ; 3*C * C
  (def qkvb_stride C3)                           ; 3*C
  (def attprojw_stride (arith.muli C C))         ; C * C
  (def attprojb_stride C)                        ; C
  (def fcw_stride (arith.muli C4 C))             ; 4*C * C
  (def fcb_stride C4)                            ; 4*C
  (def fcprojw_stride (arith.muli C C4))         ; C * 4*C
  (def fcprojb_stride C)                         ; C

  ;; =========================================================================
  ;; Load parameters
  ;; =========================================================================
  (def sizeof_f32 (: 4 i64))
  (def total_bytes (arith.muli total_params sizeof_f32))
  (def params_ptr (call !llvm.ptr malloc total_bytes))
  (print "Loading weights...\n")
  (def read_count (call i64 fread params_ptr sizeof_f32 total_params file))
  (print "Loaded %ld floats\n" read_count)
  (def _close (call i32 fclose file))
  ;; Note: header_buf leaks (1KB) but we're about to exit anyway

  ;; Store params globally
  (def g_params_addr (llvm.mlir.addressof {:global_name @g_params :result !llvm.ptr}))
  (llvm.store params_ptr g_params_addr)

  ;; Verify wte[0][0]
  (def wte_val (llvm.load {:result f32} params_ptr))
  (def wte_val_f64 (arith.extf {:result f64} wte_val))
  (print "wte[0][0] = %f\n" wte_val_f64)

  ;; Load tokenizer (hardcoded path - modify tokenizer_path global if needed)
  (def tok_path (llvm.mlir.addressof {:global_name @tokenizer_path :result !llvm.ptr}))
  (def _tok_ok (func.call {:result i32} "tokenizer_init" tok_path))

  ;; Get wte and wpe pointers (computed from dynamic offsets)
  (def wte_ptr params_ptr)
  (def wpe_ptr (ptr-at f32 params_ptr wpe_base))

  ;; Create token_ids buffer with EOT token (start of generation)
  (def token_ids (memref.alloc {:result memref<64xi32>}))
  (def eot_global (llvm.mlir.addressof {:global_name @eot_token :result !llvm.ptr}))
  (def eot_token (llvm.load {:result i32} eot_global))
  (def c0 (: 0 index))
  (def c1 (: 1 index))
  (def c64 (: 64 index))

  ;; Fill with EOT
  (scf.for c0 c64 c1
    (region
      (block [(: i index)]
        (memref.store eot_token token_ids i)
        (scf.yield))))

  (print "Prompt: <|endoftext|>\nGenerated:")

  ;; Allocate activation buffers
  (def x (memref.alloc {:result memref<64x768xf32>}))
  (def x2 (memref.alloc {:result memref<64x768xf32>}))
  (def ln_out (memref.alloc {:result memref<64x768xf32>}))
  (def qkv_out (memref.alloc {:result memref<64x2304xf32>}))
  (def attn_out (memref.alloc {:result memref<64x768xf32>}))
  (def attn_proj_out (memref.alloc {:result memref<64x768xf32>}))
  (def fc_out (memref.alloc {:result memref<64x3072xf32>}))
  (def gelu_out (memref.alloc {:result memref<64x3072xf32>}))
  (def fc_proj_out (memref.alloc {:result memref<64x768xf32>}))
  (def logits (memref.alloc {:result memref<50257xf32>}))

  ;; Layer weight buffers
  (def ln1_w (memref.alloc {:result memref<768xf32>}))
  (def ln1_b (memref.alloc {:result memref<768xf32>}))
  (def qkv_w (memref.alloc {:result memref<768x2304xf32>}))
  (def qkv_b (memref.alloc {:result memref<2304xf32>}))
  (def attn_w (memref.alloc {:result memref<768x768xf32>}))
  (def attn_b (memref.alloc {:result memref<768xf32>}))
  (def ln2_w (memref.alloc {:result memref<768xf32>}))
  (def ln2_b (memref.alloc {:result memref<768xf32>}))
  (def fc_w (memref.alloc {:result memref<768x3072xf32>}))
  (def fc_b (memref.alloc {:result memref<3072xf32>}))
  (def fcproj_w (memref.alloc {:result memref<3072x768xf32>}))
  (def fcproj_b (memref.alloc {:result memref<768xf32>}))
  (def lnf_w (memref.alloc {:result memref<768xf32>}))
  (def lnf_b (memref.alloc {:result memref<768xf32>}))

  (def c768 (: 768 index))
  (def c2304 (: 2304 index))
  (def c3072 (: 3072 index))
  (def c12 (: 12 index))

  ;; =========================================================================
  ;; GENERATION LOOP: Generate 20 tokens
  ;; =========================================================================
  (def prompt_len (: 1 index))
  (def gen_steps (: 20 index))  ; generate 20 tokens
  (def gen_end (arith.addi prompt_len gen_steps))

  (print "\nGenerating tokens:\n")

  ;; step is the position where we'll store the next token
  ;; We compute logits at position (step - 1) which is the last filled position
  (scf.for prompt_len gen_end c1
    (region
      (block [(: step index)]
        ;; 1. Embedding lookup
        (func.call "embedding_lookup" x wte_ptr wpe_ptr token_ids)

        ;; Debug: check embedding on first step
        (def is_first_step_embed (arith.cmpi {:predicate 0} step prompt_len))
        (scf.if is_first_step_embed
          (region
            (block []
              (def e0 (memref.load x (: 0 index) (: 0 index)))
              (def e1 (memref.load x (: 0 index) (: 1 index)))
              (def e2 (memref.load x (: 0 index) (: 2 index)))
              (def e3 (memref.load x (: 0 index) (: 3 index)))
              (def e4 (memref.load x (: 0 index) (: 4 index)))
              (def e496 (memref.load x (: 0 index) (: 496 index)))
              (print "After embedding:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                     (arith.extf {:result f64} e0)
                     (arith.extf {:result f64} e1)
                     (arith.extf {:result f64} e2)
                     (arith.extf {:result f64} e3)
                     (arith.extf {:result f64} e4)
                     (arith.extf {:result f64} e496))
              (scf.yield)))
          (region (block [] (scf.yield))))

        ;; Position to compute logits at (last filled position)
        (def logit_pos (arith.subi step c1))

        ;; 2. Run 12 transformer layers
        (scf.for c0 c12 c1
          (region
            (block [(: layer index)]
              (def layer_i64 (arith.index_cast {:result i64} layer))

              ;; Copy layer weights (same as gpt2_full_forward.lisp)
              ;; ln1 weights
              (def ln1w_offset (arith.addi ln1w_base (arith.muli layer_i64 ln_stride)))
              (def ln1b_offset (arith.addi ln1b_base (arith.muli layer_i64 ln_stride)))
              (scf.for c0 c768 c1
                (region
                  (block [(: i index)]
                    (def i_i64 (arith.index_cast {:result i64} i))
                    (def w_ptr (ptr-at f32 params_ptr (arith.addi ln1w_offset i_i64)))
                    (def b_ptr (ptr-at f32 params_ptr (arith.addi ln1b_offset i_i64)))
                    (memref.store (llvm.load {:result f32} w_ptr) ln1_w i)
                    (memref.store (llvm.load {:result f32} b_ptr) ln1_b i)
                    (scf.yield))))

              ;; Debug: check layer 2 weights
              (def is_layer2_check (arith.cmpi {:predicate 0} layer (: 2 index)))
              (def do_layer2_debug (arith.andi is_layer2_check is_first_step_embed))
              (scf.if do_layer2_debug
                (region
                  (block []
                    ;; Print ln1w values
                    (def w0 (memref.load ln1_w (: 0 index)))
                    (def w1 (memref.load ln1_w (: 1 index)))
                    (def w2 (memref.load ln1_w (: 2 index)))
                    (def w3 (memref.load ln1_w (: 3 index)))
                    (def w4 (memref.load ln1_w (: 4 index)))
                    (print "Layer 2 ln1w[0..4]: %.6f %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} w0)
                           (arith.extf {:result f64} w1)
                           (arith.extf {:result f64} w2)
                           (arith.extf {:result f64} w3)
                           (arith.extf {:result f64} w4))
                    ;; Print ln1b values
                    (def b0 (memref.load ln1_b (: 0 index)))
                    (def b1 (memref.load ln1_b (: 1 index)))
                    (def b2 (memref.load ln1_b (: 2 index)))
                    (def b3 (memref.load ln1_b (: 3 index)))
                    (def b4 (memref.load ln1_b (: 4 index)))
                    (print "Layer 2 ln1b[0..4]: %.6f %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} b0)
                           (arith.extf {:result f64} b1)
                           (arith.extf {:result f64} b2)
                           (arith.extf {:result f64} b3)
                           (arith.extf {:result f64} b4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; qkv weights - llm.c stores as weight[oc*C+c] (row stride = C)
              ;; Transpose load: checkpoint[oc*C+c] -> qkv_w[c][oc]
              (def qkvw_offset (arith.addi qkvw_base (arith.muli layer_i64 qkvw_stride)))
              (def qkvb_offset (arith.addi qkvb_base (arith.muli layer_i64 qkvb_stride)))
              (scf.for c0 c2304 c1
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 C))  ; row_stride = C
                    (scf.for c0 c768 c1
                      (region
                        (block [(: c index)]
                          (def c_i64 (arith.index_cast {:result i64} c))
                          (def w_ptr (ptr-at f32 params_ptr (arith.addi qkvw_offset (arith.addi row_offset c_i64))))
                          (memref.store (llvm.load {:result f32} w_ptr) qkv_w c oc)
                          (scf.yield))))
                    (scf.yield))))
              (scf.for c0 c2304 c1
                (region
                  (block [(: i index)]
                    (def i_i64 (arith.index_cast {:result i64} i))
                    (def b_ptr (ptr-at f32 params_ptr (arith.addi qkvb_offset i_i64)))
                    (memref.store (llvm.load {:result f32} b_ptr) qkv_b i)
                    (scf.yield))))

              ;; attention projection weights - llm.c stores as weight[oc*C+c] (row stride = C)
              ;; Transpose load: checkpoint[oc*C+c] -> attn_w[c][oc]
              (def attprojw_offset (arith.addi attprojw_base (arith.muli layer_i64 attprojw_stride)))
              (def attprojb_offset (arith.addi attprojb_base (arith.muli layer_i64 attprojb_stride)))
              (scf.for c0 c768 c1
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 C))  ; row_stride = C
                    (scf.for c0 c768 c1
                      (region
                        (block [(: c index)]
                          (def c_i64 (arith.index_cast {:result i64} c))
                          (def w_ptr (ptr-at f32 params_ptr (arith.addi attprojw_offset (arith.addi row_offset c_i64))))
                          (memref.store (llvm.load {:result f32} w_ptr) attn_w c oc)
                          (scf.yield))))
                    (scf.yield))))
              ;; Bias doesn't need transpose - it's just (OC)
              (scf.for c0 c768 c1
                (region
                  (block [(: i index)]
                    (def i_i64 (arith.index_cast {:result i64} i))
                    (def ab_ptr (ptr-at f32 params_ptr (arith.addi attprojb_offset i_i64)))
                    (memref.store (llvm.load {:result f32} ab_ptr) attn_b i)
                    (scf.yield))))

              ;; ln2 weights
              (def ln2w_offset (arith.addi ln2w_base (arith.muli layer_i64 ln_stride)))
              (def ln2b_offset (arith.addi ln2b_base (arith.muli layer_i64 ln_stride)))
              (scf.for c0 c768 c1
                (region
                  (block [(: i index)]
                    (def i_i64 (arith.index_cast {:result i64} i))
                    (def w_ptr (ptr-at f32 params_ptr (arith.addi ln2w_offset i_i64)))
                    (def b_ptr (ptr-at f32 params_ptr (arith.addi ln2b_offset i_i64)))
                    (memref.store (llvm.load {:result f32} w_ptr) ln2_w i)
                    (memref.store (llvm.load {:result f32} b_ptr) ln2_b i)
                    (scf.yield))))

              ;; fc weights - llm.c stores as weight[oc*C+c] (row stride = C)
              ;; Transpose load: checkpoint[oc*C+c] -> fc_w[c][oc]
              (def fcw_offset (arith.addi fcw_base (arith.muli layer_i64 fcw_stride)))
              (def fcb_offset (arith.addi fcb_base (arith.muli layer_i64 fcb_stride)))
              (scf.for c0 c3072 c1
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 C))  ; row_stride = C
                    (scf.for c0 c768 c1
                      (region
                        (block [(: c index)]
                          (def c_i64 (arith.index_cast {:result i64} c))
                          (def w_ptr (ptr-at f32 params_ptr (arith.addi fcw_offset (arith.addi row_offset c_i64))))
                          (memref.store (llvm.load {:result f32} w_ptr) fc_w c oc)
                          (scf.yield))))
                    (scf.yield))))
              (scf.for c0 c3072 c1
                (region
                  (block [(: i index)]
                    (def i_i64 (arith.index_cast {:result i64} i))
                    (def b_ptr (ptr-at f32 params_ptr (arith.addi fcb_offset i_i64)))
                    (memref.store (llvm.load {:result f32} b_ptr) fc_b i)
                    (scf.yield))))

              ;; Debug: check layer 2 fc_w weights
              (scf.if do_layer2_debug
                (region
                  (block []
                    ;; Print raw checkpoint values for fc_w
                    (def fcw_raw0 (llvm.load {:result f32} (ptr-at f32 params_ptr fcw_offset)))
                    (def fcw_raw1 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi fcw_offset (: 1 i64)))))
                    (def fcw_raw768 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi fcw_offset (: 768 i64)))))
                    (def fcw_raw3072 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi fcw_offset (: 3072 i64)))))
                    (print "Layer 2 fc_w RAW checkpoint:\n  [0]: %.6f  [1]: %.6f  [768]: %.6f  [3072]: %.6f\n"
                           (arith.extf {:result f64} fcw_raw0)
                           (arith.extf {:result f64} fcw_raw1)
                           (arith.extf {:result f64} fcw_raw768)
                           (arith.extf {:result f64} fcw_raw3072))
                    ;; Print loaded values - column 0 (expect raw[0], raw[3072], raw[6144], ...)
                    (def fw0 (memref.load fc_w (: 0 index) (: 0 index)))
                    (def fw1 (memref.load fc_w (: 1 index) (: 0 index)))
                    (def fw2 (memref.load fc_w (: 2 index) (: 0 index)))
                    (def fw3 (memref.load fc_w (: 3 index) (: 0 index)))
                    (def fw4 (memref.load fc_w (: 4 index) (: 0 index)))
                    (print "Layer 2 fc_w loaded [0..4][0] (expect raw[0,1,2,3,4]): %.6f %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} fw0)
                           (arith.extf {:result f64} fw1)
                           (arith.extf {:result f64} fw2)
                           (arith.extf {:result f64} fw3)
                           (arith.extf {:result f64} fw4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; fc projection weights - llm.c stores as weight[oc*(4*C)+ic] (row stride = 4*C)
              ;; Transpose load: checkpoint[oc*(4*C)+ic] -> fcproj_w[ic][oc]
              (def fcprojw_offset (arith.addi fcprojw_base (arith.muli layer_i64 fcprojw_stride)))
              (def fcprojb_offset (arith.addi fcprojb_base (arith.muli layer_i64 fcprojb_stride)))
              (scf.for c0 c768 c1  ; outer loop over oc (output dim = C)
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 C4))  ; row_stride = 4*C
                    (scf.for c0 c3072 c1  ; inner loop over ic (input dim = 4*C)
                      (region
                        (block [(: ic index)]
                          (def ic_i64 (arith.index_cast {:result i64} ic))
                          (def w_ptr (ptr-at f32 params_ptr (arith.addi fcprojw_offset (arith.addi row_offset ic_i64))))
                          (memref.store (llvm.load {:result f32} w_ptr) fcproj_w ic oc)
                          (scf.yield))))
                    (scf.yield))))
              (scf.for c0 c768 c1
                (region
                  (block [(: i index)]
                    (def i_i64 (arith.index_cast {:result i64} i))
                    (def b_ptr (ptr-at f32 params_ptr (arith.addi fcprojb_offset i_i64)))
                    (memref.store (llvm.load {:result f32} b_ptr) fcproj_b i)
                    (scf.yield))))

              ;; Debug: check layer 2 fcproj weights
              (scf.if do_layer2_debug
                (region
                  (block []
                    ;; Print RAW checkpoint values at different offsets
                    (def raw0 (llvm.load {:result f32} (ptr-at f32 params_ptr fcprojw_offset)))
                    (def raw1 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi fcprojw_offset (: 1 i64)))))
                    (def raw768 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi fcprojw_offset (: 768 i64)))))
                    (def raw3072 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi fcprojw_offset (: 3072 i64)))))
                    (print "Layer 2 fcprojw RAW checkpoint:\n")
                    (print "  [0]: %.6f  [1]: %.6f  [768]: %.6f  [3072]: %.6f\n"
                           (arith.extf {:result f64} raw0)
                           (arith.extf {:result f64} raw1)
                           (arith.extf {:result f64} raw768)
                           (arith.extf {:result f64} raw3072))
                    ;; fcproj_w loaded values
                    (def fpw0 (memref.load fcproj_w (: 0 index) (: 0 index)))
                    (def fpw1 (memref.load fcproj_w (: 1 index) (: 0 index)))
                    (def fpw2 (memref.load fcproj_w (: 2 index) (: 0 index)))
                    (def fpw3 (memref.load fcproj_w (: 3 index) (: 0 index)))
                    (def fpw4 (memref.load fcproj_w (: 4 index) (: 0 index)))
                    (print "Layer 2 fcprojw loaded [ic][0] (expect raw[0,1,2,3,4]): %.6f %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} fpw0)
                           (arith.extf {:result f64} fpw1)
                           (arith.extf {:result f64} fpw2)
                           (arith.extf {:result f64} fpw3)
                           (arith.extf {:result f64} fpw4))
                    ;; fcproj_b[0..4]
                    (def fpb0 (memref.load fcproj_b (: 0 index)))
                    (def fpb1 (memref.load fcproj_b (: 1 index)))
                    (def fpb2 (memref.load fcproj_b (: 2 index)))
                    (def fpb3 (memref.load fcproj_b (: 3 index)))
                    (def fpb4 (memref.load fcproj_b (: 4 index)))
                    (print "Layer 2 fcprojb[0..4]: %.6f %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} fpb0)
                           (arith.extf {:result f64} fpb1)
                           (arith.extf {:result f64} fpb2)
                           (arith.extf {:result f64} fpb3)
                           (arith.extf {:result f64} fpb4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; === Run transformer block ===

              ;; Debug: check layer 2 input (x before LN1)
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def xi0 (memref.load x (: 0 index) (: 0 index)))
                    (def xi1 (memref.load x (: 0 index) (: 1 index)))
                    (def xi2 (memref.load x (: 0 index) (: 2 index)))
                    (def xi3 (memref.load x (: 0 index) (: 3 index)))
                    (def xi4 (memref.load x (: 0 index) (: 4 index)))
                    (def xi496 (memref.load x (: 0 index) (: 496 index)))
                    (print "Layer 2 input (x):\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                           (arith.extf {:result f64} xi0)
                           (arith.extf {:result f64} xi1)
                           (arith.extf {:result f64} xi2)
                           (arith.extf {:result f64} xi3)
                           (arith.extf {:result f64} xi4)
                           (arith.extf {:result f64} xi496))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "layernorm_forward" ln_out x ln1_w ln1_b)

              ;; Debug: check layer 2 ln1 output
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def lno0 (memref.load ln_out (: 0 index) (: 0 index)))
                    (def lno1 (memref.load ln_out (: 0 index) (: 1 index)))
                    (def lno2 (memref.load ln_out (: 0 index) (: 2 index)))
                    (def lno3 (memref.load ln_out (: 0 index) (: 3 index)))
                    (def lno4 (memref.load ln_out (: 0 index) (: 4 index)))
                    (def lno496 (memref.load ln_out (: 0 index) (: 496 index)))
                    (print "Layer 2 ln1 output:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                           (arith.extf {:result f64} lno0)
                           (arith.extf {:result f64} lno1)
                           (arith.extf {:result f64} lno2)
                           (arith.extf {:result f64} lno3)
                           (arith.extf {:result f64} lno4)
                           (arith.extf {:result f64} lno496))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; Debug: check layer 0 ln1 output
              (def is_layer0 (arith.cmpi {:predicate 0} layer c0))
              (def is_debug_step (arith.andi is_layer0 is_first_step_embed))
              (scf.if is_debug_step
                (region
                  (block []
                    (def ln0 (memref.load ln_out (: 0 index) (: 0 index)))
                    (def ln1_v (memref.load ln_out (: 0 index) (: 1 index)))
                    (def ln2 (memref.load ln_out (: 0 index) (: 2 index)))
                    (def ln3 (memref.load ln_out (: 0 index) (: 3 index)))
                    (def ln4 (memref.load ln_out (: 0 index) (: 4 index)))
                    (def ln496 (memref.load ln_out (: 0 index) (: 496 index)))
                    (print "After LN1 (layer 0):\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                           (arith.extf {:result f64} ln0)
                           (arith.extf {:result f64} ln1_v)
                           (arith.extf {:result f64} ln2)
                           (arith.extf {:result f64} ln3)
                           (arith.extf {:result f64} ln4)
                           (arith.extf {:result f64} ln496))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "matmul_qkv" qkv_out ln_out qkv_w qkv_b)

              ;; Debug: check QKV output for layer 2
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def qkv0 (memref.load qkv_out (: 0 index) (: 0 index)))
                    (def qkv1 (memref.load qkv_out (: 0 index) (: 1 index)))
                    (def qkv2 (memref.load qkv_out (: 0 index) (: 2 index)))
                    (def qkv3 (memref.load qkv_out (: 0 index) (: 3 index)))
                    (def qkv4 (memref.load qkv_out (: 0 index) (: 4 index)))
                    (print "Layer 2 Q[0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} qkv0)
                           (arith.extf {:result f64} qkv1)
                           (arith.extf {:result f64} qkv2)
                           (arith.extf {:result f64} qkv3)
                           (arith.extf {:result f64} qkv4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; Debug: check QKV output for layer 0
              (scf.if is_debug_step
                (region
                  (block []
                    ;; Q values (0..4)
                    (def q0 (memref.load qkv_out (: 0 index) (: 0 index)))
                    (def q1 (memref.load qkv_out (: 0 index) (: 1 index)))
                    (def q2 (memref.load qkv_out (: 0 index) (: 2 index)))
                    (def q3 (memref.load qkv_out (: 0 index) (: 3 index)))
                    (def q4 (memref.load qkv_out (: 0 index) (: 4 index)))
                    ;; K values (768..772)
                    (def k0 (memref.load qkv_out (: 0 index) (: 768 index)))
                    (def k1 (memref.load qkv_out (: 0 index) (: 769 index)))
                    (def k2 (memref.load qkv_out (: 0 index) (: 770 index)))
                    (def k3 (memref.load qkv_out (: 0 index) (: 771 index)))
                    (def k4 (memref.load qkv_out (: 0 index) (: 772 index)))
                    ;; V values (1536..1540)
                    (def v0 (memref.load qkv_out (: 0 index) (: 1536 index)))
                    (def v1 (memref.load qkv_out (: 0 index) (: 1537 index)))
                    (def v2 (memref.load qkv_out (: 0 index) (: 1538 index)))
                    (def v3 (memref.load qkv_out (: 0 index) (: 1539 index)))
                    (def v4 (memref.load qkv_out (: 0 index) (: 1540 index)))
                    (print "After QKV (layer 0):\n  Q[0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} q0)
                           (arith.extf {:result f64} q1)
                           (arith.extf {:result f64} q2)
                           (arith.extf {:result f64} q3)
                           (arith.extf {:result f64} q4))
                    (print "  K[0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} k0)
                           (arith.extf {:result f64} k1)
                           (arith.extf {:result f64} k2)
                           (arith.extf {:result f64} k3)
                           (arith.extf {:result f64} k4))
                    (print "  V[0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} v0)
                           (arith.extf {:result f64} v1)
                           (arith.extf {:result f64} v2)
                           (arith.extf {:result f64} v3)
                           (arith.extf {:result f64} v4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "attention_forward" attn_out qkv_out)

              ;; Debug: check attention output for layer 2
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def at0 (memref.load attn_out (: 0 index) (: 0 index)))
                    (def at1 (memref.load attn_out (: 0 index) (: 1 index)))
                    (def at2 (memref.load attn_out (: 0 index) (: 2 index)))
                    (def at3 (memref.load attn_out (: 0 index) (: 3 index)))
                    (def at4 (memref.load attn_out (: 0 index) (: 4 index)))
                    (print "Layer 2 attn output:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} at0)
                           (arith.extf {:result f64} at1)
                           (arith.extf {:result f64} at2)
                           (arith.extf {:result f64} at3)
                           (arith.extf {:result f64} at4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; Debug: check attention output for layer 0
              (scf.if is_debug_step
                (region
                  (block []
                    (def a0 (memref.load attn_out (: 0 index) (: 0 index)))
                    (def a1 (memref.load attn_out (: 0 index) (: 1 index)))
                    (def a2 (memref.load attn_out (: 0 index) (: 2 index)))
                    (def a3 (memref.load attn_out (: 0 index) (: 3 index)))
                    (def a4 (memref.load attn_out (: 0 index) (: 4 index)))
                    (def a496 (memref.load attn_out (: 0 index) (: 496 index)))
                    (print "After attention (layer 0):\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                           (arith.extf {:result f64} a0)
                           (arith.extf {:result f64} a1)
                           (arith.extf {:result f64} a2)
                           (arith.extf {:result f64} a3)
                           (arith.extf {:result f64} a4)
                           (arith.extf {:result f64} a496))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "matmul_attn_proj" attn_proj_out attn_out attn_w attn_b)

              ;; Debug: attention projection for layer 2
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def ap2_0 (memref.load attn_proj_out (: 0 index) (: 0 index)))
                    (def ap2_1 (memref.load attn_proj_out (: 0 index) (: 1 index)))
                    (def ap2_2 (memref.load attn_proj_out (: 0 index) (: 2 index)))
                    (def ap2_3 (memref.load attn_proj_out (: 0 index) (: 3 index)))
                    (def ap2_4 (memref.load attn_proj_out (: 0 index) (: 4 index)))
                    (print "Layer 2 attn proj:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} ap2_0)
                           (arith.extf {:result f64} ap2_1)
                           (arith.extf {:result f64} ap2_2)
                           (arith.extf {:result f64} ap2_3)
                           (arith.extf {:result f64} ap2_4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; Debug: attention projection
              (scf.if is_debug_step
                (region
                  (block []
                    (def ap0 (memref.load attn_proj_out (: 0 index) (: 0 index)))
                    (def ap1 (memref.load attn_proj_out (: 0 index) (: 1 index)))
                    (def ap2 (memref.load attn_proj_out (: 0 index) (: 2 index)))
                    (def ap3 (memref.load attn_proj_out (: 0 index) (: 3 index)))
                    (def ap4 (memref.load attn_proj_out (: 0 index) (: 4 index)))
                    (def ap496 (memref.load attn_proj_out (: 0 index) (: 496 index)))
                    (print "After attn proj (layer 0):\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                           (arith.extf {:result f64} ap0)
                           (arith.extf {:result f64} ap1)
                           (arith.extf {:result f64} ap2)
                           (arith.extf {:result f64} ap3)
                           (arith.extf {:result f64} ap4)
                           (arith.extf {:result f64} ap496))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "residual_add" x2 x attn_proj_out)

              ;; Debug: layer 2 residual1
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def r2_0 (memref.load x2 (: 0 index) (: 0 index)))
                    (def r2_1 (memref.load x2 (: 0 index) (: 1 index)))
                    (def r2_2 (memref.load x2 (: 0 index) (: 2 index)))
                    (def r2_3 (memref.load x2 (: 0 index) (: 3 index)))
                    (def r2_4 (memref.load x2 (: 0 index) (: 4 index)))
                    (print "Layer 2 residual1:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} r2_0)
                           (arith.extf {:result f64} r2_1)
                           (arith.extf {:result f64} r2_2)
                           (arith.extf {:result f64} r2_3)
                           (arith.extf {:result f64} r2_4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; Debug: after first residual
              (scf.if is_debug_step
                (region
                  (block []
                    (def r0 (memref.load x2 (: 0 index) (: 0 index)))
                    (def r1 (memref.load x2 (: 0 index) (: 1 index)))
                    (def r2 (memref.load x2 (: 0 index) (: 2 index)))
                    (def r3 (memref.load x2 (: 0 index) (: 3 index)))
                    (def r4 (memref.load x2 (: 0 index) (: 4 index)))
                    (def r496 (memref.load x2 (: 0 index) (: 496 index)))
                    (print "After residual1 (layer 0):\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                           (arith.extf {:result f64} r0)
                           (arith.extf {:result f64} r1)
                           (arith.extf {:result f64} r2)
                           (arith.extf {:result f64} r3)
                           (arith.extf {:result f64} r4)
                           (arith.extf {:result f64} r496))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "layernorm_forward" ln_out x2 ln2_w ln2_b)

              ;; Debug: layer 2 ln2 output (fc input)
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def ln2o0 (memref.load ln_out (: 0 index) (: 0 index)))
                    (def ln2o1 (memref.load ln_out (: 0 index) (: 1 index)))
                    (def ln2o2 (memref.load ln_out (: 0 index) (: 2 index)))
                    (def ln2o3 (memref.load ln_out (: 0 index) (: 3 index)))
                    (def ln2o4 (memref.load ln_out (: 0 index) (: 4 index)))
                    (print "Layer 2 ln2 output (fc input):\n  [0..4]: %.6f %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} ln2o0)
                           (arith.extf {:result f64} ln2o1)
                           (arith.extf {:result f64} ln2o2)
                           (arith.extf {:result f64} ln2o3)
                           (arith.extf {:result f64} ln2o4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "matmul_fc" fc_out ln_out fc_w fc_b)

              ;; Debug: layer 2 fc output
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def fc0 (memref.load fc_out (: 0 index) (: 0 index)))
                    (def fc1 (memref.load fc_out (: 0 index) (: 1 index)))
                    (def fc2 (memref.load fc_out (: 0 index) (: 2 index)))
                    (def fc3 (memref.load fc_out (: 0 index) (: 3 index)))
                    (def fc4 (memref.load fc_out (: 0 index) (: 4 index)))
                    (print "Layer 2 fc output:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} fc0)
                           (arith.extf {:result f64} fc1)
                           (arith.extf {:result f64} fc2)
                           (arith.extf {:result f64} fc3)
                           (arith.extf {:result f64} fc4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "gelu_forward" gelu_out fc_out)

              ;; Debug: verify gelu at 642 immediately after call
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def fc642_check (memref.load fc_out (: 0 index) (: 642 index)))
                    (def gelu642_check (memref.load gelu_out (: 0 index) (: 642 index)))
                    ;; Manual gelu computation for fc[642]
                    (def sqrt_2_over_pi (: 0.7978845608 f32))
                    (def coeff (: 0.044715 f32))
                    (def half (: 0.5 f32))
                    (def one (: 1.0 f32))
                    (def x fc642_check)
                    (def x3 (arith.mulf x (arith.mulf x x)))
                    (def inner (arith.mulf sqrt_2_over_pi (arith.addf x (arith.mulf coeff x3))))
                    (def tanh_val (func.call {:result f32} "my_tanhf" inner))
                    (def manual_gelu (arith.mulf (arith.mulf half x) (arith.addf one tanh_val)))
                    (print "  DEBUG at 642: fc=%.6f gelu_stored=%.6f manual_gelu=%.6f tanh=%.6f\n"
                           (arith.extf {:result f64} fc642_check)
                           (arith.extf {:result f64} gelu642_check)
                           (arith.extf {:result f64} manual_gelu)
                           (arith.extf {:result f64} tanh_val))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; Debug: layer 2 gelu output
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def g0 (memref.load gelu_out (: 0 index) (: 0 index)))
                    (def g1 (memref.load gelu_out (: 0 index) (: 1 index)))
                    (def g2 (memref.load gelu_out (: 0 index) (: 2 index)))
                    (def g3 (memref.load gelu_out (: 0 index) (: 3 index)))
                    (def g4 (memref.load gelu_out (: 0 index) (: 4 index)))
                    (print "Layer 2 gelu output:\n  [0..4]: %.6f %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} g0)
                           (arith.extf {:result f64} g1)
                           (arith.extf {:result f64} g2)
                           (arith.extf {:result f64} g3)
                           (arith.extf {:result f64} g4))
                    ;; Print fc sum
                    (def c3072_fc (: 3072 index))
                    (def fc_sum (scf.for {:result f32} c0 c3072_fc c1 (: 0.0 f32)
                      (region
                        (block [(: cc index) (: acc f32)]
                          (def fv (memref.load fc_out (: 0 index) cc))
                          (scf.yield (arith.addf acc fv))))))
                    (print "  Sum of all fc[0] values: %.6f\n" (arith.extf {:result f64} fc_sum))
                    ;; Print fc at specific indices
                    (def fc500 (memref.load fc_out (: 0 index) (: 500 index)))
                    (def fc501 (memref.load fc_out (: 0 index) (: 501 index)))
                    (def fc502 (memref.load fc_out (: 0 index) (: 502 index)))
                    (def fc503 (memref.load fc_out (: 0 index) (: 503 index)))
                    (def fc504 (memref.load fc_out (: 0 index) (: 504 index)))
                    (print "  fc[500..504]: %.3f %.3f %.3f %.3f %.3f\n"
                           (arith.extf {:result f64} fc500) (arith.extf {:result f64} fc501)
                           (arith.extf {:result f64} fc502) (arith.extf {:result f64} fc503)
                           (arith.extf {:result f64} fc504))
                    ;; Print gelu sum BEFORE matmul_fc_proj
                    (def c3072_gelu (: 3072 index))
                    (def gelu_sum_before (scf.for {:result f32} c0 c3072_gelu c1 (: 0.0 f32)
                      (region
                        (block [(: cc index) (: acc f32)]
                          (def gv (memref.load gelu_out (: 0 index) cc))
                          (scf.yield (arith.addf acc gv))))))
                    (print "  Sum of gelu[0] BEFORE matmul_fc_proj: %.6f\n" (arith.extf {:result f64} gelu_sum_before))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "matmul_fc_proj" fc_proj_out gelu_out fcproj_w fcproj_b)

              ;; Debug: manually compute fc_proj[0][0] for layer 2
              (scf.if do_layer2_debug
                (region
                  (block []
                    ;; Manual: fc_proj[0][0] = sum_c gelu[0][c] * fcproj_w[c][0] + bias[0]
                    ;; Compute full sum over 3072 elements
                    (def c3072 (: 3072 index))
                    (def manual_sum (scf.for {:result f32} c0 c3072 c1 (: 0.0 f32)
                      (region
                        (block [(: c index) (: acc f32)]
                          (def gv (memref.load gelu_out (: 0 index) c))
                          (def wv (memref.load fcproj_w c (: 0 index)))
                          (scf.yield (arith.addf acc (arith.mulf gv wv)))))))
                    (def b0 (memref.load fcproj_b (: 0 index)))
                    (def manual_result (arith.addf manual_sum b0))
                    (print "Manual fc_proj[0][0]: sum=%.6f + bias=%.6f = %.6f\n"
                           (arith.extf {:result f64} manual_sum)
                           (arith.extf {:result f64} b0)
                           (arith.extf {:result f64} manual_result))
                    ;; Also check a few sample values from gelu and weights
                    (def g100 (memref.load gelu_out (: 0 index) (: 100 index)))
                    (def g1000 (memref.load gelu_out (: 0 index) (: 1000 index)))
                    (def g3000 (memref.load gelu_out (: 0 index) (: 3000 index)))
                    (print "  gelu_out[0][100,1000,3000]: %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} g100)
                           (arith.extf {:result f64} g1000)
                           (arith.extf {:result f64} g3000))
                    ;; Check weight values for column 0 at different rows
                    (def w100 (memref.load fcproj_w (: 100 index) (: 0 index)))
                    (def w1000 (memref.load fcproj_w (: 1000 index) (: 0 index)))
                    (def w3000 (memref.load fcproj_w (: 3000 index) (: 0 index)))
                    (print "  fcproj_w[100,1000,3000][0]: %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} w100)
                           (arith.extf {:result f64} w1000)
                           (arith.extf {:result f64} w3000))
                    ;; Print first 10 weights like C reference
                    (def fpw0 (memref.load fcproj_w (: 0 index) (: 0 index)))
                    (def fpw1 (memref.load fcproj_w (: 1 index) (: 0 index)))
                    (def fpw2 (memref.load fcproj_w (: 2 index) (: 0 index)))
                    (def fpw3 (memref.load fcproj_w (: 3 index) (: 0 index)))
                    (def fpw4 (memref.load fcproj_w (: 4 index) (: 0 index)))
                    (def fpw5 (memref.load fcproj_w (: 5 index) (: 0 index)))
                    (def fpw6 (memref.load fcproj_w (: 6 index) (: 0 index)))
                    (def fpw7 (memref.load fcproj_w (: 7 index) (: 0 index)))
                    (def fpw8 (memref.load fcproj_w (: 8 index) (: 0 index)))
                    (def fpw9 (memref.load fcproj_w (: 9 index) (: 0 index)))
                    (print "  fcproj_w[0..9][0]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} fpw0) (arith.extf {:result f64} fpw1)
                           (arith.extf {:result f64} fpw2) (arith.extf {:result f64} fpw3)
                           (arith.extf {:result f64} fpw4) (arith.extf {:result f64} fpw5)
                           (arith.extf {:result f64} fpw6) (arith.extf {:result f64} fpw7)
                           (arith.extf {:result f64} fpw8) (arith.extf {:result f64} fpw9))
                    ;; Print gelu sum
                    (def gelu_sum (scf.for {:result f32} c0 c3072 c1 (: 0.0 f32)
                      (region
                        (block [(: c index) (: acc f32)]
                          (def gv (memref.load gelu_out (: 0 index) c))
                          (scf.yield (arith.addf acc gv))))))
                    (print "  Sum of all gelu[0] values: %.6f\n" (arith.extf {:result f64} gelu_sum))
                    ;; Print gelu[500..509]
                    (def g500 (memref.load gelu_out (: 0 index) (: 500 index)))
                    (def g501 (memref.load gelu_out (: 0 index) (: 501 index)))
                    (def g502 (memref.load gelu_out (: 0 index) (: 502 index)))
                    (def g503 (memref.load gelu_out (: 0 index) (: 503 index)))
                    (def g504 (memref.load gelu_out (: 0 index) (: 504 index)))
                    (print "  gelu[500..504]: %.3f %.3f %.3f %.3f %.3f\n"
                           (arith.extf {:result f64} g500) (arith.extf {:result f64} g501)
                           (arith.extf {:result f64} g502) (arith.extf {:result f64} g503)
                           (arith.extf {:result f64} g504))
                    ;; Print gelu[2000..2004]
                    (def g2000 (memref.load gelu_out (: 0 index) (: 2000 index)))
                    (def g2001 (memref.load gelu_out (: 0 index) (: 2001 index)))
                    (def g2002 (memref.load gelu_out (: 0 index) (: 2002 index)))
                    (def g2003 (memref.load gelu_out (: 0 index) (: 2003 index)))
                    (def g2004 (memref.load gelu_out (: 0 index) (: 2004 index)))
                    (print "  gelu[2000..2004]: %.3f %.3f %.3f %.3f %.3f\n"
                           (arith.extf {:result f64} g2000) (arith.extf {:result f64} g2001)
                           (arith.extf {:result f64} g2002) (arith.extf {:result f64} g2003)
                           (arith.extf {:result f64} g2004))
                    ;; Print additional samples: gelu[505..509] and gelu[2005..2009]
                    (def g505 (memref.load gelu_out (: 0 index) (: 505 index)))
                    (def g506 (memref.load gelu_out (: 0 index) (: 506 index)))
                    (def g507 (memref.load gelu_out (: 0 index) (: 507 index)))
                    (def g508 (memref.load gelu_out (: 0 index) (: 508 index)))
                    (def g509 (memref.load gelu_out (: 0 index) (: 509 index)))
                    (print "  gelu[505..509]: %.3f %.3f %.3f %.3f %.3f\n"
                           (arith.extf {:result f64} g505) (arith.extf {:result f64} g506)
                           (arith.extf {:result f64} g507) (arith.extf {:result f64} g508)
                           (arith.extf {:result f64} g509))
                    (def g2005 (memref.load gelu_out (: 0 index) (: 2005 index)))
                    (def g2006 (memref.load gelu_out (: 0 index) (: 2006 index)))
                    (def g2007 (memref.load gelu_out (: 0 index) (: 2007 index)))
                    (def g2008 (memref.load gelu_out (: 0 index) (: 2008 index)))
                    (def g2009 (memref.load gelu_out (: 0 index) (: 2009 index)))
                    (print "  gelu[2005..2009]: %.3f %.3f %.3f %.3f %.3f\n"
                           (arith.extf {:result f64} g2005) (arith.extf {:result f64} g2006)
                           (arith.extf {:result f64} g2007) (arith.extf {:result f64} g2008)
                           (arith.extf {:result f64} g2009))
                    ;; Print partial sums: first half (0..1536) and second half (1536..3072)
                    (def c1536 (: 1536 index))
                    (def c3072_local (: 3072 index))
                    ;; Count iterations explicitly
                    (def count_half1 (scf.for {:result i64} c0 c1536 c1 (: 0 i64)
                      (region
                        (block [(: cc index) (: cnt i64)]
                          (scf.yield (arith.addi cnt (: 1 i64)))))))
                    (def count_half2 (scf.for {:result i64} c1536 c3072_local c1 (: 0 i64)
                      (region
                        (block [(: cc index) (: cnt i64)]
                          (scf.yield (arith.addi cnt (: 1 i64)))))))
                    (print "  loop counts: half1=%ld half2=%ld\n" count_half1 count_half2)
                    ;; Use f64 accumulation to rule out precision issues
                    (def gelu_sum_half1_f64 (scf.for {:result f64} c0 c1536 c1 (: 0.0 f64)
                      (region
                        (block [(: cc index) (: acc f64)]
                          (def gv (memref.load gelu_out (: 0 index) cc))
                          (def gv64 (arith.extf {:result f64} gv))
                          (scf.yield (arith.addf acc gv64))))))
                    (def gelu_sum_half2_f64 (scf.for {:result f64} c1536 c3072_local c1 (: 0.0 f64)
                      (region
                        (block [(: cc index) (: acc f64)]
                          (def gv (memref.load gelu_out (: 0 index) cc))
                          (def gv64 (arith.extf {:result f64} gv))
                          (scf.yield (arith.addf acc gv64))))))
                    (print "  gelu sum (f64 accum) half1: %.6f, half2: %.6f\n" gelu_sum_half1_f64 gelu_sum_half2_f64)
                    ;; Check memory layout: gelu[1][0] should be at offset 3072 from gelu[0][0]
                    (def g_row1_col0 (memref.load gelu_out (: 1 index) (: 0 index)))
                    (def fc_row1_col0 (memref.load fc_out (: 1 index) (: 0 index)))
                    (print "  Memory check: gelu[1][0]=%.6f fc[1][0]=%.6f (should be position 1's data)\n"
                           (arith.extf {:result f64} g_row1_col0)
                           (arith.extf {:result f64} fc_row1_col0))
                    ;; Print values at every 256 indices
                    (def g256 (memref.load gelu_out (: 0 index) (: 256 index)))
                    (def g512 (memref.load gelu_out (: 0 index) (: 512 index)))
                    (def g1024 (memref.load gelu_out (: 0 index) (: 1024 index)))
                    (def g1280 (memref.load gelu_out (: 0 index) (: 1280 index)))
                    (def g1792 (memref.load gelu_out (: 0 index) (: 1792 index)))
                    (def g2048 (memref.load gelu_out (: 0 index) (: 2048 index)))
                    (def g2560 (memref.load gelu_out (: 0 index) (: 2560 index)))
                    (def g2816 (memref.load gelu_out (: 0 index) (: 2816 index)))
                    (print "  gelu[@256 intervals]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} g256) (arith.extf {:result f64} g512)
                           (arith.extf {:result f64} g1024) (arith.extf {:result f64} g1280)
                           (arith.extf {:result f64} g1792) (arith.extf {:result f64} g2048)
                           (arith.extf {:result f64} g2560) (arith.extf {:result f64} g2816))
                    (def gelu_sum_half1 (scf.for {:result f32} c0 c1536 c1 (: 0.0 f32)
                      (region
                        (block [(: cc index) (: acc f32)]
                          (def gv (memref.load gelu_out (: 0 index) cc))
                          (scf.yield (arith.addf acc gv))))))
                    (def gelu_sum_half2 (scf.for {:result f32} c1536 c3072_local c1 (: 0.0 f32)
                      (region
                        (block [(: cc index) (: acc f32)]
                          (def gv (memref.load gelu_out (: 0 index) cc))
                          (scf.yield (arith.addf acc gv))))))
                    (print "  gelu sum half1[0..1536]: %.6f, half2[1536..3072]: %.6f\n"
                           (arith.extf {:result f64} gelu_sum_half1)
                           (arith.extf {:result f64} gelu_sum_half2))
                    ;; Weighted sum: sum(i * gelu[i])
                    (def weighted_sum (scf.for {:result f64} c0 c3072_local c1 (: 0.0 f64)
                      (region
                        (block [(: cc index) (: acc f64)]
                          (def cc_f64 (arith.uitofp {:result f64} (arith.index_cast {:result i64} cc)))
                          (def gv (memref.load gelu_out (: 0 index) cc))
                          (def gv64 (arith.extf {:result f64} gv))
                          (scf.yield (arith.addf acc (arith.mulf cc_f64 gv64)))))))
                    (print "  gelu weighted sum (i*v): %.6f\n" weighted_sum)
                    ;; Chunk sums for chunks 0, 10, 28, 42, 47 (to match C's pattern)
                    (def c64 (: 64 index))
                    (def c640 (: 640 index))
                    (def c704 (: 704 index))
                    (def c1792 (: 1792 index))
                    (def c1856 (: 1856 index))
                    (def c2688 (: 2688 index))
                    (def c2752 (: 2752 index))
                    (def c3008 (: 3008 index))
                    (def chunk0_sum (scf.for {:result f64} c0 c64 c1 (: 0.0 f64)
                      (region (block [(: i index) (: acc f64)]
                        (def gv (memref.load gelu_out (: 0 index) i))
                        (scf.yield (arith.addf acc (arith.extf {:result f64} gv)))))))
                    (def chunk10_sum (scf.for {:result f64} c640 c704 c1 (: 0.0 f64)
                      (region (block [(: i index) (: acc f64)]
                        (def gv (memref.load gelu_out (: 0 index) i))
                        (scf.yield (arith.addf acc (arith.extf {:result f64} gv)))))))
                    (def chunk28_sum (scf.for {:result f64} c1792 c1856 c1 (: 0.0 f64)
                      (region (block [(: i index) (: acc f64)]
                        (def gv (memref.load gelu_out (: 0 index) i))
                        (scf.yield (arith.addf acc (arith.extf {:result f64} gv)))))))
                    (def chunk42_sum (scf.for {:result f64} c2688 c2752 c1 (: 0.0 f64)
                      (region (block [(: i index) (: acc f64)]
                        (def gv (memref.load gelu_out (: 0 index) i))
                        (scf.yield (arith.addf acc (arith.extf {:result f64} gv)))))))
                    (def chunk47_sum (scf.for {:result f64} c3008 c3072_local c1 (: 0.0 f64)
                      (region (block [(: i index) (: acc f64)]
                        (def gv (memref.load gelu_out (: 0 index) i))
                        (scf.yield (arith.addf acc (arith.extf {:result f64} gv)))))))
                    (print "  gelu chunk sums [0,10,28,42,47]: %.1f %.1f %.1f %.1f %.1f\n"
                           chunk0_sum chunk10_sum chunk28_sum chunk42_sum chunk47_sum)
                    ;; Print detailed chunk 10 (640-655)
                    (def g640 (memref.load gelu_out (: 0 index) (: 640 index)))
                    (def g641 (memref.load gelu_out (: 0 index) (: 641 index)))
                    (def g642 (memref.load gelu_out (: 0 index) (: 642 index)))
                    (def g643 (memref.load gelu_out (: 0 index) (: 643 index)))
                    (def g644 (memref.load gelu_out (: 0 index) (: 644 index)))
                    (def g645 (memref.load gelu_out (: 0 index) (: 645 index)))
                    (def g646 (memref.load gelu_out (: 0 index) (: 646 index)))
                    (def g647 (memref.load gelu_out (: 0 index) (: 647 index)))
                    (print "  gelu[640..647]: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"
                           (arith.extf {:result f64} g640) (arith.extf {:result f64} g641)
                           (arith.extf {:result f64} g642) (arith.extf {:result f64} g643)
                           (arith.extf {:result f64} g644) (arith.extf {:result f64} g645)
                           (arith.extf {:result f64} g646) (arith.extf {:result f64} g647))
                    (def fc640 (memref.load fc_out (: 0 index) (: 640 index)))
                    (def fc641 (memref.load fc_out (: 0 index) (: 641 index)))
                    (def fc642 (memref.load fc_out (: 0 index) (: 642 index)))
                    (def fc643 (memref.load fc_out (: 0 index) (: 643 index)))
                    (def fc644 (memref.load fc_out (: 0 index) (: 644 index)))
                    (def fc645 (memref.load fc_out (: 0 index) (: 645 index)))
                    (def fc646 (memref.load fc_out (: 0 index) (: 646 index)))
                    (def fc647 (memref.load fc_out (: 0 index) (: 647 index)))
                    (print "  fc[640..647]: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"
                           (arith.extf {:result f64} fc640) (arith.extf {:result f64} fc641)
                           (arith.extf {:result f64} fc642) (arith.extf {:result f64} fc643)
                           (arith.extf {:result f64} fc644) (arith.extf {:result f64} fc645)
                           (arith.extf {:result f64} fc646) (arith.extf {:result f64} fc647))
                    ;; Print gelu at quarter marks: 768, 1536, 2304
                    (def g768 (memref.load gelu_out (: 0 index) (: 768 index)))
                    (def g1536 (memref.load gelu_out (: 0 index) (: 1536 index)))
                    (def g2304 (memref.load gelu_out (: 0 index) (: 2304 index)))
                    (def g3071 (memref.load gelu_out (: 0 index) (: 3071 index)))
                    (print "  gelu[768,1536,2304,3071]: %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} g768)
                           (arith.extf {:result f64} g1536)
                           (arith.extf {:result f64} g2304)
                           (arith.extf {:result f64} g3071))
                    ;; Also print fc at same positions for verification
                    (def fc768 (memref.load fc_out (: 0 index) (: 768 index)))
                    (def fc1536 (memref.load fc_out (: 0 index) (: 1536 index)))
                    (def fc2304 (memref.load fc_out (: 0 index) (: 2304 index)))
                    (def fc3071 (memref.load fc_out (: 0 index) (: 3071 index)))
                    (print "  fc[768,1536,2304,3071]: %.6f %.6f %.6f %.6f\n"
                           (arith.extf {:result f64} fc768)
                           (arith.extf {:result f64} fc1536)
                           (arith.extf {:result f64} fc2304)
                           (arith.extf {:result f64} fc3071))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              ;; Debug: layer 2 fc_proj output
              (scf.if do_layer2_debug
                (region
                  (block []
                    (def fcp0 (memref.load fc_proj_out (: 0 index) (: 0 index)))
                    (def fcp1 (memref.load fc_proj_out (: 0 index) (: 1 index)))
                    (def fcp2 (memref.load fc_proj_out (: 0 index) (: 2 index)))
                    (def fcp3 (memref.load fc_proj_out (: 0 index) (: 3 index)))
                    (def fcp4 (memref.load fc_proj_out (: 0 index) (: 4 index)))
                    (print "Layer 2 fc_proj output:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n"
                           (arith.extf {:result f64} fcp0)
                           (arith.extf {:result f64} fcp1)
                           (arith.extf {:result f64} fcp2)
                           (arith.extf {:result f64} fcp3)
                           (arith.extf {:result f64} fcp4))
                    (scf.yield)))
                (region (block [] (scf.yield))))

              (func.call "residual_add" x x2 fc_proj_out)

              ;; Debug: after each layer complete (for first 3 and last 2)
              (def is_layer1 (arith.cmpi {:predicate 0} layer c1))
              (def is_layer2 (arith.cmpi {:predicate 0} layer (: 2 index)))
              (def is_layer10 (arith.cmpi {:predicate 0} layer (: 10 index)))
              (def is_layer11 (arith.cmpi {:predicate 0} layer (: 11 index)))
              (def layer_debug (arith.ori (arith.ori (arith.ori (arith.ori is_layer0 is_layer1) is_layer2) is_layer10) is_layer11))
              (def do_layer_debug (arith.andi layer_debug is_first_step_embed))
              (scf.if do_layer_debug
                (region
                  (block []
                    (def l0 (memref.load x (: 0 index) (: 0 index)))
                    (def l1 (memref.load x (: 0 index) (: 1 index)))
                    (def l2 (memref.load x (: 0 index) (: 2 index)))
                    (def l3 (memref.load x (: 0 index) (: 3 index)))
                    (def l4 (memref.load x (: 0 index) (: 4 index)))
                    (def l496 (memref.load x (: 0 index) (: 496 index)))
                    (def layer_i32 (arith.index_cast {:result i32} layer))
                    (print "After layer %d:\n  [0..4]: %.4f %.4f %.4f %.4f %.4f\n  [496]: %.4f\n"
                           layer_i32
                           (arith.extf {:result f64} l0)
                           (arith.extf {:result f64} l1)
                           (arith.extf {:result f64} l2)
                           (arith.extf {:result f64} l3)
                           (arith.extf {:result f64} l4)
                           (arith.extf {:result f64} l496))
                    (scf.yield)))
                (region (block [] (scf.yield))))
              (scf.yield))))

        ;; 3. Final LayerNorm
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def w_ptr (ptr-at f32 params_ptr (arith.addi lnfw_base i_i64)))
              (def b_ptr (ptr-at f32 params_ptr (arith.addi lnfb_base i_i64)))
              (memref.store (llvm.load {:result f32} w_ptr) lnf_w i)
              (memref.store (llvm.load {:result f32} b_ptr) lnf_b i)
              (scf.yield))))

        ;; Debug: check values before final LN, especially position 496
        (def is_first_step2 (arith.cmpi {:predicate 0} step prompt_len))
        (scf.if is_first_step2
          (region
            (block []
              ;; Check x (input to final LN)
              (def x_pre0 (memref.load x (: 0 index) (: 0 index)))
              (def x_pre1 (memref.load x (: 0 index) (: 1 index)))
              (def x_pre496 (memref.load x (: 0 index) (: 496 index)))
              ;; Check lnf weights
              (def lnf_w0 (memref.load lnf_w (: 0 index)))
              (def lnf_w496 (memref.load lnf_w (: 496 index)))
              (def lnf_b496 (memref.load lnf_b (: 496 index)))
              (print "Before LN: x[0]=%.2f x[496]=%.2f, lnf_w[496]=%.2f lnf_b[496]=%.2f\n"
                     (arith.extf {:result f64} x_pre0)
                     (arith.extf {:result f64} x_pre496)
                     (arith.extf {:result f64} lnf_w496)
                     (arith.extf {:result f64} lnf_b496))
              ;; Check raw checkpoint values for lnf (use computed offsets)
              (def raw_lnfw0 (llvm.load {:result f32} (ptr-at f32 params_ptr lnfw_base)))
              (def raw_lnfw496 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi lnfw_base (: 496 i64)))))
              (def raw_lnfb496 (llvm.load {:result f32} (ptr-at f32 params_ptr (arith.addi lnfb_base (: 496 i64)))))
              (print "Raw checkpoint: lnfw[0]=%.4f lnfw[496]=%.4f lnfb[496]=%.4f\n"
                     (arith.extf {:result f64} raw_lnfw0)
                     (arith.extf {:result f64} raw_lnfw496)
                     (arith.extf {:result f64} raw_lnfb496))
              (scf.yield)))
          (region (block [] (scf.yield))))

        (func.call "layernorm_forward" x2 x lnf_w lnf_b)

        ;; 4. Compute logits at last filled position
        (func.call "logits_last_position" logits x2 wte_ptr logit_pos)

        ;; Debug: on first step, show hidden state and wte values
        (def is_first_step (arith.cmpi {:predicate 0} step prompt_len))  ; eq
        (scf.if is_first_step
          (region
            (block []
              ;; Print hidden state values (after final layernorm)
              (def h0 (memref.load x2 (: 0 index) (: 0 index)))
              (def h1 (memref.load x2 (: 0 index) (: 1 index)))
              (def h767 (memref.load x2 (: 0 index) (: 767 index)))

              ;; Compute sum of hidden state to see magnitude
              (def c0_idx (: 0 index))
              (def c1_idx (: 1 index))
              (def c768_idx (: 768 index))
              (def sum_h (scf.for {:result f32} c0_idx c768_idx c1_idx (: 0.0 f32)
                (region
                  (block [(: c index) (: acc f32)]
                    (def hv (memref.load x2 (: 0 index) c))
                    (scf.yield (arith.addf acc hv))))))

              ;; Check wte values for token 11 (comma)
              (def wte_11_0_ptr (ptr-at f32 wte_ptr (: 8448 i64)))  ; 11 * 768
              (def wte_11_0 (llvm.load {:result f32} wte_11_0_ptr))
              (def wte_11_1_ptr (ptr-at f32 wte_ptr (: 8449 i64)))
              (def wte_11_1 (llvm.load {:result f32} wte_11_1_ptr))

              (print "Hidden[0]: sum=%.2f, [0]=%.4f [1]=%.4f [767]=%.4f\n"
                     (arith.extf {:result f64} sum_h)
                     (arith.extf {:result f64} h0)
                     (arith.extf {:result f64} h1)
                     (arith.extf {:result f64} h767))
              (print "WTE[11]: [0]=%.4f [1]=%.4f\n"
                     (arith.extf {:result f64} wte_11_0)
                     (arith.extf {:result f64} wte_11_1))

              ;; Manually compute logit[11] = sum_c hidden[0,c] * wte[11,c]
              (def manual_logit (scf.for {:result f32} c0_idx c768_idx c1_idx (: 0.0 f32)
                (region
                  (block [(: c index) (: acc f32)]
                    (def hv (memref.load x2 (: 0 index) c))
                    (def c_i64 (arith.index_cast {:result i64} c))
                    (def wte_ptr_c (ptr-at f32 wte_ptr (arith.addi (: 8448 i64) c_i64)))
                    (def wte_v (llvm.load {:result f32} wte_ptr_c))
                    (scf.yield (arith.addf acc (arith.mulf hv wte_v)))))))

              ;; Compute sum of wte[11] embedding
              (def sum_wte11 (scf.for {:result f32} c0_idx c768_idx c1_idx (: 0.0 f32)
                (region
                  (block [(: c index) (: acc f32)]
                    (def c_i64 (arith.index_cast {:result i64} c))
                    (def wte_ptr_c (ptr-at f32 wte_ptr (arith.addi (: 8448 i64) c_i64)))
                    (def wte_v (llvm.load {:result f32} wte_ptr_c))
                    (scf.yield (arith.addf acc wte_v))))))

              ;; Compute sum of wte[464] embedding
              (def sum_wte464 (scf.for {:result f32} c0_idx c768_idx c1_idx (: 0.0 f32)
                (region
                  (block [(: c index) (: acc f32)]
                    (def c_i64 (arith.index_cast {:result i64} c))
                    (def wte_ptr_c (ptr-at f32 wte_ptr (arith.addi (: 356352 i64) c_i64)))  ; 464*768
                    (def wte_v (llvm.load {:result f32} wte_ptr_c))
                    (scf.yield (arith.addf acc wte_v))))))

              (print "Sum wte[11]=%.2f, Sum wte[464]=%.2f\n"
                     (arith.extf {:result f64} sum_wte11)
                     (arith.extf {:result f64} sum_wte464))

              ;; Find max absolute value in hidden state
              (def max_abs_h (scf.for {:result f32} c0_idx c768_idx c1_idx (: 0.0 f32)
                (region
                  (block [(: c index) (: best f32)]
                    (def hv (memref.load x2 (: 0 index) c))
                    (def abs_hv (arith.maximumf hv (arith.negf hv)))
                    (def new_best (arith.maximumf best abs_hv))
                    (scf.yield new_best)))))

              ;; Find first occurrence of max
              (def max_idx (scf.for {:result i32} c0_idx c768_idx c1_idx (: -1 i32)
                (region
                  (block [(: c index) (: found_idx i32)]
                    (def hv (memref.load x2 (: 0 index) c))
                    (def abs_hv (arith.maximumf hv (arith.negf hv)))
                    (def is_max (arith.cmpf {:predicate 1} abs_hv max_abs_h))  ; oeq
                    (def c_i32 (arith.index_cast {:result i32} c))
                    (def not_found (arith.cmpi {:predicate 2} found_idx (: 0 i32)))  ; slt
                    (def update (arith.andi is_max not_found))
                    (def new_idx (arith.select update c_i32 found_idx))
                    (scf.yield new_idx)))))
              (print "Max |hidden[c]| = %.2f at c=%d\n"
                     (arith.extf {:result f64} max_abs_h)
                     max_idx)

              ;; Check a few more positions in hidden state
              (def h100 (memref.load x2 (: 0 index) (: 100 index)))
              (def h200 (memref.load x2 (: 0 index) (: 200 index)))
              (def h500 (memref.load x2 (: 0 index) (: 500 index)))
              (print "Hidden samples: [100]=%.2f [200]=%.2f [500]=%.2f\n"
                     (arith.extf {:result f64} h100)
                     (arith.extf {:result f64} h200)
                     (arith.extf {:result f64} h500))

              ;; Check contribution from position 496 to logit[11]
              (def h496 (memref.load x2 (: 0 index) (: 496 index)))
              (def wte_11_496_ptr (ptr-at f32 wte_ptr (: 8944 i64)))  ; 11*768 + 496
              (def wte_11_496 (llvm.load {:result f32} wte_11_496_ptr))
              (def contrib_496 (arith.mulf h496 wte_11_496))
              (print "h[496]=%.2f, wte[11][496]=%.4f, contrib=%.2f\n"
                     (arith.extf {:result f64} h496)
                     (arith.extf {:result f64} wte_11_496)
                     (arith.extf {:result f64} contrib_496))

              ;; Print logits
              (def l11 (memref.load logits (: 11 index)))
              (def l464 (memref.load logits (: 464 index)))
              (print "logit[11]=%.2f, logit[464]=%.2f\n"
                     (arith.extf {:result f64} l11)
                     (arith.extf {:result f64} l464))
              (scf.yield)))
          (region (block [] (scf.yield))))

        ;; 5. Argmax to get next token
        (def next_token (func.call {:result i32} "argmax" logits))

        ;; 6. Decode and print
        (def token_str (func.call {:result !llvm.ptr} "tokenizer_decode" next_token))
        (print "%s" token_str)

        ;; 7. Store token at current step position
        (memref.store next_token token_ids step)

        (scf.yield))))

  (print "\n\nGeneration complete!\n")

  ;; Cleanup
  (memref.dealloc token_ids)
  (memref.dealloc x)
  (memref.dealloc x2)
  (memref.dealloc ln_out)
  (memref.dealloc qkv_out)
  (memref.dealloc attn_out)
  (memref.dealloc attn_proj_out)
  (memref.dealloc fc_out)
  (memref.dealloc gelu_out)
  (memref.dealloc fc_proj_out)
  (memref.dealloc logits)
  (memref.dealloc ln1_w)
  (memref.dealloc ln1_b)
  (memref.dealloc qkv_w)
  (memref.dealloc qkv_b)
  (memref.dealloc attn_w)
  (memref.dealloc attn_b)
  (memref.dealloc ln2_w)
  (memref.dealloc ln2_b)
  (memref.dealloc fc_w)
  (memref.dealloc fc_b)
  (memref.dealloc fcproj_w)
  (memref.dealloc fcproj_b)
  (memref.dealloc lnf_w)
  (memref.dealloc lnf_b)

  (call! free params_ptr)

  (func.return (: 0 i64)))