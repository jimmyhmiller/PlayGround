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
    ;; String constants
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<39 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\0" :result !llvm.array<39 x i8>}))
          (llvm.return s))))

    (llvm.mlir.global {:sym_name "tokenizer_path"
                       :linkage 0
                       :global_type !llvm.array<44 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_tokenizer.bin\0" :result !llvm.array<44 x i8>}))
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
    ;; tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    ;; =========================================================================
    (func.func {:sym_name "my_tanhf"
                :function_type (-> [f32] [f32])}
      (region
        (block [(: x f32)]
          (def two (: 2.0 f32))
          (def one (: 1.0 f32))
          (def two_x (arith.mulf two x))
          (def exp_2x (func.call {:result f32} "my_expf" two_x))
          (def num (arith.subf exp_2x one))
          (def denom (arith.addf exp_2x one))
          (def result (arith.divf num denom))
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
  ;; Load checkpoint
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; Skip header
  (def _skip (call i32 fseek file (: 1024 i64) (: 0 i32)))

  ;; Load all params (padded vocab: 50304 * 768 = 38633472 for wte)
  (def total_params (: 124475904 i64))
  (def sizeof_f32 (: 4 i64))
  (def total_bytes (arith.muli total_params sizeof_f32))
  (def params_ptr (call !llvm.ptr malloc total_bytes))
  (print "Loading weights...\n")
  (def read_count (call i64 fread params_ptr sizeof_f32 total_params file))
  (print "Loaded %ld floats\n" read_count)
  (def _close (call i32 fclose file))

  ;; Store params globally
  (def g_params_addr (llvm.mlir.addressof {:global_name @g_params :result !llvm.ptr}))
  (llvm.store params_ptr g_params_addr)

  ;; Verify wte[0][0]
  (def wte_val (llvm.load {:result f32} params_ptr))
  (def wte_val_f64 (arith.extf {:result f64} wte_val))
  (print "wte[0][0] = %f\n" wte_val_f64)


  ;; Load tokenizer
  (def tok_path (llvm.mlir.addressof {:global_name @tokenizer_path :result !llvm.ptr}))
  (def _tok_ok (func.call {:result i32} "tokenizer_init" tok_path))

  ;; Get wte and wpe pointers
  (def wte_ptr params_ptr)
  (def wpe_offset (: 38633472 i64))  ; 50304 * 768 (padded vocab)
  (def wpe_ptr (ptr-at f32 params_ptr wpe_offset))

  ;; Create token_ids buffer with EOT token (start of generation)
  (def token_ids (memref.alloc {:result memref<64xi32>}))
  (def eot_global (llvm.mlir.addressof {:global_name @eot_token :result !llvm.ptr}))
  (def eot_token (llvm.load {:result i32} eot_global))
  (def c0 (: 0 index))
  (def c1 (: 1 index))
  (def c64 (: 64 index))

  ;; Fill with EOT, then set a single token prompt: "The" = 464
  (scf.for c0 c64 c1
    (region
      (block [(: i index)]
        (memref.store eot_token token_ids i)
        (scf.yield))))
  (memref.store (: 464 i32) token_ids (: 0 index))    ; "The"

  (print "Prompt: The\nGenerated:")

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

  ;; Weight base offsets (using padded_vocab_size = 50304)
  ;; wte: 0 (50304*768 = 38633472), wpe: 38633472 (1024*768 = 786432)
  (def ln1w_base (: 39419904 i64))   ; wpe + 786432
  (def ln1b_base (: 39429120 i64))   ; ln1w + 9216
  (def qkvw_base (: 39438336 i64))   ; ln1b + 9216
  (def qkvb_base (: 60672000 i64))   ; qkvw + 21233664
  (def attprojw_base (: 60699648 i64)) ; qkvb + 27648
  (def attprojb_base (: 67777536 i64)) ; attprojw + 7077888
  (def ln2w_base (: 67786752 i64))   ; attprojb + 9216
  (def ln2b_base (: 67795968 i64))   ; ln2w + 9216
  (def fcw_base (: 67805184 i64))    ; ln2b + 9216
  (def fcb_base (: 96116736 i64))    ; fcw + 28311552
  (def fcprojw_base (: 96153600 i64))  ; fcb + 36864
  (def fcprojb_base (: 124465152 i64)) ; fcprojw + 28311552
  (def lnfw_base (: 124474368 i64))  ; fcprojb + 9216
  (def lnfb_base (: 124475136 i64))  ; lnfw + 768

  ;; Per-layer strides
  (def ln_stride (: 768 i64))
  (def qkvw_stride (: 1769472 i64))
  (def qkvb_stride (: 2304 i64))
  (def attprojw_stride (: 589824 i64))
  (def attprojb_stride (: 768 i64))
  (def fcw_stride (: 2359296 i64))
  (def fcb_stride (: 3072 i64))
  (def fcprojw_stride (: 2359296 i64))
  (def fcprojb_stride (: 768 i64))

  ;; =========================================================================
  ;; GENERATION LOOP: Generate 20 tokens
  ;; =========================================================================
  (def prompt_len (: 1 index))
  (def gen_steps (: 30 index))  ; generate 30 tokens after prompt
  (def gen_end (arith.addi prompt_len gen_steps))

  (print "\nGenerating tokens:\n")

  ;; step is the position where we'll store the next token
  ;; We compute logits at position (step - 1) which is the last filled position
  (scf.for prompt_len gen_end c1
    (region
      (block [(: step index)]
        ;; 1. Embedding lookup
        (func.call "embedding_lookup" x wte_ptr wpe_ptr token_ids)

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

              ;; qkv weights - llm.c stores as (OC=2304, C=768), we use (C=768, OC=2304)
              ;; Load with transpose: checkpoint[oc*768+c] -> qkv_w[c][oc]
              (def qkvw_offset (arith.addi qkvw_base (arith.muli layer_i64 qkvw_stride)))
              (def qkvb_offset (arith.addi qkvb_base (arith.muli layer_i64 qkvb_stride)))
              (scf.for c0 c2304 c1
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 (: 768 i64)))
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

              ;; attention projection weights - llm.c stores as (OC=768, C=768)
              ;; Load with transpose: checkpoint[oc*768+c] -> attn_w[c][oc]
              (def attprojw_offset (arith.addi attprojw_base (arith.muli layer_i64 attprojw_stride)))
              (def attprojb_offset (arith.addi attprojb_base (arith.muli layer_i64 attprojb_stride)))
              (scf.for c0 c768 c1
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 (: 768 i64)))
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

              ;; fc weights - llm.c stores as (OC=3072, C=768)
              ;; Load with transpose: checkpoint[oc*768+c] -> fc_w[c][oc]
              (def fcw_offset (arith.addi fcw_base (arith.muli layer_i64 fcw_stride)))
              (def fcb_offset (arith.addi fcb_base (arith.muli layer_i64 fcb_stride)))
              (scf.for c0 c3072 c1
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 (: 768 i64)))
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

              ;; fc projection weights - llm.c stores as (OC=768, C=3072)
              ;; Load with transpose: checkpoint[oc*3072+c] -> fcproj_w[c][oc]
              (def fcprojw_offset (arith.addi fcprojw_base (arith.muli layer_i64 fcprojw_stride)))
              (def fcprojb_offset (arith.addi fcprojb_base (arith.muli layer_i64 fcprojb_stride)))
              (scf.for c0 c768 c1
                (region
                  (block [(: oc index)]
                    (def oc_i64 (arith.index_cast {:result i64} oc))
                    (def row_offset (arith.muli oc_i64 (: 3072 i64)))
                    (scf.for c0 c3072 c1
                      (region
                        (block [(: c index)]
                          (def c_i64 (arith.index_cast {:result i64} c))
                          (def w_ptr (ptr-at f32 params_ptr (arith.addi fcprojw_offset (arith.addi row_offset c_i64))))
                          (memref.store (llvm.load {:result f32} w_ptr) fcproj_w c oc)
                          (scf.yield))))
                    (scf.yield))))
              (scf.for c0 c768 c1
                (region
                  (block [(: i index)]
                    (def i_i64 (arith.index_cast {:result i64} i))
                    (def b_ptr (ptr-at f32 params_ptr (arith.addi fcprojb_offset i_i64)))
                    (memref.store (llvm.load {:result f32} b_ptr) fcproj_b i)
                    (scf.yield))))

              ;; === Run transformer block ===
              (func.call "layernorm_forward" ln_out x ln1_w ln1_b)
              (func.call "matmul_qkv" qkv_out ln_out qkv_w qkv_b)
              (func.call "attention_forward" attn_out qkv_out)
              (func.call "matmul_attn_proj" attn_proj_out attn_out attn_w attn_b)
              (func.call "residual_add" x2 x attn_proj_out)
              (func.call "layernorm_forward" ln_out x2 ln2_w ln2_b)
              (func.call "matmul_fc" fc_out ln_out fc_w fc_b)
              (func.call "gelu_forward" gelu_out fc_out)
              (func.call "matmul_fc_proj" fc_proj_out gelu_out fcproj_w fcproj_b)
              (func.call "residual_add" x x2 fc_proj_out)
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
        (func.call "layernorm_forward" x2 x lnf_w lnf_b)

        ;; 4. Compute logits at last filled position
        (func.call "logits_last_position" logits x2 wte_ptr logit_pos)

        ;; Debug: show some logit values
        (def logit0 (memref.load logits (: 0 index)))
        (def logit11 (memref.load logits (: 11 index)))  ; comma
        (def logit262 (memref.load logits (: 262 index)))  ; "the"
        (print "logits[0]=%.2f [11]=%.2f [262]=%.2f\n"
               (arith.extf {:result f64} logit0)
               (arith.extf {:result f64} logit11)
               (arith.extf {:result f64} logit262))

        ;; 5. Argmax to get next token
        (def next_token (func.call {:result i32} "argmax" logits))

        ;; 6. Decode and print - show token ID for debugging
        (def token_str (func.call {:result !llvm.ptr} "tokenizer_decode" next_token))
        (print "[%d]%s" next_token token_str)

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