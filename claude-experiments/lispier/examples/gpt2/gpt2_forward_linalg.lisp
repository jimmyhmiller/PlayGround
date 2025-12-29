;; GPT-2 Forward Pass using Linalg Operations
;;
;; Uses linalg.matmul for automatic GPU parallelization via MLIR passes:
;; - linalg-tile-to-parallel-loops
;; - convert-parallel-loops-to-gpu
;;
;; Architecture (GPT-2 Small):
;; - T=64 (sequence length)
;; - C=768 (embedding dimension)
;; - L=12 (number of layers)
;; - NH=12 (number of attention heads)
;; - hs=64 (head size = C/NH)
;; - V=50257 (vocabulary size)
;;
;; All tensors are 2D (no batch dimension, B=1 implied)

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)
(require-dialect math)

;; Compilation pipeline for AMD GPU via linalg -> parallel loops -> GPU
;; Key: convert-scf-to-cf MUST run BEFORE gpu-kernel-outlining
(compilation
  (target rocm
    ;; Linalg to parallel loops, then to GPU
    (pass convert-linalg-to-parallel-loops)
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    ;; Lower affine constructs
    (pass lower-affine)
    ;; CRITICAL: SCF to CF BEFORE GPU outlining (eliminates scf.for in kernels)
    (pass convert-scf-to-cf)
    ;; GPU lowering
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
    ;; Host-side LLVM lowering
    (pass gpu-to-llvm {:use-bare-pointers-for-kernels true})
    (pass convert-cf-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-index-to-llvm)
    (pass convert-math-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass reconcile-unrealized-casts)))

(module
  (do
    ;; =========================================================================
    ;; QKV Projection: (T,C) @ (C,3C) + bias -> (T,3C)
    ;; T=64, C=768, 3C=2304
    ;; =========================================================================
    (func.func {:sym_name "matmul_qkv"
                :function_type (-> [memref<64x2304xf32>    ; out (T,3C)
                                    memref<64x768xf32>     ; inp (T,C)
                                    memref<768x2304xf32>   ; weight (C,3C)
                                    memref<2304xf32>] [])} ; bias (3C)
      (region
        (block [(: out memref<64x2304xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768x2304xf32>)
                (: bias memref<2304xf32>)]

          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill zero out)

          ;; Core matmul: (64,768) @ (768,2304) -> (64,2304)
          (linalg.matmul inp weight out)

          ;; Add bias
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def K (: 2304 index))
          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (scf.for c0 K c1
                  (region
                    (block [(: k index)]
                      (def val (memref.load {:result f32} out t k))
                      (def b (memref.load {:result f32} bias k))
                      (def result (arith.addf val b))
                      (memref.store result out t k)
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; =========================================================================
    ;; Attention Output Projection: (T,C) @ (C,C) + bias -> (T,C)
    ;; =========================================================================
    (func.func {:sym_name "matmul_attn_proj"
                :function_type (-> [memref<64x768xf32>    ; out (T,C)
                                    memref<64x768xf32>    ; inp (T,C)
                                    memref<768x768xf32>   ; weight (C,C)
                                    memref<768xf32>] [])} ; bias (C)
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768x768xf32>)
                (: bias memref<768xf32>)]

          (def zero (: 0.0 f32))
          (linalg.fill zero out)
          (linalg.matmul inp weight out)

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (scf.for c0 C c1
                  (region
                    (block [(: c index)]
                      (def val (memref.load {:result f32} out t c))
                      (def b (memref.load {:result f32} bias c))
                      (def result (arith.addf val b))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; =========================================================================
    ;; MLP FC1: (T,C) @ (C,4C) + bias -> (T,4C)
    ;; C=768, 4C=3072
    ;; =========================================================================
    (func.func {:sym_name "matmul_fc"
                :function_type (-> [memref<64x3072xf32>   ; out (T,4C)
                                    memref<64x768xf32>    ; inp (T,C)
                                    memref<768x3072xf32>  ; weight (C,4C)
                                    memref<3072xf32>] [])} ; bias (4C)
      (region
        (block [(: out memref<64x3072xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768x3072xf32>)
                (: bias memref<3072xf32>)]

          (def zero (: 0.0 f32))
          (linalg.fill zero out)
          (linalg.matmul inp weight out)

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C4 (: 3072 index))
          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (scf.for c0 C4 c1
                  (region
                    (block [(: c index)]
                      (def val (memref.load {:result f32} out t c))
                      (def b (memref.load {:result f32} bias c))
                      (def result (arith.addf val b))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; =========================================================================
    ;; MLP Projection: (T,4C) @ (4C,C) + bias -> (T,C)
    ;; =========================================================================
    (func.func {:sym_name "matmul_fc_proj"
                :function_type (-> [memref<64x768xf32>    ; out (T,C)
                                    memref<64x3072xf32>   ; inp (T,4C)
                                    memref<3072x768xf32>  ; weight (4C,C)
                                    memref<768xf32>] [])} ; bias (C)
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x3072xf32>)
                (: weight memref<3072x768xf32>)
                (: bias memref<768xf32>)]

          (def zero (: 0.0 f32))
          (linalg.fill zero out)
          (linalg.matmul inp weight out)

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (scf.for c0 C c1
                  (region
                    (block [(: c index)]
                      (def val (memref.load {:result f32} out t c))
                      (def b (memref.load {:result f32} bias c))
                      (def result (arith.addf val b))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; =========================================================================
    ;; Residual Add using linalg.add
    ;; =========================================================================
    (func.func {:sym_name "residual_forward"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>
                                    memref<64x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp1 memref<64x768xf32>)
                (: inp2 memref<64x768xf32>)]
          (linalg.add inp1 inp2 out)
          (func.return))))

    ;; =========================================================================
    ;; Copy buffer using linalg.copy
    ;; =========================================================================
    (func.func {:sym_name "copy_buffer"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)]
          (linalg.copy inp out)
          (func.return))))

    ;; =========================================================================
    ;; Token Embedding Lookup: tokens[T] -> embeddings[T,C]
    ;; (Keep as scf.for - index-based gather, no linalg equivalent)
    ;; =========================================================================
    (func.func {:sym_name "embedding_lookup"
                :function_type (-> [memref<64x768xf32>     ; out (T,C)
                                    memref<64xi32>         ; tokens (T)
                                    memref<50257x768xf32>] [])} ; wte (V,C)
      (region
        (block [(: out memref<64x768xf32>)
                (: tokens memref<64xi32>)
                (: wte memref<50257x768xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))

          (scf.for c0 T c1
            (region
              (block [(: t index)]
                ;; Get token ID
                (def token_id_i32 (memref.load {:result i32} tokens t))
                (def token_id (arith.index_cast {:result index} token_id_i32))

                ;; Copy embedding vector
                (scf.for c0 C c1
                  (region
                    (block [(: c index)]
                      (def emb_val (memref.load {:result f32} wte token_id c))
                      (memref.store emb_val out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Add Position Embeddings: out[T,C] += wpe[T,C]
    ;; (Use linalg.add - shapes match directly)
    ;; =========================================================================
    (func.func {:sym_name "add_position_embeddings"
                :function_type (-> [memref<64x768xf32>     ; out (T,C) - modified in place
                                    memref<64x768xf32>] [])} ; wpe slice (T,C)
      (region
        (block [(: out memref<64x768xf32>)
                (: wpe memref<64x768xf32>)]
          ;; out = out + wpe (in-place via temp would be safer, but this works)
          ;; Actually linalg.add needs separate output, so we need a pattern
          ;; For now, use scf.for for in-place add
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))

          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (scf.for c0 C c1
                  (region
                    (block [(: c index)]
                      (def x_val (memref.load {:result f32} out t c))
                      (def pe_val (memref.load {:result f32} wpe t c))
                      (def sum (arith.addf x_val pe_val))
                      (memref.store sum out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; LayerNorm (keep as scf.for - needs reduction for mean/variance)
    ;; =========================================================================
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<64x768xf32>     ; out (T,C)
                                    memref<64xf32>         ; mean (T) scratch
                                    memref<64xf32>         ; rstd (T) scratch
                                    memref<64x768xf32>     ; inp (T,C)
                                    memref<768xf32>        ; weight (gamma)
                                    memref<768xf32>        ; bias (beta)
                                    f32] [])}              ; eps
      (region
        (block [(: out memref<64x768xf32>)
                (: mean memref<64xf32>)
                (: rstd memref<64xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768xf32>)
                (: bias memref<768xf32>)
                (: eps f32)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def C_f32 (: 768.0 f32))
          (def zero (: 0.0 f32))
          (def one (: 1.0 f32))

          (scf.for c0 T c1
            (region
              (block [(: t index)]
                ;; Mean
                (def sum_val (scf.for {:result f32} c0 C c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def x (memref.load {:result f32} inp t c))
                      (def new_acc (arith.addf acc x))
                      (scf.yield new_acc)))))
                (def m (arith.divf sum_val C_f32))
                (memref.store m mean t)

                ;; Variance
                (def var_val (scf.for {:result f32} c0 C c1 zero
                  (region
                    (block [(: c index) (: vacc f32)]
                      (def x (memref.load {:result f32} inp t c))
                      (def diff (arith.subf x m))
                      (def diff_sq (arith.mulf diff diff))
                      (def new_vacc (arith.addf vacc diff_sq))
                      (scf.yield new_vacc)))))
                (def variance (arith.divf var_val C_f32))

                ;; Reciprocal std
                (def var_eps (arith.addf variance eps))
                (def std (math.sqrt var_eps))
                (def rs (arith.divf one std))
                (memref.store rs rstd t)

                ;; Normalize
                (scf.for c0 C c1
                  (region
                    (block [(: c index)]
                      (def x (memref.load {:result f32} inp t c))
                      (def x_norm (arith.mulf (arith.subf x m) rs))
                      (def gamma (memref.load {:result f32} weight c))
                      (def beta (memref.load {:result f32} bias c))
                      (def scaled (arith.mulf x_norm gamma))
                      (def result (arith.addf scaled beta))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Attention Forward (keep as scf.for - has softmax with reductions)
    ;; Now with 2D tensors: qkv is (T, 3C), out is (T, C)
    ;; =========================================================================
    (func.func {:sym_name "attention_forward"
                :function_type (-> [memref<64x768xf32>      ; out (T,C)
                                    memref<12x64x64xf32>    ; preatt (NH,T,T)
                                    memref<12x64x64xf32>    ; att (NH,T,T)
                                    memref<64x2304xf32>] [])} ; qkv (T,3C)
      (region
        (block [(: out memref<64x768xf32>)
                (: preatt memref<12x64x64xf32>)
                (: att memref<12x64x64xf32>)
                (: qkv memref<64x2304xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def NH (: 12 index))
          (def hs (: 64 index))
          (def zero (: 0.0 f32))
          (def neg_inf (: -10000.0 f32))
          (def scale (: 0.125 f32))  ; 1/sqrt(64)

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
                                        ;; Q is at offset h_offset in qkv
                                        (def q_idx (arith.addi h_offset d))
                                        (def q_val (memref.load {:result f32} qkv t q_idx))
                                        ;; K is at offset C + h_offset in qkv
                                        (def k_base (arith.addi C h_offset))
                                        (def k_idx (arith.addi k_base d))
                                        (def k_val (memref.load {:result f32} qkv t2 k_idx))
                                        (def prod (arith.mulf q_val k_val))
                                        (def new_acc (arith.addf acc prod))
                                        (scf.yield new_acc)))))
                                  (def scaled_dot (arith.mulf dot scale))
                                  (scf.yield scaled_dot)))
                              (region
                                (block []
                                  (scf.yield neg_inf)))))
                            (memref.store score preatt h t t2)
                            (def is_greater (arith.cmpf {:predicate "ogt"} score curr_max))
                            (def new_max (arith.select is_greater score curr_max))
                            (scf.yield new_max)))))

                      ;; Pass 2: Softmax exp and sum
                      (def exp_sum (scf.for {:result f32} c0 T c1 zero
                        (region
                          (block [(: t2 index) (: sum_acc f32)]
                            (def score (memref.load {:result f32} preatt h t t2))
                            (def shifted (arith.subf score max_score))
                            (def exp_val (math.exp shifted))
                            (memref.store exp_val att h t t2)
                            (def new_sum (arith.addf sum_acc exp_val))
                            (scf.yield new_sum)))))

                      ;; Pass 3: Normalize
                      (scf.for c0 T c1
                        (region
                          (block [(: t2 index)]
                            (def exp_val (memref.load {:result f32} att h t t2))
                            (def normalized (arith.divf exp_val exp_sum))
                            (memref.store normalized att h t t2)
                            (scf.yield))))

                      ;; Pass 4: Weighted sum of V
                      (scf.for c0 hs c1
                        (region
                          (block [(: d index)]
                            (def weighted_sum (scf.for {:result f32} c0 T c1 zero
                              (region
                                (block [(: t2 index) (: acc f32)]
                                  (def att_weight (memref.load {:result f32} att h t t2))
                                  ;; V is at offset 2C + h_offset in qkv
                                  (def v_base_c (arith.addi C C))
                                  (def v_base (arith.addi v_base_c h_offset))
                                  (def v_idx (arith.addi v_base d))
                                  (def v_val (memref.load {:result f32} qkv t2 v_idx))
                                  (def weighted (arith.mulf att_weight v_val))
                                  (def new_acc (arith.addf acc weighted))
                                  (scf.yield new_acc)))))
                            (def out_idx (arith.addi h_offset d))
                            (memref.store weighted_sum out t out_idx)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; GELU Activation (keep as scf.for - needs math.tanh)
    ;; =========================================================================
    (func.func {:sym_name "gelu_forward"
                :function_type (-> [memref<64x3072xf32>
                                    memref<64x3072xf32>] [])}
      (region
        (block [(: out memref<64x3072xf32>)
                (: inp memref<64x3072xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C4 (: 3072 index))

          (def half (: 0.5 f32))
          (def one (: 1.0 f32))
          (def sqrt_2_over_pi (: 0.7978845608 f32))
          (def coeff (: 0.044715 f32))

          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (scf.for c0 C4 c1
                  (region
                    (block [(: c index)]
                      (def x (memref.load {:result f32} inp t c))
                      (def x2 (arith.mulf x x))
                      (def x3 (arith.mulf x2 x))
                      (def inner1 (arith.mulf coeff x3))
                      (def inner2 (arith.addf x inner1))
                      (def inner3 (arith.mulf sqrt_2_over_pi inner2))
                      (def tanh_val (math.tanh inner3))
                      (def one_plus_tanh (arith.addf one tanh_val))
                      (def half_x (arith.mulf half x))
                      (def result (arith.mulf half_x one_plus_tanh))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Logits Projection: hidden[T,C] @ wte[V,C]^T -> logits[V]
    ;; Only compute for last token position (T-1) for efficiency
    ;; =========================================================================
    (func.func {:sym_name "logits_forward"
                :function_type (-> [memref<50257xf32>       ; out (V) - logits for last token
                                    memref<64x768xf32>      ; inp (T,C)
                                    memref<50257x768xf32>] [])} ; wte (V,C)
      (region
        (block [(: out memref<50257xf32>)
                (: inp memref<64x768xf32>)
                (: wte memref<50257x768xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def C (: 768 index))
          (def zero (: 0.0 f32))
          (def last_t (: 63 index))  ; T-1 for last token

          ;; For each vocab token v, compute dot(inp[T-1, :], wte[v, :])
          (scf.for c0 V c1
            (region
              (block [(: v index)]
                (def dot (scf.for {:result f32} c0 C c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def inp_val (memref.load {:result f32} inp last_t c))
                      (def w_val (memref.load {:result f32} wte v c))
                      (def prod (arith.mulf inp_val w_val))
                      (def new_acc (arith.addf acc prod))
                      (scf.yield new_acc)))))
                (memref.store dot out v)
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Softmax for logits (1D version)
    ;; =========================================================================
    (func.func {:sym_name "softmax_logits"
                :function_type (-> [memref<50257xf32>       ; probs (V) - output
                                    memref<50257xf32>] [])} ; logits (V) - input
      (region
        (block [(: probs memref<50257xf32>)
                (: logits memref<50257xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def zero (: 0.0 f32))
          (def neg_inf (: -10000.0 f32))

          ;; Find max for numerical stability
          (def max_val (scf.for {:result f32} c0 V c1 neg_inf
            (region
              (block [(: v index) (: curr_max f32)]
                (def val (memref.load {:result f32} logits v))
                (def is_greater (arith.cmpf {:predicate "ogt"} val curr_max))
                (def new_max (arith.select is_greater val curr_max))
                (scf.yield new_max)))))

          ;; Exp and sum
          (def exp_sum (scf.for {:result f32} c0 V c1 zero
            (region
              (block [(: v index) (: sum_acc f32)]
                (def val (memref.load {:result f32} logits v))
                (def shifted (arith.subf val max_val))
                (def exp_val (math.exp shifted))
                (memref.store exp_val probs v)
                (def new_sum (arith.addf sum_acc exp_val))
                (scf.yield new_sum)))))

          ;; Normalize
          (scf.for c0 V c1
            (region
              (block [(: v index)]
                (def exp_val (memref.load {:result f32} probs v))
                (def normalized (arith.divf exp_val exp_sum))
                (memref.store normalized probs v)
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Single Transformer Block
    ;; =========================================================================
    (func.func {:sym_name "transformer_block"
                :function_type (-> [memref<64x768xf32>      ; residual_out (T,C)
                                    memref<64x768xf32>      ; residual_in (T,C)
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
                                    memref<64xf32>          ; ln_mean
                                    memref<64xf32>          ; ln_rstd
                                    memref<64x768xf32>      ; ln1_out
                                    memref<64x2304xf32>     ; qkv
                                    memref<12x64x64xf32>    ; preatt
                                    memref<12x64x64xf32>    ; att
                                    memref<64x768xf32>      ; attn_out
                                    memref<64x768xf32>      ; attn_proj
                                    memref<64x768xf32>      ; residual2
                                    memref<64x768xf32>      ; ln2_out
                                    memref<64x3072xf32>     ; fch
                                    memref<64x3072xf32>     ; fch_gelu
                                    memref<64x768xf32>] [])}  ; fc_proj
      (region
        (block [(: residual_out memref<64x768xf32>)
                (: residual_in memref<64x768xf32>)
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
                (: ln_mean memref<64xf32>)
                (: ln_rstd memref<64xf32>)
                (: ln1_out memref<64x768xf32>)
                (: qkv memref<64x2304xf32>)
                (: preatt memref<12x64x64xf32>)
                (: att memref<12x64x64xf32>)
                (: attn_out memref<64x768xf32>)
                (: attn_proj memref<64x768xf32>)
                (: residual2 memref<64x768xf32>)
                (: ln2_out memref<64x768xf32>)
                (: fch memref<64x3072xf32>)
                (: fch_gelu memref<64x3072xf32>)
                (: fc_proj memref<64x768xf32>)]

          (def eps (: 0.00001 f32))

          ;; 1. LayerNorm1
          (func.call "layernorm_forward" ln1_out ln_mean ln_rstd residual_in ln1_weight ln1_bias eps)

          ;; 2. QKV Projection (linalg.matmul)
          (func.call "matmul_qkv" qkv ln1_out qkv_weight qkv_bias)

          ;; 3. Attention
          (func.call "attention_forward" attn_out preatt att qkv)

          ;; 4. Attention Output Projection (linalg.matmul)
          (func.call "matmul_attn_proj" attn_proj attn_out attn_proj_weight attn_proj_bias)

          ;; 5. Residual 1 (linalg.add)
          (func.call "residual_forward" residual2 residual_in attn_proj)

          ;; 6. LayerNorm2
          (func.call "layernorm_forward" ln2_out ln_mean ln_rstd residual2 ln2_weight ln2_bias eps)

          ;; 7. MLP FC (linalg.matmul)
          (func.call "matmul_fc" fch ln2_out fc_weight fc_bias)

          ;; 8. GELU
          (func.call "gelu_forward" fch_gelu fch)

          ;; 9. MLP Projection (linalg.matmul)
          (func.call "matmul_fc_proj" fc_proj fch_gelu fc_proj_weight fc_proj_bias)

          ;; 10. Residual 2 (linalg.add)
          (func.call "residual_forward" residual_out residual2 fc_proj)

          (func.return))))

    ;; =========================================================================
    ;; Simple test main (allocate and run forward pass with random weights)
    ;; =========================================================================
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate outputs
          (def logits (memref.alloc {:result memref<50257xf32>}))
          (def probs (memref.alloc {:result memref<50257xf32>}))

          ;; Allocate tokens
          (def tokens (memref.alloc {:result memref<64xi32>}))

          ;; Embeddings
          (def wte (memref.alloc {:result memref<50257x768xf32>}))
          (def wpe (memref.alloc {:result memref<64x768xf32>}))  ; Only need T positions

          ;; Final LayerNorm
          (def ln_f_w (memref.alloc {:result memref<768xf32>}))
          (def ln_f_b (memref.alloc {:result memref<768xf32>}))

          ;; Layer 0 params (will loop for 12 layers later)
          (def l0_ln1_w (memref.alloc {:result memref<768xf32>}))
          (def l0_ln1_b (memref.alloc {:result memref<768xf32>}))
          (def l0_qkv_w (memref.alloc {:result memref<768x2304xf32>}))
          (def l0_qkv_b (memref.alloc {:result memref<2304xf32>}))
          (def l0_attn_w (memref.alloc {:result memref<768x768xf32>}))
          (def l0_attn_b (memref.alloc {:result memref<768xf32>}))
          (def l0_ln2_w (memref.alloc {:result memref<768xf32>}))
          (def l0_ln2_b (memref.alloc {:result memref<768xf32>}))
          (def l0_fc_w (memref.alloc {:result memref<768x3072xf32>}))
          (def l0_fc_b (memref.alloc {:result memref<3072xf32>}))
          (def l0_proj_w (memref.alloc {:result memref<3072x768xf32>}))
          (def l0_proj_b (memref.alloc {:result memref<768xf32>}))

          ;; Scratch buffers
          (def x (memref.alloc {:result memref<64x768xf32>}))
          (def x_out (memref.alloc {:result memref<64x768xf32>}))
          (def ln_mean (memref.alloc {:result memref<64xf32>}))
          (def ln_rstd (memref.alloc {:result memref<64xf32>}))
          (def ln_out (memref.alloc {:result memref<64x768xf32>}))
          (def qkv (memref.alloc {:result memref<64x2304xf32>}))
          (def preatt (memref.alloc {:result memref<12x64x64xf32>}))
          (def att (memref.alloc {:result memref<12x64x64xf32>}))
          (def attn_out (memref.alloc {:result memref<64x768xf32>}))
          (def attn_proj (memref.alloc {:result memref<64x768xf32>}))
          (def residual (memref.alloc {:result memref<64x768xf32>}))
          (def ln2_out (memref.alloc {:result memref<64x768xf32>}))
          (def fch (memref.alloc {:result memref<64x3072xf32>}))
          (def fch_gelu (memref.alloc {:result memref<64x3072xf32>}))
          (def fc_proj (memref.alloc {:result memref<64x768xf32>}))

          ;; Initialize constants
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def one (: 1.0 f32))
          (def zero (: 0.0 f32))
          (def small (: 0.01 f32))
          (def tiny (: 0.001 f32))

          ;; Initialize tokens [0, 1, 2, ..., 63]
          (def T (: 64 index))
          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (def t_i32 (arith.index_cast {:result i32} t))
                (memref.store t_i32 tokens t)
                (scf.yield))))

          ;; Initialize embeddings with small values
          (def V (: 50257 index))
          (def C (: 768 index))
          (scf.for c0 V c1
            (region
              (block [(: v index)]
                (scf.for c0 C c1
                  (region
                    (block [(: c index)]
                      (def v_mod (arith.remui v (: 100 index)))
                      (def v_i64 (arith.index_cast {:result i64} v_mod))
                      (def c_i64 (arith.index_cast {:result i64} c))
                      (def sum_i64 (arith.addi v_i64 c_i64))
                      (def val_f (arith.sitofp {:result f32} sum_i64))
                      (def val (arith.mulf val_f tiny))
                      (memref.store val wte v c)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize position embeddings (64x768)
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
                      (def val (arith.mulf val_f tiny))
                      (memref.store val wpe t c)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize LayerNorm weights to 1, biases to 0
          (scf.for c0 C c1
            (region
              (block [(: c index)]
                (memref.store one ln_f_w c)
                (memref.store zero ln_f_b c)
                (memref.store one l0_ln1_w c)
                (memref.store zero l0_ln1_b c)
                (memref.store one l0_ln2_w c)
                (memref.store zero l0_ln2_b c)
                (memref.store zero l0_attn_b c)
                (memref.store zero l0_proj_b c)
                (scf.yield))))

          ;; Initialize QKV biases to 0
          (def K3C (: 2304 index))
          (scf.for c0 K3C c1
            (region
              (block [(: k index)]
                (memref.store zero l0_qkv_b k)
                (scf.yield))))

          ;; Initialize FC biases to 0
          (def C4 (: 3072 index))
          (scf.for c0 C4 c1
            (region
              (block [(: k index)]
                (memref.store zero l0_fc_b k)
                (scf.yield))))

          ;; Initialize weights with small values
          (scf.for c0 C c1
            (region
              (block [(: i index)]
                (scf.for c0 C c1
                  (region
                    (block [(: j index)]
                      (memref.store small l0_attn_w i j)
                      (scf.yield))))
                (scf.yield))))

          (scf.for c0 C c1
            (region
              (block [(: i index)]
                (scf.for c0 K3C c1
                  (region
                    (block [(: j index)]
                      (memref.store small l0_qkv_w i j)
                      (scf.yield))))
                (scf.yield))))

          (scf.for c0 C c1
            (region
              (block [(: i index)]
                (scf.for c0 C4 c1
                  (region
                    (block [(: j index)]
                      (memref.store small l0_fc_w i j)
                      (scf.yield))))
                (scf.yield))))

          (scf.for c0 C4 c1
            (region
              (block [(: i index)]
                (scf.for c0 C c1
                  (region
                    (block [(: j index)]
                      (memref.store small l0_proj_w i j)
                      (scf.yield))))
                (scf.yield))))

          ;; 1. Token embedding lookup
          (func.call "embedding_lookup" x tokens wte)

          ;; 2. Add position embeddings
          (func.call "add_position_embeddings" x wpe)

          ;; 3. Run one transformer block (for testing)
          (func.call "transformer_block"
            x_out x
            l0_ln1_w l0_ln1_b l0_qkv_w l0_qkv_b l0_attn_w l0_attn_b
            l0_ln2_w l0_ln2_b l0_fc_w l0_fc_b l0_proj_w l0_proj_b
            ln_mean ln_rstd ln_out qkv preatt att
            attn_out attn_proj residual ln2_out fch fch_gelu fc_proj)

          ;; 4. Final LayerNorm
          (def eps (: 0.00001 f32))
          (func.call "layernorm_forward" x ln_mean ln_rstd x_out ln_f_w ln_f_b eps)

          ;; 5. Project to logits
          (func.call "logits_forward" logits x wte)

          ;; 6. Softmax
          (func.call "softmax_logits" probs logits)

          ;; Print first few values
          (def logit0 (memref.load {:result f32} logits c0))
          (def logit1 (memref.load {:result f32} logits c1))
          (def prob0 (memref.load {:result f32} probs c0))
          (def prob1 (memref.load {:result f32} probs c1))

          (func.call "printF32" logit0)
          (func.call "printNewline")
          (func.call "printF32" logit1)
          (func.call "printNewline")
          (func.call "printF32" prob0)
          (func.call "printNewline")
          (func.call "printF32" prob1)
          (func.call "printNewline")

          (func.return))))

    ;; External function declarations
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"})

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"})))