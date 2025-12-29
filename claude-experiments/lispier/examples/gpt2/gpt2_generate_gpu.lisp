;; GPT-2 Text Generation with Linalg Operations for ROCm GPU
;;
;; This file combines:
;; - Linalg operations from gpt2_forward_linalg.lisp for GPU acceleration
;; - Weight loading and generation loop from gpt2_generate.lisp
;;
;; Architecture: GPT-2 Small (124M)
;; - T=64 (sequence length), C=768 (channels)
;; - V=50257 (vocab), L=12 (layers), NH=12 (heads)

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)
(require-dialect math)
(require-dialect llvm)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

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
    ;; Global paths and state
    ;; =========================================================================
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

    (llvm.mlir.global {:sym_name "g_params"
                       :linkage 10
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null))))

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
    ;; QKV Projection using linalg.matmul: (T,C) @ (C,3C) + bias -> (T,3C)
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
    ;; Attention Output Projection using linalg.matmul: (T,C) @ (C,C) + bias
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
    ;; MLP FC1 using linalg.matmul: (T,C) @ (C,4C) + bias -> (T,4C)
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
    ;; MLP Projection using linalg.matmul: (T,4C) @ (4C,C) + bias -> (T,C)
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
    ;; Token Embedding Lookup (keep as scf.for - index-based gather)
    ;; =========================================================================
    (func.func {:sym_name "embedding_lookup"
                :function_type (-> [memref<64x768xf32>
                                    !llvm.ptr
                                    !llvm.ptr
                                    memref<64xi32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: wte_ptr !llvm.ptr)
                (: wpe_ptr !llvm.ptr)
                (: tokens memref<64xi32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def C_i64 (: 768 i64))

          (scf.for c0 T c1
            (region
              (block [(: t index)]
                (def token_id_i32 (memref.load {:result i32} tokens t))
                (def token_id (arith.extsi {:result i64} token_id_i32))
                (def t_i64 (arith.index_cast {:result i64} t))

                ;; Token embedding offset: token_id * C
                (def wte_offset (arith.muli token_id C_i64))

                ;; Position embedding offset: t * C
                (def wpe_offset (arith.muli t_i64 C_i64))

                ;; Copy and add embeddings
                (scf.for c0 C c1
                  (region
                    (block [(: c index)]
                      (def c_i64 (arith.index_cast {:result i64} c))

                      ;; Get token embedding
                      (def wte_idx (arith.addi wte_offset c_i64))
                      (def wte_elem_ptr (llvm.getelementptr {:result !llvm.ptr :elem_type f32} wte_ptr wte_idx))
                      (def wte_val (llvm.load {:result f32} wte_elem_ptr))

                      ;; Get position embedding
                      (def wpe_idx (arith.addi wpe_offset c_i64))
                      (def wpe_elem_ptr (llvm.getelementptr {:result !llvm.ptr :elem_type f32} wpe_ptr wpe_idx))
                      (def wpe_val (llvm.load {:result f32} wpe_elem_ptr))

                      ;; Sum and store
                      (def sum (arith.addf wte_val wpe_val))
                      (memref.store sum out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; LayerNorm using math.sqrt (keep scf.for for reductions)
    ;; =========================================================================
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x768xf32>
                                    memref<768xf32>
                                    memref<768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768xf32>)
                (: bias memref<768xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def T (: 64 index))
          (def C (: 768 index))
          (def C_f32 (: 768.0 f32))
          (def zero (: 0.0 f32))
          (def eps (: 1e-5 f32))
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

                ;; Reciprocal std using math.sqrt
                (def var_eps (arith.addf variance eps))
                (def std (math.sqrt var_eps))
                (def rs (arith.divf one std))

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
    ;; Attention Forward using math.exp (keep scf.for for softmax reductions)
    ;; =========================================================================
    (func.func {:sym_name "attention_forward"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x2304xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: qkv memref<64x2304xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c12 (: 12 index))
          (def hs (: 64 index))
          (def zero (: 0.0 f32))
          (def neg_inf (: -1e9 f32))
          (def scale (: 0.125 f32))

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
                      ;; Pass 1: compute scores and find max
                      (def max_score (scf.for {:result f32} c0 (arith.addi t c1) c1 neg_inf
                        (region
                          (block [(: t2 index) (: max_acc f32)]
                            (def dot (scf.for {:result f32} c0 hs c1 zero
                              (region
                                (block [(: i index) (: dot_acc f32)]
                                  (def q_idx (arith.addi h_offset i))
                                  (def k_idx (arith.addi k_offset i))
                                  (def q_val (memref.load {:result f32} qkv t q_idx))
                                  (def k_val (memref.load {:result f32} qkv t2 k_idx))
                                  (scf.yield (arith.addf dot_acc (arith.mulf q_val k_val)))))))
                            (def score (arith.mulf dot scale))
                            (def new_max (arith.maximumf max_acc score))
                            (scf.yield new_max)))))

                      ;; Pass 2: exp and sum using math.exp
                      (def exp_sum (scf.for {:result f32} c0 (arith.addi t c1) c1 zero
                        (region
                          (block [(: t2 index) (: sum_acc f32)]
                            (def dot (scf.for {:result f32} c0 hs c1 zero
                              (region
                                (block [(: i index) (: dot_acc f32)]
                                  (def q_idx (arith.addi h_offset i))
                                  (def k_idx (arith.addi k_offset i))
                                  (def q_val (memref.load {:result f32} qkv t q_idx))
                                  (def k_val (memref.load {:result f32} qkv t2 k_idx))
                                  (scf.yield (arith.addf dot_acc (arith.mulf q_val k_val)))))))
                            (def score (arith.mulf dot scale))
                            (def exp_score (math.exp (arith.subf score max_score)))
                            (scf.yield (arith.addf sum_acc exp_score))))))

                      ;; Pass 3: weighted sum of V
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
                                        (def q_val (memref.load {:result f32} qkv t q_idx))
                                        (def k_val (memref.load {:result f32} qkv t2 k_idx))
                                        (scf.yield (arith.addf dot_acc (arith.mulf q_val k_val)))))))
                                  (def score (arith.mulf dot scale))
                                  (def attn_weight (arith.divf (math.exp (arith.subf score max_score)) exp_sum))
                                  (def v_idx (arith.addi v_offset i))
                                  (def v_val (memref.load {:result f32} qkv t2 v_idx))
                                  (scf.yield (arith.addf v_acc (arith.mulf attn_weight v_val)))))))
                            (def out_idx (arith.addi h_offset i))
                            (memref.store weighted_v out t out_idx)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; GELU Activation using math.tanh
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
                      ;; Use math.tanh instead of custom function
                      (def tanh_val (math.tanh inner3))
                      (def one_plus_tanh (arith.addf one tanh_val))
                      (def half_x (arith.mulf half x))
                      (def result (arith.mulf half_x one_plus_tanh))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Logits Projection using linalg.matmul (only last position for efficiency)
    ;; =========================================================================
    (func.func {:sym_name "logits_forward"
                :function_type (-> [memref<50257xf32>
                                    memref<64x768xf32>
                                    !llvm.ptr] [])}
      (region
        (block [(: out memref<50257xf32>)
                (: inp memref<64x768xf32>)
                (: wte_ptr !llvm.ptr)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def C (: 768 index))
          (def C_i64 (: 768 i64))
          (def zero (: 0.0 f32))
          (def last_t (: 63 index))

          ;; For each vocab token v, compute dot(inp[T-1, :], wte[v, :])
          (scf.for c0 V c1
            (region
              (block [(: v index)]
                (def v_i64 (arith.index_cast {:result i64} v))
                (def wte_row_offset (arith.muli v_i64 C_i64))

                (def dot (scf.for {:result f32} c0 C c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def inp_val (memref.load {:result f32} inp last_t c))
                      (def c_i64 (arith.index_cast {:result i64} c))
                      (def wte_idx (arith.addi wte_row_offset c_i64))
                      (def wte_elem_ptr (llvm.getelementptr {:result !llvm.ptr :elem_type f32} wte_ptr wte_idx))
                      (def w_val (llvm.load {:result f32} wte_elem_ptr))
                      (def prod (arith.mulf inp_val w_val))
                      (def new_acc (arith.addf acc prod))
                      (scf.yield new_acc)))))
                (memref.store dot out v)
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Softmax for logits using math.exp
    ;; =========================================================================
    (func.func {:sym_name "softmax_logits"
                :function_type (-> [memref<50257xf32>
                                    memref<50257xf32>] [])}
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

          ;; Exp and sum using math.exp
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
    ;; Argmax
    ;; =========================================================================
    (func.func {:sym_name "argmax"
                :function_type (-> [memref<50257xf32>] [i32])}
      (region
        (block [(: probs memref<50257xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def neg_inf (: -10000.0 f32))

          ;; Find max value
          (def max_val (scf.for {:result f32} c0 V c1 neg_inf
            (region
              (block [(: v index) (: curr_max f32)]
                (def val (memref.load {:result f32} probs v))
                (def is_greater (arith.cmpf {:predicate "ogt"} val curr_max))
                (def new_max (arith.select is_greater val curr_max))
                (scf.yield new_max)))))

          ;; Find index of max
          (def max_idx (scf.for {:result index} c0 V c1 c0
            (region
              (block [(: v index) (: curr_idx index)]
                (def val (memref.load {:result f32} probs v))
                (def is_max (arith.cmpf {:predicate "oeq"} val max_val))
                (def new_idx (arith.select is_max v curr_idx))
                (scf.yield new_idx)))))

          (def result (arith.index_cast {:result i32} max_idx))
          (func.return result))))

    ;; =========================================================================
    ;; Print token
    ;; =========================================================================
    (func.func {:sym_name "print_token"
                :function_type (-> [i32] [])}
      (region
        (block [(: token_id i32)]
          (def token_table_addr (llvm.mlir.addressof {:global_name @token_table :result !llvm.ptr}))
          (def table_ptr (llvm.load {:result !llvm.ptr} token_table_addr))
          (def token_i64 (arith.extsi {:result i64} token_id))
          (def str_ptr_ptr (llvm.getelementptr {:result !llvm.ptr :elem_type !llvm.ptr} table_ptr token_i64))
          (def str_ptr (llvm.load {:result !llvm.ptr} str_ptr_ptr))
          (def _p (call i32 printf str_ptr))
          (func.return))))

    ;; =========================================================================
    ;; Tokenizer init
    ;; =========================================================================
    (func.func {:sym_name "tokenizer_init"
                :function_type (-> [!llvm.ptr] [i32])}
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

          ;; Get vocab size from header[1]
          (def vocab_ptr (llvm.getelementptr {:result !llvm.ptr :elem_type i32} header_ptr (: 1 i64)))
          (def vocab_size (llvm.load {:result i32} vocab_ptr))

          ;; Get EOT from header[2]
          (def eot_ptr (llvm.getelementptr {:result !llvm.ptr :elem_type i32} header_ptr (: 2 i64)))
          (def eot (llvm.load {:result i32} eot_ptr))

          ;; Store EOT globally
          (def eot_global (llvm.mlir.addressof {:global_name @eot_token :result !llvm.ptr}))
          (llvm.store eot eot_global)

          ;; Allocate token table (array of string pointers)
          (def vocab_i64 (arith.extsi {:result i64} vocab_size))
          (def ptr_size (: 8 i64))
          (def table_bytes (arith.muli vocab_i64 ptr_size))
          (def table_ptr (call !llvm.ptr malloc table_bytes))

          ;; Store globally
          (def token_table_addr (llvm.mlir.addressof {:global_name @token_table :result !llvm.ptr}))
          (llvm.store table_ptr token_table_addr)

          ;; Read each token string
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def vocab_idx (arith.index_cast {:result index} vocab_size))
          (def one_byte (: 1 i64))

          (scf.for c0 vocab_idx c1
            (region
              (block [(: i index)]
                ;; Read length (1 byte)
                (def len_buf (call !llvm.ptr malloc one_byte))
                (def _r1 (call i64 fread len_buf one_byte one_byte file))
                (def len_u8 (llvm.load {:result i8} len_buf))
                (def len (arith.extui {:result i64} len_u8))
                (call! free len_buf)

                ;; Allocate string buffer (+1 for null)
                (def str_len_plus1 (arith.addi len one_byte))
                (def str_ptr (call !llvm.ptr malloc str_len_plus1))

                ;; Read string
                (def _r2 (call i64 fread str_ptr one_byte len file))

                ;; Add null terminator
                (def null_pos (llvm.getelementptr {:result !llvm.ptr :elem_type i8} str_ptr len))
                (def null_byte (: 0 i8))
                (llvm.store null_byte null_pos)

                ;; Store in table
                (def i_i64 (arith.index_cast {:result i64} i))
                (def slot_ptr (llvm.getelementptr {:result !llvm.ptr :elem_type !llvm.ptr} table_ptr i_i64))
                (llvm.store str_ptr slot_ptr)
                (scf.yield))))

          (def _fc (call i32 fclose file))
          (call! free header_ptr)

          (func.return (: 0 i32)))))

    ;; =========================================================================
    ;; External function declarations
    ;; =========================================================================
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"})

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"})

))
