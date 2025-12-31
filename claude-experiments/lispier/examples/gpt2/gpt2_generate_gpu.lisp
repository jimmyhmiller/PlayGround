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

;; NOTE: Using func.func external declarations instead of extern-fn for non-vararg functions
;; because extern-fn + GPU pipeline causes llvm-request-c-wrappers to crash
;; For printf (vararg), we use extern-fn which generates llvm.func (not affected by the pass)
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; Compilation pipeline for AMD GPU via linalg -> parallel loops -> GPU
(compilation
  (target rocm
    ;; Linalg to parallel loops, then to GPU
    (pass convert-linalg-to-parallel-loops)
    ;; Tile parallel loops to create block/thread structure (16x16 = 256 threads per block)
    (pass scf-parallel-loop-tiling {:parallel-loop-tile-sizes "16,16"})
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    ;; Lower affine constructs
    (pass lower-affine)
    ;; CRITICAL: SCF to CF BEFORE GPU outlining
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
    ;; External function declarations (using func.func instead of extern-fn)
    ;; =========================================================================
    (func.func {:sym_name "malloc"
                :function_type (-> [i64] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "free"
                :function_type (-> [!llvm.ptr] [])
                :sym_visibility "private"})

    (func.func {:sym_name "fopen"
                :function_type (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr])
                :sym_visibility "private"})

    (func.func {:sym_name "fread"
                :function_type (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64])
                :sym_visibility "private"})

    (func.func {:sym_name "fseek"
                :function_type (-> [!llvm.ptr i64 i32] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "fclose"
                :function_type (-> [!llvm.ptr] [i32])
                :sym_visibility "private"})

    (func.func {:sym_name "clock_ms"
                :function_type (-> [] [i64])
                :sym_visibility "private"})

    ;; printf is variadic - use extern-fn which generates llvm.func (not affected by llvm-request-c-wrappers)
    ;; This is declared outside module using extern-fn macro

    ;; =========================================================================
    ;; Global paths and state
    ;; =========================================================================
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
                      (def wte_elem_ptr (ptr-at f32 wte_ptr wte_idx))
                      (def wte_val (llvm.load {:result f32} wte_elem_ptr))

                      ;; Get position embedding
                      (def wpe_idx (arith.addi wpe_offset c_i64))
                      (def wpe_elem_ptr (ptr-at f32 wpe_ptr wpe_idx))
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
    ;; Reshape QKV from (T=64, 3C=2304) to separate Q/K/V as (NH=12, T=64, hs=64)
    ;; =========================================================================
    (func.func {:sym_name "reshape_qkv_to_batched"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<64x2304xf32>] [])}
      (region
        (block [(: Q_out memref<12x64x64xf32>)
                (: K_out memref<12x64x64xf32>)
                (: V_out memref<12x64x64xf32>)
                (: qkv memref<64x2304xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c12 (: 12 index))
          (def c64 (: 64 index))
          (def hs (: 64 index))
          (def c768 (: 768 index))

          ;; For each head
          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                (def h_offset (arith.muli h hs))
                (def k_base (arith.addi h_offset c768))
                (def v_base (arith.addi k_base c768))

                ;; For each time position
                (scf.for c0 c64 c1
                  (region
                    (block [(: t index)]
                      ;; For each head dimension
                      (scf.for c0 hs c1
                        (region
                          (block [(: i index)]
                            ;; Q[h,t,i] = qkv[t, h*hs + i]
                            (def q_src_idx (arith.addi h_offset i))
                            (def q_val (memref.load {:result f32} qkv t q_src_idx))
                            (memref.store q_val Q_out h t i)

                            ;; K[h,t,i] = qkv[t, 768 + h*hs + i]
                            (def k_src_idx (arith.addi k_base i))
                            (def k_val (memref.load {:result f32} qkv t k_src_idx))
                            (memref.store k_val K_out h t i)

                            ;; V[h,t,i] = qkv[t, 1536 + h*hs + i]
                            (def v_src_idx (arith.addi v_base i))
                            (def v_val (memref.load {:result f32} qkv t v_src_idx))
                            (memref.store v_val V_out h t i)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Transpose K from (NH, T, hs) to (NH, hs, T) for K^T operation
    ;; =========================================================================
    (func.func {:sym_name "transpose_k_for_attention"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: K_t memref<12x64x64xf32>)
                (: K memref<12x64x64xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c12 (: 12 index))
          (def c64 (: 64 index))

          ;; For each head
          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                ;; For each time position (becomes column in transposed)
                (scf.for c0 c64 c1
                  (region
                    (block [(: t index)]
                      ;; For each head dimension (becomes row in transposed)
                      (scf.for c0 c64 c1
                        (region
                          (block [(: i index)]
                            ;; K_t[h, i, t] = K[h, t, i]
                            (def val (memref.load {:result f32} K h t i))
                            (memref.store val K_t h i t)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Batched Q @ K^T using linalg.batch_matmul + scaling
    ;; Q: (12, 64, 64), K_t: (12, 64, 64) -> scores: (12, 64, 64)
    ;; =========================================================================
    (func.func {:sym_name "batched_qk_matmul"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: scores memref<12x64x64xf32>)
                (: Q memref<12x64x64xf32>)
                (: K_t memref<12x64x64xf32>)]

          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill zero scores)

          ;; Batched matmul: for each batch h, compute scores[h] = Q[h] @ K_t[h]
          (linalg.batch_matmul Q K_t scores)

          ;; Apply scaling (1/sqrt(64) = 0.125)
          (def scale (: 0.125 f32))
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c12 (: 12 index))
          (def c64 (: 64 index))

          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                (scf.for c0 c64 c1
                  (region
                    (block [(: t1 index)]
                      (scf.for c0 c64 c1
                        (region
                          (block [(: t2 index)]
                            (def val (memref.load {:result f32} scores h t1 t2))
                            (def scaled (arith.mulf val scale))
                            (memref.store scaled scores h t1 t2)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))

          (func.return))))

    ;; =========================================================================
    ;; Causal Softmax: Apply causal mask and softmax to attention scores
    ;; scores: (12, 64, 64), weights_out: (12, 64, 64)
    ;; =========================================================================
    (func.func {:sym_name "causal_softmax"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: weights memref<12x64x64xf32>)
                (: scores memref<12x64x64xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c12 (: 12 index))
          (def c64 (: 64 index))
          (def zero (: 0.0 f32))
          (def neg_inf (: -1e9 f32))

          ;; For each head
          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                ;; For each query position
                (scf.for c0 c64 c1
                  (region
                    (block [(: t index)]
                      ;; Causal mask: only attend to positions <= t
                      (def valid_len (arith.addi t c1))

                      ;; Pass 1: Find max for numerical stability (only valid positions)
                      (def max_score (scf.for {:result f32} c0 valid_len c1 neg_inf
                        (region
                          (block [(: t2 index) (: max_acc f32)]
                            (def score (memref.load {:result f32} scores h t t2))
                            (def new_max (arith.maximumf max_acc score))
                            (scf.yield new_max)))))

                      ;; Pass 2: Compute exp(score - max) and sum
                      (def exp_sum (scf.for {:result f32} c0 valid_len c1 zero
                        (region
                          (block [(: t2 index) (: sum_acc f32)]
                            (def score (memref.load {:result f32} scores h t t2))
                            (def shifted (arith.subf score max_score))
                            (def exp_val (math.exp shifted))
                            ;; Store intermediate exp value
                            (memref.store exp_val weights h t t2)
                            (def new_sum (arith.addf sum_acc exp_val))
                            (scf.yield new_sum)))))

                      ;; Pass 3: Normalize by sum (valid positions)
                      (scf.for c0 valid_len c1
                        (region
                          (block [(: t2 index)]
                            (def exp_val (memref.load {:result f32} weights h t t2))
                            (def normalized (arith.divf exp_val exp_sum))
                            (memref.store normalized weights h t t2)
                            (scf.yield))))

                      ;; Zero out invalid positions (t2 > t) for causal mask
                      (scf.for valid_len c64 c1
                        (region
                          (block [(: t2 index)]
                            (memref.store zero weights h t t2)
                            (scf.yield))))

                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Batched Attention @ V using linalg.batch_matmul
    ;; weights: (12, 64, 64), V: (12, 64, 64) -> out: (12, 64, 64)
    ;; =========================================================================
    (func.func {:sym_name "batched_attn_v_matmul"
                :function_type (-> [memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: out memref<12x64x64xf32>)
                (: weights memref<12x64x64xf32>)
                (: V memref<12x64x64xf32>)]

          ;; Zero output for accumulation
          (def zero (: 0.0 f32))
          (linalg.fill zero out)

          ;; Batched matmul: out[h] = weights[h] @ V[h]
          (linalg.batch_matmul weights V out)

          (func.return))))

    ;; =========================================================================
    ;; Reshape attention output from (NH=12, T=64, hs=64) back to (T=64, C=768)
    ;; =========================================================================
    (func.func {:sym_name "reshape_attn_output"
                :function_type (-> [memref<64x768xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: attn_values memref<12x64x64xf32>)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c12 (: 12 index))
          (def c64 (: 64 index))
          (def hs (: 64 index))

          ;; For each time position
          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                ;; For each head
                (scf.for c0 c12 c1
                  (region
                    (block [(: h index)]
                      (def h_offset (arith.muli h hs))
                      ;; For each head dimension
                      (scf.for c0 hs c1
                        (region
                          (block [(: i index)]
                            ;; out[t, h*hs + i] = attn_values[h, t, i]
                            (def out_idx (arith.addi h_offset i))
                            (def val (memref.load {:result f32} attn_values h t i))
                            (memref.store val out t out_idx)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Attention Forward - Orchestrates batched GPU attention
    ;; =========================================================================
    (func.func {:sym_name "attention_forward"
                :function_type (-> [memref<64x768xf32>
                                    memref<64x2304xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>
                                    memref<12x64x64xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>)
                (: qkv memref<64x2304xf32>)
                (: Q memref<12x64x64xf32>)
                (: K memref<12x64x64xf32>)
                (: V memref<12x64x64xf32>)
                (: K_t memref<12x64x64xf32>)
                (: scores memref<12x64x64xf32>)
                (: weights memref<12x64x64xf32>)
                (: values memref<12x64x64xf32>)]

          ;; Step 1: Reshape QKV to batched format
          (func.call "reshape_qkv_to_batched" Q K V qkv)

          ;; Step 2: Transpose K for K^T
          (func.call "transpose_k_for_attention" K_t K)

          ;; Step 3: Q @ K^T -> scores (GPU via linalg.batch_matmul)
          (func.call "batched_qk_matmul" scores Q K_t)

          ;; Step 4: Causal softmax on scores -> weights
          (func.call "causal_softmax" weights scores)

          ;; Step 5: weights @ V -> values (GPU via linalg.batch_matmul)
          (func.call "batched_attn_v_matmul" values weights V)

          ;; Step 6: Reshape output back to (T, C) format
          (func.call "reshape_attn_output" out values)

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
                                    !llvm.ptr
                                    index] [])}
      (region
        (block [(: out memref<50257xf32>)
                (: inp memref<64x768xf32>)
                (: wte_ptr !llvm.ptr)
                (: pos index)]

          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def V (: 50257 index))
          (def C (: 768 index))
          (def C_i64 (: 768 i64))
          (def zero (: 0.0 f32))

          ;; For each vocab token v, compute dot(inp[pos, :], wte[v, :])
          (scf.for c0 V c1
            (region
              (block [(: v index)]
                (def v_i64 (arith.index_cast {:result i64} v))
                (def wte_row_offset (arith.muli v_i64 C_i64))

                (def dot (scf.for {:result f32} c0 C c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def inp_val (memref.load {:result f32} inp pos c))
                      (def c_i64 (arith.index_cast {:result i64} c))
                      (def wte_idx (arith.addi wte_row_offset c_i64))
                      (def wte_elem_ptr (ptr-at f32 wte_ptr wte_idx))
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
          (def str_ptr_ptr (ptr-at !llvm.ptr table_ptr token_i64))
          (def str_ptr (llvm.load {:result !llvm.ptr} str_ptr_ptr))
          ;; Use llvm.call directly for variadic printf (can't use func.call with llvm.func)
          (def _p (llvm.call {:callee @printf
                              :var_callee_type !llvm.func<i32 (ptr, ...)>
                              :result i32}
                    str_ptr))
          (func.return))))

    ;; =========================================================================
    ;; Tokenizer init
    ;; =========================================================================
    (func.func {:sym_name "tokenizer_init"
                :function_type (-> [!llvm.ptr] [i32])}
      (region
        (block [(: path !llvm.ptr)]
          (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
          (def file (func.call {:result !llvm.ptr} "fopen" path mode))

          ;; Read header (256 * 4 = 1024 bytes)
          (def header_size (: 1024 i64))
          (def header_ptr (func.call {:result !llvm.ptr} "malloc" header_size))
          (def _read_h (func.call {:result i64} "fread" header_ptr (: 4 i64) (: 256 i64) file))

          ;; Check magic (should be 20240328)
          (def magic (llvm.load {:result i32} header_ptr))

          ;; Get vocab size from header[2] (header[1] is version)
          (def vocab_ptr (ptr-at i32 header_ptr (: 2 i64)))
          (def vocab_size (llvm.load {:result i32} vocab_ptr))

          ;; Get EOT from header[3]
          (def eot_ptr (ptr-at i32 header_ptr (: 3 i64)))
          (def eot (llvm.load {:result i32} eot_ptr))

          ;; Store EOT globally
          (def eot_global (llvm.mlir.addressof {:global_name @eot_token :result !llvm.ptr}))
          (llvm.store eot eot_global)

          ;; Allocate token table (array of string pointers)
          (def vocab_i64 (arith.extsi {:result i64} vocab_size))
          (def ptr_size (: 8 i64))
          (def table_bytes (arith.muli vocab_i64 ptr_size))
          (def table_ptr (func.call {:result !llvm.ptr} "malloc" table_bytes))

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
                (def len_buf (func.call {:result !llvm.ptr} "malloc" one_byte))
                (def _r1 (func.call {:result i64} "fread" len_buf one_byte one_byte file))
                (def len_u8 (llvm.load {:result i8} len_buf))
                (def len (arith.extui {:result i64} len_u8))
                (func.call "free" len_buf)

                ;; Allocate string buffer (+1 for null)
                (def str_len_plus1 (arith.addi len one_byte))
                (def str_ptr (func.call {:result !llvm.ptr} "malloc" str_len_plus1))

                ;; Read string
                (def _r2 (func.call {:result i64} "fread" str_ptr one_byte len file))

                ;; Add null terminator
                (def null_pos (ptr-at i8 str_ptr len))
                (def null_byte (: 0 i8))
                (llvm.store null_byte null_pos)

                ;; Store in table
                (def i_i64 (arith.index_cast {:result i64} i))
                (def slot_ptr (ptr-at !llvm.ptr table_ptr i_i64))
                (llvm.store str_ptr slot_ptr)
                (scf.yield))))

          (def _fc (func.call {:result i32} "fclose" file))
          (func.call "free" header_ptr)

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

    ;; =========================================================================
    ;; Main function with weight loading and generation loop
    ;; =========================================================================
    (defn main []
      ;; Load checkpoint
      (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
      (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
      (def file (func.call {:result !llvm.ptr} "fopen" path mode))
    
      ;; Read checkpoint header (256 ints) to get model config
      (def header_buf (func.call {:result !llvm.ptr} "malloc" (: 1024 i64)))
      (def _read_header (func.call {:result i64} "fread" header_buf (: 4 i64) (: 256 i64) file))
    
      ;; Extract model config from header
      (def magic_ptr header_buf)
      (def _magic (llvm.load {:result i32} magic_ptr))
    
      (def maxT_ptr (ptr-at i32 header_buf (: 2 i64)))
      (def maxT_i32 (llvm.load {:result i32} maxT_ptr))
      (def maxT (arith.extsi {:result i64} maxT_i32))
    
      (def V_ptr (ptr-at i32 header_buf (: 3 i64)))
      (def V_i32 (llvm.load {:result i32} V_ptr))
      (def V (arith.extsi {:result i64} V_i32))
    
      (def L_ptr (ptr-at i32 header_buf (: 4 i64)))
      (def L_i32 (llvm.load {:result i32} L_ptr))
      (def L (arith.extsi {:result i64} L_i32))
    
      (def NH_ptr (ptr-at i32 header_buf (: 5 i64)))
      (def NH_i32 (llvm.load {:result i32} NH_ptr))
    
      (def C_ptr (ptr-at i32 header_buf (: 6 i64)))
      (def C_i32 (llvm.load {:result i32} C_ptr))
      (def C (arith.extsi {:result i64} C_i32))
    
      (def Vp_ptr (ptr-at i32 header_buf (: 7 i64)))
      (def Vp_i32 (llvm.load {:result i32} Vp_ptr))
      (def Vp (arith.extsi {:result i64} Vp_i32))
    
      ;; Print config
      (print "[GPT-2 Config]\n")
      (print "  max_seq_len: %d\n" maxT_i32)
      (print "  vocab_size: %d\n" V_i32)
      (print "  num_layers: %d\n" L_i32)
      (print "  num_heads: %d\n" NH_i32)
      (print "  channels: %d\n" C_i32)
    
      ;; Compute derived dimensions
      (def C3 (arith.muli C (: 3 i64)))
      (def C4 (arith.muli C (: 4 i64)))
    
      ;; Compute parameter sizes
      (def size_wte (arith.muli Vp C))
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
    
      ;; Compute base offsets
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
    
      ;; Compute per-layer strides
      (def ln_stride C)
      (def qkvw_stride (arith.muli C3 C))
      (def qkvb_stride C3)
      (def attprojw_stride (arith.muli C C))
      (def attprojb_stride C)
      (def fcw_stride (arith.muli C4 C))
      (def fcb_stride C4)
      (def fcprojw_stride (arith.muli C C4))
      (def fcprojb_stride C)
    
      ;; Load parameters
      (def sizeof_f32 (: 4 i64))
      (def total_bytes (arith.muli total_params sizeof_f32))
      (def params_ptr (func.call {:result !llvm.ptr} "malloc" total_bytes))
      (print "Loading weights...\n")
      (def read_count (func.call {:result i64} "fread" params_ptr sizeof_f32 total_params file))
      (print "Loaded %ld floats\n" read_count)
      (def _close (func.call {:result i32} "fclose" file))
    
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
      (def wpe_ptr (ptr-at f32 params_ptr wpe_base))
    
      ;; Create token_ids buffer with EOT token
      (def token_ids (memref.alloc {:result memref<64xi32>}))
      (gpu.host_register (memref.cast {:result "memref<*xi32>"} token_ids))
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

      ;; Attention workspace buffers (for batched attention)
      (def Q_batched (memref.alloc {:result memref<12x64x64xf32>}))
      (def K_batched (memref.alloc {:result memref<12x64x64xf32>}))
      (def V_batched (memref.alloc {:result memref<12x64x64xf32>}))
      (def K_transposed (memref.alloc {:result memref<12x64x64xf32>}))
      (def attn_scores (memref.alloc {:result memref<12x64x64xf32>}))
      (def attn_weights (memref.alloc {:result memref<12x64x64xf32>}))
      (def attn_values (memref.alloc {:result memref<12x64x64xf32>}))

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

      ;; Register all buffers with GPU runtime
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} x))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} x2))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} qkv_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_proj_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fc_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} gelu_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fc_proj_out))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} logits))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln1_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln1_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} qkv_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} qkv_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln2_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} ln2_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fc_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fc_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fcproj_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} fcproj_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} lnf_w))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} lnf_b))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} Q_batched))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} K_batched))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} V_batched))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} K_transposed))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_scores))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_weights))
      (gpu.host_register (memref.cast {:result "memref<*xf32>"} attn_values))

      (def c768 (: 768 index))
      (def c2304 (: 2304 index))
      (def c3072 (: 3072 index))
      (def c12 (: 12 index))
    
      ;; Generation loop: Generate 20 tokens
      (def prompt_len (: 1 index))
      (def gen_steps (: 20 index))
      (def gen_end (arith.addi prompt_len gen_steps))
    
      (print "\nGenerating tokens:\n")

      (def start_time (func.call {:result i64} "clock_ms"))

      (scf.for prompt_len gen_end c1
        (region
          (block [(: step index)]
            ;; 1. Embedding lookup
            (func.call "embedding_lookup" x wte_ptr wpe_ptr token_ids)
    
            ;; 2. Run 12 transformer layers
            (scf.for c0 c12 c1
              (region
                (block [(: layer index)]
                  (def layer_i64 (arith.index_cast {:result i64} layer))
    
                  ;; Load ln1 weights
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
    
                  ;; Load qkv weights (transpose: checkpoint[oc*C+c] -> qkv_w[c][oc])
                  (def qkvw_offset (arith.addi qkvw_base (arith.muli layer_i64 qkvw_stride)))
                  (def qkvb_offset (arith.addi qkvb_base (arith.muli layer_i64 qkvb_stride)))
                  (scf.for c0 c2304 c1
                    (region
                      (block [(: oc index)]
                        (def oc_i64 (arith.index_cast {:result i64} oc))
                        (def row_offset (arith.muli oc_i64 C))
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
    
                  ;; Load attention projection weights (transpose)
                  (def attprojw_offset (arith.addi attprojw_base (arith.muli layer_i64 attprojw_stride)))
                  (def attprojb_offset (arith.addi attprojb_base (arith.muli layer_i64 attprojb_stride)))
                  (scf.for c0 c768 c1
                    (region
                      (block [(: oc index)]
                        (def oc_i64 (arith.index_cast {:result i64} oc))
                        (def row_offset (arith.muli oc_i64 C))
                        (scf.for c0 c768 c1
                          (region
                            (block [(: c index)]
                              (def c_i64 (arith.index_cast {:result i64} c))
                              (def w_ptr (ptr-at f32 params_ptr (arith.addi attprojw_offset (arith.addi row_offset c_i64))))
                              (memref.store (llvm.load {:result f32} w_ptr) attn_w c oc)
                              (scf.yield))))
                        (scf.yield))))
                  (scf.for c0 c768 c1
                    (region
                      (block [(: i index)]
                        (def i_i64 (arith.index_cast {:result i64} i))
                        (def ab_ptr (ptr-at f32 params_ptr (arith.addi attprojb_offset i_i64)))
                        (memref.store (llvm.load {:result f32} ab_ptr) attn_b i)
                        (scf.yield))))
    
                  ;; Load ln2 weights
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
    
                  ;; Load fc weights (transpose)
                  (def fcw_offset (arith.addi fcw_base (arith.muli layer_i64 fcw_stride)))
                  (def fcb_offset (arith.addi fcb_base (arith.muli layer_i64 fcb_stride)))
                  (scf.for c0 c3072 c1
                    (region
                      (block [(: oc index)]
                        (def oc_i64 (arith.index_cast {:result i64} oc))
                        (def row_offset (arith.muli oc_i64 C))
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
    
                  ;; Load fc projection weights (transpose: checkpoint[oc*(4*C)+ic] -> fcproj_w[ic][oc])
                  (def fcprojw_offset (arith.addi fcprojw_base (arith.muli layer_i64 fcprojw_stride)))
                  (def fcprojb_offset (arith.addi fcprojb_base (arith.muli layer_i64 fcprojb_stride)))
                  (scf.for c0 c768 c1
                    (region
                      (block [(: oc index)]
                        (def oc_i64 (arith.index_cast {:result i64} oc))
                        (def row_offset (arith.muli oc_i64 C4))
                        (scf.for c0 c3072 c1
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
    
                  ;; === Transformer block operations ===
                  ;; 1. LayerNorm1
                  (func.call "layernorm_forward" ln_out x ln1_w ln1_b)
    
                  ;; 2. QKV Projection (using linalg.matmul)
                  (func.call "matmul_qkv" qkv_out ln_out qkv_w qkv_b)
    
                  ;; 3. Attention (GPU batched matmul + softmax)
                  (func.call "attention_forward" attn_out qkv_out
                             Q_batched K_batched V_batched
                             K_transposed attn_scores attn_weights attn_values)
    
                  ;; 4. Attention Output Projection (using linalg.matmul)
                  (func.call "matmul_attn_proj" attn_proj_out attn_out attn_w attn_b)
    
                  ;; 5. Residual 1 (using linalg.add)
                  (func.call "residual_forward" x2 x attn_proj_out)
    
                  ;; 6. LayerNorm2
                  (func.call "layernorm_forward" ln_out x2 ln2_w ln2_b)
    
                  ;; 7. MLP FC (using linalg.matmul)
                  (func.call "matmul_fc" fc_out ln_out fc_w fc_b)
    
                  ;; 8. GELU (using math.tanh)
                  (func.call "gelu_forward" gelu_out fc_out)
    
                  ;; 9. MLP Projection (using linalg.matmul)
                  (func.call "matmul_fc_proj" fc_proj_out gelu_out fcproj_w fcproj_b)
    
                  ;; 10. Residual 2 (using linalg.add)
                  (func.call "residual_forward" x x2 fc_proj_out)
    
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
    
            ;; 4. Compute logits from position step-1 (the last token's position)
            (def logit_pos (arith.subi step c1))
            (func.call "logits_forward" logits x2 wte_ptr logit_pos)
    
            ;; 5. Argmax to get next token
            (def next_token (func.call {:result i32} "argmax" logits))
    
            ;; 6. Print token
            (func.call "print_token" next_token)
    
            ;; 7. Store token at current step position
            (memref.store next_token token_ids step)
    
            (scf.yield))))

      (def end_time (func.call {:result i64} "clock_ms"))
      (def elapsed_ms (arith.subi end_time start_time))
      (print "\n\nGeneration complete!\n")
      (print "Time for 20 tokens: %ld ms\n" elapsed_ms)
      (print "Per token: %ld ms\n" (arith.divsi elapsed_ms (: 20 i64)))
    
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
      (memref.dealloc Q_batched)
      (memref.dealloc K_batched)
      (memref.dealloc V_batched)
      (memref.dealloc K_transposed)
      (memref.dealloc attn_scores)
      (memref.dealloc attn_weights)
      (memref.dealloc attn_values)
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
    
      (func.call "free" params_ptr)

      (func.return))))