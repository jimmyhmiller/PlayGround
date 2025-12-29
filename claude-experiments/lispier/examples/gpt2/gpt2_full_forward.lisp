;; GPT-2 Full 12-Layer Forward Pass
;; Loads weights from checkpoint and runs complete inference
;;
;; Weight layout (interleaved by type across all 12 layers):
;; - wte: 50257 × 768 = 38,597,376 floats
;; - wpe: 1024 × 768 = 786,432 floats
;; - ln1w: 12 × 768 = 9,216 floats
;; - ln1b: 12 × 768 = 9,216 floats
;; - qkvw: 12 × 768 × 2304 = 21,233,664 floats
;; - qkvb: 12 × 2304 = 27,648 floats
;; - attprojw: 12 × 768 × 768 = 7,077,888 floats
;; - attprojb: 12 × 768 = 9,216 floats
;; - ln2w: 12 × 768 = 9,216 floats
;; - ln2b: 12 × 768 = 9,216 floats
;; - fcw: 12 × 768 × 3072 = 28,311,552 floats
;; - fcb: 12 × 3072 = 36,864 floats
;; - fcprojw: 12 × 3072 × 768 = 28,311,552 floats
;; - fcprojb: 12 × 768 = 9,216 floats
;; - lnfw: 768 floats
;; - lnfb: 768 floats
;; Total: 124,439,808 floats = ~497 MB

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect llvm)
(require-dialect scf)
(require-dialect math)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

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

    (llvm.mlir.global {:sym_name "read_mode"
                       :linkage 0
                       :global_type !llvm.array<3 x i8>
                       :constant true}
      (region
        (block []
          (def s (llvm.mlir.constant {:value "rb\0" :result !llvm.array<3 x i8>}))
          (llvm.return s))))

    ;; Global to store params pointer
    (llvm.mlir.global {:sym_name "g_params"
                       :linkage 10
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null))))

    ;; =========================================================================
    ;; LayerNorm forward
    ;; =========================================================================
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<64x768xf32>   ; out
                                    memref<64x768xf32>   ; inp
                                    memref<768xf32>      ; weight (gamma)
                                    memref<768xf32>      ; bias (beta)
                                    f32] [])}            ; eps
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: weight memref<768xf32>)
                (: bias memref<768xf32>)
                (: eps f32)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c768_f32 (: 768.0 f32))
          (def zero (: 0.0 f32))
          (def one (: 1.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                ;; Mean
                (def sum_val (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def x (memref.load inp t c))
                      (def new_acc (arith.addf acc x))
                      (scf.yield new_acc)))))
                (def m (arith.divf sum_val c768_f32))

                ;; Variance
                (def var_val (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: c index) (: vacc f32)]
                      (def x (memref.load inp t c))
                      (def diff (arith.subf x m))
                      (def diff_sq (arith.mulf diff diff))
                      (def new_vacc (arith.addf vacc diff_sq))
                      (scf.yield new_vacc)))))
                (def variance (arith.divf var_val c768_f32))

                ;; Reciprocal std
                (def var_eps (arith.addf variance eps))
                (def std (math.sqrt var_eps))
                (def rs (arith.divf one std))

                ;; Normalize
                (scf.for c0 c768 c1
                  (region
                    (block [(: c index)]
                      (def x (memref.load inp t c))
                      (def x_norm (arith.mulf (arith.subf x m) rs))
                      (def gamma (memref.load weight c))
                      (def beta (memref.load bias c))
                      (def scaled (arith.mulf x_norm gamma))
                      (def result (arith.addf scaled beta))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; QKV Matmul with bias: (64,768) @ (768,2304) + bias -> (64,2304)
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

          ;; out[t,k] = sum_c inp[t,c] * weight[c,k] + bias[k]
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
                            (def prod (arith.mulf inp_val w_val))
                            (def new_acc (arith.addf acc prod))
                            (scf.yield new_acc)))))
                      (def b (memref.load bias k))
                      (def result (arith.addf sum b))
                      (memref.store result out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Attention forward (simplified - single head version for testing)
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
          (def hs (: 64 index))
          (def zero (: 0.0 f32))
          (def neg_inf (: -10000.0 f32))
          (def scale (: 0.125 f32))  ; 1/sqrt(64)

          ;; Process each head
          (scf.for c0 c12 c1
            (region
              (block [(: h index)]
                (def h_offset (arith.muli h hs))
                (scf.for c0 c64 c1
                  (region
                    (block [(: t index)]
                      ;; Compute attention scores with causal mask
                      (def max_score (scf.for {:result f32} c0 c64 c1 neg_inf
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
                                        (def q_val (memref.load qkv t q_idx))
                                        (def k_base (arith.addi c768 h_offset))
                                        (def k_idx (arith.addi k_base d))
                                        (def k_val (memref.load qkv t2 k_idx))
                                        (def prod (arith.mulf q_val k_val))
                                        (def new_acc (arith.addf acc prod))
                                        (scf.yield new_acc)))))
                                  (def scaled_dot (arith.mulf dot scale))
                                  (scf.yield scaled_dot)))
                              (region
                                (block []
                                  (scf.yield neg_inf)))))
                            (def is_greater (arith.cmpf {:predicate "ogt"} score curr_max))
                            (def new_max (arith.select is_greater score curr_max))
                            (scf.yield new_max)))))

                      ;; Softmax sum
                      (def exp_sum (scf.for {:result f32} c0 c64 c1 zero
                        (region
                          (block [(: t2 index) (: sum_acc f32)]
                            (def is_valid (arith.cmpi {:predicate "ule"} t2 t))
                            (def exp_val (scf.if {:result f32} is_valid
                              (region
                                (block []
                                  (def dot (scf.for {:result f32} c0 hs c1 zero
                                    (region
                                      (block [(: d index) (: acc f32)]
                                        (def q_idx (arith.addi h_offset d))
                                        (def q_val (memref.load qkv t q_idx))
                                        (def k_base (arith.addi c768 h_offset))
                                        (def k_idx (arith.addi k_base d))
                                        (def k_val (memref.load qkv t2 k_idx))
                                        (def prod (arith.mulf q_val k_val))
                                        (def new_acc (arith.addf acc prod))
                                        (scf.yield new_acc)))))
                                  (def scaled (arith.mulf dot scale))
                                  (def shifted (arith.subf scaled max_score))
                                  (def ev (math.exp shifted))
                                  (scf.yield ev)))
                              (region
                                (block []
                                  (scf.yield zero)))))
                            (def new_sum (arith.addf sum_acc exp_val))
                            (scf.yield new_sum)))))

                      ;; Weighted sum of V
                      (scf.for c0 hs c1
                        (region
                          (block [(: d index)]
                            (def weighted_sum (scf.for {:result f32} c0 c64 c1 zero
                              (region
                                (block [(: t2 index) (: acc f32)]
                                  (def is_valid (arith.cmpi {:predicate "ule"} t2 t))
                                  (def contrib (scf.if {:result f32} is_valid
                                    (region
                                      (block []
                                        ;; Recompute attention weight
                                        (def dot (scf.for {:result f32} c0 hs c1 zero
                                          (region
                                            (block [(: d2 index) (: dacc f32)]
                                              (def q_idx (arith.addi h_offset d2))
                                              (def q_val (memref.load qkv t q_idx))
                                              (def k_base (arith.addi c768 h_offset))
                                              (def k_idx (arith.addi k_base d2))
                                              (def k_val (memref.load qkv t2 k_idx))
                                              (def prod (arith.mulf q_val k_val))
                                              (def new_dacc (arith.addf dacc prod))
                                              (scf.yield new_dacc)))))
                                        (def scaled (arith.mulf dot scale))
                                        (def shifted (arith.subf scaled max_score))
                                        (def ev (math.exp shifted))
                                        (def att_weight (arith.divf ev exp_sum))
                                        ;; V value
                                        (def v_base_c (arith.addi c768 c768))
                                        (def v_base (arith.addi v_base_c h_offset))
                                        (def v_idx (arith.addi v_base d))
                                        (def v_val (memref.load qkv t2 v_idx))
                                        (def weighted (arith.mulf att_weight v_val))
                                        (scf.yield weighted)))
                                    (region
                                      (block []
                                        (scf.yield zero)))))
                                  (def new_acc (arith.addf acc contrib))
                                  (scf.yield new_acc)))))
                            (def out_idx (arith.addi h_offset d))
                            (memref.store weighted_sum out t out_idx)
                            (scf.yield))))
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Attention projection: (64,768) @ (768,768) + bias -> (64,768)
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
                            (def prod (arith.mulf inp_val w_val))
                            (def new_acc (arith.addf acc prod))
                            (scf.yield new_acc)))))
                      (def b (memref.load bias k))
                      (def result (arith.addf sum b))
                      (memref.store result out t k)
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
                      (def sum (arith.addf va vb))
                      (memref.store sum out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; MLP FC: (64,768) @ (768,3072) + bias -> (64,3072)
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
                            (def prod (arith.mulf inp_val w_val))
                            (def new_acc (arith.addf acc prod))
                            (scf.yield new_acc)))))
                      (def b (memref.load bias k))
                      (def result (arith.addf sum b))
                      (memref.store result out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; GELU activation
    ;; =========================================================================
    (func.func {:sym_name "gelu_forward"
                :function_type (-> [memref<64x3072xf32>
                                    memref<64x3072xf32>] [])}
      (region
        (block [(: out memref<64x3072xf32>)
                (: inp memref<64x3072xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c3072 (: 3072 index))
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))
          (def sqrt_2_over_pi (: 0.7978845608 f32))
          (def coeff (: 0.044715 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c3072 c1
                  (region
                    (block [(: c index)]
                      (def x (memref.load inp t c))
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
    ;; MLP projection: (64,3072) @ (3072,768) + bias -> (64,768)
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
                            (def prod (arith.mulf inp_val w_val))
                            (def new_acc (arith.addf acc prod))
                            (scf.yield new_acc)))))
                      (def b (memref.load bias k))
                      (def result (arith.addf sum b))
                      (memref.store result out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))))

;; =========================================================================
;; Weight offset constants (in floats from start of params after header)
;; =========================================================================
;; wte_offset = 0
;; wpe_offset = 38597376
;; ln1w_offset = 39383808
;; ln1b_offset = 39393024
;; qkvw_offset = 39402240
;; qkvb_offset = 60635904
;; attprojw_offset = 60663552
;; attprojb_offset = 67741440
;; ln2w_offset = 67750656
;; ln2b_offset = 67759872
;; fcw_offset = 67769088
;; fcb_offset = 96080640
;; fcprojw_offset = 96117504
;; fcprojb_offset = 124429056
;; lnfw_offset = 124438272
;; lnfb_offset = 124439040

;; =========================================================================
;; Main function
;; =========================================================================
(defn main [] -> i64
  ;; Open checkpoint
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; Skip header (256 ints = 1024 bytes)
  (def header_size (: 1024 i64))
  (def seek_set (: 0 i32))
  (def _skip (call i32 fseek file header_size seek_set))

  ;; Allocate buffer for all params (~497 MB)
  (def total_params (: 124439808 i64))
  (def sizeof_f32 (: 4 i64))
  (def total_bytes (arith.muli total_params sizeof_f32))
  (def params_ptr (call !llvm.ptr malloc total_bytes))
  (print "Allocated %ld bytes for parameters\n" total_bytes)

  ;; Read all params
  (def read_count (call i64 fread params_ptr sizeof_f32 total_params file))
  (print "Read %ld floats\n" read_count)
  (def _close (call i32 fclose file))

  ;; Store in global
  (def g_params_addr (llvm.mlir.addressof {:global_name @g_params :result !llvm.ptr}))
  (llvm.store params_ptr g_params_addr)

  ;; Verify first wte value
  (def wte_val (llvm.load {:result f32} params_ptr))
  (def wte_val_f64 (arith.extf {:result f64} wte_val))
  (print "wte[0][0] = %f\n" wte_val_f64)

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

  ;; Allocate weight memrefs (reused for each layer)
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

  ;; Initialize input with token embeddings (tokens 0-63 for simplicity)
  ;; x = wte[token] + wpe[pos]
  (def c0 (: 0 index))
  (def c1 (: 1 index))
  (def c64 (: 64 index))
  (def c768 (: 768 index))
  (def wpe_offset (: 38597376 i64))

  (scf.for c0 c64 c1
    (region
      (block [(: t index)]
        (def t_i64 (arith.index_cast {:result i64} t))
        (def wte_row_offset (arith.muli t_i64 (: 768 i64)))
        (def wpe_row_offset (arith.addi wpe_offset (arith.muli t_i64 (: 768 i64))))
        (scf.for c0 c768 c1
          (region
            (block [(: c index)]
              (def c_i64 (arith.index_cast {:result i64} c))
              ;; wte[t, c]
              (def wte_idx (arith.addi wte_row_offset c_i64))
              (def wte_ptr (ptr-at f32 params_ptr wte_idx))
              (def wte_v (llvm.load {:result f32} wte_ptr))
              ;; wpe[t, c]
              (def wpe_idx (arith.addi wpe_row_offset c_i64))
              (def wpe_ptr (ptr-at f32 params_ptr wpe_idx))
              (def wpe_v (llvm.load {:result f32} wpe_ptr))
              ;; x = wte + wpe
              (def sum (arith.addf wte_v wpe_v))
              (memref.store sum x t c)
              (scf.yield))))
        (scf.yield))))

  (print "Initialized embeddings\n")

  ;; Print first embedding value
  (def emb_val (memref.load x c0 c0))
  (def emb_val_f64 (arith.extf {:result f64} emb_val))
  (print "x[0][0] = %f\n" emb_val_f64)

  ;; Weight base offsets
  (def ln1w_base (: 39383808 i64))
  (def ln1b_base (: 39393024 i64))
  (def qkvw_base (: 39402240 i64))
  (def qkvb_base (: 60635904 i64))
  (def attprojw_base (: 60663552 i64))
  (def attprojb_base (: 67741440 i64))
  (def ln2w_base (: 67750656 i64))
  (def ln2b_base (: 67759872 i64))
  (def fcw_base (: 67769088 i64))
  (def fcb_base (: 96080640 i64))
  (def fcprojw_base (: 96117504 i64))
  (def fcprojb_base (: 124429056 i64))
  (def lnfw_base (: 124438272 i64))
  (def lnfb_base (: 124439040 i64))

  ;; Per-layer strides
  (def ln_stride (: 768 i64))
  (def qkvw_stride (: 1769472 i64))   ; 768 * 2304
  (def qkvb_stride (: 2304 i64))
  (def attnw_stride (: 589824 i64))   ; 768 * 768
  (def attnb_stride (: 768 i64))
  (def fcw_stride (: 2359296 i64))    ; 768 * 3072
  (def fcb_stride (: 3072 i64))
  (def fcprojw_stride (: 2359296 i64)) ; 3072 * 768
  (def fcprojb_stride (: 768 i64))

  (def eps (: 0.00001 f32))
  (def c12 (: 12 index))

  ;; Process 12 transformer layers
  (scf.for c0 c12 c1
    (region
      (block [(: layer index)]
        (def layer_i64 (arith.index_cast {:result i64} layer))
        (print "Processing layer %ld\n" layer_i64)

        ;; Compute layer-specific weight offsets
        (def ln1w_offset (arith.addi ln1w_base (arith.muli layer_i64 ln_stride)))
        (def ln1b_offset (arith.addi ln1b_base (arith.muli layer_i64 ln_stride)))
        (def qkvw_offset (arith.addi qkvw_base (arith.muli layer_i64 qkvw_stride)))
        (def qkvb_offset (arith.addi qkvb_base (arith.muli layer_i64 qkvb_stride)))
        (def attnw_offset (arith.addi attprojw_base (arith.muli layer_i64 attnw_stride)))
        (def attnb_offset (arith.addi attprojb_base (arith.muli layer_i64 attnb_stride)))
        (def ln2w_offset (arith.addi ln2w_base (arith.muli layer_i64 ln_stride)))
        (def ln2b_offset (arith.addi ln2b_base (arith.muli layer_i64 ln_stride)))
        (def fcw_offset (arith.addi fcw_base (arith.muli layer_i64 fcw_stride)))
        (def fcb_offset (arith.addi fcb_base (arith.muli layer_i64 fcb_stride)))
        (def fcprojw_offset (arith.addi fcprojw_base (arith.muli layer_i64 fcprojw_stride)))
        (def fcprojb_offset (arith.addi fcprojb_base (arith.muli layer_i64 fcprojb_stride)))

        ;; Copy ln1 weights
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def w_ptr (ptr-at f32 params_ptr (arith.addi ln1w_offset i_i64)))
              (def b_ptr (ptr-at f32 params_ptr (arith.addi ln1b_offset i_i64)))
              (def w_val (llvm.load {:result f32} w_ptr))
              (def b_val (llvm.load {:result f32} b_ptr))
              (memref.store w_val ln1_w i)
              (memref.store b_val ln1_b i)
              (scf.yield))))

        ;; Copy qkv weights (768 x 2304)
        (def c2304 (: 2304 index))
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def row_base (arith.addi qkvw_offset (arith.muli i_i64 (: 2304 i64))))
              (scf.for c0 c2304 c1
                (region
                  (block [(: j index)]
                    (def j_i64 (arith.index_cast {:result i64} j))
                    (def w_ptr (ptr-at f32 params_ptr (arith.addi row_base j_i64)))
                    (def w_val (llvm.load {:result f32} w_ptr))
                    (memref.store w_val qkv_w i j)
                    (scf.yield))))
              (scf.yield))))

        ;; Copy qkv bias
        (scf.for c0 c2304 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def b_ptr (ptr-at f32 params_ptr (arith.addi qkvb_offset i_i64)))
              (def b_val (llvm.load {:result f32} b_ptr))
              (memref.store b_val qkv_b i)
              (scf.yield))))

        ;; Copy attn proj weights (768 x 768)
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def row_base (arith.addi attnw_offset (arith.muli i_i64 (: 768 i64))))
              (scf.for c0 c768 c1
                (region
                  (block [(: j index)]
                    (def j_i64 (arith.index_cast {:result i64} j))
                    (def w_ptr (ptr-at f32 params_ptr (arith.addi row_base j_i64)))
                    (def w_val (llvm.load {:result f32} w_ptr))
                    (memref.store w_val attn_w i j)
                    (scf.yield))))
              (scf.yield))))

        ;; Copy attn proj bias
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def b_ptr (ptr-at f32 params_ptr (arith.addi attnb_offset i_i64)))
              (def b_val (llvm.load {:result f32} b_ptr))
              (memref.store b_val attn_b i)
              (scf.yield))))

        ;; Copy ln2 weights
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def w_ptr (ptr-at f32 params_ptr (arith.addi ln2w_offset i_i64)))
              (def b_ptr (ptr-at f32 params_ptr (arith.addi ln2b_offset i_i64)))
              (def w_val (llvm.load {:result f32} w_ptr))
              (def b_val (llvm.load {:result f32} b_ptr))
              (memref.store w_val ln2_w i)
              (memref.store b_val ln2_b i)
              (scf.yield))))

        ;; Copy fc weights (768 x 3072)
        (def c3072 (: 3072 index))
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def row_base (arith.addi fcw_offset (arith.muli i_i64 (: 3072 i64))))
              (scf.for c0 c3072 c1
                (region
                  (block [(: j index)]
                    (def j_i64 (arith.index_cast {:result i64} j))
                    (def w_ptr (ptr-at f32 params_ptr (arith.addi row_base j_i64)))
                    (def w_val (llvm.load {:result f32} w_ptr))
                    (memref.store w_val fc_w i j)
                    (scf.yield))))
              (scf.yield))))

        ;; Copy fc bias
        (scf.for c0 c3072 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def b_ptr (ptr-at f32 params_ptr (arith.addi fcb_offset i_i64)))
              (def b_val (llvm.load {:result f32} b_ptr))
              (memref.store b_val fc_b i)
              (scf.yield))))

        ;; Copy fc proj weights (3072 x 768)
        (scf.for c0 c3072 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def row_base (arith.addi fcprojw_offset (arith.muli i_i64 (: 768 i64))))
              (scf.for c0 c768 c1
                (region
                  (block [(: j index)]
                    (def j_i64 (arith.index_cast {:result i64} j))
                    (def w_ptr (ptr-at f32 params_ptr (arith.addi row_base j_i64)))
                    (def w_val (llvm.load {:result f32} w_ptr))
                    (memref.store w_val fcproj_w i j)
                    (scf.yield))))
              (scf.yield))))

        ;; Copy fc proj bias
        (scf.for c0 c768 c1
          (region
            (block [(: i index)]
              (def i_i64 (arith.index_cast {:result i64} i))
              (def b_ptr (ptr-at f32 params_ptr (arith.addi fcprojb_offset i_i64)))
              (def b_val (llvm.load {:result f32} b_ptr))
              (memref.store b_val fcproj_b i)
              (scf.yield))))

        ;; === Run transformer block ===

        ;; 1. LayerNorm1
        (func.call "layernorm_forward" ln_out x ln1_w ln1_b eps)

        ;; 2. QKV projection
        (func.call "matmul_qkv" qkv_out ln_out qkv_w qkv_b)

        ;; 3. Attention
        (func.call "attention_forward" attn_out qkv_out)

        ;; 4. Attention projection
        (func.call "matmul_attn_proj" attn_proj_out attn_out attn_w attn_b)

        ;; 5. Residual add
        (func.call "residual_add" x2 x attn_proj_out)

        ;; 6. LayerNorm2
        (func.call "layernorm_forward" ln_out x2 ln2_w ln2_b eps)

        ;; 7. MLP FC
        (func.call "matmul_fc" fc_out ln_out fc_w fc_b)

        ;; 8. GELU
        (func.call "gelu_forward" gelu_out fc_out)

        ;; 9. MLP projection
        (func.call "matmul_fc_proj" fc_proj_out gelu_out fcproj_w fcproj_b)

        ;; 10. Residual add (x = x2 + fc_proj_out)
        (func.call "residual_add" x x2 fc_proj_out)

        (scf.yield))))

  (print "All 12 layers complete\n")

  ;; Final LayerNorm
  ;; Copy lnf weights
  (scf.for c0 c768 c1
    (region
      (block [(: i index)]
        (def i_i64 (arith.index_cast {:result i64} i))
        (def w_ptr (ptr-at f32 params_ptr (arith.addi lnfw_base i_i64)))
        (def b_ptr (ptr-at f32 params_ptr (arith.addi lnfb_base i_i64)))
        (def w_val (llvm.load {:result f32} w_ptr))
        (def b_val (llvm.load {:result f32} b_ptr))
        (memref.store w_val lnf_w i)
        (memref.store b_val lnf_b i)
        (scf.yield))))

  (func.call "layernorm_forward" x2 x lnf_w lnf_b eps)

  ;; Print final hidden state values
  (def final_val (memref.load x2 c0 c0))
  (def final_val_f64 (arith.extf {:result f64} final_val))
  (print "Final hidden[0][0] = %f\n" final_val_f64)

  (def final_val2 (memref.load x2 (: 63 index) c0))
  (def final_val2_f64 (arith.extf {:result f64} final_val2))
  (print "Final hidden[63][0] = %f\n" final_val2_f64)

  ;; Cleanup
  (memref.dealloc x)
  (memref.dealloc x2)
  (memref.dealloc ln_out)
  (memref.dealloc qkv_out)
  (memref.dealloc attn_out)
  (memref.dealloc attn_proj_out)
  (memref.dealloc fc_out)
  (memref.dealloc gelu_out)
  (memref.dealloc fc_proj_out)
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

  ;; Free params
  (call! free params_ptr)

  (func.return (: 0 i64)))