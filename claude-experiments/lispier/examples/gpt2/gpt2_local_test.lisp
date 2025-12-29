;; GPT-2 Local Test - Tests transformer forward pass with synthetic data
;; No checkpoint file needed - uses randomly initialized weights

(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect llvm)
(require-dialect scf)
(require-dialect math)

(link-library :c)

(extern-fn printf (-> [!llvm.ptr ...] [i32]))

(module
  (do
    ;; =========================================================================
    ;; LayerNorm: out = (inp - mean) / sqrt(var + eps) * gamma + beta
    ;; =========================================================================
    (func.func {:sym_name "layernorm_forward"
                :function_type (-> [memref<64x768xf32>     ; out
                                    memref<64x768xf32>     ; inp
                                    memref<768xf32>        ; gamma
                                    memref<768xf32>] [])}  ; beta
      (region
        (block [(: out memref<64x768xf32>)
                (: inp memref<64x768xf32>)
                (: gamma memref<768xf32>)
                (: beta memref<768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def eps (: 1e-5 f32))
          (def zero (: 0.0 f32))
          (def n_f32 (: 768.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                ;; Compute mean
                (def sum (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def val (memref.load inp t c))
                      (scf.yield (arith.addf acc val))))))
                (def mean (arith.divf sum n_f32))

                ;; Compute variance
                (def var_sum (scf.for {:result f32} c0 c768 c1 zero
                  (region
                    (block [(: c index) (: acc f32)]
                      (def val (memref.load inp t c))
                      (def diff (arith.subf val mean))
                      (def sq (arith.mulf diff diff))
                      (scf.yield (arith.addf acc sq))))))
                (def var (arith.divf var_sum n_f32))
                (def var_eps (arith.addf var eps))
                (def std (math.sqrt var_eps))

                ;; Normalize
                (scf.for c0 c768 c1
                  (region
                    (block [(: c index)]
                      (def val (memref.load inp t c))
                      (def norm (arith.divf (arith.subf val mean) std))
                      (def g (memref.load gamma c))
                      (def b (memref.load beta c))
                      (def result (arith.addf (arith.mulf norm g) b))
                      (memref.store result out t c)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; Matmul with bias: out = inp @ weight + bias
    ;; =========================================================================
    (func.func {:sym_name "matmul_768_768"
                :function_type (-> [memref<64x768xf32>     ; out
                                    memref<64x768xf32>     ; inp
                                    memref<768x768xf32>    ; weight
                                    memref<768xf32>] [])}  ; bias
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
                            (scf.yield (arith.addf acc prod))))))
                      (def b (memref.load bias k))
                      (memref.store (arith.addf sum b) out t k)
                      (scf.yield))))
                (scf.yield))))
          (func.return))))

    ;; =========================================================================
    ;; GELU activation
    ;; =========================================================================
    (func.func {:sym_name "gelu_forward"
                :function_type (-> [memref<64x768xf32> memref<64x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>) (: inp memref<64x768xf32>)]
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def sqrt_2_over_pi (: 0.7978845608 f32))
          (def coeff (: 0.044715 f32))
          (def half (: 0.5 f32))
          (def one (: 1.0 f32))

          (scf.for c0 c64 c1
            (region
              (block [(: t index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: c index)]
                      (def x (memref.load inp t c))
                      (def x3 (arith.mulf x (arith.mulf x x)))
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
          (func.return))))))

;; Main test
(defn main [] -> i64
  ;; Allocate buffers
  (def x (memref.alloc {:result memref<64x768xf32>}))
  (def ln_out (memref.alloc {:result memref<64x768xf32>}))
  (def attn_out (memref.alloc {:result memref<64x768xf32>}))
  (def gelu_out (memref.alloc {:result memref<64x768xf32>}))
  (def residual (memref.alloc {:result memref<64x768xf32>}))

  ;; Allocate weight buffers
  (def ln_gamma (memref.alloc {:result memref<768xf32>}))
  (def ln_beta (memref.alloc {:result memref<768xf32>}))
  (def fc_weight (memref.alloc {:result memref<768x768xf32>}))
  (def fc_bias (memref.alloc {:result memref<768xf32>}))

  ;; Initialize with small values
  (def c0 (: 0 index))
  (def c1 (: 1 index))
  (def c64 (: 64 index))
  (def c768 (: 768 index))
  (def one (: 1.0 f32))
  (def zero (: 0.0 f32))
  (def small (: 0.001 f32))

  ;; Init x to 1.0
  (scf.for c0 c64 c1
    (region
      (block [(: t index)]
        (scf.for c0 c768 c1
          (region
            (block [(: c index)]
              (memref.store one x t c)
              (scf.yield))))
        (scf.yield))))

  ;; Init ln_gamma to 1.0, ln_beta to 0.0
  (scf.for c0 c768 c1
    (region
      (block [(: c index)]
        (memref.store one ln_gamma c)
        (memref.store zero ln_beta c)
        (memref.store zero fc_bias c)
        (scf.yield))))

  ;; Init fc_weight to 0.001
  (scf.for c0 c768 c1
    (region
      (block [(: i index)]
        (scf.for c0 c768 c1
          (region
            (block [(: j index)]
              (memref.store small fc_weight i j)
              (scf.yield))))
        (scf.yield))))

  (print "Running layernorm...\n")
  (func.call "layernorm_forward" ln_out x ln_gamma ln_beta)

  ;; Check ln_out[0][0] - should be 0 since all values are equal (mean = val)
  (def ln_val (memref.load ln_out c0 c0))
  (def ln_val_f64 (arith.extf {:result f64} ln_val))
  (print "ln_out[0][0] = %f (expect 0.0)\n" ln_val_f64)

  (print "Running matmul...\n")
  (func.call "matmul_768_768" attn_out ln_out fc_weight fc_bias)

  ;; Check attn_out[0][0] - should be 0 since ln_out is all zeros
  (def attn_val (memref.load attn_out c0 c0))
  (def attn_val_f64 (arith.extf {:result f64} attn_val))
  (print "attn_out[0][0] = %f (expect 0.0)\n" attn_val_f64)

  (print "Running gelu...\n")
  (func.call "gelu_forward" gelu_out attn_out)

  (def gelu_val (memref.load gelu_out c0 c0))
  (def gelu_val_f64 (arith.extf {:result f64} gelu_val))
  (print "gelu_out[0][0] = %f (expect 0.0)\n" gelu_val_f64)

  (print "Running residual...\n")
  (func.call "residual_add" residual x gelu_out)

  ;; Check residual[0][0] - should be 1.0 + 0.0 = 1.0
  (def res_val (memref.load residual c0 c0))
  (def res_val_f64 (arith.extf {:result f64} res_val))
  (print "residual[0][0] = %f (expect 1.0)\n" res_val_f64)

  ;; Cleanup
  (memref.dealloc x)
  (memref.dealloc ln_out)
  (memref.dealloc attn_out)
  (memref.dealloc gelu_out)
  (memref.dealloc residual)
  (memref.dealloc ln_gamma)
  (memref.dealloc ln_beta)
  (memref.dealloc fc_weight)
  (memref.dealloc fc_bias)

  (print "Test complete!\n")
  (func.return (: 0 i64)))