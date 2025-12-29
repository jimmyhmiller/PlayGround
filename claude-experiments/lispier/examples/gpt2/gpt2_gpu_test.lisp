;; GPT-2 Single Transformer Block Test on GPU
;; Uses gpu.host_register for memory accessible to GPU

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect linalg)
(require-dialect scf)
(require-dialect math)

;; Compilation pipeline for AMD GPU
(compilation
  (target rocm
    (pass convert-linalg-to-parallel-loops)
    (pass gpu-map-parallel-loops)
    (pass convert-parallel-loops-to-gpu)
    (pass lower-affine)
    (pass convert-scf-to-cf)
    (pass gpu-kernel-outlining)
    (pass rocdl-attach-target)
    (pass convert-gpu-to-rocdl {:use-bare-ptr-memref-call-conv true})
    (pass gpu-module-to-binary)
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
    ;; QKV matmul: (64,768) @ (768,2304) -> (64,2304)
    (func.func {:sym_name "matmul_qkv"
                :function_type (-> [memref<64x2304xf32> memref<64x768xf32> memref<768x2304xf32>] [])}
      (region
        (block [(: out memref<64x2304xf32>) (: inp memref<64x768xf32>) (: weight memref<768x2304xf32>)]
          (def zero (: 0.0 f32))
          (linalg.fill zero out)
          (linalg.matmul inp weight out)
          (func.return))))

    ;; Attention projection: (64,768) @ (768,768) -> (64,768)
    (func.func {:sym_name "matmul_attn_proj"
                :function_type (-> [memref<64x768xf32> memref<64x768xf32> memref<768x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>) (: inp memref<64x768xf32>) (: weight memref<768x768xf32>)]
          (def zero (: 0.0 f32))
          (linalg.fill zero out)
          (linalg.matmul inp weight out)
          (func.return))))

    ;; Residual add using linalg
    (func.func {:sym_name "residual_add"
                :function_type (-> [memref<64x768xf32> memref<64x768xf32> memref<64x768xf32>] [])}
      (region
        (block [(: out memref<64x768xf32>) (: a memref<64x768xf32>) (: b memref<64x768xf32>)]
          (linalg.add a b out)
          (func.return))))

    ;; Main test
    (func.func {:sym_name "main"
                :function_type (-> [] [])}
      (region
        (block []
          ;; Allocate buffers
          (def x (memref.alloc {:result memref<64x768xf32>}))
          (def qkv_weight (memref.alloc {:result memref<768x2304xf32>}))
          (def qkv_out (memref.alloc {:result memref<64x2304xf32>}))
          (def attn_weight (memref.alloc {:result memref<768x768xf32>}))
          (def attn_out (memref.alloc {:result memref<64x768xf32>}))
          (def residual_out (memref.alloc {:result memref<64x768xf32>}))

          ;; Initialize
          (def c0 (: 0 index))
          (def c1 (: 1 index))
          (def c64 (: 64 index))
          (def c768 (: 768 index))
          (def c2304 (: 2304 index))
          (def one (: 1.0 f32))
          (def small (: 0.001 f32))

          ;; Initialize input x to 1.0
          (scf.for c0 c64 c1
            (region
              (block [(: i index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: j index)]
                      (memref.store one x i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize qkv_weight to 0.001
          (scf.for c0 c768 c1
            (region
              (block [(: i index)]
                (scf.for c0 c2304 c1
                  (region
                    (block [(: j index)]
                      (memref.store small qkv_weight i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Initialize attn_weight to 0.001
          (scf.for c0 c768 c1
            (region
              (block [(: i index)]
                (scf.for c0 c768 c1
                  (region
                    (block [(: j index)]
                      (memref.store small attn_weight i j)
                      (scf.yield))))
                (scf.yield))))

          ;; Register all buffers with GPU
          (def x_u (memref.cast {:result "memref<*xf32>"} x))
          (def qkv_w_u (memref.cast {:result "memref<*xf32>"} qkv_weight))
          (def qkv_o_u (memref.cast {:result "memref<*xf32>"} qkv_out))
          (def attn_w_u (memref.cast {:result "memref<*xf32>"} attn_weight))
          (def attn_o_u (memref.cast {:result "memref<*xf32>"} attn_out))
          (def res_o_u (memref.cast {:result "memref<*xf32>"} residual_out))

          (gpu.host_register x_u)
          (gpu.host_register qkv_w_u)
          (gpu.host_register qkv_o_u)
          (gpu.host_register attn_w_u)
          (gpu.host_register attn_o_u)
          (gpu.host_register res_o_u)

          ;; Run QKV matmul
          (func.call "matmul_qkv" qkv_out x qkv_weight)

          ;; Simulate attention by extracting first 768 values (V portion offset by 2*768)
          ;; For simplicity, just use first 768 columns of qkv_out as "attention output"
          ;; In reality, attention is more complex - this just tests the matmul chain

          ;; Run attn projection (using first 768 columns of input as mock)
          ;; We'd need proper slicing - for now just use x as mock input
          (func.call "matmul_attn_proj" attn_out x attn_weight)

          ;; Residual add
          (func.call "residual_add" residual_out x attn_out)

          ;; Print results
          ;; QKV out[0][0] = 768 * 0.001 = 0.768
          (def r1 (memref.load qkv_out c0 c0))
          (func.call "printF32" r1)
          (func.call "printNewline")

          ;; Attn out[0][0] = 768 * 0.001 = 0.768
          (def r2 (memref.load attn_out c0 c0))
          (func.call "printF32" r2)
          (func.call "printNewline")

          ;; Residual out[0][0] = 1.0 + 0.768 = 1.768
          (def r3 (memref.load residual_out c0 c0))
          (func.call "printF32" r3)
          (func.call "printNewline")

          ;; Unregister
          (gpu.host_unregister x_u)
          (gpu.host_unregister qkv_w_u)
          (gpu.host_unregister qkv_o_u)
          (gpu.host_unregister attn_w_u)
          (gpu.host_unregister attn_o_u)
          (gpu.host_unregister res_o_u)

          ;; Cleanup
          (memref.dealloc x)
          (memref.dealloc qkv_weight)
          (memref.dealloc qkv_out)
          (memref.dealloc attn_weight)
          (memref.dealloc attn_out)
          (memref.dealloc residual_out)

          (func.return))))

    ;; External declarations
    (func.func {:sym_name "printF32"
                :function_type (-> [f32] [])
                :sym_visibility "private"})

    (func.func {:sym_name "printNewline"
                :function_type (-> [] [])
                :sym_visibility "private"})))
