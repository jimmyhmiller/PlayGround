;; Multi-Head Causal Self-Attention for GPT-2
;;
;; This file implements attention using the gpu.launch pattern.
;; Uses the async GPU pattern with explicit device memory management.
;;
;; For B=1, T=64, C=768, NH=12 (GPT-2 Small):
;; - head_size (hs) = 768/12 = 64
;; - QKV input: (B, T, 3*C) where Q, K, V each have size C
;; - preatt: (B, NH, T, T) - pre-softmax attention scores
;; - att: (B, NH, T, T) - post-softmax attention weights
;; - out: (B, T, C) - attention output
;;
;; Computation per (batch, time, head):
;; 1. Q[t] dot K[t2] for t2 <= t (causal masking)
;; 2. Scale by 1/sqrt(head_size)
;; 3. Softmax over t2 dimension
;; 4. Weighted sum of V vectors
;;
;; GPU Launch Strategy:
;; - Grid: NH blocks (one per head)
;; - Threads: T threads per block (one per time position)
;; - Each thread computes the full attention for its position

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)
(require-dialect math)

;; GPU compilation pipeline (AMD ROCm)
(compilation
  (target rocm
    (pass convert-scf-to-cf)
    (pass gpu-async-region)
    (pass rocdl-attach-target :chip "gfx1151")
    (pass convert-gpu-to-rocdl)
    (pass gpu-module-to-binary)
    (pass gpu-to-llvm)
    (pass async-to-async-runtime)
    (pass convert-async-to-llvm)
    (pass convert-arith-to-llvm)
    (pass convert-math-to-llvm)
    (pass convert-vector-to-llvm)
    (pass convert-index-to-llvm)
    (pass finalize-memref-to-llvm)
    (pass convert-func-to-llvm)
    (pass convert-cf-to-llvm)
    (pass reconcile-unrealized-casts)))

;; Reference MLIR for the attention kernel:
;;
;; module attributes {gpu.container_module} {
;;   gpu.module @kernels {
;;     gpu.func @attention_head(
;;       %qkv: memref<1x64x2304xf32>,   // (B,T,3*C) for C=768
;;       %out: memref<1x64x768xf32>,     // (B,T,C)
;;       %preatt: memref<1x12x64x64xf32>, // (B,NH,T,T)
;;       %att: memref<1x12x64x64xf32>     // (B,NH,T,T)
;;     ) kernel {
;;       // Block = head, Thread = time position
;;       %h = gpu.block_id x     // head index
;;       %t = gpu.thread_id y    // time position
;;
;;       // Pass 1: Compute Q[t] dot K[t2], store to preatt, track max
;;       // Pass 2: exp(score - max), sum
;;       // Pass 3: Normalize
;;       // Pass 4: Weighted sum of V
;;       gpu.return
;;     }
;;   }
;; }
;;
;; The kernel computes attention for position t at head h:
;; - Query at qkv[0, t, h*hs : (h+1)*hs]
;; - Keys at qkv[0, t2, h*hs + C : h*hs + C + hs] for t2 <= t
;; - Values at qkv[0, t2, h*hs + 2*C : h*hs + 2*C + hs] for t2 <= t

;; Note: Full implementation is in /tmp/attention_simple.mlir
;; This file documents the structure for the lisp compiler to generate.
