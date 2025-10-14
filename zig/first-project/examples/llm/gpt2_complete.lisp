;; ============================================================================
;; GPT-2 Implementation in Lisp (Single File)
;; Based on Andrej Karpathy's llm.c
;; ============================================================================

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "string.h")
(include-header "math.h")
(link-library "m")

;; ============================================================================
;; External Function Declarations
;; ============================================================================

;; Math functions
(extern-fn expf [x F32] -> F32)
(extern-fn sqrtf [x F32] -> F32)
(extern-fn tanhf [x F32] -> F32)
(extern-fn logf [x F32] -> F32)
(extern-fn powf [x F32 y F32] -> F32)

;; File I/O
(extern-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer Void))
(extern-fn fclose [stream (Pointer Void)] -> I32)
(extern-fn fread [ptr (Pointer Void) size U64 nmemb U64 stream (Pointer Void)] -> U64)
(extern-fn fwrite [ptr (Pointer Void) size U64 nmemb U64 stream (Pointer Void)] -> U64)
(extern-fn fprintf [stream (Pointer Void) fmt (Pointer U8)] -> I32)

;; Memory
(extern-fn malloc [size U64] -> (Pointer Void))
(extern-fn calloc [nmemb U64 size U64] -> (Pointer Void))
(extern-fn free [ptr (Pointer Void)] -> Nil)
(extern-fn memset [s (Pointer Void) c I32 n U64] -> (Pointer Void))

;; I/O
(extern-fn printf [fmt (Pointer U8)] -> I32)

;; ============================================================================
;; Configuration
;; ============================================================================

(def GPT2Config (: Type)
  (Struct
    [max_seq_len I32]
    [vocab_size I32]
    [num_layers I32]
    [num_heads I32]
    [channels I32]))

;; ============================================================================
;; Tensor Operations
;; ============================================================================

;; LayerNorm Forward
;; out: (B, T, C) - output
;; inp: (B, T, C) - input
;; weight: (C) - scale
;; bias: (C) - shift
(def layernorm_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32] Nil))
  (fn [out inp weight bias B T C]
    (let [eps (: F32) 0.00001
          bt (: I32) 0]
      (while (< bt (* B T))
        ;; Compute mean
        (let [bt_offset (: I32) (* bt C)
              mean (: F32) 0.0
              c (: I32) 0]
          (while (< c C)
            (set! mean (+ mean (pointer-index-read inp (+ bt_offset c))))
            (set! c (+ c 1)))
          (set! mean (/ mean (+ C 0.0)))

          ;; Compute variance
          (let [variance (: F32) 0.0]
            (set! c 0)
            (while (< c C)
              (let [diff (: F32) (- (pointer-index-read inp (+ bt_offset c)) mean)]
                (set! variance (+ variance (* diff diff))))
              (set! c (+ c 1)))
            (set! variance (/ variance (+ C 0.0)))

            ;; Normalize and scale
            (let [rstd (: F32) (/ 1.0 (sqrtf (+ variance eps)))]
              (set! c 0)
              (while (< c C)
                (let [idx (: I32) (+ bt_offset c)
                      n (: F32) (* (- (pointer-index-read inp idx) mean) rstd)
                      w (: F32) (pointer-index-read weight c)
                      b (: F32) (pointer-index-read bias c)]
                  (pointer-index-write! out idx (+ (* n w) b)))
                (set! c (+ c 1)))
              nil)))
        (set! bt (+ bt 1))))
    nil))

;; Matrix Multiplication Forward
;; out: (B, T, OC)
;; inp: (B, T, C)
;; weight: (OC, C)
;; bias: (OC) or null
(def matmul_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32 I32] Nil))
  (fn [out inp weight bias B T C OC]
    (let [bt (: I32) 0]
      (while (< bt (* B T))
        (let [bt_out_offset (: I32) (* bt OC)
              bt_inp_offset (: I32) (* bt C)
              o (: I32) 0]
          (while (< o OC)
            (let [val (: F32) (if (pointer-equal? bias pointer-null)
                                0.0
                                (pointer-index-read bias o))
                  wrow_offset (: I32) (* o C)
                  c (: I32) 0]
              (while (< c C)
                (set! val (+ val (* (pointer-index-read inp (+ bt_inp_offset c))
                                   (pointer-index-read weight (+ wrow_offset c)))))
                (set! c (+ c 1)))
              (pointer-index-write! out (+ bt_out_offset o) val))
            (set! o (+ o 1))))
        (set! bt (+ bt 1))))
    nil))

;; GELU Activation Forward
(def gelu_forward (: (-> [(Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp N]
    (let [s (: F32) (sqrtf (/ 2.0 3.14159265))
          i (: I32) 0]
      (while (< i N)
        (let [x (: F32) (pointer-index-read inp i)
              cube (: F32) (* (* 0.044715 x) (* x x))
              y (: F32) (* (* 0.5 x) (+ 1.0 (tanhf (* s (+ x cube)))))]
          (pointer-index-write! out i y))
        (set! i (+ i 1))))
    nil))

;; Residual Connection Forward
(def residual_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp1 inp2 N]
    (let [i (: I32) 0]
      (while (< i N)
        (pointer-index-write! out i (+ (pointer-index-read inp1 i)
                                       (pointer-index-read inp2 i)))
        (set! i (+ i 1))))
    nil))

;; ============================================================================
;; Test Functions
;; ============================================================================

(def test_layernorm (: (-> [] Nil))
  (fn []
    (printf (c-str "Testing LayerNorm...\n"))
    (let [B (: I32) 2
          T (: I32) 3
          C (: I32) 4
          size (: I32) (* (* B T) C)
          inp (: (Pointer F32)) (allocate-array F32 24)
          out (: (Pointer F32)) (allocate-array F32 24)
          weight (: (Pointer F32)) (allocate-array F32 4 1.0)
          bias (: (Pointer F32)) (allocate-array F32 4 0.0)]

      ;; Initialize input with sequential values
      (let [i (: I32) 0]
        (while (< i size)
          (pointer-index-write! inp i (+ (+ i 0.0) 1.0))
          (set! i (+ i 1))))

      ;; Run layernorm
      (layernorm_forward out inp weight bias B T C)

      (printf (c-str "LayerNorm test completed\n"))
      (deallocate-array inp)
      (deallocate-array out)
      (deallocate-array weight)
      (deallocate-array bias)
      nil)))

(def test_matmul (: (-> [] Nil))
  (fn []
    (printf (c-str "Testing MatMul...\n"))
    (let [B (: I32) 1
          T (: I32) 2
          C (: I32) 3
          OC (: I32) 4
          inp (: (Pointer F32)) (allocate-array F32 6)
          weight (: (Pointer F32)) (allocate-array F32 12)
          out (: (Pointer F32)) (allocate-array F32 8)]

      ;; Initialize input
      (let [i (: I32) 0]
        (while (< i 6)
          (pointer-index-write! inp i (+ (+ i 0.0) 1.0))
          (set! i (+ i 1))))

      ;; Initialize weight
      (let [i (: I32) 0]
        (while (< i 12)
          (pointer-index-write! weight i 0.1)
          (set! i (+ i 1))))

      ;; Run matmul
      (matmul_forward out inp weight pointer-null B T C OC)

      (printf (c-str "MatMul test completed\n"))
      (deallocate-array inp)
      (deallocate-array weight)
      (deallocate-array out)
      nil)))

(def test_gelu (: (-> [] Nil))
  (fn []
    (printf (c-str "Testing GELU...\n"))
    (let [N (: I32) 5
          inp (: (Pointer F32)) (allocate-array F32 5)
          out (: (Pointer F32)) (allocate-array F32 5)]

      ;; Initialize: [-2, -1, 0, 1, 2]
      (pointer-index-write! inp 0 -2.0)
      (pointer-index-write! inp 1 -1.0)
      (pointer-index-write! inp 2 0.0)
      (pointer-index-write! inp 3 1.0)
      (pointer-index-write! inp 4 2.0)

      ;; Run GELU
      (gelu_forward out inp N)

      (printf (c-str "GELU results: [%f, %f, %f, %f, %f]\n")
              (pointer-index-read out 0)
              (pointer-index-read out 1)
              (pointer-index-read out 2)
              (pointer-index-read out 3)
              (pointer-index-read out 4))

      (deallocate-array inp)
      (deallocate-array out)
      nil)))

(def test_residual (: (-> [] Nil))
  (fn []
    (printf (c-str "Testing Residual...\n"))
    (let [N (: I32) 4
          inp1 (: (Pointer F32)) (allocate-array F32 4)
          inp2 (: (Pointer F32)) (allocate-array F32 4)
          out (: (Pointer F32)) (allocate-array F32 4)]

      ;; Initialize
      (let [i (: I32) 0]
        (while (< i N)
          (pointer-index-write! inp1 i (+ (+ i 0.0) 1.0))
          (pointer-index-write! inp2 i (+ (+ i 0.0) 10.0))
          (set! i (+ i 1))))

      ;; Run residual
      (residual_forward out inp1 inp2 N)

      (printf (c-str "Residual results: [%f, %f, %f, %f]\n")
              (pointer-index-read out 0)
              (pointer-index-read out 1)
              (pointer-index-read out 2)
              (pointer-index-read out 3))

      (deallocate-array inp1)
      (deallocate-array inp2)
      (deallocate-array out)
      nil)))

;; ============================================================================
;; Main
;; ============================================================================

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "=================================\n"))
    (printf (c-str "GPT-2 Tensor Operations Test\n"))
    (printf (c-str "=================================\n\n"))

    (test_layernorm)
    (test_matmul)
    (test_gelu)
    (test_residual)

    (printf (c-str "\n=================================\n"))
    (printf (c-str "All tests completed!\n"))
    (printf (c-str "=================================\n"))
    0))

(main-fn)
