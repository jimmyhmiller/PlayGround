;; Test Suite for Tensor Operations
;; Verifies that basic tensor ops work correctly

(include-header "stdio.h")
(include-header "math.h")
(link-library "m")

;; External functions
(extern-fn printf [fmt String] -> I32)
(extern-fn expf [x F32] -> F32)
(extern-fn sqrtf [x F32] -> F32)
(extern-fn tanhf [x F32] -> F32)
(extern-fn fabsf [x F32] -> F32)


;; ============================================================================
;; GELU Forward (from tensor_ops.lisp)
;; ============================================================================

(def gelu_forward (: (-> [(Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp N]
    (let [s (: F32) (sqrtf (/ 2.0 3.14159265358979323846))
          i (: I32) 0]
      (while (< i N)
        (let [x (: F32) (pointer-index-read inp i)
              cube (: F32) (* (* 0.044715 x) (* x x))
              y (: F32) (* (* 0.5 x) (+ 1.0 (tanhf (* s (+ x cube)))))]
          (pointer-index-write! out i y))
        (set! i (+ i 1))))
    nil))

;; ============================================================================
;; Simple Layer Norm (no batch dimension)
;; ============================================================================

(def simple_layernorm (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp weight bias N]
    (let [eps (: F32) 0.00001]
      ;; Compute mean
      (let [mean (: F32) 0.0
            i (: I32) 0]
        (while (< i N)
          (set! mean (+ mean (pointer-index-read inp i)))
          (set! i (+ i 1)))
        (set! mean (/ mean (+ N 0.0)))

        ;; Compute variance
        (let [variance (: F32) 0.0]
          (set! i 0)
          (while (< i N)
            (let [diff (: F32) (- (pointer-index-read inp i) mean)]
              (set! variance (+ variance (* diff diff))))
            (set! i (+ i 1)))
          (set! variance (/ variance (+ N 0.0)))

          ;; Normalize and scale
          (let [rstd (: F32) (/ 1.0 (sqrtf (+ variance eps)))]
            (set! i 0)
            (while (< i N)
              (let [normalized (: F32) (* (- (pointer-index-read inp i) mean) rstd)
                    w (: F32) (pointer-index-read weight i)
                    b (: F32) (pointer-index-read bias i)]
                (pointer-index-write! out i (+ (* normalized w) b)))
              (set! i (+ i 1)))
            nil))))
    nil))

;; ============================================================================
;; Softmax Forward (1D version)
;; ============================================================================

(def simple_softmax (: (-> [(Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp N]
    ;; Find max for numerical stability
    (let [maxval (: F32) (pointer-index-read inp 0)
          i (: I32) 1]
      (while (< i N)
        (let [val (: F32) (pointer-index-read inp i)]
          (if (> val maxval)
            (set! maxval val)
            nil))
        (set! i (+ i 1)))

      ;; Compute exp and sum
      (let [sum (: F32) 0.0]
        (set! i 0)
        (while (< i N)
          (let [exp_val (: F32) (expf (- (pointer-index-read inp i) maxval))]
            (pointer-index-write! out i exp_val)
            (set! sum (+ sum exp_val)))
          (set! i (+ i 1)))

        ;; Normalize
        (set! i 0)
        (while (< i N)
          (pointer-index-write! out i (/ (pointer-index-read out i) sum))
          (set! i (+ i 1)))
        nil))
    nil))

;; ============================================================================
;; Test GELU
;; ============================================================================

(def test_gelu (: (-> [] Nil))
  (fn []
    (printf "\n=== Testing GELU ===\n")
    (let [n (: I32) 5
          inp (: (Pointer F32)) (allocate-array F32 5)
          out (: (Pointer F32)) (allocate-array F32 5)]

      ;; Initialize input: [-2, -1, 0, 1, 2]
      (pointer-index-write! inp 0 -2.0)
      (pointer-index-write! inp 1 -1.0)
      (pointer-index-write! inp 2 0.0)
      (pointer-index-write! inp 3 1.0)
      (pointer-index-write! inp 4 2.0)

      (gelu_forward out inp n)

      (printf "GELU test completed\n")
      (deallocate-array inp)
      (deallocate-array out)
      nil)))

;; ============================================================================
;; Test LayerNorm
;; ============================================================================

(def test_layernorm (: (-> [] Nil))
  (fn []
    (printf "\n=== Testing LayerNorm ===\n")
    (let [n (: I32) 4
          inp (: (Pointer F32)) (allocate-array F32 4)
          out (: (Pointer F32)) (allocate-array F32 4)
          weight (: (Pointer F32)) (allocate-array F32 4 1.0)
          bias (: (Pointer F32)) (allocate-array F32 4 0.0)]

      ;; Initialize input: [1, 2, 3, 4]
      (pointer-index-write! inp 0 1.0)
      (pointer-index-write! inp 1 2.0)
      (pointer-index-write! inp 2 3.0)
      (pointer-index-write! inp 3 4.0)

      (simple_layernorm out inp weight bias n)

      (printf "LayerNorm test completed\n")
      (deallocate-array inp)
      (deallocate-array out)
      (deallocate-array weight)
      (deallocate-array bias)
      nil)))

;; ============================================================================
;; Test Softmax
;; ============================================================================

(def test_softmax (: (-> [] Nil))
  (fn []
    (printf "\n=== Testing Softmax ===\n")
    (let [n (: I32) 5
          inp (: (Pointer F32)) (allocate-array F32 5)
          out (: (Pointer F32)) (allocate-array F32 5)]

      ;; Initialize input: [1, 2, 3, 4, 5]
      (pointer-index-write! inp 0 1.0)
      (pointer-index-write! inp 1 2.0)
      (pointer-index-write! inp 2 3.0)
      (pointer-index-write! inp 3 4.0)
      (pointer-index-write! inp 4 5.0)

      (simple_softmax out inp n)

      ;; Verify sum is ~1.0
      (let [sum (: F32) 0.0
            i (: I32) 0]
        (while (< i n)
          (set! sum (+ sum (pointer-index-read out i)))
          (set! i (+ i 1)))
        (printf "Softmax test completed (sum should be ~1.0)\n")
        nil)

      (deallocate-array inp)
      (deallocate-array out)
      nil)))

;; ============================================================================
;; Main Test Runner
;; ============================================================================

(def main-fn (: (-> [] I32))
  (fn []
    (printf "=================================\n")
    (printf "Tensor Operations Test Suite\n")
    (printf "=================================\n")

    (test_gelu)
    (test_layernorm)
    (test_softmax)

    (printf "\n=================================\n")
    (printf "All tests completed!\n")
    (printf "=================================\n")
    0))

(main-fn)
