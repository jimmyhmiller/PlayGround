;; Core Tensor Operations for GPT-2
;; Implements the fundamental operations needed for transformer models

(include-header "math.h")
(link-library "m")

;; Math function externs
(extern-fn expf [x F32] -> F32)
(extern-fn sqrtf [x F32] -> F32)
(extern-fn tanhf [x F32] -> F32)
(extern-fn logf [x F32] -> F32)
(extern-fn powf [x F32 y F32] -> F32)

;; ============================================================================
;; Encoder Forward: Token + Position Embeddings
;; ============================================================================
;; out: (B, T, C)
;; inp: (B, T) token indices
;; wte: (V, C) token embedding weights
;; wpe: (maxT, C) position embedding weights
;; B = batch size, T = sequence length, C = channels, V = vocab size

(def encoder_forward (: (-> [(Pointer F32) (Pointer I32) (Pointer F32) (Pointer F32) I32 I32 I32] Nil))
  (fn [out inp wte wpe B T C]
    (let [b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            (let [out_bt (: (Pointer F32)) (+ out (* (* (+ (* b T) t) C) 4))
                  ix (: I32) (pointer-index-read inp (+ (* b T) t))
                  wte_ix (: (Pointer F32)) (+ wte (* (* ix C) 4))
                  wpe_t (: (Pointer F32)) (+ wpe (* (* t C) 4))
                  c (: I32) 0]
              (while (< c C)
                (let [token_emb (: F32) (pointer-index-read wte_ix c)
                      pos_emb (: F32) (pointer-index-read wpe_t c)]
                  (pointer-index-write! out_bt c (+ token_emb pos_emb)))
                (set! c (+ c 1))))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; Layer Normalization Forward
;; ============================================================================
;; out: (B, T, C) normalized output
;; mean: (B, T) means
;; rstd: (B, T) reciprocal standard deviations
;; inp: (B, T, C) input
;; weight: (C) scale parameters
;; bias: (C) shift parameters

(def layernorm_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32] Nil))
  (fn [out mean rstd inp weight bias B T C]
    (let [eps (: F32) 1e-5
          b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            ;; Compute mean
            (let [m (: F32) 0.0
                  c (: I32) 0
                  inp_bt (: (Pointer F32)) (+ inp (* (* (+ (* b T) t) C) 4))]
              (while (< c C)
                (set! m (+ m (pointer-index-read inp_bt c)))
                (set! c (+ c 1)))
              (set! m (/ m (+ C 0.0)))

              ;; Compute variance
              (let [v (: F32) 0.0]
                (set! c 0)
                (while (< c C)
                  (let [diff (: F32) (- (pointer-index-read inp_bt c) m)]
                    (set! v (+ v (* diff diff))))
                  (set! c (+ c 1)))
                (set! v (/ v (+ C 0.0)))

                ;; Compute reciprocal standard deviation
                (let [s (: F32) (/ 1.0 (sqrtf (+ v eps)))
                      out_bt (: (Pointer F32)) (+ out (* (* (+ (* b T) t) C) 4))
                      mean_bt_idx (: I32) (+ (* b T) t)]
                  (pointer-index-write! mean mean_bt_idx m)
                  (pointer-index-write! rstd mean_bt_idx s)

                  ;; Normalize and scale
                  (set! c 0)
                  (while (< c C)
                    (let [n (: F32) (* (- (pointer-index-read inp_bt c) m) s)
                          w (: F32) (pointer-index-read weight c)
                          b_param (: F32) (pointer-index-read bias c)]
                      (pointer-index-write! out_bt c (+ (* n w) b_param)))
                    (set! c (+ c 1)))
                  nil)))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; Matrix Multiplication Forward
;; ============================================================================
;; out: (B, T, OC) output
;; inp: (B, T, C) input
;; weight: (OC, C) weight matrix (transposed in memory)
;; bias: (OC) bias vector (can be null)

(def matmul_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32 I32] Nil))
  (fn [out inp weight bias B T C OC]
    (let [b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            (let [out_bt (: (Pointer F32)) (+ out (* (* (+ (* b T) t) OC) 4))
                  inp_bt (: (Pointer F32)) (+ inp (* (* (+ (* b T) t) C) 4))
                  o (: I32) 0]
              (while (< o OC)
                (let [val (: F32) (if (pointer-equal? bias pointer-null)
                                    0.0
                                    (pointer-index-read bias o))
                      wrow (: (Pointer F32)) (+ weight (* (* o C) 4))
                      c (: I32) 0]
                  (while (< c C)
                    (set! val (+ val (* (pointer-index-read inp_bt c)
                                       (pointer-index-read wrow c))))
                    (set! c (+ c 1)))
                  (pointer-index-write! out_bt o val))
                (set! o (+ o 1))))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; GELU Activation Forward
;; ============================================================================
;; Gaussian Error Linear Unit activation function
;; GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

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
;; Softmax Forward
;; ============================================================================
;; Numerically stable softmax over the last dimension

(def softmax_forward (: (-> [(Pointer F32) (Pointer F32) I32 I32] Nil))
  (fn [probs logits B T]
    (let [b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            (let [logits_bt (: (Pointer F32)) (+ logits (* (* (+ (* b T) t) T) 4))
                  probs_bt (: (Pointer F32)) (+ probs (* (* (+ (* b T) t) T) 4))]

              ;; Find max for numerical stability
              (let [maxval (: F32) (pointer-index-read logits_bt 0)
                    i (: I32) 1]
                (while (< i T)
                  (let [val (: F32) (pointer-index-read logits_bt i)]
                    (if (> val maxval)
                      (set! maxval val)
                      nil))
                  (set! i (+ i 1)))

                ;; Compute exp and sum
                (let [sum (: F32) 0.0]
                  (set! i 0)
                  (while (< i T)
                    (let [exp_val (: F32) (expf (- (pointer-index-read logits_bt i) maxval))]
                      (pointer-index-write! probs_bt i exp_val)
                      (set! sum (+ sum exp_val)))
                    (set! i (+ i 1)))

                  ;; Normalize
                  (set! i 0)
                  (while (< i T)
                    (pointer-index-write! probs_bt i (/ (pointer-index-read probs_bt i) sum))
                    (set! i (+ i 1)))
                  nil)))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; Residual Connection Forward
;; ============================================================================
;; out = inp1 + inp2 (element-wise addition)

(def residual_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp1 inp2 N]
    (let [i (: I32) 0]
      (while (< i N)
        (pointer-index-write! out i (+ (pointer-index-read inp1 i)
                                       (pointer-index-read inp2 i)))
        (set! i (+ i 1))))
    nil))
