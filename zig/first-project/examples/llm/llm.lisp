;; ============================================================================
;; GPT-2 Implementation - Direct Port from llm.c
;; Reference: https://github.com/karpathy/llm.c
;; ============================================================================

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "math.h")
(link-library "m")

;; ============================================================================
;; External Functions
;; ============================================================================

(extern-fn printf [fmt (Pointer U8)] -> I32)
(extern-fn sqrtf [x F32] -> F32)

;; ============================================================================
;; GPT2Config - matches C struct exactly
;; ============================================================================

(def GPT2Config (: Type)
  (Struct
    [max_seq_len I32]        ; max sequence length, e.g. 1024
    [vocab_size I32]         ; vocab size, e.g. 50257
    [padded_vocab_size I32]  ; padded to e.g. %128==0, 50304
    [num_layers I32]         ; number of layers, e.g. 12
    [num_heads I32]          ; number of heads in attention, e.g. 12
    [channels I32]))         ; number of channels, e.g. 768

;; ============================================================================
;; encoder_forward
;;
;; out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
;; inp is (B,T) of integers, holding the token ids at each (b,t) position
;; wte is (V,C) of token embeddings, short for "weight token embeddings"
;; wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
;; ============================================================================

(def encoder_forward (: (-> [(Pointer F32) (Pointer I32) (Pointer F32) (Pointer F32) I32 I32 I32] Nil))
  (fn [out inp wte wpe B T C]
    (let [b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            ;; seek to the output position in out[b,t,:]
            (let [out_bt_offset (: I32) (+ (* (* b T) C) (* t C))
                  ;; get the index of the token at inp[b, t]
                  ix (: I32) (pointer-index-read inp (+ (* b T) t))
                  ;; seek to the position in wte corresponding to the token
                  wte_ix_offset (: I32) (* ix C)
                  ;; seek to the position in wpe corresponding to the position
                  wpe_t_offset (: I32) (* t C)
                  i (: I32) 0]
              ;; add the two vectors and store the result in out[b,t,:]
              (while (< i C)
                (pointer-index-write! out (+ out_bt_offset i)
                  (+ (pointer-index-read wte (+ wte_ix_offset i))
                     (pointer-index-read wpe (+ wpe_t_offset i))))
                (set! i (+ i 1))))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; layernorm_forward
;;
;; Reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
;; Both inp and out are (B,T,C) of the activations
;; mean and rstd are (B,T) buffers, to be used later in backward pass
;; At each position (b,t) of the input, the C-dimensional vector
;; of activations gets normalized, then scaled and shifted
;; ============================================================================

(def layernorm_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32] Nil))
  (fn [out mean rstd inp weight bias B T C]
    (let [eps (: F32) 0.00001
          b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            ;; seek to the input position inp[b,t,:]
            (let [x_offset (: I32) (+ (* (* b T) C) (* t C))
                  ;; calculate the mean
                  m (: F32) 0.0
                  i (: I32) 0]
              (while (< i C)
                (set! m (+ m (pointer-index-read inp (+ x_offset i))))
                (set! i (+ i 1)))
              (set! m (/ m (+ C 0.0)))

              ;; calculate the variance (without any bias correction)
              (let [v (: F32) 0.0]
                (set! i 0)
                (while (< i C)
                  (let [xshift (: F32) (- (pointer-index-read inp (+ x_offset i)) m)]
                    (set! v (+ v (* xshift xshift))))
                  (set! i (+ i 1)))
                (set! v (/ v (+ C 0.0)))

                ;; calculate the rstd (reciprocal standard deviation)
                (let [s (: F32) (/ 1.0 (sqrtf (+ v eps)))
                      ;; seek to the output position in out[b,t,:]
                      out_bt_offset (: I32) (+ (* (* b T) C) (* t C))]
                  (set! i 0)
                  (while (< i C)
                    (let [n (: F32) (* s (- (pointer-index-read inp (+ x_offset i)) m))  ; normalize
                          o (: F32) (+ (* n (pointer-index-read weight i))
                                      (pointer-index-read bias i))]  ; scale and shift
                      (pointer-index-write! out (+ out_bt_offset i) o))  ; write
                    (set! i (+ i 1)))

                  ;; cache the mean and rstd for the backward pass later
                  (pointer-index-write! mean (+ (* b T) t) m)
                  (pointer-index-write! rstd (+ (* b T) t) s)
                  nil)))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; Test encoder_forward
;; ============================================================================

(def test_encoder_forward (: (-> [] I32))
  (fn []
    (printf (c-str "Testing encoder_forward...\n"))

    ;; Test dimensions: B=2, T=3, C=4
    (let [B (: I32) 2
          T (: I32) 3
          C (: I32) 4
          V (: I32) 10  ; vocab size
          ;; Allocate arrays
          out (: (Pointer F32)) (allocate-array F32 (* (* B T) C))
          inp (: (Pointer I32)) (allocate-array I32 (* B T))
          wte (: (Pointer F32)) (allocate-array F32 (* V C))
          wpe (: (Pointer F32)) (allocate-array F32 (* T C))]

      ;; Initialize inp with token ids: [1, 2, 3, 4, 5, 6]
      (pointer-index-write! inp 0 1)
      (pointer-index-write! inp 1 2)
      (pointer-index-write! inp 2 3)
      (pointer-index-write! inp 3 4)
      (pointer-index-write! inp 4 5)
      (pointer-index-write! inp 5 6)

      ;; Initialize wte (token embeddings) with sequential values
      (let [i (: I32) 0]
        (while (< i (* V C))
          (pointer-index-write! wte i (+ (* (+ i 0.0) 0.1) 1.0))
          (set! i (+ i 1))))

      ;; Initialize wpe (position embeddings) with sequential values
      (let [i (: I32) 0]
        (while (< i (* T C))
          (pointer-index-write! wpe i (+ (* (+ i 0.0) 0.01) 0.5))
          (set! i (+ i 1))))

      ;; Run encoder_forward
      (encoder_forward out inp wte wpe B T C)

      ;; Print first few outputs for verification
      (printf (c-str "First output values:\n"))
      (printf (c-str "  out[0] = %f\n") (pointer-index-read out 0))
      (printf (c-str "  out[1] = %f\n") (pointer-index-read out 1))
      (printf (c-str "  out[2] = %f\n") (pointer-index-read out 2))
      (printf (c-str "  out[3] = %f\n") (pointer-index-read out 3))

      ;; Clean up
      (deallocate-array out)
      (deallocate-array inp)
      (deallocate-array wte)
      (deallocate-array wpe)

      (printf (c-str "encoder_forward test completed!\n"))
      0)))

;; ============================================================================
;; Test layernorm_forward
;; ============================================================================

(def test_layernorm_forward (: (-> [] I32))
  (fn []
    (printf (c-str "Testing layernorm_forward...\n"))

    ;; Test dimensions: B=2, T=2, C=3
    (let [B (: I32) 2
          T (: I32) 2
          C (: I32) 3
          ;; Allocate arrays
          out (: (Pointer F32)) (allocate-array F32 (* (* B T) C))
          mean (: (Pointer F32)) (allocate-array F32 (* B T))
          rstd (: (Pointer F32)) (allocate-array F32 (* B T))
          inp (: (Pointer F32)) (allocate-array F32 (* (* B T) C))
          weight (: (Pointer F32)) (allocate-array F32 C 1.0)
          bias (: (Pointer F32)) (allocate-array F32 C 0.0)]

      ;; Initialize inp with sequential values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      (let [i (: I32) 0]
        (while (< i (* (* B T) C))
          (pointer-index-write! inp i (+ (+ i 0.0) 1.0))
          (set! i (+ i 1))))

      ;; Run layernorm_forward
      (layernorm_forward out mean rstd inp weight bias B T C)

      ;; Print first few outputs for verification
      (printf (c-str "First output values:\n"))
      (printf (c-str "  out[0] = %f\n") (pointer-index-read out 0))
      (printf (c-str "  out[1] = %f\n") (pointer-index-read out 1))
      (printf (c-str "  out[2] = %f\n") (pointer-index-read out 2))
      (printf (c-str "  mean[0] = %f\n") (pointer-index-read mean 0))
      (printf (c-str "  rstd[0] = %f\n") (pointer-index-read rstd 0))

      ;; Clean up
      (deallocate-array out)
      (deallocate-array mean)
      (deallocate-array rstd)
      (deallocate-array inp)
      (deallocate-array weight)
      (deallocate-array bias)

      (printf (c-str "layernorm_forward test completed!\n"))
      0)))

;; ============================================================================
;; Main
;; ============================================================================

(def main-fn (: (-> [] I32))
  (fn []
    (test_encoder_forward)
    (printf (c-str "\n"))
    (test_layernorm_forward)))

(main-fn)
