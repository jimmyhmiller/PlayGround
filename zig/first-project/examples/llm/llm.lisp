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
(extern-fn tanhf [x F32] -> F32)
(extern-fn expf [x F32] -> F32)
(extern-fn malloc [size I32] -> (Pointer U8))
(extern-fn free [ptr (Pointer U8)] -> Nil)
(extern-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer U8))
(extern-fn fclose [file (Pointer U8)] -> I32)
(extern-fn fread [ptr (Pointer U8) size I32 count I32 file (Pointer U8)] -> I32)
(extern-fn fseek [file (Pointer U8) offset I32 whence I32] -> I32)
(extern-fn exit [status I32] -> Nil)

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
;; ParameterTensors - holds all model weights (16 parameter tensors)
;; ============================================================================

(def ParameterTensors (: Type)
  (Struct
    [wte (Pointer F32)]      ; (V, C) token embeddings
    [wpe (Pointer F32)]      ; (maxT, C) position embeddings
    [ln1w (Pointer F32)]     ; (L, C) layernorm1 weights
    [ln1b (Pointer F32)]     ; (L, C) layernorm1 biases
    [qkvw (Pointer F32)]     ; (L, 3*C, C) qkv projection weights
    [qkvb (Pointer F32)]     ; (L, 3*C) qkv projection biases
    [attprojw (Pointer F32)] ; (L, C, C) attention output projection weights
    [attprojb (Pointer F32)] ; (L, C) attention output projection biases
    [ln2w (Pointer F32)]     ; (L, C) layernorm2 weights
    [ln2b (Pointer F32)]     ; (L, C) layernorm2 biases
    [fcw (Pointer F32)]      ; (L, 4*C, C) MLP first layer weights
    [fcb (Pointer F32)]      ; (L, 4*C) MLP first layer biases
    [fcprojw (Pointer F32)]  ; (L, C, 4*C) MLP projection weights
    [fcprojb (Pointer F32)]  ; (L, C) MLP projection biases
    [lnfw (Pointer F32)]     ; (C) final layernorm weights
    [lnfb (Pointer F32)]))   ; (C) final layernorm biases

;; ============================================================================
;; ActivationTensors - holds intermediate computation buffers (23 tensors)
;; ============================================================================

(def ActivationTensors (: Type)
  (Struct
    [encoded (Pointer F32)]    ; (B, T, C)
    [ln1 (Pointer F32)]        ; (L, B, T, C)
    [ln1_mean (Pointer F32)]   ; (L, B, T)
    [ln1_rstd (Pointer F32)]   ; (L, B, T)
    [qkv (Pointer F32)]        ; (L, B, T, 3*C)
    [atty (Pointer F32)]       ; (L, B, T, C)
    [preatt (Pointer F32)]     ; (L, B, NH, T, T)
    [att (Pointer F32)]        ; (L, B, NH, T, T)
    [attproj (Pointer F32)]    ; (L, B, T, C)
    [residual2 (Pointer F32)]  ; (L, B, T, C)
    [ln2 (Pointer F32)]        ; (L, B, T, C)
    [ln2_mean (Pointer F32)]   ; (L, B, T)
    [ln2_rstd (Pointer F32)]   ; (L, B, T)
    [fch (Pointer F32)]        ; (L, B, T, 4*C)
    [fch_gelu (Pointer F32)]   ; (L, B, T, 4*C)
    [fcproj (Pointer F32)]     ; (L, B, T, C)
    [residual3 (Pointer F32)]  ; (L, B, T, C)
    [lnf (Pointer F32)]        ; (B, T, C)
    [lnf_mean (Pointer F32)]   ; (B, T)
    [lnf_rstd (Pointer F32)]   ; (B, T)
    [logits (Pointer F32)]     ; (B, T, Vp)
    [probs (Pointer F32)]      ; (B, T, Vp)
    [losses (Pointer F32)]))   ; (B, T)

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
;; matmul_forward_naive
;;\
;; The most naive implementation of matrix multiplication
;; inp is (B,T,C), weight is (OC, C), bias is (OC) or NULL
;; out will be (B,T,OC)
;; ============================================================================

(def matmul_forward_naive (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32 I32] Nil))
  (fn [out inp weight bias B T C OC]
    (let [b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            (let [bt (: I32) (+ (* b T) t)
                  o (: I32) 0]
              (while (< o OC)
                (let [val (: F32) (if (pointer-equal? bias pointer-null)
                                    0.0
                                    (pointer-index-read bias o))
                      i (: I32) 0]
                  (while (< i C)
                    (set! val (+ val (* (pointer-index-read inp (+ (* bt C) i))
                                       (pointer-index-read weight (+ (* o C) i)))))
                    (set! i (+ i 1)))
                  (pointer-index-write! out (+ (* bt OC) o) val))
                (set! o (+ o 1))))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; residual_forward
;;
;; Simple element-wise addition: out[i] = inp1[i] + inp2[i]
;; ============================================================================

(def residual_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp1 inp2 N]
    (let [i (: I32) 0]
      (while (< i N)
        (pointer-index-write! out i
          (+ (pointer-index-read inp1 i)
             (pointer-index-read inp2 i)))
        (set! i (+ i 1))))
    nil))

;; ============================================================================
;; gelu_forward
;;
;; (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
;; GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
;; ============================================================================

(def gelu_forward (: (-> [(Pointer F32) (Pointer F32) I32] Nil))
  (fn [out inp N]
    (let [pi_value (: F32) 3.14159265358979323846
          GELU_SCALING_FACTOR (: F32) (sqrtf (/ 2.0 pi_value))
          i (: I32) 0]
      (while (< i N)
        (let [x (: F32) (pointer-index-read inp i)
              cube (: F32) (* (* (* 0.044715 x) x) x)
              result (: F32) (* (* 0.5 x) (+ 1.0 (tanhf (* GELU_SCALING_FACTOR (+ x cube)))))]
          (pointer-index-write! out i result))
        (set! i (+ i 1))))
    nil))

;; ============================================================================
;; attention_forward
;;
;; Multi-head self-attention with causal mask
;; Input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
;; preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
;; Output is (B, T, C)
;; ============================================================================

(def attention_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32 I32] Nil))
  (fn [out preatt att inp B T C NH]
    (let [C3 (: I32) (* C 3)
          hs (: I32) (div C NH)  ; head size
          scale (: F32) (/ 1.0 (sqrtf (+ hs 0.0)))
          b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            (let [h (: I32) 0]
              (while (< h NH)
                (let [query_t_offset (: I32) (+ (+ (* (* b T) C3) (* t C3)) (* h hs))
                      preatt_bth_offset (: I32) (+ (+ (* (* (* b NH) T) T) (* (* h T) T)) (* t T))
                      att_bth_offset (: I32) (+ (+ (* (* (* b NH) T) T) (* (* h T) T)) (* t T))

                      ;; pass 1: calculate query dot key and maxval
                      maxval (: F32) (- 0.0 10000.0)
                      t2 (: I32) 0]

                  (while (<= t2 t)
                    (let [key_t2_offset (: I32) (+ (+ (+ (* (* b T) C3) (* t2 C3)) (* h hs)) C)
                          val (: F32) 0.0
                          i (: I32) 0]
                      ;; (query_t) dot (key_t2)
                      (while (< i hs)
                        (set! val (+ val (* (pointer-index-read inp (+ query_t_offset i))
                                           (pointer-index-read inp (+ key_t2_offset i)))))
                        (set! i (+ i 1)))
                      (set! val (* val scale))
                      (if (> val maxval)
                        (set! maxval val)
                        nil)
                      (pointer-index-write! preatt (+ preatt_bth_offset t2) val))
                    (set! t2 (+ t2 1)))

                  ;; pass 2: calculate the exp and keep track of sum
                  (let [expsum (: F32) 0.0]
                    (set! t2 0)
                    (while (<= t2 t)
                      (let [expv (: F32) (expf (- (pointer-index-read preatt (+ preatt_bth_offset t2)) maxval))]
                        (set! expsum (+ expsum expv))
                        (pointer-index-write! att (+ att_bth_offset t2) expv))
                      (set! t2 (+ t2 1)))

                    (let [expsum_inv (: F32) (if (= expsum 0.0)
                                               0.0
                                               (/ 1.0 expsum))]
                      ;; pass 3: normalize to get the softmax
                      (set! t2 0)
                      (while (< t2 T)
                        (if (<= t2 t)
                          (pointer-index-write! att (+ att_bth_offset t2)
                            (* (pointer-index-read att (+ att_bth_offset t2)) expsum_inv))
                          (pointer-index-write! att (+ att_bth_offset t2) 0.0))
                        (set! t2 (+ t2 1)))

                      ;; pass 4: accumulate weighted values into the output of attention
                      (let [out_bth_offset (: I32) (+ (+ (* (* b T) C) (* t C)) (* h hs))
                            i (: I32) 0]
                        (while (< i hs)
                          (pointer-index-write! out (+ out_bth_offset i) 0.0)
                          (set! i (+ i 1)))
                        (set! t2 0)
                        (while (<= t2 t)
                          (let [value_t2_offset (: I32) (+ (+ (+ (* (* b T) C3) (* t2 C3)) (* h hs)) (* C 2))
                                att_btht2 (: F32) (pointer-index-read att (+ att_bth_offset t2))]
                            (set! i 0)
                            (while (< i hs)
                              (pointer-index-write! out (+ out_bth_offset i)
                                (+ (pointer-index-read out (+ out_bth_offset i))
                                   (* att_btht2 (pointer-index-read inp (+ value_t2_offset i)))))
                              (set! i (+ i 1))))
                          (set! t2 (+ t2 1)))))))
                (set! h (+ h 1))))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; softmax_forward
;;
;; Softmax with numerical stability (subtracts max before exp)
;; Input: logits is (B,T,Vp) of the unnormalized log probabilities
;; Output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
;; Vp is the padded vocab size (for efficiency), V is the "real" vocab size
;; ============================================================================

(def softmax_forward (: (-> [(Pointer F32) (Pointer F32) I32 I32 I32 I32] Nil))
  (fn [probs logits B T V Vp]
    (let [b (: I32) 0]
      (while (< b B)
        (let [t (: I32) 0]
          (while (< t T)
            (let [logits_bt_offset (: I32) (+ (* (* b T) Vp) (* t Vp))
                  probs_bt_offset (: I32) (+ (* (* b T) Vp) (* t Vp))

                  ;; Find maxval for numerical stability
                  maxval (: F32) (- 0.0 10000.0)
                  i (: I32) 0]

              (while (< i V)
                (let [val (: F32) (pointer-index-read logits (+ logits_bt_offset i))]
                  (if (> val maxval)
                    (set! maxval val)
                    nil))
                (set! i (+ i 1)))

              ;; Calculate exp and sum
              (let [sum (: F32) 0.0]
                (set! i 0)
                (while (< i V)
                  (let [expv (: F32) (expf (- (pointer-index-read logits (+ logits_bt_offset i)) maxval))]
                    (pointer-index-write! probs (+ probs_bt_offset i) expv)
                    (set! sum (+ sum expv)))
                  (set! i (+ i 1)))

                ;; Normalize to get probabilities
                (set! i 0)
                (while (< i V)
                  (pointer-index-write! probs (+ probs_bt_offset i)
                    (/ (pointer-index-read probs (+ probs_bt_offset i)) sum))
                  (set! i (+ i 1)))

                ;; Zero out padded dimensions
                (set! i V)
                (while (< i Vp)
                  (pointer-index-write! probs (+ probs_bt_offset i) 0.0)
                  (set! i (+ i 1)))))
            (set! t (+ t 1))))
        (set! b (+ b 1))))
    nil))

;; ============================================================================
;; gpt2_forward
;;
;; Full GPT-2 forward pass orchestration
;; This runs the model on input tokens and produces logits/probabilities
;; ============================================================================

(def gpt2_forward (: (-> [(Pointer I32) GPT2Config ParameterTensors ActivationTensors I32 I32] Nil))
  (fn [inputs config params acts B T]
    ;; Extract config values
    (let [C (: I32) (. config channels)
          L (: I32) (. config num_layers)
          NH (: I32) (. config num_heads)
          V (: I32) (. config vocab_size)
          Vp (: I32) (. config padded_vocab_size)]

      ;; Forward pass starts with encoding
      (encoder_forward (. acts encoded) inputs (. params wte) (. params wpe) B T C)

      ;; Process each transformer layer
      (let [l (: I32) 0]
        (while (< l L)
          (let [;; Calculate offsets for layer l
                BTC (: I32) (* (* B T) C)
                BT (: I32) (* B T)

                ;; Determine input residual (either encoded or previous layer's residual3)
                residual (: (Pointer F32)) (if (= l 0)
                                              (. acts encoded)
                                              (+ (. acts residual3) (* (* (- l 1) BTC) 1)))

                ;; Get pointers for this layer's weights (all layers concatenated)
                l_ln1w (: (Pointer F32)) (+ (. params ln1w) (* l C))
                l_ln1b (: (Pointer F32)) (+ (. params ln1b) (* l C))
                l_qkvw (: (Pointer F32)) (+ (. params qkvw) (* (* (* l 3) C) C))
                l_qkvb (: (Pointer F32)) (+ (. params qkvb) (* (* l 3) C))
                l_attprojw (: (Pointer F32)) (+ (. params attprojw) (* (* l C) C))
                l_attprojb (: (Pointer F32)) (+ (. params attprojb) (* l C))
                l_ln2w (: (Pointer F32)) (+ (. params ln2w) (* l C))
                l_ln2b (: (Pointer F32)) (+ (. params ln2b) (* l C))
                l_fcw (: (Pointer F32)) (+ (. params fcw) (* (* (* l 4) C) C))
                l_fcb (: (Pointer F32)) (+ (. params fcb) (* (* l 4) C))
                l_fcprojw (: (Pointer F32)) (+ (. params fcprojw) (* (* l C) (* 4 C)))
                l_fcprojb (: (Pointer F32)) (+ (. params fcprojb) (* l C))

                ;; Get pointers for this layer's activations
                l_ln1 (: (Pointer F32)) (+ (. acts ln1) (* l BTC))
                l_ln1_mean (: (Pointer F32)) (+ (. acts ln1_mean) (* l BT))
                l_ln1_rstd (: (Pointer F32)) (+ (. acts ln1_rstd) (* l BT))
                l_qkv (: (Pointer F32)) (+ (. acts qkv) (* (* (* l B) T) (* 3 C)))
                l_atty (: (Pointer F32)) (+ (. acts atty) (* l BTC))
                l_preatt (: (Pointer F32)) (+ (. acts preatt) (* (* (* (* l B) NH) T) T))
                l_att (: (Pointer F32)) (+ (. acts att) (* (* (* (* l B) NH) T) T))
                l_attproj (: (Pointer F32)) (+ (. acts attproj) (* l BTC))
                l_residual2 (: (Pointer F32)) (+ (. acts residual2) (* l BTC))
                l_ln2 (: (Pointer F32)) (+ (. acts ln2) (* l BTC))
                l_ln2_mean (: (Pointer F32)) (+ (. acts ln2_mean) (* l BT))
                l_ln2_rstd (: (Pointer F32)) (+ (. acts ln2_rstd) (* l BT))
                l_fch (: (Pointer F32)) (+ (. acts fch) (* (* (* l B) T) (* 4 C)))
                l_fch_gelu (: (Pointer F32)) (+ (. acts fch_gelu) (* (* (* l B) T) (* 4 C)))
                l_fcproj (: (Pointer F32)) (+ (. acts fcproj) (* l BTC))
                l_residual3 (: (Pointer F32)) (+ (. acts residual3) (* l BTC))]

            ;; Layer computations: attention block
            (layernorm_forward l_ln1 l_ln1_mean l_ln1_rstd residual l_ln1w l_ln1b B T C)
            (matmul_forward_naive l_qkv l_ln1 l_qkvw l_qkvb B T C (* 3 C))
            (attention_forward l_atty l_preatt l_att l_qkv B T C NH)
            (matmul_forward_naive l_attproj l_atty l_attprojw l_attprojb B T C C)
            (residual_forward l_residual2 residual l_attproj BTC)

            ;; Layer computations: MLP block
            (layernorm_forward l_ln2 l_ln2_mean l_ln2_rstd l_residual2 l_ln2w l_ln2b B T C)
            (matmul_forward_naive l_fch l_ln2 l_fcw l_fcb B T C (* 4 C))
            (gelu_forward l_fch_gelu l_fch (* (* B T) (* 4 C)))
            (matmul_forward_naive l_fcproj l_fch_gelu l_fcprojw l_fcprojb B T (* 4 C) C)
            (residual_forward l_residual3 l_residual2 l_fcproj BTC))

          (set! l (+ l 1))))

      ;; Final layer norm and projection to vocabulary
      (let [residual (: (Pointer F32)) (+ (. acts residual3) (* (* (* (- L 1) B) T) C))]
        (layernorm_forward (. acts lnf) (. acts lnf_mean) (. acts lnf_rstd)
                          residual (. params lnfw) (. params lnfb) B T C)
        (matmul_forward_naive (. acts logits) (. acts lnf) (. params wte) pointer-null B T C Vp)
        (softmax_forward (. acts probs) (. acts logits) B T V Vp)))
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
;; Test matmul_forward_naive
;; ============================================================================

(def test_matmul_forward (: (-> [] I32))
  (fn []
    (printf (c-str "Testing matmul_forward_naive...\n"))

    ;; Test dimensions: B=2, T=2, C=3, OC=4
    (let [B (: I32) 2
          T (: I32) 2
          C (: I32) 3
          OC (: I32) 4
          ;; Allocate arrays
          out (: (Pointer F32)) (allocate-array F32 (* (* B T) OC))
          inp (: (Pointer F32)) (allocate-array F32 (* (* B T) C))
          weight (: (Pointer F32)) (allocate-array F32 (* OC C))
          bias (: (Pointer F32)) (allocate-array F32 OC)]

      ;; Initialize inp with sequential values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      (let [i (: I32) 0]
        (while (< i (* (* B T) C))
          (pointer-index-write! inp i (+ (+ i 0.0) 1.0))
          (set! i (+ i 1))))

      ;; Initialize weight with sequential values
      (let [i (: I32) 0]
        (while (< i (* OC C))
          (pointer-index-write! weight i (+ (+ i 0.0) 1.0))
          (set! i (+ i 1))))

      ;; Initialize bias with sequential values
      (let [i (: I32) 0]
        (while (< i OC)
          (pointer-index-write! bias i (+ (+ i 0.0) 0.1))
          (set! i (+ i 1))))

      ;; Run matmul_forward_naive
      (matmul_forward_naive out inp weight bias B T C OC)

      ;; Print first few outputs for verification
      (printf (c-str "First output values:\n"))
      (printf (c-str "  out[0] = %f\n") (pointer-index-read out 0))
      (printf (c-str "  out[1] = %f\n") (pointer-index-read out 1))
      (printf (c-str "  out[2] = %f\n") (pointer-index-read out 2))
      (printf (c-str "  out[3] = %f\n") (pointer-index-read out 3))

      ;; Clean up
      (deallocate-array out)
      (deallocate-array inp)
      (deallocate-array weight)
      (deallocate-array bias)

      (printf (c-str "matmul_forward_naive test completed!\n"))
      0)))

;; ============================================================================
;; Test residual_forward
;; ============================================================================

(def test_residual_forward (: (-> [] I32))
  (fn []
    (printf (c-str "Testing residual_forward...\n"))

    ;; Test with N=6
    (let [N (: I32) 6
          out (: (Pointer F32)) (allocate-array F32 N)
          inp1 (: (Pointer F32)) (allocate-array F32 N)
          inp2 (: (Pointer F32)) (allocate-array F32 N)]

      ;; Initialize inp1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      (let [i (: I32) 0]
        (while (< i N)
          (pointer-index-write! inp1 i (+ (+ i 0.0) 1.0))
          (set! i (+ i 1))))

      ;; Initialize inp2: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
      (let [i (: I32) 0]
        (while (< i N)
          (pointer-index-write! inp2 i (* (+ (+ i 0.0) 1.0) 0.1))
          (set! i (+ i 1))))

      ;; Run residual_forward
      (residual_forward out inp1 inp2 N)

      ;; Print outputs
      (printf (c-str "First output values:\n"))
      (printf (c-str "  out[0] = %f (expected 1.1)\n") (pointer-index-read out 0))
      (printf (c-str "  out[1] = %f (expected 2.2)\n") (pointer-index-read out 1))
      (printf (c-str "  out[2] = %f (expected 3.3)\n") (pointer-index-read out 2))

      ;; Clean up
      (deallocate-array out)
      (deallocate-array inp1)
      (deallocate-array inp2)

      (printf (c-str "residual_forward test completed!\n"))
      0)))

;; ============================================================================
;; Test gelu_forward
;; ============================================================================

(def test_gelu_forward (: (-> [] I32))
  (fn []
    (printf (c-str "Testing gelu_forward...\n"))

    ;; Test with N=4
    (let [N (: I32) 4
          out (: (Pointer F32)) (allocate-array F32 N)
          inp (: (Pointer F32)) (allocate-array F32 N)]

      ;; Initialize inp with test values: [-1.0, 0.0, 1.0, 2.0]
      (pointer-index-write! inp 0 (- 0.0 1.0))
      (pointer-index-write! inp 1 0.0)
      (pointer-index-write! inp 2 1.0)
      (pointer-index-write! inp 3 2.0)

      ;; Run gelu_forward
      (gelu_forward out inp N)

      ;; Print outputs
      (printf (c-str "First output values:\n"))
      (printf (c-str "  out[0] = %f\n") (pointer-index-read out 0))
      (printf (c-str "  out[1] = %f\n") (pointer-index-read out 1))
      (printf (c-str "  out[2] = %f\n") (pointer-index-read out 2))
      (printf (c-str "  out[3] = %f\n") (pointer-index-read out 3))

      ;; Clean up
      (deallocate-array out)
      (deallocate-array inp)

      (printf (c-str "gelu_forward test completed!\n"))
      0)))

;; ============================================================================
;; Test attention_forward
;; ============================================================================

(def test_attention_forward (: (-> [] I32))
  (fn []
    (printf (c-str "Testing attention_forward...\n"))

    ;; Test dimensions: B=1, T=2, C=4, NH=2
    (let [B (: I32) 1
          T (: I32) 2
          C (: I32) 4
          NH (: I32) 2
          ;; Allocate arrays
          out (: (Pointer F32)) (allocate-array F32 (* (* B T) C))
          preatt (: (Pointer F32)) (allocate-array F32 (* (* (* (* B NH) T) T) 1))
          att (: (Pointer F32)) (allocate-array F32 (* (* (* (* B NH) T) T) 1))
          inp (: (Pointer F32)) (allocate-array F32 (* (* B T) (* C 3)))]

      ;; Initialize inp with sequential values (Q, K, V concatenated)
      (let [i (: I32) 0]
        (while (< i (* (* B T) (* C 3)))
          (pointer-index-write! inp i (* (+ (+ i 0.0) 1.0) 0.1))
          (set! i (+ i 1))))

      ;; Run attention_forward
      (attention_forward out preatt att inp B T C NH)

      ;; Print outputs
      (printf (c-str "First output values:\n"))
      (printf (c-str "  out[0] = %f\n") (pointer-index-read out 0))
      (printf (c-str "  out[1] = %f\n") (pointer-index-read out 1))
      (printf (c-str "  out[2] = %f\n") (pointer-index-read out 2))
      (printf (c-str "  out[3] = %f\n") (pointer-index-read out 3))

      ;; Clean up
      (deallocate-array out)
      (deallocate-array preatt)
      (deallocate-array att)
      (deallocate-array inp)

      (printf (c-str "attention_forward test completed!\n"))
      0)))

;; ============================================================================
;; Test softmax_forward
;; ============================================================================

(def test_softmax_forward (: (-> [] I32))
  (fn []
    (printf (c-str "Testing softmax_forward...\n"))

    ;; Test dimensions: B=1, T=2, V=4, Vp=8 (padded)
    (let [B (: I32) 1
          T (: I32) 2
          V (: I32) 4
          Vp (: I32) 8
          ;; Allocate arrays
          probs (: (Pointer F32)) (allocate-array F32 (* (* B T) Vp))
          logits (: (Pointer F32)) (allocate-array F32 (* (* B T) Vp))]

      ;; Initialize logits with test values
      ;; First position: [1.0, 2.0, 3.0, 4.0, 0, 0, 0, 0]
      (pointer-index-write! logits 0 1.0)
      (pointer-index-write! logits 1 2.0)
      (pointer-index-write! logits 2 3.0)
      (pointer-index-write! logits 3 4.0)
      (pointer-index-write! logits 4 0.0)
      (pointer-index-write! logits 5 0.0)
      (pointer-index-write! logits 6 0.0)
      (pointer-index-write! logits 7 0.0)

      ;; Second position: [2.0, 2.0, 2.0, 2.0, 0, 0, 0, 0]
      (pointer-index-write! logits 8 2.0)
      (pointer-index-write! logits 9 2.0)
      (pointer-index-write! logits 10 2.0)
      (pointer-index-write! logits 11 2.0)
      (pointer-index-write! logits 12 0.0)
      (pointer-index-write! logits 13 0.0)
      (pointer-index-write! logits 14 0.0)
      (pointer-index-write! logits 15 0.0)

      ;; Run softmax_forward
      (softmax_forward probs logits B T V Vp)

      ;; Print outputs for first position (should sum to 1.0)
      (printf (c-str "First position probabilities:\n"))
      (printf (c-str "  probs[0] = %f\n") (pointer-index-read probs 0))
      (printf (c-str "  probs[1] = %f\n") (pointer-index-read probs 1))
      (printf (c-str "  probs[2] = %f\n") (pointer-index-read probs 2))
      (printf (c-str "  probs[3] = %f\n") (pointer-index-read probs 3))
      (printf (c-str "  Sum = %f (should be 1.0)\n")
        (+ (+ (+ (pointer-index-read probs 0) (pointer-index-read probs 1))
                 (pointer-index-read probs 2))
              (pointer-index-read probs 3)))

      ;; Print outputs for second position (should all be 0.25)
      (printf (c-str "Second position probabilities (all equal logits):\n"))
      (printf (c-str "  probs[8] = %f\n") (pointer-index-read probs 8))
      (printf (c-str "  probs[9] = %f\n") (pointer-index-read probs 9))

      ;; Clean up
      (deallocate-array probs)
      (deallocate-array logits)

      (printf (c-str "softmax_forward test completed!\n"))
      0)))

;; ============================================================================
;; Main
;; ============================================================================

(def main-fn (: (-> [] I32))
  (fn []
    (test_encoder_forward)
    (printf (c-str "\n"))
    (test_layernorm_forward)
    (printf (c-str "\n"))
    (test_matmul_forward)
    (printf (c-str "\n"))
    (test_residual_forward)
    (printf (c-str "\n"))
    (test_gelu_forward)
    (printf (c-str "\n"))
    (test_attention_forward)
    (printf (c-str "\n"))
    (test_softmax_forward)))

(main-fn)
