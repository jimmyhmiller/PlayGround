;; ============================================================================
;; GPT-2 Implementation - Direct Port from llm.c
;; Reference: https://github.com/karpathy/llm.c
;; ============================================================================

;; Use declare-fn instead of extern-fn to avoid duplicate declarations with headers
(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "math.h")
(link-library "m")

;; Enable aggressive optimization for performance
(compiler-flag "-O3")

;; ============================================================================
;; External Functions (declared for type checker, not emitted)
;; ============================================================================

(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn free [ptr (Pointer U8)] -> Nil)
(declare-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer U8))
(declare-fn fclose [file (Pointer U8)] -> I32)
(declare-fn fread [ptr (Pointer U8) size I32 count I32 file (Pointer U8)] -> I32)
(declare-fn exit [status I32] -> Nil)
(declare-fn sqrtf [x F32] -> F32)
(declare-fn tanhf [x F32] -> F32)
(declare-fn expf [x F32] -> F32)

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
;; Checkpoint Data - struct to hold both config and params from a checkpoint
;; ============================================================================

(def CheckpointData (: Type)
  (Struct
    [config GPT2Config]
    [params ParameterTensors]))

;; ============================================================================
;; load_gpt2_checkpoint
;;
;; Loads GPT-2 model weights from a binary checkpoint file
;; File format:
;;   - Header: 256 int32 values
;;   - Parameters: contiguous float32 array (all 16 parameter tensors)
;; ============================================================================

(def load_gpt2_checkpoint (: (-> [(Pointer U8)] CheckpointData))
  (fn [checkpoint_path]
    (printf (c-str "Loading checkpoint from file...\n"))

    ;; Open file for reading
    (let [model_file (: (Pointer U8)) (fopen checkpoint_path (c-str "rb"))]

      ;; Check if file opened successfully
      (if (pointer-equal? model_file pointer-null)
        (let [dummy (: I32) 0]
          (printf (c-str "ERROR: Could not open checkpoint file\n"))
          (exit 1)
          ;; Never reached, but needed for type checking
          (CheckpointData (GPT2Config 0 0 0 0 0 0)
                         (ParameterTensors pointer-null pointer-null pointer-null pointer-null
                                          pointer-null pointer-null pointer-null pointer-null
                                          pointer-null pointer-null pointer-null pointer-null
                                          pointer-null pointer-null pointer-null pointer-null)))

        ;; File opened successfully - read header
        (let [header (: (Pointer I32)) (allocate-array I32 256)
              bytes_read (: I32) (fread (cast (Pointer U8) header) 4 256 model_file)]

          ;; Check magic number and version
          (let [magic (: I32) (pointer-index-read header 0)
                version (: I32) (pointer-index-read header 1)]

            (if (!= magic 20240326)
              (let [dummy (: I32) 0]
                (printf (c-str "ERROR: Bad magic number in model file: %d\n") magic)
                (exit 1))
              nil)

            (if (!= version 3)
              (let [dummy (: I32) 0]
                (printf (c-str "ERROR: Bad version in model file: %d (expected 3)\n") version)
                (printf (c-str "HINT: Re-run 'python train_gpt2.py' to generate correct format\n"))
                (exit 1))
              nil)

            ;; Extract config from header
            (let [maxT (: I32) (pointer-index-read header 2)
                  V (: I32) (pointer-index-read header 3)
                  L (: I32) (pointer-index-read header 4)
                  NH (: I32) (pointer-index-read header 5)
                  C (: I32) (pointer-index-read header 6)
                  Vp (: I32) (pointer-index-read header 7)]

              (printf (c-str "[GPT-2]\n"))
              (printf (c-str "max_seq_len: %d\n") maxT)
              (printf (c-str "vocab_size: %d\n") V)
              (printf (c-str "padded_vocab_size: %d\n") Vp)
              (printf (c-str "num_layers: %d\n") L)
              (printf (c-str "num_heads: %d\n") NH)
              (printf (c-str "channels: %d\n") C)

              ;; Calculate parameter sizes (16 tensors)
              (let [wte_size (: I32) (* Vp C)
                    wpe_size (: I32) (* maxT C)
                    ln1w_size (: I32) (* L C)
                    ln1b_size (: I32) (* L C)
                    qkvw_size (: I32) (* (* (* L 3) C) C)
                    qkvb_size (: I32) (* (* L 3) C)
                    attprojw_size (: I32) (* (* L C) C)
                    attprojb_size (: I32) (* L C)
                    ln2w_size (: I32) (* L C)
                    ln2b_size (: I32) (* L C)
                    fcw_size (: I32) (* (* (* L 4) C) C)
                    fcb_size (: I32) (* (* L 4) C)
                    fcprojw_size (: I32) (* (* L C) (* 4 C))
                    fcprojb_size (: I32) (* L C)
                    lnfw_size (: I32) C
                    lnfb_size (: I32) C

                    ;; Calculate total parameters
                    num_parameters (: I32) (+ (+ (+ wte_size wpe_size) (+ (+ ln1w_size ln1b_size) (+ qkvw_size qkvb_size)))
                                             (+ (+ (+ attprojw_size attprojb_size) (+ ln2w_size ln2b_size))
                                                (+ (+ (+ fcw_size fcb_size) (+ fcprojw_size fcprojb_size)) (+ lnfw_size lnfb_size))))]

                (printf (c-str "num_parameters: %d\n") num_parameters)

                ;; Allocate memory for all parameters
                (let [params_memory (: (Pointer F32)) (allocate-array F32 num_parameters)
                      params_bytes_read (: I32) (fread (cast (Pointer U8) params_memory) 4 num_parameters model_file)]

                  ;; Close file
                  (fclose model_file)

                  ;; Clean up header
                  (deallocate-array header)

                  ;; Set up pointers to individual tensors
                  (let [wte (: (Pointer F32)) params_memory
                        wpe (: (Pointer F32)) (+ wte wte_size)
                        ln1w (: (Pointer F32)) (+ wpe wpe_size)
                        ln1b (: (Pointer F32)) (+ ln1w ln1w_size)
                        qkvw (: (Pointer F32)) (+ ln1b ln1b_size)
                        qkvb (: (Pointer F32)) (+ qkvw qkvw_size)
                        attprojw (: (Pointer F32)) (+ qkvb qkvb_size)
                        attprojb (: (Pointer F32)) (+ attprojw attprojw_size)
                        ln2w (: (Pointer F32)) (+ attprojb attprojb_size)
                        ln2b (: (Pointer F32)) (+ ln2w ln2w_size)
                        fcw (: (Pointer F32)) (+ ln2b ln2b_size)
                        fcb (: (Pointer F32)) (+ fcw fcw_size)
                        fcprojw (: (Pointer F32)) (+ fcb fcb_size)
                        fcprojb (: (Pointer F32)) (+ fcprojw fcprojw_size)
                        lnfw (: (Pointer F32)) (+ fcprojb fcprojb_size)
                        lnfb (: (Pointer F32)) (+ lnfw lnfw_size)

                        config (: GPT2Config) (GPT2Config maxT V Vp L NH C)
                        params (: ParameterTensors) (ParameterTensors
                                                      wte wpe ln1w ln1b qkvw qkvb
                                                      attprojw attprojb ln2w ln2b
                                                      fcw fcb fcprojw fcprojb lnfw lnfb)]

                    (printf (c-str "Successfully loaded checkpoint!\n"))
                    (CheckpointData config params)))))))))))

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
;;
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
;; matmul_forward (OPTIMIZED)
;;
;; Optimized matrix multiplication with loop unrolling and register tiling
;; Key optimizations:
;; 1. Loop unrolling by 8 - process 8 positions at once
;; 2. Register tiling - keep 8 results in local variables
;; 3. Weight reuse - load each weight once, use 8 times (8x less memory bandwidth!)
;; 4. Falls back to naive version if B*T not divisible by 8
;; ============================================================================

(def matmul_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32 I32] Nil))
  (fn [out inp weight bias B T C OC]
    (let [LOOP_UNROLL (: I32) 8
          BT (: I32) (* B T)]

      ;; Fallback to naive if B*T not divisible by LOOP_UNROLL
      (if (!= (% BT LOOP_UNROLL) 0)
        (matmul_forward_naive out inp weight bias B T C OC)

        ;; Optimized version with loop unrolling
        (let [obt (: I32) 0]
          (while (< obt BT)
            (let [o (: I32) 0]
              (while (< o OC)
                ;; Register tiling: keep 8 results in local variables
                (let [result0 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))
                      result1 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))
                      result2 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))
                      result3 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))
                      result4 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))
                      result5 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))
                      result6 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))
                      result7 (: F32) (if (pointer-equal? bias pointer-null) 0.0 (pointer-index-read bias o))]

                  ;; Inner loops: load weight once, use 8 times
                  (let [i (: I32) 0]
                    (while (< i C)
                      ;; Load weight ONCE
                      (let [w (: F32) (pointer-index-read weight (+ i (* o C)))
                            ;; Compute 8 positions with same weight (weight reuse!)
                            bt0 (: I32) (+ obt 0)
                            bt1 (: I32) (+ obt 1)
                            bt2 (: I32) (+ obt 2)
                            bt3 (: I32) (+ obt 3)
                            bt4 (: I32) (+ obt 4)
                            bt5 (: I32) (+ obt 5)
                            bt6 (: I32) (+ obt 6)
                            bt7 (: I32) (+ obt 7)]

                        ;; Accumulate: result += inp * w (8 times, weight reused!)
                        (set! result0 (+ result0 (* (pointer-index-read inp (+ (* bt0 C) i)) w)))
                        (set! result1 (+ result1 (* (pointer-index-read inp (+ (* bt1 C) i)) w)))
                        (set! result2 (+ result2 (* (pointer-index-read inp (+ (* bt2 C) i)) w)))
                        (set! result3 (+ result3 (* (pointer-index-read inp (+ (* bt3 C) i)) w)))
                        (set! result4 (+ result4 (* (pointer-index-read inp (+ (* bt4 C) i)) w)))
                        (set! result5 (+ result5 (* (pointer-index-read inp (+ (* bt5 C) i)) w)))
                        (set! result6 (+ result6 (* (pointer-index-read inp (+ (* bt6 C) i)) w)))
                        (set! result7 (+ result7 (* (pointer-index-read inp (+ (* bt7 C) i)) w))))
                      (set! i (+ i 1))))

                  ;; Write back all 8 results to memory
                  (pointer-index-write! out (+ (* (+ obt 0) OC) o) result0)
                  (pointer-index-write! out (+ (* (+ obt 1) OC) o) result1)
                  (pointer-index-write! out (+ (* (+ obt 2) OC) o) result2)
                  (pointer-index-write! out (+ (* (+ obt 3) OC) o) result3)
                  (pointer-index-write! out (+ (* (+ obt 4) OC) o) result4)
                  (pointer-index-write! out (+ (* (+ obt 5) OC) o) result5)
                  (pointer-index-write! out (+ (* (+ obt 6) OC) o) result6)
                  (pointer-index-write! out (+ (* (+ obt 7) OC) o) result7))

                (set! o (+ o 1))))
            (set! obt (+ obt LOOP_UNROLL))))))
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
;; Sampling helper - find argmax (greedy decoding)
;; ============================================================================

;; argmax - find index of maximum value in array
;; Simple greedy sampling - always pick the most likely token
(def argmax (: (-> [(Pointer F32) I32] I32))
  (fn [probs n]
    (let [max_val (: F32) (- 0.0 999999.0)
          max_idx (: I32) 0
          i (: I32) 0]
      (while (< i n)
        (let [val (: F32) (pointer-index-read probs i)]
          (if (> val max_val)
            (let [dummy (: I32) 0]
              (set! max_val val)
              (set! max_idx i))
            nil))
        (set! i (+ i 1)))
      max_idx)))

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
            (matmul_forward l_qkv l_ln1 l_qkvw l_qkvb B T C (* 3 C))
            (attention_forward l_atty l_preatt l_att l_qkv B T C NH)
            (matmul_forward l_attproj l_atty l_attprojw l_attprojb B T C C)
            (residual_forward l_residual2 residual l_attproj BTC)

            ;; Layer computations: MLP block
            (layernorm_forward l_ln2 l_ln2_mean l_ln2_rstd l_residual2 l_ln2w l_ln2b B T C)
            (matmul_forward l_fch l_ln2 l_fcw l_fcb B T C (* 4 C))
            (gelu_forward l_fch_gelu l_fch (* (* B T) (* 4 C)))
            (matmul_forward l_fcproj l_fch_gelu l_fcprojw l_fcprojb B T (* 4 C) C)
            (residual_forward l_residual3 l_residual2 l_fcproj BTC))

          (set! l (+ l 1))))

      ;; Final layer norm and projection to vocabulary
      (let [residual (: (Pointer F32)) (+ (. acts residual3) (* (* (* (- L 1) B) T) C))]
        (layernorm_forward (. acts lnf) (. acts lnf_mean) (. acts lnf_rstd)
                          residual (. params lnfw) (. params lnfb) B T C)
        (matmul_forward (. acts logits) (. acts lnf) (. params wte) pointer-null B T C Vp)
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
;; Helper: allocate_activation_tensors
;; Allocates all activation buffers for the given configuration
;; ============================================================================

(def allocate_activation_tensors (: (-> [I32 I32 I32 I32 I32 I32] ActivationTensors))
  (fn [B T C L NH Vp]
    (let [encoded_size (: I32) (* (* B T) C)
          ln1_size (: I32) (* (* (* L B) T) C)
          ln1_mean_size (: I32) (* (* L B) T)
          ln1_rstd_size (: I32) (* (* L B) T)
          qkv_size (: I32) (* (* (* L B) T) (* 3 C))
          atty_size (: I32) (* (* (* L B) T) C)
          preatt_size (: I32) (* (* (* (* L B) NH) T) T)
          att_size (: I32) (* (* (* (* L B) NH) T) T)
          attproj_size (: I32) (* (* (* L B) T) C)
          residual2_size (: I32) (* (* (* L B) T) C)
          ln2_size (: I32) (* (* (* L B) T) C)
          ln2_mean_size (: I32) (* (* L B) T)
          ln2_rstd_size (: I32) (* (* L B) T)
          fch_size (: I32) (* (* (* L B) T) (* 4 C))
          fch_gelu_size (: I32) (* (* (* L B) T) (* 4 C))
          fcproj_size (: I32) (* (* (* L B) T) C)
          residual3_size (: I32) (* (* (* L B) T) C)
          lnf_size (: I32) (* (* B T) C)
          lnf_mean_size (: I32) (* B T)
          lnf_rstd_size (: I32) (* B T)
          logits_size (: I32) (* (* B T) Vp)
          probs_size (: I32) (* (* B T) Vp)
          losses_size (: I32) (* B T)]

      (ActivationTensors
        (allocate-array F32 encoded_size)
        (allocate-array F32 ln1_size)
        (allocate-array F32 ln1_mean_size)
        (allocate-array F32 ln1_rstd_size)
        (allocate-array F32 qkv_size)
        (allocate-array F32 atty_size)
        (allocate-array F32 preatt_size)
        (allocate-array F32 att_size)
        (allocate-array F32 attproj_size)
        (allocate-array F32 residual2_size)
        (allocate-array F32 ln2_size)
        (allocate-array F32 ln2_mean_size)
        (allocate-array F32 ln2_rstd_size)
        (allocate-array F32 fch_size)
        (allocate-array F32 fch_gelu_size)
        (allocate-array F32 fcproj_size)
        (allocate-array F32 residual3_size)
        (allocate-array F32 lnf_size)
        (allocate-array F32 lnf_mean_size)
        (allocate-array F32 lnf_rstd_size)
        (allocate-array F32 logits_size)
        (allocate-array F32 probs_size)
        (allocate-array F32 losses_size)))))

;; ============================================================================
;; Helper: allocate_parameter_tensors
;; Allocates all parameter buffers and initializes with small random values
;; ============================================================================

(def allocate_parameter_tensors (: (-> [I32 I32 I32 I32 I32] ParameterTensors))
  (fn [V maxT C L Vp]
    (let [wte_size (: I32) (* V C)
          wpe_size (: I32) (* maxT C)
          ln1w_size (: I32) (* L C)
          ln1b_size (: I32) (* L C)
          qkvw_size (: I32) (* (* (* L 3) C) C)
          qkvb_size (: I32) (* (* L 3) C)
          attprojw_size (: I32) (* (* L C) C)
          attprojb_size (: I32) (* L C)
          ln2w_size (: I32) (* L C)
          ln2b_size (: I32) (* L C)
          fcw_size (: I32) (* (* (* L 4) C) C)
          fcb_size (: I32) (* (* L 4) C)
          fcprojw_size (: I32) (* (* L C) (* 4 C))
          fcprojb_size (: I32) (* L C)
          lnfw_size (: I32) C
          lnfb_size (: I32) C

          ;; Allocate and initialize with simple values (0.01 for weights, 0 for biases)
          wte (: (Pointer F32)) (allocate-array F32 wte_size 0.01)
          wpe (: (Pointer F32)) (allocate-array F32 wpe_size 0.01)
          ln1w (: (Pointer F32)) (allocate-array F32 ln1w_size 1.0)
          ln1b (: (Pointer F32)) (allocate-array F32 ln1b_size 0.0)
          qkvw (: (Pointer F32)) (allocate-array F32 qkvw_size 0.01)
          qkvb (: (Pointer F32)) (allocate-array F32 qkvb_size 0.0)
          attprojw (: (Pointer F32)) (allocate-array F32 attprojw_size 0.01)
          attprojb (: (Pointer F32)) (allocate-array F32 attprojb_size 0.0)
          ln2w (: (Pointer F32)) (allocate-array F32 ln2w_size 1.0)
          ln2b (: (Pointer F32)) (allocate-array F32 ln2b_size 0.0)
          fcw (: (Pointer F32)) (allocate-array F32 fcw_size 0.01)
          fcb (: (Pointer F32)) (allocate-array F32 fcb_size 0.0)
          fcprojw (: (Pointer F32)) (allocate-array F32 fcprojw_size 0.01)
          fcprojb (: (Pointer F32)) (allocate-array F32 fcprojb_size 0.0)
          lnfw (: (Pointer F32)) (allocate-array F32 lnfw_size 1.0)
          lnfb (: (Pointer F32)) (allocate-array F32 lnfb_size 0.0)]

      (ParameterTensors
        wte wpe ln1w ln1b qkvw qkvb attprojw attprojb
        ln2w ln2b fcw fcb fcprojw fcprojb lnfw lnfb))))

;; ============================================================================
;; Test: integration test for complete GPT-2 forward pass
;; ============================================================================

(def test_gpt2_inference (: (-> [] I32))
  (fn []
    (printf (c-str "Testing complete GPT-2 inference pipeline...\n"))

    ;; Tiny model configuration: 1 layer, 2 heads, 64 channels, vocab=16
    (let [config (: GPT2Config) (GPT2Config 8 16 16 1 2 64)
          B (: I32) 1
          T (: I32) 4

          ;; Extract config values
          C (: I32) (. config channels)
          L (: I32) (. config num_layers)
          NH (: I32) (. config num_heads)
          V (: I32) (. config vocab_size)
          Vp (: I32) (. config padded_vocab_size)
          maxT (: I32) (. config max_seq_len)]

      (printf (c-str "Config: L=%d, NH=%d, C=%d, V=%d, Vp=%d\n") L NH C V Vp)

      ;; Allocate parameter and activation tensors
      (let [params (: ParameterTensors) (allocate_parameter_tensors V maxT C L Vp)
            acts (: ActivationTensors) (allocate_activation_tensors B T C L NH Vp)

            ;; Create input token sequence: [1, 2, 3, 4]
            inputs (: (Pointer I32)) (allocate-array I32 (* B T))]

        (pointer-index-write! inputs 0 1)
        (pointer-index-write! inputs 1 2)
        (pointer-index-write! inputs 2 3)
        (pointer-index-write! inputs 3 4)

        (printf (c-str "Input tokens: [1, 2, 3, 4]\n"))

        ;; Run the forward pass!
        (printf (c-str "Running gpt2_forward...\n"))
        (gpt2_forward inputs config params acts B T)

        ;; Get probabilities for the last position (t=T-1)
        (let [last_t (: I32) (- T 1)
              last_probs_offset (: I32) (* last_t Vp)
              probs_ptr (: (Pointer F32)) (+ (. acts probs) last_probs_offset)]

          (printf (c-str "Output probabilities for last position (first 8 values):\n"))
          (let [i (: I32) 0]
            (while (< i 8)
              (printf (c-str "  probs[%d] = %f\n") i (pointer-index-read probs_ptr i))
              (set! i (+ i 1))))

          ;; Use argmax to get predicted token
          (let [next_token (: I32) (argmax probs_ptr V)]
            (printf (c-str "Predicted next token (greedy): %d\n") next_token)))

        ;; Clean up
        (deallocate-array inputs)

        (printf (c-str "GPT-2 inference test completed!\n"))
        0))))

;; ============================================================================
;; Test: Real GPT-2 inference with loaded checkpoint
;; ============================================================================

(def test_real_gpt2_inference (: (-> [] I32))
  (fn []
    (printf (c-str "\n=== Testing GPT-2 with REAL pretrained weights ===\n\n"))

    ;; Load the checkpoint
    (let [checkpoint (: CheckpointData) (load_gpt2_checkpoint (c-str "gpt2_124M.bin"))
          config (: GPT2Config) (. checkpoint config)
          params (: ParameterTensors) (. checkpoint params)]

      (printf (c-str "\n=== Running inference ===\n\n"))

      ;; Set up inference parameters
      (let [B (: I32) 1
            T (: I32) 4

            ;; Extract config values
            C (: I32) (. config channels)
            L (: I32) (. config num_layers)
            NH (: I32) (. config num_heads)
            V (: I32) (. config vocab_size)
            Vp (: I32) (. config padded_vocab_size)]

        ;; Allocate activation tensors
        (let [acts (: ActivationTensors) (allocate_activation_tensors B T C L NH Vp)

              ;; Create input tokens (some example tokens)
              ;; GPT-2 tokenizer: 1 = "!", 2 = "\"", 3 = "#", 4 = "$"
              ;; Let's use [15496, 995, 318] which is roughly "Hello world is"
              inputs (: (Pointer I32)) (allocate-array I32 (* B T))]

          (pointer-index-write! inputs 0 15496)  ; "Hello"
          (pointer-index-write! inputs 1 995)    ; " world"
          (pointer-index-write! inputs 2 318)    ; " is"
          (pointer-index-write! inputs 3 1)      ; test token

          (printf (c-str "Input tokens: [15496, 995, 318, 1]\n"))
          (printf (c-str "Running full GPT-2 forward pass with 12 layers...\n"))

          ;; Run the forward pass!
          (gpt2_forward inputs config params acts B T)

          ;; Get probabilities for the last position
          (let [last_t (: I32) (- T 1)
                last_probs_offset (: I32) (* last_t Vp)
                probs_ptr (: (Pointer F32)) (+ (. acts probs) last_probs_offset)]

            (printf (c-str "\nOutput probabilities for last position (first 10 values):\n"))
            (let [i (: I32) 0]
              (while (< i 10)
                (printf (c-str "  token[%d] prob = %.6f\n") i (pointer-index-read probs_ptr i))
                (set! i (+ i 1))))

            ;; Get top predicted token
            (let [next_token (: I32) (argmax probs_ptr V)]
              (printf (c-str "\nPredicted next token (greedy): %d\n") next_token)
              (printf (c-str "Probability of predicted token: %.6f\n")
                (pointer-index-read probs_ptr next_token))))

          ;; Clean up
          (deallocate-array inputs)

          (printf (c-str "\nReal GPT-2 inference test completed successfully!\n"))
          0)))))

;; ============================================================================
;; Test: Autoregressive generation - generate multiple tokens
;; ============================================================================

(def test_autoregressive_generation (: (-> [] I32))
  (fn []
    (printf (c-str "\n=== Testing Autoregressive Generation ===\n\n"))

    ;; Load the checkpoint
    (let [checkpoint (: CheckpointData) (load_gpt2_checkpoint (c-str "gpt2_124M.bin"))
          config (: GPT2Config) (. checkpoint config)
          params (: ParameterTensors) (. checkpoint params)]

      (printf (c-str "\n=== Generating tokens ===\n\n"))

      (let [C (: I32) (. config channels)
            L (: I32) (. config num_layers)
            NH (: I32) (. config num_heads)
            V (: I32) (. config vocab_size)
            Vp (: I32) (. config padded_vocab_size)

            ;; Start with a prompt: "Hello world is"
            ;; Use FIXED context window for O(N) generation instead of O(N³)
            context_window (: I32) 8
            num_tokens_to_generate (: I32) 100
            sequence (: (Pointer I32)) (allocate-array I32 (+ context_window num_tokens_to_generate))]

        ;; Initialize prompt tokens
        (pointer-index-write! sequence 0 15496)  ; "Hello"
        (pointer-index-write! sequence 1 995)    ; " world"
        (pointer-index-write! sequence 2 318)    ; " is"

        (printf (c-str "Initial prompt tokens: [15496, 995, 318]\n"))
        (printf (c-str "Generating 100 new tokens with fixed context window...\n\n"))

        ;; Allocate activation tensors ONCE for context_window (outside loop!)
        (let [B (: I32) 1
              acts (: ActivationTensors) (allocate_activation_tensors B context_window C L NH Vp)]

          ;; Generate 100 new tokens using sliding window
          (let [gen_count (: I32) 0
                total_generated (: I32) 3]  ; Start with 3 prompt tokens
            (while (< gen_count num_tokens_to_generate)
              (let [;; Use only last context_window tokens for forward pass
                    window_start (: I32) (if (< total_generated context_window)
                                           0
                                           (- total_generated context_window))
                    window_len (: I32) (if (< total_generated context_window)
                                         total_generated
                                         context_window)
                    window_input (: (Pointer I32)) (+ sequence window_start)]

                ;; Run forward pass ONLY on context window (fixed size!)
                (gpt2_forward window_input config params acts B window_len)

                ;; Get probabilities for last position in window
                (let [last_t (: I32) (- window_len 1)
                      last_probs_offset (: I32) (* last_t Vp)
                      probs_ptr (: (Pointer F32)) (+ (. acts probs) last_probs_offset)

                      ;; Sample next token (greedy - pick argmax)
                      next_token (: I32) (argmax probs_ptr V)]

                  ;; Progress milestones
                  (if (= gen_count 0)
                    (let [dummy (: I32) (printf (c-str "Starting generation...\n"))] nil)
                    nil)
                  (if (= gen_count 24)
                    (let [dummy (: I32) (printf (c-str "25 tokens generated...\n"))] nil)
                    nil)
                  (if (= gen_count 49)
                    (let [dummy (: I32) (printf (c-str "50 tokens generated...\n"))] nil)
                    nil)
                  (if (= gen_count 74)
                    (let [dummy (: I32) (printf (c-str "75 tokens generated...\n"))] nil)
                    nil)

                  ;; Append to full sequence
                  (pointer-index-write! sequence total_generated next_token)
                  (set! total_generated (+ total_generated 1))))

              (set! gen_count (+ gen_count 1)))))

        ;; Print final sequence (first 20 tokens for readability)
        (printf (c-str "\nFirst 20 generated tokens:\n"))
        (printf (c-str "["))
        (let [i (: I32) 0
              print_limit (: I32) (if (< (+ 3 num_tokens_to_generate) 20)
                                    (+ 3 num_tokens_to_generate)
                                    20)]
          (while (< i print_limit)
            (if (< i (- print_limit 1))
              (printf (c-str "%d, ") (pointer-index-read sequence i))
              (printf (c-str "%d") (pointer-index-read sequence i)))
            (set! i (+ i 1))))
        (printf (c-str " ...]\n"))

        ;; Clean up
        (deallocate-array sequence)

        (printf (c-str "\nAutoregressive generation test completed!\n"))
        0))))

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
    (test_softmax_forward)
    (printf (c-str "\n"))
    (test_gpt2_inference)
    (printf (c-str "\n"))
    (test_real_gpt2_inference)
    (printf (c-str "\n"))
    (test_autoregressive_generation)))

(main-fn)
