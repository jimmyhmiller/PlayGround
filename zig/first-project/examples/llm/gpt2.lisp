;; GPT-2 Transformer Model Implementation
;; Main structures and forward pass

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "string.h")

;; File I/O (FILE* is an opaque pointer, use Pointer Void)
(extern-fn fopen [filename String mode String] -> (Pointer Void))
(extern-fn fclose [stream (Pointer Void)] -> I32)
(extern-fn fread [ptr (Pointer Void) size U64 nmemb U64 stream (Pointer Void)] -> U64)
(extern-fn fwrite [ptr (Pointer Void) size U64 nmemb U64 stream (Pointer Void)] -> U64)
(extern-fn fseek [stream (Pointer Void) offset I64 whence I32] -> I32)
(extern-fn ftell [stream (Pointer Void)] -> I64)
(extern-fn fprintf [stream (Pointer Void) fmt String] -> I32)

;; Memory utilities
(extern-fn malloc [size U64] -> (Pointer Void))
(extern-fn calloc [nmemb U64 size U64] -> (Pointer Void))
(extern-fn free [ptr (Pointer Void)] -> Nil)
(extern-fn memset [s (Pointer Void) c I32 n U64] -> (Pointer Void))
(extern-fn memcpy [dest (Pointer Void) src (Pointer Void) n U64] -> (Pointer Void))

;; Printing for debug
(extern-fn printf [fmt String] -> I32)

;; ============================================================================
;; Configuration Struct
;; ============================================================================

(def GPT2Config (: Type)
  (Struct
    [max_seq_len I32]     ; maximum sequence length (e.g., 1024)
    [vocab_size I32]      ; vocabulary size (e.g., 50257)
    [num_layers I32]      ; number of transformer layers (e.g., 12)
    [num_heads I32]       ; number of attention heads (e.g., 12)
    [channels I32]))      ; embedding dimension (e.g., 768)

;; ============================================================================
;; Parameter Tensors
;; ============================================================================
;; Stores all the learned parameters of the model
;; In a real implementation, these would be loaded from a checkpoint

(def ParameterTensors (: Type)
  (Struct
    ;; Token and position embeddings
    [wte (Pointer F32)]          ; (vocab_size, channels) token embeddings
    [wpe (Pointer F32)]          ; (max_seq_len, channels) position embeddings

    ;; Layer norm parameters (per layer)
    [ln1w (Pointer F32)]         ; (num_layers, channels) layernorm1 weights
    [ln1b (Pointer F32)]         ; (num_layers, channels) layernorm1 biases
    [ln2w (Pointer F32)]         ; (num_layers, channels) layernorm2 weights
    [ln2b (Pointer F32)]         ; (num_layers, channels) layernorm2 biases

    ;; Attention parameters (per layer)
    [qkvw (Pointer F32)]         ; (num_layers, 3*channels, channels) QKV weights
    [qkvb (Pointer F32)]         ; (num_layers, 3*channels) QKV biases
    [attprojw (Pointer F32)]     ; (num_layers, channels, channels) attention projection weights
    [attprojb (Pointer F32)]     ; (num_layers, channels) attention projection biases

    ;; MLP parameters (per layer)
    [fcw (Pointer F32)]          ; (num_layers, 4*channels, channels) first FC layer weights
    [fcb (Pointer F32)]          ; (num_layers, 4*channels) first FC layer biases
    [fcprojw (Pointer F32)]      ; (num_layers, channels, 4*channels) FC projection weights
    [fcprojb (Pointer F32)]      ; (num_layers, channels) FC projection biases

    ;; Final layer norm
    [lnfw (Pointer F32)]         ; (channels) final layernorm weights
    [lnfb (Pointer F32)]         ; (channels) final layernorm biases

    [num_parameters U64]))       ; total number of parameters

;; ============================================================================
;; Activation Tensors
;; ============================================================================
;; Stores all intermediate activations during forward pass
;; These would be needed for backward pass in training

(def ActivationTensors (: Type)
  (Struct
    [encoded (Pointer F32)]      ; (B, T, C) after embedding
    [ln1 (Pointer F32)]          ; (B, T, C) after first layernorm
    [ln1_mean (Pointer F32)]     ; (B, T) layernorm1 means
    [ln1_rstd (Pointer F32)]     ; (B, T) layernorm1 rstd
    [qkv (Pointer F32)]          ; (B, T, 3*C) query, key, value
    [atty (Pointer F32)]         ; (B, T, C) attention output
    [att (Pointer F32)]          ; (B, NH, T, T) attention weights
    [attproj (Pointer F32)]      ; (B, T, C) after attention projection
    [residual2 (Pointer F32)]    ; (B, T, C) after residual connection 2
    [ln2 (Pointer F32)]          ; (B, T, C) after second layernorm
    [ln2_mean (Pointer F32)]     ; (B, T) layernorm2 means
    [ln2_rstd (Pointer F32)]     ; (B, T) layernorm2 rstd
    [fch (Pointer F32)]          ; (B, T, 4*C) after first FC layer
    [fch_gelu (Pointer F32)]     ; (B, T, 4*C) after GELU
    [fcproj (Pointer F32)]       ; (B, T, C) after FC projection
    [residual3 (Pointer F32)]    ; (B, T, C) after residual connection 3
    [lnf (Pointer F32)]          ; (B, T, C) after final layernorm
    [lnf_mean (Pointer F32)]     ; (B, T) final layernorm means
    [lnf_rstd (Pointer F32)]     ; (B, T) final layernorm rstd
    [logits (Pointer F32)]       ; (B, T, vocab_size) output logits
    [probs (Pointer F32)]        ; (B, T, vocab_size) output probabilities

    [B I32]                      ; batch size
    [T I32]                      ; sequence length
    [C I32]                      ; channels
    [NH I32]                     ; number of heads
    [num_activations U64]))      ; total activation count

;; ============================================================================
;; Attention Forward (Simplified)
;; ============================================================================
;; Multi-head self-attention mechanism
;; This is a simplified version focusing on the core computation

(def attention_forward (: (-> [(Pointer F32) (Pointer F32) (Pointer F32) I32 I32 I32 I32] Nil))
  (fn [out att qkv B T C NH]
    ;; Head dimension
    (let [hs (: I32) (/ C NH)
          b (: I32) 0]

      (while (< b B)
        (let [h (: I32) 0]
          (while (< h NH)
            ;; For each head, compute attention scores
            (let [t (: I32) 0]
              (while (< t T)
                (let [query_t (: (Pointer F32))
                        (+ qkv (* (* (+ (+ (* (* b T) t) (* h hs)) 0) C) 4))
                      t2 (: I32) 0]

                  ;; Compute attention scores for this query against all keys
                  (while (< t2 T)
                    (let [key_t2 (: (Pointer F32))
                            (+ qkv (* (* (+ (+ (* (* b T) t2) (* h hs)) C) C) 4))
                          score (: F32) 0.0
                          i (: I32) 0]

                      ;; Dot product: Q * K^T
                      (while (< i hs)
                        (set! score (+ score
                          (* (pointer-index-read query_t i)
                             (pointer-index-read key_t2 i))))
                        (set! i (+ i 1)))

                      ;; Scale by sqrt(head_dim)
                      (set! score (/ score (sqrtf (+ hs 0.0))))

                      ;; Store attention score
                      (let [att_idx (: I32) (+ (+ (+ (* (* (* b NH) h) T) t) T) t2)]
                        (pointer-index-write! att att_idx score)))
                    (set! t2 (+ t2 1))))
                (set! t (+ t 1))))
            (set! h (+ h 1))))
        (set! b (+ b 1)))

      ;; Apply softmax to attention scores (per query)
      ;; TODO: Call softmax_forward here

      ;; Compute weighted sum of values
      (set! b 0)
      (while (< b B)
        (let [h (: I32) 0]
          (while (< h NH)
            (let [t (: I32) 0]
              (while (< t T)
                (let [i (: I32) 0]
                  (while (< i hs)
                    (let [sum (: F32) 0.0
                          t2 (: I32) 0]
                      (while (< t2 T)
                        (let [att_idx (: I32) (+ (+ (+ (* (* (* b NH) h) T) t) T) t2)
                              att_weight (: F32) (pointer-index-read att att_idx)
                              value_t2 (: (Pointer F32))
                                (+ qkv (* (* (+ (+ (* (* b T) t2) (* h hs)) (* (* 2 C) i)) C) 4))
                              val (: F32) (pointer-index-read value_t2 i)]
                          (set! sum (+ sum (* att_weight val))))
                        (set! t2 (+ t2 1)))

                      ;; Write output
                      (let [out_idx (: I32) (+ (* (+ (+ (* (* b T) t) (* h hs)) i) C) 4)]
                        (pointer-index-write! out out_idx sum)))
                    (set! i (+ i 1))))
                (set! t (+ t 1))))
            (set! h (+ h 1))))
        (set! b (+ b 1)))
      nil)))

;; ============================================================================
;; Helper: Allocate Float Array
;; ============================================================================

(def malloc_f32 (: (-> [U64] (Pointer F32)))
  (fn [n]
    (let [ptr (: (Pointer Void)) (calloc n 4)]
      ptr)))

;; ============================================================================
;; Initialize Config
;; ============================================================================

(def make_gpt2_config (: (-> [I32 I32 I32 I32 I32] GPT2Config))
  (fn [max_seq_len vocab_size num_layers num_heads channels]
    (GPT2Config max_seq_len vocab_size num_layers num_heads channels)))

;; ============================================================================
;; Print Config (for debugging)
;; ============================================================================

(def print_gpt2_config (: (-> [GPT2Config] Nil))
  (fn [config]
    (printf "GPT2 Config:\n")
    (printf "  max_seq_len: %d\n")
    (printf "  vocab_size: %d\n")
    (printf "  num_layers: %d\n")
    (printf "  num_heads: %d\n")
    (printf "  channels: %d\n")
    nil))
