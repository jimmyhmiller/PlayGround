;; GPT-2 Text Generation
;;
;; Implements autoregressive token generation using the forward pass.
;;
;; Generation Loop:
;; 1. Start with prompt tokens
;; 2. Run forward pass to get logits for next token
;; 3. Sample next token from probability distribution
;; 4. Append token to sequence
;; 5. Repeat until max length or EOS token

(require-dialect gpu)
(require-dialect func)
(require-dialect arith)
(require-dialect memref)
(require-dialect scf)

;; Generation Pseudo-code:
;;
;; func.func @generate(
;;     %prompt_tokens: memref<?xi32>,  // Initial tokens
;;     %prompt_len: index,              // Number of prompt tokens
;;     %max_tokens: index,              // Maximum generation length
;;     %params: ParameterTensors        // Model weights on GPU
;; ) -> memref<?xi32> {                 // Generated token sequence
;;
;;   // Allocate output buffer
;;   %output = memref.alloc(%max_tokens) : memref<?xi32>
;;
;;   // Copy prompt to output
;;   scf.for %i = 0 to %prompt_len {
;;     %tok = memref.load %prompt_tokens[%i]
;;     memref.store %tok, %output[%i]
;;   }
;;
;;   // Generation loop
;;   %pos = %prompt_len
;;   scf.while (%pos < %max_tokens) {
;;     // Run forward pass for current sequence
;;     %logits = call @gpt2_forward(%output[0:%pos], %params)
;;
;;     // Get logits for last position: logits[pos-1]
;;     %last_logits = memref.subview %logits[%pos-1, :] : memref<V>
;;
;;     // Sample next token (argmax for greedy, or temperature sampling)
;;     %next_token = call @sample(%last_logits)
;;
;;     // Append to output
;;     memref.store %next_token, %output[%pos]
;;
;;     // Check for EOS (token 50256)
;;     %is_eos = arith.cmpi eq, %next_token, 50256
;;     scf.if %is_eos {
;;       scf.yield  // Exit loop
;;     }
;;
;;     %pos = %pos + 1
;;   }
;;
;;   return %output
;; }

;; Sampling Strategies:
;;
;; 1. Greedy (argmax):
;;    %max_idx = 0
;;    %max_val = logits[0]
;;    for i in 1..V:
;;      if logits[i] > max_val:
;;        max_idx = i
;;        max_val = logits[i]
;;    return max_idx
;;
;; 2. Temperature Sampling:
;;    // Apply temperature
;;    for i in 0..V:
;;      logits[i] = logits[i] / temperature
;;
;;    // Softmax
;;    probs = softmax(logits)
;;
;;    // Sample from distribution
;;    rand = random()
;;    cumsum = 0
;;    for i in 0..V:
;;      cumsum += probs[i]
;;      if cumsum > rand:
;;        return i
;;
;; 3. Top-k Sampling:
;;    // Sort and keep top k tokens
;;    // Sample from reduced distribution

;; GPU Optimization Notes:
;;
;; For efficient generation:
;; 1. Keep all weights on GPU
;; 2. Keep growing context on GPU
;; 3. Only copy final token IDs back to host
;; 4. Use KV-cache to avoid recomputing attention for past tokens
;;
;; KV-Cache (future optimization):
;; - Store K and V projections from previous positions
;; - Only compute Q for new position
;; - Reduces attention from O(T²) to O(T) per token

;; Example Usage:
;;
;; // Load model
;; %params = call @load_checkpoint("gpt2_124M.bin")
;;
;; // Encode prompt
;; %prompt = "Hello, world"
;; %prompt_tokens = call @tokenize(%prompt)
;;
;; // Generate
;; %output_tokens = call @generate(%prompt_tokens, 10, 100, %params)
;;
;; // Decode
;; %output_text = call @detokenize(%output_tokens)
;; call @print(%output_text)
