;; GPT-2 Model Configuration
;;
;; This file defines the model architecture and tensor layouts for GPT-2.
;; Based on llm.c model specification.

;; Model Configurations (from llm.c):
;; | Model        | Layers | Hidden | Heads | Params |
;; |--------------|--------|--------|-------|--------|
;; | GPT-2 Small  | 12     | 768    | 12    | 124M   |
;; | GPT-2 Medium | 24     | 1024   | 16    | 355M   |
;; | GPT-2 Large  | 36     | 1280   | 20    | 774M   |
;; | GPT-2 XL     | 48     | 1600   | 25    | 1.5B   |

;; GPT-2 Small Configuration
;; B = batch size (dynamic, typically 1-8)
;; T = sequence length (up to 1024)
;; C = channels/hidden dimension = 768
;; L = number of layers = 12
;; NH = number of attention heads = 12
;; hs = head size = C/NH = 64
;; V = vocab size = 50257

;; Parameter Tensors (16 total):
;; 1. wte: (V, C) - token embeddings
;; 2. wpe: (maxT, C) - position embeddings
;; 3. ln1w: (L, C) - layernorm1 weights
;; 4. ln1b: (L, C) - layernorm1 biases
;; 5. qkvw: (L, 3*C, C) - QKV projection weights
;; 6. qkvb: (L, 3*C) - QKV projection biases
;; 7. attprojw: (L, C, C) - attention output projection weights
;; 8. attprojb: (L, C) - attention output projection biases
;; 9. ln2w: (L, C) - layernorm2 weights
;; 10. ln2b: (L, C) - layernorm2 biases
;; 11. fcw: (L, 4*C, C) - MLP first layer weights
;; 12. fcb: (L, 4*C) - MLP first layer biases
;; 13. fcprojw: (L, C, 4*C) - MLP projection weights
;; 14. fcprojb: (L, C) - MLP projection biases
;; 15. lnfw: (C,) - final layernorm weights
;; 16. lnfb: (C,) - final layernorm biases

;; Activation Tensors (for forward pass):
;; 1. encoded: (B, T, C) - token + position embeddings
;; 2. ln1: (L, B, T, C) - layernorm1 output
;; 3. qkv: (L, B, T, 3*C) - QKV projections
;; 4. atty: (L, B, T, C) - attention output
;; 5. preatt: (L, B, NH, T, T) - pre-softmax attention scores
;; 6. att: (L, B, NH, T, T) - post-softmax attention weights
;; 7. attproj: (L, B, T, C) - projected attention output
;; 8. residual2: (L, B, T, C) - after first residual
;; 9. ln2: (L, B, T, C) - layernorm2 output
;; 10. fch: (L, B, T, 4*C) - MLP hidden
;; 11. fch_gelu: (L, B, T, 4*C) - after GELU
;; 12. fcproj: (L, B, T, C) - MLP output
;; 13. residual3: (L, B, T, C) - after second residual
;; 14. lnf: (B, T, C) - final layernorm output
;; 15. logits: (B, T, V) - output logits
;; 16. probs: (B, T, V) - softmax probabilities

;; Forward Pass Structure (pseudo-MLIR):
;;
;; func.func @gpt2_forward(
;;     %tokens: memref<BxTxi32>,      // input token IDs
;;     %params: ParameterTensors,     // all model weights
;;     %acts: ActivationTensors       // scratch buffers
;; ) -> memref<BxTxVxf32> {           // output probabilities
;;
;;   // 1. Encoder: token_emb + pos_emb
;;   @encoder_forward(%acts.encoded, %tokens, %params.wte, %params.wpe)
;;
;;   // 2. Transformer Blocks (L iterations)
;;   %residual = %acts.encoded
;;   for l = 0 to L {
;;     // 2a. LayerNorm 1
;;     @layernorm_forward(%acts.ln1[l], %residual, %params.ln1w[l], %params.ln1b[l])
;;
;;     // 2b. QKV Projection (matmul)
;;     @matmul(%acts.qkv[l], %acts.ln1[l], %params.qkvw[l])
;;     @bias_add(%acts.qkv[l], %params.qkvb[l])
;;
;;     // 2c. Self-Attention
;;     @attention_forward(%acts.atty[l], %acts.preatt[l], %acts.att[l], %acts.qkv[l])
;;
;;     // 2d. Attention Output Projection
;;     @matmul(%acts.attproj[l], %acts.atty[l], %params.attprojw[l])
;;     @bias_add(%acts.attproj[l], %params.attprojb[l])
;;
;;     // 2e. Residual 1
;;     @residual_add(%acts.residual2[l], %residual, %acts.attproj[l])
;;
;;     // 2f. LayerNorm 2
;;     @layernorm_forward(%acts.ln2[l], %acts.residual2[l], %params.ln2w[l], %params.ln2b[l])
;;
;;     // 2g. MLP: FC -> GELU -> Proj
;;     @matmul(%acts.fch[l], %acts.ln2[l], %params.fcw[l])
;;     @bias_add(%acts.fch[l], %params.fcb[l])
;;     @gelu_forward(%acts.fch_gelu[l], %acts.fch[l])
;;     @matmul(%acts.fcproj[l], %acts.fch_gelu[l], %params.fcprojw[l])
;;     @bias_add(%acts.fcproj[l], %params.fcprojb[l])
;;
;;     // 2h. Residual 2
;;     @residual_add(%acts.residual3[l], %acts.residual2[l], %acts.fcproj[l])
;;     %residual = %acts.residual3[l]
;;   }
;;
;;   // 3. Final LayerNorm
;;   @layernorm_forward(%acts.lnf, %residual, %params.lnfw, %params.lnfb)
;;
;;   // 4. Output projection (share wte)
;;   @matmul(%acts.logits, %acts.lnf, %params.wte^T)
;;
;;   // 5. Softmax
;;   @softmax_forward(%acts.probs, %acts.logits)
;;
;;   return %acts.probs
;; }
