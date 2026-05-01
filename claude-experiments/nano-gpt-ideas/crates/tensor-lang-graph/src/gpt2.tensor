// GPT-2 forward pass.
//
// Symbolic dims (bound at compile time via compile_with_env):
//   B, T, vocab_size, n_embd, n_head, n_layer, mlp_hidden, head_size
//
// Named float constants (also passed in at compile time):
//   inv_d         = 1 / n_embd
//   inv_head_size = 1 / sqrt(head_size)
//
// Note: n_layer is referenced in the for-loop bound and must be a concrete
// value in `dim_values` at compile time (it must resolve to an integer
// literal for loop unrolling). B, T, vocab_size, n_embd, n_head, head_size,
// mlp_hidden can be kept symbolic.

dim B
dim T
dim vocab_size
dim n_embd
dim n_head
dim head_size
dim mlp_hidden
dim n_layer

fn layernorm(x, gamma, beta) {
    let mean = mul(sum(x, axis: 2), inv_d)
    let xc = sub(x, mean)
    let var = mul(sum(mul(xc, xc), axis: 2), inv_d)
    let std = sqrt(add(var, 0.00001))
    let normed = mul(xc, recip(std))
    add(mul(normed, gamma), beta)
}

fn clamp(x, lo, hi) {
    // clamp(x, lo, hi) = max(min(x, hi), lo)
    // min(a, b) = neg(max(neg(a), neg(b)))
    let upper = neg(max(neg(x), neg(hi)))
    max(upper, lo)
}

fn gelu(x) {
    let x3 = mul(mul(x, x), x)
    let inner = mul(0.7978845608028654, add(x, mul(0.044715, x3)))
    // Clamp for numerically stable tanh (tanh(10) ≈ 1.0)
    let clamped = clamp(inner, neg(10.0), 10.0)
    let z2 = mul(clamped, 2.0)
    let ez2 = exp(z2)
    let tanh_val = mul(sub(ez2, 1.0), recip(add(ez2, 1.0)))
    mul(mul(0.5, x), add(1.0, tanh_val))
}

fn linear(x, w, b) {
    add(matmul(x, w), b)
}

fn softmax_attn(x) {
    let m = max(x, axis: 3)
    let e = exp(sub(x, m))
    let s = sum(e, axis: 3)
    mul(recip(s), e)
}

// Shared inputs
let tokens    = load([B, T])
let wte       = load([vocab_size, n_embd])
let wpe       = load([T, n_embd])
let attn_mask = load([B, 1, T, T])

// Expand causal mask to all heads (once, reused by every layer)
let mask_full = expand(attn_mask, [B, n_head, T, T])

// Token + positional embedding (one-hot lookup via arange broadcast)
let classes = arange(vocab_size)
let cls     = reshape(classes, [1, 1, vocab_size])
let cls_exp = expand(cls, [B, T, vocab_size])
let tok_r   = reshape(tokens, [B, T, 1])
let tok_exp = expand(tok_r, [B, T, vocab_size])
let lo      = sub(cls_exp, 0.5)
let hi      = add(cls_exp, 0.5)
let one_hot = mul(cmplt(lo, tok_exp), cmplt(tok_exp, hi))
let tok_emb = matmul(one_hot, wte)
let pos_emb = reshape(wpe, [1, T, n_embd])
let x = add(tok_emb, expand(pos_emb, [B, T, n_embd]))

// Transformer stack — unrolled at compile time.
for i in 0..n_layer {
    let ln1_g       = load([n_embd])
    let ln1_b       = load([n_embd])
    let attn_qkv_w  = load([n_embd, 3*n_embd])
    let attn_qkv_b  = load([3*n_embd])
    let attn_proj_w = load([n_embd, n_embd])
    let attn_proj_b = load([n_embd])
    let ln2_g       = load([n_embd])
    let ln2_b       = load([n_embd])
    let mlp_fc_w    = load([n_embd, mlp_hidden])
    let mlp_fc_b    = load([mlp_hidden])
    let mlp_proj_w  = load([mlp_hidden, n_embd])
    let mlp_proj_b  = load([n_embd])

    // Attention block
    let ln1_out  = layernorm(x, ln1_g, ln1_b)
    let qkv      = linear(ln1_out, attn_qkv_w, attn_qkv_b)
    let qkv_r    = reshape(qkv, [B, T, 3, n_head, head_size])
    let q        = reshape(shrink(qkv_r, [[0, B], [0, T], [0, 1], [0, n_head], [0, head_size]]), [B, T, n_head, head_size])
    let k        = reshape(shrink(qkv_r, [[0, B], [0, T], [1, 2], [0, n_head], [0, head_size]]), [B, T, n_head, head_size])
    let v        = reshape(shrink(qkv_r, [[0, B], [0, T], [2, 3], [0, n_head], [0, head_size]]), [B, T, n_head, head_size])
    let q_h      = permute(q, [0, 2, 1, 3])
    let k_h      = permute(k, [0, 2, 1, 3])
    let v_h      = permute(v, [0, 2, 1, 3])
    let kt       = permute(k_h, [0, 1, 3, 2])
    let scores   = mul(matmul(q_h, kt), inv_head_size)
    let masked   = add(scores, mask_full)
    let attn     = softmax_attn(masked)
    let attn_out = matmul(attn, v_h)
    let merged   = reshape(permute(attn_out, [0, 2, 1, 3]), [B, T, n_embd])
    let attn_res = linear(merged, attn_proj_w, attn_proj_b)
    let x = add(x, attn_res)

    // MLP block
    let ln2_out  = layernorm(x, ln2_g, ln2_b)
    let fc       = gelu(linear(ln2_out, mlp_fc_w, mlp_fc_b))
    let mlp_res  = linear(fc, mlp_proj_w, mlp_proj_b)
    let x = add(x, mlp_res)
}

// Final layer norm + weight-tied output projection.
let ln_f_g = load([n_embd])
let ln_f_b = load([n_embd])
let x_norm = layernorm(x, ln_f_g, ln_f_b)
let wte_t  = permute(reshape(wte, [1, vocab_size, n_embd]), [0, 2, 1])
let wte_2d = reshape(wte_t, [n_embd, vocab_size])
let logits = matmul(x_norm, wte_2d)
