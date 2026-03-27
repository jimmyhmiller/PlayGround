/// Generate a nanoGPT forward-pass program in our tensor DSL.
///
/// The generated program expects inputs loaded in this order:
///   0: tokens [B, T] (float indices)
///   1: wte [vocab_size, n_embd]
///   2: wpe [T, n_embd]
///   3: attn_mask [B, 1, T, T] (pre-scaled: 0.0=attend, -1e6=masked)
///   Then for each layer i (0..n_layer):
///     4 + i*12 + 0:  ln1_gamma [n_embd]
///     4 + i*12 + 1:  ln1_beta  [n_embd]
///     4 + i*12 + 2:  attn_qkv_w [n_embd, 3*n_embd]  (pre-transposed)
///     4 + i*12 + 3:  attn_qkv_b [3*n_embd]
///     4 + i*12 + 4:  attn_proj_w [n_embd, n_embd]    (pre-transposed)
///     4 + i*12 + 5:  attn_proj_b [n_embd]
///     4 + i*12 + 6:  ln2_gamma [n_embd]
///     4 + i*12 + 7:  ln2_beta  [n_embd]
///     4 + i*12 + 8:  mlp_fc_w [n_embd, 4*n_embd]     (pre-transposed)
///     4 + i*12 + 9:  mlp_fc_b [4*n_embd]
///     4 + i*12 + 10: mlp_proj_w [4*n_embd, n_embd]    (pre-transposed)
///     4 + i*12 + 11: mlp_proj_b [n_embd]
///   Final:
///     4 + n_layer*12 + 0: ln_f_gamma [n_embd]
///     4 + n_layer*12 + 1: ln_f_beta  [n_embd]
///   (lm_head reuses wte due to weight tying)
///
/// Returns the DSL source string with concrete seq_len.
pub fn generate_nanogpt_program(
    batch: usize,
    seq_len: usize,
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
) -> String {
    generate_nanogpt_program_inner(batch, &seq_len.to_string(), vocab_size, n_embd, n_head, n_layer, false)
}

/// Returns the DSL source string with symbolic seq_len (dimension parameter "T").
pub fn generate_nanogpt_program_symbolic(
    batch: usize,
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
) -> String {
    generate_nanogpt_program_inner(batch, "T", vocab_size, n_embd, n_head, n_layer, true)
}

fn generate_nanogpt_program_inner(
    batch: usize,
    seq_len: &str,
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    symbolic: bool,
) -> String {
    let head_size = n_embd / n_head;
    let mlp_hidden = 4 * n_embd;
    let inv_d = 1.0 / n_embd as f64;
    let inv_head_size = 1.0 / (head_size as f64).sqrt();

    let mut p = String::new();

    // Declare symbolic dim if needed
    if symbolic {
        p.push_str("dim T\n");
    }

    // Helper functions
    p.push_str(&format!(r#"
fn layernorm(x, gamma, beta) {{
    let mean = mul(sum(x, axis: 2), {inv_d})
    let xc = sub(x, mean)
    let var = mul(sum(mul(xc, xc), axis: 2), {inv_d})
    let std = sqrt(add(var, 0.00001))
    let normed = mul(xc, recip(std))
    add(mul(normed, gamma), beta)
}}

fn clamp(x, lo, hi) {{
    // clamp(x, lo, hi) = max(min(x, hi), lo)
    // min(a, b) = neg(max(neg(a), neg(b)))
    let upper = neg(max(neg(x), neg(hi)))
    max(upper, lo)
}}

fn gelu(x) {{
    let x3 = mul(mul(x, x), x)
    let inner = mul(0.7978845608028654, add(x, mul(0.044715, x3)))
    // Clamp for numerically stable tanh (tanh(10) ≈ 1.0)
    let clamped = clamp(inner, neg(10.0), 10.0)
    let z2 = mul(clamped, 2.0)
    let ez2 = exp(z2)
    let tanh_val = mul(sub(ez2, 1.0), recip(add(ez2, 1.0)))
    mul(mul(0.5, x), add(1.0, tanh_val))
}}

fn linear(x, w, b) {{
    add(matmul(x, w), b)
}}

fn softmax_attn(x) {{
    let m = max(x, axis: 3)
    let e = exp(sub(x, m))
    let s = sum(e, axis: 3)
    mul(recip(s), e)
}}

"#));

    // Load shared weights and attention mask
    p.push_str(&format!("let tokens = load([{batch}, {seq_len}])\n"));
    p.push_str(&format!("let wte = load([{vocab_size}, {n_embd}])\n"));
    p.push_str(&format!("let wpe = load([{seq_len}, {n_embd}])\n"));
    p.push_str(&format!("let attn_mask = load([{batch}, 1, {seq_len}, {seq_len}])\n"));

    // Expand mask to all heads (once, reused by all layers)
    p.push_str(&format!("let mask_full = expand(attn_mask, [{batch}, {n_head}, {seq_len}, {seq_len}])\n"));

    // Embedding: one-hot tokens @ wte + wpe
    // one_hot: for each position, create a [vocab_size] indicator vector
    p.push_str(&format!(r#"
let classes = arange({vocab_size})
let cls = reshape(classes, [1, 1, {vocab_size}])
let cls_exp = expand(cls, [{batch}, {seq_len}, {vocab_size}])
let tok_r = reshape(tokens, [{batch}, {seq_len}, 1])
let tok_exp = expand(tok_r, [{batch}, {seq_len}, {vocab_size}])
let lo = sub(cls_exp, 0.5)
let hi = add(cls_exp, 0.5)
let one_hot = mul(cmplt(lo, tok_exp), cmplt(tok_exp, hi))
let tok_emb = matmul(one_hot, wte)
let pos_emb = reshape(wpe, [1, {seq_len}, {n_embd}])
let x = add(tok_emb, expand(pos_emb, [{batch}, {seq_len}, {n_embd}]))
"#));

    // Transformer layers
    for i in 0..n_layer {
        // Load layer weights
        p.push_str(&format!("\n// Layer {i}\n"));
        p.push_str(&format!("let ln1_g_{i} = load([{n_embd}])\n"));
        p.push_str(&format!("let ln1_b_{i} = load([{n_embd}])\n"));
        p.push_str(&format!("let attn_qkv_w_{i} = load([{n_embd}, {}])\n", 3 * n_embd));
        p.push_str(&format!("let attn_qkv_b_{i} = load([{}])\n", 3 * n_embd));
        p.push_str(&format!("let attn_proj_w_{i} = load([{n_embd}, {n_embd}])\n"));
        p.push_str(&format!("let attn_proj_b_{i} = load([{n_embd}])\n"));
        p.push_str(&format!("let ln2_g_{i} = load([{n_embd}])\n"));
        p.push_str(&format!("let ln2_b_{i} = load([{n_embd}])\n"));
        p.push_str(&format!("let mlp_fc_w_{i} = load([{n_embd}, {mlp_hidden}])\n"));
        p.push_str(&format!("let mlp_fc_b_{i} = load([{mlp_hidden}])\n"));
        p.push_str(&format!("let mlp_proj_w_{i} = load([{mlp_hidden}, {n_embd}])\n"));
        p.push_str(&format!("let mlp_proj_b_{i} = load([{n_embd}])\n"));

        // Attention block
        p.push_str(&format!(r#"
let ln1_{i} = layernorm(x, ln1_g_{i}, ln1_b_{i})
let qkv_{i} = linear(ln1_{i}, attn_qkv_w_{i}, attn_qkv_b_{i})
let qkv_r_{i} = reshape(qkv_{i}, [{batch}, {seq_len}, 3, {n_head}, {head_size}])
let q_{i} = reshape(shrink(qkv_r_{i}, [[0, {batch}], [0, {seq_len}], [0, 1], [0, {n_head}], [0, {head_size}]]), [{batch}, {seq_len}, {n_head}, {head_size}])
let k_{i} = reshape(shrink(qkv_r_{i}, [[0, {batch}], [0, {seq_len}], [1, 2], [0, {n_head}], [0, {head_size}]]), [{batch}, {seq_len}, {n_head}, {head_size}])
let v_{i} = reshape(shrink(qkv_r_{i}, [[0, {batch}], [0, {seq_len}], [2, 3], [0, {n_head}], [0, {head_size}]]), [{batch}, {seq_len}, {n_head}, {head_size}])
let q_h_{i} = permute(q_{i}, [0, 2, 1, 3])
let k_h_{i} = permute(k_{i}, [0, 2, 1, 3])
let v_h_{i} = permute(v_{i}, [0, 2, 1, 3])
let kt_{i} = permute(k_h_{i}, [0, 1, 3, 2])
let scores_{i} = mul(matmul(q_h_{i}, kt_{i}), {inv_head_size})
let masked_{i} = add(scores_{i}, mask_full)
let attn_{i} = softmax_attn(masked_{i})
let attn_out_{i} = matmul(attn_{i}, v_h_{i})
let merged_{i} = reshape(permute(attn_out_{i}, [0, 2, 1, 3]), [{batch}, {seq_len}, {n_embd}])
let attn_proj_{i} = linear(merged_{i}, attn_proj_w_{i}, attn_proj_b_{i})
let x = add(x, attn_proj_{i})
"#));

        // MLP block
        p.push_str(&format!(r#"
let ln2_{i} = layernorm(x, ln2_g_{i}, ln2_b_{i})
let fc_{i} = gelu(linear(ln2_{i}, mlp_fc_w_{i}, mlp_fc_b_{i}))
let mlp_out_{i} = linear(fc_{i}, mlp_proj_w_{i}, mlp_proj_b_{i})
let x = add(x, mlp_out_{i})
"#));
    }

    // Final layer norm + output projection (weight-tied with wte)
    p.push_str(&format!("let ln_f_g = load([{n_embd}])\n"));
    p.push_str(&format!("let ln_f_b = load([{n_embd}])\n"));
    p.push_str(&format!(r#"
let x_norm = layernorm(x, ln_f_g, ln_f_b)
let wte_t = permute(reshape(wte, [1, {vocab_size}, {n_embd}]), [0, 2, 1])
let wte_2d = reshape(wte_t, [{n_embd}, {vocab_size}])
let logits = matmul(x_norm, wte_2d)
"#));

    p
}

/// Return the total number of inputs the generated program expects.
pub fn nanogpt_input_count(n_layer: usize) -> usize {
    4 + n_layer * 12 + 2  // tokens, wte, wpe, attn_mask, 12 per layer, ln_f_gamma, ln_f_beta
}

/// Return the input names in order with their shapes.
pub fn nanogpt_input_layout(
    batch: usize,
    seq_len: usize,
    vocab_size: usize,
    n_embd: usize,
    n_layer: usize,
) -> Vec<(String, Vec<usize>)> {
    let mlp_hidden = 4 * n_embd;
    let mut inputs = vec![
        ("tokens".into(), vec![batch, seq_len]),
        ("wte".into(), vec![vocab_size, n_embd]),
        ("wpe".into(), vec![seq_len, n_embd]),
        ("attn_mask".into(), vec![batch, 1, seq_len, seq_len]),
    ];
    for i in 0..n_layer {
        inputs.extend([
            (format!("ln1_g_{i}"), vec![n_embd]),
            (format!("ln1_b_{i}"), vec![n_embd]),
            (format!("attn_qkv_w_{i}"), vec![n_embd, 3 * n_embd]),
            (format!("attn_qkv_b_{i}"), vec![3 * n_embd]),
            (format!("attn_proj_w_{i}"), vec![n_embd, n_embd]),
            (format!("attn_proj_b_{i}"), vec![n_embd]),
            (format!("ln2_g_{i}"), vec![n_embd]),
            (format!("ln2_b_{i}"), vec![n_embd]),
            (format!("mlp_fc_w_{i}"), vec![n_embd, mlp_hidden]),
            (format!("mlp_fc_b_{i}"), vec![mlp_hidden]),
            (format!("mlp_proj_w_{i}"), vec![mlp_hidden, n_embd]),
            (format!("mlp_proj_b_{i}"), vec![n_embd]),
        ]);
    }
    inputs.extend([
        ("ln_f_g".into(), vec![n_embd]),
        ("ln_f_b".into(), vec![n_embd]),
    ]);
    inputs
}
