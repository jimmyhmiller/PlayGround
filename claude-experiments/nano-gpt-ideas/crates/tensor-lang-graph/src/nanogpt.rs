/// Static GPT-2 source written in the tensor DSL itself (no Rust
/// templating). Compile via [`compile_gpt2`].
///
/// Input order (matches what [`compile_gpt2`] produces as `load(...)`
/// nodes, in textual order through the static DSL):
///   0: tokens     [B, T] (float indices)
///   1: wte        [vocab_size, n_embd]
///   2: wpe        [T, n_embd]
///   3: attn_mask  [B, 1, T, T] (pre-scaled: 0.0=attend, -1e6=masked)
///   For each layer i in 0..n_layer:
///     4 + i*12 + 0:  ln1_gamma   [n_embd]
///     4 + i*12 + 1:  ln1_beta    [n_embd]
///     4 + i*12 + 2:  attn_qkv_w  [n_embd, 3*n_embd]  (pre-transposed)
///     4 + i*12 + 3:  attn_qkv_b  [3*n_embd]
///     4 + i*12 + 4:  attn_proj_w [n_embd, n_embd]    (pre-transposed)
///     4 + i*12 + 5:  attn_proj_b [n_embd]
///     4 + i*12 + 6:  ln2_gamma   [n_embd]
///     4 + i*12 + 7:  ln2_beta    [n_embd]
///     4 + i*12 + 8:  mlp_fc_w    [n_embd, 4*n_embd]   (pre-transposed)
///     4 + i*12 + 9:  mlp_fc_b    [4*n_embd]
///     4 + i*12 + 10: mlp_proj_w  [4*n_embd, n_embd]   (pre-transposed)
///     4 + i*12 + 11: mlp_proj_b  [n_embd]
///   Final:
///     4 + n_layer*12 + 0: ln_f_gamma [n_embd]
///     4 + n_layer*12 + 1: ln_f_beta  [n_embd]
///   (lm_head reuses wte due to weight tying)
pub const GPT2_DSL: &str = include_str!("gpt2.tensor");

/// Compile the static `gpt2.tensor` source into a [`crate::Graph`] with the
/// given configuration. `seq_len = None` leaves `T` symbolic (bind at
/// runtime via `run_with_dim_params`); `Some(t)` bakes `T = t` in.
/// `n_layer` must be concrete because the layer loop is unrolled at
/// compile time.
pub fn compile_gpt2(
    batch: usize,
    seq_len: Option<usize>,
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
) -> crate::Graph {
    use std::collections::HashMap;
    let head_size = n_embd / n_head;
    let mlp_hidden = 4 * n_embd;

    let mut dims: HashMap<String, usize> = HashMap::new();
    dims.insert("B".into(), batch);
    if let Some(t) = seq_len {
        dims.insert("T".into(), t);
    }
    dims.insert("vocab_size".into(), vocab_size);
    dims.insert("n_embd".into(), n_embd);
    dims.insert("n_head".into(), n_head);
    dims.insert("head_size".into(), head_size);
    dims.insert("mlp_hidden".into(), mlp_hidden);
    dims.insert("n_layer".into(), n_layer);

    let inv_d = 1.0 / n_embd as f64;
    let inv_head_size = 1.0 / (head_size as f64).sqrt();

    let mut consts: HashMap<String, f64> = HashMap::new();
    consts.insert("inv_d".into(), inv_d);
    consts.insert("inv_head_size".into(), inv_head_size);

    crate::compile_with_env(GPT2_DSL, &dims, &consts)
}

/// Return the total number of inputs the compiled GPT-2 graph expects.
///   4 fixed inputs (tokens, wte, wpe, attn_mask)
/// + 12 per transformer layer
/// + 2 final (ln_f gamma/beta)
pub fn nanogpt_input_count(n_layer: usize) -> usize {
    4 + n_layer * 12 + 2
}
