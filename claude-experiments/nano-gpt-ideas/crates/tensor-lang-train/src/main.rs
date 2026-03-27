mod midi_parse;

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use rand::Rng;
use tensor_lang_backend::assemblyscript::AssemblyScriptBackend;
use tensor_lang_graph::{dims, Dim, Graph, NodeId, Op};

// ─── Model config ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct ModelConfig {
    vocab_size: usize,
    seq_len: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    batch_size: usize,
}

impl ModelConfig {
    fn head_size(&self) -> usize { self.n_embd / self.n_head }
    fn mlp_hidden(&self) -> usize { 4 * self.n_embd }
}

// ─── Graph construction ────────────────────────────────────────────────────

struct TrainingGraph {
    graph: Graph,
    loss: NodeId,
    grad_ids: Vec<NodeId>,
}

fn add_input(g: &mut Graph, name: &str, shape: &[usize]) -> NodeId {
    let id = g.add_node(Op::Input { name: name.into() }, vec![]);
    g.set_input_shape(id, dims(shape));
    id
}

fn build_layernorm(g: &mut Graph, x: NodeId, gamma: NodeId, beta: NodeId, n_embd: usize) -> NodeId {
    let inv_d = 1.0 / n_embd as f64;
    let inv_d_const = g.add_node(Op::Constant(inv_d), vec![]);
    let sum_x = g.add_node(Op::ReduceSum { axis: 2 }, vec![x]);
    let mean = g.add_node(Op::Mul, vec![sum_x, inv_d_const]);
    let neg_mean = g.add_node(Op::Neg, vec![mean]);
    let xc = g.add_node(Op::Add, vec![x, neg_mean]);
    let xc2 = g.add_node(Op::Mul, vec![xc, xc]);
    let sum_xc2 = g.add_node(Op::ReduceSum { axis: 2 }, vec![xc2]);
    let var = g.add_node(Op::Mul, vec![sum_xc2, inv_d_const]);
    let eps = g.add_node(Op::Constant(1e-5), vec![]);
    let var_eps = g.add_node(Op::Add, vec![var, eps]);
    let std = g.add_node(Op::Sqrt, vec![var_eps]);
    let inv_std = g.add_node(Op::Recip, vec![std]);
    let normed = g.add_node(Op::Mul, vec![xc, inv_std]);
    let scaled = g.add_node(Op::Mul, vec![normed, gamma]);
    g.add_node(Op::Add, vec![scaled, beta])
}

fn build_gelu(g: &mut Graph, x: NodeId) -> NodeId {
    let x2 = g.add_node(Op::Mul, vec![x, x]);
    let x3 = g.add_node(Op::Mul, vec![x2, x]);
    let c1 = g.add_node(Op::Constant(0.044715), vec![]);
    let c1x3 = g.add_node(Op::Mul, vec![c1, x3]);
    let inner_sum = g.add_node(Op::Add, vec![x, c1x3]);
    let c2 = g.add_node(Op::Constant(0.7978845608028654), vec![]);
    let inner = g.add_node(Op::Mul, vec![c2, inner_sum]);
    let ten = g.add_node(Op::Constant(10.0), vec![]);
    let neg_ten = g.add_node(Op::Neg, vec![ten]);
    let neg_inner = g.add_node(Op::Neg, vec![inner]);
    let max1 = g.add_node(Op::Max, vec![neg_inner, neg_ten]);
    let clamped = g.add_node(Op::Neg, vec![max1]);
    let neg_ten2 = g.add_node(Op::Constant(-10.0), vec![]);
    let clamped2 = g.add_node(Op::Max, vec![clamped, neg_ten2]);
    let two = g.add_node(Op::Constant(2.0), vec![]);
    let z2 = g.add_node(Op::Mul, vec![clamped2, two]);
    let log2e = g.add_node(Op::Constant(std::f64::consts::LOG2_E), vec![]);
    let z2_scaled = g.add_node(Op::Mul, vec![z2, log2e]);
    let ez2 = g.add_node(Op::Exp2, vec![z2_scaled]);
    let one = g.add_node(Op::Constant(1.0), vec![]);
    let neg_one = g.add_node(Op::Constant(-1.0), vec![]);
    let ez2_m1 = g.add_node(Op::Add, vec![ez2, neg_one]);
    let ez2_p1 = g.add_node(Op::Add, vec![ez2, one]);
    let inv_ez2_p1 = g.add_node(Op::Recip, vec![ez2_p1]);
    let tanh_val = g.add_node(Op::Mul, vec![ez2_m1, inv_ez2_p1]);
    let one2 = g.add_node(Op::Constant(1.0), vec![]);
    let one_plus_tanh = g.add_node(Op::Add, vec![one2, tanh_val]);
    let half = g.add_node(Op::Constant(0.5), vec![]);
    let half_x = g.add_node(Op::Mul, vec![half, x]);
    g.add_node(Op::Mul, vec![half_x, one_plus_tanh])
}

fn build_matmul(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    let a_shape = g.nodes[a.0].shape.clone();
    let b_shape = g.nodes[b.0].shape.clone();
    let ndim_a = a_shape.len();
    let ndim_b = b_shape.len();
    let m = a_shape[ndim_a - 2].clone();
    let k = a_shape[ndim_a - 1].clone();
    let n = b_shape[ndim_b - 1].clone();
    let batch_a = &a_shape[..ndim_a - 2];
    let batch_b = &b_shape[..ndim_b - 2];
    let batch = tensor_lang_graph::broadcast_shapes(batch_a, batch_b);
    let mut a_reshaped: Vec<Dim> = batch_a.to_vec();
    a_reshaped.extend([m.clone(), k.clone(), Dim::Lit(1)]);
    let mut b_reshaped: Vec<Dim> = batch_b.to_vec();
    b_reshaped.extend([Dim::Lit(1), k.clone(), n.clone()]);
    let mut expanded: Vec<Dim> = batch.clone();
    expanded.extend([m.clone(), k, n.clone()]);
    let a_r = g.add_node(Op::Reshape { shape: a_reshaped }, vec![a]);
    let b_r = g.add_node(Op::Reshape { shape: b_reshaped }, vec![b]);
    let a_e = g.add_node(Op::Expand { shape: expanded.clone() }, vec![a_r]);
    let b_e = g.add_node(Op::Expand { shape: expanded }, vec![b_r]);
    let prod = g.add_node(Op::Mul, vec![a_e, b_e]);
    let k_axis = batch.len() + 1;
    let summed = g.add_node(Op::ReduceSum { axis: k_axis }, vec![prod]);
    let mut out_shape: Vec<Dim> = batch;
    out_shape.extend([m, n]);
    g.add_node(Op::Reshape { shape: out_shape }, vec![summed])
}

fn build_linear(g: &mut Graph, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let mm = build_matmul(g, x, w);
    g.add_node(Op::Add, vec![mm, b])
}

fn build_softmax(g: &mut Graph, x: NodeId, axis: usize) -> NodeId {
    let mx = g.add_node(Op::ReduceMax { axis }, vec![x]);
    let neg_mx = g.add_node(Op::Neg, vec![mx]);
    let shifted = g.add_node(Op::Add, vec![x, neg_mx]);
    let log2e = g.add_node(Op::Constant(std::f64::consts::LOG2_E), vec![]);
    let scaled = g.add_node(Op::Mul, vec![shifted, log2e]);
    let ex = g.add_node(Op::Exp2, vec![scaled]);
    let sum_ex = g.add_node(Op::ReduceSum { axis }, vec![ex]);
    let inv_sum = g.add_node(Op::Recip, vec![sum_ex]);
    g.add_node(Op::Mul, vec![ex, inv_sum])
}

fn build_log_softmax(g: &mut Graph, x: NodeId, axis: usize) -> NodeId {
    let mx = g.add_node(Op::ReduceMax { axis }, vec![x]);
    let neg_mx = g.add_node(Op::Neg, vec![mx]);
    let shifted = g.add_node(Op::Add, vec![x, neg_mx]);
    let log2e = g.add_node(Op::Constant(std::f64::consts::LOG2_E), vec![]);
    let scaled = g.add_node(Op::Mul, vec![shifted, log2e]);
    let ex = g.add_node(Op::Exp2, vec![scaled]);
    let sum_ex = g.add_node(Op::ReduceSum { axis }, vec![ex]);
    let log2_sum = g.add_node(Op::Log2, vec![sum_ex]);
    let ln2 = g.add_node(Op::Constant(std::f64::consts::LN_2), vec![]);
    let log_sum = g.add_node(Op::Mul, vec![log2_sum, ln2]);
    let neg_log_sum = g.add_node(Op::Neg, vec![log_sum]);
    g.add_node(Op::Add, vec![shifted, neg_log_sum])
}

fn build_one_hot(g: &mut Graph, tokens: NodeId, v: usize, b: usize, t: usize) -> NodeId {
    let classes = g.add_node(Op::Arange { size: Dim::Lit(v) }, vec![]);
    let cls = g.add_node(Op::Reshape { shape: dims(&[1, 1, v]) }, vec![classes]);
    let cls_exp = g.add_node(Op::Expand { shape: dims(&[b, t, v]) }, vec![cls]);
    let tok_r = g.add_node(Op::Reshape { shape: dims(&[b, t, 1]) }, vec![tokens]);
    let tok_exp = g.add_node(Op::Expand { shape: dims(&[b, t, v]) }, vec![tok_r]);
    let half = g.add_node(Op::Constant(0.5), vec![]);
    let neg_half = g.add_node(Op::Neg, vec![half]);
    let lo = g.add_node(Op::Add, vec![cls_exp, neg_half]);
    let half2 = g.add_node(Op::Constant(0.5), vec![]);
    let hi = g.add_node(Op::Add, vec![cls_exp, half2]);
    let lt_lo = g.add_node(Op::CmpLt, vec![lo, tok_exp]);
    let lt_hi = g.add_node(Op::CmpLt, vec![tok_exp, hi]);
    g.add_node(Op::Mul, vec![lt_lo, lt_hi])
}

/// Build transformer forward pass, returns (logits_node, weight_ids)
fn build_transformer(g: &mut Graph, tokens: NodeId, cfg: &ModelConfig) -> (NodeId, Vec<NodeId>) {
    let b = cfg.batch_size;
    let t = cfg.seq_len;
    let v = cfg.vocab_size;
    let d = cfg.n_embd;
    let h = cfg.n_head;
    let hs = cfg.head_size();
    let mlp_h = cfg.mlp_hidden();

    let wte = add_input(g, "wte", &[v, d]);
    let wpe = add_input(g, "wpe", &[t, d]);
    let mut weight_ids = vec![wte, wpe];

    // Embedding
    let one_hot_tok = build_one_hot(g, tokens, v, b, t);
    let tok_emb = build_matmul(g, one_hot_tok, wte);
    let pos_emb_raw = g.add_node(Op::Reshape { shape: dims(&[1, t, d]) }, vec![wpe]);
    let pos_emb = g.add_node(Op::Expand { shape: dims(&[b, t, d]) }, vec![pos_emb_raw]);
    let mut x = g.add_node(Op::Add, vec![tok_emb, pos_emb]);

    // Causal mask
    let arange_t = g.add_node(Op::Arange { size: Dim::Lit(t) }, vec![]);
    let row = g.add_node(Op::Reshape { shape: dims(&[1, 1, t, 1]) }, vec![arange_t]);
    let row_exp = g.add_node(Op::Expand { shape: dims(&[1, 1, t, t]) }, vec![row]);
    let arange_t2 = g.add_node(Op::Arange { size: Dim::Lit(t) }, vec![]);
    let col = g.add_node(Op::Reshape { shape: dims(&[1, 1, 1, t]) }, vec![arange_t2]);
    let col_exp = g.add_node(Op::Expand { shape: dims(&[1, 1, t, t]) }, vec![col]);
    let future = g.add_node(Op::CmpLt, vec![row_exp, col_exp]);
    let neg_big = g.add_node(Op::Constant(-1e6), vec![]);
    let mask = g.add_node(Op::Mul, vec![future, neg_big]);
    let mask_full = g.add_node(Op::Expand { shape: dims(&[b, h, t, t]) }, vec![mask]);

    for layer in 0..cfg.n_layer {
        let ln1_g = add_input(g, &format!("ln1_g_{layer}"), &[d]);
        let ln1_b = add_input(g, &format!("ln1_b_{layer}"), &[d]);
        let qkv_w = add_input(g, &format!("qkv_w_{layer}"), &[d, 3 * d]);
        let qkv_b = add_input(g, &format!("qkv_b_{layer}"), &[3 * d]);
        let proj_w = add_input(g, &format!("proj_w_{layer}"), &[d, d]);
        let proj_b = add_input(g, &format!("proj_b_{layer}"), &[d]);
        let ln2_g = add_input(g, &format!("ln2_g_{layer}"), &[d]);
        let ln2_b = add_input(g, &format!("ln2_b_{layer}"), &[d]);
        let fc_w = add_input(g, &format!("fc_w_{layer}"), &[d, mlp_h]);
        let fc_b = add_input(g, &format!("fc_b_{layer}"), &[mlp_h]);
        let mlp_w = add_input(g, &format!("mlp_w_{layer}"), &[mlp_h, d]);
        let mlp_b = add_input(g, &format!("mlp_b_{layer}"), &[d]);
        weight_ids.extend([ln1_g, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                          ln2_g, ln2_b, fc_w, fc_b, mlp_w, mlp_b]);

        let ln1 = build_layernorm(g, x, ln1_g, ln1_b, d);
        let qkv = build_linear(g, ln1, qkv_w, qkv_b);
        let qkv_r = g.add_node(Op::Reshape { shape: dims(&[b, t, 3, h, hs]) }, vec![qkv]);
        let q_s = g.add_node(Op::Shrink { bounds: vec![
            (Dim::Lit(0), Dim::Lit(b)), (Dim::Lit(0), Dim::Lit(t)),
            (Dim::Lit(0), Dim::Lit(1)), (Dim::Lit(0), Dim::Lit(h)), (Dim::Lit(0), Dim::Lit(hs)),
        ]}, vec![qkv_r]);
        let q = g.add_node(Op::Reshape { shape: dims(&[b, t, h, hs]) }, vec![q_s]);
        let k_s = g.add_node(Op::Shrink { bounds: vec![
            (Dim::Lit(0), Dim::Lit(b)), (Dim::Lit(0), Dim::Lit(t)),
            (Dim::Lit(1), Dim::Lit(2)), (Dim::Lit(0), Dim::Lit(h)), (Dim::Lit(0), Dim::Lit(hs)),
        ]}, vec![qkv_r]);
        let k = g.add_node(Op::Reshape { shape: dims(&[b, t, h, hs]) }, vec![k_s]);
        let v_s = g.add_node(Op::Shrink { bounds: vec![
            (Dim::Lit(0), Dim::Lit(b)), (Dim::Lit(0), Dim::Lit(t)),
            (Dim::Lit(2), Dim::Lit(3)), (Dim::Lit(0), Dim::Lit(h)), (Dim::Lit(0), Dim::Lit(hs)),
        ]}, vec![qkv_r]);
        let v_val = g.add_node(Op::Reshape { shape: dims(&[b, t, h, hs]) }, vec![v_s]);
        let q_h = g.add_node(Op::Permute { order: vec![0, 2, 1, 3] }, vec![q]);
        let k_h = g.add_node(Op::Permute { order: vec![0, 2, 1, 3] }, vec![k]);
        let v_h = g.add_node(Op::Permute { order: vec![0, 2, 1, 3] }, vec![v_val]);
        let kt = g.add_node(Op::Permute { order: vec![0, 1, 3, 2] }, vec![k_h]);
        let scores = build_matmul(g, q_h, kt);
        let inv_hs = g.add_node(Op::Constant(1.0 / (hs as f64).sqrt()), vec![]);
        let scores_scaled = g.add_node(Op::Mul, vec![scores, inv_hs]);
        let scores_masked = g.add_node(Op::Add, vec![scores_scaled, mask_full]);
        let attn = build_softmax(g, scores_masked, 3);
        let attn_out = build_matmul(g, attn, v_h);
        let merged = g.add_node(Op::Permute { order: vec![0, 2, 1, 3] }, vec![attn_out]);
        let merged_flat = g.add_node(Op::Reshape { shape: dims(&[b, t, d]) }, vec![merged]);
        let proj = build_linear(g, merged_flat, proj_w, proj_b);
        x = g.add_node(Op::Add, vec![x, proj]);

        let ln2 = build_layernorm(g, x, ln2_g, ln2_b, d);
        let fc = build_linear(g, ln2, fc_w, fc_b);
        let fc_act = build_gelu(g, fc);
        let mlp_out = build_linear(g, fc_act, mlp_w, mlp_b);
        x = g.add_node(Op::Add, vec![x, mlp_out]);
    }

    let ln_f_g = add_input(g, "ln_f_g", &[d]);
    let ln_f_b = add_input(g, "ln_f_b", &[d]);
    weight_ids.extend([ln_f_g, ln_f_b]);
    let x_norm = build_layernorm(g, x, ln_f_g, ln_f_b, d);
    let wte_t = g.add_node(Op::Permute { order: vec![1, 0] }, vec![wte]);
    let logits = build_matmul(g, x_norm, wte_t);

    (logits, weight_ids)
}

fn build_training_graph(cfg: &ModelConfig) -> TrainingGraph {
    let mut g = Graph::new();
    let b = cfg.batch_size;
    let t = cfg.seq_len;
    let v = cfg.vocab_size;

    let tokens = add_input(&mut g, "tokens", &[b, t]);
    let targets = add_input(&mut g, "targets", &[b, t]);
    let (logits, weight_ids) = build_transformer(&mut g, tokens, cfg);

    // Cross-entropy loss
    let log_probs = build_log_softmax(&mut g, logits, 2);
    let target_one_hot = build_one_hot(&mut g, targets, v, b, t);
    let prod = g.add_node(Op::Mul, vec![target_one_hot, log_probs]);
    let sum_per_token = g.add_node(Op::ReduceSum { axis: 2 }, vec![prod]);
    let sum_batch = g.add_node(Op::ReduceSum { axis: 0 }, vec![sum_per_token]);
    let sum_all = g.add_node(Op::ReduceSum { axis: 0 }, vec![sum_batch]);
    let scalar_sum = g.add_node(Op::Reshape { shape: vec![] }, vec![sum_all]);
    let neg_sum = g.add_node(Op::Neg, vec![scalar_sum]);
    let inv_bt = g.add_node(Op::Constant(1.0 / (b * t) as f64), vec![]);
    let loss = g.add_node(Op::Mul, vec![neg_sum, inv_bt]);

    let grad_ids = g.grad(loss, &weight_ids);
    TrainingGraph { graph: g, loss, grad_ids }
}

fn build_inference_graph(cfg: &ModelConfig) -> Graph {
    let inf_cfg = ModelConfig { batch_size: 1, ..cfg.clone() };
    let mut g = Graph::new();
    let tokens = add_input(&mut g, "tokens", &[1, inf_cfg.seq_len]);
    let (_logits, _weight_ids) = build_transformer(&mut g, tokens, &inf_cfg);
    g
}

// ─── Weight init ───────────────────────────────────────────────────────────

fn weight_sizes(cfg: &ModelConfig) -> Vec<usize> {
    let d = cfg.n_embd;
    let v = cfg.vocab_size;
    let t = cfg.seq_len;
    let mlp_h = cfg.mlp_hidden();
    let mut sizes = vec![v * d, t * d];
    for _ in 0..cfg.n_layer {
        sizes.extend([d, d, d * 3 * d, 3 * d, d * d, d, d, d, d * mlp_h, mlp_h, mlp_h * d, d]);
    }
    sizes.extend([d, d]);
    sizes
}

fn init_weights(cfg: &ModelConfig) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let sizes = weight_sizes(cfg);
    let d = cfg.n_embd;
    sizes.iter().enumerate().map(|(i, &size)| {
        if i <= 1 {
            (0..size).map(|_| rng.r#gen::<f32>() * 0.04 - 0.02).collect()
        } else {
            let layer_offset = (i - 2) % 12;
            let is_ln_gamma = layer_offset == 0 || layer_offset == 6
                || i == sizes.len() - 2;
            let is_ln_beta = layer_offset == 1 || layer_offset == 7
                || i == sizes.len() - 1;
            if is_ln_gamma && size == d {
                vec![1.0f32; size]
            } else if (is_ln_beta && size == d)
                || size == 3 * d || size == d || size == cfg.mlp_hidden() {
                // Could be a bias — check if it's a small 1D tensor
                if size <= cfg.mlp_hidden() && (size == d || size == 3 * d || size == cfg.mlp_hidden()) {
                    // Likely a bias, but we can't perfectly distinguish from here.
                    // Biases that are d-sized and ln_gamma overlap. Use position logic:
                    vec![0.0f32; size]
                } else {
                    let std = 0.02;
                    (0..size).map(|_| rng.r#gen::<f32>() * 2.0 * std - std).collect()
                }
            } else {
                let std = 0.02;
                (0..size).map(|_| rng.r#gen::<f32>() * 2.0 * std - std).collect()
            }
        }
    }).collect()
}

// ─── WASM runner ───────────────────────────────────────────────────────────

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf()
}

fn compile_wasm(as_code: &str) -> std::path::PathBuf {
    let root = project_root();
    let tmp = std::env::temp_dir();
    let src_path = tmp.join("train_model.ts");
    let wasm_path = tmp.join("train_model.wasm");
    std::fs::write(&src_path, as_code).unwrap();
    let output = Command::new("npx")
        .args(["asc", src_path.to_str().unwrap(),
               "--outFile", wasm_path.to_str().unwrap(),
               "--exportRuntime", "--optimize"])
        .current_dir(&root).output().expect("failed to run asc");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("asc compilation failed:\n{stderr}");
    }
    wasm_path
}

fn run_wasm(wasm_path: &std::path::Path, inputs: &[&[f32]]) -> Vec<f32> {
    let root = project_root();
    let tmp = std::env::temp_dir();
    let bin_path = tmp.join("train_inputs.bin");
    let manifest_path = tmp.join("train_manifest.json");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&bin_path).unwrap());
        for arr in inputs {
            for v in *arr { f.write_all(&v.to_le_bytes()).unwrap(); }
        }
    }
    let entries: Vec<String> = inputs.iter()
        .map(|arr| format!("{{\"n_elements\":{}}}", arr.len())).collect();
    let manifest = format!("{{\"inputs\":[{}]}}", entries.join(","));
    std::fs::write(&manifest_path, &manifest).unwrap();
    let runner = root.join("test_runner_bin.mjs");
    let output = Command::new("node")
        .args([runner.to_str().unwrap(), wasm_path.to_str().unwrap(),
               bin_path.to_str().unwrap(), manifest_path.to_str().unwrap()])
        .current_dir(&root).output().expect("failed to run node");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("WASM execution failed:\n{stderr}");
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let s = stdout.trim().trim_start_matches('[').trim_end_matches(']');
    if s.is_empty() { vec![] }
    else { s.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect() }
}

// ─── Adam optimizer ────────────────────────────────────────────────────────

struct Adam {
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    m: Vec<Vec<f32>>, v: Vec<Vec<f32>>, t: usize,
}

impl Adam {
    fn new(sizes: &[usize], lr: f32) -> Self {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            m: sizes.iter().map(|&s| vec![0.0f32; s]).collect(),
            v: sizes.iter().map(|&s| vec![0.0f32; s]).collect(), t: 0 }
    }
    fn step(&mut self, weights: &mut [Vec<f32>], grads: &[Vec<f32>]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, (w, grad)) in weights.iter_mut().zip(grads.iter()).enumerate() {
            for j in 0..w.len() {
                let g = grad[j];
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;
                w[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

// ─── Weight save/load ──────────────────────────────────────────────────────

fn save_model(weights: &[Vec<f32>], cfg: &ModelConfig, vocab: &midi_parse::Vocab) {
    let path = project_root().join("midi_model.bin");
    let mut data: Vec<u8> = Vec::new();
    // Config
    for &v in &[cfg.vocab_size, cfg.seq_len, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.batch_size] {
        data.extend_from_slice(&(v as u32).to_le_bytes());
    }
    // Vocab pitches
    data.extend_from_slice(&(vocab.pitches.len() as u32).to_le_bytes());
    for &p in &vocab.pitches { data.push(p); }
    data.extend_from_slice(&(vocab.max_time_shift as u32).to_le_bytes());
    // Weights
    data.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    for w in weights {
        data.extend_from_slice(&(w.len() as u32).to_le_bytes());
        for &v in w { data.extend_from_slice(&v.to_le_bytes()); }
    }
    std::fs::write(&path, &data).unwrap();
    eprintln!("Saved model to {}", path.display());
}

fn load_model() -> (Vec<Vec<f32>>, ModelConfig, midi_parse::Vocab) {
    let path = project_root().join("midi_model.bin");
    let data = std::fs::read(&path).unwrap_or_else(|_| {
        panic!("No model found at {}. Run train first.", path.display())
    });
    let mut off = 0;
    let mut read_u32 = |o: &mut usize| -> u32 {
        let v = u32::from_le_bytes(data[*o..*o+4].try_into().unwrap());
        *o += 4; v
    };
    let vocab_size = read_u32(&mut off) as usize;
    let seq_len = read_u32(&mut off) as usize;
    let n_embd = read_u32(&mut off) as usize;
    let n_head = read_u32(&mut off) as usize;
    let n_layer = read_u32(&mut off) as usize;
    let batch_size = read_u32(&mut off) as usize;
    let cfg = ModelConfig { vocab_size, seq_len, n_embd, n_head, n_layer, batch_size };

    let n_pitches = read_u32(&mut off) as usize;
    let mut pitches = Vec::new();
    for _ in 0..n_pitches { pitches.push(data[off]); off += 1; }
    let max_time_shift = read_u32(&mut off) as usize;
    let vocab = midi_parse::Vocab { pitches, max_time_shift };

    let n_weights = read_u32(&mut off) as usize;
    let mut weights = Vec::new();
    for _ in 0..n_weights {
        let len = read_u32(&mut off) as usize;
        let w: Vec<f32> = data[off..off + len * 4].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        off += len * 4;
        weights.push(w);
    }
    (weights, cfg, vocab)
}

// ─── Main ──────────────────────────────────────────────────────────────────

const DEFAULT_MIDI: &str = "/Users/jimmyhmiller/Downloads/[Midi] Erik Satie - Gymnopedie No. 1.mid";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "train" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_MIDI);
            cmd_train(path);
        }
        "generate" => cmd_generate(false),
        "random" => cmd_generate(true),
        "parse" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_MIDI);
            cmd_parse(path);
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  train [midi-file]    Train on a MIDI file");
            eprintln!("  generate             Generate from trained weights");
            eprintln!("  random               Generate from random weights");
            eprintln!("  parse [midi-file]    Analyze a MIDI file");
        }
    }
}

fn cmd_parse(path: &str) {
    eprintln!("Parsing: {path}");
    let events = midi_parse::parse_midi_file(path);
    let (tokens, vocab) = midi_parse::tokenize(&events, 120, 32);
    midi_parse::print_summary(&events, &tokens, &vocab);
}

fn cmd_train(midi_path: &str) {
    eprintln!("=== MIDI Trainer ===");
    eprintln!("Source: {midi_path}");

    // Parse MIDI
    let events = midi_parse::parse_midi_file(midi_path);
    let ticks_per_step = 120;
    let max_time_shift = 32;
    let (tokens, vocab) = midi_parse::tokenize(&events, ticks_per_step, max_time_shift);
    midi_parse::print_summary(&events, &tokens, &vocab);

    let cfg = ModelConfig {
        vocab_size: vocab.size(),
        seq_len: 128,
        n_embd: 64,
        n_head: 4,
        n_layer: 2,
        batch_size: 4,
    };
    eprintln!("Model: vocab={} seq={} embd={} heads={} layers={}",
        cfg.vocab_size, cfg.seq_len, cfg.n_embd, cfg.n_head, cfg.n_layer);

    // Create training windows (sliding window over token sequence)
    let mut windows: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let stride = cfg.seq_len / 2;
    let mut pos = 0;
    while pos + cfg.seq_len + 1 <= tokens.len() {
        let input: Vec<f32> = tokens[pos..pos + cfg.seq_len].iter().map(|&t| t as f32).collect();
        let target: Vec<f32> = tokens[pos + 1..pos + cfg.seq_len + 1].iter().map(|&t| t as f32).collect();
        windows.push((input, target));
        pos += stride;
    }
    eprintln!("Training windows: {} (stride={})", windows.len(), stride);

    // Build graph
    eprintln!("Building computation graph...");
    let t0 = Instant::now();
    let tg = build_training_graph(&cfg);
    eprintln!("Graph: {} nodes ({:.1}ms)", tg.graph.nodes.len(), t0.elapsed().as_secs_f64() * 1000.0);

    // Compile
    eprintln!("Compiling to WASM...");
    let t0 = Instant::now();
    let mut output_ids = vec![tg.loss];
    output_ids.extend_from_slice(&tg.grad_ids);
    let backend = AssemblyScriptBackend;
    let as_code = backend.emit_fused_multi_output(&tg.graph, &output_ids);
    eprintln!("Generated {} lines AS ({:.1}ms)", as_code.lines().count(), t0.elapsed().as_secs_f64() * 1000.0);
    let t0 = Instant::now();
    let wasm_path = compile_wasm(&as_code);
    eprintln!("WASM compiled in {:.1}s", t0.elapsed().as_secs_f64());

    // Init
    let sizes = weight_sizes(&cfg);
    let mut weights = init_weights(&cfg);
    let total_params: usize = sizes.iter().sum();
    eprintln!("Total parameters: {total_params}");

    let mut optimizer = Adam::new(&sizes, 3e-4);
    let mut rng = rand::thread_rng();
    let n_epochs = 200;
    let batch_size = cfg.batch_size;

    eprintln!("\nTraining...");
    for epoch in 0..n_epochs {
        let t_epoch = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0;

        // Shuffle
        let mut indices: Vec<usize> = (0..windows.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        for batch_start in (0..windows.len()).step_by(batch_size) {
            if batch_start + batch_size > windows.len() { break; }

            let mut token_data = vec![0.0f32; batch_size * cfg.seq_len];
            let mut target_data = vec![0.0f32; batch_size * cfg.seq_len];
            for bi in 0..batch_size {
                let (ref inp, ref tgt) = windows[indices[batch_start + bi]];
                token_data[bi * cfg.seq_len..(bi + 1) * cfg.seq_len].copy_from_slice(inp);
                target_data[bi * cfg.seq_len..(bi + 1) * cfg.seq_len].copy_from_slice(tgt);
            }

            let mut input_refs: Vec<&[f32]> = vec![&token_data, &target_data];
            for w in &weights { input_refs.push(w); }

            let result = run_wasm(&wasm_path, &input_refs);
            let loss_val = result[0];
            epoch_loss += loss_val;
            n_batches += 1;

            let mut offset = 1;
            let mut grads: Vec<Vec<f32>> = Vec::new();
            for &size in &sizes {
                grads.push(result[offset..offset + size].to_vec());
                offset += size;
            }

            // Gradient clipping
            let mut norm_sq = 0.0f32;
            for grad in &grads { for &g in grad { norm_sq += g * g; } }
            let norm = norm_sq.sqrt();
            if norm > 1.0 {
                let scale = 1.0 / norm;
                for grad in &mut grads { for g in grad.iter_mut() { *g *= scale; } }
            }

            optimizer.step(&mut weights, &grads);
        }

        let avg_loss = epoch_loss / n_batches as f32;
        let epoch_time = t_epoch.elapsed().as_secs_f64();
        eprintln!("Epoch {:3}/{}: loss={:.4}  ({} batches, {:.1}s)",
            epoch + 1, n_epochs, avg_loss, n_batches, epoch_time);
    }

    save_model(&weights, &cfg, &vocab);
    eprintln!("\nTo generate: cargo run --release -p tensor-lang-train -- generate");
}

fn cmd_generate(random: bool) {
    if random {
        eprintln!("=== Generate (random weights) ===");
    } else {
        eprintln!("=== Generate from trained model ===");
    }

    let (weights, cfg, vocab) = if random {
        let events = midi_parse::parse_midi_file(DEFAULT_MIDI);
        let (_, vocab) = midi_parse::tokenize(&events, 120, 32);
        let cfg = ModelConfig {
            vocab_size: vocab.size(), seq_len: 128, n_embd: 64,
            n_head: 4, n_layer: 2, batch_size: 4,
        };
        let weights = init_weights(&cfg);
        (weights, cfg, vocab)
    } else {
        load_model()
    };

    eprintln!("Compiling inference graph...");
    let t0 = Instant::now();
    let backend = AssemblyScriptBackend;
    let inf_graph = build_inference_graph(&cfg);
    let inf_code = backend.emit_fused(&inf_graph);
    let inf_wasm = compile_wasm(&inf_code);
    eprintln!("Compiled in {:.1}s", t0.elapsed().as_secs_f64());

    let mut rng = rand::thread_rng();
    let gen_tokens = 300;
    let temperature = 0.9f32;

    // Seed with first few tokens from the piece
    let seed_len = 4;
    let mut tokens: Vec<usize> = vec![vocab.pad(); seed_len];
    // Start with a note_on for a root note
    if let Some(&first_pitch) = vocab.pitches.first() {
        tokens[0] = vocab.note_on(first_pitch);
    }

    eprintln!("\nGenerating {} tokens...", gen_tokens);
    for step in 0..gen_tokens {
        let ctx_start = if tokens.len() > cfg.seq_len { tokens.len() - cfg.seq_len } else { 0 };
        let ctx = &tokens[ctx_start..];
        let mut token_data = vec![0.0f32; cfg.seq_len];
        for (i, &t) in ctx.iter().enumerate() { token_data[i] = t as f32; }

        let mut input_refs: Vec<&[f32]> = vec![&token_data];
        for w in &weights { input_refs.push(w); }

        let logits = run_wasm(&inf_wasm, &input_refs);
        let pos = ctx.len() - 1;
        let start_idx = pos * cfg.vocab_size;
        let pos_logits = &logits[start_idx..start_idx + cfg.vocab_size];

        // Sample
        let max_logit = pos_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = pos_logits.iter()
            .map(|&l| ((l - max_logit) / temperature).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum).collect();

        let mut r: f32 = rng.gen_range(0.0..1.0);
        let mut next = 0;
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 { next = i; break; }
        }
        // Don't generate PAD
        if next == 0 { next = 1; }

        if step < 60 {
            eprint!("{} ", vocab.token_name(next));
            if step % 20 == 19 { eprintln!(); }
        }
        tokens.push(next);
    }
    eprintln!("\n");

    // Decode tokens to MIDI
    let midi_path = project_root().join("generated.mid");
    tokens_to_midi(&midi_path, &tokens[seed_len..], &vocab);
    eprintln!("Wrote MIDI to: {}", midi_path.display());
    eprintln!("Play it:  open {}", midi_path.display());
}

// ─── Token sequence → MIDI file ───────────────────────────────────────────

fn tokens_to_midi(path: &std::path::Path, tokens: &[usize], vocab: &midi_parse::Vocab) {
    let ticks_per_beat: u16 = 480;
    let ticks_per_step: u32 = 120; // must match tokenization
    let mut data: Vec<u8> = Vec::new();

    // MThd
    data.extend_from_slice(b"MThd");
    data.extend_from_slice(&6u32.to_be_bytes());
    data.extend_from_slice(&0u16.to_be_bytes()); // format 0
    data.extend_from_slice(&1u16.to_be_bytes()); // 1 track
    data.extend_from_slice(&ticks_per_beat.to_be_bytes());

    let mut track: Vec<u8> = Vec::new();
    // Tempo: 72 BPM (Gymnopedie is slow)
    track.push(0x00);
    track.extend_from_slice(&[0xFF, 0x51, 0x03]);
    let tempo: u32 = 833_333; // 60_000_000 / 72
    track.push((tempo >> 16) as u8);
    track.push((tempo >> 8) as u8);
    track.push(tempo as u8);

    // Program change: acoustic grand piano
    track.push(0x00);
    track.push(0xC0);
    track.push(0);

    let mut pending_delta: u32 = 0;

    for &token in tokens {
        if let Some(pitch) = vocab.is_note_on(token) {
            write_vlq(&mut track, pending_delta);
            pending_delta = 0;
            track.push(0x90);
            track.push(pitch);
            track.push(80); // velocity
        } else if let Some(pitch) = vocab.is_note_off(token) {
            write_vlq(&mut track, pending_delta);
            pending_delta = 0;
            track.push(0x80);
            track.push(pitch);
            track.push(0);
        } else if let Some(steps) = vocab.is_time_shift(token) {
            pending_delta += steps as u32 * ticks_per_step;
        }
        // Skip PAD tokens
    }

    // End of track
    track.push(0x00);
    track.extend_from_slice(&[0xFF, 0x2F, 0x00]);

    data.extend_from_slice(b"MTrk");
    data.extend_from_slice(&(track.len() as u32).to_be_bytes());
    data.extend_from_slice(&track);

    std::fs::write(path, &data).unwrap();
}

fn write_vlq(buf: &mut Vec<u8>, mut value: u32) {
    let mut bytes = Vec::new();
    bytes.push((value & 0x7F) as u8);
    value >>= 7;
    while value > 0 {
        bytes.push((value & 0x7F) as u8 | 0x80);
        value >>= 7;
    }
    bytes.reverse();
    buf.extend_from_slice(&bytes);
}
