use std::collections::HashMap;
use rand::Rng;
use tensor_lang_graph::{compile_with_env, Graph, NodeId, Op};
use tensor_lang_gpu::plan::{self, GpuPlan};
use tensor_lang_gpu::runtime::GpuRuntime;

// ─── Simple tensor container (replaces ndarray) ─────────────────────────────

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>().max(1);
        Tensor { shape, data: vec![0.0; len] }
    }

    pub fn from_shape_vec(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Tensor { shape, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }
}

// ─── Chord grammar vocabulary ────────────���────────────────────────────────

pub const TOKENS: &[&str] = &[
    "<pad>", "<bos>", "<eos>",
    "I", "ii", "iii", "IV", "V", "vi",
    "V/X", "ii/X", "bII7/X",
    "V7", "viio", "bVI", "bVII",
];

pub const VOCAB_SIZE: usize = 16;

pub fn tok(name: &str) -> usize {
    TOKENS.iter().position(|&t| t == name).unwrap()
}

pub fn token_name(id: usize) -> &'static str {
    TOKENS[id]
}

// ─── Grammar data generation ──────────────────────────────────────────────

pub fn generate_sequence(rng: &mut impl Rng) -> Vec<usize> {
    let mut seq = vec![tok("<bos>")];
    let base: Vec<&str> = match rng.gen_range(0..10) {
        0 => vec!["I"],
        1..=2 => vec!["ii", "V", "I"],
        3 => vec!["IV", "I"],
        4..=5 => vec!["vi", "ii", "V", "I"],
        6 => vec!["ii", "V", "vi"],
        7 => vec!["IV", "V", "vi"],
        8 => vec!["bVI", "bVII", "I"],
        _ => vec!["IV", "I"],
    };

    for chord in &base {
        if rng.gen_bool(0.25) && *chord != "I" && *chord != "bVI" && *chord != "bVII" {
            expand_x(&mut seq, rng);
        }
        if *chord == "V" {
            match rng.gen_range(0..5) {
                0 => seq.push(tok("V7")),
                1 => seq.push(tok("viio")),
                _ => seq.push(tok("V")),
            }
        } else {
            seq.push(tok(chord));
        }
    }

    if rng.gen_bool(0.2) {
        seq.push(tok("IV"));
        seq.push(tok("I"));
    }

    seq.push(tok("<eos>"));
    seq
}

fn expand_x(seq: &mut Vec<usize>, rng: &mut impl Rng) {
    match rng.gen_range(0..3) {
        0 => {
            if rng.gen_bool(0.2) { seq.push(tok("bII7/X")); }
            else { seq.push(tok("V/X")); }
        }
        1 => {
            seq.push(tok("ii/X"));
            if rng.gen_bool(0.2) { seq.push(tok("bII7/X")); }
            else { seq.push(tok("V/X")); }
        }
        _ => {
            seq.push(tok("V/X"));
            if rng.gen_bool(0.2) { seq.push(tok("bII7/X")); }
            else { seq.push(tok("V/X")); }
        }
    }
}

// ─── Model config ──────────────���──────────────���───────────────────────────

#[derive(Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub batch_size: usize,
}

impl ModelConfig {
    pub fn head_size(&self) -> usize { self.n_embd / self.n_head }
    pub fn mlp_hidden(&self) -> usize { 4 * self.n_embd }
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            vocab_size: VOCAB_SIZE,
            seq_len: 16,
            n_embd: 32,
            n_head: 2,
            n_layer: 1,
            batch_size: 1,
        }
    }
}

// ─── Key activation indices ────────────���──────────────────────────────────

pub struct ActivationMap {
    pub token_embeddings: usize,
    pub attn_weights: usize,
    pub logits: usize,
    pub attn_output: usize,
}

// ──�� Model runner ────────────────────��────────────────────────────────────

pub struct ModelRunner {
    pub inf_graph: Graph,
    pub train_graph: Graph,
    pub train_loss: NodeId,
    pub train_grad_ids: Vec<NodeId>,
    pub train_weight_ids: Vec<NodeId>,
    pub inf_map: ActivationMap,
    pub cfg: ModelConfig,
    pub weights: Vec<Vec<f32>>,
    // Boundary indices for graph visualization
    pub inf_logits_idx: usize,       // last forward node in inference graph
    pub train_logits_idx: usize,     // last model-forward node in training graph
    pub train_loss_idx: usize,       // loss node index (boundary: everything after = backprop)
    // GPU execution
    gpu_rt: GpuRuntime,
    inf_plan: GpuPlan,
    inf_all_output_ids: Vec<NodeId>,    // which nodes are materialized in inference
    inf_output_sizes: Vec<usize>,       // size of each output
    train_plan: GpuPlan,
    train_output_sizes: Vec<usize>,
}

fn model_source() -> String {
    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("tensor-lang-train")
        .join("model_tiny.tensor");
    std::fs::read_to_string(&model_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", model_path.display()))
}

fn model_env(cfg: &ModelConfig) -> (HashMap<String, usize>, HashMap<String, f64>) {
    let dims: HashMap<String, usize> = [
        ("B", cfg.batch_size), ("T", cfg.seq_len), ("V", cfg.vocab_size),
        ("D", cfg.n_embd), ("H", cfg.n_head), ("S", cfg.head_size()),
        ("M", cfg.mlp_hidden()),
    ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();

    let constants: HashMap<String, f64> = [
        ("INV_D", 1.0 / cfg.n_embd as f64),
        ("INV_SQRT_S", 1.0 / (cfg.head_size() as f64).sqrt()),
        ("NEG_EIGHT_OVER_H", -8.0 / cfg.n_head as f64),
    ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();

    (dims, constants)
}

fn weight_sizes(cfg: &ModelConfig) -> Vec<usize> {
    let d = cfg.n_embd;
    let v = cfg.vocab_size;
    let mlp_h = cfg.mlp_hidden();
    let mut sizes = vec![v * d]; // wte
    for _ in 0..cfg.n_layer {
        sizes.extend([d, d, d * 3 * d, 3 * d, d * d, d, d, d, d * mlp_h, mlp_h, mlp_h * d, d]);
    }
    sizes.extend([d, d]); // ln_f_g, ln_f_b
    sizes
}

fn init_weights(cfg: &ModelConfig) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let sizes = weight_sizes(cfg);
    let d = cfg.n_embd;
    sizes.iter().enumerate().map(|(i, &size)| {
        if i == 0 {
            (0..size).map(|_| rng.r#gen::<f32>() * 0.04 - 0.02).collect()
        } else {
            let layer_offset = (i - 1) % 12;
            let is_gamma = (layer_offset == 0 || layer_offset == 6) && size == d
                || i == sizes.len() - 2;
            if is_gamma {
                vec![1.0f32; size]
            } else if size == d || size == 3 * d || size == cfg.mlp_hidden() {
                vec![0.0f32; size]
            } else {
                let std = 0.02;
                (0..size).map(|_| rng.r#gen::<f32>() * 2.0 * std - std).collect()
            }
        }
    }).collect()
}

fn find_activation_map(graph: &Graph, cfg: &ModelConfig) -> ActivationMap {
    let h = cfg.n_head;
    let t = cfg.seq_len;
    let d = cfg.n_embd;
    let _v = cfg.vocab_size;

    let logits = graph.nodes.len() - 1;

    // Find attention weights: shape [1, H, T, T] that is result of Mul (softmax output)
    let mut attn_weights = 0;
    let mut token_embeddings = 0;
    let mut attn_output = 0;

    for (i, node) in graph.nodes.iter().enumerate() {
        let shape: Vec<usize> = node.shape.iter()
            .filter_map(|d| d.as_usize())
            .collect();

        // Attention weights: [1, H, T, T] from a Mul (recip * exp pattern in softmax)
        if shape == vec![1, h, t, t] {
            if let Op::Mul = &node.op {
                attn_weights = i;
            }
        }

        // Token embeddings: first [1, T, D] node that comes from a MatMul (one_hot @ wte)
        if shape == vec![1, t, d] && token_embeddings == 0 {
            if let Op::Reshape { .. } = &node.op {
                // skip reshapes
            } else if matches!(&node.op, Op::Add | Op::Mul) {
                // these come later (layernorm etc)
            } else {
                token_embeddings = i;
            }
        }
    }

    // Find the residual after attention: it's the Add [1,T,D] right after attn projection
    // Walk forward from attn_weights to find the next Add with shape [1, T, D]
    for i in attn_weights..graph.nodes.len() {
        let shape: Vec<usize> = graph.nodes[i].shape.iter()
            .filter_map(|d| d.as_usize())
            .collect();
        if shape == vec![1, t, d] {
            if let Op::Add = &graph.nodes[i].op {
                attn_output = i;
                break;
            }
        }
    }

    ActivationMap {
        token_embeddings,
        attn_weights,
        logits,
        attn_output,
    }
}

// Returns (graph, loss_id, grad_ids, weight_ids, logits_idx, loss_idx)
fn compile_for_training(cfg: &ModelConfig) -> (Graph, NodeId, Vec<NodeId>, Vec<NodeId>, usize, usize) {
    let source = model_source();
    let (dims, constants) = model_env(cfg);
    let mut graph = compile_with_env(&source, &dims, &constants);

    let b = cfg.batch_size;
    let t = cfg.seq_len;
    let v = cfg.vocab_size;

    let logits = NodeId(graph.nodes.len() - 1);

    let weight_ids: Vec<NodeId> = graph.nodes.iter().enumerate()
        .filter_map(|(i, n)| {
            if let Op::Input { name } = &n.op {
                if name != "input_0" { Some(NodeId(i)) } else { None }
            } else { None }
        })
        .collect();

    let targets = graph.add_node(Op::Input { name: "targets".into() }, vec![]);
    graph.set_input_shape(targets, tensor_lang_graph::dims(&[b, t]));

    // One-hot targets
    let classes = graph.add_node(Op::Arange { size: tensor_lang_graph::Dim::Lit(v) }, vec![]);
    let cls = graph.add_node(Op::Reshape { shape: tensor_lang_graph::dims(&[1, 1, v]) }, vec![classes]);
    let cls_exp = graph.add_node(Op::Expand { shape: tensor_lang_graph::dims(&[b, t, v]) }, vec![cls]);
    let tok_r = graph.add_node(Op::Reshape { shape: tensor_lang_graph::dims(&[b, t, 1]) }, vec![targets]);
    let tok_exp = graph.add_node(Op::Expand { shape: tensor_lang_graph::dims(&[b, t, v]) }, vec![tok_r]);
    let half = graph.add_node(Op::Constant(0.5), vec![]);
    let neg_half = graph.add_node(Op::Neg, vec![half]);
    let lo = graph.add_node(Op::Add, vec![cls_exp, neg_half]);
    let half2 = graph.add_node(Op::Constant(0.5), vec![]);
    let hi = graph.add_node(Op::Add, vec![cls_exp, half2]);
    let lt_lo = graph.add_node(Op::CmpLt, vec![lo, tok_exp]);
    let lt_hi = graph.add_node(Op::CmpLt, vec![tok_exp, hi]);
    let target_one_hot = graph.add_node(Op::Mul, vec![lt_lo, lt_hi]);

    // Log-softmax
    let mx = graph.add_node(Op::ReduceMax { axis: 2 }, vec![logits]);
    let neg_mx = graph.add_node(Op::Neg, vec![mx]);
    let shifted = graph.add_node(Op::Add, vec![logits, neg_mx]);
    let log2e = graph.add_node(Op::Constant(std::f64::consts::LOG2_E), vec![]);
    let scaled = graph.add_node(Op::Mul, vec![shifted, log2e]);
    let ex = graph.add_node(Op::Exp2, vec![scaled]);
    let sum_ex = graph.add_node(Op::ReduceSum { axis: 2 }, vec![ex]);
    let log2_sum = graph.add_node(Op::Log2, vec![sum_ex]);
    let ln2 = graph.add_node(Op::Constant(std::f64::consts::LN_2), vec![]);
    let log_sum = graph.add_node(Op::Mul, vec![log2_sum, ln2]);
    let neg_log_sum = graph.add_node(Op::Neg, vec![log_sum]);
    let log_probs = graph.add_node(Op::Add, vec![shifted, neg_log_sum]);

    // Cross-entropy loss — reduce all dims: [B, T, V] → scalar
    let prod = graph.add_node(Op::Mul, vec![target_one_hot, log_probs]);
    let sum_v = graph.add_node(Op::ReduceSum { axis: 2 }, vec![prod]);    // [B, T, 1]
    let sum_t = graph.add_node(Op::ReduceSum { axis: 1 }, vec![sum_v]);   // [B, 1, 1]
    let sum_b = graph.add_node(Op::ReduceSum { axis: 0 }, vec![sum_t]);   // [1, 1, 1]
    let scalar = graph.add_node(Op::Reshape { shape: vec![] }, vec![sum_b]);
    let neg = graph.add_node(Op::Neg, vec![scalar]);
    let inv_bt = graph.add_node(Op::Constant(1.0 / (b * t) as f64), vec![]);
    let loss = graph.add_node(Op::Mul, vec![neg, inv_bt]);

    let logits_idx = logits.0;
    let loss_idx = loss.0;

    let grad_ids = graph.grad(loss, &weight_ids);

    (graph, loss, grad_ids, weight_ids, logits_idx, loss_idx)
}

pub fn init_weights_pub(cfg: &ModelConfig) -> Vec<Vec<f32>> {
    init_weights(cfg)
}

/// Compute output sizes for each node in a graph.
fn node_sizes(graph: &Graph) -> Vec<usize> {
    graph.nodes.iter().map(|n| {
        n.shape.iter()
            .map(|d| d.as_usize().expect("symbolic dim in viz"))
            .product::<usize>()
            .max(1) // scalars are 1 element
    }).collect()
}

/// Compile inference WASM that outputs ALL non-input nodes.
fn compile_inf_plan(graph: &Graph) -> (GpuPlan, Vec<NodeId>, Vec<usize>) {
    // Output every non-input node so the graph explorer can inspect all values
    let output_ids: Vec<NodeId> = (0..graph.nodes.len())
        .filter(|&i| !matches!(&graph.nodes[i].op, Op::Input { .. }))
        .map(NodeId)
        .collect();
    let output_sizes: Vec<usize> = output_ids.iter()
        .map(|id| {
            graph.nodes[id.0].shape.iter()
                .map(|d| d.as_usize().unwrap())
                .product::<usize>()
                .max(1)
        })
        .collect();

    let gpu_plan = plan::build_plan_multi_output(graph, &output_ids);
    (gpu_plan, output_ids, output_sizes)
}

/// Compile training GPU plan that outputs loss + gradients.
fn compile_train_plan(
    graph: &Graph,
    loss: NodeId,
    grad_ids: &[NodeId],
) -> (GpuPlan, Vec<usize>) {
    let mut output_ids = vec![loss];
    output_ids.extend_from_slice(grad_ids);
    let output_sizes: Vec<usize> = output_ids.iter()
        .map(|id| {
            graph.nodes[id.0].shape.iter()
                .map(|d| d.as_usize().unwrap())
                .product::<usize>()
                .max(1)
        })
        .collect();

    let gpu_plan = plan::build_plan_multi_output(graph, &output_ids);
    (gpu_plan, output_sizes)
}

impl ModelRunner {
    pub fn new() -> Self {
        let cfg = ModelConfig::default();
        Self::build(cfg, init_weights(&ModelConfig::default()))
    }

    fn build(cfg: ModelConfig, weights: Vec<Vec<f32>>) -> Self {
        // Compile inference graph
        let source = model_source();
        let (dims, constants) = model_env(&cfg);
        let inf_graph = compile_with_env(&source, &dims, &constants);
        let inf_map = find_activation_map(&inf_graph, &cfg);
        let inf_logits_idx = inf_graph.nodes.len() - 1;

        // Compile inference GPU plan (outputs all nodes)
        let (inf_plan, inf_all_output_ids, inf_output_sizes) =
            compile_inf_plan(&inf_graph);

        // Compile training graph
        let (train_graph, train_loss, train_grad_ids, train_weight_ids, train_logits_idx, train_loss_idx) =
            compile_for_training(&cfg);

        // Compile training GPU plan (outputs loss + grads)
        let (train_plan, train_output_sizes) =
            compile_train_plan(&train_graph, train_loss, &train_grad_ids);

        // Initialize GPU runtime
        let gpu_rt = GpuRuntime::new();

        ModelRunner {
            inf_graph,
            train_graph,
            train_loss,
            train_grad_ids,
            train_weight_ids,
            inf_map,
            inf_logits_idx,
            train_logits_idx,
            train_loss_idx,
            cfg,
            weights,
            gpu_rt,
            inf_plan,
            inf_all_output_ids,
            inf_output_sizes,
            train_plan,
            train_output_sizes,
        }
    }

    /// Build flat input arrays in graph order.
    fn collect_inputs(&self, graph: &Graph, tokens: &[f32], targets: Option<&[f32]>) -> Vec<Vec<f32>> {
        let mut inputs = Vec::new();
        for node in &graph.nodes {
            if let Op::Input { name } = &node.op {
                if name == "input_0" {
                    inputs.push(tokens.to_vec());
                } else if name == "targets" {
                    inputs.push(targets.unwrap_or(&[]).to_vec());
                } else if name.starts_with("input_") {
                    let num: usize = name[6..].parse().unwrap();
                    let weight_idx = num - 1;
                    inputs.push(self.weights[weight_idx].clone());
                }
            }
        }
        inputs
    }

    pub fn forward(&self, tokens: &[f32]) -> Vec<Tensor> {
        let flat_inputs = self.collect_inputs(&self.inf_graph, tokens, None);
        let input_refs: Vec<&[f32]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

        let total_output: usize = self.inf_output_sizes.iter().sum();
        let result = self.gpu_rt.run(&self.inf_plan, &input_refs, total_output);

        // Split concatenated output into per-node Tensor values.
        let n_nodes = self.inf_graph.nodes.len();
        let mut all_values: Vec<Tensor> = Vec::with_capacity(n_nodes);

        let mut output_map: HashMap<usize, usize> = HashMap::new();
        for (oi, id) in self.inf_all_output_ids.iter().enumerate() {
            output_map.insert(id.0, oi);
        }

        let mut offset = 0usize;
        let mut offsets = Vec::with_capacity(self.inf_output_sizes.len());
        for &sz in &self.inf_output_sizes {
            offsets.push(offset);
            offset += sz;
        }

        for i in 0..n_nodes {
            let shape: Vec<usize> = self.inf_graph.nodes[i].shape.iter()
                .map(|d| d.as_usize().unwrap_or(1))
                .collect();
            if let Some(&oi) = output_map.get(&i) {
                let start = offsets[oi];
                let sz = self.inf_output_sizes[oi];
                let data = result[start..start + sz].to_vec();
                all_values.push(Tensor::from_shape_vec(shape, data));
            } else {
                all_values.push(Tensor::zeros(shape));
            }
        }

        all_values
    }

    pub fn train_step(&mut self, tokens: &[f32], targets: &[f32]) -> (f32, Vec<Vec<f32>>) {
        let flat_inputs = self.collect_inputs(&self.train_graph, tokens, Some(targets));
        let input_refs: Vec<&[f32]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

        let total_output: usize = self.train_output_sizes.iter().sum();
        let result = self.gpu_rt.run(&self.train_plan, &input_refs, total_output);

        // First output is loss (1 element), rest are gradients
        let loss = result[0];

        let mut off = self.train_output_sizes[0]; // skip loss
        let grads: Vec<Vec<f32>> = self.train_output_sizes[1..].iter()
            .map(|&sz| {
                let g = result[off..off + sz].to_vec();
                off += sz;
                g
            })
            .collect();

        (loss, grads)
    }

    pub fn from_pretrained(path: &std::path::Path) -> Self {
        let data = std::fs::read(path).unwrap_or_else(|e| {
            panic!("Cannot read {}: {e}", path.display());
        });
        let mut off = 0;
        let r32 = |o: &mut usize| -> u32 {
            let v = u32::from_le_bytes(data[*o..*o+4].try_into().unwrap());
            *o += 4; v
        };
        let cfg = ModelConfig {
            vocab_size: r32(&mut off) as usize,
            seq_len: r32(&mut off) as usize,
            n_embd: r32(&mut off) as usize,
            n_head: r32(&mut off) as usize,
            n_layer: r32(&mut off) as usize,
            batch_size: 1,
        };
        let _saved_batch = r32(&mut off);

        let n_w = r32(&mut off) as usize;
        let mut weights = Vec::new();
        for _ in 0..n_w {
            let len = r32(&mut off) as usize;
            let mut w = Vec::with_capacity(len);
            for _ in 0..len {
                let v = f32::from_le_bytes(data[off..off+4].try_into().unwrap());
                off += 4;
                w.push(v);
            }
            weights.push(w);
        }
        eprintln!("Loaded {} weight tensors from {} (n_embd={}, n_head={}, n_layer={})",
            n_w, path.display(), cfg.n_embd, cfg.n_head, cfg.n_layer);

        Self::build(cfg, weights)
    }

    pub fn pad_sequence(seq: &[usize], seq_len: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len];
        for (i, &t) in seq.iter().enumerate().take(seq_len) {
            out[i] = t as f32;
        }
        out
    }
}

// ─── Node description for graph view ──────────────────────────────────────

pub fn describe_node(graph: &Graph, idx: usize) -> String {
    let node = &graph.nodes[idx];
    let op_str = match &node.op {
        Op::Input { name } => format!("Input({})", name),
        Op::Constant(v) => {
            if *v == (*v as i64) as f64 { format!("Const({})", *v as i64) }
            else { format!("Const({:.4})", v) }
        }
        Op::Arange { size } => format!("Arange({})", size.as_usize().unwrap_or(0)),
        Op::Neg => "Neg".into(),
        Op::Recip => "Recip".into(),
        Op::Exp2 => "Exp2".into(),
        Op::Log2 => "Log2".into(),
        Op::Sqrt => "Sqrt".into(),
        Op::Add => "Add".into(),
        Op::Mul => "Mul".into(),
        Op::Max => "Max".into(),
        Op::CmpLt => "CmpLt".into(),
        Op::ReduceSum { axis } => format!("Sum(axis={})", axis),
        Op::ReduceMax { axis } => format!("Max(axis={})", axis),
        Op::Reshape { shape } => {
            let dims: Vec<String> = shape.iter().map(|d| d.as_usize().map(|v| v.to_string()).unwrap_or("?".into())).collect();
            format!("Reshape({})", dims.join(","))
        }
        Op::Permute { order } => {
            let o: Vec<String> = order.iter().map(|x| x.to_string()).collect();
            format!("Permute({})", o.join(","))
        }
        Op::Expand { shape } => {
            let dims: Vec<String> = shape.iter().map(|d| d.as_usize().map(|v| v.to_string()).unwrap_or("?".into())).collect();
            format!("Expand({})", dims.join(","))
        }
        Op::Pad { .. } => "Pad".into(),
        Op::Shrink { .. } => "Shrink".into(),
    };
    let shape: Vec<String> = node.shape.iter()
        .map(|d| d.as_usize().map(|v| v.to_string()).unwrap_or("?".into()))
        .collect();
    let inputs: Vec<String> = node.inputs.iter().map(|id| id.0.to_string()).collect();
    if inputs.is_empty() {
        format!("[{}] {} -> [{}]", idx, op_str, shape.join(","))
    } else {
        format!("[{}] {}({}) -> [{}]", idx, op_str, inputs.join(","), shape.join(","))
    }
}

// ─── Adam optimizer ───────────────────────────────────────────────────────

pub struct Adam {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(sizes: &[usize], lr: f32) -> Self {
        Adam {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            m: sizes.iter().map(|&s| vec![0.0f32; s]).collect(),
            v: sizes.iter().map(|&s| vec![0.0f32; s]).collect(),
            t: 0,
        }
    }

    pub fn step(&mut self, weights: &mut [Vec<f32>], grads: &[Vec<f32>]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, (w, grad)) in weights.iter_mut().zip(grads.iter()).enumerate() {
            for j in 0..w.len() {
                let g = grad[j];
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                w[j] -= self.lr * (self.m[i][j] / bc1) / ((self.v[i][j] / bc2).sqrt() + self.eps);
            }
        }
    }
}
