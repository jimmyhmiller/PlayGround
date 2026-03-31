use std::collections::HashMap;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, Track, TrackEvent, TrackEventKind};
use midly::num::{u4, u7, u15, u24, u28};
use rand::Rng;
use tensor_lang_backend::assemblyscript::AssemblyScriptBackend;
use tensor_lang_graph::{compile_with_env, Graph, NodeId, Op};

// ─── Chord grammar vocabulary ─────────────────────────────────────────────

const TOKENS: &[&str] = &[
    "<pad>", "<bos>", "<eos>",
    "I", "ii", "iii", "IV", "V", "vi",
    "V/X", "ii/X", "bII7/X",
    "V7", "viio", "bVI", "bVII",
];

fn tok(name: &str) -> usize {
    TOKENS.iter().position(|&t| t == name).unwrap()
}

fn token_name(id: usize) -> &'static str {
    TOKENS[id]
}

const VOCAB_SIZE: usize = 16; // must match TOKENS.len()

// ─── Grammar-based data generation ────────────────────────────────────────
//
// Base progressions (S rules):
//   S -> I
//   S -> ii V I
//   S -> IV I
//   S -> vi ii V I
//
// New rules:
//   S -> ii V vi          (deceptive cadence)
//   S -> IV V vi          (deceptive cadence)
//   S -> S IV I           (plagal tag — append to any progression)
//   S -> bVI bVII I       (Mario cadence)
//
// V substitution: V -> V7 | viio (anywhere V appears)
// Secondary dominants: V/X, ii/X V/X, bII7/X before any non-I diatonic chord
//
// ─────────────────────────────────────────────────────────────────────────

fn generate_sequence(rng: &mut impl Rng) -> Vec<usize> {
    let mut seq = vec![tok("<bos>")];

    // Pick a base progression (weighted: common ones more likely)
    let base: Vec<&str> = match rng.gen_range(0..10) {
        0 => vec!["I"],
        1..=2 => vec!["ii", "V", "I"],
        3 => vec!["IV", "I"],
        4..=5 => vec!["vi", "ii", "V", "I"],
        6 => vec!["ii", "V", "vi"],          // deceptive
        7 => vec!["IV", "V", "vi"],           // deceptive
        8 => vec!["bVI", "bVII", "I"],        // Mario cadence
        _ => vec!["IV", "I"],
    };

    // For each chord in the base, optionally prepend secondary dominant chain
    for chord in &base {
        // 25% chance of secondary dominant approach (not before I or bVI/bVII)
        if rng.gen_bool(0.25) && *chord != "I" && *chord != "bVI" && *chord != "bVII" {
            expand_x(&mut seq, rng);
        }
        // V substitution: V -> V7 or viio
        if *chord == "V" {
            match rng.gen_range(0..5) {
                0 => seq.push(tok("V7")),     // 20% V7
                1 => seq.push(tok("viio")),   // 20% viio
                _ => seq.push(tok("V")),      // 60% plain V
            }
        } else {
            seq.push(tok(chord));
        }
    }

    // Optional plagal tag: 20% chance to append IV I
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
            // V/X (or bII7/X substitution)
            if rng.gen_bool(0.2) {
                seq.push(tok("bII7/X"));
            } else {
                seq.push(tok("V/X"));
            }
        }
        1 => {
            // ii/X V/X
            seq.push(tok("ii/X"));
            if rng.gen_bool(0.2) {
                seq.push(tok("bII7/X"));
            } else {
                seq.push(tok("V/X"));
            }
        }
        _ => {
            // Chain: V/X V/X
            seq.push(tok("V/X"));
            if rng.gen_bool(0.2) {
                seq.push(tok("bII7/X"));
            } else {
                seq.push(tok("V/X"));
            }
        }
    }
}

fn generate_dataset(n: usize) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| generate_sequence(&mut rng)).collect()
}

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

// ─── DSL compilation ──────────────────────────────────────────────────────

fn model_source() -> String {
    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("model_tiny.tensor");
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

struct TrainingGraph {
    graph: Graph,
    loss: NodeId,
    grad_ids: Vec<NodeId>,
}

fn compile_for_training(cfg: &ModelConfig) -> TrainingGraph {
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

    let prod = graph.add_node(Op::Mul, vec![target_one_hot, log_probs]);
    let sum_v = graph.add_node(Op::ReduceSum { axis: 2 }, vec![prod]);
    let sum_b = graph.add_node(Op::ReduceSum { axis: 0 }, vec![sum_v]);
    let sum_all = graph.add_node(Op::ReduceSum { axis: 0 }, vec![sum_b]);
    let scalar = graph.add_node(Op::Reshape { shape: vec![] }, vec![sum_all]);
    let neg = graph.add_node(Op::Neg, vec![scalar]);
    let inv_bt = graph.add_node(Op::Constant(1.0 / (b * t) as f64), vec![]);
    let loss = graph.add_node(Op::Mul, vec![neg, inv_bt]);

    let grad_ids = graph.grad(loss, &weight_ids);

    TrainingGraph { graph, loss, grad_ids }
}

fn compile_for_inference(cfg: &ModelConfig) -> Graph {
    let inf_cfg = ModelConfig { batch_size: 1, ..cfg.clone() };
    let source = model_source();
    let (dims, constants) = model_env(&inf_cfg);
    compile_with_env(&source, &dims, &constants)
}

// ─── Weight init ───────────────────────────────────────────────────────────

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

// ─── WASM runner ───────────────────────────────────────────────────────────

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf()
}

fn compile_wasm_named(as_code: &str, name: &str) -> std::path::PathBuf {
    let root = project_root();
    let tmp = std::env::temp_dir();
    let src_path = tmp.join(format!("{name}.ts"));
    let wasm_path = tmp.join(format!("{name}.wasm"));
    std::fs::write(&src_path, as_code).unwrap();
    let output = Command::new("npx")
        .args(["asc", src_path.to_str().unwrap(),
               "--outFile", wasm_path.to_str().unwrap(),
               "--exportRuntime", "--optimize",
               "--initialMemory", "256",
               "--maximumMemory", "4096"])
        .current_dir(&root).output().expect("failed to run asc");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("asc compilation failed:\n{stderr}");
    }
    wasm_path
}

struct WasmRunner {
    child: std::process::Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    stdout: std::process::ChildStdout,
}

impl WasmRunner {
    fn new(wasm_path: &std::path::Path) -> Self {
        let root = project_root();
        let runner = root.join("persistent_runner.mjs");
        let mut child = Command::new("node")
            .args([runner.to_str().unwrap(), wasm_path.to_str().unwrap()])
            .current_dir(&root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("failed to spawn node");

        let stdin = std::io::BufWriter::new(child.stdin.take().unwrap());
        let stdout = child.stdout.take().unwrap();
        WasmRunner { child, stdin, stdout }
    }

    fn run(&mut self, inputs: &[&[f32]]) -> Vec<f32> {
        use std::io::Read;

        let n = inputs.len() as u32;
        self.stdin.write_all(&n.to_le_bytes()).unwrap();
        for arr in inputs {
            let size = arr.len() as u32;
            self.stdin.write_all(&size.to_le_bytes()).unwrap();
            for &v in *arr {
                self.stdin.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        self.stdin.flush().unwrap();

        let mut header = [0u8; 4];
        self.stdout.read_exact(&mut header).expect("failed to read output header");
        let n_outputs = u32::from_le_bytes(header) as usize;

        let mut data = vec![0u8; n_outputs * 4];
        self.stdout.read_exact(&mut data).expect("failed to read output data");

        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl Drop for WasmRunner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
                w[j] -= self.lr * (self.m[i][j] / bc1) / ((self.v[i][j] / bc2).sqrt() + self.eps);
            }
        }
    }
}

// ─── Pad sequences to fixed length ────────────────────────────────────────

fn pad_sequence(seq: &[usize], seq_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len];
    for (i, &t) in seq.iter().enumerate().take(seq_len) {
        out[i] = t as f32;
    }
    out
}

// ─── Grammar validation ───────────────────────────────────────────────────

fn is_valid_sequence(tokens: &[usize]) -> bool {
    let core: Vec<usize> = tokens.iter()
        .copied()
        .filter(|&t| t != tok("<pad>") && t != tok("<bos>") && t != tok("<eos>"))
        .collect();

    if core.is_empty() { return false; }

    // Strip secondary dominant prefixes and normalize V-equivalents,
    // then check if base chords match a valid pattern (with optional plagal tag)
    let mut pos = 0;
    let mut base_chords = Vec::new();

    while pos < core.len() {
        // Skip secondary dominant chain
        while pos < core.len() && is_secondary(core[pos]) {
            pos += 1;
        }
        if pos >= core.len() { return false; }
        if !is_chord(core[pos]) { return false; }
        // Normalize: V7 and viio count as V for pattern matching
        let normalized = if core[pos] == tok("V7") || core[pos] == tok("viio") {
            tok("V")
        } else {
            core[pos]
        };
        base_chords.push(normalized);
        pos += 1;
    }

    // Check if base_chords ends with IV I (plagal tag) — strip it and check the rest
    let base = if base_chords.len() >= 3
        && base_chords[base_chords.len() - 2] == tok("IV")
        && base_chords[base_chords.len() - 1] == tok("I")
    {
        // Could be a plagal tag — check if what's before it is a valid base
        let without_tag = &base_chords[..base_chords.len() - 2];
        if is_valid_base(without_tag) {
            return true;
        }
        // Otherwise check the whole thing as-is
        &base_chords[..]
    } else {
        &base_chords[..]
    };

    is_valid_base(base)
}

fn is_valid_base(chords: &[usize]) -> bool {
    let patterns: &[&[usize]] = &[
        &[tok("I")],
        &[tok("ii"), tok("V"), tok("I")],
        &[tok("IV"), tok("I")],
        &[tok("vi"), tok("ii"), tok("V"), tok("I")],
        &[tok("ii"), tok("V"), tok("vi")],          // deceptive
        &[tok("IV"), tok("V"), tok("vi")],           // deceptive
        &[tok("bVI"), tok("bVII"), tok("I")],        // Mario cadence
    ];
    patterns.iter().any(|p| chords == *p)
}

fn is_secondary(t: usize) -> bool {
    t == tok("V/X") || t == tok("ii/X") || t == tok("bII7/X")
}

fn is_chord(t: usize) -> bool {
    t == tok("I") || t == tok("ii") || t == tok("iii") || t == tok("IV")
        || t == tok("V") || t == tok("vi")
        || t == tok("V7") || t == tok("viio")
        || t == tok("bVI") || t == tok("bVII")
}

// ─── Commands ──────────────────────────────────────────────────────────────

fn main() {
    assert_eq!(TOKENS.len(), VOCAB_SIZE, "VOCAB_SIZE mismatch");

    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("train");
    match cmd {
        "train" => cmd_train(),
        "generate" => cmd_generate(),
        "generate-midi" => cmd_generate_midi(),
        "random-midi" => cmd_random_midi(),
        "data" => cmd_show_data(),
        "midi" => cmd_midi(),
        _ => {
            eprintln!("Usage:");
            eprintln!("  train           Train on grammar-generated data");
            eprintln!("  generate        Generate sequences from trained model");
            eprintln!("  generate-midi   Generate from model → MIDI file");
            eprintln!("  random-midi     Generate from random weights → MIDI file");
            eprintln!("  data            Show example training data");
            eprintln!("  midi            Generate MIDI from random grammar sequences");
        }
    }
}

fn cmd_show_data() {
    let data = generate_dataset(20);
    for seq in &data {
        let names: Vec<&str> = seq.iter().map(|&t| token_name(t)).collect();
        println!("{}", names.join(" "));
    }
    eprintln!("\nGenerated 20 examples. Lengths: {:?}",
        data.iter().map(|s| s.len()).collect::<Vec<_>>());
}

fn cmd_train() {
    eprintln!("=== Chord Grammar Transformer ===");
    eprintln!("Learning a context-free grammar with attention\n");

    // Ultra small config
    let cfg = ModelConfig {
        vocab_size: VOCAB_SIZE,
        seq_len: 16,
        n_embd: 16,
        n_head: 2,
        n_layer: 1,
        batch_size: 16,
    };

    let sizes = weight_sizes(&cfg);
    let total_params: usize = sizes.iter().sum();
    eprintln!("Model: vocab={} seq={} embd={} heads={} layers={} batch={}",
        cfg.vocab_size, cfg.seq_len, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.batch_size);
    eprintln!("Total parameters: {}", total_params);

    // Generate training data
    let n_train = 2000;
    let dataset = generate_dataset(n_train);
    eprintln!("Training sequences: {}", dataset.len());

    // Show a few examples
    eprintln!("\nExamples:");
    for seq in dataset.iter().take(5) {
        let names: Vec<&str> = seq.iter().map(|&t| token_name(t)).collect();
        eprintln!("  {}", names.join(" "));
    }

    // Create training windows: input = seq[:-1], target = seq[1:]
    let mut windows: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for seq in &dataset {
        if seq.len() < 2 { continue; }
        let inp = pad_sequence(seq, cfg.seq_len);
        // Target is shifted by 1
        let mut tgt_seq: Vec<usize> = seq[1..].to_vec();
        // Pad target with 0s (pad token)
        while tgt_seq.len() < cfg.seq_len {
            tgt_seq.push(0);
        }
        let tgt: Vec<f32> = tgt_seq.iter().take(cfg.seq_len).map(|&t| t as f32).collect();
        windows.push((inp, tgt));
    }
    eprintln!("Training windows: {}\n", windows.len());

    // Compile
    eprintln!("Compiling DSL → graph → WASM...");
    let t0 = Instant::now();
    let tg = compile_for_training(&cfg);
    eprintln!("Graph: {} nodes ({:.1}ms)", tg.graph.nodes.len(), t0.elapsed().as_secs_f64() * 1000.0);

    let mut output_ids = vec![tg.loss];
    output_ids.extend_from_slice(&tg.grad_ids);
    let backend = AssemblyScriptBackend;
    let as_code = backend.emit_fused_multi_output(&tg.graph, &output_ids);
    eprintln!("Generated {} lines AS", as_code.lines().count());

    let t0 = Instant::now();
    let wasm_path = compile_wasm_named(&as_code, "chord_train");
    eprintln!("WASM compiled in {:.1}s", t0.elapsed().as_secs_f64());

    let mut weights = init_weights(&cfg);
    let mut runner = WasmRunner::new(&wasm_path);
    let mut optimizer = Adam::new(&sizes, 1e-3);
    let mut rng = rand::thread_rng();

    eprintln!("\nTraining...");
    let n_epochs = 200;
    for epoch in 0..n_epochs {
        let t_epoch = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0;

        // Shuffle
        let mut indices: Vec<usize> = (0..windows.len()).collect();
        for i in (1..indices.len()).rev() {
            indices.swap(i, rng.gen_range(0..=i));
        }

        for batch_start in (0..windows.len()).step_by(cfg.batch_size) {
            if batch_start + cfg.batch_size > windows.len() { break; }
            let mut tok_data = vec![0.0f32; cfg.batch_size * cfg.seq_len];
            let mut tgt_data = vec![0.0f32; cfg.batch_size * cfg.seq_len];
            for bi in 0..cfg.batch_size {
                let (ref inp, ref target) = windows[indices[batch_start + bi]];
                tok_data[bi*cfg.seq_len..(bi+1)*cfg.seq_len].copy_from_slice(inp);
                tgt_data[bi*cfg.seq_len..(bi+1)*cfg.seq_len].copy_from_slice(target);
            }

            let mut refs: Vec<&[f32]> = vec![&tok_data];
            for w in &weights { refs.push(w); }
            refs.push(&tgt_data);

            let result = runner.run(&refs);
            let expected_len = 1 + sizes.iter().sum::<usize>();
            if result.len() != expected_len {
                panic!("WASM returned {} values, expected {}", result.len(), expected_len);
            }
            epoch_loss += result[0];
            n_batches += 1;

            let mut off = 1;
            let grads: Vec<Vec<f32>> = sizes.iter().map(|&s| {
                let g = result[off..off+s].to_vec(); off += s; g
            }).collect();

            let norm: f32 = grads.iter().flat_map(|g| g.iter()).map(|g| g * g).sum::<f32>().sqrt();
            let mut clipped = grads;
            if norm > 1.0 {
                let s = 1.0 / norm;
                for g in &mut clipped { for v in g.iter_mut() { *v *= s; } }
            }
            optimizer.step(&mut weights, &clipped);
        }

        let avg_loss = epoch_loss / n_batches as f32;
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            eprintln!("Epoch {:3}/{}: loss={:.4}  ({:.1}s)",
                epoch+1, n_epochs, avg_loss, t_epoch.elapsed().as_secs_f64());
        }
    }

    // Save weights
    let save_path = project_root().join("chord_grammar_model.bin");
    save_model(&weights, &cfg, &save_path);

    // Quick generation test
    eprintln!("\n=== Generation test ===");
    generate_sequences(&weights, &cfg, 20);
}

fn cmd_generate() {
    let model_path = project_root().join("chord_grammar_model.bin");
    let (weights, cfg) = load_model(&model_path);
    eprintln!("Loaded model from {}", model_path.display());
    generate_sequences(&weights, &cfg, 50);
}

fn cmd_random_midi() {
    let args: Vec<String> = std::env::args().collect();
    let key_name = args.get(2).map(|s| s.as_str()).unwrap_or("C");
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("random_chords.mid");
    let n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);

    let key_root: u8 = match key_name {
        "C" => 0, "Db" | "C#" => 1, "D" => 2, "Eb" | "D#" => 3,
        "E" => 4, "F" => 5, "Gb" | "F#" => 6, "G" => 7,
        "Ab" | "G#" => 8, "A" => 9, "Bb" | "A#" => 10, "B" => 11,
        _ => { eprintln!("Unknown key: {key_name}, defaulting to C"); 0 }
    };

    let cfg = ModelConfig {
        vocab_size: VOCAB_SIZE, seq_len: 16, n_embd: 16,
        n_head: 2, n_layer: 1, batch_size: 16,
    };
    let weights = init_weights(&cfg);
    eprintln!("Using random (untrained) weights");

    let seqs = generate_sequences(&weights, &cfg, n);
    render_seqs_to_midi(&seqs, key_root, out_path);
}

fn cmd_generate_midi() {
    let args: Vec<String> = std::env::args().collect();
    let key_name = args.get(2).map(|s| s.as_str()).unwrap_or("C");
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("generated_chords.mid");
    let n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);

    let key_root: u8 = match key_name {
        "C" => 0, "Db" | "C#" => 1, "D" => 2, "Eb" | "D#" => 3,
        "E" => 4, "F" => 5, "Gb" | "F#" => 6, "G" => 7,
        "Ab" | "G#" => 8, "A" => 9, "Bb" | "A#" => 10, "B" => 11,
        _ => { eprintln!("Unknown key: {key_name}, defaulting to C"); 0 }
    };

    let model_path = project_root().join("chord_grammar_model.bin");
    let (weights, cfg) = load_model(&model_path);
    eprintln!("Loaded model from {}", model_path.display());

    let seqs = generate_sequences(&weights, &cfg, n);
    render_seqs_to_midi(&seqs, key_root, out_path);
}

fn generate_sequences(weights: &[Vec<f32>], cfg: &ModelConfig, n: usize) -> Vec<Vec<usize>> {
    let backend = AssemblyScriptBackend;
    let inf_graph = compile_for_inference(cfg);
    let inf_code = backend.emit_fused(&inf_graph);
    let inf_wasm = compile_wasm_named(&inf_code, "chord_infer");
    let mut runner = WasmRunner::new(&inf_wasm);
    let mut rng = rand::thread_rng();

    let mut valid = 0;
    let mut total = 0;
    let mut all_seqs = Vec::new();

    for _ in 0..n {
        let mut tokens = vec![tok("<bos>")];

        for _ in 0..14 { // max generation length
            let mut token_data = vec![0.0f32; cfg.seq_len];
            for (i, &t) in tokens.iter().enumerate().take(cfg.seq_len) {
                token_data[i] = t as f32;
            }

            let mut refs: Vec<&[f32]> = vec![&token_data];
            for w in weights { refs.push(w); }

            let logits = runner.run(&refs);
            let pos = tokens.len() - 1;
            let pos_logits = &logits[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];

            // Temperature sampling
            let temperature = 0.5f32;
            let max_l = pos_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = pos_logits.iter().map(|&l| ((l - max_l) / temperature).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

            // Sample
            let r: f32 = rng.r#gen();
            let mut cum = 0.0;
            let mut next_tok = 0;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if r < cum {
                    next_tok = i;
                    break;
                }
            }

            tokens.push(next_tok);
            if next_tok == tok("<eos>") { break; }
        }

        let names: Vec<&str> = tokens.iter().map(|&t| token_name(t)).collect();
        let valid_flag = is_valid_sequence(&tokens);
        if valid_flag { valid += 1; }
        total += 1;
        eprintln!("  {} {}", if valid_flag { "✓" } else { "✗" }, names.join(" "));
        all_seqs.push(tokens);
    }

    eprintln!("\nValid: {}/{} ({:.0}%)", valid, total, valid as f32 / total as f32 * 100.0);
    all_seqs
}

// ─── Save/Load ─────────────────────────────────────────────────────────────

fn save_model(weights: &[Vec<f32>], cfg: &ModelConfig, path: &std::path::Path) {
    let mut data: Vec<u8> = Vec::new();
    for &v in &[cfg.vocab_size, cfg.seq_len, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.batch_size] {
        data.extend_from_slice(&(v as u32).to_le_bytes());
    }
    data.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    for w in weights {
        data.extend_from_slice(&(w.len() as u32).to_le_bytes());
        for &v in w { data.extend_from_slice(&v.to_le_bytes()); }
    }
    std::fs::write(path, &data).unwrap();
    eprintln!("Saved model to {}", path.display());
}

fn load_model(path: &std::path::Path) -> (Vec<Vec<f32>>, ModelConfig) {
    let data = std::fs::read(path).unwrap_or_else(|_| {
        panic!("No model at {}. Run train first.", path.display())
    });
    let mut off = 0;
    let mut r32 = |o: &mut usize| -> u32 {
        let v = u32::from_le_bytes(data[*o..*o+4].try_into().unwrap()); *o += 4; v
    };
    let cfg = ModelConfig {
        vocab_size: r32(&mut off) as usize, seq_len: r32(&mut off) as usize,
        n_embd: r32(&mut off) as usize, n_head: r32(&mut off) as usize,
        n_layer: r32(&mut off) as usize, batch_size: r32(&mut off) as usize,
    };
    let n_w = r32(&mut off) as usize;
    let mut weights = Vec::new();
    for _ in 0..n_w {
        let len = r32(&mut off) as usize;
        let w: Vec<f32> = data[off..off+len*4].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        off += len * 4;
        weights.push(w);
    }
    (weights, cfg)
}

// ─── MIDI rendering ───────────────────────────────────────────────────────

/// Given a Roman numeral token name and a key root (MIDI note number for tonic),
/// return (bass_note, [chord_notes]) where chord_notes are in a nice piano range.
/// `target_root` is the root of the chord that secondary dominants resolve to.
/// Returns (bass_note, chord_notes_right_hand).
/// Bass is octave 2-3, right hand is octave 4.
fn chord_notes(token: &str, key_root: u8, target_root: Option<u8>) -> (u8, Vec<u8>) {
    match token {
        "I"    => build_chord(key_root, 0, &[0, 4, 7]),
        "ii"   => build_chord(key_root, 2, &[0, 3, 7]),
        "iii"  => build_chord(key_root, 4, &[0, 3, 7]),
        "IV"   => build_chord(key_root, 5, &[0, 4, 7]),
        "V"    => build_chord(key_root, 7, &[0, 4, 7]),
        "V7"   => build_chord(key_root, 7, &[0, 4, 7, 10]),
        "viio" => build_chord(key_root, 11, &[0, 3, 6]),
        "vi"   => build_chord(key_root, 9, &[0, 3, 7]),
        "bVI"  => build_chord(key_root, 8, &[0, 4, 7]),
        "bVII" => build_chord(key_root, 10, &[0, 4, 7]),
        "V/X" => {
            let tr = target_root.unwrap_or(key_root);
            let offset = (tr.wrapping_sub(key_root).wrapping_add(7)) % 12;
            build_chord(key_root, offset, &[0, 4, 7, 10]) // dominant 7th for color
        }
        "ii/X" => {
            let tr = target_root.unwrap_or(key_root);
            let offset = (tr.wrapping_sub(key_root).wrapping_add(2)) % 12;
            build_chord(key_root, offset, &[0, 3, 7])
        }
        "bII7/X" => {
            let tr = target_root.unwrap_or(key_root);
            let offset = (tr.wrapping_sub(key_root).wrapping_add(1)) % 12;
            build_chord(key_root, offset, &[0, 4, 7, 10]) // dom7
        }
        _ => build_chord(key_root, 0, &[0, 4, 7]),
    }
}

fn midi_note_name(n: u8) -> String {
    let names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
    let octave = n / 12;
    format!("{}{}", names[n as usize % 12], octave as i8 - 1)
}

fn build_chord(key_root: u8, offset: u8, intervals: &[u8]) -> (u8, Vec<u8>) {
    let root = key_root + offset;
    let bass = root + 36; // bass in octave 2 (C2=36)
    // Right hand: close voicing around C4 (60)
    let notes: Vec<u8> = intervals.iter().map(|&i| root + 60 + i).collect();
    (bass, notes)
}

/// Resolve the sequence of tokens into (token_name, target_root_offset) pairs
/// so secondary dominants know what they're targeting.
fn resolve_targets(tokens: &[&str], key_root: u8) -> Vec<(String, Option<u8>)> {
    let chord_root_offset = |t: &str| -> Option<u8> {
        match t {
            "I" => Some(0), "ii" => Some(2), "iii" => Some(4),
            "IV" => Some(5), "V" | "V7" => Some(7), "vi" => Some(9),
            "viio" => Some(11), "bVI" => Some(8), "bVII" => Some(10),
            _ => None,
        }
    };

    let mut result = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == "V/X" || tokens[i] == "ii/X" || tokens[i] == "bII7/X" {
            // Find the next diatonic chord — that's the target
            let mut target = None;
            for j in (i + 1)..tokens.len() {
                if let Some(off) = chord_root_offset(tokens[j]) {
                    target = Some(key_root + off);
                    break;
                }
            }
            result.push((tokens[i].to_string(), target));
        } else {
            result.push((tokens[i].to_string(), None));
        }
        i += 1;
    }
    result
}

/// Voice lead: try the chord at -12, 0, +12 and pick the shift that
/// minimizes movement from prev chord while staying in a good range.
fn voice_lead(prev: &[u8], current: &[u8]) -> Vec<u8> {
    if prev.is_empty() {
        return current.to_vec();
    }
    let mut best = current.to_vec();
    let mut best_cost = i32::MAX;
    for shift in [-12i8, 0, 12] {
        let candidate: Vec<u8> = current.iter().map(|&n| {
            (n as i32 + shift as i32) as u8
        }).collect();
        // Reject if any note is out of reasonable range
        if candidate.iter().any(|&n| n < 48 || n > 84) {
            continue;
        }
        // Cost: movement from prev + penalty for straying from home range (C4=60..C5=72)
        let move_cost: i32 = candidate.iter()
            .map(|&c| prev.iter().map(|&p| (c as i32 - p as i32).abs()).min().unwrap_or(100))
            .sum();
        // Small penalty for being far from the center of the range (66 = F#4)
        let center_cost: i32 = candidate.iter()
            .map(|&c| ((c as i32 - 66).abs() / 3))
            .sum();
        let cost = move_cost + center_cost;
        if cost < best_cost {
            best_cost = cost;
            best = candidate;
        }
    }
    best.sort();
    best
}

fn progression_to_midi(tokens: &[&str], key_root: u8) -> Vec<u8> {
    let resolved = resolve_targets(tokens, key_root);

    let ticks_per_beat = 480u16;
    let chord_beats = 4u32; // each chord lasts 2 beats (half note)
    let chord_ticks = chord_beats * ticks_per_beat as u32;

    let mut smf = Smf::new(Header::new(
        Format::SingleTrack,
        Timing::Metrical(u15::new(ticks_per_beat)),
    ));

    let mut track: Track = Vec::new();

    // Tempo: 100 BPM = 600000 microseconds per beat
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(600_000))),
    });

    // Program change: piano (channel 0)
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Midi {
            channel: u4::new(0),
            message: MidiMessage::ProgramChange { program: u7::new(0) },
        },
    });

    // Program change: acoustic bass (channel 1)
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Midi {
            channel: u4::new(1),
            message: MidiMessage::ProgramChange { program: u7::new(32) }, // acoustic bass
        },
    });

    let mut prev_chord_notes: Vec<u8> = Vec::new();
    struct NoteEvt {
        abs_tick: u32,
        channel: u8,
        note: u8,
        vel: u8,
        on: bool,
    }

    let mut events: Vec<NoteEvt> = Vec::new();

    for (idx, (tok_name, target)) in resolved.iter().enumerate() {
        let (bass, raw_chord) = chord_notes(tok_name, key_root, *target);
        let voiced = voice_lead(&prev_chord_notes, &raw_chord);
        prev_chord_notes = voiced.clone();

        let vel: u8 = if idx == 0 || tok_name == "I" { 80 } else { 70 };
        let tick = idx as u32 * chord_ticks;

        // Bass note
        events.push(NoteEvt { abs_tick: tick, channel: 1, note: bass, vel: vel - 10, on: true });
        events.push(NoteEvt { abs_tick: tick + chord_ticks - 10, channel: 1, note: bass, vel: 0, on: false });

        // Chord notes
        for &n in &voiced {
            events.push(NoteEvt { abs_tick: tick, channel: 0, note: n, vel, on: true });
            events.push(NoteEvt { abs_tick: tick + chord_ticks - 10, channel: 0, note: n, vel: 0, on: false });
        }
    }

    // Sort by absolute tick, note-offs before note-ons at same tick
    events.sort_by_key(|e| (e.abs_tick, if e.on { 1u8 } else { 0 }));

    let mut last_tick = 0u32;
    for evt in &events {
        let delta = evt.abs_tick - last_tick;
        last_tick = evt.abs_tick;

        let message = if evt.on {
            MidiMessage::NoteOn { key: u7::new(evt.note), vel: u7::new(evt.vel) }
        } else {
            MidiMessage::NoteOff { key: u7::new(evt.note), vel: u7::new(0) }
        };

        track.push(TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(evt.channel),
                message,
            },
        });
    }

    // End of track
    track.push(TrackEvent {
        delta: u28::new(chord_ticks),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    smf.tracks.push(track);

    let mut buf = Vec::new();
    smf.write(&mut buf).unwrap();
    buf
}

fn render_seqs_to_midi(seqs: &[Vec<usize>], key_root: u8, out_path: &str) {
    let progressions: Vec<Vec<&str>> = seqs.iter().map(|seq| {
        seq.iter()
            .map(|&t| token_name(t))
            .filter(|&t| t != "<pad>" && t != "<bos>" && t != "<eos>")
            .collect()
    }).collect();

    let ticks_per_beat = 480u16;
    let chord_beats = 4u32;
    let chord_ticks = chord_beats * ticks_per_beat as u32;
    let rest_ticks = 0u32;
    let eighth = ticks_per_beat as u32 / 2;

    let mut smf = Smf::new(Header::new(
        Format::SingleTrack,
        Timing::Metrical(u15::new(ticks_per_beat)),
    ));

    let mut track: Track = Vec::new();

    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(600_000))),
    });
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Midi {
            channel: u4::new(0),
            message: MidiMessage::ProgramChange { program: u7::new(0) },
        },
    });

    struct NoteEvt { abs_tick: u32, note: u8, vel: u8, on: bool }
    let mut all_events: Vec<NoteEvt> = Vec::new();
    let mut global_offset: u32 = 0;

    for prog in &progressions {
        if prog.is_empty() { continue; }

        let resolved = resolve_targets(prog, key_root);
        let mut prev_chord_notes: Vec<u8> = Vec::new();

        for (idx, (tok_name, target)) in resolved.iter().enumerate() {
            let (bass, raw_chord) = chord_notes(tok_name, key_root, *target);
            let voiced = voice_lead(&prev_chord_notes, &raw_chord);
            prev_chord_notes = voiced.clone();

            let base_vel: u8 = if idx == 0 { 80 } else { 70 };
            let tick = global_offset + idx as u32 * chord_ticks;

            // Left hand: just bass root
            let lh_vel = base_vel - 10;
            all_events.push(NoteEvt { abs_tick: tick, note: bass, vel: lh_vel, on: true });
            all_events.push(NoteEvt { abs_tick: tick + chord_ticks - 10, note: bass, vel: 0, on: false });

            // Right hand: arpeggio
            let mut arp_notes: Vec<u8> = voiced.clone();
            if let Some(&root) = voiced.first() {
                arp_notes.push(root + 12);
            }
            let pattern: Vec<usize> = if arp_notes.len() >= 4 {
                vec![0, 1, 2, 3, 2, 1, 0, 1]
            } else {
                vec![0, 1, 2, 1, 0, 1, 2, 1]
            };
            for (step, &pat_idx) in pattern.iter().enumerate() {
                let note = arp_notes[pat_idx % arp_notes.len()];
                let step_tick = tick + step as u32 * eighth;
                let vel = if step % 4 == 0 { base_vel } else { base_vel - 12 };
                let dur = eighth - 20;
                all_events.push(NoteEvt { abs_tick: step_tick, note, vel, on: true });
                all_events.push(NoteEvt { abs_tick: step_tick + dur, note, vel: 0, on: false });
            }
        }

        global_offset += resolved.len() as u32 * chord_ticks + rest_ticks;
    }

    all_events.sort_by_key(|e| (e.abs_tick, if e.on { 1u8 } else { 0 }));

    let mut last_tick = 0u32;
    for evt in &all_events {
        let delta = evt.abs_tick - last_tick;
        last_tick = evt.abs_tick;
        let message = if evt.on {
            MidiMessage::NoteOn { key: u7::new(evt.note), vel: u7::new(evt.vel) }
        } else {
            MidiMessage::NoteOff { key: u7::new(evt.note), vel: u7::new(0) }
        };
        track.push(TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi { channel: u4::new(0), message },
        });
    }

    track.push(TrackEvent {
        delta: u28::new(chord_ticks),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    smf.tracks.push(track);
    let mut buf = Vec::new();
    smf.write(&mut buf).unwrap();
    std::fs::write(out_path, &buf).unwrap();
    eprintln!("\nWrote {} bytes to {out_path}", buf.len());
}

fn cmd_midi() {
    let args: Vec<String> = std::env::args().collect();
    let key_name = args.get(2).map(|s| s.as_str()).unwrap_or("C");
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("chord_progressions.mid");
    let n_progressions: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);

    let key_root: u8 = match key_name {
        "C" => 0, "Db" | "C#" => 1, "D" => 2, "Eb" | "D#" => 3,
        "E" => 4, "F" => 5, "Gb" | "F#" => 6, "G" => 7,
        "Ab" | "G#" => 8, "A" => 9, "Bb" | "A#" => 10, "B" => 11,
        _ => { eprintln!("Unknown key: {key_name}, defaulting to C"); 0 }
    };

    eprintln!("Generating {n_progressions} progressions in {key_name} major → {out_path}");

    let mut rng = rand::thread_rng();

    // Generate progressions and concatenate them into one MIDI file
    let ticks_per_beat = 480u16;
    let chord_beats = 4u32;
    let chord_ticks = chord_beats * ticks_per_beat as u32;
    let rest_ticks = 0u32; // 2 beat rest between progressions

    let mut smf = Smf::new(Header::new(
        Format::SingleTrack,
        Timing::Metrical(u15::new(ticks_per_beat)),
    ));

    let mut track: Track = Vec::new();

    // Tempo: 100 BPM
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(600_000))),
    });

    // Piano on ch 0 (both hands)
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Midi {
            channel: u4::new(0),
            message: MidiMessage::ProgramChange { program: u7::new(0) },
        },
    });

    struct NoteEvt {
        abs_tick: u32,
        channel: u8,
        note: u8,
        vel: u8,
        on: bool,
    }

    let mut all_events: Vec<NoteEvt> = Vec::new();
    let mut global_offset: u32 = 0;

    for prog_i in 0..n_progressions {
        let seq = generate_sequence(&mut rng);
        let core: Vec<&str> = seq.iter()
            .map(|&t| token_name(t))
            .filter(|&t| t != "<pad>" && t != "<bos>" && t != "<eos>")
            .collect();

        eprintln!("  {}: {}", prog_i + 1, core.join(" "));

        let resolved = resolve_targets(&core, key_root);
        let mut prev_chord_notes: Vec<u8> = Vec::new();

        for (idx, (tok_name, target)) in resolved.iter().enumerate() {
            let (bass, raw_chord) = chord_notes(tok_name, key_root, *target);
            let voiced = voice_lead(&prev_chord_notes, &raw_chord);
            prev_chord_notes = voiced.clone();

            // Debug: print voicing
            let note_names: Vec<String> = voiced.iter().map(|&n| midi_note_name(n)).collect();
            eprint!("    {}: bass={} rh=[{}]", tok_name, midi_note_name(bass), note_names.join(" "));
            if let Some(t) = target { eprint!(" (→{})", midi_note_name(*t + 60)); }
            eprintln!();

            let base_vel: u8 = if idx == 0 { 80 } else { 70 };
            let tick = global_offset + idx as u32 * chord_ticks;
            let eighth = ticks_per_beat as u32 / 2; // 240 ticks

            // Left hand: just the bass root, held for full duration
            let lh_vel = base_vel - 10;
            all_events.push(NoteEvt { abs_tick: tick, channel: 0, note: bass, vel: lh_vel, on: true });
            all_events.push(NoteEvt { abs_tick: tick + chord_ticks - 10, channel: 0, note: bass, vel: 0, on: false });

            // Arpeggio: build a pattern from chord tones
            let mut arp_notes: Vec<u8> = voiced.clone();
            // Add octave-up root for the top of the arpeggio
            if let Some(&root) = voiced.first() {
                arp_notes.push(root + 12);
            }
            // Build the up-down pattern
            let pattern: Vec<usize> = if arp_notes.len() >= 4 {
                vec![0, 1, 2, 3, 2, 1, 0, 1] // up to octave, back down
            } else {
                // 3-note triad: up then down
                vec![0, 1, 2, 1, 0, 1, 2, 1]
            };

            for (step, &pat_idx) in pattern.iter().enumerate() {
                let note = arp_notes[pat_idx % arp_notes.len()];
                let step_tick = tick + step as u32 * eighth;
                // Accent beats 1 and 3 (steps 0 and 4)
                let vel = if step % 4 == 0 { base_vel } else { base_vel - 12 };
                let dur = eighth - 20; // slight gap between notes

                all_events.push(NoteEvt { abs_tick: step_tick, channel: 0, note, vel, on: true });
                all_events.push(NoteEvt { abs_tick: step_tick + dur, channel: 0, note, vel: 0, on: false });
            }
        }

        global_offset += resolved.len() as u32 * chord_ticks + rest_ticks;
    }

    all_events.sort_by_key(|e| (e.abs_tick, if e.on { 1u8 } else { 0 }));

    let mut last_tick = 0u32;
    for evt in &all_events {
        let delta = evt.abs_tick - last_tick;
        last_tick = evt.abs_tick;

        let message = if evt.on {
            MidiMessage::NoteOn { key: u7::new(evt.note), vel: u7::new(evt.vel) }
        } else {
            MidiMessage::NoteOff { key: u7::new(evt.note), vel: u7::new(0) }
        };

        track.push(TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(evt.channel),
                message,
            },
        });
    }

    track.push(TrackEvent {
        delta: u28::new(chord_ticks),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    smf.tracks.push(track);

    let mut buf = Vec::new();
    smf.write(&mut buf).unwrap();
    std::fs::write(out_path, &buf).unwrap();
    eprintln!("Wrote {} bytes to {out_path}", buf.len());
}
