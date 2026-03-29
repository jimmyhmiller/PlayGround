mod midi_parse;

use std::io::Write;
use std::process::Command;
use std::time::Instant;

use rand::Rng;
use tensor_lang_backend::assemblyscript::AssemblyScriptBackend;
use std::collections::HashMap;
use tensor_lang_graph::{compile_with_env, Graph, NodeId, Op};

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

// ─── DSL model definition ──────────────────────────────────────────────────

fn model_source() -> String {
    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("model.tensor");
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

// ─── Compile DSL for training (adds loss + backward) ───────────────────────

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

    // The compiled graph has logits as its last node.
    let logits = NodeId(graph.nodes.len() - 1);

    // Find learnable weight input nodes (everything except tokens = input_0).
    let weight_ids: Vec<NodeId> = graph.nodes.iter().enumerate()
        .filter_map(|(i, n)| {
            if let Op::Input { name } = &n.op {
                if name != "input_0" { Some(NodeId(i)) } else { None }
            } else { None }
        })
        .collect();

    // Add targets input
    let targets = graph.add_node(Op::Input { name: "targets".into() }, vec![]);
    graph.set_input_shape(targets, tensor_lang_graph::dims(&[b, t]));

    // One-hot encode targets
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

    // Log-softmax of logits
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

    // Cross-entropy: -mean(sum(one_hot * log_probs))
    let prod = graph.add_node(Op::Mul, vec![target_one_hot, log_probs]);
    let sum_v = graph.add_node(Op::ReduceSum { axis: 2 }, vec![prod]);
    let sum_b = graph.add_node(Op::ReduceSum { axis: 0 }, vec![sum_v]);
    let sum_all = graph.add_node(Op::ReduceSum { axis: 0 }, vec![sum_b]);
    let scalar = graph.add_node(Op::Reshape { shape: vec![] }, vec![sum_all]);
    let neg = graph.add_node(Op::Neg, vec![scalar]);
    let inv_bt = graph.add_node(Op::Constant(1.0 / (b * t) as f64), vec![]);
    let loss = graph.add_node(Op::Mul, vec![neg, inv_bt]);

    // Backward pass
    let grad_ids = graph.grad(loss, &weight_ids);

    TrainingGraph { graph, loss, grad_ids }
}

/// Compile DSL for inference (forward only, returns logits)
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
    let mut sizes = vec![v * d]; // wte (no wpe — ALiBi handles position)
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
            // wte embedding
            (0..size).map(|_| rng.r#gen::<f32>() * 0.04 - 0.02).collect()
        } else {
            // Layer weights: pattern repeats every 12 params
            // [ln1_g, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_g, ln2_b, fc_w, fc_b, mlp_w, mlp_b]
            // Then final: ln_f_g, ln_f_b
            let layer_offset = (i - 1) % 12;
            let is_gamma = (layer_offset == 0 || layer_offset == 6) && size == d
                || i == sizes.len() - 2;
            if is_gamma {
                vec![1.0f32; size]
            } else if size == d || size == 3 * d || size == cfg.mlp_hidden() {
                vec![0.0f32; size] // biases
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
    compile_wasm_named(as_code, "train_model")
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
               "--initialMemory", "4096",
               "--maximumMemory", "65536"])
        .current_dir(&root).output().expect("failed to run asc");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("asc compilation failed:\n{stderr}");
    }
    wasm_path
}

/// A persistent WASM runner — one node process stays alive for many calls.
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
            .args(["--max-old-space-size=4096",
                   runner.to_str().unwrap(),
                   wasm_path.to_str().unwrap()])
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

        // Write request: [n_inputs: u32] then for each [size: u32] [data: f32×size]
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

        // Read response: [n_outputs: u32] [data: f32×n_outputs]
        let mut header = [0u8; 4];
        self.stdout.read_exact(&mut header).expect("failed to read output header from WASM runner");
        let n_outputs = u32::from_le_bytes(header) as usize;

        let mut data = vec![0u8; n_outputs * 4];
        self.stdout.read_exact(&mut data).expect("failed to read output data from WASM runner");

        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl Drop for WasmRunner {
    fn drop(&mut self) {
        // Kill the child process to ensure it exits
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

// ─── Model save/load ───────────────────────────────────────────────────────

fn save_model(weights: &[Vec<f32>], cfg: &ModelConfig, vocab: &midi_parse::Vocab) {
    save_model_to(weights, cfg, vocab, &project_root().join("midi_model.bin"));
}

fn save_model_to(weights: &[Vec<f32>], cfg: &ModelConfig, vocab: &midi_parse::Vocab, path: &std::path::Path) {
    let mut data: Vec<u8> = Vec::new();
    for &v in &[cfg.vocab_size, cfg.seq_len, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.batch_size] {
        data.extend_from_slice(&(v as u32).to_le_bytes());
    }
    data.extend_from_slice(&(vocab.pitches.len() as u32).to_le_bytes());
    for &p in &vocab.pitches { data.push(p); }
    data.extend_from_slice(&(vocab.max_time_shift as u32).to_le_bytes());
    data.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    for w in weights {
        data.extend_from_slice(&(w.len() as u32).to_le_bytes());
        for &v in w { data.extend_from_slice(&v.to_le_bytes()); }
    }
    std::fs::write(&path, &data).unwrap();
    eprintln!("Saved model to {}", path.display());
}

fn load_model_from(path: &std::path::Path) -> (Vec<Vec<f32>>, ModelConfig, midi_parse::Vocab) {
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
    let n_pitches = r32(&mut off) as usize;
    let mut pitches = Vec::new();
    for _ in 0..n_pitches { pitches.push(data[off]); off += 1; }
    let max_time_shift = r32(&mut off) as usize;
    let vocab = midi_parse::Vocab { pitches, max_time_shift };
    let n_w = r32(&mut off) as usize;
    let mut weights = Vec::new();
    for _ in 0..n_w {
        let len = r32(&mut off) as usize;
        let w: Vec<f32> = data[off..off+len*4].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        off += len * 4;
        weights.push(w);
    }
    (weights, cfg, vocab)
}

// ─── Commands ──────────────────────────────────────────────────────────────

const DEFAULT_MIDI: &str = "/Users/jimmyhmiller/Downloads/satie";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "train" => cmd_train(args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_MIDI)),
        "train-rl" => cmd_train_rl(args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_MIDI)),
        "generate" => {
            let mut checkpoint = None;
            let mut temp = None;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--temp" | "--temperature" => { i += 1; temp = Some(args[i].parse::<f32>().expect("invalid temperature")); }
                    other => { if checkpoint.is_none() { checkpoint = Some(other.to_string()); } }
                }
                i += 1;
            }
            cmd_generate(checkpoint.as_deref(), false, temp)
        }
        "random" => cmd_generate(None, true, None),
        "parse" => cmd_parse(args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_MIDI)),
        _ => {
            eprintln!("Usage:");
            eprintln!("  train [midi]         Train on MIDI file (supervised)");
            eprintln!("  train-rl [midi]      RL fine-tune for bass note sustain");
            eprintln!("  generate [checkpoint] Generate from trained/checkpoint weights");
            eprintln!("  random               Generate from random weights");
            eprintln!("  parse [midi]         Analyze MIDI file");
        }
    }
}

fn cmd_parse(path: &str) {
    let files = collect_midi_files(path);
    for f in &files {
        eprintln!("=== {} ===", f);
        let events = midi_parse::parse_midi_file(f);
        let (tokens, vocab) = midi_parse::tokenize(&events, 120, 32);
        midi_parse::print_summary(&events, &tokens, &vocab);
        analyze_density(&tokens, &vocab);
        // Test reward on windows — show quality over time
        let window = 128.min(tokens.len());
        let stride = 64;
        let mut pos = 0;
        let (mut total, mut rewarded) = (0, 0);
        while pos + window <= tokens.len() {
            let (r, bd) = midi_parse::compute_satie_reward(&tokens[pos..pos+window], &vocab);
            let pct = pos as f32 / tokens.len() as f32 * 100.0;
            eprintln!("  @{:>4} ({:>2.0}%): r={:.0} [{}]", pos, pct, r, bd);
            total += 1;
            if r > 0.5 { rewarded += 1; }
            pos += stride;
        }
        eprintln!("  Reward: {}/{} ({:.0}%)", rewarded, total, rewarded as f32 / total.max(1) as f32 * 100.0);
    }
}

fn analyze_density(tokens: &[usize], vocab: &midi_parse::Vocab) {
    // Measure: for every 16-step window (1 measure), count note events and dead time
    let mut time_pos = 0usize;
    let mut events_per_measure: Vec<usize> = Vec::new();
    let mut dead_steps_per_measure: Vec<usize> = Vec::new();

    let mut measure_events = 0usize;
    let mut measure_start = 0usize;
    let mut last_event_time = 0usize;
    let mut max_gap_in_measure = 0usize;

    for &tok in tokens {
        if let Some(s) = vocab.is_time_shift(tok) {
            time_pos += s;
            // Check if we crossed a measure boundary (every 16 steps)
            while time_pos >= measure_start + 16 {
                let dead = max_gap_in_measure;
                events_per_measure.push(measure_events);
                dead_steps_per_measure.push(dead);
                measure_start += 16;
                measure_events = 0;
                max_gap_in_measure = 0;
                last_event_time = measure_start;
            }
        } else if vocab.is_note_on(tok).is_some() {
            let gap = time_pos.saturating_sub(last_event_time);
            max_gap_in_measure = max_gap_in_measure.max(gap);
            last_event_time = time_pos;
            measure_events += 1;
        }
    }

    if !events_per_measure.is_empty() {
        let avg_events: f32 = events_per_measure.iter().sum::<usize>() as f32 / events_per_measure.len() as f32;
        let avg_dead: f32 = dead_steps_per_measure.iter().sum::<usize>() as f32 / dead_steps_per_measure.len() as f32;
        let max_dead = dead_steps_per_measure.iter().max().unwrap_or(&0);
        eprintln!("  Density: {:.1} note-ons/measure, avg max-gap={:.1} steps, worst gap={} steps",
            avg_events, avg_dead, max_dead);
        eprintln!("  Measures: {}", events_per_measure.len());

        // Time shift distribution
        let mut shift_counts = vec![0usize; 33];
        for &tok in tokens {
            if let Some(s) = vocab.is_time_shift(tok) {
                if s < shift_counts.len() { shift_counts[s] += 1; }
            }
        }
        eprint!("  Time shifts: ");
        for (s, &c) in shift_counts.iter().enumerate() {
            if c > 0 { eprint!("T+{}:{}  ", s, c); }
        }
        eprintln!();

        // Consecutive time shift analysis (total gap between note events)
        let mut gaps: Vec<usize> = Vec::new();
        let mut current_gap = 0usize;
        let mut in_gap = false;
        for &tok in tokens {
            if let Some(s) = vocab.is_time_shift(tok) {
                current_gap += s;
                in_gap = true;
            } else if vocab.is_note_on(tok).is_some() || vocab.is_note_off(tok).is_some() {
                if in_gap && current_gap > 0 {
                    gaps.push(current_gap);
                }
                current_gap = 0;
                in_gap = false;
            }
        }
        gaps.sort();
        if !gaps.is_empty() {
            let median = gaps[gaps.len() / 2];
            let p90 = gaps[gaps.len() * 9 / 10];
            let max = gaps[gaps.len() - 1];
            eprintln!("  Event gaps: median={} p90={} max={} (in time steps)", median, p90, max);
        }
    }
}

fn collect_midi_files(path: &str) -> Vec<String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let mut files: Vec<String> = std::fs::read_dir(p).unwrap()
            .filter_map(|e| {
                let e = e.unwrap();
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".mid") || name.ends_with(".midi") {
                    Some(e.path().to_string_lossy().to_string())
                } else { None }
            })
            .collect();
        files.sort();
        files
    } else {
        vec![path.to_string()]
    }
}

fn cmd_train(midi_path: &str) {
    eprintln!("=== MIDI Trainer ===");

    let files = collect_midi_files(midi_path);
    eprintln!("Source files:");
    for f in &files { eprintln!("  {f}"); }

    let (tokens, vocab) = midi_parse::tokenize_multi(&files, 120, 32);
    eprintln!("Combined: {} tokens, vocab size {}", tokens.len(), vocab.size());
    eprintln!("  {} unique pitches", vocab.pitches.len());
    eprint!("  ");
    for p in &vocab.pitches { eprint!("{} ", midi_parse::pitch_name(*p)); }
    eprintln!();

    let cfg = ModelConfig {
        vocab_size: vocab.size(), seq_len: 128, n_embd: 64,
        n_head: 4, n_layer: 2, batch_size: 1,
    };
    eprintln!("Model: vocab={} seq={} embd={} heads={} layers={} batch={}",
        cfg.vocab_size, cfg.seq_len, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.batch_size);

    // Training windows
    let stride = cfg.seq_len / 2;
    let mut windows: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let mut pos = 0;
    while pos + cfg.seq_len + 1 <= tokens.len() {
        let inp: Vec<f32> = tokens[pos..pos+cfg.seq_len].iter().map(|&t| t as f32).collect();
        let tgt: Vec<f32> = tokens[pos+1..pos+cfg.seq_len+1].iter().map(|&t| t as f32).collect();
        windows.push((inp, tgt));
        pos += stride;
    }
    eprintln!("Training windows: {} (stride={})", windows.len(), stride);

    // Build training graph from DSL
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
    let wasm_path = compile_wasm(&as_code);
    eprintln!("WASM compiled in {:.1}s", t0.elapsed().as_secs_f64());

    let sizes = weight_sizes(&cfg);
    let mut weights = init_weights(&cfg);
    eprintln!("Parameters: {}", sizes.iter().sum::<usize>());

    // Start persistent WASM runner (one process for all training steps)
    let mut runner = WasmRunner::new(&wasm_path);
    eprintln!("WASM runner started");

    let mut optimizer = Adam::new(&sizes, 3e-4);
    let mut rng = rand::thread_rng();

    eprintln!("\nTraining...");
    for epoch in 0..200 {
        let t_epoch = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0;
        let mut indices: Vec<usize> = (0..windows.len()).collect();
        for i in (1..indices.len()).rev() {
            indices.swap(i, rng.gen_range(0..=i));
        }

        for batch_start in (0..windows.len()).step_by(cfg.batch_size) {
            if batch_start + cfg.batch_size > windows.len() { break; }
            let mut tok = vec![0.0f32; cfg.batch_size * cfg.seq_len];
            let mut tgt = vec![0.0f32; cfg.batch_size * cfg.seq_len];
            for bi in 0..cfg.batch_size {
                let (ref inp, ref target) = windows[indices[batch_start + bi]];
                tok[bi*cfg.seq_len..(bi+1)*cfg.seq_len].copy_from_slice(inp);
                tgt[bi*cfg.seq_len..(bi+1)*cfg.seq_len].copy_from_slice(target);
            }

            // Input order: tokens, weights (wte, wpe, layer weights, ln_f), targets
            let mut refs: Vec<&[f32]> = vec![&tok];
            for w in &weights { refs.push(w); }
            refs.push(&tgt);

            let result = runner.run(&refs);
            let expected_len = 1 + sizes.iter().sum::<usize>();
            if result.len() != expected_len {
                // Check if input had NaN
                let nan_count: usize = refs.iter()
                    .flat_map(|r| r.iter())
                    .filter(|v| !v.is_finite()).count();
                panic!("WASM returned {} values, expected {}. Input NaN/Inf count: {}",
                    result.len(), expected_len, nan_count);
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

            // Check for NaN/Inf in weights (these corrupt WASM execution)
            for (wi, w) in weights.iter().enumerate() {
                for (j, &v) in w.iter().enumerate() {
                    if !v.is_finite() {
                        panic!("NaN/Inf in weight[{}][{}] = {} after epoch {} batch {}",
                            wi, j, v, epoch + 1, n_batches);
                    }
                }
            }
        }

        let avg_loss = epoch_loss / n_batches as f32;
        eprintln!("Epoch {:3}/200: loss={:.4}  ({} batches, {:.1}s)",
            epoch+1, avg_loss, n_batches, t_epoch.elapsed().as_secs_f64());

        // Checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0 {
            let checkpoint_path = project_root().join(format!("checkpoint_epoch_{}.bin", epoch + 1));
            save_model_to(&weights, &cfg, &vocab, &checkpoint_path);
            eprintln!("  → Saved checkpoint: {}", checkpoint_path.display());
        }
    }

    save_model(&weights, &cfg, &vocab);
    eprintln!("\nTo generate: cargo run --release -p tensor-lang-train -- generate");
}

fn cmd_generate(checkpoint: Option<&str>, random: bool, temp_override: Option<f32>) {
    let (weights, cfg, vocab) = if random {
        eprintln!("=== Generate (random weights) ===");
        let files = collect_midi_files(DEFAULT_MIDI);
        let (_, vocab) = midi_parse::tokenize_multi(&files, 120, 32);
        let cfg = ModelConfig {
            vocab_size: vocab.size(), seq_len: 128, n_embd: 64,
            n_head: 4, n_layer: 2, batch_size: 1,
        };
        (init_weights(&cfg), cfg, vocab)
    } else {
        let path = match checkpoint {
            Some(p) => std::path::PathBuf::from(p),
            None => project_root().join("midi_model.bin"),
        };
        eprintln!("=== Generate from {} ===", path.display());
        load_model_from(&path)
    };

    eprintln!("Compiling DSL → inference WASM...");
    let t0 = Instant::now();
    let backend = AssemblyScriptBackend;
    let inf_graph = compile_for_inference(&cfg);
    let inf_code = backend.emit_fused(&inf_graph);
    let inf_wasm = compile_wasm(&inf_code);
    eprintln!("Compiled in {:.1}s", t0.elapsed().as_secs_f64());

    let mut inf_runner = WasmRunner::new(&inf_wasm);
    let mut rng = rand::thread_rng();
    let temperature = temp_override.unwrap_or(0.3f32);

    // Seed selection
    let seed_from_start = std::env::var("SEED_START").is_ok();
    let custom_seed = std::env::var("SEED_CUSTOM").ok();
    let files = collect_midi_files(DEFAULT_MIDI);
    let file = if seed_from_start { &files[0] } else { &files[rng.gen_range(0..files.len())] };
    let seed_events = midi_parse::parse_midi_file(file);
    let seed_tokens = midi_parse::tokenize_with_vocab(&seed_events, &vocab, 120);

    // Find a random 2-measure window by accumulating time shifts
    let mut measure_starts: Vec<usize> = vec![0]; // token indices where each measure boundary falls
    let mut accumulated_steps: usize = 0;
    for (i, &tok) in seed_tokens.iter().enumerate() {
        if let Some(s) = vocab.is_time_shift(tok) {
            accumulated_steps += s;
            // Every 16 steps (1 measure in 4/4 at 4 steps/beat) is a measure boundary
            while accumulated_steps >= 16 && measure_starts.last().map_or(true, |&last| last < i + 1) {
                measure_starts.push(i + 1);
                accumulated_steps -= 16;
            }
        }
    }

    let mut tokens: Vec<usize> = if let Some(ref custom) = custom_seed {
        // Parse custom seed like "ON(C2) T+4 ON(E3) ON(G3) ON(B3) T+8 OFF(B3) OFF(G3) OFF(E3) OFF(C2)"
        let mut toks = Vec::new();
        for part in custom.split_whitespace() {
            if part.starts_with("ON(") && part.ends_with(")") {
                let name = &part[3..part.len()-1];
                if let Some(pitch) = parse_pitch_name(name) {
                    toks.push(vocab.note_on(pitch));
                }
            } else if part.starts_with("OFF(") && part.ends_with(")") {
                let name = &part[4..part.len()-1];
                if let Some(pitch) = parse_pitch_name(name) {
                    toks.push(vocab.note_off(pitch));
                }
            } else if part.starts_with("T+") {
                if let Ok(s) = part[2..].parse::<usize>() {
                    toks.push(vocab.time_shift(s));
                }
            }
        }
        eprintln!("Custom seed: {} tokens", toks.len());
        toks
    } else if seed_from_start {
        // Use the very beginning of the file
        let end_idx = if measure_starts.len() >= 3 { measure_starts[2] } else { seed_tokens.len().min(cfg.seq_len) };
        seed_tokens[..end_idx.min(seed_tokens.len())].to_vec()
    } else if measure_starts.len() >= 3 {
        // Pick a random 2-measure start (need at least 2 measures ahead)
        let max_start = measure_starts.len() - 2;
        let pick = rng.gen_range(0..max_start);
        let start_idx = measure_starts[pick];
        let end_idx = if pick + 2 < measure_starts.len() {
            measure_starts[pick + 2]
        } else {
            seed_tokens.len()
        };
        seed_tokens[start_idx..end_idx.min(seed_tokens.len())].to_vec()
    } else {
        // Song too short, just take what we have
        seed_tokens[..seed_tokens.len().min(cfg.seq_len)].to_vec()
    };
    eprintln!("Seed from: {}", file);
    eprint!("Seed: ");
    for &t in &tokens { eprint!("{} ", vocab.token_name(t)); }
    eprintln!("\n");

    eprintln!("Generating 500 tokens (temp={temperature})...");
    for step in 0..500 {
        let ctx_start = if tokens.len() > cfg.seq_len { tokens.len() - cfg.seq_len } else { 0 };
        let ctx = &tokens[ctx_start..];
        let mut token_data = vec![0.0f32; cfg.seq_len];
        for (i, &t) in ctx.iter().enumerate() { token_data[i] = t as f32; }

        let mut refs: Vec<&[f32]> = vec![&token_data];
        for w in &weights { refs.push(w); }

        let logits = inf_runner.run(&refs);
        let pos = ctx.len() - 1;
        let pos_logits = &logits[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];

        // Temperature-scaled softmax
        let max_l = pos_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = pos_logits.iter().map(|&l| ((l - max_l) / temperature).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

        // Top-p (nucleus) sampling: keep only tokens with cumulative prob <= top_p
        let top_p = 0.9f32;
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cumulative = 0.0f32;
        let mut cutoff = sorted_indices.len();
        for (i, &idx) in sorted_indices.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative > top_p {
                cutoff = i + 1;
                break;
            }
        }
        // Renormalize over kept tokens
        let kept = &sorted_indices[..cutoff];
        let kept_sum: f32 = kept.iter().map(|&i| probs[i]).sum();

        let mut r: f32 = rng.gen_range(0.0..1.0) * kept_sum;
        let mut next = kept[0];
        for &idx in kept { r -= probs[idx]; if r <= 0.0 { next = idx; break; } }
        if next == 0 { next = 1; } // skip PAD

        if step < 60 {
            eprint!("{} ", vocab.token_name(next));
            if step % 15 == 14 { eprintln!(); }
        }
        tokens.push(next);
    }
    eprintln!();

    let midi_path = project_root().join("generated.mid");
    tokens_to_midi(&midi_path, &tokens, &vocab);
    eprintln!("\nWrote: {}", midi_path.display());
    eprintln!("Play:  open {}", midi_path.display());
}

// ─── Rollout generation ───────────────────────────────────────────────────

/// Pick a random 2-measure seed from a random training file.
fn pick_random_seed(
    files: &[String],
    vocab: &midi_parse::Vocab,
    max_len: usize,
    rng: &mut impl rand::Rng,
) -> Vec<usize> {
    let file = &files[rng.gen_range(0..files.len())];
    let events = midi_parse::parse_midi_file(file);
    let tokens = midi_parse::tokenize_with_vocab(&events, vocab, 120);

    let mut measure_starts: Vec<usize> = vec![0];
    let mut accumulated_steps: usize = 0;
    for (i, &tok) in tokens.iter().enumerate() {
        if let Some(s) = vocab.is_time_shift(tok) {
            accumulated_steps += s;
            while accumulated_steps >= 16 && measure_starts.last().map_or(true, |&last| last < i + 1) {
                measure_starts.push(i + 1);
                accumulated_steps -= 16;
            }
        }
    }

    if measure_starts.len() >= 3 {
        let max_start = measure_starts.len() - 2;
        let pick = rng.gen_range(0..max_start);
        let start_idx = measure_starts[pick];
        let end_idx = if pick + 2 < measure_starts.len() {
            measure_starts[pick + 2]
        } else {
            tokens.len()
        };
        tokens[start_idx..end_idx.min(tokens.len())].to_vec()
    } else {
        tokens[..tokens.len().min(max_len)].to_vec()
    }
}

/// Generate a token sequence autoregressively from a seed.
fn generate_rollout(
    runner: &mut WasmRunner,
    weights: &[Vec<f32>],
    cfg: &ModelConfig,
    _vocab: &midi_parse::Vocab,
    seed_tokens: &[usize],
    num_tokens: usize,
    temperature: f32,
    rng: &mut impl rand::Rng,
) -> Vec<usize> {
    let mut tokens = seed_tokens.to_vec();
    for _ in 0..num_tokens {
        let ctx_start = if tokens.len() > cfg.seq_len { tokens.len() - cfg.seq_len } else { 0 };
        let ctx = &tokens[ctx_start..];
        let mut token_data = vec![0.0f32; cfg.seq_len];
        for (i, &t) in ctx.iter().enumerate() { token_data[i] = t as f32; }

        let mut refs: Vec<&[f32]> = vec![&token_data];
        for w in weights { refs.push(w); }

        let logits = runner.run(&refs);
        let pos = ctx.len() - 1;
        let pos_logits = &logits[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];

        // Temperature-scaled softmax
        let max_l = pos_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = pos_logits.iter().map(|&l| ((l - max_l) / temperature).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

        // Top-p sampling
        let top_p = 0.9f32;
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cumulative = 0.0f32;
        let mut cutoff = sorted_indices.len();
        for (i, &idx) in sorted_indices.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative > top_p { cutoff = i + 1; break; }
        }
        let kept = &sorted_indices[..cutoff];
        let kept_sum: f32 = kept.iter().map(|&i| probs[i]).sum();
        let mut r: f32 = rng.gen_range(0.0..1.0) * kept_sum;
        let mut next = kept[0];
        for &idx in kept { r -= probs[idx]; if r <= 0.0 { next = idx; break; } }
        if next == 0 { next = 1; } // skip PAD

        tokens.push(next);
    }
    tokens
}

// ─── RL graph compilation ─────────────────────────────────────────────────

/// Compile the model graph with REINFORCE policy gradient loss.
/// Inputs: tokens [B, T], weights..., advantages [B]
/// Output: loss (scalar) + gradients for each weight
fn compile_for_rl(cfg: &ModelConfig) -> TrainingGraph {
    let source = model_source();
    let (dims, constants) = model_env(cfg);
    let mut graph = compile_with_env(&source, &dims, &constants);

    let b = cfg.batch_size;
    let t = cfg.seq_len;
    let v = cfg.vocab_size;

    let logits = NodeId(graph.nodes.len() - 1);

    // Find learnable weight input nodes (everything except tokens = input_0)
    let weight_ids: Vec<NodeId> = graph.nodes.iter().enumerate()
        .filter_map(|(i, n)| {
            if let Op::Input { name } = &n.op {
                if name != "input_0" { Some(NodeId(i)) } else { None }
            } else { None }
        })
        .collect();

    // Add advantages input [B]
    let advantages = graph.add_node(Op::Input { name: "advantages".into() }, vec![]);
    graph.set_input_shape(advantages, tensor_lang_graph::dims(&[b]));

    // Log-softmax of logits [B, T, V]
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

    // One-hot encode input tokens (the rollout actions) to select log_probs
    let tokens_node = NodeId(0); // input_0 = tokens [B, T]
    let classes = graph.add_node(Op::Arange { size: tensor_lang_graph::Dim::Lit(v) }, vec![]);
    let cls = graph.add_node(Op::Reshape { shape: tensor_lang_graph::dims(&[1, 1, v]) }, vec![classes]);
    let cls_exp = graph.add_node(Op::Expand { shape: tensor_lang_graph::dims(&[b, t, v]) }, vec![cls]);
    let tok_r = graph.add_node(Op::Reshape { shape: tensor_lang_graph::dims(&[b, t, 1]) }, vec![tokens_node]);
    let tok_exp = graph.add_node(Op::Expand { shape: tensor_lang_graph::dims(&[b, t, v]) }, vec![tok_r]);
    let half = graph.add_node(Op::Constant(0.5), vec![]);
    let neg_half = graph.add_node(Op::Neg, vec![half]);
    let lo = graph.add_node(Op::Add, vec![cls_exp, neg_half]);
    let half2 = graph.add_node(Op::Constant(0.5), vec![]);
    let hi = graph.add_node(Op::Add, vec![cls_exp, half2]);
    let lt_lo = graph.add_node(Op::CmpLt, vec![lo, tok_exp]);
    let lt_hi = graph.add_node(Op::CmpLt, vec![tok_exp, hi]);
    let action_one_hot = graph.add_node(Op::Mul, vec![lt_lo, lt_hi]);

    // Select log_probs of chosen actions: sum(one_hot * log_probs, axis=2)
    let prod = graph.add_node(Op::Mul, vec![action_one_hot, log_probs]);
    let chosen_lp = graph.add_node(Op::ReduceSum { axis: 2 }, vec![prod]); // [B, T, 1]

    // Sum log_probs across time for each sequence
    let seq_lp = graph.add_node(Op::ReduceSum { axis: 1 }, vec![chosen_lp]); // [B, 1, 1]

    // Weight by advantages: reshape [B] -> [B, 1, 1]
    let adv_r = graph.add_node(Op::Reshape { shape: tensor_lang_graph::dims(&[b, 1, 1]) }, vec![advantages]);
    let weighted = graph.add_node(Op::Mul, vec![seq_lp, adv_r]); // [B, 1, 1]

    // Loss = -mean(advantage * sum_t(log_pi))
    let sum_b = graph.add_node(Op::ReduceSum { axis: 0 }, vec![weighted]); // [1, 1, 1]
    let scalar = graph.add_node(Op::Reshape { shape: vec![] }, vec![sum_b]);
    let neg = graph.add_node(Op::Neg, vec![scalar]);
    let inv_b = graph.add_node(Op::Constant(1.0 / b as f64), vec![]);
    let loss = graph.add_node(Op::Mul, vec![neg, inv_b]);

    let grad_ids = graph.grad(loss, &weight_ids);
    TrainingGraph { graph, loss, grad_ids }
}

// ─── RL training command ──────────────────────────────────────────────────

fn cmd_train_rl(midi_path: &str) {
    eprintln!("=== REINFORCE RL Training (Satie style) ===");

    // Load pretrained model
    let model_path = project_root().join("midi_model.bin");
    let (mut weights, base_cfg, vocab) = load_model_from(&model_path);
    eprintln!("Loaded model: vocab={} seq={} embd={} heads={} layers={}",
        base_cfg.vocab_size, base_cfg.seq_len, base_cfg.n_embd, base_cfg.n_head, base_cfg.n_layer);

    let rl_batch_size = 4usize;
    let rollout_tokens = 128usize; // long rollouts so tail quality is tested
    let temperature = 0.5f32;

    // Compile both WASM modules upfront (but only run one at a time)
    let inf_cfg = ModelConfig { batch_size: 1, ..base_cfg.clone() };
    let rl_cfg = ModelConfig { batch_size: rl_batch_size, ..base_cfg.clone() };

    eprintln!("Compiling WASM modules...");
    let t0 = Instant::now();
    let backend = AssemblyScriptBackend;

    let inf_graph = compile_for_inference(&inf_cfg);
    let inf_code = backend.emit_fused(&inf_graph);
    let inf_wasm = compile_wasm_named(&inf_code, "rl_inference");

    let rl_tg = compile_for_rl(&rl_cfg);
    eprintln!("  RL graph: {} nodes", rl_tg.graph.nodes.len());
    let mut output_ids = vec![rl_tg.loss];
    output_ids.extend_from_slice(&rl_tg.grad_ids);
    let rl_code = backend.emit_fused_multi_output(&rl_tg.graph, &output_ids);
    let rl_wasm = compile_wasm_named(&rl_code, "rl_training");
    eprintln!("  Both WASM modules compiled in {:.1}s", t0.elapsed().as_secs_f64());

    let sizes = weight_sizes(&rl_cfg);
    let mut optimizer = Adam::new(&sizes, 1e-5);
    let mut rng = rand::thread_rng();

    let files = collect_midi_files(midi_path);
    eprintln!("Training MIDI files: {}", files.len());

    let num_episodes = 200;
    let rollouts_per_batch = 20; // generate many rollouts at once to amortize runner startup
    eprintln!("\nRL training: {} episodes, {} rollouts/batch, {} batch size\n",
        num_episodes, rollouts_per_batch, rl_batch_size);

    let mut episode = 0;
    while episode < num_episodes {
        // Phase 1: Generate a big batch of rollouts (only inference runner alive)
        eprintln!("--- Generating {} rollouts ---", rollouts_per_batch);
        let mut all_rollouts: Vec<Vec<usize>> = Vec::new();
        let mut all_rewards: Vec<f32> = Vec::new();
        {
            let mut inf_runner = WasmRunner::new(&inf_wasm);
            for ri in 0..rollouts_per_batch {
                let t_roll = Instant::now();
                let seed = pick_random_seed(&files, &vocab, inf_cfg.seq_len, &mut rng);
                let rollout = generate_rollout(
                    &mut inf_runner, &weights, &inf_cfg, &vocab,
                    &seed, rollout_tokens, temperature, &mut rng,
                );
                // Score the TAIL of the rollout — this is what the model generates
                // (the head is the seed from training data, which is always good)
                let tail_start = rollout.len() / 2;
                let (reward, breakdown) = midi_parse::compute_satie_reward(&rollout[tail_start..], &vocab);
                eprintln!("  rollout {}/{}: {} tokens, tail r={:.0} [{}] ({:.1}s)",
                    ri + 1, rollouts_per_batch, rollout.len(), reward, breakdown, t_roll.elapsed().as_secs_f64());
                all_rollouts.push(rollout);
                all_rewards.push(reward);
            }
            // inf_runner dropped here
        }

        // Phase 2: Process rollouts in mini-batches (only RL runner alive)
        eprintln!("--- Training on rollouts ---");
        {
            let mut rl_runner = WasmRunner::new(&rl_wasm);
            for batch_start in (0..all_rollouts.len()).step_by(rl_batch_size) {
                if batch_start + rl_batch_size > all_rollouts.len() { break; }
                let t_ep = Instant::now();

                let batch_rollouts = &all_rollouts[batch_start..batch_start + rl_batch_size];
                let batch_rewards = &all_rewards[batch_start..batch_start + rl_batch_size];

                let mean_reward: f32 = batch_rewards.iter().sum::<f32>() / batch_rewards.len() as f32;
                let advantages: Vec<f32> = batch_rewards.iter().map(|&r| r - mean_reward).collect();

                let adv_nonzero = advantages.iter().any(|&a| a.abs() > 1e-8);
                if !adv_nonzero {
                    eprintln!("Episode {:3}/{}: no signal (all rewards={:.2}), skipping",
                        episode + 1, num_episodes, mean_reward);
                    episode += 1;
                    continue;
                }

                let seq_len = rl_cfg.seq_len;
                let mut tok_flat = vec![0.0f32; rl_batch_size * seq_len];
                for (bi, rollout) in batch_rollouts.iter().enumerate() {
                    let start = if rollout.len() > seq_len { rollout.len() - seq_len } else { 0 };
                    let ctx = &rollout[start..];
                    for (j, &t) in ctx.iter().enumerate() {
                        tok_flat[bi * seq_len + j] = t as f32;
                    }
                }

                let mut refs: Vec<&[f32]> = vec![&tok_flat];
                for w in &weights { refs.push(w); }
                refs.push(&advantages);

                let result = rl_runner.run(&refs);
                let expected_len = 1 + sizes.iter().sum::<usize>();
                if result.len() != expected_len {
                    panic!("RL WASM returned {} values, expected {}", result.len(), expected_len);
                }

                let loss = result[0];

                let mut off = 1;
                let grads: Vec<Vec<f32>> = sizes.iter().map(|&s| {
                    let g = result[off..off + s].to_vec(); off += s; g
                }).collect();

                let norm: f32 = grads.iter().flat_map(|g| g.iter()).map(|g| g * g).sum::<f32>().sqrt();
                let mut clipped = grads;
                if norm > 1.0 {
                    let s = 1.0 / norm;
                    for g in &mut clipped { for v in g.iter_mut() { *v *= s; } }
                }

                if !loss.is_finite() || !norm.is_finite() {
                    eprintln!("Episode {:3}/{}: NaN/Inf (loss={}, norm={}), skipping",
                        episode + 1, num_episodes, loss, norm);
                    episode += 1;
                    continue;
                }

                optimizer.step(&mut weights, &clipped);

                let n_success = batch_rewards.iter().filter(|&&r| r > 0.5).count();
                eprintln!("Episode {:3}/{}: loss={:.4} reward={:.2} ({}/{} success) norm={:.1} ({:.1}s)",
                    episode + 1, num_episodes, loss, mean_reward,
                    n_success, rl_batch_size, norm, t_ep.elapsed().as_secs_f64());

                episode += 1;
                if episode >= num_episodes { break; }
            }
            // rl_runner dropped here
        }

        // Checkpoint
        let path = project_root().join(format!("rl_checkpoint_{}.bin", episode));
        save_model_to(&weights, &base_cfg, &vocab, &path);
    }

    // Save final model
    let final_path = project_root().join("midi_model_rl.bin");
    save_model_to(&weights, &base_cfg, &vocab, &final_path);
    eprintln!("\nRL training complete. Model saved to {}", final_path.display());
    eprintln!("Generate with: cargo run --release -p tensor-lang-train -- generate {}", final_path.display());
}

fn parse_pitch_name(name: &str) -> Option<u8> {
    let names = [("C", 0), ("C#", 1), ("D", 2), ("D#", 3), ("E", 4), ("F", 5),
                 ("F#", 6), ("G", 7), ("G#", 8), ("A", 9), ("A#", 10), ("B", 11)];
    for &(n, semitone) in &names {
        if name.starts_with(n) && name.len() > n.len() {
            if let Ok(octave) = name[n.len()..].parse::<i32>() {
                return Some(((octave + 1) * 12 + semitone) as u8);
            }
        }
    }
    None
}

// ─── Tokens → MIDI ────────────────────────────────────────────────────────

fn tokens_to_midi(path: &std::path::Path, tokens: &[usize], vocab: &midi_parse::Vocab) {
    let ticks_per_beat: u16 = 480;
    let ticks_per_step: u32 = 120;
    let mut data: Vec<u8> = Vec::new();

    data.extend_from_slice(b"MThd");
    data.extend_from_slice(&6u32.to_be_bytes());
    data.extend_from_slice(&0u16.to_be_bytes());
    data.extend_from_slice(&1u16.to_be_bytes());
    data.extend_from_slice(&ticks_per_beat.to_be_bytes());

    let mut track: Vec<u8> = Vec::new();
    // Tempo: 72 BPM
    track.push(0x00);
    track.extend_from_slice(&[0xFF, 0x51, 0x03]);
    let tempo: u32 = 833_333;
    track.push((tempo >> 16) as u8);
    track.push((tempo >> 8) as u8);
    track.push(tempo as u8);
    // Piano
    track.push(0x00);
    track.push(0xC0);
    track.push(0);

    let mut pending_delta: u32 = 0;
    for &token in tokens {
        if let Some(pitch) = vocab.is_note_on(token) {
            write_vlq(&mut track, pending_delta); pending_delta = 0;
            track.push(0x90); track.push(pitch); track.push(80);
        } else if let Some(pitch) = vocab.is_note_off(token) {
            write_vlq(&mut track, pending_delta); pending_delta = 0;
            track.push(0x80); track.push(pitch); track.push(0);
        } else if let Some(steps) = vocab.is_time_shift(token) {
            pending_delta += steps as u32 * ticks_per_step;
        }
    }
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
    while value > 0 { bytes.push((value & 0x7F) as u8 | 0x80); value >>= 7; }
    bytes.reverse();
    buf.extend_from_slice(&bytes);
}
