use std::path::Path;
use std::sync::mpsc;

use tensor_lang_graph::{compile, nanogpt, Graph, NodeId, Op, TensorRuntime};
use tensor_lang_backend::arm::ArmBackend;
use tensor_lang_backend::arm_runtime::ArmRuntime;

pub const MAX_SEQ_LEN: usize = 8;

pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
}

#[derive(Clone)]
pub struct NodeInfo {
    pub graph_idx: usize,
    pub shape: Vec<usize>,
    pub op_name: String,
    pub size: usize,
}

pub struct LogitsResult {
    pub logits: Vec<f32>,
    pub n_tokens: usize,
}

impl LogitsResult {
    pub fn top_k_predictions(&self, config: &Gpt2Config, k: usize) -> Vec<(usize, f32)> {
        let v = config.vocab_size;
        let last_start = (self.n_tokens - 1) * v;
        if last_start + v > self.logits.len() { return vec![]; }
        let last = &self.logits[last_start..last_start + v];
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.iter().take(k).map(|&i| (i, last[i])).collect()
    }
    pub fn next_token(&self, config: &Gpt2Config) -> usize {
        self.top_k_predictions(config, 1)[0].0
    }
}

/// Messages sent from background thread to main thread.
pub enum ForwardMessage {
    /// Logits ready (sent first).
    Logits(LogitsResult),
    /// A single tile's data is ready: (node_info_index, data).
    Tile(usize, Vec<f32>),
    /// All done, runtimes returned.
    Done(ArmRuntime, ArmRuntime),
}

pub struct Gpt2Model {
    pub config: Gpt2Config,
    pub node_infos: Vec<NodeInfo>,
    pub graph: Graph,
    full_output_sizes: Vec<usize>,
    fast_output_size: usize,
    input_names: Vec<String>,
    input_data: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
    /// Runtimes live here when not in use by a background thread.
    runtimes: Option<(ArmRuntime, ArmRuntime)>,
    /// Channel for streaming results from background thread.
    pending: Option<mpsc::Receiver<ForwardMessage>>,
}

fn op_name(op: &Op) -> String {
    match op {
        Op::Input { name } => format!("Input({})", name),
        Op::Constant(v) => if *v == (*v as i64) as f64 { format!("Const({})", *v as i64) } else { format!("Const({:.4})", v) },
        Op::Arange { .. } => "Arange".into(),
        Op::Neg => "Neg".into(), Op::Recip => "Recip".into(),
        Op::Exp2 => "Exp2".into(), Op::Log2 => "Log2".into(), Op::Sqrt => "Sqrt".into(),
        Op::Add => "Add".into(), Op::Mul => "Mul".into(), Op::Max => "Max".into(),
        Op::CmpLt => "CmpLt".into(),
        Op::ReduceSum { axis } => format!("Sum(ax={})", axis),
        Op::ReduceMax { axis } => format!("Max(ax={})", axis),
        Op::Reshape { shape } => { let d: Vec<String> = shape.iter().map(|d| d.as_usize().map(|v| v.to_string()).unwrap_or("?".into())).collect(); format!("Reshape({})", d.join(",")) }
        Op::Permute { order } => format!("Perm({})", order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
        Op::Expand { shape } => { let d: Vec<String> = shape.iter().map(|d| d.as_usize().map(|v| v.to_string()).unwrap_or("?".into())).collect(); format!("Expand({})", d.join(",")) }
        Op::Pad { .. } => "Pad".into(), Op::Shrink { .. } => "Shrink".into(),
    }
}

fn build_input_data(graph: &Graph, weights: &[Vec<f32>], seq_len: usize, n_embd: usize) -> (Vec<String>, Vec<Vec<f32>>) {
    let mut names = Vec::new();
    let mut data = Vec::new();
    for node in &graph.nodes {
        if let Op::Input { name } = &node.op {
            names.push(name.clone());
            if name == "input_0" { data.push(vec![0.0f32; seq_len]); }
            else if name == "input_3" { data.push(vec![0.0f32; seq_len * seq_len]); }
            else if name.starts_with("input_") {
                let num: usize = name[6..].parse().unwrap();
                let mi = if num <= 2 { num - 1 } else { num - 2 };
                if mi == 1 { data.push(weights[1][..seq_len * n_embd].to_vec()); }
                else { data.push(weights[mi].clone()); }
            }
        }
    }
    (names, data)
}

fn patch_inputs(data: &mut [Vec<f32>], names: &[String], token_ids: &[u32], n_tokens: usize) {
    let seq_len = MAX_SEQ_LEN;
    for (name, buf) in names.iter().zip(data.iter_mut()) {
        if name == "input_0" {
            buf.fill(0.0);
            for (i, &id) in token_ids.iter().enumerate().take(n_tokens) { buf[i] = id as f32; }
        } else if name == "input_3" {
            buf.fill(-1e6);
            for row in 0..seq_len { for col in 0..=row {
                if row < n_tokens && col < n_tokens { buf[row * seq_len + col] = 0.0; }
            }}
        }
    }
}

impl Gpt2Model {
    pub fn load(weights_dir: &Path) -> Self {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(weights_dir.join("manifest.json")).unwrap(),
        ).unwrap();
        let weights_bin = std::fs::read(weights_dir.join("weights.bin")).unwrap();
        let c = &manifest["config"];
        let vocab_size = c["vocab_size"].as_u64().unwrap() as usize;
        let n_embd = c["n_embd"].as_u64().unwrap() as usize;
        let n_head = c["n_head"].as_u64().unwrap() as usize;
        let n_layer = c["n_layer"].as_u64().unwrap() as usize;
        let config = Gpt2Config { vocab_size, n_embd, n_head, n_layer };
        eprintln!("GPT-2: V={vocab_size} D={n_embd} H={n_head} L={n_layer}");

        let tensors = manifest["tensors"].as_array().unwrap();
        let weights: Vec<Vec<f32>> = tensors.iter().map(|ti| {
            let o = ti["offset"].as_u64().unwrap() as usize;
            let n = ti["n_elements"].as_u64().unwrap() as usize;
            weights_bin[o..o+n*4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
        }).collect();
        eprintln!("Loaded {} weights ({:.0} MB)", weights.len(), weights.iter().map(|w| w.len()*4).sum::<usize>() as f64/1e6);

        let seq_len = MAX_SEQ_LEN;
        eprintln!("Compiling graph (T={seq_len})...");
        let program = nanogpt::generate_nanogpt_program(1, seq_len, vocab_size, n_embd, n_head, n_layer);
        let graph = compile(&program);
        eprintln!("  {} graph nodes", graph.nodes.len());

        let logits_id = (0..graph.nodes.len()).rev().find(|&i| {
            !matches!(&graph.nodes[i].op, Op::Reshape{..}|Op::Expand{..}|Op::Permute{..}|Op::Input{..})
        }).unwrap_or(graph.nodes.len() - 1);

        let max_per = seq_len * n_embd * 5;
        let max_total = 300_000_000;
        let mut total = 0usize;
        let mut output_ids: Vec<NodeId> = Vec::new();
        let mut node_infos: Vec<NodeInfo> = Vec::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            let shape: Vec<usize> = node.shape.iter().filter_map(|d| d.as_usize()).collect();
            let size = shape.iter().product::<usize>().max(1);
            if matches!(&node.op, Op::Input{..}) { continue; }
            if size <= 1 { continue; }
            if matches!(&node.op, Op::Reshape{..}|Op::Expand{..}|Op::Permute{..}
                |Op::Shrink{..}|Op::Pad{..}|Op::Constant(_)|Op::Arange{..}) { continue; }
            if i != logits_id && size > max_per { continue; }
            if total + size > max_total && i != logits_id { continue; }
            output_ids.push(NodeId(i));
            node_infos.push(NodeInfo { graph_idx: i, shape, op_name: op_name(&node.op), size });
            total += size;
        }
        let full_output_sizes: Vec<usize> = node_infos.iter().map(|n| n.size).collect();
        eprintln!("  {} materialized nodes ({:.1} MB)", node_infos.len(), total as f64*4.0/1e6);

        let (input_names, input_data) = build_input_data(&graph, &weights, seq_len, n_embd);

        let fast_output_size = graph.nodes[logits_id].shape.iter()
            .map(|d| d.as_usize().unwrap()).product::<usize>().max(1);

        eprintln!("Compiling ARM plans...");
        let t0 = std::time::Instant::now();
        let fast_rt = ArmRuntime::new(&ArmBackend.emit_fused_multi_output(&graph, &[NodeId(logits_id)]));
        let full_rt = ArmRuntime::new(&ArmBackend.emit_fused_multi_output(&graph, &output_ids));
        eprintln!("  ARM ready ({:.0}ms)", t0.elapsed().as_secs_f64()*1000.0);

        Gpt2Model {
            config, node_infos, graph, full_output_sizes, fast_output_size,
            input_names, input_data, weights,
            runtimes: Some((fast_rt, full_rt)),
            pending: None,
        }
    }

    pub fn wte(&self) -> &[f32] { &self.weights[0] }

    /// Launch both fast (logits) and full (all nodes) forward passes on a background thread.
    /// Non-blocking. Returns false if runtimes aren't available yet.
    pub fn launch_async(&mut self, token_ids: &[u32]) -> bool {
        // Try to reclaim runtimes from a previous run (non-blocking)
        self.try_reclaim();

        let (mut fast_rt, mut full_rt) = match self.runtimes.take() {
            Some(r) => r,
            None => {
                eprintln!("Runtimes busy, skipping launch");
                return false;
            }
        };

        let n_tokens = token_ids.len().min(MAX_SEQ_LEN);
        let mut input_data = self.input_data.clone();
        patch_inputs(&mut input_data, &self.input_names, token_ids, n_tokens);
        let sizes = self.full_output_sizes.clone();
        let fast_size = self.fast_output_size;

        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);

        std::thread::spawn(move || {
            let refs: Vec<&[f32]> = input_data.iter().map(|d| d.as_slice()).collect();

            // Fast forward first (logits) — send immediately
            let t0 = std::time::Instant::now();
            let logits_data = fast_rt.run(&refs, fast_size);
            eprintln!("  Fast: {:.0}ms", t0.elapsed().as_secs_f64()*1000.0);
            let _ = tx.send(ForwardMessage::Logits(LogitsResult { logits: logits_data, n_tokens }));

            // Full forward (all intermediates)
            let t0 = std::time::Instant::now();
            let total: usize = sizes.iter().sum();
            let result = full_rt.run(&refs, total);
            eprintln!("  Full: {:.0}ms", t0.elapsed().as_secs_f64()*1000.0);

            // Stream each tile individually
            let mut offset = 0;
            for (i, &sz) in sizes.iter().enumerate() {
                let _ = tx.send(ForwardMessage::Tile(i, result[offset..offset+sz].to_vec()));
                offset += sz;
            }

            // Return runtimes
            let _ = tx.send(ForwardMessage::Done(fast_rt, full_rt));
        });

        true
    }

    /// Non-blocking poll: drain all available messages.
    /// Returns (new_logits, new_tiles, all_done).
    pub fn poll(&mut self) -> (Option<LogitsResult>, Vec<(usize, Vec<f32>)>, bool) {
        let Some(rx) = &self.pending else { return (None, vec![], false); };
        let mut logits = None;
        let mut tiles = Vec::new();
        let mut done = false;
        loop {
            match rx.try_recv() {
                Ok(ForwardMessage::Logits(l)) => logits = Some(l),
                Ok(ForwardMessage::Tile(idx, data)) => tiles.push((idx, data)),
                Ok(ForwardMessage::Done(fast, full)) => { self.runtimes = Some((fast, full)); done = true; break; }
                Err(_) => break,
            }
        }
        if done { self.pending = None; }
        (logits, tiles, done)
    }

    fn try_reclaim(&mut self) {
        if self.runtimes.is_some() { return; }
        if let Some(rx) = &self.pending {
            loop {
                match rx.try_recv() {
                    Ok(ForwardMessage::Done(fast, full)) => { self.runtimes = Some((fast, full)); break; }
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
            if self.runtimes.is_some() { self.pending = None; }
        }
    }

    pub fn ready(&self) -> bool { self.runtimes.is_some() }
}
