use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ndarray::ArrayD;
use tensor_lang_backend::{assemblyscript::AssemblyScriptBackend, Backend};
use tensor_lang_graph::{compile, nanogpt, Op};
use tensor_lang_test_oracle::eval_with_inputs;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1000);

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

fn run_assemblyscript(as_code: &str, inputs: &[&[f32]]) -> (Vec<f32>, std::time::Duration) {
    let root = project_root();
    let tmp_dir = std::env::temp_dir();
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let tid = std::thread::current().id();

    let src_path = tmp_dir.join(format!("tensor_bench_{id}_{tid:?}.ts"));
    let wasm_path = tmp_dir.join(format!("tensor_bench_{id}_{tid:?}.wasm"));
    std::fs::write(&src_path, as_code).unwrap();

    // Compile
    let t_compile = Instant::now();
    let asc_output = Command::new("npx")
        .args(["asc", src_path.to_str().unwrap(),
               "--outFile", wasm_path.to_str().unwrap(),
               "--exportRuntime", "--optimize"])
        .current_dir(&root)
        .output()
        .expect("failed to run asc");
    let compile_time = t_compile.elapsed();

    if !asc_output.status.success() {
        let stderr = String::from_utf8_lossy(&asc_output.stderr);
        panic!("asc compilation failed:\n{stderr}");
    }

    // Build inputs JSON
    let inputs_json: Vec<Vec<f32>> = inputs.iter().map(|s| s.to_vec()).collect();
    let inner: Vec<String> = inputs_json.iter().map(|arr| {
        let elems: Vec<String> = arr.iter().map(|v| format!("{v}")).collect();
        format!("[{}]", elems.join(","))
    }).collect();
    let inputs_str = format!("[{}]", inner.join(","));

    // Run
    let t_run = Instant::now();
    let runner = root.join("test_runner.mjs");
    let node_output = Command::new("node")
        .args([runner.to_str().unwrap(), wasm_path.to_str().unwrap(), &inputs_str])
        .current_dir(&root)
        .output()
        .expect("failed to run node");
    let run_time = t_run.elapsed();

    if !node_output.status.success() {
        let stderr = String::from_utf8_lossy(&node_output.stderr);
        panic!("node execution failed:\n{stderr}");
    }

    let stdout = String::from_utf8_lossy(&node_output.stdout);
    let s = stdout.trim().trim_start_matches('[').trim_end_matches(']');
    let result: Vec<f32> = if s.is_empty() {
        vec![]
    } else {
        s.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect()
    };

    eprintln!("  AS compile: {:.1}ms, AS run: {:.1}ms",
        compile_time.as_secs_f64() * 1000.0, run_time.as_secs_f64() * 1000.0);

    (result, run_time)
}

fn build_inputs(graph: &tensor_lang_graph::Graph, vocab_size: usize) -> HashMap<String, ArrayD<f32>> {
    let mut inputs = HashMap::new();
    for node in &graph.nodes {
        if let Op::Input { name } = &node.op {
            let shape: Vec<usize> = node.shape.iter().map(|d| d.as_usize().unwrap()).collect();
            let size: usize = shape.iter().product();
            let data: Vec<f32> = if name == "input_0" {
                (0..size).map(|i| (i % vocab_size) as f32).collect()
            } else {
                (0..size).map(|i| (i as f32 * 0.1).sin() * 0.1).collect()
            };
            inputs.insert(name.clone(), ArrayD::from_shape_vec(shape, data).unwrap());
        }
    }
    inputs
}

#[test]
fn bench_oracle_vs_as() {
    // Config: B=1, T=3, vocab=4, d=8, heads=2, layers=1
    let batch = 1; let seq_len = 3; let vocab_size = 4;
    let n_embd = 8; let n_head = 2; let n_layer = 1;

    let program = nanogpt::generate_nanogpt_program(batch, seq_len, vocab_size, n_embd, n_head, n_layer);
    let graph = compile(&program);
    let inputs = build_inputs(&graph, vocab_size);

    // Oracle
    let t0 = Instant::now();
    let oracle_results = eval_with_inputs(&graph, &inputs);
    let oracle_time = t0.elapsed();
    let oracle_output = oracle_results.last().unwrap();
    eprintln!("Oracle: {:.1}ms", oracle_time.as_secs_f64() * 1000.0);

    // AS backend
    let backend = AssemblyScriptBackend;
    let as_code = backend.emit(&graph);
    eprintln!("Generated AS code: {} bytes, {} lines", as_code.len(), as_code.lines().count());

    let mut ordered_inputs: Vec<(&str, &ArrayD<f32>)> = Vec::new();
    for node in &graph.nodes {
        if let Op::Input { name } = &node.op {
            ordered_inputs.push((name.as_str(), inputs.get(name.as_str()).unwrap()));
        }
    }
    let flat_inputs: Vec<Vec<f32>> = ordered_inputs.iter()
        .map(|(_, arr)| arr.iter().copied().collect()).collect();
    let input_slices: Vec<&[f32]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

    let (as_output, as_run_time) = run_assemblyscript(&as_code, &input_slices);

    eprintln!("\nOracle time:  {:.1}ms", oracle_time.as_secs_f64() * 1000.0);
    eprintln!("AS run time:  {:.1}ms", as_run_time.as_secs_f64() * 1000.0);
    eprintln!("Ratio:        {:.1}x", as_run_time.as_secs_f64() / oracle_time.as_secs_f64());

    // Verify they match
    let oracle_flat: Vec<f32> = oracle_output.iter().copied().collect();
    let max_diff = oracle_flat.iter().zip(as_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Max diff:     {:.6}", max_diff);
    assert!(max_diff < 0.01);
}

#[test]
fn bench_oracle_gpt2_single_token() {
    // Time a single GPT-2 forward pass (oracle only - AS is impractical at this scale)
    let root = project_root();
    if !root.join("gpt2_weights/manifest.json").exists() {
        eprintln!("Skipping: run export_gpt2.py first");
        return;
    }

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("gpt2_weights/manifest.json")).unwrap()
    ).unwrap();
    let weights_bin = std::fs::read(root.join("gpt2_weights/weights.bin")).unwrap();
    let config = &manifest["config"];
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let n_embd = config["n_embd"].as_u64().unwrap() as usize;
    let n_head = config["n_head"].as_u64().unwrap() as usize;
    let n_layer = config["n_layer"].as_u64().unwrap() as usize;
    let seq_len = 3; // "Hello, world"

    let program = nanogpt::generate_nanogpt_program(1, seq_len, vocab_size, n_embd, n_head, n_layer);

    let t0 = Instant::now();
    let graph = compile(&program);
    let compile_time = t0.elapsed();
    eprintln!("DSL compile:  {:.0}ms ({} nodes)", compile_time.as_secs_f64() * 1000.0, graph.nodes.len());

    // Load inputs
    let tensors_meta = manifest["tensors"].as_array().unwrap();
    let input_nodes: Vec<(String, Vec<usize>)> = graph.nodes.iter()
        .filter_map(|n| {
            if let Op::Input { name } = &n.op {
                Some((name.clone(), n.shape.iter().map(|d| d.as_usize().unwrap()).collect()))
            } else { None }
        }).collect();

    let mut inputs = HashMap::new();
    // Tokens
    inputs.insert(input_nodes[0].0.clone(),
        ArrayD::from_shape_vec(vec![1, seq_len], vec![15496.0f32, 11.0, 995.0]).unwrap());

    for (idx, (name, shape)) in input_nodes.iter().enumerate().skip(1) {
        let t = &tensors_meta[idx - 1];
        let offset = t["offset"].as_u64().unwrap() as usize;
        let n_elements = t["n_elements"].as_u64().unwrap() as usize;
        let manifest_shape: Vec<usize> = t["shape"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        let bytes = &weights_bin[offset..offset + n_elements * 4];
        let data: Vec<f32> = bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let tensor = ArrayD::from_shape_vec(manifest_shape, data).unwrap();

        if t["name"].as_str() == Some("wpe") && tensor.shape()[0] > shape[0] {
            let sliced = tensor.slice(ndarray::s![..shape[0], ..]).to_owned();
            inputs.insert(name.clone(), sliced.into_dyn());
        } else {
            inputs.insert(name.clone(), tensor);
        }
    }

    // Run 3 times and report
    for i in 0..3 {
        let t0 = Instant::now();
        let results = eval_with_inputs(&graph, &inputs);
        let elapsed = t0.elapsed();
        let logits = results.last().unwrap();
        let top = logits.slice(ndarray::s![0, seq_len - 1, ..])
            .iter().copied().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        eprintln!("Run {}: {:.1}s (top token: {} logit={:.1})", i, elapsed.as_secs_f64(), top.0, top.1);
    }
}
