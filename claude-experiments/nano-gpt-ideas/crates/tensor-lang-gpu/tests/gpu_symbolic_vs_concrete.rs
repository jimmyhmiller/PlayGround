//! Compare symbolic vs concrete path to isolate divergence source.

use tensor_lang_backend::runtime::WasmRuntime;
use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_gpu::plan;
use tensor_lang_gpu::runtime::GpuRuntime;
use tensor_lang_graph::{nanogpt, Op};

use std::path::PathBuf;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf()
}

fn load_weights() -> Option<Vec<Vec<f32>>> {
    let weights_dir = project_root().join("gpt2_weights");
    if !weights_dir.join("manifest.json").exists() { return None; }
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(weights_dir.join("manifest.json")).unwrap()
    ).unwrap();
    let bin = std::fs::read(weights_dir.join("weights.bin")).unwrap();
    let mut weights = Vec::new();
    for t in manifest["tensors"].as_array().unwrap() {
        let off = t["offset"].as_u64().unwrap() as usize;
        let n = t["n_elements"].as_u64().unwrap() as usize;
        weights.push(bin[off..off+n*4].chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect());
    }
    Some(weights)
}

fn build_inputs(graph: &tensor_lang_graph::Graph, weights: &[Vec<f32>], seq_len: usize) -> Vec<Vec<f32>> {
    let n_inputs = graph.nodes.iter().filter(|n| matches!(&n.op, Op::Input { .. })).count();
    let token_input: Vec<f32> = (0..seq_len).map(|i| (464 + i) as f32).collect();
    let wpe_slice = weights[1][..seq_len * 768].to_vec();
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len { for j in 0..seq_len { if j > i { mask[i * seq_len + j] = -1_000_000.0; } } }
    let mut all: Vec<Vec<f32>> = vec![vec![]; n_inputs];
    all[0] = token_input;
    all[1] = weights[0].clone();
    all[2] = wpe_slice;
    all[3] = mask;
    let mut wi = 2;
    for idx in 4..n_inputs { all[idx] = weights[wi].clone(); wi += 1; }
    all
}

#[test]
fn test_symbolic_vs_concrete_wasm() {
    let Some(weights) = load_weights() else { eprintln!("Skipping"); return; };
    let seq_len = 4;

    // Concrete
    let graph_c = nanogpt::compile_gpt2(1, Some(seq_len), 50257, 768, 12, 1);
    let inputs_c = build_inputs(&graph_c, &weights, seq_len);
    let slices_c: Vec<&[f32]> = inputs_c.iter().map(|v| v.as_slice()).collect();
    let out_size = seq_len * 50257;
    let wasm_c = WasmBackend::default().emit_fused(&graph_c);
    let mut rt_c = WasmRuntime::new(&wasm_c).unwrap();
    let out_c = rt_c.run(&slices_c, out_size);

    // Symbolic
    let graph_s = nanogpt::compile_gpt2(1, None, 50257, 768, 12, 1);
    let inputs_s = build_inputs(&graph_s, &weights, seq_len);
    let slices_s: Vec<&[f32]> = inputs_s.iter().map(|v| v.as_slice()).collect();
    let wasm_s = WasmBackend::default().emit_fused(&graph_s);
    let mut rt_s = WasmRuntime::new(&wasm_s).unwrap();
    let out_s = rt_s.run_with_dim_params(&[seq_len as i32], &slices_s, out_size);

    let max_diff: f32 = out_c.iter().zip(out_s.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("WASM concrete vs symbolic: max_diff={max_diff:.6}");
    assert!(max_diff < 0.001, "WASM concrete vs symbolic diverges: {max_diff}");
}

#[test]
fn test_symbolic_vs_concrete_gpu() {
    let Some(weights) = load_weights() else { eprintln!("Skipping"); return; };
    let seq_len = 4;

    // Concrete
    let graph_c = nanogpt::compile_gpt2(1, Some(seq_len), 50257, 768, 12, 1);
    let inputs_c = build_inputs(&graph_c, &weights, seq_len);
    let slices_c: Vec<&[f32]> = inputs_c.iter().map(|v| v.as_slice()).collect();
    let out_size = seq_len * 50257;
    let plan_c = plan::build_plan(&graph_c);
    let gpu_rt = GpuRuntime::new();
    let out_c = gpu_rt.run(&plan_c, &slices_c, out_size);

    // Symbolic
    let graph_s = nanogpt::compile_gpt2(1, None, 50257, 768, 12, 1);
    let inputs_s = build_inputs(&graph_s, &weights, seq_len);
    let slices_s: Vec<&[f32]> = inputs_s.iter().map(|v| v.as_slice()).collect();
    let plan_s = plan::build_plan(&graph_s);
    let out_s = gpu_rt.run_with_dim_params(&plan_s, &[seq_len as u32], &slices_s, out_size);

    let max_diff: f32 = out_c.iter().zip(out_s.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("GPU concrete vs symbolic: max_diff={max_diff:.6}");
    assert!(max_diff < 0.001, "GPU concrete vs symbolic diverges: {max_diff}");
}
