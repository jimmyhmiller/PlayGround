//! Compare GPU directly against the ndarray oracle (ground truth).
//! This bypasses WASM entirely.

use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::ArrayD;
use tensor_lang_gpu::plan;
use tensor_lang_gpu::runtime::GpuRuntime;
use tensor_lang_graph::{nanogpt, Op};
use tensor_lang_test_oracle::eval_with_inputs;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn load_weights() -> Option<Vec<Vec<f32>>> {
    let weights_dir = project_root().join("gpt2_weights");
    let manifest_path = weights_dir.join("manifest.json");
    if !manifest_path.exists() {
        return None;
    }
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let weights_bin = std::fs::read(weights_dir.join("weights.bin")).unwrap();
    let tensors_meta = manifest["tensors"].as_array().unwrap();
    let mut weights = Vec::new();
    for t in tensors_meta {
        let offset = t["offset"].as_u64().unwrap() as usize;
        let n_elements = t["n_elements"].as_u64().unwrap() as usize;
        let bytes = &weights_bin[offset..offset + n_elements * 4];
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        weights.push(data);
    }
    Some(weights)
}

#[test]
fn test_gpu_vs_oracle_gpt2_1layer() {
    let Some(weights) = load_weights() else {
        eprintln!("Skipping: GPT-2 weights not found");
        return;
    };

    let seq_len = 4;
    let vocab_size = 50257;
    let n_embd = 768;
    let n_head = 12;

    let graph = nanogpt::compile_gpt2(1, Some(seq_len), vocab_size, n_embd, n_head, 1);

    let token_input: Vec<f32> = (0..seq_len).map(|i| (464 + i) as f32).collect();
    let wpe_slice = weights[1][..seq_len * n_embd].to_vec();
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = -1_000_000.0;
            }
        }
    }

    // Build inputs by iterating graph nodes in order
    let mut oracle_inputs: HashMap<String, ArrayD<f32>> = HashMap::new();
    let mut flat_inputs_data: Vec<Vec<f32>> = Vec::new();
    let mut wi = 2; // weights[0]=wte, weights[1]=wpe, layer weights start at [2]

    for node in &graph.nodes {
        if let Op::Input { name } = &node.op {
            let shape: Vec<usize> = node.shape.iter()
                .map(|d| d.as_usize().unwrap())
                .collect();
            let expected_size: usize = shape.iter().product();

            let data = if flat_inputs_data.is_empty() {
                // First input: tokens
                token_input.clone()
            } else if flat_inputs_data.len() == 1 {
                // Second input: wte
                weights[0].clone()
            } else if flat_inputs_data.len() == 2 {
                // Third input: wpe (sliced)
                wpe_slice.clone()
            } else if flat_inputs_data.len() == 3 {
                // Fourth input: attn_mask
                mask.clone()
            } else {
                // Remaining: layer weights
                let d = weights[wi].clone();
                wi += 1;
                d
            };

            assert_eq!(
                data.len(),
                expected_size,
                "input {name}: data len {} != expected {}",
                data.len(),
                expected_size
            );

            let arr = ArrayD::from_shape_vec(shape, data.clone()).unwrap();
            oracle_inputs.insert(name.clone(), arr);
            flat_inputs_data.push(data);
        }
    }

    let input_slices: Vec<&[f32]> = flat_inputs_data.iter().map(|v| v.as_slice()).collect();

    // Oracle
    let oracle_results = eval_with_inputs(&graph, &oracle_inputs);
    let oracle_output = oracle_results.last().unwrap();
    let oracle_flat: Vec<f32> = oracle_output.iter().copied().collect();

    // GPU
    let gpu_plan = plan::build_plan(&graph);
    let gpu_rt = GpuRuntime::new();
    let output_size = oracle_flat.len();
    let gpu_out = gpu_rt.run(&gpu_plan, &input_slices, output_size);

    // Compare
    let max_diff: f32 = oracle_flat
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = oracle_flat
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / output_size as f32;

    eprintln!("GPU vs Oracle (1 layer, real weights): max_diff={max_diff:.6}, mean_diff={mean_diff:.8}");

    assert!(
        max_diff < 0.01,
        "GPU vs Oracle: max_diff={max_diff} exceeds 0.01"
    );
}
