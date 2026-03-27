//! Test our nanoGPT implementation against real GPT-2 weights.
//!
//! Requires running `python3 export_gpt2.py` first to generate gpt2_weights/.

use std::collections::HashMap;
use std::path::Path;

use ndarray::ArrayD;
use tensor_lang_graph::{compile, nanogpt, Op};
use tensor_lang_test_oracle::eval_with_inputs;

fn project_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

/// Load a manifest.json and weights.bin, returning tensors keyed by index.
fn load_weights() -> (serde_json::Value, Vec<u8>) {
    let root = project_root();
    let manifest_path = root.join("gpt2_weights/manifest.json");
    let weights_path = root.join("gpt2_weights/weights.bin");

    if !manifest_path.exists() {
        panic!("gpt2_weights/manifest.json not found. Run: python3 export_gpt2.py");
    }

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path).unwrap()
    ).unwrap();

    let weights = std::fs::read(&weights_path).unwrap();

    (manifest, weights)
}

/// Extract a tensor from the binary weights blob.
fn extract_tensor(weights: &[u8], tensor_info: &serde_json::Value) -> ArrayD<f32> {
    let offset = tensor_info["offset"].as_u64().unwrap() as usize;
    let n_elements = tensor_info["n_elements"].as_u64().unwrap() as usize;
    let shape: Vec<usize> = tensor_info["shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();

    let bytes = &weights[offset..offset + n_elements * 4];
    let data: Vec<f32> = bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    ArrayD::from_shape_vec(shape, data).unwrap()
}

#[test]
fn test_gpt2_real_weights() {
    let root = project_root();
    if !root.join("gpt2_weights/manifest.json").exists() {
        eprintln!("Skipping test_gpt2_real_weights: run `python3 export_gpt2.py` first");
        return;
    }

    let (manifest, weights) = load_weights();
    let config = &manifest["config"];
    let tensors = manifest["tensors"].as_array().unwrap();

    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let n_embd = config["n_embd"].as_u64().unwrap() as usize;
    let n_head = config["n_head"].as_u64().unwrap() as usize;
    let n_layer = config["n_layer"].as_u64().unwrap() as usize;
    let seq_len = config["seq_len"].as_u64().unwrap() as usize;

    println!("Config: vocab={vocab_size}, d={n_embd}, heads={n_head}, layers={n_layer}, T={seq_len}");

    // Generate the DSL program
    let program = nanogpt::generate_nanogpt_program(1, seq_len, vocab_size, n_embd, n_head, n_layer);

    println!("Compiling DSL program...");
    let graph = compile(&program);

    // Collect input nodes in order
    let input_nodes: Vec<(usize, String, Vec<usize>)> = graph.nodes.iter().enumerate()
        .filter_map(|(i, n)| {
            if let Op::Input { name } = &n.op {
                let shape: Vec<usize> = n.shape.iter().map(|d| d.as_usize().unwrap()).collect();
                Some((i, name.clone(), shape))
            } else {
                None
            }
        })
        .collect();

    println!("Graph has {} nodes, {} inputs", graph.nodes.len(), input_nodes.len());

    // Load reference input tokens
    let input_bytes = std::fs::read(root.join("gpt2_weights/reference_input.bin")).unwrap();
    let tokens: Vec<f32> = input_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("Tokens: {:?}", tokens);

    // Build input map
    let mut inputs = HashMap::new();

    // First input is tokens
    let (_, ref token_name, ref token_shape) = input_nodes[0];
    inputs.insert(token_name.clone(),
        ArrayD::from_shape_vec(token_shape.clone(), tokens).unwrap());

    // Remaining inputs are weights from the manifest, in order
    for (input_idx, (_, name, shape)) in input_nodes.iter().enumerate().skip(1) {
        // Find matching tensor in manifest (they're in the same order)
        let tensor_info = &tensors[input_idx - 1]; // -1 because tokens aren't in the manifest
        let tensor = extract_tensor(&weights, tensor_info);

        // wpe may need slicing from [1024, n_embd] to [seq_len, n_embd]
        if tensor_info["name"].as_str() == Some("wpe") && tensor.shape()[0] > shape[0] {
            let sliced = tensor.slice(ndarray::s![..shape[0], ..]).to_owned();
            inputs.insert(name.clone(), sliced.into_dyn());
        } else {
            assert_eq!(tensor.shape(), shape.as_slice(),
                "Shape mismatch for {}: manifest {:?} vs program {:?}",
                name, tensor.shape(), shape);
            inputs.insert(name.clone(), tensor);
        }
    }

    println!("Running forward pass through oracle...");
    let results = eval_with_inputs(&graph, &inputs);

    // Find first node with NaN to diagnose
    for (i, (result, node)) in results.iter().zip(graph.nodes.iter()).enumerate() {
        let has_nan = result.iter().any(|v| v.is_nan());
        let has_inf = result.iter().any(|v| v.is_infinite());
        if has_nan || has_inf {
            println!("FIRST BAD NODE: {} (op={:?}, shape={:?})", i, node.op, node.shape);
            println!("  inputs: {:?}", node.inputs);
            for &inp_id in &node.inputs {
                let inp = &results[inp_id.0];
                let min = inp.iter().copied().fold(f32::INFINITY, f32::min);
                let max = inp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let inp_nan = inp.iter().any(|v| v.is_nan());
                println!("  input {}: shape={:?}, min={}, max={}, has_nan={}",
                    inp_id.0, graph.nodes[inp_id.0].shape, min, max, inp_nan);
            }
            let min = result.iter().copied().fold(f32::INFINITY, f32::min);
            let max = result.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!("  output: min={}, max={}", min, max);
            break;
        }
    }

    let logits = results.last().unwrap();

    println!("Output logits shape: {:?}", logits.shape());
    assert_eq!(logits.shape(), &[1, seq_len, vocab_size]);

    // Load reference output
    let ref_bytes = std::fs::read(root.join("gpt2_weights/reference_output.bin")).unwrap();
    let ref_logits: Vec<f32> = ref_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let our_logits: Vec<f32> = logits.iter().copied().collect();

    // Check for NaN/Inf
    let n_nan = our_logits.iter().filter(|v| v.is_nan()).count();
    let n_inf = our_logits.iter().filter(|v| v.is_infinite()).count();
    let n_finite = our_logits.iter().filter(|v| v.is_finite()).count();
    println!("Our logits: {} finite, {} NaN, {} Inf", n_finite, n_nan, n_inf);

    if n_nan > 0 || n_inf > 0 {
        // Print some sample values
        for (i, v) in our_logits.iter().enumerate().take(10) {
            println!("  logit[{i}] = {v}");
        }
    }

    // Compare: check that the top predicted tokens match
    println!("\nLast position predictions:");
    let last_start = (seq_len - 1) * vocab_size;

    // Get top-5 from reference
    let ref_last = &ref_logits[last_start..last_start + vocab_size];
    let mut ref_indices: Vec<usize> = (0..vocab_size).collect();
    ref_indices.sort_by(|&a, &b| ref_last[b].partial_cmp(&ref_last[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Get top-5 from ours
    let our_last = &our_logits[last_start..last_start + vocab_size];
    let mut our_indices: Vec<usize> = (0..vocab_size).collect();
    our_indices.sort_by(|&a, &b| our_last[b].partial_cmp(&our_last[a]).unwrap_or(std::cmp::Ordering::Equal));

    println!("Reference top-5: {:?}", &ref_indices[..5]);
    println!("Our top-5:       {:?}", &our_indices[..5]);

    for i in 0..5 {
        println!("  ref[{}]: idx={} logit={:.3}  |  ours[{}]: idx={} logit={:.3}",
            i, ref_indices[i], ref_last[ref_indices[i]],
            i, our_indices[i], our_last[our_indices[i]]);
    }

    // The top-1 prediction should match
    assert_eq!(ref_indices[0], our_indices[0],
        "Top-1 prediction mismatch! ref={} ours={}", ref_indices[0], our_indices[0]);

    // Check that top-5 overlap significantly
    let ref_top5: std::collections::HashSet<usize> = ref_indices[..5].iter().copied().collect();
    let our_top5: std::collections::HashSet<usize> = our_indices[..5].iter().copied().collect();
    let overlap = ref_top5.intersection(&our_top5).count();
    println!("Top-5 overlap: {}/5", overlap);
    assert!(overlap >= 3, "Top-5 overlap too low: {}/5", overlap);

    // Also check mean absolute error of logits
    let mae: f32 = ref_logits.iter().zip(our_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / ref_logits.len() as f32;
    println!("Mean absolute error: {:.6}", mae);

    // For f32 accumulation over 768-dim vectors through 12 layers,
    // some error is expected. Be generous.
    assert!(mae < 1.0, "MAE too high: {}", mae);

    println!("\nGPT-2 verification PASSED!");
}
