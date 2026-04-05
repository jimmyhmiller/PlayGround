//! Dump WASM SIMD logits for verification against C baseline.
//! Usage: cargo run --release -p tensor-lang-backend --example dump_wasm_logits -- [seq_len] [token_file]

use std::io::Write;
use std::path::PathBuf;

use tensor_lang_backend::runtime::WasmRuntime;
use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_graph::{compile, nanogpt, Op};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seq_len: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(3);
    let token_file = args.get(2).map(|s| s.as_str());

    let root = project_root();
    let weights_dir = root.join("gpt2_weights");

    // Load config
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(weights_dir.join("manifest.json")).unwrap(),
    ).unwrap();
    let config = &manifest["config"];
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let n_embd = config["n_embd"].as_u64().unwrap() as usize;
    let n_head = config["n_head"].as_u64().unwrap() as usize;
    let n_layer = config["n_layer"].as_u64().unwrap() as usize;

    // Load weights
    let weights_bin = std::fs::read(weights_dir.join("weights.bin")).unwrap();
    let tensors_meta = manifest["tensors"].as_array().unwrap();
    let mut weights: Vec<Vec<f32>> = Vec::new();
    for t in tensors_meta {
        let offset = t["offset"].as_u64().unwrap() as usize;
        let n_elements = t["n_elements"].as_u64().unwrap() as usize;
        let bytes = &weights_bin[offset..offset + n_elements * 4];
        let data: Vec<f32> = bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        weights.push(data);
    }

    // Load tokens
    let token_input: Vec<f32> = if let Some(path) = token_file {
        let bytes = std::fs::read(path).unwrap();
        bytes.chunks_exact(4)
            .take(seq_len)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        (0..seq_len).map(|i| (464 + i) as f32).collect()
    };
    eprintln!("Tokens: {:?}", token_input);

    // Compile graph (symbolic T)
    let program = nanogpt::generate_nanogpt_program_symbolic(1, vocab_size, n_embd, n_head, n_layer);
    let graph = compile(&program);
    let n_inputs = graph.nodes.iter().filter(|n| matches!(&n.op, Op::Input { .. })).count();

    // Emit WASM SIMD
    let wasm = WasmBackend { use_simd: true }.emit_fused(&graph);
    let mut rt = WasmRuntime::new(&wasm).unwrap();

    // Prepare inputs
    let wpe_data = &weights[1];
    let wpe_slice = &wpe_data[..seq_len * n_embd];

    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i { mask[i * seq_len + j] = -1_000_000.0; }
        }
    }

    let mut flat_inputs: Vec<&[f32]> = vec![&[]; n_inputs];
    flat_inputs[0] = &token_input;
    flat_inputs[1] = &weights[0]; // wte
    flat_inputs[2] = wpe_slice;   // wpe (sliced)
    flat_inputs[3] = &mask;       // attn_mask

    let mut wi = 2;
    for idx in 4..n_inputs {
        flat_inputs[idx] = &weights[wi];
        wi += 1;
    }

    let output_size = seq_len * vocab_size;
    let dim_params = &[seq_len as i32];

    // Run
    let out = rt.run_with_dim_params(dim_params, &flat_inputs, output_size);

    // Dump to file
    let dump_path = root.join("bench/wasm_logits.bin");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&dump_path).unwrap());
    for v in &out {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    eprintln!("Dumped {} floats to {}", out.len(), dump_path.display());

    // Print top-5 for last position
    let last = &out[(seq_len - 1) * vocab_size..];
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    eprintln!("Top-5:");
    for i in 0..5 {
        eprintln!("  {}: idx={} logit={:.4}", i + 1, indices[i], last[indices[i]]);
    }
}
