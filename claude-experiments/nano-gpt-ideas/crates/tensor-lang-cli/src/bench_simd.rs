//! Emit SIMD and scalar WASM for GPT-2, prepare inputs, then invoke the
//! Node.js benchmark runner that times both.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use tensor_lang_backend::wasm::WasmBackend;
use tensor_lang_graph::{compile, nanogpt, Op};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

fn main() {
    let root = project_root();
    let weights_dir = root.join("gpt2_weights");

    // --- Load weights ---
    eprintln!("Loading GPT-2 weights...");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(weights_dir.join("manifest.json"))
            .expect("Run `python3 export_gpt2.py` first"),
    )
    .unwrap();

    let config = &manifest["config"];
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let n_embd = config["n_embd"].as_u64().unwrap() as usize;
    let n_head = config["n_head"].as_u64().unwrap() as usize;
    let n_layer = config["n_layer"].as_u64().unwrap() as usize;

    let weights_bin = std::fs::read(weights_dir.join("weights.bin")).unwrap();
    let tensors_meta = manifest["tensors"].as_array().unwrap();

    let mut weights: Vec<Vec<f32>> = Vec::new();
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

    eprintln!(
        "  vocab={vocab_size}, d={n_embd}, heads={n_head}, layers={n_layer}"
    );

    // --- Parse args ---
    let args: Vec<String> = std::env::args().collect();
    let seq_len: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(8);
    let warmup: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);

    eprintln!("Benchmark: T={seq_len}, warmup={warmup}, iters={iters}");

    // --- Compile graph with symbolic T ---
    eprintln!("Compiling graph...");
    let t0 = Instant::now();
    let program = nanogpt::generate_nanogpt_program_symbolic(
        1, vocab_size, n_embd, n_head, n_layer,
    );
    let graph = compile(&program);
    eprintln!("  {} nodes, compiled in {:.1}s", graph.nodes.len(), t0.elapsed().as_secs_f64());

    // --- Emit both WASM variants ---
    let tmp = std::env::temp_dir();

    eprintln!("Emitting SIMD WASM...");
    let t1 = Instant::now();
    let simd_backend = WasmBackend { use_simd: true };
    let simd_wasm = simd_backend.emit_fused(&graph);
    let simd_path = tmp.join("gpt2_simd.wasm");
    std::fs::write(&simd_path, &simd_wasm).unwrap();
    eprintln!("  {} bytes, {:.1}s", simd_wasm.len(), t1.elapsed().as_secs_f64());

    eprintln!("Emitting scalar WASM...");
    let t2 = Instant::now();
    let scalar_backend = WasmBackend { use_simd: false };
    let scalar_wasm = scalar_backend.emit_fused(&graph);
    let scalar_path = tmp.join("gpt2_scalar.wasm");
    std::fs::write(&scalar_path, &scalar_wasm).unwrap();
    eprintln!("  {} bytes, {:.1}s", scalar_wasm.len(), t2.elapsed().as_secs_f64());

    // --- Prepare inputs ---
    let n_inputs = graph.nodes.iter()
        .filter(|n| matches!(&n.op, Op::Input { .. }))
        .count();

    // Token input: dummy sequence of length seq_len
    let token_input: Vec<f32> = (0..seq_len).map(|i| (464 + i) as f32).collect(); // "The" token ids

    // wpe: first seq_len rows
    let wpe_data = &weights[1];
    let wpe_slice = &wpe_data[..seq_len * n_embd];

    // Causal mask
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = -1_000_000.0;
            }
        }
    }

    // Assemble inputs in graph order
    let mut flat_inputs: Vec<&[f32]> = vec![&[]; n_inputs];
    flat_inputs[0] = &token_input;
    flat_inputs[1] = &weights[0]; // wte
    flat_inputs[2] = wpe_slice;   // wpe (sliced)
    flat_inputs[3] = &mask;       // attn_mask

    let mut wi = 2; // skip wte, wpe already placed
    for idx in 4..n_inputs {
        flat_inputs[idx] = &weights[wi];
        wi += 1;
    }

    // Write inputs binary
    let inputs_bin_path = tmp.join("gpt2_bench_inputs.bin");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&inputs_bin_path).unwrap());
        for arr in &flat_inputs {
            for v in *arr {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    // Write manifest
    let manifest_entries: Vec<String> = flat_inputs
        .iter()
        .map(|arr| format!("{{\"n_elements\":{}}}", arr.len()))
        .collect();
    let output_size = seq_len * vocab_size;
    let manifest_json = format!(
        "{{\"dim_params\":[{}],\"inputs\":[{}],\"output_size\":{}}}",
        seq_len,
        manifest_entries.join(","),
        output_size
    );
    let manifest_path = tmp.join("gpt2_bench_manifest.json");
    std::fs::write(&manifest_path, &manifest_json).unwrap();

    // --- Run Node.js benchmark ---
    let bench_script = root.join("bench_simd.mjs");
    eprintln!("\nRunning benchmark via Node.js...\n");

    let output = Command::new("node")
        .args([
            "--max-old-space-size=8192",
            bench_script.to_str().unwrap(),
            simd_path.to_str().unwrap(),
            scalar_path.to_str().unwrap(),
            inputs_bin_path.to_str().unwrap(),
            manifest_path.to_str().unwrap(),
            &warmup.to_string(),
            &iters.to_string(),
        ])
        .current_dir(&root)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .expect("failed to run node");

    if !output.success() {
        eprintln!("Benchmark failed!");
        std::process::exit(1);
    }

    // Cleanup
    let _ = std::fs::remove_file(&simd_path);
    let _ = std::fs::remove_file(&scalar_path);
    let _ = std::fs::remove_file(&inputs_bin_path);
    let _ = std::fs::remove_file(&manifest_path);
}
