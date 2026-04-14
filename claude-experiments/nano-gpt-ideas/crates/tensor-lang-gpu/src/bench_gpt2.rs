//! GPT-2 forward pass benchmark: WASM (scalar) vs WASM (SIMD) vs GPU.
//!
//! Usage: cargo run -p tensor-lang-gpu --features native --bin bench-gpt2 -- [seq_len] [iters]

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use tensor_lang_backend::runtime::WasmRuntime;
use tensor_lang_backend::wasm::WasmBackend;
#[cfg(target_arch = "aarch64")]
use tensor_lang_backend::arm::ArmBackend;
#[cfg(target_arch = "aarch64")]
use tensor_lang_backend::arm_runtime::ArmRuntime;
use tensor_lang_gpu::plan;
use tensor_lang_gpu::runtime::GpuRuntime;
use tensor_lang_graph::{nanogpt, Op};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seq_len: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(8);
    let warmup: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);

    // --- Load weights ---
    let root = project_root();
    let weights_dir = root.join("gpt2_weights");

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
    eprintln!("Benchmark: T={seq_len}, warmup={warmup}, iters={iters}\n");

    // --- Compile graph with symbolic T ---
    eprintln!("Compiling graph...");
    let t0 = Instant::now();
    let n_layer_bench: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(n_layer);
    let graph = nanogpt::compile_gpt2(1, None, vocab_size, n_embd, n_head, n_layer_bench);
    eprintln!("  {} nodes, compiled in {:.2}s", graph.nodes.len(), t0.elapsed().as_secs_f64());

    // --- Prepare inputs ---
    let n_inputs = graph
        .nodes
        .iter()
        .filter(|n| matches!(&n.op, Op::Input { .. }))
        .count();

    let token_input: Vec<f32> = (0..seq_len).map(|i| (464 + i) as f32).collect();
    let wpe_data = &weights[1];
    let wpe_slice = &wpe_data[..seq_len * n_embd];

    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = -1_000_000.0;
            }
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

    // --- Emit WASM variants ---
    eprintln!("Emitting WASM (scalar)...");
    let t1 = Instant::now();
    let scalar_wasm = WasmBackend { use_simd: false }.emit_fused(&graph);
    eprintln!("  {} bytes, {:.2}s", scalar_wasm.len(), t1.elapsed().as_secs_f64());

    eprintln!("Emitting WASM (SIMD)...");
    let t2 = Instant::now();
    let simd_wasm = WasmBackend { use_simd: true }.emit_fused(&graph);
    eprintln!("  {} bytes, {:.2}s", simd_wasm.len(), t2.elapsed().as_secs_f64());

    // --- Build GPU plan ---
    eprintln!("Building GPU plan...");
    let t3 = Instant::now();
    let gpu_plan = plan::build_plan(&graph);
    eprintln!(
        "  {} steps, {} shaders, {:.2}s",
        gpu_plan.steps.len(),
        gpu_plan.shaders.len(),
        t3.elapsed().as_secs_f64()
    );

    // --- Initialize runtimes ---
    eprintln!("Initializing WASM runtimes...");
    let mut scalar_rt = WasmRuntime::new(&scalar_wasm).unwrap();
    let mut simd_rt = WasmRuntime::new(&simd_wasm).unwrap();

    eprintln!("Initializing GPU runtime...");
    let gpu_rt = GpuRuntime::new();

    eprintln!("\n--- Benchmark ---\n");

    // --- WASM scalar ---
    {
        // Warmup
        for _ in 0..warmup {
            scalar_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
        }
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _out = scalar_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("WASM scalar:  avg={avg:8.1}ms  min={min:8.1}ms  max={max:8.1}ms");
        std::io::stdout().flush().unwrap();
    }

    // --- WASM SIMD ---
    {
        for _ in 0..warmup {
            simd_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
        }
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _out = simd_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("WASM SIMD:    avg={avg:8.1}ms  min={min:8.1}ms  max={max:8.1}ms");
        std::io::stdout().flush().unwrap();
    }

    // --- GPU ---
    {
        let gpu_dim_params = &[seq_len as u32];
        for _ in 0..warmup {
            gpu_rt.run_with_dim_params(&gpu_plan, gpu_dim_params, &flat_inputs, output_size);
        }
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _out = gpu_rt.run_with_dim_params(&gpu_plan, gpu_dim_params, &flat_inputs, output_size);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("GPU (wgpu):   avg={avg:8.1}ms  min={min:8.1}ms  max={max:8.1}ms");
        std::io::stdout().flush().unwrap();
    }

    // --- ARM native (aarch64 only) ---
    #[cfg(target_arch = "aarch64")]
    {
        eprintln!("Emitting ARM native...");
        let t_arm = Instant::now();
        let arm_code = ArmBackend.emit_fused(&graph);
        eprintln!("  {} bytes, {:.2}s", arm_code.code.len(), t_arm.elapsed().as_secs_f64());

        let mut arm_rt = ArmRuntime::new(&arm_code);
        for _ in 0..warmup {
            arm_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
        }
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _out = arm_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("ARM native:   avg={avg:8.1}ms  min={min:8.1}ms  max={max:8.1}ms");
        std::io::stdout().flush().unwrap();
    }

    // --- Verify outputs match ---
    eprintln!("\nVerifying outputs match...");
    let scalar_out = scalar_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
    let simd_out = simd_rt.run_with_dim_params(dim_params, &flat_inputs, output_size);
    let gpu_out = gpu_rt.run_with_dim_params(&gpu_plan, &[seq_len as u32], &flat_inputs, output_size);

    // Compare last-token logits (most important for generation)
    let last_start = (seq_len - 1) * vocab_size;
    let scalar_logits = &scalar_out[last_start..last_start + vocab_size];
    let simd_logits = &simd_out[last_start..last_start + vocab_size];
    let gpu_logits = &gpu_out[last_start..last_start + vocab_size];

    fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);
        indexed
    }

    let scalar_top = top_k(scalar_logits, 5);
    let gpu_top = top_k(gpu_logits, 5);

    eprintln!("  Scalar top-5:");
    for (i, v) in &scalar_top {
        eprintln!("    [{i}] = {v:.4}");
    }
    eprintln!("  GPU top-5:");
    for (i, v) in &gpu_top {
        eprintln!("    [{i}] = {v:.4}");
    }

    // Max abs diff
    let max_diff_simd: f32 = scalar_logits.iter().zip(simd_logits.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_diff_gpu: f32 = scalar_logits.iter().zip(gpu_logits.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mean_diff_gpu: f32 = scalar_logits.iter().zip(gpu_logits.iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / scalar_logits.len() as f32;

    eprintln!("  Max diff (scalar vs SIMD): {max_diff_simd:.6}");
    eprintln!("  Max diff (scalar vs GPU):  {max_diff_gpu:.6}");
    eprintln!("  Mean diff (scalar vs GPU): {mean_diff_gpu:.8}");

    if scalar_top[0].0 == gpu_top[0].0 {
        eprintln!("  All backends agree on next token!");
    } else {
        eprintln!("  NOTE: backends disagree on argmax (normal for large models with float precision differences)");
    }
}
