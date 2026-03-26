use std::process::Command;
use std::time::Instant;

use tensor_lang_backend::assemblyscript::AssemblyScriptBackend;
use tensor_lang_graph::{compile, nanogpt, Op};

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

fn parse_json_f32_array(s: &str) -> Vec<f32> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    if s.is_empty() { return vec![]; }
    s.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect()
}

#[test]
fn bench_as_full_gpt2() {
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
    let seq_len = 3;

    eprintln!("Generating DSL program for full GPT-2...");
    let program = nanogpt::generate_nanogpt_program(1, seq_len, vocab_size, n_embd, n_head, n_layer);

    eprintln!("Compiling DSL to graph...");
    let t0 = Instant::now();
    let graph = compile(&program);
    eprintln!("  {} nodes in {:.0}ms", graph.nodes.len(), t0.elapsed().as_secs_f64() * 1000.0);

    // Emit fused AssemblyScript
    eprintln!("Emitting fused AssemblyScript...");
    let t0 = Instant::now();
    let backend = AssemblyScriptBackend;
    let as_code = backend.emit_fused(&graph);
    eprintln!("  {} bytes, {} lines in {:.0}ms",
        as_code.len(), as_code.lines().count(), t0.elapsed().as_secs_f64() * 1000.0);

    // Write AS source
    let tmp_dir = std::env::temp_dir();
    let src_path = tmp_dir.join("gpt2_full_bench.ts");
    let wasm_path = tmp_dir.join("gpt2_full_bench.wasm");
    std::fs::write(&src_path, &as_code).unwrap();

    // Compile with asc
    eprintln!("Compiling AssemblyScript to WASM with asc...");
    let t0 = Instant::now();
    let asc_output = Command::new("npx")
        .args(["asc", src_path.to_str().unwrap(),
               "--outFile", wasm_path.to_str().unwrap(),
               "--exportRuntime", "--optimize",
               "--initialMemory", "2048",
               "--maximumMemory", "65536"])
        .current_dir(&root)
        .output()
        .expect("failed to run asc");
    eprintln!("  asc took {:.1}s", t0.elapsed().as_secs_f64());

    if !asc_output.status.success() {
        let stderr = String::from_utf8_lossy(&asc_output.stderr);
        let stdout = String::from_utf8_lossy(&asc_output.stdout);
        eprintln!("asc FAILED:");
        eprintln!("stderr: {}", &stderr[..stderr.len().min(2000)]);
        eprintln!("stdout: {}", &stdout[..stdout.len().min(2000)]);
        panic!("asc compilation failed");
    }

    eprintln!("  WASM size: {} bytes", std::fs::metadata(&wasm_path).unwrap().len());

    // Build inputs
    let tensors_meta = manifest["tensors"].as_array().unwrap();
    let input_nodes: Vec<(String, Vec<usize>)> = graph.nodes.iter()
        .filter_map(|n| {
            if let Op::Input { name } = &n.op {
                Some((name.clone(), n.shape.clone()))
            } else { None }
        }).collect();

    let mut flat_inputs: Vec<Vec<f32>> = Vec::new();

    // Token input
    flat_inputs.push(vec![15496.0f32, 11.0, 995.0]);

    // Weight inputs
    for (idx, (_, shape)) in input_nodes.iter().enumerate().skip(1) {
        let t = &tensors_meta[idx - 1];
        let offset = t["offset"].as_u64().unwrap() as usize;
        let n_elements = t["n_elements"].as_u64().unwrap() as usize;

        let bytes = &weights_bin[offset..offset + n_elements * 4];
        let data: Vec<f32> = bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        // Slice wpe if needed
        if t["name"].as_str() == Some("wpe") && data.len() / n_embd > shape[0] {
            flat_inputs.push(data[..shape[0] * n_embd].to_vec());
        } else {
            flat_inputs.push(data);
        }
    }

    // Debug: verify input sizes match graph expectations
    for (i, (name, shape)) in input_nodes.iter().enumerate() {
        let expected: usize = shape.iter().product();
        let actual = flat_inputs[i].len();
        if expected != actual {
            eprintln!("  INPUT MISMATCH: {} (input_{}) expected {} elements (shape {:?}), got {}",
                name, i, expected, shape, actual);
        }
    }

    // Write inputs as binary + manifest
    let inputs_bin_path = tmp_dir.join("gpt2_full_bench_inputs.bin");
    let inputs_manifest_path = tmp_dir.join("gpt2_full_bench_manifest.json");
    eprintln!("Writing binary inputs...");
    let t0 = Instant::now();
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&inputs_bin_path).unwrap());
        for arr in &flat_inputs {
            for v in arr {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }
    // Write manifest (just element counts)
    let manifest_entries: Vec<String> = flat_inputs.iter()
        .map(|arr| format!("{{\"n_elements\":{}}}", arr.len()))
        .collect();
    std::fs::write(&inputs_manifest_path, format!("[{}]", manifest_entries.join(","))).unwrap();
    eprintln!("  Wrote inputs in {:.1}s ({} bytes)",
        t0.elapsed().as_secs_f64(),
        std::fs::metadata(&inputs_bin_path).unwrap().len());

    // Run with node using binary runner
    eprintln!("Running WASM with node...");
    let runner = root.join("test_runner_bin.mjs");
    let t0 = Instant::now();
    let node_output = Command::new("node")
        .args(["--max-old-space-size=8192",
               runner.to_str().unwrap(),
               wasm_path.to_str().unwrap(),
               inputs_bin_path.to_str().unwrap(),
               inputs_manifest_path.to_str().unwrap()])
        .current_dir(&root)
        .output()
        .expect("failed to run node");
    let run_time = t0.elapsed();

    if !node_output.status.success() {
        let stderr = String::from_utf8_lossy(&node_output.stderr);
        eprintln!("Node FAILED:");
        eprintln!("{}", &stderr[..stderr.len().min(3000)]);
        panic!("node execution failed");
    }

    let stdout = String::from_utf8_lossy(&node_output.stdout);
    let as_output = parse_json_f32_array(stdout.trim());
    eprintln!("  WASM run took {:.1}s", run_time.as_secs_f64());
    eprintln!("  Output: {} elements", as_output.len());

    // Compare with reference
    let ref_bytes = std::fs::read(root.join("gpt2_weights/reference_output.bin")).unwrap();
    let ref_logits: Vec<f32> = ref_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Check top predictions for last position
    let last_start = (seq_len - 1) * vocab_size;
    let our_last = &as_output[last_start..last_start + vocab_size];
    let ref_last = &ref_logits[last_start..last_start + vocab_size];

    let mut our_indices: Vec<usize> = (0..vocab_size).collect();
    our_indices.sort_by(|&a, &b| our_last[b].partial_cmp(&our_last[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ref_indices: Vec<usize> = (0..vocab_size).collect();
    ref_indices.sort_by(|&a, &b| ref_last[b].partial_cmp(&ref_last[a]).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\nRef top-5:  {:?}", &ref_indices[..5]);
    eprintln!("WASM top-5: {:?}", &our_indices[..5]);

    assert_eq!(ref_indices[0], our_indices[0], "Top-1 mismatch!");
    eprintln!("\nFull GPT-2 via fused WASM: PASSED!");
}

// Unfused backend OOMs on full GPT-2 — the expand intermediates for matmul
// exceed WASM's 4GB memory limit. This test documents that; don't run by default.
#[test]
#[ignore]
fn bench_as_full_gpt2_unfused() {
    use tensor_lang_backend::Backend;
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
    let seq_len = 3;

    eprintln!("Generating DSL program for full GPT-2 (UNFUSED)...");
    let program = nanogpt::generate_nanogpt_program(1, seq_len, vocab_size, n_embd, n_head, n_layer);

    eprintln!("Compiling DSL to graph...");
    let t0 = Instant::now();
    let graph = compile(&program);
    eprintln!("  {} nodes in {:.0}ms", graph.nodes.len(), t0.elapsed().as_secs_f64() * 1000.0);

    // Emit UNFUSED AssemblyScript
    eprintln!("Emitting UNFUSED AssemblyScript...");
    let t0 = Instant::now();
    let backend = AssemblyScriptBackend;
    let as_code = backend.emit(&graph);
    eprintln!("  {} bytes, {} lines in {:.0}ms",
        as_code.len(), as_code.lines().count(), t0.elapsed().as_secs_f64() * 1000.0);

    // Write AS source
    let tmp_dir = std::env::temp_dir();
    let src_path = tmp_dir.join("gpt2_unfused_bench.ts");
    let wasm_path = tmp_dir.join("gpt2_unfused_bench.wasm");
    std::fs::write(&src_path, &as_code).unwrap();

    // Compile with asc
    eprintln!("Compiling AssemblyScript to WASM with asc...");
    let t0 = Instant::now();
    let asc_output = Command::new("npx")
        .args(["asc", src_path.to_str().unwrap(),
               "--outFile", wasm_path.to_str().unwrap(),
               "--exportRuntime", "--optimize",
               "--initialMemory", "2048",
               "--maximumMemory", "65536"])
        .current_dir(&root)
        .output()
        .expect("failed to run asc");
    eprintln!("  asc took {:.1}s", t0.elapsed().as_secs_f64());

    if !asc_output.status.success() {
        let stderr = String::from_utf8_lossy(&asc_output.stderr);
        let stdout = String::from_utf8_lossy(&asc_output.stdout);
        eprintln!("asc FAILED:");
        eprintln!("stderr: {}", &stderr[..stderr.len().min(2000)]);
        eprintln!("stdout: {}", &stdout[..stdout.len().min(2000)]);
        panic!("asc compilation failed");
    }

    eprintln!("  WASM size: {} bytes", std::fs::metadata(&wasm_path).unwrap().len());

    // Build inputs
    let tensors_meta = manifest["tensors"].as_array().unwrap();
    let input_nodes: Vec<(String, Vec<usize>)> = graph.nodes.iter()
        .filter_map(|n| {
            if let Op::Input { name } = &n.op {
                Some((name.clone(), n.shape.clone()))
            } else { None }
        }).collect();

    let mut flat_inputs: Vec<Vec<f32>> = Vec::new();
    flat_inputs.push(vec![15496.0f32, 11.0, 995.0]);

    for (idx, (_, shape)) in input_nodes.iter().enumerate().skip(1) {
        let t = &tensors_meta[idx - 1];
        let offset = t["offset"].as_u64().unwrap() as usize;
        let n_elements = t["n_elements"].as_u64().unwrap() as usize;
        let bytes = &weights_bin[offset..offset + n_elements * 4];
        let data: Vec<f32> = bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if t["name"].as_str() == Some("wpe") && data.len() / n_embd > shape[0] {
            flat_inputs.push(data[..shape[0] * n_embd].to_vec());
        } else {
            flat_inputs.push(data);
        }
    }

    // Write inputs as binary + manifest
    let inputs_bin_path = tmp_dir.join("gpt2_unfused_bench_inputs.bin");
    let inputs_manifest_path = tmp_dir.join("gpt2_unfused_bench_manifest.json");
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&inputs_bin_path).unwrap());
        for arr in &flat_inputs {
            for v in arr {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }
    let manifest_entries: Vec<String> = flat_inputs.iter()
        .map(|arr| format!("{{\"n_elements\":{}}}", arr.len()))
        .collect();
    std::fs::write(&inputs_manifest_path, format!("[{}]", manifest_entries.join(","))).unwrap();

    // Run with node
    eprintln!("Running WASM with node...");
    let runner = root.join("test_runner_bin.mjs");
    let t0 = Instant::now();
    let node_output = Command::new("node")
        .args(["--max-old-space-size=8192",
               runner.to_str().unwrap(),
               wasm_path.to_str().unwrap(),
               inputs_bin_path.to_str().unwrap(),
               inputs_manifest_path.to_str().unwrap()])
        .current_dir(&root)
        .output()
        .expect("failed to run node");
    let run_time = t0.elapsed();

    if !node_output.status.success() {
        let stderr = String::from_utf8_lossy(&node_output.stderr);
        eprintln!("Node FAILED:");
        eprintln!("{}", &stderr[..stderr.len().min(3000)]);
        panic!("node execution failed");
    }

    let stdout = String::from_utf8_lossy(&node_output.stdout);
    let as_output = parse_json_f32_array(stdout.trim());
    eprintln!("  WASM run took {:.1}s", run_time.as_secs_f64());
    eprintln!("  Output: {} elements", as_output.len());
    eprintln!("\nFull GPT-2 via UNFUSED WASM: completed");
}
