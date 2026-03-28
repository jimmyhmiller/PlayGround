use std::io::{self, Write, BufRead};
use rand::RngExt;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use tensor_lang_backend::assemblyscript::AssemblyScriptBackend;
use tensor_lang_graph::{compile, nanogpt, Op};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf()
}

struct Model {
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    /// Raw weight data: (name, flat f32 data, original shape)
    weights: Vec<(String, Vec<f32>, Vec<usize>)>,
    tokenizer: tokenizers::Tokenizer,
}

impl Model {
    fn load() -> Self {
        let root = project_root();
        let weights_dir = root.join("gpt2_weights");

        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(weights_dir.join("manifest.json"))
                .expect("Run `python3 export_gpt2.py` first")
        ).unwrap();

        let config = &manifest["config"];
        let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
        let n_embd = config["n_embd"].as_u64().unwrap() as usize;
        let n_head = config["n_head"].as_u64().unwrap() as usize;
        let n_layer = config["n_layer"].as_u64().unwrap() as usize;

        let weights_bin = std::fs::read(weights_dir.join("weights.bin")).unwrap();

        let tensors_meta = manifest["tensors"].as_array().unwrap();
        let mut weights = Vec::new();

        for t in tensors_meta {
            let name = t["name"].as_str().unwrap().to_string();
            let offset = t["offset"].as_u64().unwrap() as usize;
            let n_elements = t["n_elements"].as_u64().unwrap() as usize;
            let shape: Vec<usize> = t["shape"].as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect();

            let bytes = &weights_bin[offset..offset + n_elements * 4];
            let data: Vec<f32> = bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            weights.push((name, data, shape));
        }

        let tokenizer = tokenizers::Tokenizer::from_file(
            weights_dir.join("tokenizer/tokenizer.json")
        ).expect("Run `python3 export_tokenizer.py` first");

        Model { vocab_size, n_embd, n_head, n_layer, weights, tokenizer }
    }

    fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) {
        let root = project_root();
        let tmp_dir = std::env::temp_dir();
        let encoding = self.tokenizer.encode(prompt, false).unwrap();
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        // Compile once with symbolic seq_len (T)
        eprintln!("Compiling graph with symbolic seq_len...");
        let t_compile = Instant::now();

        let program = nanogpt::generate_nanogpt_program_symbolic(
            1, self.vocab_size, self.n_embd, self.n_head, self.n_layer,
        );
        let graph = compile(&program);

        let backend = AssemblyScriptBackend;
        let as_code = backend.emit_fused(&graph);
        eprintln!("  {} nodes, {} lines AS", graph.nodes.len(), as_code.lines().count());

        let src_path = tmp_dir.join("gpt2_cached.ts");
        let wasm_path = tmp_dir.join("gpt2_cached.wasm");
        std::fs::write(&src_path, &as_code).unwrap();

        let asc_output = Command::new("npx")
            .args(["asc", src_path.to_str().unwrap(),
                   "--outFile", wasm_path.to_str().unwrap(),
                   "--exportRuntime", "--optimize",
                   "--initialMemory", "2048",
                   "--maximumMemory", "65536"])
            .current_dir(&root)
            .output()
            .expect("failed to run asc");

        if !asc_output.status.success() {
            let stderr = String::from_utf8_lossy(&asc_output.stderr);
            eprintln!("asc compilation failed: {}", &stderr[..stderr.len().min(500)]);
            return;
        }
        eprintln!("Compiled in {:.1}s", t_compile.elapsed().as_secs_f64());

        // Count inputs (tokens=0, wte=1, wpe=2, attn_mask=3, then weights)
        let n_inputs = graph.nodes.iter()
            .filter(|n| matches!(&n.op, Op::Input { .. }))
            .count();
        let mask_input_idx = 3;

        // Prepare weight inputs once (they don't change between steps)
        let mut weight_data: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut wi = 0;
        for idx in 1..n_inputs {
            if idx == mask_input_idx {
                continue; // mask is dynamic, built per step
            }
            let (ref _wname, ref data, _) = self.weights[wi];
            wi += 1;
            weight_data.push((idx, data.clone()));
        }

        print!("{}", prompt);
        io::stdout().flush().unwrap();

        let t_gen = Instant::now();
        let runner = root.join("test_runner_bin.mjs");
        let inputs_bin_path = tmp_dir.join("gpt2_cli_inputs.bin");
        let inputs_manifest_path = tmp_dir.join("gpt2_cli_manifest.json");

        for step in 0..max_tokens {
            let actual_t = tokens.len();
            if actual_t > 1024 {
                eprintln!("Sequence exceeds 1024 tokens");
                break;
            }

            // Build dynamic inputs sized to actual_t (NOT padded!)
            let token_input: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();

            // wpe: first actual_t rows of the full positional embedding table
            let wpe_data = &self.weights[1].1; // wpe is the 2nd weight
            let wpe_slice = &wpe_data[..actual_t * self.n_embd];

            // Attention mask: causal only, sized to actual_t x actual_t
            let mut mask = vec![0.0f32; actual_t * actual_t];
            for i in 0..actual_t {
                for j in 0..actual_t {
                    if j > i {
                        mask[i * actual_t + j] = -1000000.0;
                    }
                }
            }

            // Assemble all inputs in graph order
            let mut flat_inputs: Vec<&[f32]> = vec![&[]; n_inputs];
            flat_inputs[0] = &token_input;
            // wte at index 1 — full table, doesn't change
            flat_inputs[1] = &self.weights[0].1;
            // wpe at index 2 — sliced to actual_t rows
            flat_inputs[2] = wpe_slice;
            // mask at index 3
            flat_inputs[mask_input_idx] = &mask;
            // remaining weights
            for (idx, data) in &weight_data {
                flat_inputs[*idx] = data;
            }

            // Write binary + manifest with dim_params
            {
                let mut f = io::BufWriter::new(std::fs::File::create(&inputs_bin_path).unwrap());
                for arr in &flat_inputs {
                    for v in *arr {
                        f.write_all(&v.to_le_bytes()).unwrap();
                    }
                }
            }
            let manifest_entries: Vec<String> = flat_inputs.iter()
                .map(|arr| format!("{{\"n_elements\":{}}}", arr.len()))
                .collect();
            let manifest = format!(
                "{{\"dim_params\":[{}],\"inputs\":[{}]}}",
                actual_t,
                manifest_entries.join(",")
            );
            std::fs::write(&inputs_manifest_path, &manifest).unwrap();

            // Run WASM (no recompilation — same .wasm, just different T)
            let t_step = Instant::now();
            let node_output = Command::new("node")
                .args(["--max-old-space-size=8192",
                       runner.to_str().unwrap(),
                       wasm_path.to_str().unwrap(),
                       inputs_bin_path.to_str().unwrap(),
                       inputs_manifest_path.to_str().unwrap()])
                .current_dir(&root)
                .output()
                .expect("failed to run node");

            if !node_output.status.success() {
                let stderr = String::from_utf8_lossy(&node_output.stderr);
                eprintln!("WASM execution failed: {}", &stderr[..stderr.len().min(500)]);
                return;
            }

            // Parse output logits
            let stdout = String::from_utf8_lossy(&node_output.stdout);
            let s = stdout.trim().trim_start_matches('[').trim_end_matches(']');
            let logits: Vec<f32> = s.split(',')
                .map(|v| v.trim().parse::<f32>().unwrap())
                .collect();

            // Get logits for the last position (output is [1, T, vocab])
            let last_start = (actual_t - 1) * self.vocab_size;
            let last_logits = &logits[last_start..last_start + self.vocab_size];
            let next_token = if temperature <= 0.0 {
                // Greedy argmax
                last_logits.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap().0 as u32
            } else {
                // Temperature sampling
                let max_l = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = last_logits.iter()
                    .map(|&l| ((l - max_l) / temperature).exp())
                    .collect();
                let sum: f32 = exp.iter().sum();
                let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();
                let r: f32 = rand::rng().random();
                let mut cumulative = 0.0;
                let mut chosen = 0u32;
                for (i, &p) in probs.iter().enumerate() {
                    cumulative += p;
                    if r < cumulative {
                        chosen = i as u32;
                        break;
                    }
                }
                chosen
            };

            // Decode and print
            let text = self.tokenizer.decode(&[next_token], false).unwrap();
            print!("{}", text);
            io::stdout().flush().unwrap();

            let step_time = t_step.elapsed().as_secs_f64();
            if step == 0 {
                eprintln!("  (first token: {step_time:.1}s)");
            }

            tokens.push(next_token);

            if next_token == 50256 { break; } // <|endoftext|>
        }

        // Cleanup
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&wasm_path);
        let _ = std::fs::remove_file(&inputs_bin_path);
        let _ = std::fs::remove_file(&inputs_manifest_path);

        println!();
        let generated = tokens.len() - encoding.get_ids().len();
        eprintln!("({} tokens in {:.1}s)", generated, t_gen.elapsed().as_secs_f64());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut temperature = 0.0f32;
    let mut max_tokens = 20usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" | "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("invalid temperature value");
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("invalid max-tokens value");
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!("Loading GPT-2 (124M) weights...");
    let t0 = Instant::now();
    let model = Model::load();
    eprintln!("Loaded in {:.1}s", t0.elapsed().as_secs_f64());
    eprintln!("vocab={}, d={}, heads={}, layers={}",
        model.vocab_size, model.n_embd, model.n_head, model.n_layer);
    eprintln!("Backend: fused WASM (AssemblyScript)");
    eprintln!("Temperature: {temperature}, Max tokens: {max_tokens}");
    eprintln!();
    eprintln!("Type a prompt and press Enter. Ctrl-C to quit.");
    eprintln!();

    let stdin = io::stdin();
    loop {
        eprint!("> ");
        io::stderr().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }
        let prompt = line.trim_end_matches('\n').trim_end_matches('\r');
        if prompt.is_empty() { continue; }

        model.generate(prompt, max_tokens, temperature);
        eprintln!();
    }
}
