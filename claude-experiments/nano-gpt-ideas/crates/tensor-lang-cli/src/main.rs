use std::io::{self, Write, BufRead};
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

    fn generate(&self, prompt: &str, max_tokens: usize) {
        let root = project_root();
        let tmp_dir = std::env::temp_dir();
        let encoding = self.tokenizer.encode(prompt, false).unwrap();
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

        print!("{}", prompt);
        io::stdout().flush().unwrap();

        let t_gen = Instant::now();

        for step in 0..max_tokens {
            let seq_len = tokens.len();

            // Generate DSL program and compile to graph
            let program = nanogpt::generate_nanogpt_program(
                1, seq_len, self.vocab_size, self.n_embd, self.n_head, self.n_layer,
            );
            let graph = compile(&program);

            // Emit fused AssemblyScript
            let backend = AssemblyScriptBackend;
            let as_code = backend.emit_fused(&graph);

            // Write AS source and compile to WASM
            let src_path = tmp_dir.join(format!("gpt2_cli_{step}.ts"));
            let wasm_path = tmp_dir.join(format!("gpt2_cli_{step}.wasm"));
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

            // Build inputs: token IDs + weights in graph order
            let input_nodes: Vec<(String, Vec<usize>)> = graph.nodes.iter()
                .filter_map(|n| {
                    if let Op::Input { name } = &n.op {
                        Some((name.clone(), n.shape.clone()))
                    } else { None }
                }).collect();

            let mut flat_inputs: Vec<Vec<f32>> = Vec::new();

            // Token input
            let token_floats: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
            flat_inputs.push(token_floats);

            // Weight inputs
            for (input_idx, (_, shape)) in input_nodes.iter().enumerate().skip(1) {
                let (ref wname, ref data, _) = self.weights[input_idx - 1];

                if wname == "wpe" && data.len() / self.n_embd > shape[0] {
                    flat_inputs.push(data[..shape[0] * self.n_embd].to_vec());
                } else {
                    flat_inputs.push(data.clone());
                }
            }

            // Write inputs as binary + manifest
            let inputs_bin_path = tmp_dir.join(format!("gpt2_cli_{step}_inputs.bin"));
            let inputs_manifest_path = tmp_dir.join(format!("gpt2_cli_{step}_manifest.json"));
            {
                let mut f = io::BufWriter::new(std::fs::File::create(&inputs_bin_path).unwrap());
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

            // Run WASM
            let runner = root.join("test_runner_bin.mjs");
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

            // Get logits for the last position, pick greedy
            let last_start = (seq_len - 1) * self.vocab_size;
            let last_logits = &logits[last_start..last_start + self.vocab_size];
            let next_token = last_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap().0 as u32;

            // Decode and print
            let text = self.tokenizer.decode(&[next_token], false).unwrap();
            print!("{}", text);
            io::stdout().flush().unwrap();


            // Cleanup temp files
            let _ = std::fs::remove_file(&src_path);
            let _ = std::fs::remove_file(&wasm_path);
            let _ = std::fs::remove_file(&inputs_bin_path);
            let _ = std::fs::remove_file(&inputs_manifest_path);

            tokens.push(next_token);

            if next_token == 50256 { break; } // <|endoftext|>
        }

        println!();
        let generated = tokens.len() - encoding.get_ids().len();
        eprintln!("({} tokens in {:.1}s)", generated, t_gen.elapsed().as_secs_f64());
    }
}

fn main() {
    eprintln!("Loading GPT-2 (124M) weights...");
    let t0 = Instant::now();
    let model = Model::load();
    eprintln!("Loaded in {:.1}s", t0.elapsed().as_secs_f64());
    eprintln!("vocab={}, d={}, heads={}, layers={}",
        model.vocab_size, model.n_embd, model.n_head, model.n_layer);
    eprintln!("Backend: fused WASM (AssemblyScript)");
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

        model.generate(prompt, 20);
        eprintln!();
    }
}
