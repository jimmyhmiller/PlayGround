mod gpt2;
mod viewer;
mod viz;

use std::path::Path;

pub struct AppState {
    pub model: gpt2::Gpt2Model,
    pub tokenizer: tokenizers::Tokenizer,
    /// All token ids for the full sequence.
    pub all_token_ids: Vec<u32>,
    pub all_token_strings: Vec<String>,
    pub vocab_strings: Vec<String>,
    pub prompt: String,
    /// Which token position's activations are highlighted.
    pub focus_token: usize,
    /// Latest logits from the current window.
    pub logits: Option<gpt2::LogitsResult>,
    /// Per-node tile values. None = not yet computed.
    pub tile_values: Vec<Option<Vec<f32>>>,
    pub computing: bool,
}

impl AppState {
    /// Number of tokens the model is running on (capped at T).
    pub fn n_tokens(&self) -> usize {
        self.all_token_ids.len().min(gpt2::MAX_SEQ_LEN)
    }

    /// Launch background forward on all tokens. Non-blocking.
    pub fn launch(&mut self) {
        let ids: Vec<u32> = self.all_token_ids.iter().take(gpt2::MAX_SEQ_LEN).copied().collect();
        self.computing = self.model.launch_async(&ids);
    }

    /// Poll for background results.
    pub fn poll(&mut self) -> bool {
        let (new_logits, new_tiles, done) = self.model.poll();
        let mut changed = false;
        if let Some(l) = new_logits {
            self.logits = Some(l);
            changed = true;
        }
        for (idx, data) in new_tiles {
            if idx < self.tile_values.len() {
                self.tile_values[idx] = Some(data);
            }
            changed = true;
        }
        if done { self.computing = false; }
        changed
    }

    /// Move focus to next token position.
    pub fn focus_next(&mut self) {
        let max = self.n_tokens().saturating_sub(1);
        if self.focus_token < max {
            self.focus_token += 1;
        }
    }

    /// Move focus to previous token position.
    pub fn focus_prev(&mut self) {
        if self.focus_token > 0 {
            self.focus_token -= 1;
        }
    }

    /// Generate next token and relaunch.
    pub fn generate_one(&mut self) {
        let Some(logits) = &self.logits else {
            eprintln!("No logits yet");
            return;
        };
        if self.all_token_ids.len() >= gpt2::MAX_SEQ_LEN {
            eprintln!("Max sequence length reached");
            return;
        }
        let next = logits.next_token(&self.model.config);
        let next_str = self.tokenizer.id_to_token(next as u32)
            .unwrap_or_else(|| format!("[{next}]"));
        eprintln!("Generated: {:?}", next_str);
        self.all_token_ids.push(next as u32);
        self.all_token_strings.push(next_str);
        // Focus the new token
        self.focus_token = self.n_tokens() - 1;
        self.launch();
    }

    pub fn n_nodes(&self) -> usize { self.model.node_infos.len() }
    pub fn n_computed(&self) -> usize { self.tile_values.iter().filter(|v| v.is_some()).count() }

    pub fn focus_token_str(&self) -> &str {
        self.all_token_strings.get(self.focus_token)
            .map(|s| s.as_str()).unwrap_or("?")
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = if args.len() > 1 { args[1..].join(" ") }
        else { "The quick brown fox jumps over the lazy dog".to_string() };

    let weights_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().join("gpt2_weights");
    if !weights_dir.join("manifest.json").exists() {
        eprintln!("ERROR: gpt2_weights/ not found. Run: python3 export_gpt2.py");
        std::process::exit(1);
    }

    let tokenizer = tokenizers::Tokenizer::from_file(weights_dir.join("tokenizer/tokenizer.json"))
        .unwrap_or_else(|e| { eprintln!("tokenizer: {e}"); std::process::exit(1); });

    let encoding = tokenizer.encode(prompt.as_str(), false).unwrap();
    let all_token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let all_token_strings: Vec<String> = encoding.get_tokens().iter().map(|s| s.to_string()).collect();
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Tokens ({}): {all_token_strings:?}", all_token_ids.len());

    let mut model = gpt2::Gpt2Model::load(&weights_dir);
    let n_nodes = model.node_infos.len();

    let vocab_strings: Vec<String> = (0..model.config.vocab_size)
        .map(|i| tokenizer.id_to_token(i as u32).unwrap_or_else(|| format!("[{i}]")))
        .collect();

    // Initial window: first T tokens
    let initial_ids: Vec<u32> = all_token_ids.iter().take(gpt2::MAX_SEQ_LEN).copied().collect();
    model.launch_async(&initial_ids);

    let state = AppState {
        model, tokenizer,
        all_token_ids, all_token_strings, vocab_strings, prompt,
        focus_token: 0,
        logits: None,
        tile_values: vec![None; n_nodes],
        computing: true,
    };

    eprintln!("\n{n_nodes} nodes (T={}), computing...", gpt2::MAX_SEQ_LEN);
    eprintln!("  L-drag=pan  R-drag=tilt  wheel=zoom");
    eprintln!("  left/right=step window  space=generate  +/-=size");

    viewer::App::new(state).run();
}
