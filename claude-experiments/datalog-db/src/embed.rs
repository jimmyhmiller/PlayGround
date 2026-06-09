//! Pure-Rust text embeddings via `candle` (BERT / sentence-transformer models
//! such as `BAAI/bge-small-en-v1.5`).
//!
//! Runs on the **CPU** backend. There is no production-ready ROCm path for
//! candle on this machine's GPU (gfx1151) — every AMD-GPU option is an
//! unmerged fork — so CPU is the correct engineering choice here, not a silent
//! fallback. bge-small is small (~33M params); batch-embedding a corpus is a
//! one-time job measured in minutes.
//!
//! Gated behind the `embed` cargo feature so the core database stays
//! dependency-light. Build/run with `--features embed`.

use std::path::PathBuf;

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;

/// A loaded embedding model ready to encode text into dense vectors.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    /// Output embedding dimension (the model's hidden size).
    pub dim: usize,
    /// Cap on tokens per input; longer inputs are truncated by the tokenizer.
    max_tokens: usize,
}

/// How to combine per-token hidden states into one vector.
#[derive(Debug, Clone, Copy)]
pub enum Pooling {
    /// Mean over non-padding tokens (the sentence-transformer default).
    Mean,
    /// The `[CLS]` token's hidden state.
    Cls,
}

impl Embedder {
    /// Load a model from the Hugging Face hub by id (downloads + caches on
    /// first use), e.g. `"BAAI/bge-small-en-v1.5"`. `revision` defaults to
    /// "main" when `None`.
    pub fn from_hub(model_id: &str, revision: Option<&str>) -> anyhow::Result<Self> {
        use hf_hub::api::sync::Api;
        use hf_hub::{Repo, RepoType};

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        ));
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        // Prefer safetensors; fall back to the PyTorch checkpoint name.
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.safetensors"))?;
        Self::from_files(&config_path, &tokenizer_path, &weights_path)
    }

    /// Load a model from explicit local files.
    pub fn from_files(
        config_path: &PathBuf,
        tokenizer_path: &PathBuf,
        weights_path: &PathBuf,
    ) -> anyhow::Result<Self> {
        let device = Device::Cpu;

        let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let dim = config.hidden_size;
        let max_tokens = config.max_position_embeddings.min(512);

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        // Enable padding + truncation so a batch tokenizes to a rectangular
        // tensor and over-long inputs are clipped to the model's max length.
        let pad_id = config.pad_token_id as u32;
        tokenizer
            .with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                pad_id,
                pad_token: "[PAD]".to_string(),
                ..Default::default()
            }))
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_tokens,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("configure tokenizer: {e}"))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DTYPE, &device)?
        };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            max_tokens,
        })
    }

    /// Embed a batch of texts into L2-normalized vectors (one per input).
    /// Normalizing makes dot-product == cosine similarity, which matches the
    /// `cosine` metric used by the vector-search query op.
    pub fn embed_batch(&self, texts: &[String], pooling: Pooling) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let n = encodings.len();
        let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut ids = Vec::with_capacity(n * seq_len);
        let mut mask = Vec::with_capacity(n * seq_len);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().copied());
            mask.extend(enc.get_attention_mask().iter().copied());
        }

        let input_ids = Tensor::from_vec(ids, (n, seq_len), &self.device)?;
        let attn_u32 = Tensor::from_vec(mask, (n, seq_len), &self.device)?;
        // Token type ids are all zeros for single-segment inputs.
        let token_type_ids = input_ids.zeros_like()?;

        // [n, seq, hidden]
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attn_u32))?;

        let pooled = match pooling {
            Pooling::Cls => hidden.i((.., 0, ..))?, // first token
            Pooling::Mean => {
                // Mean over non-padding tokens, using the attention mask as
                // weights so padding doesn't drag the average.
                let mask_f = attn_u32.to_dtype(DTYPE)?; // [n, seq]
                let mask_3d = mask_f.unsqueeze(2)?; // [n, seq, 1]
                let summed = hidden.broadcast_mul(&mask_3d)?.sum(1)?; // [n, hidden]
                let counts = mask_f.sum(1)?.unsqueeze(1)?; // [n, 1]
                summed.broadcast_div(&counts.clamp(1e-9f64, f64::INFINITY)?)?
            }
        };

        // L2-normalize each row.
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?; // [n, 1]
        let normed = pooled.broadcast_div(&norm.clamp(1e-12f64, f64::INFINITY)?)?;

        let rows: Vec<Vec<f32>> = normed.to_vec2()?;
        Ok(rows)
    }

    /// Embed a single text.
    pub fn embed(&self, text: &str, pooling: Pooling) -> anyhow::Result<Vec<f32>> {
        let mut out = self.embed_batch(&[text.to_string()], pooling)?;
        Ok(out.pop().unwrap_or_default())
    }

    /// Token budget per input (for chunking decisions upstream).
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}
