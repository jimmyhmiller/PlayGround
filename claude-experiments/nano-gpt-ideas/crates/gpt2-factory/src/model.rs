//! GPT-2 model resource + forward-pass polling.
//!
//! We load the model in `main()` (blocking), insert it as a Bevy `Resource`,
//! kick off a background forward pass, and every frame poll for new tiles.
//! The residual-stream belt subscribes to changes via `GptState::version`.

use bevy::prelude::*;
use gpt2_viz::gpt2::{Gpt2Model, LogitsResult, MAX_SEQ_LEN};

/// Which stage of the residual stream a piece of data belongs to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StageKind {
    Embedding,
    PostAttn(usize),
    PostMlp(usize),
}

/// Not a regular `Resource` because `Gpt2Model` contains raw pointers and
/// an `mpsc::Receiver` that aren't `Sync`. Inserted via
/// `App::insert_non_send_resource` and accessed with `NonSend`/`NonSendMut`.
#[allow(dead_code)]
pub struct GptState {
    pub model: Gpt2Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub token_ids: Vec<u32>,
    pub token_strings: Vec<String>,
    pub tile_values: Vec<Option<Vec<f32>>>,
    pub logits: Option<LogitsResult>,
    pub computing: bool,
    /// Bumped whenever tile_values changes so subscribers can re-render.
    pub version: u64,
}

impl GptState {
    pub fn n_tokens(&self) -> usize {
        self.token_ids.len().min(MAX_SEQ_LEN)
    }

    pub fn n_embd(&self) -> usize {
        self.model.config.n_embd
    }

    /// Look up the residual-stream tensor for a given stage, if it's been
    /// computed yet. Shape is `[1, T, D]` flattened row-major.
    pub fn get_residual(&self, kind: StageKind) -> Option<&Vec<f32>> {
        let ni = match kind {
            StageKind::Embedding => self.model.layout.embedding?,
            StageKind::PostAttn(l) => self.model.layout.layers.get(l)?.residual_attn?,
            StageKind::PostMlp(l) => self.model.layout.layers.get(l)?.residual_mlp?,
        };
        self.tile_values.get(ni)?.as_ref()
    }
}

pub struct ModelPlugin;

impl Plugin for ModelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, poll_forward);
    }
}

fn poll_forward(mut state: NonSendMut<GptState>) {
    let (new_logits, new_tiles, done) = state.model.poll();
    let mut changed = false;
    if let Some(l) = new_logits {
        state.logits = Some(l);
        changed = true;
    }
    for (idx, data) in new_tiles {
        if idx < state.tile_values.len() {
            state.tile_values[idx] = Some(data);
            changed = true;
        }
    }
    if done {
        state.computing = false;
        let n_done = state.tile_values.iter().filter(|v| v.is_some()).count();
        info!("Forward pass complete: {}/{} tiles", n_done, state.tile_values.len());
    }
    if changed {
        state.version = state.version.wrapping_add(1);
    }
}
