//! Classifier prompts + result types.
//!
//! Each classifier is a (system prompt, user-template, output-shape)
//! triple. Keep them small and concrete — the smaller model behind a
//! cheap endpoint is fine for binary-ish decisions but easily confused
//! by ambiguous instructions.

use serde::Deserialize;

use crate::llm::{self, LlmConfig, LlmError};

/// Output of [`classify_default_cwd`]. The model is asked to pick
/// `good_default = true` only when the path is plausibly the canonical
/// working directory for the named project.
#[derive(Debug, Deserialize, Clone)]
pub struct DefaultCwdClassification {
    pub good_default: bool,
    /// 0.0 to 1.0. Caller should threshold (suggested >= 0.6) before
    /// surfacing the suggestion to the user.
    pub confidence: f32,
    /// One short sentence the user can see if we ever build a UI for
    /// these suggestions. Kept under ~120 chars by the prompt.
    pub reason: String,
}

const DEFAULT_CWD_SYSTEM: &str = "\
You classify whether a filesystem path is the canonical working \
directory for a developer's project. A canonical working directory is \
a stable, identifiable project root: a git repository root, a folder \
containing a build manifest (Cargo.toml, package.json, pyproject.toml, \
go.mod, Makefile, etc.), or an obvious project name match. \
Generic locations like the user's $HOME, $HOME/Downloads, /tmp, \
or unrelated system directories are NOT good defaults.

Respond with ONLY a JSON object matching this exact shape:
{\"good_default\": true|false, \"confidence\": 0.0..1.0, \"reason\": \"<one short sentence>\"}
No prose outside the JSON. Keep `reason` under 120 characters.";

/// Ask the model whether `cwd` is a reasonable default working
/// directory for the project named `project_name`. Blocking; call from
/// a thread.
pub fn classify_default_cwd(
    cfg: &LlmConfig,
    project_name: &str,
    cwd: &str,
) -> Result<DefaultCwdClassification, LlmError> {
    let user = format!("Project name: {project_name}\nCandidate cwd: {cwd}");
    llm::classify(cfg, DEFAULT_CWD_SYSTEM, &user)
}
