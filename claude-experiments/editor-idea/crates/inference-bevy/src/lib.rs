//! Library of classifier prompts + a tiny OpenAI-compatible HTTP
//! client. Intentionally **not a dispatcher** — when to fire which
//! classifier depends on host context (which pane belongs to which
//! project, whether the project is already settled, etc.) that lives
//! in the editor crate. Keeping dispatch out of here means the library
//! never has to import terminal-bevy or know about `Projects`.
//!
//! Typical use from a host:
//!
//! ```ignore
//! let cfg = inference_bevy::llm::LlmConfig::from_env()?;
//! std::thread::spawn(move || {
//!     let verdict = inference_bevy::classifiers::classify_default_cwd(
//!         &cfg, "my-project", "/Users/me/code/my-project")?;
//!     /* publish verdict event yourself */
//! });
//! ```
//!
//! The crate still ships the `infer-classify-cwd` smoke-test bin so
//! the LLM path can be exercised end-to-end without the editor.

pub mod classifiers;
pub mod llm;

/// Event-kind constants shared with consumers. Keeping them here so
/// publishers and subscribers can match on the same strings without
/// drifting across crates.
pub mod event_kinds {
    pub const TERMINAL_CWD_CHANGED: &str = "terminal.cwd_changed";
    pub const PROJECT_DEFAULT_CWD_SUGGESTED: &str = "inference.project_default_cwd_suggested";
    /// A command finished in a terminal. Payload:
    /// `{session_id, command, cwd, exit_code}`. Emitted by the worker
    /// from OSC 133 shell-integration marks.
    pub const TERMINAL_COMMAND_EXECUTED: &str = "terminal.command_executed";
    /// The command classifier thinks a command is worth a saved
    /// run-button. Payload: `{session_id, command, cwd, title,
    /// confidence, reason, worth_suggesting}`. Consumed into the
    /// suggestion drawer.
    pub const COMMAND_PANE_SUGGESTED: &str = "inference.command_pane_suggested";
}
