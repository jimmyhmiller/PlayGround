//! Decides *when* to invoke `inference-bevy`'s classifiers and
//! publishes the verdicts back to the bus. Lives in terminal-bevy
//! because the firing rules depend on host state (`Projects`,
//! `ProjectMembership`) that the library crate has no business
//! knowing about.
//!
//! Current rule for the cwd-default classifier: fire when **all** of:
//!
//! - the event is `terminal.cwd_changed`
//! - the new cwd is not `$HOME` (the spawn default — no preference
//!   expressed)
//! - the owning project does not already have a `default_cwd`
//!   remembered (the user's instruction: only infer when nothing is
//!   set yet)
//! - we haven't already started a classification for this
//!   (session_id, cwd) pair this run (the shell-integration shim
//!   re-emits OSC 7 on every prompt; we don't want N duplicate LLM
//!   calls for the same dir)
//!
//! Failing any of these we silently no-op. The matching pane → project
//! lookup also has to succeed; standalone test panes without a
//! `ProjectMembership` are skipped.

use std::collections::HashSet;

use bevy::prelude::*;

use claude_bus_bevy::ClaudeBusEvent;
use inference_bevy::classifiers::{classify_command_suggestion, classify_default_cwd};
use inference_bevy::event_kinds::{
    COMMAND_PANE_SUGGESTED, PROJECT_DEFAULT_CWD_SUGGESTED, TERMINAL_COMMAND_EXECUTED,
    TERMINAL_CWD_CHANGED,
};
use inference_bevy::llm::LlmConfig;

use crate::drawer::Drawer;
use crate::projects::Projects;
use crate::TerminalSession;
use pane_bevy::PaneProject;

pub struct InferenceDispatchPlugin;

impl Plugin for InferenceDispatchPlugin {
    fn build(&self, app: &mut App) {
        // Resolve LLM config once at startup so the per-event hot path
        // doesn't repeatedly hit env / log warnings.
        let cfg = match LlmConfig::from_env() {
            Ok(c) => {
                info!(
                    "[inference-dispatch] LLM configured (base_url={} model={})",
                    c.base_url, c.model
                );
                Some(c)
            }
            Err(e) => {
                warn!(
                    "[inference-dispatch] classifier disabled ({}); \
                     set LLM_API_KEY or DEEPSEEK_KEY to enable",
                    e
                );
                None
            }
        };
        app.insert_resource(InferenceConfig { llm: cfg })
            .init_resource::<ClassifiedCwds>()
            .init_resource::<ClassifiedCommands>()
            .add_systems(
                Update,
                (
                    dispatch_cwd_classifier,
                    dispatch_command_classifier,
                    consume_command_suggestions,
                ),
            );
    }
}

#[derive(Resource)]
struct InferenceConfig {
    llm: Option<LlmConfig>,
}

/// Per-(session_id, cwd) memo so the OSC 7 emitter's per-prompt
/// repeats don't each trigger a fresh LLM call. Session-scoped; not
/// persisted (a restart of the GUI rebuilds it from scratch and we
/// can afford the extra calls — typically the project already has a
/// `default_cwd` after the first run, so the project-state gate
/// short-circuits anyway).
#[derive(Resource, Default)]
struct ClassifiedCwds {
    seen: HashSet<(u64, String)>,
}

fn dispatch_cwd_classifier(
    mut events: MessageReader<ClaudeBusEvent>,
    cfg: Res<InferenceConfig>,
    mut seen: ResMut<ClassifiedCwds>,
    panes: Query<(&TerminalSession, Option<&PaneProject>)>,
    projects: Res<Projects>,
) {
    for ev in events.read() {
        if ev.kind != TERMINAL_CWD_CHANGED {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<CwdChangedPayload>(&ev.payload_json) else {
            warn!(
                "[inference-dispatch] malformed cwd_changed payload: {}",
                ev.payload_json
            );
            continue;
        };
        let Ok(session_id) = ev.terminal_session_id.parse::<u64>() else {
            continue;
        };

        // Skip $HOME — the spawn default. The user navigating to home
        // isn't expressing a preference.
        if let Some(home) = std::env::var_os("HOME") {
            if std::path::Path::new(&payload.cwd) == std::path::Path::new(&home) {
                continue;
            }
        }

        // Skip if we've already classified this exact (session, cwd)
        // in this run.
        if !seen.seen.insert((session_id, payload.cwd.clone())) {
            continue;
        }

        // Resolve owning project; skip if untagged.
        let Some(project_id) = panes
            .iter()
            .find(|(ts, _)| ts.0 == session_id)
            .and_then(|(_, pp)| pp.map(|p| p.0))
        else {
            continue;
        };

        // GATE: skip if the project already has a remembered cwd. The
        // user's rule — once we've decided, don't re-classify on
        // every subsequent navigation.
        if projects.default_cwd_of(project_id).is_some() {
            continue;
        }

        // Need a model to do anything meaningful from here.
        let Some(llm_cfg) = cfg.llm.clone() else {
            continue;
        };

        let project_name = projects
            .name_of(project_id)
            .unwrap_or("unknown")
            .to_string();
        let cwd = payload.cwd.clone();
        let session_id_str = ev.terminal_session_id.clone();
        std::thread::Builder::new()
            .name("inference-classify-cwd".into())
            .spawn(move || {
                run_classification(llm_cfg, session_id_str, session_id, project_name, cwd);
            })
            .ok();
    }
}

#[derive(serde::Deserialize)]
struct CwdChangedPayload {
    cwd: String,
}

fn run_classification(
    cfg: LlmConfig,
    session_id_str: String,
    session_id: u64,
    project_name: String,
    cwd: String,
) {
    let result = match classify_default_cwd(&cfg, &project_name, &cwd) {
        Ok(r) => r,
        Err(e) => {
            warn!(
                "[inference-dispatch] cwd classifier failed for session={} cwd={}: {}",
                session_id, cwd, e
            );
            return;
        }
    };
    info!(
        "[inference-dispatch] cwd verdict session={} cwd={} good_default={} confidence={:.2}",
        session_id, cwd, result.good_default, result.confidence
    );
    publish_suggestion(&session_id_str, session_id, &project_name, &cwd, &result);
}

fn publish_suggestion(
    session_id_str: &str,
    session_id: u64,
    project_name: &str,
    cwd: &str,
    result: &inference_bevy::classifiers::DefaultCwdClassification,
) {
    let Some(socket) = claude_bus::socket_path() else {
        return;
    };
    let payload = serde_json::json!({
        "session_id": session_id,
        "project_name": project_name,
        "cwd": cwd,
        "good_default": result.good_default,
        "confidence": result.confidence,
        "reason": result.reason,
    })
    .to_string();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Err(e) = claude_bus::client::publish_oneshot(
        &socket,
        PROJECT_DEFAULT_CWD_SUGGESTED,
        ts,
        session_id_str,
        std::process::id(),
        &payload,
    ) {
        warn!("[inference-dispatch] failed to publish suggestion: {}", e);
    }
}

// ---------- Command-suggestion classifier ----------

/// Below this the classifier's verdict isn't surfaced into the drawer.
const COMMAND_SUGGEST_THRESHOLD: f32 = 0.7;
/// Commands longer than this are skipped (huge heredocs / paste blobs
/// aren't useful run-buttons and bloat the prompt).
const MAX_COMMAND_LEN: usize = 400;

/// Allowlist of task-runner / build-tool programs. ONLY commands whose
/// leading program is one of these ever reach the LLM classifier —
/// everything else (cd, ls, cat, grep, git, vim, …) is dropped with no
/// model call. This is the cheap gate that keeps us from classifying
/// every keystroke. Edit freely as the toolset grows.
const COMMAND_RUNNERS: &[&str] = &[
    // Rust
    "cargo",
    // Node / JS / TS
    "npm", "pnpm", "yarn", "bun", "node", "deno", "npx",
    // Python
    "python", "python3", "pytest", "uv",
    // Clojure
    "clojure", "clj", "bb", "lein",
    // Beagle
    "beag",
    // Go / Zig / Swift
    "go", "zig", "swift",
    // generic build drivers
    "make", "just", "cmake", "ninja", "gradle", "mvn",
];

/// True for a leading `NAME=value` environment assignment (so
/// `RUST_LOG=debug cargo run` still resolves to `cargo`).
fn is_env_assignment(tok: &str) -> bool {
    match tok.split_once('=') {
        Some((name, _)) => {
            !name.is_empty()
                && name
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_ascii_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        None => false,
    }
}

/// If `command`'s leading program (after skipping `VAR=val` prefixes and
/// a `sudo`/`env`/`time`/`nice` wrapper, and stripping any path) is a
/// known runner, return it; otherwise `None`.
fn command_runner(command: &str) -> Option<&'static str> {
    for tok in command.split_whitespace() {
        if is_env_assignment(tok) || matches!(tok, "sudo" | "env" | "time" | "nice") {
            continue;
        }
        let prog = tok.rsplit('/').next().unwrap_or(tok);
        return COMMAND_RUNNERS.iter().copied().find(|&r| r == prog);
    }
    None
}

/// Per-(session, command) memo so a command re-run in the same session
/// this run doesn't trigger a duplicate LLM call. Session-scoped, not
/// persisted (the drawer's own dedup handles cross-restart repeats).
#[derive(Resource, Default)]
struct ClassifiedCommands {
    seen: std::collections::HashSet<(u64, String)>,
}

#[derive(serde::Deserialize)]
struct CommandExecutedPayload {
    command: String,
    #[serde(default)]
    cwd: String,
    #[serde(default)]
    exit_code: i32,
}

/// Fire the command classifier on each `terminal.command_executed`,
/// once per unique (session, command) this run. Resolves the owning
/// project (for the prompt label + later scoping); an untagged pane
/// just classifies with an "unknown" label and lands unscoped.
fn dispatch_command_classifier(
    mut events: MessageReader<ClaudeBusEvent>,
    cfg: Res<InferenceConfig>,
    mut seen: ResMut<ClassifiedCommands>,
    panes: Query<(&TerminalSession, Option<&PaneProject>)>,
    projects: Res<Projects>,
) {
    for ev in events.read() {
        if ev.kind != TERMINAL_COMMAND_EXECUTED {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<CommandExecutedPayload>(&ev.payload_json) else {
            warn!(
                "[inference-dispatch] malformed command_executed payload: {}",
                ev.payload_json
            );
            continue;
        };
        let Ok(session_id) = ev.terminal_session_id.parse::<u64>() else {
            continue;
        };

        let command = payload.command.trim().to_string();
        if command.is_empty() || command.len() > MAX_COMMAND_LEN {
            continue;
        }

        // Cheap gate: only known task-runners reach the model. Skips
        // navigation/inspection/editor/git commands with no LLM call.
        if command_runner(&command).is_none() {
            continue;
        }

        // One classification per (session, command) per run.
        if !seen.seen.insert((session_id, command.clone())) {
            continue;
        }

        // Need a model to do anything.
        let Some(llm_cfg) = cfg.llm.clone() else {
            continue;
        };

        // Owning project (if the pane is tagged) → label for the prompt.
        let project_name = panes
            .iter()
            .find(|(ts, _)| ts.0 == session_id)
            .and_then(|(_, pp)| pp.map(|p| p.0))
            .and_then(|id| projects.name_of(id))
            .unwrap_or("unknown")
            .to_string();

        let session_id_str = ev.terminal_session_id.clone();
        let cwd = payload.cwd.clone();
        let exit_code = payload.exit_code;
        std::thread::Builder::new()
            .name("inference-classify-cmd".into())
            .spawn(move || {
                run_command_classification(
                    llm_cfg,
                    session_id_str,
                    session_id,
                    project_name,
                    command,
                    cwd,
                    exit_code,
                );
            })
            .ok();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_command_classification(
    cfg: LlmConfig,
    session_id_str: String,
    session_id: u64,
    project_name: String,
    command: String,
    cwd: String,
    exit_code: i32,
) {
    let result = match classify_command_suggestion(&cfg, &project_name, &command, &cwd, exit_code) {
        Ok(r) => r,
        Err(e) => {
            warn!(
                "[inference-dispatch] command classifier failed (session={} cmd={:?}): {}",
                session_id, command, e
            );
            return;
        }
    };
    info!(
        "[inference-dispatch] command verdict session={} cmd={:?} worth={} confidence={:.2}",
        session_id, command, result.worth_suggesting, result.confidence
    );
    if !result.worth_suggesting || result.confidence < COMMAND_SUGGEST_THRESHOLD {
        return;
    }
    let Some(socket) = claude_bus::socket_path() else {
        return;
    };
    let payload = serde_json::json!({
        "session_id": session_id,
        "command": command,
        "cwd": cwd,
        "title": result.title,
        "reason": result.reason,
        "confidence": result.confidence,
    })
    .to_string();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Err(e) = claude_bus::client::publish_oneshot(
        &socket,
        COMMAND_PANE_SUGGESTED,
        ts,
        &session_id_str,
        std::process::id(),
        &payload,
    ) {
        warn!("[inference-dispatch] failed to publish command suggestion: {}", e);
    }
}

#[derive(serde::Deserialize)]
struct CommandSuggestedPayload {
    command: String,
    #[serde(default)]
    cwd: String,
    title: String,
    #[serde(default)]
    reason: String,
}

/// Land accepted command suggestions in the drawer, scoped to the
/// originating terminal's project (via session → pane → membership);
/// unscoped if the pane is untagged. Deduped against what's already
/// parked so a repeated command doesn't stack up.
fn consume_command_suggestions(
    mut events: MessageReader<ClaudeBusEvent>,
    panes: Query<(&TerminalSession, Option<&PaneProject>)>,
    mut drawer: ResMut<Drawer>,
) {
    for ev in events.read() {
        if ev.kind != COMMAND_PANE_SUGGESTED {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<CommandSuggestedPayload>(&ev.payload_json) else {
            continue;
        };
        let session_id = ev.terminal_session_id.parse::<u64>().ok();
        let project_id = session_id.and_then(|sid| {
            panes
                .iter()
                .find(|(ts, _)| ts.0 == sid)
                .and_then(|(_, pp)| pp.map(|p| p.0))
        });

        if drawer.has_command(project_id, &payload.command) {
            continue;
        }

        let config = serde_json::json!({
            "command": payload.command,
            "title": payload.title,
            "cwd": payload.cwd,
        });
        let reason = (!payload.reason.is_empty()).then_some(payload.reason);
        drawer.push("run-button".to_string(), payload.title, reason, config, project_id);
    }
}

#[cfg(test)]
mod tests {
    use super::command_runner;

    #[test]
    fn runners_pass() {
        assert_eq!(command_runner("cargo test --workspace"), Some("cargo"));
        assert_eq!(command_runner("npm run dev"), Some("npm"));
        assert_eq!(command_runner("RUST_LOG=debug cargo run"), Some("cargo"));
        assert_eq!(command_runner("env FOO=bar npm test"), Some("npm"));
        assert_eq!(command_runner("sudo make install"), Some("make"));
        assert_eq!(command_runner("/usr/local/bin/cargo build"), Some("cargo"));
        assert_eq!(command_runner("clojure -M:test"), Some("clojure"));
    }

    #[test]
    fn non_runners_skip() {
        assert_eq!(command_runner("ls -la"), None);
        assert_eq!(command_runner("cd ~/code"), None);
        assert_eq!(command_runner("git status"), None);
        assert_eq!(command_runner("cat README.md"), None);
        assert_eq!(command_runner("vim src/main.rs"), None);
        assert_eq!(command_runner(""), None);
    }
}
