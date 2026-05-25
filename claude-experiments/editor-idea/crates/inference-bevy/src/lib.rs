//! Event-driven inference layer for the editor.
//!
//! Subscribes to the existing `claude-bus` event stream (via the
//! `claude-bus-bevy` plugin the host app already adds) and runs small
//! LLM-backed classifiers in response to specific events. Results are
//! published back to the bus as their own event kinds so any UI or
//! tooling can decide how to surface them.
//!
//! v1 wires a single case end-to-end:
//!
//!   `terminal.cwd_changed` →  cwd-default classifier
//!                          →  `inference.project_default_cwd_suggested`
//!
//! Classifier calls happen on a dedicated thread (one per pending
//! request) so the Bevy main thread is never blocked on network I/O.
//! The classifier thread publishes its own bus event when done; nothing
//! is sent back through the Bevy `App` itself.
//!
//! Adding a new classifier:
//!  - extend `classifiers.rs` with a system prompt + output shape
//!  - add a Bevy system that filters on the trigger event kind, spawns
//!    a thread, runs the classifier, and publishes the result event
//!
//! Deliberately keep this layer thin: heuristics that can be answered
//! without a model should live in the systems here (cheap, sync); only
//! genuinely ambiguous decisions should call out.

pub mod classifiers;
pub mod llm;

use bevy::prelude::*;
use claude_bus_bevy::ClaudeBusEvent;

use crate::classifiers::classify_default_cwd;
use crate::llm::LlmConfig;

/// Bus event kinds this crate publishes. Kept as constants here so the
/// downstream subscribers can match on the same strings without
/// drifting.
pub mod event_kinds {
    pub const TERMINAL_CWD_CHANGED: &str = "terminal.cwd_changed";
    pub const PROJECT_DEFAULT_CWD_SUGGESTED: &str = "inference.project_default_cwd_suggested";
}

/// Plugin entry point. Add after `claude_bus_bevy::BusEventPlugin`.
#[derive(Default, Clone)]
pub struct InferencePlugin;

impl Plugin for InferencePlugin {
    fn build(&self, app: &mut App) {
        // Resolve LLM config up front so we can warn the user once at
        // startup rather than every event. The Resource still exists in
        // an "unconfigured" form so the systems can no-op cleanly.
        let cfg = match LlmConfig::from_env() {
            Ok(c) => {
                info!(
                    "inference-bevy: LLM configured (base_url={} model={})",
                    c.base_url, c.model
                );
                Some(c)
            }
            Err(e) => {
                warn!(
                    "inference-bevy: classifier disabled ({}); \
                     set LLM_API_KEY or DEEPSEEK_KEY to enable",
                    e
                );
                None
            }
        };
        app.insert_resource(InferenceConfig { llm: cfg })
            .init_resource::<SeenCwds>()
            .add_systems(Update, on_cwd_changed);
    }
}

#[derive(Resource)]
struct InferenceConfig {
    llm: Option<LlmConfig>,
}

/// Per-session memory of the last cwd we already classified, so a
/// shell that repeatedly emits OSC 7 (every prompt) only triggers one
/// classification per unique destination. The "first cwd after open"
/// rule is implicit: $HOME (the spawn default) is added on first
/// sight, and any other cwd we haven't classified gets a call.
#[derive(Resource, Default)]
struct SeenCwds {
    by_session: std::collections::HashMap<String, std::collections::HashSet<String>>,
}

fn on_cwd_changed(
    mut events: MessageReader<ClaudeBusEvent>,
    cfg: Res<InferenceConfig>,
    mut seen: ResMut<SeenCwds>,
) {
    for ev in events.read() {
        if ev.kind != event_kinds::TERMINAL_CWD_CHANGED {
            continue;
        }
        let Some(payload) = serde_json::from_str::<CwdChangedPayload>(&ev.payload_json).ok()
        else {
            warn!("inference-bevy: malformed cwd_changed payload: {}", ev.payload_json);
            continue;
        };

        let session_seen = seen
            .by_session
            .entry(ev.terminal_session_id.clone())
            .or_default();
        if !session_seen.insert(payload.cwd.clone()) {
            // Already classified this (session, cwd) pair.
            continue;
        }

        // Skip $HOME — the spawn default. A user who hasn't actually
        // navigated anywhere isn't expressing a preference.
        if let Some(home) = std::env::var_os("HOME") {
            if std::path::Path::new(&payload.cwd) == std::path::Path::new(&home) {
                continue;
            }
        }

        // Need a model to do anything meaningful from here.
        let Some(llm_cfg) = cfg.llm.clone() else {
            continue;
        };

        // Project name we pass to the classifier. v1: the trailing
        // path component as a stand-in until we wire real project
        // membership through the cwd_changed event payload.
        let project_name = std::path::Path::new(&payload.cwd)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let session_id = ev.terminal_session_id.clone();
        let cwd = payload.cwd.clone();
        std::thread::Builder::new()
            .name("inference-classify-cwd".into())
            .spawn(move || {
                run_cwd_classification(llm_cfg, session_id, project_name, cwd);
            })
            .ok();
    }
}

#[derive(serde::Deserialize)]
struct CwdChangedPayload {
    #[serde(default)]
    #[allow(dead_code)]
    session_id: u64,
    cwd: String,
}

fn run_cwd_classification(
    cfg: LlmConfig,
    session_id: String,
    project_name: String,
    cwd: String,
) {
    let result = match classify_default_cwd(&cfg, &project_name, &cwd) {
        Ok(r) => r,
        Err(e) => {
            warn!(
                "inference-bevy: cwd classifier failed for session={} cwd={}: {}",
                session_id, cwd, e
            );
            return;
        }
    };
    info!(
        "inference-bevy: cwd suggestion session={} cwd={} good_default={} confidence={:.2}",
        session_id, cwd, result.good_default, result.confidence
    );
    publish_suggestion(&session_id, &project_name, &cwd, &result);
}

fn publish_suggestion(
    session_id: &str,
    project_name: &str,
    cwd: &str,
    result: &classifiers::DefaultCwdClassification,
) {
    let Some(socket) = claude_bus::socket_path() else {
        return;
    };
    let payload = serde_json::json!({
        "session_id": session_id.parse::<u64>().unwrap_or(0),
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
        event_kinds::PROJECT_DEFAULT_CWD_SUGGESTED,
        ts,
        session_id,
        std::process::id(),
        &payload,
    ) {
        warn!("inference-bevy: failed to publish suggestion: {}", e);
    }
}
