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
use inference_bevy::classifiers::classify_default_cwd;
use inference_bevy::event_kinds::{PROJECT_DEFAULT_CWD_SUGGESTED, TERMINAL_CWD_CHANGED};
use inference_bevy::llm::LlmConfig;

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
            .add_systems(Update, dispatch_cwd_classifier);
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
