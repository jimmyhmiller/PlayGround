//! Snapshot tests for inspector views against the deploy-monitor fixture.
//!
//! Each test compares against a committed file under `tests/snapshots/`.
//! To regenerate after a deliberate change:
//!
//!     UPDATE_SNAPSHOTS=1 cargo test -p inspector

use std::env;
use std::fs;
use std::path::PathBuf;

use ir::manifest::Manifest;
use inspector::{handler_card, program_map, state_cell, validate_report};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ir/examples/deploy_monitor.json")
}

fn snapshots_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
}

fn load() -> Manifest {
    let raw = fs::read_to_string(fixture_path()).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn snapshot(name: &str, actual: &str) {
    let update = env::var("UPDATE_SNAPSHOTS").is_ok();
    let dir = snapshots_dir();
    if update {
        fs::create_dir_all(&dir).unwrap();
    }
    let path = dir.join(format!("{name}.txt"));
    if update {
        fs::write(&path, actual).unwrap();
        return;
    }
    let expected = fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "snapshot missing: {} (re-run with UPDATE_SNAPSHOTS=1)",
            path.display()
        )
    });
    if expected != actual {
        panic!(
            "snapshot mismatch for `{name}` at {}:\n--- expected ---\n{expected}\n--- got ---\n{actual}",
            path.display()
        );
    }
}

#[test]
fn show_matches_snapshot() {
    snapshot("show", &program_map(&load()));
}

#[test]
fn handler_kick_off_deploy_matches_snapshot() {
    let m = load();
    let text = handler_card(&m, "kick_off_deploy").expect("handler exists");
    snapshot("handler_kick_off_deploy", &text);
}

#[test]
fn handler_poll_in_flight_matches_snapshot() {
    let m = load();
    let text = handler_card(&m, "poll_in_flight").expect("handler exists");
    snapshot("handler_poll_in_flight", &text);
}

#[test]
fn state_in_progress_matches_snapshot() {
    let m = load();
    let text = state_cell(&m, "in_progress").expect("cell exists");
    snapshot("state_in_progress", &text);
}

#[test]
fn state_subscribers_matches_snapshot() {
    let m = load();
    let text = state_cell(&m, "subscribers").expect("cell exists");
    snapshot("state_subscribers", &text);
}

#[test]
fn validate_returns_ok_for_good_manifest() {
    let (ok, text) = validate_report(&load());
    assert!(ok, "deploy-monitor should validate cleanly");
    assert!(text.contains("OK"), "got: {text}");
}

#[test]
fn validate_reports_issues_for_bad_manifest() {
    // Hand-build a manifest with a reference to a non-existent event.
    let bad = serde_json::json!({
        "name": "bad",
        "version": "0.0.0",
        "schemas": {},
        "events": {},
        "state": {},
        "effects": {},
        "handlers": [
            {
                "name": "h",
                "on": "NoSuchEvent",
                "read": [],
                "write": [],
                "emit": [],
                "body": { "hash": "x", "uri": "x" }
            }
        ]
    });
    let manifest: Manifest = serde_json::from_value(bad).unwrap();
    let (ok, text) = validate_report(&manifest);
    assert!(!ok);
    assert!(text.contains("INVALID"));
    assert!(text.contains("NoSuchEvent"));
}

#[test]
fn unknown_handler_returns_none() {
    assert!(handler_card(&load(), "nope").is_none());
}

#[test]
fn unknown_state_returns_none() {
    assert!(state_cell(&load(), "nope").is_none());
}
