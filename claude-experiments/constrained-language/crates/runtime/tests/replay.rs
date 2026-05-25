//! Replay tests for the deploy-monitor program.
//!
//! Two flavors:
//!   1. **State reconstruction** — apply the log's writes to a fresh
//!      state store, verify the result matches the original.
//!   2. **Deterministic body replay** — re-run the bodies in a fresh
//!      runtime where adapters return the *recorded* outcomes from the
//!      original log. The two logs should be byte-identical.

use std::path::PathBuf;

use serde_json::json;

use ir::manifest::Manifest;
use runtime::{
    apply_writes_from_log, BodyError, BodyCtx, InboundEvent, MockAdapter, ReplayAdapter, Runtime,
    StateStore,
};

fn load_manifest() -> Manifest {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ir/examples/deploy_monitor.json");
    let raw = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&raw).unwrap()
}

fn kick_off_body(ctx: &mut BodyCtx<'_>) -> Result<(), BodyError> {
    let event = ctx.event.clone();
    let sha = event.get("sha").cloned().unwrap_or(json!(""));
    let repo = event.get("repo").cloned().unwrap_or(json!(""));

    let subs = ctx
        .read_map_entry("subscribers", &repo)?
        .unwrap_or(json!([]));

    let deploy_id = format!("dep-{}", ctx.derive_id("deploy_id"));
    let record = json!({
        "id": deploy_id,
        "sha": sha,
        "repo": repo,
        "status": { "tag": "queued" },
        "started_at": ctx.arrival_ts,
    });
    ctx.put_map("in_progress", json!(deploy_id), record)?;

    ctx.emit(
        "HttpRequest",
        json!({ "method": "POST", "url": format!("https://deploy.example.com/{}",
                repo.as_str().unwrap_or("")), "body": [] }),
    )?;

    if let Some(list) = subs.as_array() {
        for ch in list {
            ctx.emit(
                "SlackPost",
                json!({ "channel": ch, "text": format!("queued {}", sha.as_str().unwrap_or("")) }),
            )?;
        }
    }
    ctx.emit("LogWrite", json!({ "level": { "tag": "info" }, "msg": "kicked off" }))?;
    Ok(())
}

fn register_bodies(rt: &mut Runtime) {
    rt.bodies.register("components/kick_off_deploy.wasm", kick_off_body);
    rt.bodies.register("components/update_status.wasm", |_| Ok(()));
    rt.bodies.register("components/poll_in_flight.wasm", |_| Ok(()));
    rt.bodies.register("components/do_rollback.wasm", |_| Ok(()));
}

fn seed_subscribers(rt: &mut Runtime) {
    rt.state.put_map_entry(
        "subscribers",
        json!("acme/widget"),
        json!(["#deploys", "#alerts"]),
    );
    rt.state
        .put_map_entry("subscribers", json!("acme/api"), json!(["#deploys"]));
}

fn drive(rt: &mut Runtime, shas: &[&str], repos: &[&str]) {
    for (sha, repo) in shas.iter().zip(repos.iter()) {
        rt.enqueue(InboundEvent::new(
            "MergeToMain",
            json!({ "sha": sha, "author": "alice", "repo": repo }),
        ))
        .unwrap();
    }
    rt.run_to_quiescence().unwrap();
}

fn live_runtime() -> Runtime {
    let m = load_manifest();
    let mut rt = Runtime::new(m).expect("validates");
    register_bodies(&mut rt);
    rt.adapters.register(
        "HttpRequest",
        Box::new(MockAdapter::with_response(json!({ "status": 200, "body": [] }))),
    );
    rt.adapters
        .register("SlackPost", Box::new(MockAdapter::with_response(json!({}))));
    rt.adapters
        .register("LogWrite", Box::new(MockAdapter::with_response(json!({}))));
    seed_subscribers(&mut rt);
    rt
}

#[test]
fn state_reconstructs_from_log() {
    let mut rt = live_runtime();
    drive(
        &mut rt,
        &["aa", "bb", "cc"],
        &["acme/widget", "acme/api", "acme/widget"],
    );

    let original_atoms = rt.state.atoms().clone();
    let original_maps = rt.state.maps().clone();

    // Fresh state: same starting point (manifest defaults + initial seed).
    let m = load_manifest();
    let mut fresh = StateStore::from_manifest(&m);
    fresh.put_map_entry(
        "subscribers",
        json!("acme/widget"),
        json!(["#deploys", "#alerts"]),
    );
    fresh.put_map_entry("subscribers", json!("acme/api"), json!(["#deploys"]));

    let writes = apply_writes_from_log(&mut fresh, &rt.log);
    assert!(writes > 0, "expected at least one write replayed");

    assert_eq!(
        fresh.atoms(),
        &original_atoms,
        "reconstructed atoms differ"
    );
    assert_eq!(fresh.maps(), &original_maps, "reconstructed maps differ");
}

#[test]
fn deterministic_replay_produces_identical_log() {
    // First run: real (mock) adapters, recording outcomes.
    let mut rt1 = live_runtime();
    drive(&mut rt1, &["aa", "bb"], &["acme/widget", "acme/api"]);

    let original_log_json: Vec<serde_json::Value> = rt1
        .log
        .entries()
        .iter()
        .map(|e| serde_json::to_value(e).unwrap())
        .collect();

    // Second run: same manifest + bodies + initial state, but adapters
    // return the recorded outcomes from rt1's log instead of fresh mocks.
    let m = load_manifest();
    let mut rt2 = Runtime::new(m).expect("validates");
    register_bodies(&mut rt2);
    rt2.adapters.register(
        "HttpRequest",
        Box::new(ReplayAdapter::from_log(&rt1.log, "HttpRequest")),
    );
    rt2.adapters.register(
        "SlackPost",
        Box::new(ReplayAdapter::from_log(&rt1.log, "SlackPost")),
    );
    rt2.adapters.register(
        "LogWrite",
        Box::new(ReplayAdapter::from_log(&rt1.log, "LogWrite")),
    );
    seed_subscribers(&mut rt2);

    drive(&mut rt2, &["aa", "bb"], &["acme/widget", "acme/api"]);

    let replayed_log_json: Vec<serde_json::Value> = rt2
        .log
        .entries()
        .iter()
        .map(|e| serde_json::to_value(e).unwrap())
        .collect();

    assert_eq!(
        replayed_log_json.len(),
        original_log_json.len(),
        "log lengths differ"
    );

    for (i, (a, b)) in original_log_json
        .iter()
        .zip(replayed_log_json.iter())
        .enumerate()
    {
        assert_eq!(
            a, b,
            "log entry {i} differs:\nORIGINAL: {a}\nREPLAYED: {b}"
        );
    }
}

#[test]
fn state_after_replay_matches_state_after_live_run() {
    let mut rt1 = live_runtime();
    drive(&mut rt1, &["aa", "bb"], &["acme/widget", "acme/widget"]);

    let m = load_manifest();
    let mut rt2 = Runtime::new(m).expect("validates");
    register_bodies(&mut rt2);
    rt2.adapters.register(
        "HttpRequest",
        Box::new(ReplayAdapter::from_log(&rt1.log, "HttpRequest")),
    );
    rt2.adapters.register(
        "SlackPost",
        Box::new(ReplayAdapter::from_log(&rt1.log, "SlackPost")),
    );
    rt2.adapters.register(
        "LogWrite",
        Box::new(ReplayAdapter::from_log(&rt1.log, "LogWrite")),
    );
    seed_subscribers(&mut rt2);
    drive(&mut rt2, &["aa", "bb"], &["acme/widget", "acme/widget"]);

    assert_eq!(rt1.state.atoms(), rt2.state.atoms());
    assert_eq!(rt1.state.maps(), rt2.state.maps());
}
