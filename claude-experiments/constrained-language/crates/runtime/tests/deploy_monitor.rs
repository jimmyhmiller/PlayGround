//! End-to-end test for the deploy-monitor program against the v0.1 runtime.
//!
//! We register a native body for each of the four handlers, register mock
//! effect adapters, push a `MergeToMain` event, run to quiescence, and
//! verify that:
//!
//! * the right state mutations were applied,
//! * the right effects were emitted (and in the right shape),
//! * the event log contains the events / handler invocations / effect rows,
//! * the runtime enforces declared footprints (a body that tries to write
//!   to an undeclared cell errors out).

use std::path::PathBuf;

use serde_json::json;

use ir::manifest::Manifest;
use runtime::body::BodyError;
use runtime::log::LogEntryKind;
use runtime::{InboundEvent, MockAdapter, Runtime, RuntimeError, Value};

fn load_manifest() -> Manifest {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ir/examples/deploy_monitor.json");
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    serde_json::from_str(&raw).expect("parse fixture")
}

fn register_deploy_monitor_bodies(rt: &mut Runtime) {
    // kick_off_deploy: on MergeToMain
    //   read subscribers[$event.repo]
    //   write in_progress[*]
    //   emit HttpRequest, SlackPost, LogWrite
    rt.bodies.register("components/kick_off_deploy.wasm", |ctx| {
        let event = ctx.event.clone();
        let sha = event.get("sha").cloned().unwrap_or(Value::Null);
        let repo = event.get("repo").cloned().unwrap_or(Value::Null);

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
            json!({
                "method": "POST",
                "url": format!("https://deploy.example.com/{}", repo.as_str().unwrap_or("")),
                "body": [],
            }),
        )?;

        if let Some(list) = subs.as_array() {
            for ch in list {
                ctx.emit(
                    "SlackPost",
                    json!({
                        "channel": ch,
                        "text": format!("Deploy queued for {}", sha.as_str().unwrap_or("")),
                    }),
                )?;
            }
        }

        ctx.emit(
            "LogWrite",
            json!({
                "level": { "tag": "info" },
                "msg": format!("kick_off_deploy ran for {}", repo.as_str().unwrap_or("")),
            }),
        )?;

        Ok(())
    });

    // update_status — not exercised by this test, but registered so the runtime
    // won't reject the manifest if a future event triggers it. Minimal body.
    rt.bodies
        .register("components/update_status.wasm", |_ctx| Ok(()));
    rt.bodies
        .register("components/poll_in_flight.wasm", |_ctx| Ok(()));
    rt.bodies
        .register("components/do_rollback.wasm", |_ctx| Ok(()));
}

fn make_runtime_with_seed() -> (Runtime, MockAdapter, MockAdapter, MockAdapter) {
    let manifest = load_manifest();
    let mut rt = Runtime::new(manifest).expect("manifest validates");
    register_deploy_monitor_bodies(&mut rt);

    let http = MockAdapter::with_response(json!({ "status": 200, "body": [] }));
    let slack = MockAdapter::with_response(json!({}));
    let log = MockAdapter::with_response(json!({}));

    rt.adapters
        .register("HttpRequest", Box::new(http.clone()));
    rt.adapters
        .register("SlackPost", Box::new(slack.clone()));
    rt.adapters
        .register("LogWrite", Box::new(log.clone()));

    // Seed subscribers[acme/widget] = ["#deploys", "#alerts"]
    rt.state.put_map_entry(
        "subscribers",
        json!("acme/widget"),
        json!(["#deploys", "#alerts"]),
    );

    (rt, http, slack, log)
}

#[test]
fn merge_to_main_kicks_off_a_deploy() {
    let (mut rt, http, slack, log) = make_runtime_with_seed();

    rt.enqueue(InboundEvent::new(
        "MergeToMain",
        json!({ "sha": "deadbeef", "author": "alice", "repo": "acme/widget" }),
    ))
    .expect("enqueue");
    rt.run_to_quiescence().expect("run");

    // State: one entry in in_progress.
    let entries = rt.state.list_map("in_progress");
    assert_eq!(entries.len(), 1, "expected exactly one in-flight deploy");
    let (_id, record) = &entries[0];
    assert_eq!(record["sha"], json!("deadbeef"));
    assert_eq!(record["repo"], json!("acme/widget"));
    assert_eq!(record["status"], json!({ "tag": "queued" }));

    // Adapters: HTTP request was issued.
    let http_calls = http.calls();
    assert_eq!(http_calls.len(), 1, "exactly one HttpRequest emitted");
    assert_eq!(http_calls[0]["method"], json!("POST"));
    assert!(http_calls[0]["url"]
        .as_str()
        .unwrap()
        .contains("acme/widget"));

    // Adapters: two Slack posts, one per subscriber.
    let slack_calls = slack.calls();
    assert_eq!(slack_calls.len(), 2, "one SlackPost per subscriber");
    let channels: Vec<&str> = slack_calls
        .iter()
        .map(|c| c["channel"].as_str().unwrap_or(""))
        .collect();
    assert!(channels.contains(&"#deploys"));
    assert!(channels.contains(&"#alerts"));

    // Adapters: one log line.
    assert_eq!(log.calls().len(), 1);
}

#[test]
fn event_log_records_event_handler_and_effects() {
    let (mut rt, _http, _slack, _log) = make_runtime_with_seed();

    rt.enqueue(InboundEvent::new(
        "MergeToMain",
        json!({ "sha": "deadbeef", "author": "alice", "repo": "acme/widget" }),
    ))
    .unwrap();
    rt.run_to_quiescence().unwrap();

    let kinds: Vec<&LogEntryKind> = rt.log.entries().iter().map(|e| &e.kind).collect();

    // First entry: EventEnqueued for MergeToMain.
    assert!(matches!(
        kinds[0],
        LogEntryKind::EventEnqueued { event, .. } if event == "MergeToMain"
    ));

    // Then exactly one HandlerInvoked for kick_off_deploy with one write and 4 emits
    // (1 HTTP + 2 Slack + 1 Log).
    let invocations: Vec<_> = kinds
        .iter()
        .filter_map(|k| match k {
            LogEntryKind::HandlerInvoked {
                handler,
                writes,
                emits,
                ..
            } => Some((handler.as_str(), writes.len(), emits.len())),
            _ => None,
        })
        .collect();
    assert_eq!(invocations, vec![("kick_off_deploy", 1, 4)]);

    // Then 4 EffectFulfilled entries.
    let fulfilled = kinds
        .iter()
        .filter(|k| matches!(k, LogEntryKind::EffectFulfilled { .. }))
        .count();
    assert_eq!(fulfilled, 4);
}

#[test]
fn body_violating_footprint_errors() {
    let manifest = load_manifest();
    let mut rt = Runtime::new(manifest).expect("validates");

    // A malicious kick_off_deploy that tries to write to `history`, which is
    // NOT in its declared footprint.
    rt.bodies.register("components/kick_off_deploy.wasm", |ctx| {
        ctx.put_map(
            "history",
            json!("forbidden"),
            json!({"id": "x", "sha": "x", "repo": "x", "status": {"tag": "queued"}, "started_at": 0}),
        )?;
        Ok(())
    });
    // Stub bodies for handlers that aren't invoked by this event.
    rt.bodies
        .register("components/update_status.wasm", |_| Ok(()));
    rt.bodies
        .register("components/poll_in_flight.wasm", |_| Ok(()));
    rt.bodies
        .register("components/do_rollback.wasm", |_| Ok(()));

    let http = MockAdapter::with_response(json!({"status": 200, "body": []}));
    let slack = MockAdapter::with_response(json!({}));
    let log = MockAdapter::with_response(json!({}));
    rt.adapters.register("HttpRequest", Box::new(http));
    rt.adapters.register("SlackPost", Box::new(slack));
    rt.adapters.register("LogWrite", Box::new(log));

    rt.enqueue(InboundEvent::new(
        "MergeToMain",
        json!({ "sha": "x", "author": "x", "repo": "x" }),
    ))
    .unwrap();
    let err = rt.run_to_quiescence().expect_err("body should be rejected");
    match err {
        RuntimeError::Body { handler, source } => {
            assert_eq!(handler, "kick_off_deploy");
            assert!(matches!(source, BodyError::UndeclaredWrite(c) if c == "history"));
        }
        other => panic!("expected RuntimeError::Body, got {:?}", other),
    }
}

#[test]
fn body_reading_undeclared_cell_errors() {
    let manifest = load_manifest();
    let mut rt = Runtime::new(manifest).expect("validates");

    rt.bodies.register("components/kick_off_deploy.wasm", |ctx| {
        // poll_in_flight has `read: in_progress[*]`. kick_off_deploy does NOT.
        let _ = ctx.list_map("in_progress")?;
        Ok(())
    });
    rt.bodies
        .register("components/update_status.wasm", |_| Ok(()));
    rt.bodies
        .register("components/poll_in_flight.wasm", |_| Ok(()));
    rt.bodies
        .register("components/do_rollback.wasm", |_| Ok(()));

    let http = MockAdapter::with_response(json!({"status": 200, "body": []}));
    let slack = MockAdapter::with_response(json!({}));
    let log = MockAdapter::with_response(json!({}));
    rt.adapters.register("HttpRequest", Box::new(http));
    rt.adapters.register("SlackPost", Box::new(slack));
    rt.adapters.register("LogWrite", Box::new(log));

    rt.enqueue(InboundEvent::new(
        "MergeToMain",
        json!({ "sha": "x", "author": "x", "repo": "x" }),
    ))
    .unwrap();
    let err = rt.run_to_quiescence().expect_err("body should be rejected");
    assert!(matches!(err, RuntimeError::Body { source: BodyError::UndeclaredRead(_), .. }));
}
