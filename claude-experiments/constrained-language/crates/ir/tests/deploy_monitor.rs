//! End-to-end test for the deploy-monitor fixture:
//! * parse the manifest JSON
//! * validate it
//! * canonical encoding is stable
//! * generate WIT worlds for every handler and snapshot them
//!
//! Snapshot files live under `tests/snapshots/`. To regenerate them after a
//! deliberate change, run:
//!
//!     UPDATE_SNAPSHOTS=1 cargo test -p ir

use std::env;
use std::fs;
use std::path::PathBuf;

use ir::canonical::to_canonical_string;
use ir::manifest::Manifest;
use ir::{validate, wit};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/deploy_monitor.json")
}

fn snapshots_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
}

fn load_fixture() -> Manifest {
    let raw = fs::read_to_string(fixture_path()).expect("read fixture");
    serde_json::from_str(&raw).expect("parse fixture")
}

#[test]
fn fixture_parses() {
    let m = load_fixture();
    assert_eq!(m.name, "deploy-monitor");
    assert_eq!(m.handlers.len(), 4);
}

#[test]
fn fixture_validates() {
    let m = load_fixture();
    if let Err(e) = validate::validate(&m) {
        let report: Vec<String> = e.issues().iter().map(|i| i.to_string()).collect();
        panic!("validation failed:\n  - {}", report.join("\n  - "));
    }
}

#[test]
fn canonical_encoding_is_stable() {
    let m = load_fixture();
    let a = to_canonical_string(&m).expect("encode 1");
    let b = to_canonical_string(&m).expect("encode 2");
    assert_eq!(a, b, "two encodings of the same value must be identical");

    // Sanity: top-level keys are sorted. Parse back and inspect.
    let v: serde_json::Value = serde_json::from_str(&a).expect("re-parse canonical");
    let obj = v.as_object().expect("top-level is object");
    let keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "top-level keys must be sorted");
}

#[test]
fn wit_worlds_match_snapshots() {
    let m = load_fixture();
    validate::validate(&m).expect("manifest validates");

    let update = env::var("UPDATE_SNAPSHOTS").is_ok();
    let dir = snapshots_dir();
    if update {
        fs::create_dir_all(&dir).expect("create snapshots dir");
    }

    let mut failures = Vec::<String>::new();
    for h in &m.handlers {
        let world = wit::generate_world(h, &m);
        let snapshot_path = dir.join(format!("{}.wit", h.name));

        if update {
            fs::write(&snapshot_path, &world).expect("write snapshot");
            continue;
        }

        let expected = match fs::read_to_string(&snapshot_path) {
            Ok(s) => s,
            Err(_) => {
                failures.push(format!(
                    "snapshot missing: {} (re-run with UPDATE_SNAPSHOTS=1 to create)",
                    snapshot_path.display()
                ));
                continue;
            }
        };
        if world != expected {
            failures.push(format!(
                "snapshot mismatch for `{}` at {}:\n--- expected ---\n{}\n--- got ---\n{}",
                h.name,
                snapshot_path.display(),
                expected,
                world
            ));
        }
    }

    if !failures.is_empty() {
        panic!("{}", failures.join("\n\n"));
    }
}
