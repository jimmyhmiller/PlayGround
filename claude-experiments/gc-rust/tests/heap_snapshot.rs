//! Heap-explorer P2: the on-demand mid-execution `heap_snapshot()` intrinsic.
//! A running program snapshots its OWN heap at a chosen point; objects live on
//! the stack at that point are correctly reported reachable (real mid-execution
//! roots), unlike the program-end dump where everything is garbage.

use std::process::Command;

#[test]
fn heap_snapshot_intrinsic_captures_live_roots_midexec() {
    // x and y are live (rooted on the stack) when heap_snapshot() runs.
    let src = "\
struct Box3 { a: i64, b: i64, c: i64 }
fn main() -> i64 {
    let x = Box3 { a: 1, b: 2, c: 3 };
    let y = Box3 { a: 4, b: 5, c: 6 };
    heap_snapshot();
    x.a + y.c
}
";
    let dir = std::env::temp_dir().join(format!("gcr_snap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srcp = dir.join("snap.gcr");
    std::fs::write(&srcp, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["run", srcp.to_str().unwrap()])
        .env("GCR_HEAP_SNAPSHOT_DIR", &dir)
        .output()
        .expect("run gcr");
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let snap = dir.join("snapshot-0000.json");
    assert!(snap.exists(), "snapshot file not written to GCR_HEAP_SNAPSHOT_DIR");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&snap).unwrap()).expect("valid snapshot JSON");
    let s = &json["summary"];
    assert_eq!(s["objects"].as_i64(), Some(2), "two Box3 allocated");
    // The whole point of a mid-execution snapshot: x and y are live → reachable
    // (program-end dump would show 0 reachable). Proves the frame-publish +
    // real-roots scan works for the requesting thread.
    assert_eq!(
        s["reachable_objects"].as_i64(),
        Some(2),
        "live stack roots must be reachable mid-execution; summary: {s}"
    );
    assert_eq!(s["roots"].as_array().map(|a| a.len()), Some(2), "two rooted objects");
    assert!(s["snapshot_pause_ns"].as_i64().unwrap_or(0) > 0, "STW pause must be measured");

    let _ = std::fs::remove_dir_all(&dir);
}
