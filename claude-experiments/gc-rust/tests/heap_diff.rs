//! `gcr heap-diff` — the leak-hunting workflow over two `GCR_HEAP_DUMP=json`
//! snapshots. Verifies per-type growth detection (NEW / GREW) + the summary delta.

use std::process::Command;

fn write_tmp(name: &str, content: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("gcr_hd_{}_{}.json", std::process::id(), name));
    std::fs::write(&p, content).unwrap();
    p
}

#[test]
fn heap_diff_reports_per_type_growth() {
    // Before: 2 Pairs. After: 3 Pairs (GREW) + 1 new Trio (NEW).
    let a = r#"{"summary":{"version":1,"objects":2,"bytes":64,"reachable_objects":0,
        "by_type":[{"name":"Pair","count":2,"bytes":64}]},"objects":[]}"#;
    let b = r#"{"summary":{"version":1,"objects":4,"bytes":136,"reachable_objects":0,
        "by_type":[{"name":"Pair","count":3,"bytes":96},{"name":"Trio","count":1,"bytes":40}]},"objects":[]}"#;
    let pa = write_tmp("a", a);
    let pb = write_tmp("b", b);

    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["heap-diff", pa.to_str().unwrap(), pb.to_str().unwrap()])
        .output()
        .expect("run gcr heap-diff");
    assert!(out.status.success(), "heap-diff exit: {:?}", out.status);
    let s = String::from_utf8_lossy(&out.stdout);

    assert!(s.contains("Trio") && s.contains("[NEW]"), "expected Trio NEW; got:\n{s}");
    assert!(s.contains("Pair") && s.contains("[GREW]"), "expected Pair GREW; got:\n{s}");
    assert!(s.contains("+72"), "expected net +72 bytes; got:\n{s}");
    assert!(s.contains("objects 2 -> 4"), "expected object delta; got:\n{s}");

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn heap_diff_rejects_bad_json() {
    let pa = write_tmp("bad", "not json");
    let pb = write_tmp("ok", r#"{"summary":{"by_type":[]}}"#);
    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["heap-diff", pa.to_str().unwrap(), pb.to_str().unwrap()])
        .output()
        .expect("run gcr heap-diff");
    assert!(!out.status.success(), "bad JSON must fail, not silently pass");
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}
