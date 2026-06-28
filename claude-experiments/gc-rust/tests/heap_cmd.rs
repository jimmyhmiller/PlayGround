//! `gcr heap <file>` — the heap-explorer data source. Runs a program and emits a
//! single JSON heap snapshot, preferring a mid-execution `heap_snapshot()` (live
//! stack roots) and falling back to the program-end dump. This is what the
//! `gcr_heap.ft` widget drives; the test pins the JSON shape it depends on.

use std::process::Command;

fn run_heap_tagged(tag: &str, src: &str, out_flag: bool) -> (bool, String, String) {
    let dir = std::env::temp_dir().join(format!("gcr_heapcmd_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srcp = dir.join("prog.gcr");
    std::fs::write(&srcp, src).unwrap();
    let outp = dir.join("snap.json");

    let mut args = vec!["heap".to_string(), srcp.to_str().unwrap().to_string()];
    if out_flag {
        args.push("--out".to_string());
        args.push(outp.to_str().unwrap().to_string());
    }
    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(&args)
        .output()
        .expect("run gcr heap");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let json = if out_flag {
        std::fs::read_to_string(&outp).unwrap_or_default()
    } else {
        stdout.clone()
    };
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), json, stdout)
}

#[test]
fn heap_cmd_prefers_midexec_snapshot_with_live_roots() {
    // A `heap_snapshot()` call mid-execution → the chosen snapshot has live roots
    // (x and y are still on the stack), unlike a program-end dump.
    let src = "\
struct P { x: i64, y: i64 }
fn main() -> i64 {
    let a = P { x: 1, y: 2 };
    let b = P { x: 3, y: 4 };
    heap_snapshot();
    a.x + b.y
}
";
    let (ok, json, _) = run_heap_tagged("midexec", src, true);
    assert!(ok, "gcr heap should succeed");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid snapshot JSON");
    let s = &v["summary"];
    assert_eq!(s["objects"].as_i64(), Some(2), "two P allocated");
    assert_eq!(s["reachable_objects"].as_i64(), Some(2), "both live mid-exec");
    assert_eq!(s["roots"].as_array().map(|a| a.len()), Some(2), "two live roots");
    assert!(v["objects"].as_array().map_or(false, |a| a.len() == 2));
}

#[test]
fn heap_cmd_reports_shared_object_incoming_refs() {
    // A single heap object referenced by two cells → two incoming edges. The
    // explorer's "retained by" view reads exactly these `refs` arrays.
    let src = "\
struct P { x: i64 }
enum L { Nil, Cons(P, L) }
fn main() -> i64 {
    let shared = P { x: 99 };
    let pair = L::Cons(shared, L::Cons(shared, L::Nil));
    heap_snapshot();
    match pair { L::Nil => 0, L::Cons(p, _) => p.x }
}
";
    let (ok, json, _) = run_heap_tagged("shared", src, true);
    assert!(ok, "gcr heap should succeed");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid snapshot JSON");
    let objs = v["objects"].as_array().expect("objects array");
    // Find the object with two incoming refs (the shared P).
    let mut incoming: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for o in objs {
        for r in o["refs"].as_array().into_iter().flatten() {
            *incoming.entry(r.as_i64().unwrap()).or_default() += 1;
        }
    }
    let shared = incoming.values().filter(|&&c| c >= 2).count();
    assert_eq!(shared, 1, "exactly one object should have two incoming refs (the shared P): {incoming:?}");
}

#[test]
fn heap_cmd_falls_back_to_program_end_dump() {
    // No `heap_snapshot()` call → the program-end dump is used. The objects are
    // unreachable at end (main returned), but the snapshot is still well-formed.
    let src = "\
struct Box { v: i64 }
fn main() -> i64 {
    let a = Box { v: 1 };
    let b = Box { v: 2 };
    a.v + b.v
}
";
    let (ok, json, _) = run_heap_tagged("fallback", src, true);
    assert!(ok, "gcr heap should succeed");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid snapshot JSON");
    assert_eq!(v["summary"]["objects"].as_i64(), Some(2), "two Box still on the heap");
}

#[test]
fn heap_cmd_series_bundles_snapshots_in_order() {
    // Several `heap_snapshot()` calls over a growing list → `--series` bundles
    // them in call order; the explorer's growth tab reads this monotonic climb.
    let src = "\
struct N { v: i64 }
enum L { Nil, Cons(N, L) }
fn main() -> i64 {
    let mut acc = L::Nil;
    let mut i = 0;
    while i < 4 {
        acc = L::Cons(N { v: i }, acc);
        heap_snapshot();
        i = i + 1;
    }
    0
}
";
    let dir = std::env::temp_dir().join(format!("gcr_heapseries_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srcp = dir.join("prog.gcr");
    std::fs::write(&srcp, src).unwrap();
    let outp = dir.join("series.json");
    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["heap", srcp.to_str().unwrap(), "--series", outp.to_str().unwrap()])
        .output()
        .expect("run gcr heap --series");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&outp).unwrap()).expect("valid series JSON");
    let snaps = v["snapshots"].as_array().expect("snapshots array");
    assert_eq!(snaps.len(), 4, "four snapshots, one per iteration");
    // The List count must climb monotonically across the series (the leak signal).
    let list_count = |s: &serde_json::Value| -> i64 {
        s["summary"]["by_type"].as_array().unwrap().iter()
            .find(|t| t["name"] == "L").and_then(|t| t["count"].as_i64()).unwrap_or(0)
    };
    let counts: Vec<i64> = snaps.iter().map(list_count).collect();
    assert_eq!(counts, vec![2, 3, 4, 5], "List count climbs by one each snapshot: {counts:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn heap_cmd_without_out_prints_json_to_stdout() {
    let src = "struct Q { n: i64 } fn main() -> i64 { let q = Q { n: 7 }; heap_snapshot(); q.n }";
    let (ok, _, stdout) = run_heap_tagged("stdout", src, false);
    assert!(ok);
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("stdout is valid JSON");
    assert!(v["summary"]["objects"].as_i64().unwrap() >= 1);
}
