//! `gcr bench <prog…>` — the general benchmark runner. Runs any gc-rust
//! program(s) and emits the `gcr-bench/1` metrics schema: every program is a
//! group with a `gc-rust` series carrying runtime + compile + size + the
//! runtime-only metrics (GC cycles/pauses, allocation churn, peak heap). This is
//! the "run any program, get any benchmark" data source for the bench toolkit.

use std::process::Command;

fn bench(tag: &str, src: &str) -> serde_json::Value {
    let dir = std::env::temp_dir().join(format!("gcr_benchcmd_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srcp = dir.join("prog.gcr");
    std::fs::write(&srcp, src).unwrap();
    let outp = dir.join("bench.json");
    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["bench", srcp.to_str().unwrap(), "--runs", "2", "--json", outp.to_str().unwrap()])
        .output()
        .expect("run gcr bench");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let v = serde_json::from_str(&std::fs::read_to_string(&outp).unwrap()).expect("valid bench JSON");
    let _ = std::fs::remove_dir_all(&dir);
    v
}

#[test]
fn bench_emits_general_schema_with_all_metrics() {
    let v = bench("schema", "fn main() -> i64 { 21 + 21 }");
    assert_eq!(v["schema"], "gcr-bench/1");
    let keys: Vec<&str> = v["metrics"].as_array().unwrap().iter()
        .map(|m| m["key"].as_str().unwrap()).collect();
    for want in [
        "wall_ms", "compile_ms", "binary_kb", "gc_minor", "gc_major",
        "gc_pause_max_ms", "gc_pause_total_ms", "alloc_objects", "alloc_bytes", "peak_heap_bytes",
    ] {
        assert!(keys.contains(&want), "metric {want} missing from {keys:?}");
    }
    // Each metric def carries units + lower_better so the views are schema-driven.
    let m0 = &v["metrics"][0];
    assert!(m0["unit"].is_string() && m0["lower_better"].is_boolean());

    let g = &v["groups"][0];
    assert_eq!(g["name"], "prog");
    let s = &g["series"][0];
    assert_eq!(s["label"], "gc-rust");
    // wall_ms is a distribution; compile/size are scalars.
    assert!(s["values"]["wall_ms"]["mean"].is_number());
    assert!(s["values"]["binary_kb"].as_f64().unwrap() > 0.0);
}

#[test]
fn bench_captures_allocation_and_gc_metrics() {
    // An allocation-heavy program must report nonzero allocation churn (and the
    // GC counters must be present, even if the small heap never triggers a cycle).
    let src = "\
enum T { Leaf, Node(T, T) }
fn make(d: i64) -> T { if d == 0 { T::Leaf } else { T::Node(make(d-1), make(d-1)) } }
fn count(t: T) -> i64 { match t { T::Leaf => 1, T::Node(l, r) => 1 + count(l) + count(r) } }
fn main() -> i64 {
    let mut total = 0; let mut i = 0;
    while i < 20 { total = total + count(make(10)); i = i + 1; }
    total
}
";
    let v = bench("alloc", src);
    let vals = &v["groups"][0]["series"][0]["values"];
    assert!(vals["alloc_objects"].as_f64().unwrap() > 1000.0, "should record many allocations: {vals}");
    assert!(vals["alloc_bytes"].as_f64().unwrap() > 1000.0, "should record allocation bytes");
    assert!(vals["gc_minor"].is_number(), "gc_minor present");
}

#[test]
fn bench_vary_nursery_produces_a_series_per_value_with_distinct_gc() {
    // The same program under different GC nursery sizes → one series per size,
    // and a smaller nursery must collect more often (the tuning tradeoff). This
    // is a benchmark extracted from an unchanged program purely via a runtime knob.
    let src = "\
enum T { Leaf, Node(T, T) }
fn make(d: i64) -> T { if d == 0 { T::Leaf } else { T::Node(make(d-1), make(d-1)) } }
fn count(t: T) -> i64 { match t { T::Leaf => 1, T::Node(l, r) => 1 + count(l) + count(r) } }
fn main() -> i64 {
    let mut total = 0; let mut i = 0;
    while i < 30 { total = total + count(make(11)); i = i + 1; }
    total
}
";
    let dir = std::env::temp_dir().join(format!("gcr_benchvary_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srcp = dir.join("prog.gcr");
    std::fs::write(&srcp, src).unwrap();
    let outp = dir.join("out.json");
    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["bench", srcp.to_str().unwrap(), "--nursery", "1,16", "--runs", "1",
               "--json", outp.to_str().unwrap()])
        .output()
        .expect("run gcr bench --nursery");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&outp).unwrap()).unwrap();
    assert_eq!(v["meta"]["vary"]["env"], "GCR_NURSERY_MB");
    let series = v["groups"][0]["series"].as_array().unwrap();
    assert_eq!(series.len(), 2, "one series per nursery value");
    assert_eq!(series[0]["label"], "nursery 1MB");
    assert_eq!(series[1]["label"], "nursery 16MB");
    let gc_small = series[0]["values"]["gc_minor"].as_f64().unwrap();
    let gc_big = series[1]["values"]["gc_minor"].as_f64().unwrap();
    assert!(gc_small > gc_big, "1MB nursery must collect more than 16MB: {gc_small} vs {gc_big}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bench_runs_multiple_programs_as_groups() {
    let dir = std::env::temp_dir().join(format!("gcr_benchmulti_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("a.gcr");
    let b = dir.join("b.gcr");
    std::fs::write(&a, "fn main() -> i64 { 1 }").unwrap();
    std::fs::write(&b, "fn main() -> i64 { 2 }").unwrap();
    let outp = dir.join("out.json");
    let out = Command::new(env!("CARGO_BIN_EXE_gcr"))
        .args(["bench", a.to_str().unwrap(), b.to_str().unwrap(), "--runs", "1", "--json", outp.to_str().unwrap()])
        .output()
        .expect("run gcr bench");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&outp).unwrap()).unwrap();
    let names: Vec<&str> = v["groups"].as_array().unwrap().iter()
        .map(|g| g["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["a", "b"], "one group per program, in order");
    let _ = std::fs::remove_dir_all(&dir);
}
