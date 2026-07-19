//! Counter emission in `memscope perfetto`, as a test.
//!
//! > **Skipping unchanged counter samples is lossless: replaying the emitted
//! > samples with step-hold semantics reproduces the exact live-byte curve, for
//! > the total and for every type.**
//!
//! `perfetto` used to write every counter on every event. An allocation touches
//! exactly one type, so that cost `events × types` where the information content
//! is `events` — on a 41M-event recording, 95 GB of which ~99.5% was the same
//! numbers restated. It also defaulted to `--top-types 10`, silently discarding
//! all but ten types, which is why peak-composition analysis came back ~11%
//! attributed.
//!
//! The fix only skips samples a Perfetto counter track would have held anyway.
//! That is the claim under test: reconstruct from the trace, compare against the
//! live set folded independently from the same events, and require equality at
//! every timestamp. The second test pins that *every* type gets a track by
//! default, so nothing is dropped without being asked for.

use std::collections::HashMap;
use std::process::Command;

use memscope_proto::{recfmt, EventKind, RawEvent, SiteId};

/// One synthetic allocation event.
struct Ev {
    kind: EventKind,
    addr: u64,
    size: u64,
    site: u32,
    ts: u64,
}

/// Types, one per site id. Two sites deliberately share a label (`Vec<u8>`):
/// counter tracks are keyed by name, so co-named sites must fold into one track
/// rather than fight over a single curve.
const SITE_TYPES: &[&str] = &["Vec<u8>", "String", "Box<Node>", "Vec<u8>", "HashMap<K, V>"];

/// A run with churn (alloc/free of the same address), several events sharing a
/// timestamp, types that go to zero and come back, and a type touched exactly
/// once — the cases where "only on change" could plausibly drop something real.
fn events() -> Vec<Ev> {
    let mut v = Vec::new();
    let mut push = |kind, addr, size, site, ts| v.push(Ev { kind, addr, size, site, ts });
    let a = EventKind::Alloc;
    let d = EventKind::Dealloc;

    push(a, 0x1000, 100, 0, 1_000_000); // Vec<u8> site 0
    push(a, 0x2000, 50, 1, 1_000_000); // same ts as above
    push(a, 0x3000, 200, 2, 2_000_000);
    push(d, 0x2000, 50, 1, 3_000_000); // String -> 0
    push(a, 0x4000, 30, 3, 3_000_000); // Vec<u8> site 3, same label as site 0
    push(a, 0x5000, 70, 1, 4_000_000); // String back from 0
    push(d, 0x1000, 100, 0, 5_000_000);
    push(a, 0x6000, 8, 4, 5_000_000); // HashMap touched exactly once
    push(a, 0x1000, 100, 0, 6_000_000); // address reused after free
    push(d, 0x3000, 200, 2, 7_000_000);
    push(d, 0x4000, 30, 3, 8_000_000);
    push(a, 0x7000, 1, 2, 8_000_000);
    push(d, 0x7000, 1, 2, 9_000_000); // net zero within the run
    v
}

fn write_recording(path: &str) {
    let evs = events();
    let mut b = Vec::new();
    recfmt::encode_header(&mut b, 4242, "/nonexistent/perfetto-counter-test", 0);
    for (i, ty) in SITE_TYPES.iter().enumerate() {
        // Resolved sites (type already known), so the reader needs no dSYM. No
        // shape: a shape wraps the label (`Vec<String>`), and these fixtures name
        // the labels directly so the expected counter names stay readable.
        recfmt::encode_site(&mut b, i as u32, Some(ty), None, &[]);
    }
    recfmt::encode_events_header(&mut b, evs.len() as u32);
    for (seq, e) in evs.iter().enumerate() {
        recfmt::encode_event(
            &mut b,
            &RawEvent {
                kind: e.kind,
                seq: seq as u64,
                ts_nanos: e.ts,
                addr: e.addr,
                size: e.size,
                align: 8,
                site: SiteId(e.site),
                thread: 1,
            },
        );
    }
    std::fs::write(path, &b).expect("write recording");
}

/// Fold the events directly: `ts -> (total, {label -> live bytes})` at the end of
/// each timestamp. This is the ground truth the trace must reproduce.
fn expected() -> Vec<(u64, u64, HashMap<String, u64>)> {
    let mut live: HashMap<u64, (u64, u32)> = HashMap::new();
    let mut per_label: HashMap<String, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut out: Vec<(u64, u64, HashMap<String, u64>)> = Vec::new();
    let evs = events();
    for (i, e) in evs.iter().enumerate() {
        match e.kind {
            EventKind::Alloc => {
                total += e.size;
                *per_label.entry(SITE_TYPES[e.site as usize].to_string()).or_default() += e.size;
                live.insert(e.addr, (e.size, e.site));
            }
            EventKind::Dealloc => {
                if let Some((size, site)) = live.remove(&e.addr) {
                    total -= size;
                    let c = per_label.entry(SITE_TYPES[site as usize].to_string()).or_default();
                    *c -= size;
                }
            }
            _ => {}
        }
        // Record only at the end of a timestamp — one point per ts is all the
        // trace format can carry, so that is the resolution being compared.
        if evs.get(i + 1).map(|n| n.ts) != Some(e.ts) {
            out.push((e.ts, total, per_label.clone()));
        }
    }
    out
}

/// Every counter sample in the trace, in file order.
fn counter_samples(trace: &serde_json::Value) -> Vec<(String, u64, u64)> {
    trace["traceEvents"]
        .as_array()
        .expect("traceEvents array")
        .iter()
        .filter(|e| e["ph"] == "C")
        .map(|e| {
            let ts_us = e["ts"].as_f64().expect("ts");
            (
                e["name"].as_str().expect("name").to_string(),
                (ts_us * 1000.0).round() as u64,
                e["args"]["bytes"].as_u64().expect("bytes"),
            )
        })
        .collect()
}

fn run_perfetto(tag: &str, extra: &[&str]) -> serde_json::Value {
    let dir = std::env::temp_dir();
    let rec = dir.join(format!("memscope-ctr-{tag}.mscope"));
    let out = dir.join(format!("memscope-ctr-{tag}.json"));
    write_recording(rec.to_str().expect("utf-8 path"));

    let status = Command::new(env!("CARGO_BIN_EXE_memscope"))
        .arg("perfetto")
        .arg(&rec)
        .arg("--out")
        .arg(&out)
        .args(extra)
        .output()
        .expect("run memscope perfetto");
    assert!(
        status.status.success(),
        "perfetto failed: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let text = std::fs::read_to_string(&out).expect("read trace");
    serde_json::from_str(&text).expect("trace is valid JSON")
}

/// The invariant: step-hold replay of the emitted samples equals the true curve.
#[test]
fn counters_are_lossless_after_dropping_unchanged_samples() {
    let trace = run_perfetto("lossless", &[]);
    let samples = counter_samples(&trace);

    // Replay exactly as Perfetto renders a counter track: a value persists until
    // the next sample on that track.
    let mut held: HashMap<String, u64> = HashMap::new();
    let mut by_ts: HashMap<u64, HashMap<String, u64>> = HashMap::new();
    let mut timestamps: Vec<u64> = Vec::new();
    for (name, ts, bytes) in &samples {
        held.insert(name.clone(), *bytes);
        if by_ts.insert(*ts, held.clone()).is_none() {
            timestamps.push(*ts);
        } else {
            by_ts.insert(*ts, held.clone());
        }
    }

    for (ts, total, per_label) in expected() {
        let state = by_ts.get(&ts).unwrap_or_else(|| {
            // No sample at this ts means nothing changed; carry the last state.
            let prev = timestamps.iter().filter(|t| **t < ts).max().expect("a prior sample");
            by_ts.get(prev).expect("state at prior ts")
        });
        assert_eq!(
            state.get("live_bytes").copied().unwrap_or(0),
            total,
            "live_bytes wrong at ts={ts}"
        );
        for (label, want) in &per_label {
            let got = state.get(&format!("live: {label}")).copied().unwrap_or(0);
            assert_eq!(got, *want, "type {label:?} wrong at ts={ts} (want {want}, got {got})");
        }
    }
}

/// No sample may restate a value the track already holds — that redundancy is
/// what made real traces 95 GB.
#[test]
fn no_sample_restates_an_unchanged_value() {
    let trace = run_perfetto("nodup", &[]);
    let mut held: HashMap<String, u64> = HashMap::new();
    for (name, ts, bytes) in counter_samples(&trace) {
        if let Some(prev) = held.insert(name.clone(), bytes) {
            assert_ne!(prev, bytes, "counter {name:?} restated {bytes} at ts={ts}");
        }
    }
}

/// Every distinct type gets a counter track by default — no truncation unless
/// `--top-types` explicitly asks for it. `Vec<u8>` is allocated from two sites
/// and must appear as *one* track carrying their sum.
#[test]
fn every_type_is_tracked_by_default_and_co_named_sites_merge() {
    let trace = run_perfetto("alltypes", &[]);
    let names: std::collections::HashSet<String> =
        counter_samples(&trace).into_iter().map(|(n, _, _)| n).collect();

    let mut want: Vec<String> = SITE_TYPES.iter().map(|t| format!("live: {t}")).collect();
    want.sort();
    want.dedup();
    for w in &want {
        assert!(names.contains(w), "missing counter track {w:?}; got {names:?}");
    }

    // Sites 0 and 3 are both Vec<u8>: at ts=3ms both are live (100 + 30).
    let merged = counter_samples(&trace)
        .into_iter()
        .filter(|(n, ts, _)| n == "live: Vec<u8>" && *ts == 3_000_000)
        .map(|(_, _, b)| b)
        .next()
        .expect("a Vec<u8> sample at ts=3ms");
    assert_eq!(merged, 130, "co-named sites must fold into one track");
}
