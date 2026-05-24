//! Chrome Trace Event Format (chrome://tracing JSON).
//!
//! Two top-level forms accepted:
//! - Array form: `[{...event...}, ...]`
//! - Object form: `{"traceEvents": [...], "displayTimeUnit": "ms", ...}`
//!
//! Phases handled in v1: `B`/`E` (paired into slices via per-(pid,tid) open stack),
//! `X` (complete, has explicit `dur`), `i`/`I` (instant — recorded as zero-width
//! slice), `M` (metadata — `thread_name`, `process_name`).
//!
//! Phases skipped (with a `log::warn` once per kind): `b`/`e`/`n` async, `s`/`t`/`f`
//! flow, `C` counter, `R`/`O`/`N`/`D` object events, `c`/`(`/`)` clock sync, `P`/`p`
//! samples (we already have folded for that).

use ahash::AHashMap;
use flame_core::{
    LoadError, LoadResult, ProcessId, ProfileBuilder, TraceSource, TrackId, TrackKind,
};
use serde::Deserialize;

pub struct ChromeSource;

pub const SOURCE: &dyn TraceSource = &ChromeSource;

#[derive(Deserialize, Debug)]
struct Event {
    #[serde(default)]
    name: String,
    #[serde(default)]
    cat: String,
    /// Phase. Single character.
    ph: String,
    /// Timestamp in microseconds.
    #[serde(default)]
    ts: f64,
    /// Duration (X phase) in microseconds.
    #[serde(default)]
    dur: Option<f64>,
    #[serde(default)]
    pid: Option<i64>,
    #[serde(default)]
    tid: Option<i64>,
    /// Metadata payload (for `M` events).
    #[serde(default)]
    args: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum TraceFile {
    Object {
        #[serde(rename = "traceEvents")]
        trace_events: Vec<Event>,
    },
    Array(Vec<Event>),
}

impl TraceSource for ChromeSource {
    fn name(&self) -> &'static str {
        "Chrome Trace JSON"
    }

    fn detect(&self, input: &[u8], filename: Option<&str>) -> bool {
        // Look at the first non-whitespace byte.
        let head: &[u8] = &input[..input.len().min(8192)];
        let mut i = 0;
        while i < head.len() && head[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= head.len() {
            return false;
        }
        let first = head[i] as char;
        if first != '[' && first != '{' {
            return false;
        }

        // Strong signal: substring `"ph"` or `"traceEvents"` near the front.
        let s = std::str::from_utf8(head).unwrap_or("");
        if s.contains("\"traceEvents\"") || s.contains("\"ph\"") {
            return true;
        }

        // Weaker signal: known extensions only when JSON-shaped.
        if let Some(name) = filename {
            let lower = name.to_ascii_lowercase();
            if lower.ends_with(".trace.json")
                || lower.ends_with(".chrome.json")
                || lower.ends_with(".ctf.json")
            {
                return true;
            }
        }
        false
    }

    fn load(&self, input: &[u8], builder: &mut ProfileBuilder) -> LoadResult<()> {
        let parsed: TraceFile = serde_json::from_slice(input)
            .map_err(|e| LoadError::Parse(format!("chrome json: {e}")))?;
        let events = match parsed {
            TraceFile::Array(v) => v,
            TraceFile::Object { trace_events } => trace_events,
        };

        // Per-(pid, tid) state: track id, plus a stack of currently-open `X`
        // (complete) events' end timestamps. B/E nesting is handled by the
        // builder's own open-stack; X events have explicit durations and don't
        // touch that stack, so we compute their depth from time containment
        // here. Assumes X events on a track arrive in start-time order with
        // parent before child, which the Chrome spec requires.
        struct ThreadState {
            track: TrackId,
            open_x_end_ns: Vec<u64>,
        }
        let mut state: AHashMap<(i64, i64), ThreadState> = AHashMap::new();
        let mut process_ids: AHashMap<i64, ProcessId> = AHashMap::new();

        let mut warned_async = false;
        let mut warned_flow = false;
        let mut warned_counter = false;
        let mut warned_unknown = false;

        for ev in &events {
            let pid = ev.pid.unwrap_or(0);
            let tid = ev.tid.unwrap_or(0);

            // Metadata phase: set process or thread names. No timestamp use.
            if ev.ph == "M" {
                if ev.name == "thread_name" {
                    let name = extract_args_string(&ev.args, "name");
                    let proc_id = *process_ids
                        .entry(pid)
                        .or_insert_with(|| builder.add_process(pid, ""));
                    let thread_id = builder.add_thread(Some(proc_id), tid, name.as_deref().unwrap_or(""));
                    // Refresh the track's name if we already created one.
                    if let Some(s) = state.get(&(pid, tid)) {
                        let _ = s; // track name not currently mutable post-creation; v2 polish
                    }
                    let _ = thread_id;
                } else if ev.name == "process_name" {
                    let name = extract_args_string(&ev.args, "name");
                    let proc_id = builder.add_process(pid, name.as_deref().unwrap_or(""));
                    process_ids.insert(pid, proc_id);
                }
                continue;
            }

            // Get-or-create track for (pid, tid).
            let st = state.entry((pid, tid)).or_insert_with(|| {
                let proc_id = *process_ids
                    .entry(pid)
                    .or_insert_with(|| builder.add_process(pid, ""));
                let thread = builder.add_thread(Some(proc_id), tid, "");
                let label = format!("PID {pid} TID {tid}");
                let track = builder.add_track(TrackKind::Thread(thread), &label, None);
                ThreadState { track, open_x_end_ns: Vec::new() }
            });
            let track = st.track;

            // Convert µs -> ns.
            let ts_ns = (ev.ts * 1000.0).max(0.0) as u64;
            let category = if ev.cat.is_empty() {
                flame_core::CategoryId::DEFAULT
            } else {
                builder.intern_category(&ev.cat)
            };

            match ev.ph.as_str() {
                "B" => {
                    let name = builder.intern_string(&ev.name);
                    builder.begin_slice(track, ts_ns, name, category);
                }
                "E" => {
                    builder.end_slice(track, ts_ns);
                }
                "X" => {
                    let dur_ns = ev.dur.unwrap_or(0.0).max(0.0) * 1000.0;
                    let dur_ns = dur_ns as u64;
                    let end_ns = ts_ns + dur_ns;
                    let name = builder.intern_string(&ev.name);
                    // Pop any open X parents that ended at or before this event's
                    // start, then nest under whatever's still open.
                    while st.open_x_end_ns.last().map_or(false, |&e| e <= ts_ns) {
                        st.open_x_end_ns.pop();
                    }
                    let depth = builder.open_stack_depth(track) + st.open_x_end_ns.len() as u16;
                    builder.add_complete_slice(track, depth, ts_ns, dur_ns, name, category, None);
                    st.open_x_end_ns.push(end_ns);
                }
                "i" | "I" => {
                    let name = builder.intern_string(&ev.name);
                    while st.open_x_end_ns.last().map_or(false, |&e| e <= ts_ns) {
                        st.open_x_end_ns.pop();
                    }
                    let depth = builder.open_stack_depth(track) + st.open_x_end_ns.len() as u16;
                    // Zero-width slice as instant marker. Renderer can decide to draw
                    // these specially in v2; v1 just shows them as a 1px line.
                    builder.add_complete_slice(track, depth, ts_ns, 0, name, category, None);
                }
                "b" | "e" | "n" => {
                    if !warned_async {
                        log::warn!("chrome: async events (b/e/n) not yet supported, skipping");
                        warned_async = true;
                    }
                }
                "s" | "t" | "f" => {
                    if !warned_flow {
                        log::warn!("chrome: flow events (s/t/f) not yet supported, skipping");
                        warned_flow = true;
                    }
                }
                "C" => {
                    if !warned_counter {
                        log::warn!("chrome: counter events (C) not yet supported, skipping");
                        warned_counter = true;
                    }
                }
                _ => {
                    if !warned_unknown {
                        log::warn!("chrome: phase {:?} not handled, skipping", ev.ph);
                        warned_unknown = true;
                    }
                }
            }
        }

        // Close any still-open B/E pairs at the max ts seen so far. The builder
        // tracks max_ts; expose via close_open_slices.
        builder.close_open_slices_at_max();

        Ok(())
    }
}

fn extract_args_string(args: &Option<serde_json::Value>, key: &str) -> Option<String> {
    args.as_ref()
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_b_e_pair() {
        let input = br#"[
            {"name":"foo","ph":"B","ts":0,"pid":1,"tid":1},
            {"name":"bar","ph":"B","ts":10,"pid":1,"tid":1},
            {"name":"bar","ph":"E","ts":20,"pid":1,"tid":1},
            {"name":"foo","ph":"E","ts":30,"pid":1,"tid":1}
        ]"#;
        assert!(ChromeSource.detect(input, None));
        let mut b = ProfileBuilder::new();
        ChromeSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 2);
        assert_eq!(p.duration_ns(), 30_000); // 30µs -> 30000ns
    }

    #[test]
    fn x_events_nest_by_containment() {
        // Parent (dur=100) fully contains child (ts=10, dur=50).
        let input = br#"{"traceEvents":[
            {"name":"parent","ph":"X","ts":0,"dur":100,"pid":1,"tid":1},
            {"name":"child","ph":"X","ts":10,"dur":50,"pid":1,"tid":1},
            {"name":"sibling","ph":"X","ts":200,"dur":5,"pid":1,"tid":1}
        ]}"#;
        let mut b = ProfileBuilder::new();
        ChromeSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 3);
        // finish() sorts by (track, depth, start_ns): parent, sibling, child.
        let mut by_start: Vec<(u64, u16)> = (0..p.slices.len())
            .map(|i| (p.slices.start_ns[i], p.slices.depth[i]))
            .collect();
        by_start.sort_by_key(|&(s, _)| s);
        assert_eq!(by_start, vec![(0, 0), (10_000, 1), (200_000, 0)]);
    }

    #[test]
    fn parse_x_complete() {
        let input = br#"{"traceEvents":[
            {"name":"foo","ph":"X","ts":0,"dur":15,"pid":1,"tid":1}
        ]}"#;
        let mut b = ProfileBuilder::new();
        ChromeSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 1);
        assert_eq!(p.slices.dur_ns[0], 15_000);
    }
}
