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

        // Per-(pid, tid) state: track id, plus a *buffer* of this track's `X`
        // (complete) and instant events. B/E nesting is handled by the builder's
        // own open-stack; X events carry explicit durations and don't touch that
        // stack, so their depth is purely a function of time containment.
        //
        // We cannot compute that depth in a single streaming pass: the Chrome
        // spec recommends, but does not require, that parents are emitted before
        // children. Real producers (e.g. memscope's allocation timeline) walk the
        // tree in post-order, so a wide parent interval arrives *after* the
        // narrower child intervals it contains. A forward-only "pop ends <= start"
        // stack then buries the parent underneath its own children at too great a
        // depth, which renders as staircases, overlaps, and gaps.
        //
        // Instead we collect every X/instant event per track, then in `flush`
        // sort by (start asc, end desc) and assign depth with a containment stack.
        // That is order-independent and correct for any properly-nested input.
        // `base_depth` captures the open B/E depth at read time so X events still
        // nest under any enclosing B/E slice.
        struct XEvent {
            start_ns: u64,
            dur_ns: u64,
            name: flame_core::StringId,
            category: flame_core::CategoryId,
            base_depth: u16,
        }
        struct ThreadState {
            track: TrackId,
            x_events: Vec<XEvent>,
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
                ThreadState { track, x_events: Vec::new() }
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
                    let dur_ns = (ev.dur.unwrap_or(0.0).max(0.0) * 1000.0) as u64;
                    let name = builder.intern_string(&ev.name);
                    let base_depth = builder.open_stack_depth(track);
                    // Defer depth assignment to flush — see ThreadState docs. We
                    // cannot nest correctly until every X event on this track is
                    // known, because parents may arrive after their children.
                    st.x_events.push(XEvent { start_ns: ts_ns, dur_ns, name, category, base_depth });
                }
                "i" | "I" => {
                    let name = builder.intern_string(&ev.name);
                    let base_depth = builder.open_stack_depth(track);
                    // Zero-width slice as instant marker. Renderer can decide to draw
                    // these specially in v2; v1 just shows them as a 1px line.
                    st.x_events.push(XEvent { start_ns: ts_ns, dur_ns: 0, name, category, base_depth });
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

        // Flush each track's buffered X/instant events. Sort by (start asc, end
        // desc) so any interval that *contains* another sorts before it, then
        // assign depth with a containment stack: pop every open interval that
        // ends at or before this one's start, and this event's containment depth
        // is whatever remains open. This is independent of the order events were
        // emitted, so post-order (child-before-parent) producers nest correctly.
        for st in state.values_mut() {
            let evs = &st.x_events;
            let mut order: Vec<u32> = (0..evs.len() as u32).collect();
            order.sort_by_key(|&i| {
                let e = &evs[i as usize];
                (e.start_ns, std::cmp::Reverse(e.start_ns + e.dur_ns))
            });
            let mut open_end: Vec<u64> = Vec::new();
            for &i in &order {
                let e = &evs[i as usize];
                while open_end.last().map_or(false, |&end| end <= e.start_ns) {
                    open_end.pop();
                }
                let depth = e.base_depth + open_end.len() as u16;
                builder.add_complete_slice(
                    st.track, depth, e.start_ns, e.dur_ns, e.name, e.category, None,
                );
                open_end.push(e.start_ns + e.dur_ns);
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
    fn x_events_nest_when_parent_emitted_after_children() {
        // Post-order producer (memscope allocation timeline): the wide parent
        // interval [0,100] is emitted AFTER the two child intervals it contains.
        // A forward-only stack would bury the parent below its children; the
        // containment-sort flush must place it at depth 0 with children at 1.
        let input = br#"{"traceEvents":[
            {"name":"childA","ph":"X","ts":0,"dur":40,"pid":1,"tid":1},
            {"name":"childB","ph":"X","ts":40,"dur":60,"pid":1,"tid":1},
            {"name":"parent","ph":"X","ts":0,"dur":100,"pid":1,"tid":1}
        ]}"#;
        let mut b = ProfileBuilder::new();
        ChromeSource.load(input, &mut b).unwrap();
        let p = b.finish();
        let depth_of = |name: &str| -> u16 {
            let sid = p.strings.lookup(name).expect("name interned");
            (0..p.slices.len())
                .find(|&i| p.slices.name[i] == sid)
                .map(|i| p.slices.depth[i])
                .expect("slice present")
        };
        assert_eq!(depth_of("parent"), 0, "parent must be the containing frame");
        assert_eq!(depth_of("childA"), 1);
        assert_eq!(depth_of("childB"), 1);
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
