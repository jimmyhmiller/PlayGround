//! Firefox Profiler / Gecko **processed** profile JSON. This is what `samply`
//! emits and what profiler.firefox.com loads.
//!
//! Supports both layouts:
//! - **v60+** with hoisted `shared` tables: `shared.{stackTable, frameTable,
//!   funcTable, resourceTable, stringArray}`.
//! - **v24+** with per-thread tables (older fixtures).
//!
//! Source of truth: `firefox-devtools/profiler` `src/types/profile.ts`.
//!
//! All tables are column-array: `frameTable: { length, func: [...], line: [...], ... }`.
//!
//! Sample → name resolution chain:
//! ```text
//! samples.stack[i] -> stackTable.frame[s] -> frameTable.func[f]
//!                  -> funcTable.name[fn]  -> stringArray[sn]
//! ```
//!
//! Slices are synthesized from sample runs: consecutive samples that share the
//! same stack-prefix at depth `d` collapse into a single slice. Markers (Phase
//! 1 = Interval) become their own slices on a separate marker track per thread.

#![allow(dead_code)]

use flame_core::{
    CategoryId, FrameId, LoadError, LoadResult, ProcessId, ProfileBuilder, TraceSource, TrackId,
    TrackKind,
};
use serde::Deserialize;

pub struct FirefoxSource;

pub const SOURCE: &dyn TraceSource = &FirefoxSource;

#[derive(Deserialize, Debug)]
struct File {
    meta: Meta,
    #[serde(default)]
    libs: Vec<serde_json::Value>,
    #[serde(default)]
    shared: Option<SharedTables>,
    threads: Vec<RawThread>,
}

#[derive(Deserialize, Debug)]
struct Meta {
    /// Sampling interval in ms (informational; we don't use it for layout).
    #[serde(default)]
    interval: f64,
    /// Trace start time in unix epoch ms. We rebase samples to t=0 = startTime.
    #[serde(default)]
    #[serde(rename = "startTime")]
    start_time: f64,
    /// Schema version. Determines whether shared tables are hoisted.
    #[serde(default)]
    #[serde(rename = "preprocessedProfileVersion")]
    preprocessed_profile_version: u32,
    #[serde(default)]
    categories: Vec<MetaCategory>,
}

#[derive(Deserialize, Debug, Default)]
struct MetaCategory {
    #[serde(default)]
    name: String,
    #[serde(default)]
    color: String,
}

/// In v60+, these tables are hoisted out of each thread.
#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct SharedTables {
    #[serde(rename = "stringArray")]
    string_array: Vec<String>,
    #[serde(rename = "stackTable")]
    stack_table: StackTableJson,
    #[serde(rename = "frameTable")]
    frame_table: FrameTableJson,
    #[serde(rename = "funcTable")]
    func_table: FuncTableJson,
}

#[derive(Deserialize, Debug)]
struct RawThread {
    #[serde(default)]
    name: String,
    #[serde(default)]
    pid: serde_json::Value, // can be number or string
    #[serde(default)]
    tid: serde_json::Value,

    /// Per-thread tables (v24 layout). For v60 these are `null`/missing on
    /// the thread and live under the top-level `shared`.
    #[serde(default)]
    #[serde(rename = "stringArray")]
    string_array: Option<Vec<String>>,
    #[serde(default)]
    #[serde(rename = "stackTable")]
    stack_table: Option<StackTableJson>,
    #[serde(default)]
    #[serde(rename = "frameTable")]
    frame_table: Option<FrameTableJson>,
    #[serde(default)]
    #[serde(rename = "funcTable")]
    func_table: Option<FuncTableJson>,

    samples: SamplesJson,
    #[serde(default)]
    markers: Option<MarkersJson>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct StackTableJson {
    length: usize,
    /// Each element is the parent stack index, or null/None for roots.
    prefix: Vec<Option<u32>>,
    /// Each element is an index into `frameTable`.
    frame: Vec<u32>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct FrameTableJson {
    length: usize,
    /// Index into `funcTable`.
    func: Vec<u32>,
    /// Optional line number.
    line: Vec<Option<u32>>,
    /// Optional category index into meta.categories.
    category: Vec<Option<u32>>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct FuncTableJson {
    length: usize,
    /// Index into stringArray.
    name: Vec<u32>,
    /// Optional resource (file) index. We don't fully resolve it for v1.
    #[serde(default)]
    resource: Vec<Option<i32>>,
    /// Optional source filename (string-table index) introduced in newer schemas.
    #[serde(default)]
    #[serde(rename = "fileName")]
    file_name: Vec<Option<u32>>,
    #[serde(default)]
    #[serde(rename = "lineNumber")]
    line_number: Vec<Option<u32>>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct SamplesJson {
    length: usize,
    stack: Vec<Option<u32>>,
    /// Either `time` or `timeDeltas` is present. `time` is absolute ms; `timeDeltas`
    /// accumulates from 0.
    #[serde(default)]
    time: Option<Vec<f64>>,
    #[serde(default)]
    #[serde(rename = "timeDeltas")]
    time_deltas: Option<Vec<f64>>,
    #[serde(default)]
    weight: Option<Vec<f64>>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
struct MarkersJson {
    length: usize,
    name: Vec<u32>,
    #[serde(rename = "startTime")]
    start_time: Vec<Option<f64>>,
    #[serde(rename = "endTime")]
    end_time: Vec<Option<f64>>,
    /// 0=Instant, 1=Interval, 2=IntervalStart, 3=IntervalEnd.
    phase: Vec<u8>,
    #[serde(default)]
    category: Vec<Option<u32>>,
}

impl TraceSource for FirefoxSource {
    fn name(&self) -> &'static str {
        "Firefox Profiler JSON"
    }

    fn detect(&self, input: &[u8], filename: Option<&str>) -> bool {
        let head: &[u8] = &input[..input.len().min(16384)];
        let s = std::str::from_utf8(head).unwrap_or("");
        // Strong signals: shape-specific keys that don't appear in other JSON formats.
        if s.contains("\"preprocessedProfileVersion\"")
            || s.contains("\"stackTable\"") && s.contains("\"frameTable\"")
            || s.contains("\"libs\"") && s.contains("\"threads\"") && s.contains("\"meta\"")
        {
            return true;
        }
        if let Some(name) = filename {
            let lower = name.to_ascii_lowercase();
            if lower.ends_with(".processed.json") || lower.ends_with(".profile.json") {
                return true;
            }
        }
        false
    }

    fn load(&self, input: &[u8], builder: &mut ProfileBuilder) -> LoadResult<()> {
        let file: File = serde_json::from_slice(input)
            .map_err(|e| LoadError::Parse(format!("firefox json: {e}")))?;

        // Categories from meta — interned into our category table by name.
        let category_ids: Vec<CategoryId> = file
            .meta
            .categories
            .iter()
            .map(|c| {
                let name = if c.name.is_empty() { "uncategorized" } else { &c.name };
                builder.intern_category(name)
            })
            .collect();
        let default_category = if category_ids.is_empty() {
            CategoryId::DEFAULT
        } else {
            category_ids[0]
        };

        // One process per profile (Firefox traces typically describe one process).
        // Use any thread's pid as the process id; threads without pid get 0.
        let process_id = builder.add_process(0, "firefox");

        for (thread_idx, thread) in file.threads.iter().enumerate() {
            let pid = json_to_i64(&thread.pid).unwrap_or(0);
            let tid = json_to_i64(&thread.tid).unwrap_or(thread_idx as i64);
            // If the file shows a real pid, register it (de-duped by builder).
            let proc = if pid != 0 {
                builder.add_process(pid, "firefox")
            } else {
                process_id
            };

            let label = if thread.name.is_empty() {
                format!("thread {tid}")
            } else {
                thread.name.clone()
            };
            let thread_obj = builder.add_thread(Some(proc), tid, &label);
            let track = builder.add_track(TrackKind::Thread(thread_obj), &label, None);

            // Pick the source of tables for this thread: shared (v60+) or per-thread (v24+).
            let local_strings;
            let strings: &[String];
            let stack_table;
            let frame_table;
            let func_table;
            if let Some(shared) = &file.shared {
                strings = &shared.string_array;
                stack_table = &shared.stack_table;
                frame_table = &shared.frame_table;
                func_table = &shared.func_table;
            } else {
                local_strings = thread.string_array.as_deref().unwrap_or(&[]);
                strings = local_strings;
                stack_table = thread.stack_table.as_ref().ok_or_else(|| {
                    LoadError::Parse(format!(
                        "thread {tid}: missing stackTable (and no shared)"
                    ))
                })?;
                frame_table = thread.frame_table.as_ref().ok_or_else(|| {
                    LoadError::Parse(format!("thread {tid}: missing frameTable"))
                })?;
                func_table = thread.func_table.as_ref().ok_or_else(|| {
                    LoadError::Parse(format!("thread {tid}: missing funcTable"))
                })?;
            }

            // Pre-intern frames: for each FrameTable entry, resolve func -> string.
            let mut frame_ids = Vec::with_capacity(frame_table.length);
            for fi in 0..frame_table.length {
                let func_idx = *frame_table.func.get(fi).unwrap_or(&0) as usize;
                let name_idx =
                    *func_table.name.get(func_idx).unwrap_or(&0) as usize;
                let name = strings.get(name_idx).map(|s| s.as_str()).unwrap_or("?");
                let file_str = func_table
                    .file_name
                    .get(func_idx)
                    .copied()
                    .flatten()
                    .and_then(|si| strings.get(si as usize))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let line = func_table
                    .line_number
                    .get(func_idx)
                    .copied()
                    .flatten()
                    .or_else(|| frame_table.line.get(fi).copied().flatten())
                    .unwrap_or(0);
                frame_ids.push(builder.intern_frame(name, file_str, line, 0));
            }

            // Pre-resolve stacks: for each stack node, walk prefix to root and build
            // a (depth, top_frame_id) plus a vector of all frames root-first. We
            // use this twice — for slice synthesis and for any sample stack lookup.
            //
            // We compute root_first_frames lazily on demand (only stacks that get
            // touched by samples). For typical traces only a fraction of stacks are
            // referenced; lazy is significantly cheaper.
            let stack_count = stack_table.length;
            let mut stack_depth: Vec<u16> = vec![0; stack_count];
            // Compute depth iteratively: depth(s) = depth(prefix) + 1, or 0 for root.
            // Iteration order is not topological, so do two passes if needed. The
            // typical case is that prefix < self, so a single forward pass works.
            for s in 0..stack_count {
                if let Some(p) = stack_table.prefix.get(s).copied().flatten() {
                    let p = p as usize;
                    if p < s {
                        stack_depth[s] = stack_depth[p].saturating_add(1);
                    }
                }
            }

            // Collect samples sorted by time.
            let n = thread.samples.length;
            let times = collect_times(&thread.samples);
            let stacks = &thread.samples.stack;

            // Run-length-encode by stack prefix at each depth.
            // open_runs[d] = (frame_id, start_ns) for the currently-open slice
            // at depth d.
            let mut open_runs: Vec<(FrameId, u64)> = Vec::new();
            let category = default_category;
            let mut max_depth: u16 = 0;

            // We need to compare stack-prefixes between consecutive samples. To do
            // that we materialize each sample's frames root-first.
            // (A more memory-efficient alternative is to walk just the prefix
            // chain via stack_table, but the working sets are small here.)
            let mut prev_frames: Vec<u32> = Vec::new();
            let mut next_frames: Vec<u32> = Vec::new();

            for i in 0..n {
                let t_ms = times.get(i).copied().unwrap_or(0.0);
                let t_ns = ms_to_ns(t_ms);

                next_frames.clear();
                if let Some(Some(stack_idx)) = stacks.get(i).copied().map(Some).flatten() {
                    walk_stack_root_first(
                        stack_idx,
                        stack_table,
                        &mut next_frames,
                    );
                }

                // Find longest common prefix with prev.
                let common = prev_frames
                    .iter()
                    .zip(next_frames.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                // Close runs deeper than common.
                while open_runs.len() > common {
                    let (frame_id, start_ns) = open_runs.pop().unwrap();
                    let depth = open_runs.len() as u16;
                    if t_ns > start_ns {
                        let name_id = builder.stacks_frame_name(frame_id);
                        builder.add_complete_slice(
                            track,
                            depth,
                            start_ns,
                            t_ns - start_ns,
                            name_id,
                            category,
                            None,
                        );
                    }
                }

                // Open new runs for the deeper part of next.
                for &fi in &next_frames[common..] {
                    let fid = frame_ids
                        .get(fi as usize)
                        .copied()
                        .unwrap_or(frame_ids[0]);
                    open_runs.push((fid, t_ns));
                    max_depth = max_depth.max(open_runs.len() as u16);
                }

                std::mem::swap(&mut prev_frames, &mut next_frames);
            }

            // Close any remaining open runs at the final sample time.
            let final_t_ns = times.last().copied().map(ms_to_ns).unwrap_or(0);
            while let Some((frame_id, start_ns)) = open_runs.pop() {
                let depth = open_runs.len() as u16;
                if final_t_ns > start_ns {
                    let name_id = builder.stacks_frame_name(frame_id);
                    builder.add_complete_slice(
                        track,
                        depth,
                        start_ns,
                        final_t_ns - start_ns,
                        name_id,
                        category,
                        None,
                    );
                }
            }

            // Markers — emit Interval markers (phase=1) on a sibling track.
            if let Some(markers) = &thread.markers {
                if markers.length > 0 {
                    let marker_track = emit_marker_track(
                        builder, proc, &label, markers, strings, &category_ids,
                    );
                    let _ = marker_track; // populated inside helper
                }
            }

            log::debug!(
                "firefox: thread {label} ({tid}): {} samples, {} stacks, {} frames",
                n,
                stack_count,
                frame_table.length
            );
        }

        Ok(())
    }
}

fn collect_times(s: &SamplesJson) -> Vec<f64> {
    if let Some(time) = &s.time {
        return time.clone();
    }
    if let Some(deltas) = &s.time_deltas {
        let mut t = 0.0_f64;
        let mut out = Vec::with_capacity(deltas.len());
        for d in deltas {
            t += *d;
            out.push(t);
        }
        return out;
    }
    Vec::new()
}

fn walk_stack_root_first(stack_idx: u32, st: &StackTableJson, out: &mut Vec<u32>) {
    let mut cur = Some(stack_idx);
    let mut leaf_first = Vec::new();
    while let Some(s) = cur {
        let s = s as usize;
        if s >= st.length {
            break;
        }
        let frame = *st.frame.get(s).unwrap_or(&0);
        leaf_first.push(frame);
        cur = st.prefix.get(s).copied().flatten();
    }
    leaf_first.reverse();
    out.extend_from_slice(&leaf_first);
}

fn emit_marker_track(
    builder: &mut ProfileBuilder,
    proc: ProcessId,
    thread_label: &str,
    m: &MarkersJson,
    strings: &[String],
    categories: &[CategoryId],
) -> TrackId {
    let label = format!("{thread_label} markers");
    // Synthetic thread for the markers, using process pid as group.
    let thread = builder.add_thread(Some(proc), 0, &label);
    let track = builder.add_track(TrackKind::Thread(thread), &label, None);

    let default_cat = builder.intern_category("marker");
    for i in 0..m.length {
        let phase = *m.phase.get(i).unwrap_or(&0);
        let name_idx = *m.name.get(i).unwrap_or(&0) as usize;
        let name = strings.get(name_idx).map(|s| s.as_str()).unwrap_or("marker");
        let cat = m
            .category
            .get(i)
            .copied()
            .flatten()
            .and_then(|c| categories.get(c as usize).copied())
            .unwrap_or(default_cat);

        let start_ms = m.start_time.get(i).copied().flatten();
        let end_ms = m.end_time.get(i).copied().flatten();

        let (start_ns, dur_ns) = match phase {
            // Instant
            0 => {
                let s = start_ms.or(end_ms).unwrap_or(0.0);
                (ms_to_ns(s), 0)
            }
            // Interval
            1 => {
                let s = start_ms.unwrap_or(0.0);
                let e = end_ms.unwrap_or(s);
                (ms_to_ns(s), ms_to_ns(e).saturating_sub(ms_to_ns(s)))
            }
            // IntervalStart / IntervalEnd — pairing isn't trivial; v1 emits as instants
            // (a future pass can pair by `name` if needed).
            2 | 3 => {
                let s = start_ms.or(end_ms).unwrap_or(0.0);
                (ms_to_ns(s), 0)
            }
            _ => continue,
        };

        let name_id = builder.intern_string(name);
        builder.add_complete_slice(track, 0, start_ns, dur_ns, name_id, cat, None);
    }
    track
}

fn ms_to_ns(ms: f64) -> u64 {
    if !ms.is_finite() || ms < 0.0 {
        return 0;
    }
    (ms * 1_000_000.0) as u64
}

fn json_to_i64(v: &serde_json::Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}
