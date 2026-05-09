//! Speedscope's own JSON format. Spec:
//! <https://github.com/jlfwong/speedscope/blob/main/src/lib/file-format-spec.ts>
//!
//! Top-level shape:
//! ```json
//! { "$schema": "https://www.speedscope.app/file-format-schema.json",
//!   "shared": { "frames": [{"name":..., "file":..., "line":...}, ...] },
//!   "profiles": [ <EventedProfile | SampledProfile>, ... ] }
//! ```
//!
//! - **EventedProfile**: `events: [{type:"O"|"C", frame: idx, at: time}, ...]`
//!   open/close pairs that nest into a flame chart.
//! - **SampledProfile**: `samples: [[frameIdx, ...], ...]` plus parallel
//!   `weights: [number, ...]`. Each sample is a stack root-first.

use flame_core::{
    LoadError, LoadResult, ProcessId, ProfileBuilder, TraceSource, TrackId, TrackKind,
};
use serde::Deserialize;

pub struct SpeedscopeSource;

pub const SOURCE: &dyn TraceSource = &SpeedscopeSource;

#[derive(Deserialize, Debug)]
struct File {
    #[serde(default)]
    name: Option<String>,
    shared: Shared,
    profiles: Vec<Profile>,
}

#[derive(Deserialize, Debug)]
struct Shared {
    frames: Vec<FrameSpec>,
}

#[derive(Deserialize, Debug, Default)]
struct FrameSpec {
    #[serde(default)]
    name: String,
    #[serde(default)]
    file: String,
    #[serde(default)]
    line: Option<u32>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum Profile {
    #[serde(rename = "evented")]
    Evented {
        #[serde(default)]
        name: Option<String>,
        unit: String,
        #[serde(rename = "startValue", default)]
        start_value: f64,
        #[serde(rename = "endValue", default)]
        end_value: f64,
        events: Vec<EventedEvent>,
    },
    #[serde(rename = "sampled")]
    Sampled {
        #[serde(default)]
        name: Option<String>,
        unit: String,
        #[serde(rename = "startValue", default)]
        start_value: f64,
        #[serde(rename = "endValue", default)]
        end_value: f64,
        samples: Vec<Vec<u32>>,
        weights: Vec<f64>,
    },
}

#[derive(Deserialize, Debug)]
struct EventedEvent {
    #[serde(rename = "type")]
    kind: String, // "O" open, "C" close
    frame: u32,
    at: f64,
}

impl TraceSource for SpeedscopeSource {
    fn name(&self) -> &'static str {
        "Speedscope JSON"
    }

    fn detect(&self, input: &[u8], filename: Option<&str>) -> bool {
        let head: &[u8] = &input[..input.len().min(8192)];
        let s = std::str::from_utf8(head).unwrap_or("");
        if s.contains("speedscope.app/file-format") || s.contains("\"shared\"") && s.contains("\"profiles\"") {
            return true;
        }
        if let Some(name) = filename {
            let lower = name.to_ascii_lowercase();
            if lower.ends_with(".speedscope.json") {
                return true;
            }
        }
        false
    }

    fn load(&self, input: &[u8], builder: &mut ProfileBuilder) -> LoadResult<()> {
        let file: File = serde_json::from_slice(input)
            .map_err(|e| LoadError::Parse(format!("speedscope json: {e}")))?;

        let process = builder.add_process(0, file.name.as_deref().unwrap_or("speedscope"));

        // Pre-intern frames into our FrameId space.
        let mut frame_map = Vec::with_capacity(file.shared.frames.len());
        for f in &file.shared.frames {
            let fid = builder.intern_frame(&f.name, &f.file, f.line.unwrap_or(0), 0);
            frame_map.push(fid);
        }

        for (i, p) in file.profiles.iter().enumerate() {
            match p {
                Profile::Evented {
                    name, unit, start_value, end_value: _, events,
                } => {
                    load_evented(
                        builder, process, i, name.as_deref(), unit, *start_value, events, &frame_map,
                    )?;
                }
                Profile::Sampled {
                    name, unit, start_value: _, end_value: _, samples, weights,
                } => {
                    load_sampled(
                        builder, process, i, name.as_deref(), unit, samples, weights, &frame_map,
                    )?;
                }
            }
        }

        builder.close_open_slices_at_max();
        Ok(())
    }
}

fn unit_to_ns_factor(unit: &str) -> f64 {
    match unit {
        "nanoseconds" => 1.0,
        "microseconds" => 1_000.0,
        "milliseconds" => 1_000_000.0,
        "seconds" => 1_000_000_000.0,
        // Speedscope's "none" unit means values are dimensionless. Treat as ns so
        // the timeline still has *some* scale; the user can read it as opaque units.
        _ => 1.0,
    }
}

fn track_for_profile(
    builder: &mut ProfileBuilder,
    process: ProcessId,
    profile_idx: usize,
    name: Option<&str>,
) -> TrackId {
    let label = name.map(String::from).unwrap_or_else(|| format!("profile {profile_idx}"));
    let thread = builder.add_thread(Some(process), profile_idx as i64, &label);
    builder.add_track(TrackKind::Thread(thread), &label, None)
}

fn load_evented(
    builder: &mut ProfileBuilder,
    process: ProcessId,
    profile_idx: usize,
    name: Option<&str>,
    unit: &str,
    start_value: f64,
    events: &[EventedEvent],
    frame_map: &[flame_core::FrameId],
) -> LoadResult<()> {
    let track = track_for_profile(builder, process, profile_idx, name);
    let category = builder.intern_category("evented");
    let factor = unit_to_ns_factor(unit);

    for ev in events {
        let frame_idx = ev.frame as usize;
        if frame_idx >= frame_map.len() {
            return Err(LoadError::Parse(format!(
                "speedscope evented: frame index {frame_idx} out of range"
            )));
        }
        let ts_ns = ((ev.at - start_value).max(0.0) * factor) as u64;
        match ev.kind.as_str() {
            "O" => {
                let fname = builder.stacks_frame_name(frame_map[frame_idx]);
                builder.begin_slice(track, ts_ns, fname, category);
            }
            "C" => {
                builder.end_slice(track, ts_ns);
            }
            other => {
                return Err(LoadError::Parse(format!(
                    "speedscope evented: unknown event type {other:?}"
                )));
            }
        }
    }
    Ok(())
}

fn load_sampled(
    builder: &mut ProfileBuilder,
    process: ProcessId,
    profile_idx: usize,
    name: Option<&str>,
    unit: &str,
    samples: &[Vec<u32>],
    weights: &[f64],
    frame_map: &[flame_core::FrameId],
) -> LoadResult<()> {
    if samples.len() != weights.len() {
        return Err(LoadError::Parse(format!(
            "speedscope sampled: samples ({}) != weights ({})",
            samples.len(),
            weights.len()
        )));
    }
    let track = track_for_profile(builder, process, profile_idx, name);
    let category = builder.intern_category("sampled");
    let factor = unit_to_ns_factor(unit);

    // Walk each sample as a root-first stack. Lay out timeline-wise: timestamp
    // increments by the cumulative weight (sample weights are per-sample
    // durations in `unit`). For a true flame chart, slices are at consecutive
    // ts. We emit one synthesized "complete" slice per (depth, run-of-equal-stack-prefix).
    //
    // Simpler v1 implementation: emit one slice per sample per frame at depth.
    // That gives correct widths but lots of adjacent slices with the same name —
    // the renderer can merge visually later. For now we collapse adjacent
    // identical (depth, frame) runs into single slices.
    let mut t_ns: u64 = 0;
    let mut prev_run: Vec<(u32, u64)> = Vec::new(); // (frame_idx, start_ns) per depth

    for (s_idx, stack) in samples.iter().enumerate() {
        let dur_ns = (weights[s_idx].max(0.0) * factor) as u64;
        let next_t = t_ns + dur_ns;

        // For each depth: if the frame matches prev_run[depth], extend; else
        // flush prev_run from this depth onward into emitted slices and restart.
        // We can't extend after the fact (slices are emit-once), so the standard
        // trick is to emit only when the run *changes* — track open runs and
        // close them when they no longer match.
        let common_prefix = stack
            .iter()
            .zip(prev_run.iter())
            .take_while(|(a, b)| **a == b.0)
            .count();

        // Close runs deeper than common_prefix.
        for depth in (common_prefix..prev_run.len()).rev() {
            let (frame_idx, start_ns) = prev_run[depth];
            emit_run(builder, track, depth as u16, frame_idx, frame_map, start_ns, t_ns, category);
        }
        prev_run.truncate(common_prefix);

        // Open new runs for the deeper part of `stack`.
        for &fid_idx in &stack[common_prefix..] {
            prev_run.push((fid_idx, t_ns));
        }

        t_ns = next_t;
    }

    // Close any remaining runs at the end timestamp.
    for depth in (0..prev_run.len()).rev() {
        let (frame_idx, start_ns) = prev_run[depth];
        emit_run(builder, track, depth as u16, frame_idx, frame_map, start_ns, t_ns, category);
    }
    Ok(())
}

fn emit_run(
    builder: &mut ProfileBuilder,
    track: TrackId,
    depth: u16,
    frame_idx: u32,
    frame_map: &[flame_core::FrameId],
    start_ns: u64,
    end_ns: u64,
    category: flame_core::CategoryId,
) {
    let Some(&fid) = frame_map.get(frame_idx as usize) else { return };
    let name = builder.stacks_frame_name(fid);
    builder.add_complete_slice(track, depth, start_ns, end_ns - start_ns, name, category, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_evented() {
        let input = br#"{
            "$schema":"https://www.speedscope.app/file-format-schema.json",
            "shared":{"frames":[{"name":"main"},{"name":"work"}]},
            "profiles":[{
                "type":"evented","name":"p","unit":"microseconds",
                "startValue":0,"endValue":100,
                "events":[
                    {"type":"O","frame":0,"at":0},
                    {"type":"O","frame":1,"at":10},
                    {"type":"C","frame":1,"at":50},
                    {"type":"C","frame":0,"at":100}
                ]
            }]
        }"#;
        assert!(SpeedscopeSource.detect(input, None));
        let mut b = ProfileBuilder::new();
        SpeedscopeSource.load(input, &mut b).unwrap();
        let p = b.finish();
        assert_eq!(p.slices.len(), 2);
        assert_eq!(p.duration_ns(), 100_000); // 100µs in ns
    }

    #[test]
    fn parse_sampled() {
        let input = br#"{
            "$schema":"https://www.speedscope.app/file-format-schema.json",
            "shared":{"frames":[{"name":"a"},{"name":"b"},{"name":"c"}]},
            "profiles":[{
                "type":"sampled","name":"p","unit":"milliseconds",
                "startValue":0,"endValue":3,
                "samples":[[0,1],[0,1],[0,2]],
                "weights":[1,1,1]
            }]
        }"#;
        let mut b = ProfileBuilder::new();
        SpeedscopeSource.load(input, &mut b).unwrap();
        let p = b.finish();
        // Expected: depth 0 = a (3ms), depth 1 = b (2ms run) + c (1ms run) = 3 slices.
        assert_eq!(p.slices.len(), 3);
        assert_eq!(p.duration_ns(), 3_000_000);
    }
}
