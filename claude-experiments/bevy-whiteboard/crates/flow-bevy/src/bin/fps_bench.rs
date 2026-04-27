//! Windowed FPS benchmark for flow-bevy.
//!
//! Spins up the real app — full GPU pipeline, real wgpu submission, real
//! present — with vsync disabled so frame rate is bounded by what the
//! engine can actually push, not the display refresh. Loads a
//! `.whiteboard` canvas, forces `HideAll` off so the packet cloud
//! actually rasterises (canvases like `life_30x30_random` set
//! `hide_all: true` in `visual.json`), samples per-frame
//! `Time<Real>::delta` for the requested duration, and prints
//! frame-time stats on `AppExit`.
//!
//! Usage:
//!     cargo run --release -p flow-bevy --bin fps_bench -- \
//!         examples/life_30x30_random.whiteboard [duration_seconds]

use std::collections::HashMap;
use std::path::PathBuf;

use bevy::app::AppExit;
use bevy::dev_tools::fps_overlay::{FpsOverlayConfig, FpsOverlayPlugin};
use bevy::diagnostic::{
    DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::window::PresentMode;

use flow_bevy::edges::HideAll;
use flow_bevy::perf::{print_report, PhaseTimings, WorkerPerf};
use flow_bevy::{CanvasSeedPlugin, FlowBevyPlugins};
use bevy::app::Last;

const WARMUP_SECS: f64 = 1.0;
const DEFAULT_DURATION_SECS: f64 = 10.0;
/// How many of the slowest frames to dump with full per-phase breakdown.
const SLOW_FRAME_DUMP: usize = 10;

fn main() {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args
        .next()
        .expect("usage: fps_bench <whiteboard_path> [duration_seconds]")
        .into();
    let duration_s: f64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_DURATION_SECS);

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "flow-bevy fps_bench".into(),
            resolution: (1400u32, 900u32).into(),
            present_mode: PresentMode::AutoNoVsync,
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(FlowBevyPlugins);
    app.add_plugins(CanvasSeedPlugin(path.clone()));
    app.add_plugins(RenderDiagnosticsPlugin);
    app.add_plugins(EntityCountDiagnosticsPlugin::default());
    // FrameTimeDiagnosticsPlugin computes the FPS / frame-time
    // numbers; FpsOverlayPlugin paints them as a HUD in the corner.
    app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    app.add_plugins(FpsOverlayPlugin {
        config: FpsOverlayConfig {
            text_config: TextFont { font_size: 16.0, ..default() },
            ..default()
        },
    });

    app.insert_resource(BenchState {
        path,
        duration_s,
        samples: Vec::with_capacity(60_000),
        per_frame: Vec::with_capacity(60_000),
        last_phase_lens: HashMap::new(),
        reported: false,
    });
    app.init_resource::<DiagSamples>();
    app.add_systems(Update, force_show_packets);
    // Per-frame capture runs in `Last`, after every Update/PostUpdate
    // system has recorded into `PhaseTimings` for THIS frame —
    // including the schedule-boundary markers in PerfPlugin
    // (transform_propagate, visibility_propagate, main_world_total).
    // Capturing in Update would smear samples across frame boundaries.
    app.add_systems(
        Last,
        (sample_diagnostics, sample_and_maybe_exit)
            .chain()
            .after(flow_bevy::perf::finish_main_world),
    );
    app.run();
}

/// Per-frame snapshot of phase timings: the sum of any new samples
/// recorded into `PhaseTimings` since the previous frame, keyed by
/// phase name. Only main-thread phases — worker-thread sim samples
/// live in `WorkerPerf` and are not on the frame's critical path.
struct FrameRecord {
    frame_ms: f64,
    phases: Vec<(&'static str, f64)>,
}

#[derive(Resource)]
struct BenchState {
    path: PathBuf,
    duration_s: f64,
    samples: Vec<f64>,
    per_frame: Vec<FrameRecord>,
    /// Length of each phase's sample vec at the end of the previous
    /// frame, so we can compute "what got recorded this frame" by
    /// delta. Sims that fire 0 or 50 times this frame both work
    /// correctly — we just sum the new samples.
    last_phase_lens: HashMap<&'static str, usize>,
    reported: bool,
}

fn force_show_packets(mut hide: ResMut<HideAll>) {
    if hide.0 {
        hide.0 = false;
    }
}

#[derive(Resource, Default)]
struct DiagSamples {
    samples: HashMap<String, (Option<&'static str>, Vec<f64>)>,
    last_seen: HashMap<String, f64>,
}

fn sample_diagnostics(
    diagnostics: Res<DiagnosticsStore>,
    state: Res<BenchState>,
    mut diag: ResMut<DiagSamples>,
) {
    if state.reported || state.samples.is_empty() {
        return;
    }
    for d in diagnostics.iter() {
        let path = d.path().as_str();
        if path.starts_with("fps")
            || path.starts_with("frame_time")
            || path.starts_with("frame_count")
        {
            continue;
        }
        let Some(value) = d.value() else { continue };
        if let Some(prev) = diag.last_seen.get(path) {
            if (*prev - value).abs() < f64::EPSILON {
                continue;
            }
        }
        diag.last_seen.insert(path.to_string(), value);
        diag.samples
            .entry(path.to_string())
            .or_insert_with(|| (Some("ms"), Vec::with_capacity(1024)))
            .1
            .push(value);
    }
}

fn print_diag_report(diag: &DiagSamples) {
    if diag.samples.is_empty() {
        return;
    }
    let mut rows: Vec<(&String, &Vec<f64>)> = diag
        .samples
        .iter()
        .map(|(k, (_, v))| (k, v))
        .filter(|(_, v)| !v.is_empty())
        .collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));

    println!();
    println!("=== bevy diagnostics ===");
    println!(
        "  {:<58} {:>7}  {:>10} {:>10} {:>10} {:>10} {:>10}",
        "path", "samples", "mean", "p50", "p95", "p99", "max",
    );
    for (path, vs) in rows {
        let mut sorted = vs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let q = |p: f64| sorted[((n - 1) as f64 * p) as usize];
        println!(
            "  {:<58} {:>7}  {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            path,
            n,
            mean,
            q(0.50),
            q(0.95),
            q(0.99),
            sorted[n - 1],
        );
    }
}

fn sample_and_maybe_exit(
    time: Res<Time<Real>>,
    mut state: ResMut<BenchState>,
    mut timings: ResMut<PhaseTimings>,
    mut worker_perf: ResMut<WorkerPerf>,
    mut diag: ResMut<DiagSamples>,
    mut exit: bevy::ecs::message::MessageWriter<AppExit>,
) {
    let elapsed = time.elapsed_secs_f64();
    let dt_ms = time.delta_secs_f64() * 1000.0;

    if state.reported {
        return;
    }
    if elapsed >= WARMUP_SECS && state.samples.is_empty() {
        *timings = PhaseTimings::default();
        *worker_perf = WorkerPerf::default();
        *diag = DiagSamples::default();
        state.last_phase_lens.clear();
    }
    if elapsed >= WARMUP_SECS {
        // Record this frame's per-phase deltas (main-thread phases only).
        let phases = timings.delta_since(&mut state.last_phase_lens);
        state.per_frame.push(FrameRecord {
            frame_ms: dt_ms,
            phases,
        });
        state.samples.push(dt_ms);
    }
    if elapsed >= WARMUP_SECS + state.duration_s {
        report(&state);
        print_slowest_frames(&state);
        println!();
        println!("=== aggregated main-thread phase timings ===");
        print_report(&timings);
        println!();
        println!("=== worker-thread sim sub-phases (NOT on frame critical path) ===");
        print_report(&worker_perf.0);
        print_diag_report(&diag);
        state.reported = true;
        exit.write(AppExit::Success);
    }
}

fn report(state: &BenchState) {
    let n = state.samples.len();
    if n == 0 {
        eprintln!("fps_bench: no frames captured (warmup {WARMUP_SECS}s + {} s)", state.duration_s);
        return;
    }
    let mut sorted = state.samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
    let q = |p: f64| sorted[((n - 1) as f64 * p) as usize];

    let line = |label: &str, ms: f64| {
        println!("  {label:<14}: {ms:7.3} ms  ({:6.1} fps)", 1000.0 / ms);
    };

    println!();
    println!("=== fps_bench: {:?} ===", state.path);
    println!("  frames        : {n}");
    println!("  window        : {:.2}s after {:.2}s warmup", state.duration_s, WARMUP_SECS);
    line("mean", mean);
    line("p50", q(0.50));
    line("p95", q(0.95));
    line("p99", q(0.99));
    line("max", sorted[n - 1]);
}

/// Sort frames by frame_ms desc, print the top N with their per-phase
/// breakdown so we can see which phases were live on slow frames vs.
/// fast ones.
fn print_slowest_frames(state: &BenchState) {
    if state.per_frame.is_empty() {
        return;
    }
    let mut sorted_idx: Vec<usize> = (0..state.per_frame.len()).collect();
    sorted_idx.sort_by(|a, b| {
        state.per_frame[*b]
            .frame_ms
            .partial_cmp(&state.per_frame[*a].frame_ms)
            .unwrap()
    });

    // Collect every phase name we ever saw, sorted, so the output
    // table has a consistent column layout.
    let mut all_phases: Vec<&'static str> = state
        .per_frame
        .iter()
        .flat_map(|f| f.phases.iter().map(|(n, _)| *n))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    all_phases.sort();

    println!();
    println!(
        "=== slowest {} frames (per-frame main-thread phase breakdown, ms) ===",
        SLOW_FRAME_DUMP.min(state.per_frame.len())
    );
    print!("  {:<6} {:>10}", "frame#", "frame_ms");
    for p in &all_phases {
        print!("  {:>10}", short_phase(p));
    }
    println!();

    let take = SLOW_FRAME_DUMP.min(state.per_frame.len());
    for &i in sorted_idx.iter().take(take) {
        let r = &state.per_frame[i];
        let phase_map: HashMap<&'static str, f64> = r.phases.iter().copied().collect();
        let mut accounted = 0.0_f64;
        print!("  {:<6} {:>10.3}", i, r.frame_ms);
        for p in &all_phases {
            let v_ms = phase_map.get(p).copied().unwrap_or(0.0) / 1000.0;
            accounted += v_ms;
            print!("  {:>10.3}", v_ms);
        }
        let unaccounted = (r.frame_ms - accounted).max(0.0);
        println!("  | unacct {:.3} ms ({:.0}%)", unaccounted, 100.0 * unaccounted / r.frame_ms);
    }

    // Also print the median frame for comparison so the slow ones
    // have a baseline.
    let mut by_frame_ms = state.per_frame.iter().collect::<Vec<_>>();
    by_frame_ms.sort_by(|a, b| a.frame_ms.partial_cmp(&b.frame_ms).unwrap());
    let mid = &by_frame_ms[by_frame_ms.len() / 2];
    let phase_map: HashMap<&'static str, f64> = mid.phases.iter().copied().collect();
    let mut accounted = 0.0_f64;
    print!("\n  {:<6} {:>10.3}", "p50", mid.frame_ms);
    for p in &all_phases {
        let v_ms = phase_map.get(p).copied().unwrap_or(0.0) / 1000.0;
        accounted += v_ms;
        print!("  {:>10.3}", v_ms);
    }
    let unaccounted = (mid.frame_ms - accounted).max(0.0);
    println!("  | unacct {:.3} ms ({:.0}%)  [median frame]", unaccounted, 100.0 * unaccounted / mid.frame_ms);
}

/// Compress a phase name into something fitting a 10-char column.
fn short_phase(name: &str) -> String {
    let trimmed = name
        .strip_prefix("edges.").map(|s| format!("ed.{s}"))
        .or_else(|| name.strip_prefix("nodes.").map(|s| format!("nd.{s}")))
        .or_else(|| name.strip_prefix("packet_cloud.").map(|s| format!("pc.{s}")))
        .or_else(|| name.strip_prefix("bridge.").map(|s| format!("br.{s}")))
        .or_else(|| name.strip_prefix("sim.").map(|s| format!("si.{s}")))
        .unwrap_or_else(|| name.to_string());
    if trimmed.len() > 10 { trimmed[..10].to_string() } else { trimmed }
}
