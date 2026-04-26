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
use bevy::diagnostic::{DiagnosticsStore, EntityCountDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::window::PresentMode;

use flow_bevy::edges::HideAll;
use flow_bevy::perf::{print_report, PhaseTimings};
use flow_bevy::{CanvasSeedPlugin, FlowBevyPlugins};

const WARMUP_SECS: f64 = 1.0;
const DEFAULT_DURATION_SECS: f64 = 10.0;

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
    // Per-render-pass CPU spans (extract / queue / sort / opaque_2d /
    // transparent_2d / gizmo passes etc.) plus a running entity count.
    // GPU timestamps are unsupported on Metal — we get CPU-side spans
    // only, which is exactly what we want for "where is the 17 ms going"
    // (the unaccounted-for portion of our frame is CPU work in the
    // render world, not GPU work).
    app.add_plugins(RenderDiagnosticsPlugin);
    app.add_plugins(EntityCountDiagnosticsPlugin::default());

    app.insert_resource(BenchState {
        path,
        duration_s,
        samples: Vec::with_capacity(60_000),
        reported: false,
    });
    app.init_resource::<DiagSamples>();
    app.add_systems(
        Update,
        (force_show_packets, sample_diagnostics, sample_and_maybe_exit).chain(),
    );
    app.run();
}

#[derive(Resource)]
struct BenchState {
    path: PathBuf,
    duration_s: f64,
    samples: Vec<f64>,
    reported: bool,
}

fn force_show_packets(mut hide: ResMut<HideAll>) {
    if hide.0 {
        hide.0 = false;
    }
}

/// Cache of every Bevy diagnostic value we observed during the
/// measurement window, keyed by diagnostic path (e.g.
/// `render/main_transparent_pass_2d/elapsed_cpu`). Sampled per-frame
/// from `DiagnosticsStore` — same sample-stream as
/// `LogDiagnosticsPlugin` but accumulated for stat aggregation rather
/// than logged once.
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
        // Either bench finished, or we're still in warmup.
        return;
    }
    for d in diagnostics.iter() {
        let path = d.path().as_str();
        // Skip the FPS / frame_time / frame_count diagnostics — we
        // measure those directly from `Time<Real>` to avoid double-
        // counting.
        if path.starts_with("fps")
            || path.starts_with("frame_time")
            || path.starts_with("frame_count")
        {
            continue;
        }
        let Some(value) = d.value() else { continue };
        // Render diagnostics keep their last value alive across frames
        // (CPU spans only update once per render-graph run). Skip
        // duplicates so our percentiles aren't dominated by carry-over.
        if let Some(prev) = diag.last_seen.get(path) {
            if (*prev - value).abs() < f64::EPSILON {
                continue;
            }
        }
        diag.last_seen.insert(path.to_string(), value);
        let suffix = d.suffix.is_empty().then_some("").map(|_| "");
        let _ = suffix;
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
    mut diag: ResMut<DiagSamples>,
    mut exit: bevy::ecs::message::MessageWriter<AppExit>,
) {
    let elapsed = time.elapsed_secs_f64();
    let dt_ms = time.delta_secs_f64() * 1000.0;

    if state.reported {
        return;
    }
    // Drop warmup-frame samples once, the moment we cross into the
    // measurement window, so the reported stats reflect the bench
    // window only.
    if elapsed >= WARMUP_SECS && state.samples.is_empty() {
        *timings = PhaseTimings::default();
        *diag = DiagSamples::default();
    }
    if elapsed >= WARMUP_SECS {
        state.samples.push(dt_ms);
    }
    if elapsed >= WARMUP_SECS + state.duration_s {
        report(&state);
        print_report(&timings);
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
