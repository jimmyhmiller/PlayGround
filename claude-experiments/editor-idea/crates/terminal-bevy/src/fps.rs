//! On-screen FPS / frame-time + per-pane/subsystem profiler overlay.
//! Toggle with Cmd+Shift+F.
//!
//! Two layers of detail:
//!   - top-right meter: avg fps, avg ms, and the worst single frame in
//!     the window (the hitch — a big gap from avg means an intermittent
//!     stall, not a uniformly heavy frame);
//!   - top-left breakdown: every pane (terminal/editor/widget) and shared
//!     subsystem (input/layout/chrome) that ran this frame, sorted by ms.
//!     Fed by `pane_bevy::prof`, which the instrumented systems write into.
//!
//! When the overlay is on, also forces winit into Continuous update mode
//! so the readings reflect real per-frame cost rather than reactive idle,
//! and flips `pane_bevy::prof` on so the breakdown gets data. When off,
//! winit goes back to whatever `maintain_winit_mode_for_animation` wants
//! and the profiler stops collecting (its guards become no-ops).
//!
//! Renders Text2d on the menu overlay layer so it sits above every pane.
//! Repositioned each frame against the primary window so it tracks
//! resizes without an extra event hookup.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bevy::app::MainScheduleOrder;
use bevy::camera::visibility::RenderLayers;
use bevy::diagnostic::SystemInformationDiagnosticsPlugin;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use pane_bevy::prof;
use pane_bevy::{PaneProject, PaneTitle};

use crate::projects::Projects;
use crate::{MonoFont, FONT_SIZE, MENU_OVERLAY_LAYER};

const MARGIN: f32 = 8.0;
// Above context_menu's MENU_Z (700) so the meter doesn't get hidden
// behind a context menu, but well inside the Camera2d default depth
// range (±1000). Z=10000 falls outside the frustum and renders as
// invisible — learned that the fun way.
const Z: f32 = 950.0;

pub struct FpsOverlayPlugin;

impl Plugin for FpsOverlayPlugin {
    fn build(&self, app: &mut App) {
        // Process/system CPU + memory usage, sampled on a background
        // sysinfo refresh. Feeds the CPU graph (and is cheap when idle).
        if !app.is_plugin_added::<SystemInformationDiagnosticsPlugin>() {
            app.add_plugins(SystemInformationDiagnosticsPlugin);
        }
        // `PROF_SHOW=1` brings the overlay up at startup (otherwise it's
        // off until Cmd+Shift+F). Handy for headless screenshot checks and
        // for always-on profiling sessions.
        let show_at_start = std::env::var("PROF_SHOW").is_ok();
        if show_at_start {
            prof::set_enabled(true);
        }
        // Frame-time + render-world (GPU pass) diagnostics, always on so
        // the overlay can attribute the "untracked" remainder to GPU vs
        // idle/present — that's the whole point of the chart.
        if !app.is_plugin_added::<bevy::diagnostic::FrameTimeDiagnosticsPlugin>() {
            app.add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default());
        }
        if !app.is_plugin_added::<bevy::render::diagnostic::RenderDiagnosticsPlugin>() {
            app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
        }

        app.insert_resource(FpsOverlayState {
            enabled: show_at_start,
            ..default()
        })
        .init_resource::<FrameProfileReadout>()
        .init_resource::<CpuGraph>()
        .init_resource::<ProfHistory>()
        .init_resource::<ProfChart>()
        .init_resource::<FrameClock>()
        .init_resource::<StageClock>()
        // Stamp the start of the frame's app work as early as possible so
        // `active` = (Last − First) captures the whole schedule's wall time.
        .add_systems(First, mark_frame_start)
        .add_systems(
            Update,
            (
                toggle_overlay,
                sync_overlay_visibility,
                update_overlay,
                update_prof_panel,
                update_cpu_graph,
                update_chart,
            )
                .chain()
                // `maintain_vsync_mode` and `sync_overlay_visibility` both
                // force Continuous; `maintain_winit_mode_for_animation`
                // forces reactive when idle. All three write WinitSettings,
                // so without an explicit order the executor serializes them
                // arbitrarily and the animation one can win. Run our chain
                // AFTER it so the overlay/vsync intent is authoritative.
                .after(crate::maintain_winit_mode_for_animation),
        )
        // End of frame, after every instrumented system has run:
        // snapshot the accumulator, then fold it into the history ring.
        .add_systems(Last, (collect_profile, record_history).chain());

        // Insert stage-boundary marker schedules between the main schedules
        // (they run exactly at each boundary because schedules are
        // sequential), then attach the timestamping system to each.
        {
            let mut order = app.world_mut().resource_mut::<MainScheduleOrder>();
            order.insert_before(First, ProfMark(255)); // reset before First
            order.insert_after(First, ProfMark(0)); // closes First
            order.insert_after(PreUpdate, ProfMark(1)); // closes PreUpdate
            order.insert_after(Update, ProfMark(2)); // closes Update
            order.insert_after(PostUpdate, ProfMark(3)); // closes PostUpdate
            order.insert_after(Last, ProfMark(4)); // closes Last
        }
        app.add_systems(ProfMark(255), |mut c: ResMut<StageClock>| {
            if prof::enabled() {
                c.prev = Some(Instant::now());
            }
        });
        for idx in 0..STAGE_NAMES.len() {
            app.add_systems(
                ProfMark(idx as u8),
                move |mut c: ResMut<StageClock>| stage_record(&mut c, idx),
            );
        }

        // `TBPROF=1` collects the per-pane/subsystem breakdown headlessly
        // (no overlay needed) and dumps it (plus the render diagnostics
        // wired above) to the diagnostics log every second.
        if std::env::var("TBPROF").is_ok() {
            prof::set_enabled(true);
            app.insert_resource(ProfForced);
            app.add_systems(Last, dump_profile_log.after(collect_profile));
        }
    }
}

/// Present when `TBPROF` forced the profiler on at startup, so toggling
/// the overlay off must NOT stop collection (the headless dump needs it).
#[derive(Resource)]
struct ProfForced;

/// Stamps the start of each frame's app work (in the `First` schedule).
#[derive(Resource, Default)]
struct FrameClock {
    start: Option<Instant>,
}

fn mark_frame_start(mut clock: ResMut<FrameClock>) {
    clock.start = Some(Instant::now());
}

// ---- per-schedule (stage) timing ----
//
// Bevy's built-in schedules (First, PreUpdate, Update, PostUpdate, Last)
// run strictly one after another, so a marker schedule inserted *between*
// two of them runs exactly at that boundary. By timestamping at each
// boundary we attribute 100% of the active frame to a stage — including
// the heavy engine-internal work (transform propagation, visibility,
// text/sprite prep in PostUpdate) that our per-system spans can't see.
//
// One label type, distinct values = distinct schedules (schedules are
// keyed by the label's value).
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct ProfMark(u8);

/// Stage names, indexed to match `StageClock::dur_ms`.
const STAGE_NAMES: [&str; 5] = ["First", "PreUpdate", "Update", "PostUpdate", "Last"];

#[derive(Resource, Default)]
struct StageClock {
    prev: Option<Instant>,
    /// Per-stage wall time (ms), same order as `STAGE_NAMES`.
    dur_ms: [f32; 5],
}

/// Boundary marker `idx` closes stage `idx` (0-based into `dur_ms`) and
/// opens the next. Marker `usize::MAX` is the pre-`First` reset.
fn stage_record(clock: &mut StageClock, idx: usize) {
    if !prof::enabled() {
        return;
    }
    let now = Instant::now();
    if let (Some(prev), true) = (clock.prev, idx < clock.dur_ms.len()) {
        clock.dur_ms[idx] = now.duration_since(prev).as_secs_f32() * 1000.0;
    }
    clock.prev = Some(now);
}

/// Last-frame breakdown, published by `collect_profile` for the panel and
/// the headless dump. Not reset on read so multiple consumers (overlay +
/// `TBPROF` dump) see the same frame.
///
/// Time model, all from the same frame:
///   `frame_ms`  = wall time of the whole frame (Time::delta).
///   `active_ms` = wall time the app's systems actually ran (First→Last).
///   remainder   = `frame_ms − active_ms` = main thread parked (vsync/present/idle).
/// Within active: panes + subsystems + uncategorized == active.
#[derive(Resource, Default)]
struct FrameProfileReadout {
    data: Option<prof::FrameData>,
    frame_ms: f32,
    active_ms: f32,
}

/// End-of-frame: pull the accumulated spans (which resets the global
/// accumulator for next frame) and publish them, tagged with this frame's
/// total wall time and the active (systems-running) wall time.
fn collect_profile(
    time: Res<Time<Real>>,
    clock: Res<FrameClock>,
    mut readout: ResMut<FrameProfileReadout>,
) {
    if let Some(d) = prof::take_frame() {
        readout.data = Some(d);
        readout.frame_ms = time.delta_secs() * 1000.0;
        readout.active_ms = clock
            .start
            .map(|s| s.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);
    }
}

#[derive(Component)]
struct FpsOverlay;

#[derive(Resource, Default)]
struct FpsOverlayState {
    enabled: bool,
    accum_secs: f64,
    accum_frames: u32,
    /// Largest single-frame dt seen in the current window — the hitch.
    accum_max_dt: f64,
    last_fps: f32,
    last_ms: f32,
    /// Worst frame time (ms) over the last window. Average frame time
    /// hides stalls; this is the number that explains a choppy feel.
    last_max_ms: f32,
    entity: Option<Entity>,
}

fn toggle_overlay(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<FpsOverlayState>,
    forced: Option<Res<ProfForced>>,
) {
    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    for ev in events.read() {
        if ev.state.is_pressed() && cmd && shift && matches!(ev.key_code, KeyCode::KeyF) {
            state.enabled = !state.enabled;
            state.accum_secs = 0.0;
            state.accum_frames = 0;
            state.accum_max_dt = 0.0;
            // Drive the cross-crate profiler from the same toggle so the
            // breakdown only collects (and only pays its span cost) while
            // the overlay is up — unless TBPROF forced it on headlessly,
            // in which case collection must persist regardless.
            if forced.is_none() {
                prof::set_enabled(state.enabled);
            }
        }
    }
}

fn sync_overlay_visibility(
    mut commands: Commands,
    mut state: ResMut<FpsOverlayState>,
    mut settings: ResMut<bevy::winit::WinitSettings>,
    font: Option<Res<MonoFont>>,
) {
    if state.enabled {
        // Pin to Continuous so the meter reflects real per-frame cost,
        // not reactive idle. Runs each frame so
        // `maintain_winit_mode_for_animation` can't flip it back.
        let want = bevy::winit::UpdateMode::Continuous;
        if settings.focused_mode != want {
            settings.focused_mode = want;
        }
        if settings.unfocused_mode != want {
            settings.unfocused_mode = want;
        }
        if state.entity.is_none() {
            let Some(font) = font else { return };
            let e = commands
                .spawn((
                    FpsOverlay,
                    Text2d::new("fps --"),
                    TextFont {
                        font: font.0.clone(),
                        font_size: FONT_SIZE,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 1.0, 0.4)),
                    Anchor::TOP_RIGHT,
                    Transform::from_xyz(0.0, 0.0, Z),
                    RenderLayers::layer(MENU_OVERLAY_LAYER),
                ))
                .id();
            state.entity = Some(e);
        }
    } else if let Some(e) = state.entity.take() {
        commands.entity(e).despawn();
    }
}

fn update_overlay(
    time: Res<Time<Real>>,
    windows: Query<&Window>,
    mut state: ResMut<FpsOverlayState>,
    mut q: Query<(&mut Text2d, &mut Transform), With<FpsOverlay>>,
) {
    if !state.enabled {
        return;
    }

    let dt = time.delta_secs_f64();
    state.accum_secs += dt;
    state.accum_frames += 1;
    if dt > state.accum_max_dt {
        state.accum_max_dt = dt;
    }

    if state.accum_secs >= 0.25 {
        let avg_dt = state.accum_secs / state.accum_frames as f64;
        state.last_fps = (1.0 / avg_dt) as f32;
        state.last_ms = (avg_dt * 1000.0) as f32;
        state.last_max_ms = (state.accum_max_dt * 1000.0) as f32;
        state.accum_secs = 0.0;
        state.accum_frames = 0;
        state.accum_max_dt = 0.0;
    }

    let Ok(window) = windows.single() else { return };
    let win_w = window.width();
    let win_h = window.height();

    let Ok((mut text, mut tx)) = q.single_mut() else {
        return;
    };
    // avg frame time + worst frame in the window. A big gap between the
    // two (e.g. 16 ms avg / 120 ms max) means an intermittent stall, not
    // a uniformly heavy frame.
    text.0 = format!(
        "{:>5.1} fps  {:>5.2} ms avg  {:>6.2} ms max",
        state.last_fps, state.last_ms, state.last_max_ms,
    );
    tx.translation.x = win_w * 0.5 - MARGIN;
    tx.translation.y = win_h * 0.5 - MARGIN;
}

#[derive(Component)]
struct ProfPanel;

/// Max rows of each kind so the panel can't run off the screen on a busy
/// canvas; the lists are pre-sorted by cost so the truncation drops the
/// cheapest, and a trailing "+N more" line keeps the omission honest.
const MAX_ROWS: usize = 12;

fn ms(d: Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

/// Builds the top-left per-pane + subsystem table from the last frame's
/// accumulator, resolving each pane Entity to a human label via its
/// `PaneTitle`/`PaneProject`. Spawns the panel on demand and despawns it
/// when the overlay is toggled off.
fn update_prof_panel(
    mut commands: Commands,
    time: Res<Time<Real>>,
    state: Res<FpsOverlayState>,
    readout: Res<FrameProfileReadout>,
    stages: Res<StageClock>,
    font: Option<Res<MonoFont>>,
    windows: Query<&Window>,
    titles: Query<&PaneTitle>,
    pane_projects: Query<&PaneProject>,
    projects: Option<Res<Projects>>,
    mut panel: Query<(Entity, &mut Text2d, &mut Transform), With<ProfPanel>>,
    mut throttle: Local<f32>,
) {
    if !state.enabled {
        for (e, ..) in &panel {
            commands.entity(e).despawn();
        }
        *throttle = 0.0;
        return;
    }

    // Per-frame numbers are exact but jittery; refresh the text a few
    // times a second so it's readable while still reconciling each shown
    // snapshot. (The panel entity persists between refreshes.)
    *throttle += time.delta_secs();
    if *throttle < 0.2 && !panel.is_empty() {
        return;
    }
    *throttle = 0.0;

    let Some(data) = readout.data.as_ref() else {
        return;
    };

    let proj_name = |id: u64| -> String {
        projects
            .as_ref()
            .and_then(|p| p.name_of(id))
            .map(|s| s.to_string())
            .unwrap_or_default()
    };

    let mut lines: Vec<String> = Vec::new();

    // Time model: frame = active + remainder.
    //   active   = wall time the app's systems ran (First→Last).
    //   remainder = frame − active = main thread parked (vsync/present/idle).
    // Within active: panes + subsystems (non-nested) + uncategorized.
    // Nested subsystems (taffy) are inside a pane total — shown for detail,
    // not added to the sum.
    let pane_total: Duration = data.panes.iter().map(|p| p.total).sum();
    let sys_top_total: Duration = data
        .subsystems
        .iter()
        .filter(|s| !s.nested)
        .map(|s| s.total)
        .sum();
    let active = readout.active_ms.max(ms(pane_total) + ms(sys_top_total));
    let frame_ms = readout.frame_ms.max(active);
    let remainder = (frame_ms - active).max(0.0);

    lines.push(format!(
        "PROFILER   frame {:.2} ms  ({:.0} fps)",
        frame_ms,
        if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 },
    ));
    // Name the remainder by frame length. A short frame with a big
    // remainder is the present-wait (compositor syncing to refresh); a long
    // frame is the app sleeping (reactive mode), not a render stall.
    let rem_kind = if remainder / frame_ms.max(0.001) < 0.3 {
        "active"
    } else if frame_ms > 25.0 {
        "reactive sleep"
    } else {
        "present wait"
    };
    lines.push(format!(
        "  ACTIVE {:.2}ms  +  remainder {:.2}ms ({})  = {:.2}ms",
        active, remainder, rem_kind, frame_ms,
    ));

    // ---- stages: the PRIMARY breakdown of active. By schedule, so it
    // covers 100% of active including the engine-internal work (transform
    // propagation, visibility, sprite/text prep) our per-system spans can't
    // see. Sorted by cost so the dominant stage (usually PostUpdate) leads.
    let stage_sum: f32 = stages.dur_ms.iter().sum();
    lines.push(format!("-- active by stage  {:.2}ms --", stage_sum));
    let mut stage_rows: Vec<(&str, f32)> = STAGE_NAMES
        .iter()
        .zip(stages.dur_ms.iter())
        .map(|(n, d)| (*n, *d))
        .collect();
    stage_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, d) in stage_rows {
        let bar = "#".repeat(((d / stage_sum.max(0.001)) * 20.0) as usize);
        lines.push(format!("{:>7.2}ms  {:<10} {}", d, name, bar));
    }

    // ---- panes (detail; these spans live within the stages above) ----
    lines.push(format!(
        "-- panes  {:.2}ms / {} --",
        ms(pane_total),
        data.panes.len(),
    ));
    if data.panes.is_empty() {
        lines.push("   (no pane work this frame)".into());
    }
    for p in data.panes.iter().take(MAX_ROWS) {
        let entity = Entity::from_bits(p.entity_bits);
        let title = titles
            .get(entity)
            .map(|t| t.0.clone())
            .unwrap_or_else(|_| format!("#{}", entity.index()));
        let project = pane_projects
            .get(entity)
            .map(|pp| proj_name(pp.0))
            .unwrap_or_default();
        let mut title = title;
        if title.chars().count() > 18 {
            title = title.chars().take(17).collect::<String>() + "…";
        }
        lines.push(format!(
            "{:>7.2}ms {:<18} {:<8} {}",
            ms(p.total),
            title,
            p.kind,
            project,
        ));
    }
    if data.panes.len() > MAX_ROWS {
        lines.push(format!("   +{} more panes", data.panes.len() - MAX_ROWS));
    }

    // ---- subsystems (top-level counted in the budget; nested shown with
    // a leading dot and the note that they're already inside a pane) ----
    lines.push(format!("-- subsystems  {:.2}ms --", ms(sys_top_total)));
    for s in data.subsystems.iter().take(MAX_ROWS) {
        let marker = if s.nested { "·" } else { " " };
        let suffix = if s.nested { "  (in panes)" } else { "" };
        lines.push(format!(
            "{}{:>6.2}ms x{:<4} {}{}",
            marker,
            ms(s.total),
            s.hits,
            s.name,
            suffix,
        ));
    }
    if data.subsystems.len() > MAX_ROWS {
        lines.push(format!(
            "   +{} more subsystems",
            data.subsystems.len() - MAX_ROWS
        ));
    }

    let text = lines.join("\n");

    let Ok(window) = windows.single() else { return };
    // Top-left, a little below where the memory-warning overlay would sit.
    let x = -window.width() * 0.5 + MARGIN;
    let y = window.height() * 0.5 - MARGIN - FONT_SIZE * 1.5;

    // A touch smaller than the body font so a dozen-plus rows fit.
    let size = FONT_SIZE * 0.85;

    if let Ok((_, mut t, mut tx)) = panel.single_mut() {
        t.0 = text;
        tx.translation.x = x;
        tx.translation.y = y;
    } else {
        let Some(font) = font else { return };
        commands.spawn((
            ProfPanel,
            Text2d::new(text),
            TextFont {
                font: font.0.clone(),
                font_size: size,
                ..default()
            },
            TextColor(Color::srgb(0.6, 1.0, 0.7)),
            Anchor::TOP_LEFT,
            Transform::from_xyz(x, y, Z),
            RenderLayers::layer(MENU_OVERLAY_LAYER),
        ));
    }
}

/// How often the headless `TBPROF` dump writes a block to the log. The
/// per-pane numbers are a single-frame sample at each tick; the Bevy
/// diagnostics (frame time, GPU passes) are already smoothed.
const DUMP_INTERVAL_SECS: f64 = 1.0;

/// `TBPROF` headless dump: every second, append the latest frame's
/// per-pane + subsystem breakdown plus Bevy's own diagnostics (frame
/// time, and under `TBPROF=all` the render-world GPU pass timings) to
/// `~/.terminal-bevy/diagnostics.log`. No window required.
fn dump_profile_log(
    time: Res<Time<Real>>,
    readout: Res<FrameProfileReadout>,
    stages: Res<StageClock>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    titles: Query<&PaneTitle>,
    pane_projects: Query<&PaneProject>,
    projects: Option<Res<Projects>>,
    mut accum: Local<f64>,
) {
    *accum += time.delta_secs_f64();
    if *accum < DUMP_INTERVAL_SECS {
        return;
    }
    *accum = 0.0;

    crate::diagnostics::append_log("[tbprof] ---- frame profile ----");
    let stage_line = STAGE_NAMES
        .iter()
        .zip(stages.dur_ms.iter())
        .map(|(n, d)| format!("{}={:.2}ms", n, d))
        .collect::<Vec<_>>()
        .join(" ");
    crate::diagnostics::append_log(&format!("[tbprof] stages {}", stage_line));

    if let Some(data) = readout.data.as_ref() {
        // Same disjoint budget as the overlay: panes + non-nested subsystems.
        let pane_total: f32 = data.panes.iter().map(|p| ms(p.total)).sum();
        let sys_top: f32 = data
            .subsystems
            .iter()
            .filter(|s| !s.nested)
            .map(|s| ms(s.total))
            .sum();
        let attributed = pane_total + sys_top;
        let active = readout.active_ms.max(attributed);
        let frame_ms = readout.frame_ms.max(active);
        crate::diagnostics::append_log(&format!(
            "[tbprof] frame {:.2}ms = active {:.2}ms (panes {:.2} + subsys {:.2} + uncateg {:.2}) + remainder {:.2}ms",
            frame_ms,
            active,
            pane_total,
            sys_top,
            (active - attributed).max(0.0),
            (frame_ms - active).max(0.0),
        ));
        let proj_name = |id: u64| -> String {
            projects
                .as_ref()
                .and_then(|p| p.name_of(id))
                .map(|s| s.to_string())
                .unwrap_or_default()
        };
        for p in &data.panes {
            let entity = Entity::from_bits(p.entity_bits);
            let title = titles
                .get(entity)
                .map(|t| t.0.clone())
                .unwrap_or_else(|_| format!("#{}", entity.index()));
            let project = pane_projects
                .get(entity)
                .map(|pp| proj_name(pp.0))
                .unwrap_or_default();
            crate::diagnostics::append_log(&format!(
                "[tbprof] pane {:>7.2}ms x{} {} [{}] {}",
                ms(p.total),
                p.hits,
                p.kind,
                project,
                title,
            ));
        }
        for s in &data.subsystems {
            crate::diagnostics::append_log(&format!(
                "[tbprof] sys  {:>7.2}ms x{} {}{}",
                ms(s.total),
                s.hits,
                s.name,
                if s.nested { " (in panes)" } else { "" },
            ));
        }
    }

    // Bevy's own diagnostics: frame time + (TBPROF=all) every render-world
    // GPU pass span. Sorted by path so the log is stable frame-to-frame.
    let mut diags: Vec<(String, f64, String)> = diagnostics
        .iter()
        .filter_map(|d| {
            d.smoothed()
                .map(|v| (d.path().to_string(), v, d.suffix.to_string()))
        })
        .collect();
    diags.sort_by(|a, b| a.0.cmp(&b.0));
    for (path, value, suffix) in diags {
        crate::diagnostics::append_log(&format!(
            "[tbprof] diag {:>10.3}{} {}",
            value, suffix, path,
        ));
    }
}

// ---- CPU usage graph ----

/// Number of bars / history samples in the CPU graph.
const CPU_BARS: usize = 100;
/// Seconds between CPU samples pushed into the ring. The underlying
/// sysinfo value refreshes ~once a second, so this is the visual cadence,
/// not the true resolution — ~0.2s × 100 bars ≈ 20s of history on screen.
const CPU_SAMPLE_SECS: f64 = 0.2;
const CPU_BAR_W: f32 = 2.0;
const CPU_BAR_STEP: f32 = 3.0;
const CPU_GRAPH_H: f32 = 56.0;

#[derive(Component)]
struct CpuBar;

#[derive(Component)]
struct CpuGraphLabel;

#[derive(Resource, Default)]
struct CpuGraph {
    /// Recent process-CPU% samples, oldest at front. One core fully busy
    /// reads ~100, so this can exceed 100 on a multi-core spike.
    samples: VecDeque<f32>,
    bars: Vec<Entity>,
    label: Option<Entity>,
    accum: f64,
    latest: f32,
    /// Previous CPU-time reading (ns) + the wall-clock elapsed at that
    /// reading, to difference into a percentage. `None` until first read.
    prev_cpu_ns: Option<u64>,
    prev_wall: f64,
    wall: f64,
}

/// Draws a scrolling bar graph of process CPU% in the top-right, under the
/// fps meter. Bars and the label are spawned on demand and despawned when
/// the overlay is toggled off. Y-scale is dynamic (max of the window,
/// floored at 100%) so a multi-core spike past 100% still fits.
fn update_cpu_graph(
    mut commands: Commands,
    time: Res<Time<Real>>,
    state: Res<FpsOverlayState>,
    windows: Query<&Window>,
    font: Option<Res<MonoFont>>,
    mut graph: ResMut<CpuGraph>,
    mut bars_q: Query<(&mut Sprite, &mut Transform), With<CpuBar>>,
    mut label_q: Query<
        (&mut Text2d, &mut Transform),
        (With<CpuGraphLabel>, Without<CpuBar>),
    >,
) {
    // Keep a running wall clock even while disabled so the first sample
    // after re-enabling differences against a fresh interval, not a stale
    // one from minutes ago (which would read as a huge spike).
    graph.wall += time.delta_secs_f64();

    if !state.enabled {
        for e in graph.bars.drain(..) {
            commands.entity(e).try_despawn();
        }
        if let Some(e) = graph.label.take() {
            commands.entity(e).try_despawn();
        }
        graph.samples.clear();
        graph.prev_cpu_ns = None;
        return;
    }

    // Sample on the visual cadence. CPU% = Δ(process cpu-time) / Δ(wall),
    // computed from `proc_pid_rusage` since Bevy's per-process metric reads
    // a flat 0 here.
    graph.accum += time.delta_secs_f64();
    if graph.accum >= CPU_SAMPLE_SECS || graph.samples.is_empty() {
        graph.accum = 0.0;
        if let Some(cpu_ns) = crate::diagnostics::process_cpu_time_ns() {
            if let Some(prev_ns) = graph.prev_cpu_ns {
                let d_cpu = cpu_ns.saturating_sub(prev_ns) as f64;
                let d_wall = (graph.wall - graph.prev_wall).max(1e-6);
                let pct = (d_cpu / 1.0e9) / d_wall * 100.0;
                graph.latest = pct as f32;
                graph.samples.push_back(pct as f32);
                while graph.samples.len() > CPU_BARS {
                    graph.samples.pop_front();
                }
            }
            graph.prev_cpu_ns = Some(cpu_ns);
            graph.prev_wall = graph.wall;
        }
    }

    let Ok(window) = windows.single() else { return };
    let right = window.width() * 0.5 - MARGIN;
    // Below the fps meter line (top-right).
    let baseline = window.height() * 0.5 - MARGIN - FONT_SIZE * 1.6 - CPU_GRAPH_H;

    // Dynamic vertical scale: tallest sample in the window, never below
    // 100% so a calm graph doesn't amplify noise into full-height bars.
    let peak = graph
        .samples
        .iter()
        .copied()
        .fold(100.0_f32, f32::max)
        .max(1.0);

    // Ensure the bar pool exists.
    if graph.bars.len() != CPU_BARS {
        for e in graph.bars.drain(..) {
            commands.entity(e).try_despawn();
        }
        let mut ids = Vec::with_capacity(CPU_BARS);
        for _ in 0..CPU_BARS {
            let id = commands
                .spawn((
                    CpuBar,
                    Sprite {
                        color: Color::srgb(0.3, 0.9, 0.4),
                        custom_size: Some(Vec2::new(CPU_BAR_W, 1.0)),
                        ..default()
                    },
                    Anchor::BOTTOM_CENTER,
                    Transform::from_xyz(right, baseline, Z - 1.0),
                    RenderLayers::layer(MENU_OVERLAY_LAYER),
                ))
                .id();
            ids.push(id);
        }
        graph.bars = ids;
    }

    // Newest sample on the right. `samples` is oldest-front, so map the
    // last `n` samples to the rightmost `n` bars.
    let n = graph.samples.len();
    for (slot, &bar_entity) in graph.bars.iter().enumerate() {
        let Ok((mut sprite, mut tx)) = bars_q.get_mut(bar_entity) else {
            continue;
        };
        // slot 0 = leftmost bar. Map so the newest sample is rightmost.
        let from_right = CPU_BARS - 1 - slot;
        let x = right - CPU_BAR_W * 0.5 - from_right as f32 * CPU_BAR_STEP;
        if from_right < n {
            let sample = graph.samples[n - 1 - from_right];
            let h = (sample / peak * CPU_GRAPH_H).clamp(1.0, CPU_GRAPH_H);
            sprite.custom_size = Some(Vec2::new(CPU_BAR_W, h));
            sprite.color = cpu_color(sample);
            tx.translation = Vec3::new(x, baseline, Z - 1.0);
        } else {
            // No sample yet for this bar — collapse it.
            sprite.custom_size = Some(Vec2::new(CPU_BAR_W, 0.0));
            tx.translation = Vec3::new(x, baseline, Z - 1.0);
        }
    }

    // Label above the graph: current value + the scale ceiling.
    let label_text = format!("CPU {:>5.0}%   (scale 0-{:.0}%)", graph.latest, peak);
    let label_y = baseline + CPU_GRAPH_H + 2.0;
    if let Ok((mut text, mut tx)) = label_q.single_mut() {
        text.0 = label_text;
        tx.translation = Vec3::new(right, label_y, Z);
    } else if let Some(font) = font {
        let id = commands
            .spawn((
                CpuGraphLabel,
                Text2d::new(label_text),
                TextFont {
                    font: font.0.clone(),
                    font_size: FONT_SIZE * 0.85,
                    ..default()
                },
                TextColor(Color::srgb(0.3, 0.9, 0.4)),
                Anchor::TOP_RIGHT,
                Transform::from_xyz(right, label_y, Z),
                RenderLayers::layer(MENU_OVERLAY_LAYER),
            ))
            .id();
        graph.label = Some(id);
    }
}

/// Green when calm, through yellow, to red as one process saturates more
/// cores. Tuned so ~1 core (100%) is yellow and ~3+ cores is solid red.
fn cpu_color(cpu: f32) -> Color {
    let t = (cpu / 300.0).clamp(0.0, 1.0);
    // green -> red, with a yellow midpoint.
    let r = (t * 2.0).clamp(0.0, 1.0);
    let g = (2.0 - t * 2.0).clamp(0.0, 1.0);
    Color::srgb(0.2 + 0.8 * r, 0.3 + 0.6 * g, 0.3)
}

// ================= interactive frame chart =================

/// One frame's composition, kept in the history ring for the chart.
///   total  = whole frame wall time.
///   active = app systems' wall time (First→Last). panes+systop+uncategorized.
///   remainder = total − active = main thread parked (vsync/present/idle).
/// `gpu` (render-world pass time) and `cpu_pct` (process CPU, all threads)
/// are concurrent measures, not part of the stack — they tell you whether a
/// big remainder is just idle, GPU-bound, or masking a busy worker thread.
#[derive(Clone, Copy, Default)]
struct FrameSample {
    total: f32,
    active: f32,
    /// Per-schedule active time (First, PreUpdate, Update, PostUpdate, Last).
    stages: [f32; 5],
    panes: f32,
    systop: f32,
    gpu: f32,
    cpu_pct: f32,
    top_pane_bits: u64,
    top_pane_ms: f32,
    top_pane_kind: &'static str,
    top_sys: &'static str,
    top_sys_ms: f32,
}

impl FrameSample {
    /// Frame time the main thread was parked (vsync/present/idle).
    fn remainder(&self) -> f32 {
        (self.total - self.active).max(0.0)
    }
}

const HIST_LEN: usize = 120;

#[derive(Resource, Default)]
struct ProfHistory {
    samples: VecDeque<FrameSample>,
    /// Click-to-freeze: when paused, the ring stops advancing so you can
    /// hover a captured spike.
    paused: bool,
    prev_cpu_ns: Option<u64>,
    wall: f64,
    prev_wall: f64,
}

/// Sum render-world GPU pass times (ms) from Bevy's `RenderDiagnostics`.
/// Prefers actual `elapsed_gpu` timestamps; falls back to the CPU-side
/// `elapsed_cpu` of the passes when GPU timestamp queries aren't supported.
fn sum_render_gpu(diagnostics: &bevy::diagnostic::DiagnosticsStore) -> f32 {
    let mut gpu = 0.0f32;
    let mut cpu = 0.0f32;
    for d in diagnostics.iter() {
        let path = d.path().to_string();
        if !path.starts_with("render/") {
            continue;
        }
        let Some(v) = d.smoothed() else { continue };
        if path.ends_with("/elapsed_gpu") {
            gpu += v as f32;
        } else if path.ends_with("/elapsed_cpu") {
            cpu += v as f32;
        }
    }
    if gpu > 0.0 {
        gpu
    } else {
        cpu
    }
}

/// End-of-frame: fold the just-collected frame into the history ring,
/// attributing GPU (render diagnostics) and process CPU% (rusage delta).
fn record_history(
    time: Res<Time<Real>>,
    readout: Res<FrameProfileReadout>,
    stage_clock: Res<StageClock>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    mut hist: ResMut<ProfHistory>,
) {
    // Keep the wall clock advancing regardless, so the CPU% delta window
    // is correct even across paused stretches.
    hist.wall += time.delta_secs_f64();

    if hist.paused {
        return;
    }
    let Some(data) = readout.data.as_ref() else {
        return;
    };

    let panes: f32 = data.panes.iter().map(|p| ms(p.total)).sum();
    let systop: f32 = data
        .subsystems
        .iter()
        .filter(|s| !s.nested)
        .map(|s| ms(s.total))
        .sum();
    let active = readout.active_ms.max(panes + systop);
    let total = readout.frame_ms.max(active);

    // Process CPU% (all threads) over this frame's wall interval.
    let cpu_pct = match crate::diagnostics::process_cpu_time_ns() {
        Some(ns) => {
            let pct = match hist.prev_cpu_ns {
                Some(prev) => {
                    let d_cpu = ns.saturating_sub(prev) as f64 / 1.0e9;
                    let d_wall = (hist.wall - hist.prev_wall).max(1e-6);
                    (d_cpu / d_wall * 100.0) as f32
                }
                None => 0.0,
            };
            hist.prev_cpu_ns = Some(ns);
            hist.prev_wall = hist.wall;
            pct
        }
        None => 0.0,
    };

    let top_pane = data.panes.first();
    let top_sys = data.subsystems.iter().find(|s| !s.nested);

    let sample = FrameSample {
        total,
        active,
        stages: stage_clock.dur_ms,
        panes,
        systop,
        gpu: sum_render_gpu(&diagnostics),
        cpu_pct,
        top_pane_bits: top_pane.map(|p| p.entity_bits).unwrap_or(0),
        top_pane_ms: top_pane.map(|p| ms(p.total)).unwrap_or(0.0),
        top_pane_kind: top_pane.map(|p| p.kind).unwrap_or(""),
        top_sys: top_sys.map(|s| s.name).unwrap_or(""),
        top_sys_ms: top_sys.map(|s| ms(s.total)).unwrap_or(0.0),
    };
    hist.samples.push_back(sample);
    while hist.samples.len() > HIST_LEN {
        hist.samples.pop_front();
    }
}

#[derive(Component)]
struct ChartPart;

/// Number of stacked segments per bar: 5 stages + remainder.
const SEGS_PER_BAR: usize = 6;

/// Per-stage colors (First, PreUpdate, Update, PostUpdate, Last). PostUpdate
/// is orange so the usual dominant stage pops.
fn stage_color(i: usize) -> Color {
    match i {
        0 => Color::srgb(0.4, 0.8, 0.8),  // First — teal
        1 => Color::srgb(0.45, 0.6, 1.0), // PreUpdate — blue
        2 => Color::srgb(0.3, 0.85, 0.45),// Update — green
        3 => Color::srgb(1.0, 0.65, 0.3), // PostUpdate — orange
        _ => Color::srgb(0.75, 0.5, 0.9), // Last — purple
    }
}

/// Sprite/text pool for the chart, rebuilt in place each frame.
#[derive(Resource, Default)]
struct ProfChart {
    /// `HIST_LEN * SEGS_PER_BAR` stacked segment sprites.
    segs: Vec<Entity>,
    background: Option<Entity>,
    reflines: Vec<Entity>,
    hairline: Option<Entity>,
    detail: Option<Entity>,
    title: Option<Entity>,
}

const CHART_STEP: f32 = 5.0;
const CHART_W: f32 = HIST_LEN as f32 * CHART_STEP;
const CHART_H: f32 = 130.0;
const SEG_W: f32 = 4.0;

/// Remainder (idle/vsync): deliberately dim so the eye reads the bright
/// "active" stage stack as the real cost and treats this as background.
fn col_remainder() -> Color {
    Color::srgba(0.35, 0.35, 0.42, 0.55)
}

/// Draws the interactive frame-composition chart (bottom-left) and handles
/// hover + click-to-freeze. Stacked bars = panes + subsystems +
/// uncategorized (= active) + remainder (= idle), summing to the frame.
/// Hovering a bar shows that frame's full breakdown.
#[allow(clippy::too_many_arguments)]
fn update_chart(
    mut commands: Commands,
    state: Res<FpsOverlayState>,
    mut hist: ResMut<ProfHistory>,
    mut chart: ResMut<ProfChart>,
    font: Option<Res<MonoFont>>,
    windows: Query<&Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    titles: Query<&PaneTitle>,
    pane_projects: Query<&PaneProject>,
    projects: Option<Res<Projects>>,
    mut sprites: Query<
        (&mut Sprite, &mut Transform, &mut Visibility),
        (With<ChartPart>, Without<Text2d>),
    >,
    mut texts: Query<
        (&mut Text2d, &mut Transform, &mut Visibility),
        (With<ChartPart>, Without<Sprite>),
    >,
) {
    if !state.enabled {
        despawn_chart(&mut commands, &mut chart);
        return;
    }
    let Some(font) = font else { return };
    let Ok(window) = windows.single() else { return };
    let win_w = window.width();
    let win_h = window.height();

    let left = -win_w * 0.5 + MARGIN;
    let bottom = -win_h * 0.5 + MARGIN + FONT_SIZE * 1.2;
    let right = left + CHART_W;

    ensure_chart(&mut commands, &mut chart, &font);

    // Vertical scale: tallest frame in view, floored at 33.3 ms (so a
    // smooth app doesn't blow tiny frames up to full height), rounded up.
    let peak = hist
        .samples
        .iter()
        .map(|s| s.total)
        .fold(33.3_f32, f32::max)
        .max(1.0);
    let y_of = |ms_val: f32| (ms_val / peak * CHART_H).clamp(0.0, CHART_H);

    // ---- hover + freeze: map cursor to a sample index ----
    let n = hist.samples.len();
    let cursor = window.cursor_position().map(|p| {
        // cursor is top-left origin window space; convert to our centered
        // 1:1 overlay space (y up).
        Vec2::new(p.x - win_w * 0.5, win_h * 0.5 - p.y)
    });
    let over_chart = cursor
        .map(|c| c.x >= left - 4.0 && c.x <= right + 4.0 && c.y >= bottom - 4.0 && c.y <= bottom + CHART_H + 4.0)
        .unwrap_or(false);
    let hovered: Option<usize> = if over_chart && n > 0 {
        let c = cursor.unwrap();
        let from_right = ((right - c.x) / CHART_STEP).round() as i32;
        let idx = n as i32 - 1 - from_right;
        (0..n as i32).contains(&idx).then_some(idx as usize)
    } else {
        None
    };
    if over_chart && mouse.just_pressed(MouseButton::Left) {
        hist.paused = !hist.paused;
    }

    // ---- background panel ----
    if let Some(bg) = chart.background {
        if let Ok((mut sp, mut tx, mut vis)) = sprites.get_mut(bg) {
            sp.color = Color::srgba(0.05, 0.06, 0.09, 0.82);
            sp.custom_size = Some(Vec2::new(CHART_W + 12.0, CHART_H + 14.0));
            tx.translation = Vec3::new(left - 6.0, bottom - 7.0, Z - 3.0);
            *vis = Visibility::Visible;
        }
    }

    // ---- reference lines at 60fps (16.6ms) and 30fps (33.3ms) ----
    for (i, refms) in [16.6_f32, 33.3].into_iter().enumerate() {
        if let Some(&e) = chart.reflines.get(i) {
            if let Ok((mut sp, mut tx, mut vis)) = sprites.get_mut(e) {
                let y = bottom + y_of(refms);
                sp.color = Color::srgba(1.0, 1.0, 1.0, 0.18);
                sp.custom_size = Some(Vec2::new(CHART_W, 1.0));
                tx.translation = Vec3::new(left, y, Z - 2.0);
                *vis = if refms <= peak { Visibility::Visible } else { Visibility::Hidden };
            }
        }
    }

    // ---- stacked bars ----
    for slot in 0..HIST_LEN {
        let from_right = HIST_LEN - 1 - slot;
        let x = right - SEG_W * 0.5 - from_right as f32 * CHART_STEP;
        let sample = if from_right < n {
            Some(hist.samples[n - 1 - from_right])
        } else {
            None
        };
        let is_hover = hovered.map(|h| h == n.wrapping_sub(1 + from_right)).unwrap_or(false);
        let segs = [
            (sample.map(|s| s.stages[0]).unwrap_or(0.0), stage_color(0)),
            (sample.map(|s| s.stages[1]).unwrap_or(0.0), stage_color(1)),
            (sample.map(|s| s.stages[2]).unwrap_or(0.0), stage_color(2)),
            (sample.map(|s| s.stages[3]).unwrap_or(0.0), stage_color(3)),
            (sample.map(|s| s.stages[4]).unwrap_or(0.0), stage_color(4)),
            (sample.map(|s| s.remainder()).unwrap_or(0.0), col_remainder()),
        ];
        let mut acc = 0.0f32;
        for (si, (val, color)) in segs.into_iter().enumerate() {
            let Some(&e) = chart.segs.get(slot * SEGS_PER_BAR + si) else { continue };
            let Ok((mut sp, mut tx, mut vis)) = sprites.get_mut(e) else { continue };
            if sample.is_none() || val <= 0.0 {
                *vis = Visibility::Hidden;
                acc += val.max(0.0);
                continue;
            }
            let h = y_of(val);
            let y0 = bottom + y_of(acc);
            let c = if is_hover { brighten(color) } else { color };
            sp.color = c;
            sp.custom_size = Some(Vec2::new(SEG_W, h.max(0.5)));
            tx.translation = Vec3::new(x, y0, Z - 1.0);
            *vis = Visibility::Visible;
            acc += val;
        }
    }

    // ---- hairline at the hovered bar ----
    if let Some(hl) = chart.hairline {
        if let Ok((mut sp, mut tx, mut vis)) = sprites.get_mut(hl) {
            if let Some(h) = hovered {
                let from_right = n - 1 - h;
                let x = right - from_right as f32 * CHART_STEP;
                sp.color = Color::srgba(1.0, 0.9, 0.3, 0.7);
                sp.custom_size = Some(Vec2::new(1.0, CHART_H));
                tx.translation = Vec3::new(x, bottom, Z - 0.5);
                *vis = Visibility::Visible;
            } else {
                *vis = Visibility::Hidden;
            }
        }
    }

    // ---- title line (just below the chart) ----
    let latest = hist.samples.back().copied().unwrap_or_default();
    if let Some(t) = chart.title {
        if let Ok((mut text, mut tx, mut vis)) = texts.get_mut(t) {
            text.0 = format!(
                "FRAME CHART 0-{:.0}ms {} [click=freeze] stages: teal=First blu=Pre grn=Update org=Post pur=Last dim=remainder",
                peak,
                if hist.paused { "PAUSED" } else { "live" },
            );
            tx.translation = Vec3::new(left, bottom - 3.0, Z);
            *vis = Visibility::Visible;
        }
    }

    // ---- detail box (hovered sample, or the latest), above the chart ----
    let shown = hovered.map(|h| hist.samples[h]).unwrap_or(latest);
    let detail = build_detail(&shown, &titles, &pane_projects, projects.as_deref());
    if let Some(d) = chart.detail {
        if let Ok((mut text, mut tx, mut vis)) = texts.get_mut(d) {
            text.0 = detail;
            tx.translation = Vec3::new(left, bottom + CHART_H + 8.0, Z);
            *vis = Visibility::Visible;
        }
    }
}

fn brighten(c: Color) -> Color {
    let s = c.to_srgba();
    Color::srgb(
        (s.red + 0.2).min(1.0),
        (s.green + 0.2).min(1.0),
        (s.blue + 0.2).min(1.0),
    )
}

/// Human-readable breakdown of one frame: the active/remainder split, the
/// active sub-categories, and a one-line note on what the remainder is.
fn build_detail(
    s: &FrameSample,
    titles: &Query<&PaneTitle>,
    pane_projects: &Query<&PaneProject>,
    projects: Option<&Projects>,
) -> String {
    let rem = s.remainder();
    let pct = |v: f32| if s.total > 0.0 { v / s.total * 100.0 } else { 0.0 };

    let top_pane_label = if s.top_pane_bits != 0 {
        let e = Entity::from_bits(s.top_pane_bits);
        let title = titles
            .get(e)
            .map(|t| t.0.clone())
            .unwrap_or_else(|_| format!("#{}", e.index()));
        let project = pane_projects
            .get(e)
            .ok()
            .and_then(|p| projects.and_then(|pr| pr.name_of(p.0)))
            .unwrap_or("");
        format!("{} ({} {})", title, s.top_pane_kind, project)
    } else {
        "—".into()
    };

    // What is the remainder? cpu_pct counts ALL threads, so high cpu with a
    // big remainder ⇒ a worker thread is busy while the main thread parks.
    // High render-pass ms ⇒ GPU. A long frame ⇒ the app slept (reactive
    // mode). Otherwise it's the present-wait: the compositor syncing the
    // window to display refresh (capped ~60/120 — a windowed-app limit, see
    // bevy#12097, not fixable without exclusive fullscreen).
    let rem_note = if rem / s.total.max(0.001) < 0.3 {
        "mostly active this frame"
    } else if s.cpu_pct > 90.0 {
        "busy worker thread (cpu high, main parked off the frame loop)"
    } else if s.gpu > s.total * 0.4 {
        "GPU-bound (render passes dominate)"
    } else if s.total > 25.0 {
        "reactive sleep — app idle (wakes on input/anim)"
    } else {
        "present wait — compositor sync to refresh (windowed cap)"
    };

    // Stage breakdown (covers 100% of active), then the per-pane/subsystem
    // detail of whichever stage they live in.
    let stage_line = STAGE_NAMES
        .iter()
        .zip(s.stages.iter())
        .map(|(n, d)| format!("{} {:.2}", n, d))
        .collect::<Vec<_>>()
        .join("  ");

    format!(
        "frame {:.2}ms   cpu {:.0}%   gpu(render) {:.2}ms\n\
         ACTIVE {:.2}ms ({:.0}%)   REMAINDER {:.2}ms ({:.0}%) — {}\n\
         stages: {}\n\
         detail: panes {:.2}ms (top {}) · subsys {:.2}ms (top {} {:.2}ms)",
        s.total,
        s.cpu_pct,
        s.gpu,
        s.active,
        pct(s.active),
        rem,
        pct(rem),
        rem_note,
        stage_line,
        s.panes,
        top_pane_label,
        s.systop,
        if s.top_sys.is_empty() { "—" } else { s.top_sys },
        s.top_sys_ms,
    )
}

fn despawn_chart(commands: &mut Commands, chart: &mut ProfChart) {
    for e in chart.segs.drain(..) {
        commands.entity(e).try_despawn();
    }
    for e in chart.reflines.drain(..) {
        commands.entity(e).try_despawn();
    }
    for e in [chart.background, chart.hairline, chart.detail, chart.title]
        .into_iter()
        .flatten()
    {
        commands.entity(e).try_despawn();
    }
    chart.background = None;
    chart.hairline = None;
    chart.detail = None;
    chart.title = None;
}

fn ensure_chart(commands: &mut Commands, chart: &mut ProfChart, font: &MonoFont) {
    let spawn_sprite = |commands: &mut Commands, anchor: Anchor| {
        commands
            .spawn((
                ChartPart,
                Sprite {
                    color: Color::NONE,
                    custom_size: Some(Vec2::splat(1.0)),
                    ..default()
                },
                anchor,
                Transform::from_xyz(0.0, 0.0, Z - 1.0),
                Visibility::Hidden,
                RenderLayers::layer(MENU_OVERLAY_LAYER),
            ))
            .id()
    };

    if chart.background.is_none() {
        chart.background = Some(spawn_sprite(commands, Anchor::BOTTOM_LEFT));
    }
    if chart.reflines.len() != 2 {
        chart.reflines = (0..2).map(|_| spawn_sprite(commands, Anchor::BOTTOM_LEFT)).collect();
    }
    if chart.segs.len() != HIST_LEN * SEGS_PER_BAR {
        chart.segs = (0..HIST_LEN * SEGS_PER_BAR)
            .map(|_| spawn_sprite(commands, Anchor::BOTTOM_CENTER))
            .collect();
    }
    if chart.hairline.is_none() {
        chart.hairline = Some(spawn_sprite(commands, Anchor::BOTTOM_CENTER));
    }
    if chart.title.is_none() {
        chart.title = Some(
            commands
                .spawn((
                    ChartPart,
                    Text2d::new(""),
                    TextFont {
                        font: font.0.clone(),
                        font_size: FONT_SIZE * 0.8,
                        ..default()
                    },
                    TextColor(Color::srgb(0.7, 0.75, 0.85)),
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(-1000.0, -1000.0, Z),
                    Visibility::Hidden,
                    RenderLayers::layer(MENU_OVERLAY_LAYER),
                ))
                .id(),
        );
    }
    if chart.detail.is_none() {
        chart.detail = Some(
            commands
                .spawn((
                    ChartPart,
                    Text2d::new(""),
                    TextFont {
                        font: font.0.clone(),
                        font_size: FONT_SIZE * 0.85,
                        ..default()
                    },
                    TextColor(Color::srgb(0.85, 0.9, 0.8)),
                    Anchor::BOTTOM_LEFT,
                    Transform::from_xyz(-1000.0, -1000.0, Z),
                    Visibility::Hidden,
                    RenderLayers::layer(MENU_OVERLAY_LAYER),
                ))
                .id(),
        );
    }
}
