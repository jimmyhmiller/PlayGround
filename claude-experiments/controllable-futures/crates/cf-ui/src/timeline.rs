//! Worker-lane timeline. Reconstructs span intervals from PollStart/PollEnd
//! pairs in the event log and renders them as colored blocks per worker.
//! Wakes / spawns / completes appear as small glyphs overlaid on the lane
//! of the *target* task's last-known worker.
//!
//! The view supports panning, zooming, and a scrubbable cursor. When the
//! cursor is set, the side panels can filter to "events near cursor" so the
//! operator can answer "what was happening at this exact moment?"

use cf_runtime::scheduler::TaskMetaSnapshot;
use cf_runtime::task::TaskId;
use cf_runtime::{Event, EventKind};
use eframe::egui;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Persistent state for the timeline panel — the parts that survive between
/// frames (zoom, pan, cursor). The actual span computation happens fresh
/// each frame from the event log snapshot.
pub struct TimelineState {
    /// Width of the visible time window in nanoseconds. Smaller = more
    /// zoom. Auto-tuned on first frame.
    pub window_nanos: u64,
    /// Right edge of the visible window, expressed as nanoseconds before
    /// `now`. 0 = follow latest. Positive = paused at a moment in the past.
    pub view_lag_nanos: u64,
    /// If set, the user has placed a scrub cursor at this absolute Instant.
    /// The right panel / event log filter to events near it.
    pub cursor: Option<Instant>,
    /// If true, the timeline auto-scrolls to follow the latest event.
    pub follow: bool,
    /// Reference instant for converting Instant → nanoseconds-since-start.
    /// Captured the first frame the panel renders.
    pub origin: Option<Instant>,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            // 200ms default window. Tokio task polls are microseconds; a
            // 2s window made stepped polls render as <1px slivers.
            window_nanos: 200_000_000,
            view_lag_nanos: 0,
            cursor: None,
            follow: true,
            origin: None,
        }
    }
}

/// One reconstructed span: a task ran on a worker over [start, end).
struct Span {
    task: TaskId,
    worker: usize,
    start: Instant,
    end: Instant,
}

/// One marker event (wake / spawn / complete) we want to overlay on a lane.
struct Marker {
    task: TaskId,
    /// Lane on which to draw — the worker most recently associated with the
    /// task. None = render in a "global" row at the top.
    worker: Option<usize>,
    at: Instant,
    kind: MarkerKind,
    label: String,
}

#[derive(Copy, Clone)]
enum MarkerKind {
    Wake,
    Spawn,
    Completed,
    User,
}

/// Build spans + markers from the events. We walk in seq order maintaining
/// a per-task "currently running on worker X since T" map. Polls that don't
/// have a matching PollEnd (i.e. still in progress) produce a span ending
/// at `now`.
fn build_spans(events: &[Event], now: Instant) -> (Vec<Span>, Vec<Marker>) {
    let mut active: HashMap<TaskId, (usize, Instant)> = HashMap::new();
    let mut last_worker: HashMap<TaskId, usize> = HashMap::new();
    let mut spans = Vec::new();
    let mut markers = Vec::new();
    for e in events {
        match &e.kind {
            EventKind::PollStart => {
                if let (Some(t), Some(w)) = (e.task, e.worker) {
                    active.insert(t, (w, e.at));
                    last_worker.insert(t, w);
                }
            }
            EventKind::PollEnd { .. } => {
                if let Some(t) = e.task {
                    if let Some((w, start)) = active.remove(&t) {
                        spans.push(Span {
                            task: t,
                            worker: w,
                            start,
                            end: e.at,
                        });
                    }
                }
            }
            EventKind::Wake { from_task, .. } => {
                if let Some(t) = e.task {
                    let label = match from_task {
                        Some(src) => format!("wake (from #{})", src.0),
                        None => "wake".to_string(),
                    };
                    markers.push(Marker {
                        task: t,
                        worker: last_worker.get(&t).copied(),
                        at: e.at,
                        kind: MarkerKind::Wake,
                        label,
                    });
                }
            }
            EventKind::Spawned { name, .. } => {
                if let Some(t) = e.task {
                    markers.push(Marker {
                        task: t,
                        worker: None,
                        at: e.at,
                        kind: MarkerKind::Spawn,
                        label: format!("spawn {}", name),
                    });
                }
            }
            EventKind::Completed => {
                if let Some(t) = e.task {
                    markers.push(Marker {
                        task: t,
                        worker: last_worker.get(&t).copied(),
                        at: e.at,
                        kind: MarkerKind::Completed,
                        label: "done".into(),
                    });
                }
            }
            EventKind::User { category, detail } => {
                if let Some(t) = e.task {
                    markers.push(Marker {
                        task: t,
                        worker: last_worker.get(&t).copied(),
                        at: e.at,
                        kind: MarkerKind::User,
                        label: if detail.is_empty() {
                            (*category).to_string()
                        } else {
                            format!("{category}: {detail}")
                        },
                    });
                }
            }
            _ => {}
        }
    }
    // Anything still in `active` is currently executing — close the span at
    // `now` so the lane shows a live block.
    for (t, (w, start)) in active {
        spans.push(Span {
            task: t,
            worker: w,
            start,
            end: now,
        });
    }
    (spans, markers)
}

/// Stable color per task id. Hash-based; not perceptually balanced but good
/// enough to distinguish neighbours.
fn task_color(id: TaskId) -> egui::Color32 {
    let h = (id.0.wrapping_mul(2654435761)) as u32;
    let r = ((h >> 16) & 0xFF) as u8;
    let g = ((h >> 8) & 0xFF) as u8;
    let b = (h & 0xFF) as u8;
    // Bias up the channels so colors aren't too dark on dark theme.
    let lift = |c: u8| (c.saturating_add(80)).min(220);
    egui::Color32::from_rgb(lift(r), lift(g), lift(b))
}

/// Format a duration as a short human-readable string for axis labels.
fn fmt_dur_short(d: Duration) -> String {
    let ns = d.as_nanos() as u64;
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}µs", ns as f64 / 1000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.1}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

pub struct TimelineDraw<'a> {
    pub state: &'a mut TimelineState,
    pub events: &'a [Event],
    pub tasks: &'a [TaskMetaSnapshot],
    pub n_workers: usize,
    pub selected_task: &'a mut Option<TaskId>,
}

impl<'a> TimelineDraw<'a> {
    pub fn show(self, ui: &mut egui::Ui) {
        let TimelineDraw {
            state,
            events,
            tasks,
            n_workers,
            selected_task,
        } = self;

        // Always anchor "now" to the latest event time when there are
        // events. Wall-clock time advancing past static historical data
        // is meaningless for visualization — events would just scroll
        // off-screen at real-time. This keeps the view stable whether
        // the workload is running, paused, or finished.
        let now = events
            .last()
            .map(|e| e.at)
            .unwrap_or_else(Instant::now);
        if state.origin.is_none() {
            state.origin = Some(now);
        }

        // Auto-frame: only fires once when a build's last event is
        // already older than the current window. Sets follow=false so
        // it doesn't keep re-firing and "flash" the view across frames.
        // The user can hit `→ live` to re-engage follow mode.
        if state.follow {
            if let (Some(first), Some(last)) = (events.first(), events.last()) {
                let lag_now = (now - last.at).as_nanos() as u64;
                if lag_now > state.window_nanos {
                    let span = (last.at - first.at).as_nanos() as u64;
                    // Add 5% margin on each side so spans aren't flush
                    // against the edges.
                    state.window_nanos = (span + span / 10).max(50_000_000);
                    state.view_lag_nanos = lag_now.saturating_sub(span / 20);
                    state.follow = false;
                }
            }
        }

        // Controls row.
        ui.horizontal(|ui| {
            ui.heading("timeline");
            ui.separator();
            ui.checkbox(&mut state.follow, "follow");
            ui.separator();
            ui.label("zoom:");
            if ui.small_button("−").clicked() {
                state.window_nanos = state.window_nanos.saturating_mul(2).min(600_000_000_000);
            }
            ui.label(fmt_dur_short(Duration::from_nanos(state.window_nanos)));
            if ui.small_button("+").clicked() {
                state.window_nanos = (state.window_nanos / 2).max(1_000_000);
            }
            // Fit-to-events: span the visible window over the entire
            // range of events. Identical math to the auto-frame logic
            // so behavior is consistent.
            if ui.small_button("fit").clicked() {
                if let (Some(first), Some(last)) =
                    (events.first(), events.last())
                {
                    let span = (last.at - first.at).as_nanos() as u64;
                    state.window_nanos = (span + span / 10).max(50_000_000);
                    let lag = (now - last.at).as_nanos() as u64;
                    state.view_lag_nanos = lag.saturating_sub(span / 20);
                    state.follow = false;
                }
            }
            ui.separator();
            if state.cursor.is_some() {
                if ui.small_button("clear cursor").clicked() {
                    state.cursor = None;
                }
            }
            if !state.follow {
                if ui.small_button("→ live").clicked() {
                    state.follow = true;
                    state.view_lag_nanos = 0;
                }
            }
        });

        // Determine the visible time window.
        let view_end = if state.follow {
            now
        } else {
            now - Duration::from_nanos(state.view_lag_nanos)
        };
        let view_start = view_end - Duration::from_nanos(state.window_nanos);

        // Allocate drawing area. Reserve enough height for: 1 axis + 1 global
        // marker row + N worker rows.
        let row_h = 22.0;
        let axis_h = 18.0;
        let global_h = 14.0;
        let total_h = axis_h + global_h + (n_workers as f32) * row_h + 10.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), total_h),
            egui::Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);

        // Background.
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 20, 24));

        let total_span = (view_end - view_start).as_nanos() as f64;
        let x_for = |t: Instant| -> f32 {
            let ns = if t < view_start {
                0.0
            } else {
                (t - view_start).as_nanos() as f64
            };
            (rect.min.x + (ns / total_span * rect.width() as f64) as f32)
                .clamp(rect.min.x, rect.max.x)
        };
        let visible = |t: Instant| t >= view_start && t <= view_end;

        // Axis: draw a few tick marks.
        let axis_y = rect.min.y + axis_h * 0.6;
        painter.line_segment(
            [
                egui::pos2(rect.min.x, axis_y),
                egui::pos2(rect.max.x, axis_y),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        );
        for i in 0..=8 {
            let frac = i as f32 / 8.0;
            let x = rect.min.x + frac * rect.width();
            painter.line_segment(
                [
                    egui::pos2(x, axis_y - 3.0),
                    egui::pos2(x, axis_y + 3.0),
                ],
                egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
            );
            // Time label: relative to view_end (negative offset).
            let ns_back = ((1.0 - frac) as f64 * total_span) as u64;
            let label = if ns_back == 0 {
                "now".to_string()
            } else {
                format!("-{}", fmt_dur_short(Duration::from_nanos(ns_back)))
            };
            painter.text(
                egui::pos2(x + 2.0, rect.min.y),
                egui::Align2::LEFT_TOP,
                label,
                egui::FontId::monospace(10.0),
                egui::Color32::from_gray(140),
            );
        }

        // Global row (spawns).
        let global_y = axis_y + 6.0;
        painter.text(
            egui::pos2(rect.min.x + 4.0, global_y),
            egui::Align2::LEFT_CENTER,
            "global",
            egui::FontId::monospace(10.0),
            egui::Color32::from_gray(120),
        );

        // Worker rows.
        let worker_top = global_y + global_h * 0.5;
        let worker_y = |w: usize| worker_top + w as f32 * row_h;
        for w in 0..n_workers {
            let y0 = worker_y(w);
            let row_rect = egui::Rect::from_min_size(
                egui::pos2(rect.min.x, y0),
                egui::vec2(rect.width(), row_h),
            );
            // Alternating row backgrounds for legibility.
            if w % 2 == 0 {
                painter.rect_filled(row_rect, 0.0, egui::Color32::from_rgb(28, 28, 32));
            }
            painter.text(
                egui::pos2(rect.min.x + 4.0, y0 + row_h * 0.5),
                egui::Align2::LEFT_CENTER,
                format!("w{}", w),
                egui::FontId::monospace(11.0),
                egui::Color32::from_gray(140),
            );
        }

        // Build spans and markers.
        let (spans, markers) = build_spans(events, now);

        // Render spans. Enforce a minimum visual width so microsecond
        // polls (typical for tokio task wakeup) don't render as 0-pixel
        // ghost rectangles in a multi-second view window.
        const MIN_SPAN_PX: f32 = 3.0;
        for s in &spans {
            if s.end < view_start || s.start > view_end {
                continue;
            }
            if s.worker >= n_workers {
                continue;
            }
            let x0 = x_for(s.start.max(view_start));
            let x1 = x_for(s.end.min(view_end)).max(x0 + MIN_SPAN_PX);
            let y0 = worker_y(s.worker) + 3.0;
            let h = row_h - 6.0;
            let mut color = task_color(s.task);
            // Highlight selected task.
            if Some(s.task) == *selected_task {
                color = egui::Color32::from_rgb(255, 240, 120);
            }
            let span_rect =
                egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(x1 - x0, h));
            painter.rect_filled(span_rect, 2.0, color);
            // Inline label if wide enough.
            if x1 - x0 > 30.0 {
                let name = tasks
                    .iter()
                    .find(|t| t.id == s.task)
                    .map(|t| t.name.as_str())
                    .unwrap_or("?");
                let txt = format!("#{} {}", s.task.0, name);
                painter.text(
                    egui::pos2(x0 + 3.0, y0 + h * 0.5),
                    egui::Align2::LEFT_CENTER,
                    txt,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(20, 20, 24),
                );
            }

            // Click on a span selects the task.
            if let Some(pos) = resp.interact_pointer_pos() {
                if resp.clicked() && span_rect.contains(pos) {
                    *selected_task = Some(s.task);
                }
            }
        }

        // Render markers.
        for m in &markers {
            if !visible(m.at) {
                continue;
            }
            let x = x_for(m.at);
            let y = match m.worker {
                Some(w) if w < n_workers => worker_y(w) + row_h * 0.5,
                _ => global_y,
            };
            let (color, glyph) = match m.kind {
                MarkerKind::Wake => (egui::Color32::from_rgb(120, 200, 255), "↑"),
                MarkerKind::Spawn => (egui::Color32::from_rgb(220, 220, 100), "+"),
                MarkerKind::Completed => (egui::Color32::from_rgb(180, 180, 180), "✓"),
                MarkerKind::User => (egui::Color32::from_rgb(255, 140, 200), "•"),
            };
            painter.text(
                egui::pos2(x, y),
                egui::Align2::CENTER_CENTER,
                glyph,
                egui::FontId::monospace(11.0),
                color,
            );
        }

        // Cursor handling: dragging or single-clicking the area sets it.
        // (Span clicks are handled above; we only place a cursor on
        // background clicks where no span is hit. To keep this simple, we
        // always update cursor on any drag.)
        if resp.dragged() || (resp.clicked() && selected_task.is_none()) {
            if let Some(pos) = resp.interact_pointer_pos() {
                if rect.contains(pos) {
                    let frac = ((pos.x - rect.min.x) / rect.width())
                        .clamp(0.0, 1.0) as f64;
                    let ns_into = (frac * total_span) as u64;
                    let cursor_t = view_start + Duration::from_nanos(ns_into);
                    state.cursor = Some(cursor_t);
                    state.follow = false;
                }
            }
        }

        // Draw cursor line.
        if let Some(c) = state.cursor {
            if visible(c) {
                let x = x_for(c);
                painter.line_segment(
                    [
                        egui::pos2(x, rect.min.y + axis_h * 0.5),
                        egui::pos2(x, rect.max.y),
                    ],
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 80, 80)),
                );
                let label = format!("-{}", fmt_dur_short(now - c));
                painter.text(
                    egui::pos2(x + 4.0, rect.min.y + axis_h * 0.5),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(255, 80, 80),
                );
            }
        }

        // Hover tooltip: which span/marker is the mouse over?
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            if rect.contains(pos) {
                // Check spans first.
                let mut tip = None;
                for s in &spans {
                    if s.worker >= n_workers {
                        continue;
                    }
                    let x0 = x_for(s.start.max(view_start));
                    let x1 = x_for(s.end.min(view_end)).max(x0 + 1.0);
                    let y0 = worker_y(s.worker) + 3.0;
                    let h = row_h - 6.0;
                    let r =
                        egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(x1 - x0, h));
                    if r.contains(pos) {
                        let task_meta = tasks.iter().find(|t| t.id == s.task);
                        let name = task_meta.map(|t| t.name.as_str()).unwrap_or("?");
                        let dur = s.end - s.start;
                        let mut t = format!(
                            "#{} {}\nworker {}\nrun for {}",
                            s.task.0,
                            name,
                            s.worker,
                            fmt_dur_short(dur)
                        );
                        if let Some(m) = task_meta {
                            if let Some(reason) = &m.wait_reason {
                                t.push_str(&format!("\nwaiting on: {reason}"));
                            }
                            if m.future_size_bytes > 0 {
                                t.push_str(&format!(
                                    "\nfuture size: {} bytes",
                                    m.future_size_bytes
                                ));
                            }
                        }
                        tip = Some(t);
                        break;
                    }
                }
                if let Some(t) = tip {
                    egui::show_tooltip(ui.ctx(), ui.layer_id(), egui::Id::new("tl-tip"), |ui| {
                        ui.label(egui::RichText::new(t).monospace());
                    });
                }
            }
        }
    }
}
