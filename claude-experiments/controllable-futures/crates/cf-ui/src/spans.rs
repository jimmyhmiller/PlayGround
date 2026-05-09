//! Span hierarchy view.
//!
//! Reconstructs the tree of `tracing` spans from the event log and
//! renders it as a flame-graph-style horizontal-bars-on-vertical-rows
//! layout. Each row is a span at a given depth; bars span the
//! enter→exit time range; child spans render on rows below their parent
//! and inside their parent's horizontal extent.
//!
//! This is the layer that turns "5000 anonymous tokio polls" into
//! "Build > Resolve > Parse > Transform > Emit". You can read what the
//! application is doing.

use cf_runtime::task::TaskId;
use cf_runtime::{Event, EventKind};
use eframe::egui;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Reconstructed span. Built from SpanEnter/SpanExit pairs in the log.
struct Span {
    span_id: u64,
    name: String,
    target: String,
    fields: String,
    parent_id: Option<u64>,
    depth: usize,
    start: Instant,
    end: Instant,
    /// First task that hosted this span. If a span moves between
    /// tokio tasks (rare for tracing because spans are usually
    /// thread-local), only the first attribution is kept.
    task: Option<TaskId>,
}

#[derive(Default)]
pub struct SpansState {
    pub selected_span: Option<u64>,
    /// Visible-window state for the flame chart's horizontal axis.
    /// Mirrors the timeline's panning logic; users can click+drag to
    /// pan and use zoom buttons.
    pub view_window_nanos: u64,
    pub view_lag_nanos: u64,
    /// When true, the critical path is highlighted on the chart and
    /// summarized below.
    pub show_critical_path: bool,
}

pub struct SpansView<'a> {
    pub events: &'a [Event],
    pub state: &'a mut SpansState,
}

impl<'a> SpansView<'a> {
    pub fn show(self, ui: &mut egui::Ui) {
        let SpansView { events, state } = self;
        let spans = build_spans(events);
        if spans.is_empty() {
            ui.label("(no spans yet)");
            return;
        }
        let max_depth = spans.iter().map(|s| s.depth).max().unwrap_or(0);
        let span_count = spans.len();

        let now = events.last().map(|e| e.at).unwrap_or_else(Instant::now);

        // Initialize view to span the whole event range on first frame.
        if state.view_window_nanos == 0 {
            if let (Some(first_evt), Some(last_evt)) = (events.first(), events.last()) {
                let span = (last_evt.at - first_evt.at).as_nanos() as u64;
                state.view_window_nanos = (span + span / 10).max(50_000_000);
                state.view_lag_nanos = 0;
            }
        }

        // Compute critical path (cheap; tens of thousands of spans
        // resolve in single-digit ms). Done unconditionally so we can
        // show the summary when the toggle is on.
        let crit_path = crate::critical::compute(events);
        let crit_set: std::collections::HashSet<u64> = if state.show_critical_path
        {
            crit_path.spans.iter().map(|s| s.span_id).collect()
        } else {
            std::collections::HashSet::new()
        };

        // Controls.
        ui.horizontal(|ui| {
            ui.heading("spans");
            ui.separator();
            ui.label(format!("{span_count} reconstructed, depth {max_depth}"));
            ui.separator();
            ui.checkbox(&mut state.show_critical_path, "critical path");
            if !crit_path.spans.is_empty() {
                ui.label(
                    egui::RichText::new(format!(
                        "(longest chain: {}, total {})",
                        crit_path.spans.len(),
                        fmt_dur_short(Duration::from_nanos(crit_path.total_ns))
                    ))
                    .color(egui::Color32::from_gray(160)),
                );
            }
            ui.separator();
            ui.label("zoom:");
            if ui.small_button("−").clicked() {
                state.view_window_nanos = state
                    .view_window_nanos
                    .saturating_mul(2)
                    .min(600_000_000_000);
            }
            ui.label(fmt_dur_short(Duration::from_nanos(state.view_window_nanos)));
            if ui.small_button("+").clicked() {
                state.view_window_nanos = (state.view_window_nanos / 2).max(1_000_000);
            }
            if ui.small_button("fit").clicked() {
                if let (Some(first_evt), Some(last_evt)) =
                    (events.first(), events.last())
                {
                    let span = (last_evt.at - first_evt.at).as_nanos() as u64;
                    state.view_window_nanos = (span + span / 10).max(50_000_000);
                    state.view_lag_nanos = 0;
                }
            }
        });

        let view_end = now - Duration::from_nanos(state.view_lag_nanos);
        let view_start =
            view_end - Duration::from_nanos(state.view_window_nanos);

        let row_h = 18.0;
        let total_h = (max_depth + 2) as f32 * row_h + 30.0;
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let avail_w = ui.available_width().max(800.0);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(avail_w, total_h),
                    egui::Sense::click_and_drag(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 20, 24));

                let total_span_ns =
                    (view_end - view_start).as_nanos() as f64;
                let x_for = |t: Instant| -> f32 {
                    let ns = if t < view_start {
                        0.0
                    } else {
                        (t - view_start).as_nanos() as f64
                    };
                    rect.min.x
                        + (ns / total_span_ns * rect.width() as f64) as f32
                };

                // Time axis.
                let axis_y = rect.min.y + 12.0;
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
                    let ns_back = ((1.0 - frac) as f64 * total_span_ns) as u64;
                    let label = if ns_back == 0 {
                        "end".to_string()
                    } else {
                        format!("-{}", fmt_dur_short(Duration::from_nanos(ns_back)))
                    };
                    painter.text(
                        egui::pos2(x, rect.min.y + 2.0),
                        egui::Align2::LEFT_TOP,
                        label,
                        egui::FontId::monospace(10.0),
                        egui::Color32::from_gray(140),
                    );
                }

                // Render spans as bars.
                for (idx, s) in spans.iter().enumerate() {
                    if s.end < view_start || s.start > view_end {
                        continue;
                    }
                    let x0 = x_for(s.start.max(view_start));
                    let x1 = x_for(s.end.min(view_end)).max(x0 + 2.0);
                    let y0 = axis_y + 8.0 + s.depth as f32 * row_h;
                    let bar_h = row_h - 2.0;
                    let bar_rect = egui::Rect::from_min_size(
                        egui::pos2(x0, y0),
                        egui::vec2(x1 - x0, bar_h),
                    );
                    let on_critical = crit_set.contains(&s.span_id);
                    let mut color = span_color(&s.name);
                    if state.show_critical_path && !on_critical {
                        // Dim non-critical spans so the path stands out.
                        let [r, g, b, _] = color.to_array();
                        color = egui::Color32::from_rgba_unmultiplied(
                            r / 3,
                            g / 3,
                            b / 3,
                            255,
                        );
                    }
                    if Some(s.span_id) == state.selected_span {
                        color = egui::Color32::from_rgb(255, 240, 120);
                    }
                    painter.rect_filled(bar_rect, 2.0, color);
                    if on_critical && state.show_critical_path {
                        // Red outline on the path itself.
                        painter.rect_stroke(
                            bar_rect,
                            2.0,
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 100, 100)),
                        );
                    }
                    if x1 - x0 > 30.0 {
                        let label = if s.fields.is_empty() {
                            s.name.clone()
                        } else {
                            format!("{} {{ {} }}", s.name, s.fields)
                        };
                        painter.text(
                            egui::pos2(x0 + 3.0, y0 + bar_h * 0.5),
                            egui::Align2::LEFT_CENTER,
                            label,
                            egui::FontId::monospace(10.0),
                            egui::Color32::from_rgb(20, 20, 24),
                        );
                    }
                    if let Some(pos) = resp.interact_pointer_pos() {
                        if resp.clicked() && bar_rect.contains(pos) {
                            state.selected_span = Some(s.span_id);
                        }
                    }
                    let _ = idx;
                }

                // Drag-to-pan.
                if resp.dragged() {
                    let drag = resp.drag_delta().x;
                    let ns_per_px = total_span_ns / rect.width() as f64;
                    let lag_change = (drag as f64 * ns_per_px) as i64;
                    state.view_lag_nanos =
                        (state.view_lag_nanos as i64 + lag_change).max(0) as u64;
                }
            });

        // Detail for selected span.
        if let Some(sid) = state.selected_span {
            if let Some(s) = spans.iter().find(|s| s.span_id == sid) {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("◉ {}", s.name))
                        .strong()
                        .size(14.0),
                );
                ui.label(format!("target:   {}", s.target));
                ui.label(format!("depth:    {}", s.depth));
                let dur = s.end - s.start;
                ui.label(format!("duration: {}", fmt_dur_short(dur)));
                if !s.fields.is_empty() {
                    ui.label(format!("fields:   {}", s.fields));
                }
                if let Some(t) = s.task {
                    ui.label(format!("on task:  #{}", t.0));
                }
                if let Some(p) = s.parent_id {
                    if let Some(parent) = spans.iter().find(|x| x.span_id == p) {
                        ui.label(format!("parent:   {} (span#{})", parent.name, p));
                    }
                }
            }
        }

        // Critical-path summary list.
        if state.show_critical_path && !crit_path.spans.is_empty() {
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "critical path — total {} (sum of self-time)",
                    fmt_dur_short(Duration::from_nanos(crit_path.total_ns))
                ))
                .strong(),
            );
            egui::ScrollArea::vertical()
                .max_height(150.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for ps in &crit_path.spans {
                        ui.horizontal(|ui| {
                            // Indent by depth so the chain reads visually.
                            ui.add_space(ps.depth as f32 * 12.0);
                            ui.monospace(fmt_dur_short(Duration::from_nanos(ps.self_ns)));
                            ui.label(
                                egui::RichText::new(format!("/ {}", &ps.name))
                                    .color(egui::Color32::from_rgb(255, 180, 180)),
                            );
                            if ui
                                .small_button("→")
                                .on_hover_text("focus this span")
                                .clicked()
                            {
                                state.selected_span = Some(ps.span_id);
                            }
                        });
                    }
                });
        }
    }
}

/// Reconstruct spans from a chronological event slice.
fn build_spans(events: &[Event]) -> Vec<Span> {
    let mut open: HashMap<u64, OpenSpan> = HashMap::new();
    let mut depth_of: HashMap<u64, usize> = HashMap::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut active_stack: Vec<u64> = Vec::new();
    for e in events {
        match &e.kind {
            EventKind::SpanEnter {
                span_id,
                name,
                target,
                parent_id,
                fields,
            } => {
                let depth = parent_id
                    .and_then(|p| depth_of.get(&p).copied())
                    .map(|d| d + 1)
                    .unwrap_or(0);
                depth_of.insert(*span_id, depth);
                open.insert(
                    *span_id,
                    OpenSpan {
                        name: name.to_string(),
                        target: target.to_string(),
                        fields: fields.clone(),
                        parent_id: *parent_id,
                        depth,
                        start: e.at,
                        task: e.task,
                    },
                );
                active_stack.push(*span_id);
            }
            EventKind::SpanExit { span_id } => {
                if let Some(o) = open.remove(span_id) {
                    spans.push(Span {
                        span_id: *span_id,
                        name: o.name,
                        target: o.target,
                        fields: o.fields,
                        parent_id: o.parent_id,
                        depth: o.depth,
                        start: o.start,
                        end: e.at,
                        task: o.task,
                    });
                }
                if let Some(pos) = active_stack.iter().rposition(|x| x == span_id) {
                    active_stack.remove(pos);
                }
            }
            _ => {}
        }
    }
    // Spans still open: close at the time of the last event.
    if let Some(last) = events.last() {
        for (span_id, o) in open {
            spans.push(Span {
                span_id,
                name: o.name,
                target: o.target,
                fields: o.fields,
                parent_id: o.parent_id,
                depth: o.depth,
                start: o.start,
                end: last.at,
                task: o.task,
            });
        }
    }
    spans
}

struct OpenSpan {
    name: String,
    target: String,
    fields: String,
    parent_id: Option<u64>,
    depth: usize,
    start: Instant,
    task: Option<TaskId>,
}

fn span_color(name: &str) -> egui::Color32 {
    // Hash the name into an HSL-ish color so each span name has a
    // consistent stable color across frames and across rows.
    let mut h: u32 = 0;
    for b in name.bytes() {
        h = h.wrapping_mul(2654435761).wrapping_add(b as u32);
    }
    let r = ((h >> 16) & 0xFF) as u8;
    let g = ((h >> 8) & 0xFF) as u8;
    let b = (h & 0xFF) as u8;
    let lift = |c: u8| (c.saturating_add(70)).min(220);
    egui::Color32::from_rgb(lift(r), lift(g), lift(b))
}

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
