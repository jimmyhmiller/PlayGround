//! Per-function aggregation view.
//!
//! Groups spans by name, computes count + total/mean/p50/p95/p99
//! duration, renders as a sortable table. The "where is the build
//! actually spending its time" view — the answer is usually one or two
//! span names dominate, and this surfaces them immediately.

use cf_runtime::{Event, EventKind};
use eframe::egui;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct FuncsState {
    pub sort_by: SortBy,
    /// Optional name filter to grep through long lists.
    pub filter: String,
    pub selected: Option<String>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SortBy {
    TotalTime,
    Count,
    P50,
    P95,
    P99,
    Max,
    Mean,
    Name,
}

impl Default for SortBy {
    fn default() -> Self {
        Self::TotalTime
    }
}

#[derive(Clone)]
struct FuncStats {
    name: String,
    count: u64,
    total_ns: u64,
    durations_sorted_ns: Vec<u64>,
}

impl FuncStats {
    fn pct(&self, p: f64) -> u64 {
        if self.durations_sorted_ns.is_empty() {
            return 0;
        }
        let idx = ((self.durations_sorted_ns.len() as f64 - 1.0) * p).round()
            as usize;
        self.durations_sorted_ns[idx.min(self.durations_sorted_ns.len() - 1)]
    }
    fn mean(&self) -> u64 {
        if self.count == 0 {
            return 0;
        }
        self.total_ns / self.count
    }
    fn max(&self) -> u64 {
        self.durations_sorted_ns.last().copied().unwrap_or(0)
    }
}

pub struct FuncsView<'a> {
    pub events: &'a [Event],
    pub state: &'a mut FuncsState,
}

impl<'a> FuncsView<'a> {
    pub fn show(self, ui: &mut egui::Ui) {
        let stats = aggregate(self.events);
        if stats.is_empty() {
            ui.label("(no spans yet)");
            return;
        }

        ui.horizontal(|ui| {
            ui.heading("functions");
            ui.separator();
            ui.label(format!("{} unique span names", stats.len()));
            ui.separator();
            ui.label("filter:");
            ui.text_edit_singleline(&mut self.state.filter);
        });

        let filter_lc = self.state.filter.to_lowercase();
        let mut rows: Vec<&FuncStats> = stats
            .values()
            .filter(|s| filter_lc.is_empty() || s.name.to_lowercase().contains(&filter_lc))
            .collect();
        match self.state.sort_by {
            SortBy::TotalTime => rows.sort_by(|a, b| b.total_ns.cmp(&a.total_ns)),
            SortBy::Count => rows.sort_by(|a, b| b.count.cmp(&a.count)),
            SortBy::P50 => rows.sort_by(|a, b| b.pct(0.5).cmp(&a.pct(0.5))),
            SortBy::P95 => rows.sort_by(|a, b| b.pct(0.95).cmp(&a.pct(0.95))),
            SortBy::P99 => rows.sort_by(|a, b| b.pct(0.99).cmp(&a.pct(0.99))),
            SortBy::Max => rows.sort_by(|a, b| b.max().cmp(&a.max())),
            SortBy::Mean => rows.sort_by(|a, b| b.mean().cmp(&a.mean())),
            SortBy::Name => rows.sort_by(|a, b| a.name.cmp(&b.name)),
        }

        // Bar-chart spotlight: visualize total time as a horizontal bar
        // beneath each row. Lets the eye see relative cost without
        // reading numbers.
        let max_total = rows.iter().map(|r| r.total_ns).max().unwrap_or(1) as f32;

        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut header = |ui: &mut egui::Ui, txt: &str, w: f32, sort: Option<SortBy>| {
                    ui.allocate_ui(egui::vec2(w, 18.0), |ui| {
                        let active = sort == Some(self.state.sort_by);
                        let label = if active {
                            egui::RichText::new(format!("▼ {txt}"))
                                .color(egui::Color32::WHITE)
                                .strong()
                        } else {
                            egui::RichText::new(txt).color(egui::Color32::GRAY)
                        };
                        if ui.selectable_label(false, label).clicked() {
                            if let Some(s) = sort {
                                self.state.sort_by = s;
                            }
                        }
                    });
                };
                ui.horizontal(|ui| {
                    header(ui, "name", 320.0, Some(SortBy::Name));
                    header(ui, "count", 70.0, Some(SortBy::Count));
                    header(ui, "total", 90.0, Some(SortBy::TotalTime));
                    header(ui, "mean", 80.0, Some(SortBy::Mean));
                    header(ui, "p50", 80.0, Some(SortBy::P50));
                    header(ui, "p95", 80.0, Some(SortBy::P95));
                    header(ui, "p99", 80.0, Some(SortBy::P99));
                    header(ui, "max", 80.0, Some(SortBy::Max));
                    header(ui, "share", 200.0, None);
                });
                ui.separator();
                for s in &rows {
                    let selected = self.state.selected.as_deref() == Some(s.name.as_str());
                    let resp = ui.horizontal(|ui| {
                        ui.allocate_ui(egui::vec2(320.0, 18.0), |ui| {
                            ui.label(&s.name);
                        });
                        ui.allocate_ui(egui::vec2(70.0, 18.0), |ui| {
                            ui.monospace(format!("{}", s.count));
                        });
                        ui.allocate_ui(egui::vec2(90.0, 18.0), |ui| {
                            ui.monospace(fmt_ns(s.total_ns));
                        });
                        ui.allocate_ui(egui::vec2(80.0, 18.0), |ui| {
                            ui.monospace(fmt_ns(s.mean()));
                        });
                        ui.allocate_ui(egui::vec2(80.0, 18.0), |ui| {
                            ui.monospace(fmt_ns(s.pct(0.5)));
                        });
                        ui.allocate_ui(egui::vec2(80.0, 18.0), |ui| {
                            ui.monospace(fmt_ns(s.pct(0.95)));
                        });
                        ui.allocate_ui(egui::vec2(80.0, 18.0), |ui| {
                            ui.monospace(fmt_ns(s.pct(0.99)));
                        });
                        ui.allocate_ui(egui::vec2(80.0, 18.0), |ui| {
                            ui.monospace(fmt_ns(s.max()));
                        });
                        ui.allocate_ui(egui::vec2(200.0, 18.0), |ui| {
                            // Bar: width proportional to total time.
                            let frac = (s.total_ns as f32 / max_total).clamp(0.0, 1.0);
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(195.0, 12.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(
                                rect,
                                2.0,
                                egui::Color32::from_rgb(40, 40, 50),
                            );
                            let bar_rect = egui::Rect::from_min_size(
                                rect.min,
                                egui::vec2(rect.width() * frac, rect.height()),
                            );
                            ui.painter().rect_filled(
                                bar_rect,
                                2.0,
                                egui::Color32::from_rgb(120, 200, 255),
                            );
                        });
                    });
                    let row_rect = resp.response.rect;
                    let click = ui.interact(
                        row_rect,
                        ui.id().with(("func-row", s.name.as_str())),
                        egui::Sense::click(),
                    );
                    if selected {
                        ui.painter().rect_filled(
                            row_rect,
                            2.0,
                            egui::Color32::from_rgba_unmultiplied(80, 120, 200, 40),
                        );
                    }
                    if click.clicked() {
                        self.state.selected = Some(s.name.clone());
                    }
                }
            });
    }
}

/// Build per-name aggregates from the event log. Pairs SpanEnter with
/// SpanExit by span_id, computes durations, groups by name.
fn aggregate(events: &[Event]) -> HashMap<String, FuncStats> {
    let mut starts: HashMap<u64, (String, Instant)> = HashMap::new();
    let mut by_name: HashMap<String, FuncStats> = HashMap::new();
    for e in events {
        match &e.kind {
            EventKind::SpanEnter { span_id, name, .. } => {
                starts.insert(*span_id, ((*name).to_string(), e.at));
            }
            EventKind::SpanExit { span_id } => {
                if let Some((name, start)) = starts.remove(span_id) {
                    let dur = (e.at - start).as_nanos() as u64;
                    let entry = by_name.entry(name.clone()).or_insert_with(|| FuncStats {
                        name,
                        count: 0,
                        total_ns: 0,
                        durations_sorted_ns: Vec::new(),
                    });
                    entry.count += 1;
                    entry.total_ns += dur;
                    entry.durations_sorted_ns.push(dur);
                }
            }
            _ => {}
        }
    }
    // Sort each function's durations once so percentile lookups are O(1).
    for s in by_name.values_mut() {
        s.durations_sorted_ns.sort_unstable();
    }
    by_name
}

fn fmt_ns(ns: u64) -> String {
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

// silence unused warning for Duration if we don't end up using it
#[allow(dead_code)]
fn _suppress(_: Duration) {}
