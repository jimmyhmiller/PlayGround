//! egui-based inspector for cf-runtime.
//!
//! Layout (left → right):
//!  - Sidebar: scheduler controls (mode, run/pause/step), counters
//!  - Center: task list (sortable, click to select)
//!  - Right: detail panel for the selected task
//!  - Bottom: event log scrubber
//!
//! The UI thread holds a `RuntimeHandle`, snapshots the registry and event log
//! each frame, and sends control commands through `Controller`. It does not
//! itself execute futures.

mod callgraph;
mod critical;
mod funcs;
mod resources;
mod spans;
mod timeline;
mod tree;
mod wakegraph;

use cf_runtime::control::SchedulerMode;
use cf_runtime::scheduler::{TaskMetaSnapshot, WorkerStatus};
use cf_runtime::task::{TaskId, TaskState};
use cf_runtime::{Event, EventKind, RuntimeHandle};

fn worker_status_view(s: WorkerStatus) -> (&'static str, egui::Color32) {
    match s {
        WorkerStatus::Parked => ("parked", egui::Color32::from_gray(120)),
        WorkerStatus::Searching => ("search", egui::Color32::from_rgb(120, 180, 240)),
        WorkerStatus::GateBlocked => ("GATE", egui::Color32::from_rgb(255, 160, 60)),
        WorkerStatus::Running => ("running", egui::Color32::from_rgb(80, 200, 120)),
        WorkerStatus::Exited => ("exited", egui::Color32::from_rgb(200, 80, 80)),
    }
}
use eframe::egui;
use std::time::Duration;

pub fn run(handle: RuntimeHandle) -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("cf-runtime inspector"),
        ..Default::default()
    };
    eframe::run_native(
        "cf-runtime inspector",
        opts,
        Box::new(move |_cc| Ok(Box::new(App::new(Source::Runtime(handle))))),
    )
}

/// UI driver for an Observer (real tokio + hooks). No scheduler available
/// since tokio's scheduler isn't ours; manual mode and worker-state lane
/// are hidden in this mode. Pause/step still work because they're enforced
/// in patched tokio via the same Controller.
pub fn run_observer(observer: std::sync::Arc<cf_runtime::hooks::Observer>) -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("cf-runtime inspector — turbopack"),
        ..Default::default()
    };
    eframe::run_native(
        "cf-runtime inspector",
        opts,
        Box::new(move |_cc| Ok(Box::new(App::new(Source::Observer(observer))))),
    )
}

/// Where the UI's data and control commands flow. Both variants expose
/// the same registry/log/controller/resources surface; only `Runtime`
/// also has a scheduler with manual-queue and worker-status state.
enum Source {
    Runtime(RuntimeHandle),
    Observer(std::sync::Arc<cf_runtime::hooks::Observer>),
}

impl Source {
    fn registry(&self) -> std::sync::Arc<cf_runtime::scheduler::TaskRegistry> {
        match self {
            Source::Runtime(h) => h.registry(),
            Source::Observer(o) => o.registry.clone(),
        }
    }
    fn resources(&self) -> std::sync::Arc<cf_runtime::resource::ResourceRegistry> {
        match self {
            Source::Runtime(h) => h.resources(),
            Source::Observer(o) => o.resources.clone(),
        }
    }
    fn log(&self) -> std::sync::Arc<cf_runtime::EventLog> {
        match self {
            Source::Runtime(h) => h.log(),
            Source::Observer(o) => o.log.clone(),
        }
    }
    fn controller(&self) -> std::sync::Arc<cf_runtime::control::Controller> {
        match self {
            Source::Runtime(h) => h.controller(),
            Source::Observer(o) => o.controller.clone(),
        }
    }
    /// `None` for Observer mode — tokio owns the scheduler and the
    /// scheduler-specific UI surfaces (manual queue, worker status lane,
    /// migrate_to_manual) are hidden.
    fn scheduler(&self) -> Option<std::sync::Arc<cf_runtime::scheduler::Scheduler>> {
        match self {
            Source::Runtime(h) => Some(h.scheduler()),
            Source::Observer(_) => None,
        }
    }
}

struct App {
    handle: Source,
    selected: Option<TaskId>,
    tasks: Vec<TaskMetaSnapshot>,
    events_tail: Vec<Event>,
    sort_by: Sort,
    event_filter: EventFilter,
    central_tab: CentralTab,
    timeline: timeline::TimelineState,
    wakegraph: wakegraph::WakeGraphState,
    spans: spans::SpansState,
    funcs: funcs::FuncsState,
    callgraph: callgraph::CallGraphState,
    /// Time-travel cursor. When set, all event-driven views filter to
    /// events with `at <= cursor`, effectively rewinding the world to
    /// that moment. Synced with the timeline's per-state cursor — set
    /// in either place, both follow.
    time_cursor: Option<std::time::Instant>,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum CentralTab {
    Tasks,
    Timeline,
    Tree,
    Resources,
    WakeGraph,
    Spans,
    Functions,
    CallGraph,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Sort {
    Id,
    Name,
    Polls,
    Busy,
    State,
}

struct EventFilter {
    show_polls: bool,
    show_state_changes: bool,
    show_wakes: bool,
    show_spawns: bool,
    show_completed: bool,
    show_control: bool,
    show_user: bool,
    show_spans: bool,
}

impl EventFilter {
    fn permit(&self, e: &Event) -> bool {
        match &e.kind {
            EventKind::PollStart | EventKind::PollEnd { .. } => self.show_polls,
            EventKind::StateChanged { .. } => self.show_state_changes,
            EventKind::Wake { .. } => self.show_wakes,
            EventKind::Spawned { .. } => self.show_spawns,
            EventKind::Completed | EventKind::Aborted => self.show_completed,
            EventKind::Control(_) => self.show_control,
            EventKind::User { .. } => self.show_user,
            EventKind::SpanEnter { .. }
            | EventKind::SpanExit { .. }
            | EventKind::SpanClose { .. }
            | EventKind::SpanEvent { .. }
            | EventKind::SpanAllocs { .. } => self.show_spans,
        }
    }
}

impl App {
    fn new(handle: Source) -> Self {
        Self {
            handle,
            selected: None,
            tasks: Vec::new(),
            events_tail: Vec::new(),
            sort_by: Sort::Id,
            event_filter: EventFilter {
                show_polls: true,
                show_state_changes: true,
                show_wakes: true,
                show_spawns: true,
                show_completed: true,
                show_control: true,
                show_user: true,
                show_spans: true,
            },
            central_tab: CentralTab::Functions,
            timeline: timeline::TimelineState::default(),
            wakegraph: wakegraph::WakeGraphState::default(),
            spans: spans::SpansState::default(),
            funcs: funcs::FuncsState::default(),
            callgraph: callgraph::CallGraphState::default(),
            time_cursor: None,
        }
    }

    fn refresh(&mut self, frozen: bool) {
        // When frozen (runtime paused), don't re-snapshot anything —
        // the user explicitly asked the world to stop. Re-snapshotting
        // would let new spawn events sneak into the view (they happen
        // on the block_on driver thread, which isn't gated by the
        // worker pause), causing top-N membership to churn and the
        // wake graph to re-relax constantly.
        if frozen {
            return;
        }
        self.tasks = Source::registry(&self.handle).snapshot();
        match self.sort_by {
            Sort::Id => self.tasks.sort_by_key(|t| t.id.0),
            Sort::Name => self.tasks.sort_by(|a, b| a.name.cmp(&b.name)),
            Sort::Polls => self.tasks.sort_by(|a, b| b.poll_count.cmp(&a.poll_count)),
            Sort::Busy => self.tasks.sort_by(|a, b| b.busy_nanos.cmp(&a.busy_nanos)),
            Sort::State => self.tasks.sort_by_key(|t| state_sort_key(t.state)),
        }
        // Pull a much wider window now that tracing spans inflate event
        // counts (~5–10× the bare tokio events). Capped on the runtime
        // side at the EventLog's ring capacity.
        self.events_tail = self.handle.log().tail(150_000);
    }
}

fn state_sort_key(s: TaskState) -> u8 {
    match s {
        TaskState::Running => 0,
        TaskState::Runnable => 1,
        TaskState::Suspended => 2,
        TaskState::Fresh => 3,
        TaskState::PausedByUser => 4,
        TaskState::Completed => 5,
        TaskState::Aborted => 6,
    }
}

fn state_color(s: TaskState) -> egui::Color32 {
    match s {
        TaskState::Running => egui::Color32::from_rgb(80, 200, 120),
        TaskState::Runnable => egui::Color32::from_rgb(120, 180, 240),
        TaskState::Suspended => egui::Color32::from_rgb(180, 180, 180),
        TaskState::Fresh => egui::Color32::from_rgb(220, 220, 100),
        TaskState::PausedByUser => egui::Color32::from_rgb(240, 180, 60),
        TaskState::Completed => egui::Color32::from_rgb(120, 120, 120),
        TaskState::Aborted => egui::Color32::from_rgb(220, 80, 80),
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let controller = self.handle.controller();
        let scheduler = self.handle.scheduler();
        let paused = controller.is_paused();

        // When paused, don't request periodic repaints — only repaint on
        // actual user input. This kills the every-100ms churn that was
        // making the wake graph and timeline appear to flicker even with
        // no underlying state changes.
        if !paused {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        self.refresh(paused);
        // Keep app-level `time_cursor` synchronized with the timeline's
        // local cursor. Setting it from either side makes the other
        // follow on the next frame.
        if self.timeline.cursor != self.time_cursor {
            // Timeline holds priority — its UI is where users place
            // the cursor. App-level just mirrors.
            self.time_cursor = self.timeline.cursor;
        }
        // Compute the time-filtered event slice once. All event-driven
        // views read this instead of `events_tail` so they all rewind
        // together when a cursor is placed.
        let visible_events: &[cf_runtime::Event] = match self.time_cursor {
            Some(c) => {
                let idx = self
                    .events_tail
                    .partition_point(|e| e.at <= c);
                &self.events_tail[..idx]
            }
            None => &self.events_tail,
        };

        // Top banner — the most visible way to confirm at a glance whether
        // the runtime is running freely, paused, or in manual mode.
        let mode = controller.mode();
        // Time-travel banner takes priority over pause/manual banners
        // so the user sees the most-relevant state when several apply.
        if let Some(c) = self.time_cursor {
            let lag = self
                .events_tail
                .last()
                .map(|e| e.at.saturating_duration_since(c))
                .unwrap_or_default();
            egui::TopBottomPanel::top("time-travel-banner").show(ctx, |ui| {
                let resp = ui.allocate_response(
                    egui::vec2(ui.available_width(), 24.0),
                    egui::Sense::click(),
                );
                ui.painter().rect_filled(
                    resp.rect,
                    0.0,
                    egui::Color32::from_rgb(220, 100, 220),
                );
                let txt = format!(
                    "⏱  TIME-TRAVEL — frozen at -{}ms behind latest event   (click to release)",
                    lag.as_millis(),
                );
                ui.painter().text(
                    resp.rect.left_center() + egui::vec2(8.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    txt,
                    egui::FontId::proportional(14.0),
                    egui::Color32::BLACK,
                );
                if resp.clicked() {
                    self.time_cursor = None;
                    self.timeline.cursor = None;
                }
            });
        } else if paused || mode == SchedulerMode::Manual {
            egui::TopBottomPanel::top("status-banner").show(ctx, |ui| {
                let (txt, fg, bg) = if paused {
                    (
                        "⏸  PAUSED — workers blocked at next poll gate",
                        egui::Color32::BLACK,
                        egui::Color32::from_rgb(255, 200, 80),
                    )
                } else {
                    (
                        "🛠  MANUAL MODE — only released tasks run",
                        egui::Color32::BLACK,
                        egui::Color32::from_rgb(160, 200, 255),
                    )
                };
                let resp = ui.allocate_response(
                    egui::vec2(ui.available_width(), 24.0),
                    egui::Sense::hover(),
                );
                ui.painter().rect_filled(resp.rect, 0.0, bg);
                ui.painter().text(
                    resp.rect.left_center() + egui::vec2(8.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    txt,
                    egui::FontId::proportional(14.0),
                    fg,
                );
            });
        }

        egui::SidePanel::left("controls")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("scheduler");
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(mode == SchedulerMode::Auto, "Auto")
                        .clicked()
                    {
                        controller.set_mode(SchedulerMode::Auto);
                    }
                    // Manual mode is only meaningful when we own the
                    // scheduler. With patched-tokio we can pause but not
                    // run-one-at-a-time the same way.
                    if scheduler.is_some()
                        && ui
                            .selectable_label(mode == SchedulerMode::Manual, "Manual")
                            .clicked()
                    {
                        controller.set_mode(SchedulerMode::Manual);
                        if let Some(ref s) = scheduler {
                            s.migrate_to_manual();
                        }
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    if paused {
                        if ui.button("▶ resume").clicked() {
                            controller.resume();
                        }
                    } else if ui.button("⏸ pause").clicked() {
                        controller.pause();
                    }
                    if ui.button("step 1").clicked() {
                        controller.step(1);
                    }
                    if ui.button("step 10").clicked() {
                        controller.step(10);
                    }
                });
                ui.separator();
                if let Some(ref s) = scheduler {
                    ui.label(format!("workers: {}", s.n_workers()));
                    let status = s.worker_status.snapshot();
                    for (i, st) in status.iter().enumerate() {
                        let (txt, color) = worker_status_view(*st);
                        ui.horizontal(|ui| {
                            ui.monospace(format!("  w{i}"));
                            ui.label(egui::RichText::new(txt).color(color).monospace());
                        });
                    }
                } else {
                    ui.label("workers: (tokio)");
                }
                ui.label(format!("tasks:   {}", self.tasks.len()));
                let count = |s: TaskState| {
                    self.tasks.iter().filter(|t| t.state == s).count()
                };
                ui.label(format!("  running:   {}", count(TaskState::Running)));
                ui.label(format!("  runnable:  {}", count(TaskState::Runnable)));
                ui.label(format!("  suspended: {}", count(TaskState::Suspended)));
                ui.label(format!("  completed: {}", count(TaskState::Completed)));

                if mode == SchedulerMode::Manual {
                    if let Some(ref s) = scheduler {
                        ui.separator();
                        ui.heading("manual queue");
                        let pending = s.manual_queue();
                        if pending.is_empty() {
                            ui.label("(empty)");
                        } else {
                            for (slab_key, task_id) in pending {
                                let name = self
                                    .tasks
                                    .iter()
                                    .find(|t| t.id == task_id)
                                    .map(|t| t.name.clone())
                                    .unwrap_or_else(|| "?".into());
                                ui.horizontal(|ui| {
                                    ui.label(format!("#{} {}", task_id.0, name));
                                    if ui.small_button("run").clicked() {
                                        s.manual_run(slab_key);
                                    }
                                });
                            }
                        }
                    }
                }
            });

        egui::TopBottomPanel::bottom("events")
            .resizable(true)
            .default_height(220.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("events");
                    ui.separator();
                    ui.checkbox(&mut self.event_filter.show_polls, "polls");
                    ui.checkbox(&mut self.event_filter.show_state_changes, "state");
                    ui.checkbox(&mut self.event_filter.show_wakes, "wakes");
                    ui.checkbox(&mut self.event_filter.show_spawns, "spawns");
                    ui.checkbox(&mut self.event_filter.show_completed, "done");
                    ui.checkbox(&mut self.event_filter.show_control, "ctl");
                    ui.checkbox(&mut self.event_filter.show_user, "user");
                    if let Some(id) = self.selected {
                        ui.separator();
                        ui.label(format!("filtering to #{}", id.0));
                        if ui.small_button("clear").clicked() {
                            self.selected = None;
                        }
                    }
                });
                let filter = &self.event_filter;
                let selected = self.selected;
                let events: Vec<&Event> = self
                    .events_tail
                    .iter()
                    .filter(|e| filter.permit(e))
                    .filter(|e| selected.is_none() || e.task == selected)
                    .collect();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for e in events {
                            ui.horizontal(|ui| {
                                ui.monospace(format!("{:>6}", e.seq));
                                if let Some(w) = e.worker {
                                    ui.monospace(format!("w{w}"));
                                } else {
                                    ui.monospace("  ");
                                }
                                if let Some(t) = e.task {
                                    let name = self
                                        .tasks
                                        .iter()
                                        .find(|x| x.id == t)
                                        .map(|x| x.name.as_str())
                                        .unwrap_or("?");
                                    ui.monospace(format!("#{:>3} {:18}", t.0, name));
                                } else {
                                    ui.monospace(format!("{:>3} {:18}", "-", ""));
                                }
                                ui.label(format_event_kind(&e.kind));
                            });
                        }
                    });
            });

        egui::SidePanel::right("detail")
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.heading("task detail");
                let Some(id) = self.selected else {
                    ui.label("(no task selected)");
                    return;
                };
                let Some(t) = self.tasks.iter().find(|t| t.id == id).cloned() else {
                    ui.label("(task gone)");
                    return;
                };
                ui.label(format!("id:        #{}", t.id.0));
                ui.label(format!("name:      {}", t.name));
                if let Some(p) = t.parent {
                    ui.label(format!("parent:    #{}", p.0));
                } else {
                    ui.label("parent:    (none)");
                }
                ui.label(
                    egui::RichText::new(format!("state:     {:?}", t.state))
                        .color(state_color(t.state)),
                );
                ui.label(format!("polls:     {}", t.poll_count));
                ui.label(format!("busy:      {} µs", t.busy_nanos / 1000));
                ui.label(format!("age:       {} ms", t.age_nanos / 1_000_000));
                if let Some(w) = t.last_worker {
                    ui.label(format!("last wkr:  {w}"));
                }
                ui.label(format!("wake pend: {}", t.wake_pending));
                ui.label(format!(
                    "future sz: {} bytes",
                    t.future_size_bytes
                ));
                if let Some(reason) = &t.wait_reason {
                    ui.separator();
                    ui.label(
                        egui::RichText::new("waiting on:")
                            .color(egui::Color32::from_gray(140)),
                    );
                    ui.label(
                        egui::RichText::new(format!("{reason}"))
                            .monospace()
                            .color(egui::Color32::from_rgb(180, 220, 255)),
                    );
                }
                if !t.recent_poll_nanos.is_empty() {
                    let mut s = t.recent_poll_nanos.clone();
                    s.sort_unstable();
                    let pct = |p: f64| {
                        let i = ((s.len() as f64 - 1.0) * p).round() as usize;
                        s[i.min(s.len() - 1)]
                    };
                    let p50 = pct(0.5);
                    let p95 = pct(0.95);
                    let p99 = pct(0.99);
                    let max = *s.last().unwrap();
                    ui.separator();
                    ui.label(format!(
                        "poll µs (last {}): p50={} p95={} p99={} max={}",
                        s.len(),
                        p50 / 1000,
                        p95 / 1000,
                        p99 / 1000,
                        max / 1000,
                    ));
                    // Tiny sparkline of the most recent durations (in their
                    // original arrival order, not sorted).
                    let raw = &t.recent_poll_nanos;
                    let max_v = *raw.iter().max().unwrap_or(&1) as f32;
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width().min(280.0), 24.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(
                        rect,
                        2.0,
                        egui::Color32::from_rgb(20, 20, 24),
                    );
                    let n = raw.len() as f32;
                    if n >= 2.0 {
                        let mut prev: Option<egui::Pos2> = None;
                        for (i, v) in raw.iter().enumerate() {
                            let x = rect.min.x + (i as f32 / (n - 1.0)) * rect.width();
                            let y = rect.max.y
                                - ((*v as f32 / max_v) * (rect.height() - 2.0));
                            let p = egui::pos2(x, y);
                            if let Some(prev) = prev {
                                ui.painter().line_segment(
                                    [prev, p],
                                    egui::Stroke::new(
                                        1.0,
                                        egui::Color32::from_rgb(120, 200, 255),
                                    ),
                                );
                            }
                            prev = Some(p);
                        }
                    }
                }
                if let Some(susp_ns) = t.suspended_for_nanos {
                    let ms = susp_ns / 1_000_000;
                    let label = if ms > 5_000 {
                        egui::RichText::new(format!("suspended {} ms", ms))
                            .color(egui::Color32::from_rgb(255, 160, 60))
                    } else {
                        egui::RichText::new(format!("suspended {} ms", ms))
                            .color(egui::Color32::from_gray(180))
                    };
                    ui.label(label);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    let task_paused = controller.is_task_paused(id);
                    if task_paused {
                        if ui.button("resume task").clicked() {
                            controller.resume_task(id);
                        }
                    } else if ui.button("pause task").clicked() {
                        controller.pause_task(id);
                    }
                    if ui.button("break next poll").clicked() {
                        controller.add_breakpoint(id);
                    }
                });

                // Wake stats — count incoming wakes by source within the
                // most-recent event log slice. Cheap; we already have the
                // events_tail snapshot.
                ui.separator();
                let mut total = 0usize;
                let mut from_reactor = 0usize;
                let mut by_task: std::collections::HashMap<TaskId, usize> =
                    std::collections::HashMap::new();
                for e in &self.events_tail {
                    if e.task != Some(id) {
                        continue;
                    }
                    if let EventKind::Wake { from_task, .. } = &e.kind {
                        total += 1;
                        match from_task {
                            Some(t) => *by_task.entry(*t).or_insert(0) += 1,
                            None => from_reactor += 1,
                        }
                    }
                }
                ui.label(
                    egui::RichText::new(format!(
                        "wakes (last {}): {}",
                        self.events_tail.len(),
                        total
                    ))
                    .color(egui::Color32::from_gray(180)),
                );
                if from_reactor > 0 {
                    ui.label(format!("  reactor: {from_reactor}"));
                }
                let mut sources: Vec<_> = by_task.into_iter().collect();
                sources.sort_by(|a, b| b.1.cmp(&a.1));
                for (src, n) in sources.into_iter().take(5) {
                    let name = self
                        .tasks
                        .iter()
                        .find(|t| t.id == src)
                        .map(|t| t.name.as_str())
                        .unwrap_or("?");
                    ui.label(format!("  #{} {}: {}×", src.0, name, n));
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                for (label, t) in [
                    ("Functions", CentralTab::Functions),
                    ("Spans", CentralTab::Spans),
                    ("Call graph", CentralTab::CallGraph),
                    ("Timeline", CentralTab::Timeline),
                    ("Tasks", CentralTab::Tasks),
                    ("Tree", CentralTab::Tree),
                    ("Resources", CentralTab::Resources),
                    ("Wake graph", CentralTab::WakeGraph),
                ] {
                    if ui.selectable_label(self.central_tab == t, label).clicked() {
                        self.central_tab = t;
                    }
                }
                ui.separator();
                if matches!(self.central_tab, CentralTab::Tasks) {
                    ui.label("sort:");
                    for (label, s) in [
                        ("id", Sort::Id),
                        ("name", Sort::Name),
                        ("polls", Sort::Polls),
                        ("busy", Sort::Busy),
                        ("state", Sort::State),
                    ] {
                        if ui.selectable_label(self.sort_by == s, label).clicked() {
                            self.sort_by = s;
                        }
                    }
                }
            });
            ui.separator();
            match self.central_tab {
                CentralTab::Timeline => {
                    // For Observer-mode (real tokio), we don't know the worker
                    // count from the scheduler — guess from CPU count. Refine
                    // later by reading tokio's metrics.
                    let n = scheduler
                        .as_ref()
                        .map(|s| s.n_workers())
                        .unwrap_or_else(|| {
                            std::thread::available_parallelism()
                                .map(|p| p.get())
                                .unwrap_or(4)
                        });
                    timeline::TimelineDraw {
                        state: &mut self.timeline,
                        events: visible_events,
                        tasks: &self.tasks,
                        n_workers: n,
                        selected_task: &mut self.selected,
                    }
                    .show(ui);
                    return;
                }
                CentralTab::Tree => {
                    tree::TreeView {
                        tasks: &self.tasks,
                        selected: &mut self.selected,
                    }
                    .show(ui);
                    return;
                }
                CentralTab::Resources => {
                    let snap = self.handle.resources().snapshot();
                    resources::ResourcesView { items: &snap }.show(ui);
                    return;
                }
                CentralTab::WakeGraph => {
                    wakegraph::WakeGraph {
                        tasks: &self.tasks,
                        events: visible_events,
                        selected: &mut self.selected,
                        state: &mut self.wakegraph,
                    }
                    .show(ui);
                    return;
                }
                CentralTab::Spans => {
                    spans::SpansView {
                        events: visible_events,
                        state: &mut self.spans,
                    }
                    .show(ui);
                    return;
                }
                CentralTab::Functions => {
                    funcs::FuncsView {
                        events: visible_events,
                        state: &mut self.funcs,
                    }
                    .show(ui);
                    return;
                }
                CentralTab::CallGraph => {
                    callgraph::CallGraphView {
                        events: visible_events,
                        state: &mut self.callgraph,
                    }
                    .show(ui);
                    return;
                }
                CentralTab::Tasks => {}
            }
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let header = |ui: &mut egui::Ui, txt: &str, w: f32| {
                        ui.allocate_ui(egui::vec2(w, 18.0), |ui| {
                            ui.label(egui::RichText::new(txt).color(egui::Color32::GRAY));
                        });
                    };
                    ui.horizontal(|ui| {
                        header(ui, "id", 40.0);
                        header(ui, "name", 200.0);
                        header(ui, "state", 80.0);
                        header(ui, "polls", 50.0);
                        header(ui, "busy(µs)", 70.0);
                        header(ui, "wkr", 30.0);
                        header(ui, "waiting on", 280.0);
                    });
                    for t in self.tasks.clone() {
                        let selected = self.selected == Some(t.id);
                        let resp = ui.horizontal(|ui| {
                            ui.allocate_ui(egui::vec2(40.0, 18.0), |ui| {
                                ui.monospace(format!("#{}", t.id.0));
                            });
                            ui.allocate_ui(egui::vec2(200.0, 18.0), |ui| {
                                ui.label(&t.name);
                            });
                            ui.allocate_ui(egui::vec2(80.0, 18.0), |ui| {
                                ui.label(
                                    egui::RichText::new(format!("{:?}", t.state))
                                        .color(state_color(t.state)),
                                );
                            });
                            ui.allocate_ui(egui::vec2(50.0, 18.0), |ui| {
                                ui.monospace(format!("{}", t.poll_count));
                            });
                            ui.allocate_ui(egui::vec2(70.0, 18.0), |ui| {
                                ui.monospace(format!("{}", t.busy_nanos / 1000));
                            });
                            ui.allocate_ui(egui::vec2(30.0, 18.0), |ui| {
                                if let Some(w) = t.last_worker {
                                    ui.monospace(format!("{w}"));
                                }
                            });
                            ui.allocate_ui(egui::vec2(280.0, 18.0), |ui| {
                                if let Some(reason) = &t.wait_reason {
                                    ui.label(
                                        egui::RichText::new(format!("{reason}"))
                                            .monospace()
                                            .size(10.0)
                                            .color(egui::Color32::from_rgb(180, 220, 255)),
                                    );
                                }
                            });
                        });
                        let row_rect = resp.response.rect;
                        let click_resp = ui.interact(
                            row_rect,
                            ui.id().with(("row", t.id.0)),
                            egui::Sense::click(),
                        );
                        if selected {
                            ui.painter().rect_filled(
                                row_rect,
                                2.0,
                                egui::Color32::from_rgba_unmultiplied(80, 120, 200, 40),
                            );
                        }
                        if click_resp.clicked() {
                            self.selected = Some(t.id);
                        }
                    }
                });
        });
    }
}

fn format_event_kind(k: &EventKind) -> String {
    match k {
        EventKind::Spawned { name, parent } => {
            format!("spawned {} parent={:?}", name, parent.map(|p| p.0))
        }
        EventKind::PollStart => "poll start".to_string(),
        EventKind::PollEnd {
            rescheduled,
            duration_nanos,
        } => format!(
            "poll end{} {}ns",
            if *rescheduled { " (resched)" } else { "" },
            duration_nanos
        ),
        EventKind::Wake {
            from_worker,
            from_task,
        } => match from_task {
            Some(t) => format!("wake from task #{} (w{:?})", t.0, from_worker),
            None => format!("wake from={:?}", from_worker),
        },
        EventKind::StateChanged { from, to } => format!("state {:?}→{:?}", from, to),
        EventKind::Completed => "completed".to_string(),
        EventKind::Aborted => "aborted".to_string(),
        EventKind::Control(s) => format!("control: {s}"),
        EventKind::User { category, detail } => {
            if detail.is_empty() {
                format!("user[{category}]")
            } else {
                format!("user[{category}] {detail}")
            }
        }
        EventKind::SpanEnter {
            span_id, name, fields, ..
        } => {
            if fields.is_empty() {
                format!("→ span#{span_id} {name}")
            } else {
                format!("→ span#{span_id} {name} {{ {fields} }}")
            }
        }
        EventKind::SpanExit { span_id } => format!("← span#{span_id}"),
        EventKind::SpanClose { span_id } => format!("✕ span#{span_id}"),
        EventKind::SpanEvent {
            target,
            level,
            message,
            ..
        } => format!("[{level} {target}] {message}"),
        EventKind::SpanAllocs {
            span_id,
            bytes_delta,
            count_delta,
        } => format!(
            "↗ span#{span_id} allocs {:+}B / {:+}",
            bytes_delta, count_delta
        ),
    }
}
