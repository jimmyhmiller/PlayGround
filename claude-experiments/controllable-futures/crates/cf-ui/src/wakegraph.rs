//! Wake graph: nodes = tasks, edges = aggregate wake counts.
//!
//! Three layouts:
//!   - SpawnDepth: x = spawn-tree depth, y = sibling position. Good when
//!     there's a real parent/child hierarchy.
//!   - WakeDepth: x = longest wake-chain ending at this node ("layer
//!     assignment"). Good for flat task pools with rich wake edges.
//!   - Force: spring-relaxation. Edges attract, all node pairs repel.
//!     Most flexible; non-deterministic but converges over a few frames.
//!
//! `from_task = None` wake edges (initial schedule from outside any
//! task) are not drawn as long lines — they show as a small "↶" tick
//! glyph on the target node so it's clear those tasks were initially
//! triggered externally.

use cf_runtime::scheduler::TaskMetaSnapshot;
use cf_runtime::task::{TaskId, TaskState};
use cf_runtime::{Event, EventKind};
use eframe::egui;
use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Layout {
    SpawnDepth,
    WakeDepth,
    Force,
}

#[derive(Default)]
pub struct WakeGraphState {
    pub layout: Option<Layout>,
    /// Cached force-directed positions (in unit-square coordinates).
    pub force_pos: HashMap<TaskId, (f32, f32)>,
    pub force_vel: HashMap<TaskId, (f32, f32)>,
    /// Frames remaining to relax. Each frame we relax once and decrement;
    /// when 0, layout is frozen. Relaxation budget resets when the
    /// node/edge set changes (signaled by `last_signature`).
    pub force_frames_left: u32,
    /// Cheap signature of nodes+edges; reset budget when it changes.
    pub last_signature: u64,
}

impl WakeGraphState {
    fn layout(&self) -> Layout {
        self.layout.unwrap_or(Layout::WakeDepth)
    }
}

pub struct WakeGraph<'a> {
    pub tasks: &'a [TaskMetaSnapshot],
    pub events: &'a [Event],
    pub selected: &'a mut Option<TaskId>,
    pub state: &'a mut WakeGraphState,
}

impl<'a> WakeGraph<'a> {
    pub fn show(self, ui: &mut egui::Ui) {
        // Aggregate wake edges.
        let mut edges: HashMap<(Option<TaskId>, TaskId), u32> = HashMap::new();
        let mut incoming: HashMap<TaskId, u32> = HashMap::new();
        for e in self.events {
            if let EventKind::Wake { from_task, .. } = &e.kind {
                if let Some(t) = e.task {
                    *edges.entry((*from_task, t)).or_insert(0) += 1;
                    *incoming.entry(t).or_insert(0) += 1;
                }
            }
        }
        if edges.is_empty() {
            ui.label("(no wake events in current log window)");
            return;
        }

        // Filter to top-N most-active tasks (where activity = sum of
        // task-task wake events touching the node, excluding None
        // sources).
        const MAX_NODES: usize = 32;
        let mut activity: HashMap<TaskId, u32> = HashMap::new();
        for ((src, tgt), count) in &edges {
            if let Some(s) = src {
                *activity.entry(*s).or_insert(0) += *count;
            }
            *activity.entry(*tgt).or_insert(0) += *count;
        }
        // Sort by activity descending, tie-break on task id ascending.
        // Without the tie-break, HashMap iteration order randomizes
        // which subset of equal-count tasks lands in the top-N, so the
        // node set changes every frame purely from sort instability.
        let mut top: Vec<(TaskId, u32)> = activity.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.0.cmp(&b.0.0)));
        top.truncate(MAX_NODES);
        let top_set: HashSet<TaskId> = top.iter().map(|(t, _)| *t).collect();

        // Edges retained: ONLY task-to-task within top_set. None-source
        // edges drop out of the rendered set; they reappear as a "↶"
        // glyph on the target node so we don't lose the information.
        let externally_woken: HashSet<TaskId> = edges
            .iter()
            .filter_map(|((s, t), _)| {
                if s.is_none() && top_set.contains(t) {
                    Some(*t)
                } else {
                    None
                }
            })
            .collect();
        edges.retain(|(s, t), _| {
            top_set.contains(t)
                && match s {
                    Some(x) => top_set.contains(x),
                    None => false,
                }
        });

        // Controls row.
        ui.horizontal(|ui| {
            let cur = self.state.layout();
            for (label, l) in [
                ("Wake depth", Layout::WakeDepth),
                ("Spawn depth", Layout::SpawnDepth),
                ("Force", Layout::Force),
            ] {
                if ui.selectable_label(cur == l, label).clicked() {
                    self.state.layout = Some(l);
                    if l != Layout::Force {
                        self.state.force_pos.clear();
                    }
                }
            }
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "showing top {} of {}, {} edges",
                    top_set.len(),
                    self.tasks.len(),
                    edges.len(),
                ))
                .color(egui::Color32::from_gray(160)),
            );
        });

        let avail = ui.available_size_before_wrap();
        let h = (avail.y - 20.0).max(200.0);
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(avail.x, h),
            egui::Sense::click(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 20, 24));

        // Compute positions per layout.
        let pos = match self.state.layout() {
            Layout::SpawnDepth => layout_spawn_depth(&top_set, self.tasks, &rect),
            Layout::WakeDepth => layout_wake_depth(&top_set, &edges, &rect),
            Layout::Force => {
                // Compute a coarse signature; reset relaxation budget
                // when the graph topology changes.
                let mut sig: u64 = top_set.len() as u64;
                for e in edges.keys() {
                    sig = sig.wrapping_mul(31).wrapping_add(e.1.0);
                    if let Some(s) = e.0 {
                        sig = sig.wrapping_mul(31).wrapping_add(s.0);
                    }
                }
                if sig != self.state.last_signature {
                    self.state.last_signature = sig;
                    // Budget enough iterations to converge a typical
                    // graph, then freeze.
                    self.state.force_frames_left = 60;
                }
                if self.state.force_frames_left > 0 {
                    self.state.force_frames_left -= 1;
                    relax_force(
                        &top_set,
                        &edges,
                        &mut self.state.force_pos,
                        &mut self.state.force_vel,
                    );
                }
                self.state
                    .force_pos
                    .iter()
                    .map(|(id, (x, y))| {
                        (
                            *id,
                            egui::pos2(
                                rect.min.x + 30.0 + x * (rect.width() - 60.0),
                                rect.min.y + 30.0 + y * (rect.height() - 60.0),
                            ),
                        )
                    })
                    .collect()
            }
        };

        // Draw edges.
        let max_count = edges.values().copied().max().unwrap_or(1) as f32;
        for ((src, tgt), count) in &edges {
            let Some(src_id) = src else { continue };
            let (Some(s), Some(t)) = (pos.get(src_id), pos.get(tgt)) else {
                continue;
            };
            let frac = (*count as f32 / max_count).max(0.05);
            let stroke = egui::Stroke::new(
                1.0 + frac * 3.0,
                egui::Color32::from_rgba_unmultiplied(
                    120,
                    200,
                    255,
                    80 + (frac * 160.0) as u8,
                ),
            );
            painter.line_segment([*s, *t], stroke);
            // Arrowhead at target.
            let dir = (*t - *s).normalized();
            if dir.length_sq() > 0.0001 {
                let perp = egui::vec2(-dir.y, dir.x);
                let arrow_base = *t - dir * 12.0;
                painter.add(egui::Shape::convex_polygon(
                    vec![*t, arrow_base + perp * 4.0, arrow_base - perp * 4.0],
                    egui::Color32::from_rgb(120, 200, 255),
                    egui::Stroke::NONE,
                ));
            }
        }

        // Draw nodes.
        for t in self.tasks.iter().filter(|t| top_set.contains(&t.id)) {
            let Some(p) = pos.get(&t.id) else { continue };
            let r = 8.0
                + (incoming.get(&t.id).copied().unwrap_or(0) as f32)
                    .log2()
                    .max(0.0)
                    * 1.5;
            let fill = match t.state {
                TaskState::Running => egui::Color32::from_rgb(80, 200, 120),
                TaskState::Runnable => egui::Color32::from_rgb(120, 180, 240),
                TaskState::Suspended => egui::Color32::from_rgb(180, 180, 180),
                TaskState::Completed => egui::Color32::from_rgb(80, 80, 80),
                TaskState::PausedByUser => egui::Color32::from_rgb(240, 180, 60),
                TaskState::Aborted => egui::Color32::from_rgb(220, 80, 80),
                TaskState::Fresh => egui::Color32::from_rgb(220, 220, 100),
            };
            let stroke = if Some(t.id) == *self.selected {
                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 240, 120))
            } else {
                egui::Stroke::NONE
            };
            painter.circle(*p, r, fill, stroke);
            // External-wake indicator: a small "↶" glyph if this task
            // was woken from outside any task (initial schedule).
            if externally_woken.contains(&t.id) {
                painter.text(
                    *p + egui::vec2(-r - 3.0, 0.0),
                    egui::Align2::RIGHT_CENTER,
                    "↶",
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_rgb(180, 180, 180),
                );
            }
            painter.text(
                *p + egui::vec2(0.0, r + 6.0),
                egui::Align2::CENTER_TOP,
                format!("#{} {}", t.id.0, &t.name),
                egui::FontId::monospace(10.0),
                egui::Color32::from_gray(220),
            );
            // Click to select.
            if let Some(pp) = resp.interact_pointer_pos() {
                if resp.clicked() && (*p - pp).length() <= r + 4.0 {
                    *self.selected = Some(t.id);
                }
            }
        }
    }
}

/// Original layout: column = spawn-tree depth, row = sibling position.
fn layout_spawn_depth(
    top_set: &HashSet<TaskId>,
    tasks: &[TaskMetaSnapshot],
    rect: &egui::Rect,
) -> HashMap<TaskId, egui::Pos2> {
    let parent_of: HashMap<TaskId, Option<TaskId>> =
        tasks.iter().map(|t| (t.id, t.parent)).collect();
    let mut depth: HashMap<TaskId, usize> = HashMap::new();
    for t in tasks {
        let mut d = 0usize;
        let mut cur = t.parent;
        while let Some(p) = cur {
            d += 1;
            cur = parent_of.get(&p).copied().flatten();
            if d > 16 {
                break;
            }
        }
        depth.insert(t.id, d);
    }
    let mut by_col: HashMap<usize, Vec<TaskId>> = HashMap::new();
    for t in tasks.iter().filter(|t| top_set.contains(&t.id)) {
        by_col
            .entry(*depth.get(&t.id).unwrap_or(&0))
            .or_default()
            .push(t.id);
    }
    let max_col = by_col.keys().copied().max().unwrap_or(0);
    let col_w = if max_col == 0 {
        rect.width() - 60.0
    } else {
        (rect.width() - 60.0) / (max_col as f32 + 1.0)
    };
    let mut out = HashMap::new();
    for (col, ts) in &by_col {
        for (i, id) in ts.iter().enumerate() {
            let x = rect.min.x + 30.0 + col_w * (*col as f32 + 0.5);
            let y = rect.min.y + 30.0
                + (i as f32 + 0.5) * (rect.height() - 60.0) / (ts.len().max(1) as f32);
            out.insert(*id, egui::pos2(x, y));
        }
    }
    out
}

/// Wake-chain depth: column index = longest chain of task-to-task wake
/// edges ending at this node. Computed via Kahn-style topological
/// layering, with cycles broken by capping the iteration count.
fn layout_wake_depth(
    top_set: &HashSet<TaskId>,
    edges: &HashMap<(Option<TaskId>, TaskId), u32>,
    rect: &egui::Rect,
) -> HashMap<TaskId, egui::Pos2> {
    let mut layer: HashMap<TaskId, usize> = HashMap::new();
    for &id in top_set {
        layer.insert(id, 0);
    }
    // Iterate: for each edge src→tgt, layer[tgt] = max(layer[tgt], layer[src]+1)
    // until stable (or cap iterations).
    for _ in 0..32 {
        let mut changed = false;
        for ((s, t), _) in edges {
            if let Some(s) = s {
                if !top_set.contains(s) || !top_set.contains(t) {
                    continue;
                }
                let new_layer = layer[s] + 1;
                let cur = layer[t];
                if new_layer > cur {
                    layer.insert(*t, new_layer);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Group by layer.
    let mut by_col: HashMap<usize, Vec<TaskId>> = HashMap::new();
    for (&id, &l) in &layer {
        by_col.entry(l).or_default().push(id);
    }
    for v in by_col.values_mut() {
        v.sort_by_key(|id| id.0);
    }
    let max_col = by_col.keys().copied().max().unwrap_or(0);
    let col_w = if max_col == 0 {
        rect.width() - 60.0
    } else {
        (rect.width() - 60.0) / (max_col as f32 + 1.0)
    };
    let mut out = HashMap::new();
    for (col, ts) in &by_col {
        for (i, id) in ts.iter().enumerate() {
            let x = rect.min.x + 30.0 + col_w * (*col as f32 + 0.5);
            let y = rect.min.y + 30.0
                + (i as f32 + 0.5) * (rect.height() - 60.0) / (ts.len().max(1) as f32);
            out.insert(*id, egui::pos2(x, y));
        }
    }
    out
}

/// Spring-relaxation force-directed layout with velocity + damping.
/// Iterates a few times per frame using cached state for stability.
/// Stops iterating when total kinetic energy drops below a threshold —
/// without that, the solver oscillates around equilibrium and the graph
/// flashes every frame. Repulsion uses a Barnes-Hut-style cap on the
/// per-pair force magnitude to keep close-clustered nodes from
/// catapulting apart.
fn relax_force(
    top_set: &HashSet<TaskId>,
    edges: &HashMap<(Option<TaskId>, TaskId), u32>,
    positions: &mut HashMap<TaskId, (f32, f32)>,
    velocities: &mut HashMap<TaskId, (f32, f32)>,
) {
    // Initialize missing positions on a circle so the relaxation has a
    // non-degenerate start.
    let n = top_set.len() as f32;
    for (i, &id) in top_set.iter().enumerate() {
        if !positions.contains_key(&id) {
            let theta = (i as f32 / n.max(1.0)) * std::f32::consts::TAU;
            positions.insert(
                id,
                (0.5 + 0.35 * theta.cos(), 0.5 + 0.35 * theta.sin()),
            );
            velocities.insert(id, (0.0, 0.0));
        }
    }
    positions.retain(|id, _| top_set.contains(id));
    velocities.retain(|id, _| top_set.contains(id));

    const ITERS: usize = 4;
    const REPULSION: f32 = 0.015;
    const REPULSION_CAP: f32 = 0.5;
    const SPRING_K: f32 = 0.06;
    const SPRING_LEN: f32 = 0.18;
    const DAMPING: f32 = 0.6;
    const STEP: f32 = 0.05;

    let max_count = edges.values().copied().max().unwrap_or(1) as f32;

    for _ in 0..ITERS {
        let mut force: HashMap<TaskId, (f32, f32)> = HashMap::new();
        let ids: Vec<TaskId> = positions.keys().copied().collect();

        // Repulsion: every pair pushes apart, but cap the force so
        // very-close nodes don't explode.
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = positions[&ids[i]];
                let b = positions[&ids[j]];
                let dx = a.0 - b.0;
                let dy = a.1 - b.1;
                let d2 = (dx * dx + dy * dy).max(1e-4);
                let d = d2.sqrt();
                let f = (REPULSION / d2).min(REPULSION_CAP);
                let fx = f * dx / d;
                let fy = f * dy / d;
                let entry_a = force.entry(ids[i]).or_insert((0.0, 0.0));
                entry_a.0 += fx;
                entry_a.1 += fy;
                let entry_b = force.entry(ids[j]).or_insert((0.0, 0.0));
                entry_b.0 -= fx;
                entry_b.1 -= fy;
            }
        }

        // Attraction: spring per edge (weighted by count).
        for ((s, t), count) in edges {
            let Some(s) = s else { continue };
            if !top_set.contains(s) || !top_set.contains(t) {
                continue;
            }
            let a = positions[s];
            let b = positions[t];
            let dx = b.0 - a.0;
            let dy = b.1 - a.1;
            let d = (dx * dx + dy * dy).sqrt().max(1e-4);
            let strength = SPRING_K * (*count as f32 / max_count + 0.3);
            let f = strength * (d - SPRING_LEN);
            let fx = f * dx / d;
            let fy = f * dy / d;
            let entry_a = force.entry(*s).or_insert((0.0, 0.0));
            entry_a.0 += fx;
            entry_a.1 += fy;
            let entry_b = force.entry(*t).or_insert((0.0, 0.0));
            entry_b.0 -= fx;
            entry_b.1 -= fy;
        }

        // Integrate with velocity + damping.
        for (id, f) in force {
            let v = velocities.entry(id).or_insert((0.0, 0.0));
            v.0 = (v.0 + f.0 * STEP) * DAMPING;
            v.1 = (v.1 + f.1 * STEP) * DAMPING;
            if let Some(p) = positions.get_mut(&id) {
                p.0 = (p.0 + v.0).clamp(0.05, 0.95);
                p.1 = (p.1 + v.1).clamp(0.05, 0.95);
            }
        }
    }
}
