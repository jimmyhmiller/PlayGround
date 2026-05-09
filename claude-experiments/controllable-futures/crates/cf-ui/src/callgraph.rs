//! Call-graph view: aggregates parent-child span relationships into a
//! directed graph where each node is a unique span *name* (not span id)
//! and each edge is "callers of name A invoked something named B, K
//! times, totalling Tns."
//!
//! For turbopack: this is the actual computation graph — `Build` calls
//! `Resolve module` calls `Read file before write` calls `Parse JS`,
//! etc. Independent of which physical task instance any single span
//! belonged to.
//!
//! Layout: force-directed, same algorithm as wakegraph but operating
//! on names. Nodes are sized by total time spent in that name; edges
//! are sized by call count.

use cf_runtime::{Event, EventKind};
use eframe::egui;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct CallGraphState {
    pub force_pos: HashMap<String, (f32, f32)>,
    pub force_vel: HashMap<String, (f32, f32)>,
    pub frames_left: u32,
    pub last_signature: u64,
    pub selected: Option<String>,
}

pub struct CallGraphView<'a> {
    pub events: &'a [Event],
    pub state: &'a mut CallGraphState,
}

#[derive(Clone)]
struct NodeAgg {
    name: String,
    count: u64,
    total_ns: u64,
}

impl<'a> CallGraphView<'a> {
    pub fn show(self, ui: &mut egui::Ui) {
        let CallGraphView { events, state } = self;

        // Aggregate.
        let mut node_aggs: HashMap<String, NodeAgg> = HashMap::new();
        let mut edge_counts: HashMap<(String, String), u64> = HashMap::new();
        let mut starts: HashMap<u64, (String, Option<u64>, Instant)> =
            HashMap::new();
        // Span-id → name, for parent lookup.
        let mut id_to_name: HashMap<u64, String> = HashMap::new();
        for e in events {
            match &e.kind {
                EventKind::SpanEnter {
                    span_id,
                    name,
                    parent_id,
                    ..
                } => {
                    starts.insert(*span_id, ((*name).to_string(), *parent_id, e.at));
                    id_to_name.insert(*span_id, (*name).to_string());
                }
                EventKind::SpanExit { span_id } => {
                    if let Some((name, parent_id, start)) = starts.remove(span_id) {
                        let dur = (e.at - start).as_nanos() as u64;
                        let entry =
                            node_aggs.entry(name.clone()).or_insert(NodeAgg {
                                name: name.clone(),
                                count: 0,
                                total_ns: 0,
                            });
                        entry.count += 1;
                        entry.total_ns += dur;
                        if let Some(p) = parent_id {
                            if let Some(parent_name) = id_to_name.get(&p) {
                                let key =
                                    (parent_name.clone(), name.clone());
                                *edge_counts.entry(key).or_insert(0) += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if node_aggs.is_empty() {
            ui.label("(no spans yet)");
            return;
        }

        // Top-N by total_ns.
        const MAX_NODES: usize = 32;
        let mut top: Vec<&NodeAgg> = node_aggs.values().collect();
        top.sort_by(|a, b| b.total_ns.cmp(&a.total_ns).then(a.name.cmp(&b.name)));
        top.truncate(MAX_NODES);
        let top_set: HashSet<String> =
            top.iter().map(|n| n.name.clone()).collect();
        let top_clone = top.clone();
        edge_counts.retain(|(s, t), _| top_set.contains(s) && top_set.contains(t));

        ui.horizontal(|ui| {
            ui.heading("call graph");
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "{} unique span names, showing top {}, {} edges",
                    node_aggs.len(),
                    top_set.len(),
                    edge_counts.len(),
                ))
                .color(egui::Color32::from_gray(160)),
            );
        });

        // Reset relaxation budget when topology changes.
        let mut sig: u64 = top_set.len() as u64;
        for n in &top_set {
            sig = sig.wrapping_mul(31).wrapping_add(hash_str(n));
        }
        for ((a, b), c) in &edge_counts {
            sig = sig.wrapping_mul(31).wrapping_add(hash_str(a));
            sig = sig.wrapping_mul(31).wrapping_add(hash_str(b));
            sig = sig.wrapping_mul(31).wrapping_add(*c);
        }
        if sig != state.last_signature {
            state.last_signature = sig;
            state.frames_left = 80;
        }

        let avail = ui.available_size_before_wrap();
        let h = (avail.y - 20.0).max(300.0);
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(avail.x, h),
            egui::Sense::click(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 20, 24));

        // Layout.
        if state.frames_left > 0 {
            state.frames_left -= 1;
            relax(&top_set, &edge_counts, &mut state.force_pos, &mut state.force_vel);
        }
        let pos: HashMap<String, egui::Pos2> = state
            .force_pos
            .iter()
            .filter(|(n, _)| top_set.contains(*n))
            .map(|(n, (x, y))| {
                (
                    n.clone(),
                    egui::pos2(
                        rect.min.x + 30.0 + x * (rect.width() - 60.0),
                        rect.min.y + 30.0 + y * (rect.height() - 60.0),
                    ),
                )
            })
            .collect();

        // Edges.
        let max_count = edge_counts.values().copied().max().unwrap_or(1) as f32;
        for ((src, tgt), count) in &edge_counts {
            let (Some(s), Some(t)) = (pos.get(src), pos.get(tgt)) else {
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
            let dir = (*t - *s).normalized();
            if dir.length_sq() > 0.0001 {
                let perp = egui::vec2(-dir.y, dir.x);
                let arrow_base = *t - dir * 14.0;
                painter.add(egui::Shape::convex_polygon(
                    vec![*t, arrow_base + perp * 5.0, arrow_base - perp * 5.0],
                    egui::Color32::from_rgb(120, 200, 255),
                    egui::Stroke::NONE,
                ));
            }
        }

        // Nodes.
        let max_total =
            top_clone.iter().map(|n| n.total_ns).max().unwrap_or(1) as f32;
        for n in &top_clone {
            let Some(p) = pos.get(&n.name) else { continue };
            // Size proportional to log(time spent) so a 100x range
            // doesn't crowd out smaller nodes.
            let r = 8.0 + ((n.total_ns as f32 / max_total).max(0.001).ln().abs() * 1.5);
            let r = r.clamp(8.0, 28.0);
            let fill = node_color(&n.name);
            let stroke = if state.selected.as_deref() == Some(n.name.as_str())
            {
                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 240, 120))
            } else {
                egui::Stroke::NONE
            };
            painter.circle(*p, r, fill, stroke);
            let label = format!(
                "{} ({}× / {})",
                truncate(&n.name, 28),
                n.count,
                fmt_dur_short(Duration::from_nanos(n.total_ns))
            );
            painter.text(
                *p + egui::vec2(0.0, r + 6.0),
                egui::Align2::CENTER_TOP,
                label,
                egui::FontId::monospace(10.0),
                egui::Color32::from_gray(220),
            );
            if let Some(pp) = resp.interact_pointer_pos() {
                if resp.clicked() && (*p - pp).length() <= r + 4.0 {
                    state.selected = Some(n.name.clone());
                }
            }
        }
    }
}

fn relax(
    top_set: &HashSet<String>,
    edges: &HashMap<(String, String), u64>,
    positions: &mut HashMap<String, (f32, f32)>,
    velocities: &mut HashMap<String, (f32, f32)>,
) {
    let n = top_set.len() as f32;
    for (i, name) in top_set.iter().enumerate() {
        if !positions.contains_key(name) {
            let theta = (i as f32 / n.max(1.0)) * std::f32::consts::TAU;
            positions.insert(
                name.clone(),
                (0.5 + 0.35 * theta.cos(), 0.5 + 0.35 * theta.sin()),
            );
            velocities.insert(name.clone(), (0.0, 0.0));
        }
    }
    positions.retain(|k, _| top_set.contains(k));
    velocities.retain(|k, _| top_set.contains(k));

    const ITERS: usize = 4;
    const REPULSION: f32 = 0.018;
    const REPULSION_CAP: f32 = 0.5;
    const SPRING_K: f32 = 0.07;
    const SPRING_LEN: f32 = 0.18;
    const DAMPING: f32 = 0.6;
    const STEP: f32 = 0.05;

    let max_count = edges.values().copied().max().unwrap_or(1) as f32;
    for _ in 0..ITERS {
        let mut force: HashMap<String, (f32, f32)> = HashMap::new();
        let names: Vec<String> = positions.keys().cloned().collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let a = positions[&names[i]];
                let b = positions[&names[j]];
                let dx = a.0 - b.0;
                let dy = a.1 - b.1;
                let d2 = (dx * dx + dy * dy).max(1e-4);
                let d = d2.sqrt();
                let f = (REPULSION / d2).min(REPULSION_CAP);
                let fx = f * dx / d;
                let fy = f * dy / d;
                let entry_a = force.entry(names[i].clone()).or_insert((0.0, 0.0));
                entry_a.0 += fx;
                entry_a.1 += fy;
                let entry_b = force.entry(names[j].clone()).or_insert((0.0, 0.0));
                entry_b.0 -= fx;
                entry_b.1 -= fy;
            }
        }
        for ((s, t), count) in edges {
            let (Some(a), Some(b)) = (positions.get(s), positions.get(t)) else {
                continue;
            };
            let dx = b.0 - a.0;
            let dy = b.1 - a.1;
            let d = (dx * dx + dy * dy).sqrt().max(1e-4);
            let strength = SPRING_K * (*count as f32 / max_count + 0.3);
            let f = strength * (d - SPRING_LEN);
            let fx = f * dx / d;
            let fy = f * dy / d;
            let entry_a = force.entry(s.clone()).or_insert((0.0, 0.0));
            entry_a.0 += fx;
            entry_a.1 += fy;
            let entry_b = force.entry(t.clone()).or_insert((0.0, 0.0));
            entry_b.0 -= fx;
            entry_b.1 -= fy;
        }
        for (name, f) in force {
            let v = velocities.entry(name.clone()).or_insert((0.0, 0.0));
            v.0 = (v.0 + f.0 * STEP) * DAMPING;
            v.1 = (v.1 + f.1 * STEP) * DAMPING;
            if let Some(p) = positions.get_mut(&name) {
                p.0 = (p.0 + v.0).clamp(0.05, 0.95);
                p.1 = (p.1 + v.1).clamp(0.05, 0.95);
            }
        }
    }
}

fn node_color(name: &str) -> egui::Color32 {
    let h = hash_str(name) as u32;
    let r = ((h >> 16) & 0xFF) as u8;
    let g = ((h >> 8) & 0xFF) as u8;
    let b = (h & 0xFF) as u8;
    let lift = |c: u8| (c.saturating_add(80)).min(220);
    egui::Color32::from_rgb(lift(r), lift(g), lift(b))
}

fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0;
    for byte in s.bytes() {
        h = h.wrapping_mul(2654435761).wrapping_add(byte as u64);
    }
    h
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}…", &s[..max - 1])
    } else {
        s.to_string()
    }
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
