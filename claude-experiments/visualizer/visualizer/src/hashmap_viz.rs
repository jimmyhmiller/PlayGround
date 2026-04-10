use std::collections::VecDeque;

use vello::kurbo::{Affine, Circle, Point, RoundedRect};
use vello::peniko::{Color, Fill};
use vello::Scene;

use crate::anim::{Easing, Spring, Spring2D, Tween};
use crate::tweakables::Tweakables;

fn bucket_x(index: usize, tw: &Tweakables) -> f64 {
    let bw = tw.get("bucket_width");
    let gap = tw.get("bucket_gap");
    let n = tw.get("num_buckets") as usize;
    let total = n as f64 * bw + (n - 1) as f64 * gap;
    let start = (800.0 - total) / 2.0;
    start + index as f64 * (bw + gap)
}

fn bucket_center_x(index: usize, tw: &Tweakables) -> f64 {
    bucket_x(index, tw) + tw.get("bucket_width") / 2.0
}

fn slot_y(slot: usize, tw: &Tweakables) -> f64 {
    let by = tw.get("bucket_y");
    let bh = tw.get("bucket_height");
    let ih = tw.get("item_height");
    let ig = tw.get("item_gap");
    by + bh - ih - 8.0 - slot as f64 * (ih + ig)
}

fn hash_key(key: &str, num_buckets: usize) -> usize {
    let mut h: u32 = 5381;
    for b in key.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h as usize % num_buckets
}

#[derive(Clone)]
struct Entry {
    key: String,
    color: [f32; 4],
}

struct VisualEntry {
    entry: Entry,
    pos: Spring2D,
    alpha: Spring,
    scale: Tween<f64>,
    bucket_idx: usize,
    slot: usize,
}

struct HashArrow {
    x: Spring,
    alpha: Tween<f64>,
    active: bool,
}

struct BucketFlash {
    bucket: usize,
    tween: Tween<f64>,
    color: [f32; 4],
}

enum AnimPhase {
    Idle,
    ShowKey { timer: f64 },
    Hashing { timer: f64 },
    Checking { timer: f64 },
    Inserting,
    Settling { timer: f64 },
}

struct Op {
    key: String,
}

pub struct HashMapViz {
    buckets: Vec<Vec<Entry>>,
    visual_entries: Vec<VisualEntry>,
    staging: Option<VisualEntry>,
    arrow: HashArrow,
    flashes: Vec<BucketFlash>,
    bucket_glow: Vec<Spring>,
    phase: AnimPhase,
    current_op: Option<Op>,
    current_target_bucket: usize,
    ops: VecDeque<Op>,
    color_idx: usize,
}

fn entry_color(idx: usize) -> [f32; 4] {
    let colors: &[[f32; 4]] = &[
        [0.35, 0.65, 0.95, 1.0],
        [0.95, 0.45, 0.35, 1.0],
        [0.40, 0.85, 0.50, 1.0],
        [0.95, 0.75, 0.30, 1.0],
        [0.75, 0.45, 0.90, 1.0],
        [0.95, 0.55, 0.70, 1.0],
        [0.40, 0.85, 0.85, 1.0],
        [0.90, 0.65, 0.40, 1.0],
        [0.60, 0.80, 0.45, 1.0],
        [0.55, 0.55, 0.85, 1.0],
    ];
    colors[idx % colors.len()]
}

pub fn register_tweakables(tw: &mut Tweakables) {
    // Layout
    tw.register("num_buckets", 8.0, 2.0, 16.0, 1.0, "layout");
    tw.register("bucket_width", 70.0, 30.0, 120.0, 1.0, "layout");
    tw.register("bucket_height", 200.0, 100.0, 400.0, 1.0, "layout");
    tw.register("bucket_gap", 16.0, 0.0, 40.0, 1.0, "layout");
    tw.register("bucket_y", 280.0, 100.0, 450.0, 1.0, "layout");
    tw.register("item_width", 54.0, 20.0, 100.0, 1.0, "layout");
    tw.register("item_height", 36.0, 16.0, 60.0, 1.0, "layout");
    tw.register("item_gap", 4.0, 0.0, 20.0, 1.0, "layout");
    tw.register("staging_y", 80.0, 20.0, 200.0, 1.0, "layout");
    tw.register("item_radius", 6.0, 0.0, 20.0, 1.0, "layout");
    tw.register("bucket_radius", 8.0, 0.0, 20.0, 1.0, "layout");

    // Springs
    tw.register("entry_stiffness", 300.0, 50.0, 800.0, 10.0, "springs");
    tw.register("entry_damping", 18.0, 1.0, 40.0, 0.5, "springs");
    tw.register("arrow_stiffness", 300.0, 50.0, 800.0, 10.0, "springs");
    tw.register("arrow_damping", 18.0, 1.0, 40.0, 0.5, "springs");
    tw.register("glow_stiffness", 200.0, 50.0, 600.0, 10.0, "springs");
    tw.register("glow_damping", 20.0, 1.0, 40.0, 0.5, "springs");
    tw.register("alpha_stiffness", 200.0, 50.0, 600.0, 10.0, "springs");
    tw.register("alpha_damping", 15.0, 1.0, 40.0, 0.5, "springs");

    // Timing
    tw.register("show_key_duration", 0.6, 0.1, 2.0, 0.05, "timing");
    tw.register("hash_duration", 0.5, 0.1, 2.0, 0.05, "timing");
    tw.register("check_duration", 0.3, 0.1, 2.0, 0.05, "timing");
    tw.register("collision_delay", 0.3, 0.0, 1.0, 0.05, "timing");
    tw.register("settle_duration", 0.4, 0.1, 2.0, 0.05, "timing");
    tw.register("scale_duration", 0.4, 0.1, 2.0, 0.05, "timing");
    tw.register("flash_duration", 0.5, 0.1, 2.0, 0.05, "timing");
    tw.register("arrow_fade_duration", 0.3, 0.05, 1.0, 0.05, "timing");

    // Colors (as 0-1 floats)
    tw.register("bg_r", 0.08, 0.0, 1.0, 0.01, "colors");
    tw.register("bg_g", 0.09, 0.0, 1.0, 0.01, "colors");
    tw.register("bg_b", 0.13, 0.0, 1.0, 0.01, "colors");
    tw.register("glow_r", 0.3, 0.0, 1.0, 0.01, "colors");
    tw.register("glow_g", 0.6, 0.0, 1.0, 0.01, "colors");
    tw.register("glow_b", 1.0, 0.0, 1.0, 0.01, "colors");
    tw.register("glow_alpha", 0.3, 0.0, 1.0, 0.01, "colors");
    tw.register("collision_r", 1.0, 0.0, 1.0, 0.01, "colors");
    tw.register("collision_g", 0.3, 0.0, 1.0, 0.01, "colors");
    tw.register("collision_b", 0.2, 0.0, 1.0, 0.01, "colors");
    tw.register("success_r", 0.2, 0.0, 1.0, 0.01, "colors");
    tw.register("success_g", 0.9, 0.0, 1.0, 0.01, "colors");
    tw.register("success_b", 0.4, 0.0, 1.0, 0.01, "colors");
}

impl HashMapViz {
    pub fn new(tw: &Tweakables) -> Self {
        let n = tw.get("num_buckets") as usize;
        let ops: VecDeque<Op> = [
            "name", "age", "city", "job", "lang", "os", "ide", "db", "color", "food",
        ]
        .iter()
        .map(|k| Op { key: k.to_string() })
        .collect();

        Self {
            buckets: vec![Vec::new(); n],
            visual_entries: Vec::new(),
            staging: None,
            arrow: HashArrow {
                x: Spring::with_params(400.0, tw.get("arrow_stiffness"), tw.get("arrow_damping")),
                alpha: Tween::new(0.0, 0.0, 0.3, Easing::Linear),
                active: false,
            },
            flashes: Vec::new(),
            bucket_glow: (0..n)
                .map(|_| Spring::with_params(0.0, tw.get("glow_stiffness"), tw.get("glow_damping")))
                .collect(),
            phase: AnimPhase::Idle,
            current_op: None,
            current_target_bucket: 0,
            ops,
            color_idx: 0,
        }
    }

    pub fn on_click(&mut self, tw: &Tweakables) {
        if matches!(self.phase, AnimPhase::Idle) {
            self.start_next_op(tw);
        }
    }

    fn start_next_op(&mut self, tw: &Tweakables) {
        let Some(op) = self.ops.pop_front() else {
            return;
        };

        let color = entry_color(self.color_idx);
        self.color_idx += 1;

        let stiffness = tw.get("entry_stiffness");
        let damping = tw.get("entry_damping");
        let staging_y = tw.get("staging_y");
        let scale_dur = tw.get("scale_duration");

        let staging = VisualEntry {
            entry: Entry { key: op.key.clone(), color },
            pos: Spring2D::with_params(400.0, staging_y, stiffness, damping),
            alpha: Spring::with_params(0.0, tw.get("alpha_stiffness"), tw.get("alpha_damping")),
            scale: Tween::new(0.0, 1.0, scale_dur, Easing::BackOut),
            bucket_idx: 0,
            slot: 0,
        };

        self.staging = Some(staging);
        self.current_op = Some(op);
        self.phase = AnimPhase::ShowKey { timer: 0.0 };
    }

    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        let entry_stiffness = tw.get("entry_stiffness");
        let entry_damping = tw.get("entry_damping");

        // Live-update spring params on all entries
        for ve in &mut self.visual_entries {
            ve.pos.x.stiffness = entry_stiffness;
            ve.pos.x.damping = entry_damping;
            ve.pos.y.stiffness = entry_stiffness;
            ve.pos.y.damping = entry_damping;
            ve.pos.tick(dt);
            ve.alpha.tick(dt);
            ve.scale.tick(dt);

            // Recompute target positions from current tweakable layout
            let tx = bucket_center_x(ve.bucket_idx, tw);
            let ty = slot_y(ve.slot, tw);
            ve.pos.set_target(tx, ty);
        }

        if let Some(s) = &mut self.staging {
            s.pos.x.stiffness = entry_stiffness;
            s.pos.x.damping = entry_damping;
            s.pos.y.stiffness = entry_stiffness;
            s.pos.y.damping = entry_damping;
            s.pos.tick(dt);
            s.alpha.tick(dt);
            s.scale.tick(dt);
        }

        self.arrow.x.stiffness = tw.get("arrow_stiffness");
        self.arrow.x.damping = tw.get("arrow_damping");
        self.arrow.x.tick(dt);
        self.arrow.alpha.tick(dt);

        for f in &mut self.flashes {
            f.tween.tick(dt);
        }
        self.flashes.retain(|f| !f.tween.done());

        let glow_stiffness = tw.get("glow_stiffness");
        let glow_damping = tw.get("glow_damping");
        for g in &mut self.bucket_glow {
            g.stiffness = glow_stiffness;
            g.damping = glow_damping;
            g.tick(dt);
        }

        // State machine
        let num_buckets = tw.get("num_buckets") as usize;

        match &mut self.phase {
            AnimPhase::Idle => {}
            AnimPhase::ShowKey { timer } => {
                *timer += dt;
                if let Some(s) = &mut self.staging {
                    s.alpha.set_target(1.0);
                }
                if *timer > tw.get("show_key_duration") {
                    if let Some(op) = &self.current_op {
                        let target = hash_key(&op.key, num_buckets);
                        self.current_target_bucket = target;
                        self.arrow.x.set_target(bucket_center_x(target, tw));
                        self.arrow.alpha = Tween::new(0.0, 1.0, tw.get("arrow_fade_duration"), Easing::CubicOut);
                        self.arrow.active = true;
                    }
                    self.phase = AnimPhase::Hashing { timer: 0.0 };
                }
            }
            AnimPhase::Hashing { timer } => {
                *timer += dt;
                if *timer > tw.get("hash_duration") {
                    let idx = self.current_target_bucket;
                    if idx < self.bucket_glow.len() {
                        self.bucket_glow[idx].set_target(1.0);
                    }
                    self.phase = AnimPhase::Checking { timer: 0.0 };
                }
            }
            AnimPhase::Checking { timer } => {
                *timer += dt;
                if *timer > tw.get("check_duration") {
                    let idx = self.current_target_bucket;
                    let occupied = if idx < self.buckets.len() { self.buckets[idx].len() } else { 0 };
                    let flash_dur = tw.get("flash_duration");

                    if occupied > 0 && *timer < tw.get("check_duration") + tw.get("collision_delay") + 0.01 {
                        // Collision flash
                        self.flashes.push(BucketFlash {
                            bucket: idx,
                            tween: Tween::new(1.0, 0.0, flash_dur, Easing::CubicOut),
                            color: [
                                tw.get("collision_r") as f32,
                                tw.get("collision_g") as f32,
                                tw.get("collision_b") as f32,
                                0.6,
                            ],
                        });
                        *timer = tw.get("check_duration") - tw.get("collision_delay");
                    } else {
                        // Insert
                        let slot = occupied;
                        let target_x = bucket_center_x(idx, tw);
                        let target_y = slot_y(slot, tw);

                        if let Some(mut s) = self.staging.take() {
                            s.pos.set_target(target_x, target_y);
                            s.bucket_idx = idx;
                            s.slot = slot;

                            let entry = s.entry.clone();
                            if idx < self.buckets.len() {
                                self.buckets[idx].push(entry);
                            }
                            self.visual_entries.push(s);
                        }

                        self.flashes.push(BucketFlash {
                            bucket: idx,
                            tween: Tween::new(1.0, 0.0, flash_dur * 0.8, Easing::CubicOut),
                            color: [
                                tw.get("success_r") as f32,
                                tw.get("success_g") as f32,
                                tw.get("success_b") as f32,
                                0.5,
                            ],
                        });

                        self.phase = AnimPhase::Inserting;
                    }
                }
            }
            AnimPhase::Inserting => {
                let settled = self.visual_entries.last()
                    .map(|ve| ve.pos.x.settled() && ve.pos.y.settled())
                    .unwrap_or(true);
                if settled {
                    self.arrow.alpha = Tween::new(1.0, 0.0, tw.get("arrow_fade_duration"), Easing::CubicIn);
                    self.arrow.active = false;
                    for g in &mut self.bucket_glow {
                        g.set_target(0.0);
                    }
                    self.phase = AnimPhase::Settling { timer: 0.0 };
                }
            }
            AnimPhase::Settling { timer } => {
                *timer += dt;
                if *timer > tw.get("settle_duration") {
                    self.current_op = None;
                    self.phase = AnimPhase::Idle;
                }
            }
        }
    }

    pub fn draw(&self, scene: &mut Scene, tw: &Tweakables) {
        let num_buckets = tw.get("num_buckets") as usize;
        let bw = tw.get("bucket_width");
        let bh = tw.get("bucket_height");
        let by = tw.get("bucket_y");
        let br = tw.get("bucket_radius");

        // Bucket index dots
        for i in 0..num_buckets {
            let cx = bucket_center_x(i, tw);
            let cy = by - 12.0;
            let circle = Circle::new(Point::new(cx, cy), 3.0);
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([0.5, 0.5, 0.6, 0.6]), None, &circle);
        }

        // Buckets
        for i in 0..num_buckets {
            let bx = bucket_x(i, tw);

            // Glow
            if i < self.bucket_glow.len() {
                let glow = self.bucket_glow[i].position;
                if glow > 0.01 {
                    let gr = tw.get("glow_r") as f32;
                    let gg = tw.get("glow_g") as f32;
                    let gb = tw.get("glow_b") as f32;
                    let ga = tw.get("glow_alpha") as f32 * glow as f32;
                    let glow_rect = RoundedRect::new(
                        bx - 4.0, by - 4.0,
                        bx + bw + 4.0, by + bh + 4.0,
                        br + 4.0,
                    );
                    scene.fill(Fill::NonZero, Affine::IDENTITY,
                        Color::new([gr, gg, gb, ga]), None, &glow_rect);
                }
            }

            // Flashes
            for flash in &self.flashes {
                if flash.bucket == i {
                    let intensity = flash.tween.value();
                    if intensity > 0.01 {
                        let flash_rect = RoundedRect::new(
                            bx - 2.0, by - 2.0,
                            bx + bw + 2.0, by + bh + 2.0,
                            br + 2.0,
                        );
                        scene.fill(Fill::NonZero, Affine::IDENTITY,
                            Color::new([
                                flash.color[0], flash.color[1], flash.color[2],
                                flash.color[3] * intensity as f32,
                            ]), None, &flash_rect);
                    }
                }
            }

            // Bucket body
            let rect = RoundedRect::new(bx, by, bx + bw, by + bh, br);
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([0.12, 0.13, 0.18, 1.0]), None, &rect);

            let border = RoundedRect::new(bx + 1.0, by + 1.0, bx + bw - 1.0, by + bh - 1.0, (br - 1.0).max(0.0));
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([0.2, 0.22, 0.28, 1.0]), None, &border);

            let bgr = tw.get("bg_r") as f32;
            let bgg = tw.get("bg_g") as f32;
            let bgb = tw.get("bg_b") as f32;
            let inner = RoundedRect::new(bx + 2.0, by + 2.0, bx + bw - 2.0, by + bh - 2.0, (br - 2.0).max(0.0));
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([bgr, bgg, bgb, 1.0]), None, &inner);
        }

        // Hash arrow
        if self.arrow.active || self.arrow.alpha.value() > 0.01 {
            let ax = self.arrow.x.position;
            let ay = by - 30.0;
            let alpha = self.arrow.alpha.value() as f32;

            let tri_size = 10.0;
            let tri = vello::kurbo::BezPath::from_vec(vec![
                vello::kurbo::PathEl::MoveTo(Point::new(ax, ay + tri_size)),
                vello::kurbo::PathEl::LineTo(Point::new(ax - tri_size, ay - tri_size)),
                vello::kurbo::PathEl::LineTo(Point::new(ax + tri_size, ay - tri_size)),
                vello::kurbo::PathEl::ClosePath,
            ]);
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([1.0, 0.8, 0.2, alpha]), None, &tri);

            let stem = RoundedRect::new(ax - 2.0, ay - tri_size - 15.0, ax + 2.0, ay - tri_size, 1.0);
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([1.0, 0.8, 0.2, alpha]), None, &stem);
        }

        // Staging item
        if let Some(s) = &self.staging {
            self.draw_entry(scene, s, tw);
        }

        // Placed entries
        for ve in &self.visual_entries {
            self.draw_entry(scene, ve, tw);
        }

        // Click hint
        if matches!(self.phase, AnimPhase::Idle) && !self.ops.is_empty() {
            let remaining = self.ops.len();
            let pulse = (web_time::Instant::now().elapsed().as_secs_f64() * 3.0).sin() * 0.3 + 0.7;
            let circle = Circle::new(Point::new(400.0, 550.0), 8.0);
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([0.5, 0.7, 1.0, pulse as f32]), None, &circle);

            for i in 0..remaining.min(10) {
                let dx = (i as f64 - remaining.min(10) as f64 / 2.0) * 12.0;
                let dot = Circle::new(Point::new(400.0 + dx, 570.0), 2.5);
                scene.fill(Fill::NonZero, Affine::IDENTITY,
                    Color::new([0.4, 0.4, 0.5, 0.6]), None, &dot);
            }
        }

        // Done indicator
        if matches!(self.phase, AnimPhase::Idle) && self.ops.is_empty() && !self.visual_entries.is_empty() {
            let circle = Circle::new(Point::new(400.0, 550.0), 10.0);
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([0.3, 0.9, 0.4, 0.8]), None, &circle);
        }
    }

    fn draw_entry(&self, scene: &mut Scene, ve: &VisualEntry, tw: &Tweakables) {
        let pos = ve.pos.position();
        let alpha = ve.alpha.position.clamp(0.0, 1.0) as f32;
        let scale = ve.scale.value();

        if alpha < 0.01 || scale < 0.01 {
            return;
        }

        let iw = tw.get("item_width");
        let ih = tw.get("item_height");
        let ir = tw.get("item_radius");

        let w = iw * scale;
        let h = ih * scale;
        let x = pos[0] - w / 2.0;
        let y = pos[1] - h / 2.0;

        let rect = RoundedRect::new(x, y, x + w, y + h, ir * scale);
        scene.fill(Fill::NonZero, Affine::IDENTITY,
            Color::new([
                ve.entry.color[0], ve.entry.color[1], ve.entry.color[2], alpha,
            ]), None, &rect);

        // Key stripe
        let stripe_h = h * 0.45;
        let stripe = RoundedRect::new(
            x + 3.0 * scale, y + 2.0 * scale,
            x + w - 3.0 * scale, y + stripe_h,
            3.0 * scale,
        );
        scene.fill(Fill::NonZero, Affine::IDENTITY,
            Color::new([1.0, 1.0, 1.0, 0.2 * alpha]), None, &stripe);

        // Corner dot
        let dot = Circle::new(Point::new(x + w - 6.0 * scale, y + 6.0 * scale), 3.0 * scale);
        scene.fill(Fill::NonZero, Affine::IDENTITY,
            Color::new([1.0, 1.0, 1.0, 0.5 * alpha]), None, &dot);

        // Key-length bars
        let bars = (ve.entry.key.len() % 3) + 1;
        for b in 0..bars {
            let bx = x + 4.0 * scale + b as f64 * 8.0 * scale;
            let by = y + stripe_h + 4.0 * scale;
            let bar = RoundedRect::new(bx, by, bx + 5.0 * scale, by + 8.0 * scale, 1.5 * scale);
            scene.fill(Fill::NonZero, Affine::IDENTITY,
                Color::new([1.0, 1.0, 1.0, 0.3 * alpha]), None, &bar);
        }
    }
}
