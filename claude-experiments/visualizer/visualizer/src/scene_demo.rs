use std::collections::HashMap;

use vello::Scene;

use crate::anim::Easing;
use crate::animated::{AnimatedColor, AnimatedPos, AnimatedValue};
use crate::collection::{Collection, EnterAnim, ExitAnim};
use crate::scene::{CircleNode, GroupNode, Node, RectNode, SceneGraph};
use crate::state::State;
use crate::tweakables::Tweakables;

pub struct SceneDemo {
    graph: SceneGraph,
    entries: Collection,
    state: State,
    ops: Vec<&'static str>,
    op_idx: usize,
    bucket_counts: Vec<usize>,
}

const NUM_BUCKETS: usize = 8;
const BUCKET_W: f64 = 70.0;
const BUCKET_H: f64 = 200.0;
const GAP: f64 = 16.0;
const BUCKET_Y: f64 = 280.0;

fn bucket_cx(i: usize) -> f64 {
    let total = NUM_BUCKETS as f64 * BUCKET_W + (NUM_BUCKETS - 1) as f64 * GAP;
    let start = (800.0 - total) / 2.0;
    start + i as f64 * (BUCKET_W + GAP) + BUCKET_W / 2.0
}

fn slot_y(slot: usize) -> f64 {
    BUCKET_Y + BUCKET_H - 26.0 - slot as f64 * 44.0
}

fn hash_key(key: &str) -> usize {
    let mut h: u32 = 5381;
    for b in key.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h as usize % NUM_BUCKETS
}

fn color_for(idx: usize) -> [f32; 4] {
    let colors: &[[f32; 4]] = &[
        [0.35, 0.65, 0.95, 1.0],
        [0.95, 0.45, 0.35, 1.0],
        [0.40, 0.85, 0.50, 1.0],
        [0.95, 0.75, 0.30, 1.0],
        [0.75, 0.45, 0.90, 1.0],
        [0.95, 0.55, 0.70, 1.0],
        [0.40, 0.85, 0.85, 1.0],
        [0.90, 0.65, 0.40, 1.0],
    ];
    colors[idx % colors.len()]
}

impl SceneDemo {
    pub fn new() -> Self {
        let mut graph = SceneGraph::new();

        // Buckets
        let mut buckets_group = GroupNode::new();
        buckets_group.props.id = Some("buckets".into());
        for i in 0..NUM_BUCKETS {
            let cx = bucket_cx(i);
            let cy = BUCKET_Y + BUCKET_H / 2.0;

            let mut bucket = RectNode::new(cx, cy, BUCKET_W, BUCKET_H);
            bucket.corner_radius = AnimatedValue::constant(8.0);
            bucket.fill = AnimatedColor::spring(0.12, 0.14, 0.20, 1.0, 150.0, 12.0);
            bucket.props.id = Some(format!("bucket-{i}"));
            buckets_group.add(Node::Rect(bucket));

            let mut dot = CircleNode::new(cx, BUCKET_Y - 12.0, 3.0);
            dot.fill = AnimatedColor::constant(0.5, 0.5, 0.6, 0.6);
            buckets_group.add(Node::Circle(dot));
        }
        graph.add(Node::Group(buckets_group));

        // Entries collection with enter/exit animations
        let entries = Collection::new("entries")
            .with_enter(EnterAnim {
                scale_from: 0.0,
                opacity_from: 0.0,
                duration: 0.4,
                easing: Easing::BackOut,
            })
            .with_exit(ExitAnim {
                scale_to: 0.0,
                opacity_to: 0.0,
                duration: 0.3,
                easing: Easing::CubicIn,
            });

        // Hint
        let mut hint = CircleNode::new(400.0, 550.0, 8.0);
        hint.fill = AnimatedColor::constant(0.5, 0.7, 1.0, 0.7);
        hint.props.id = Some("hint".into());
        graph.add(Node::Circle(hint));

        let mut state = State::new();
        state.set("num_entries", 0.0);

        Self {
            graph,
            entries,
            state,
            ops: vec!["name", "age", "city", "job", "lang", "os", "ide", "db", "color", "food"],
            op_idx: 0,
            bucket_counts: vec![0; NUM_BUCKETS],
        }
    }

    pub fn on_click(&mut self) {
        if self.op_idx >= self.ops.len() {
            return;
        }

        let key = self.ops[self.op_idx];
        let bucket = hash_key(key);
        let slot = self.bucket_counts[bucket];
        self.bucket_counts[bucket] += 1;
        let color = color_for(self.op_idx);

        let target_x = bucket_cx(bucket);
        let target_y = slot_y(slot);

        // Create entry node — position springs from staging area to bucket
        let mut entry = RectNode::new(0.0, 0.0, 54.0, 36.0);
        entry.props.pos = AnimatedPos::spring(400.0, 60.0, 300.0, 18.0);
        entry.props.pos.set_target(target_x, target_y);
        entry.corner_radius = AnimatedValue::constant(6.0);
        entry.fill = AnimatedColor::constant(color[0], color[1], color[2], color[3]);
        entry.props.id = Some(format!("entry-{}", self.op_idx));

        // Collection handles the enter animation (scale + opacity)
        self.entries.push(Node::Rect(entry));

        // Update state
        self.op_idx += 1;
        self.state.set("num_entries", self.op_idx as f64);

        // Store entry data in state
        let mut item = HashMap::new();
        item.insert("bucket".to_string(), bucket as f64);
        item.insert("slot".to_string(), slot as f64);
        item.insert("target_x".to_string(), target_x);
        item.insert("target_y".to_string(), target_y);
        self.state.list_push("entries", item);

        // Flash the target bucket
        if let Some(Node::Group(buckets)) = self.graph.find_mut("buckets") {
            if let Some(bucket_node) = buckets.children.get_mut(bucket * 2) {
                if let Some(rect) = bucket_node.as_rect_mut() {
                    let flash_color = if slot > 0 {
                        // Collision — flash red briefly
                        [0.8, 0.3, 0.2, 1.0]
                    } else {
                        [0.2, 0.6, 0.3, 1.0]
                    };
                    rect.fill.set_target(flash_color[0], flash_color[1], flash_color[2], flash_color[3]);
                    // Spring will pull it back toward base color, but we need to reset target
                    // after a beat. For now, set it back immediately — spring overshoots = flash.
                    rect.fill.set_target(0.12, 0.14, 0.20, 1.0);
                }
            }
        }

        // Hide hint when done
        if self.op_idx >= self.ops.len() {
            if let Some(hint) = self.graph.find_mut("hint") {
                hint.props_mut().opacity.set_target(0.0);
            }
        }
    }

    pub fn tick(&mut self, dt: f64, tw: &Tweakables) {
        self.graph.tick(dt, tw);
        self.entries.tick(dt, tw);
    }

    pub fn draw(&self, scene: &mut Scene, tw: &Tweakables) {
        self.graph.draw(scene, tw);
        // Draw entries collection on top
        for child in &self.entries.group.children {
            child.draw(scene, tw, 1.0);
        }
    }
}
