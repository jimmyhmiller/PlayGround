//! Per-layer attention visualization.
//!
//! Every attention hall gets 12 panels on its back wall — one per head — laid
//! out in a 6×2 grid. Each panel is an 8×8 grid of cells (`q` along Y, `k`
//! along Z) that protrude forward (+X) from the wall by an amount proportional
//! to the softmax attention weight for that (query, key) pair in that head.
//!
//! Colors are hue-shifted per head so each panel reads as its own "lane". The
//! lower triangle of each panel lights up (causal mask), the upper triangle
//! stays flat against the wall — you can see causality at a glance.
//!
//! Also spawns decorative Q/K/V colored pillars at the +Z entrance of each
//! hall. They don't bind to data yet — they're spatial landmarks for phase 3.

use bevy::prelude::*;
use gpt2_viz::gpt2::MAX_SEQ_LEN;

use crate::config::*;
use crate::labels::SpawnLabel;
use crate::model::GptState;

pub struct AttentionPlugin;

impl Plugin for AttentionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AttentionAssets>().add_systems(
            Startup,
            (
                setup_attention_assets,
                spawn_attention_panels,
                spawn_qkv_pillars,
            )
                .chain(),
        );
        app.add_systems(Update, update_attention_panels);
    }
}

pub const HEADS_PER_LAYER: usize = 12;
pub const HEAD_GRID_COLS: usize = 6;
pub const HEAD_GRID_ROWS: usize = 2;
/// Number of discrete brightness levels per head. More = smoother gradient,
/// more StandardMaterial instances. 16 × 12 = 192 materials total.
pub const BRIGHTNESS_BUCKETS: usize = 16;

const PANEL_CELL_SIZE: f32 = 0.42;
const PANEL_SPACING_Z: f32 = 3.6;
const PANEL_SPACING_Y: f32 = 3.8;
const MAX_CELL_PROTRUDE: f32 = 2.2;
const MIN_CELL_PROTRUDE: f32 = 0.04;

#[derive(Resource, Default)]
pub struct AttentionAssets {
    pub cell_mesh: Handle<Mesh>,
    /// `head_mats[head][bucket]` — dim at bucket 0, glowing at max bucket.
    pub head_mats: Vec<Vec<Handle<StandardMaterial>>>,
    pub panel_backdrop_mat: Handle<StandardMaterial>,
    pub last_version: u64,
}

#[derive(Component, Copy, Clone)]
pub struct AttentionCell {
    pub layer: usize,
    pub head: usize,
    pub q: usize,
    pub k: usize,
    pub base_x: f32,
}

/// Evenly-spaced hue around the color wheel for a given head index,
/// returned as linear-ish (r, g, b) in [0, 1].
fn head_hue_rgb(h: usize) -> (f32, f32, f32) {
    let t = h as f32 / HEADS_PER_LAYER as f32;
    let hprime = t * 6.0;
    let c = 0.95;
    let x = c * (1.0 - ((hprime % 2.0) - 1.0).abs());
    let (r, g, b) = match hprime as usize {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r, g, b)
}

fn head_hue_color(h: usize) -> Color {
    let (r, g, b) = head_hue_rgb(h);
    Color::srgb(r, g, b)
}

fn setup_attention_assets(
    mut assets: ResMut<AttentionAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    assets.cell_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0));

    // Per head, build a brightness ramp. Bucket 0 = near-black (unlit /
    // weight ≈ 0), bucket N-1 = saturated glow. Weight is encoded in
    // emissive intensity so it's legible regardless of viewing angle or
    // scene lighting.
    for h in 0..HEADS_PER_LAYER {
        let (hr, hg, hb) = head_hue_rgb(h);
        let mut per_head = Vec::with_capacity(BRIGHTNESS_BUCKETS);
        for b in 0..BRIGHTNESS_BUCKETS {
            let t = b as f32 / (BRIGHTNESS_BUCKETS - 1) as f32;
            // Base color: dark at bucket 0 so unlit cells recede, but
            // bright enough at the top to reflect passing point lights —
            // the activation pulse needs something to illuminate.
            let base_f = 0.10 + t * 0.45;
            let base = Color::srgb(hr * base_f, hg * base_f, hb * base_f);
            // Emissive: quadratic ramp so low weights stay dim and high
            // weights punch through bloom.
            let em_scale = 0.12 + t * t * 7.5;
            let emissive = LinearRgba::rgb(hr * em_scale, hg * em_scale, hb * em_scale);
            per_head.push(materials.add(StandardMaterial {
                base_color: base,
                emissive,
                perceptual_roughness: 0.5,
                metallic: 0.0,
                ..default()
            }));
        }
        assets.head_mats.push(per_head);
    }

    assets.panel_backdrop_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.02, 0.025, 0.04),
        emissive: LinearRgba::rgb(0.005, 0.008, 0.015),
        perceptual_roughness: 0.95,
        metallic: 0.0,
        ..default()
    });
}

/// World-space center of head `head`'s panel in layer `layer`.
fn head_panel_center(layer: usize, head: usize) -> Vec3 {
    let col = head % HEAD_GRID_COLS;
    let row = head / HEAD_GRID_COLS;
    let panel_z =
        (col as f32 - (HEAD_GRID_COLS as f32 - 1.0) * 0.5) * PANEL_SPACING_Z;
    let panel_y = attn_center_y(layer)
        + (row as f32 - (HEAD_GRID_ROWS as f32 - 1.0) * 0.5) * PANEL_SPACING_Y;
    let panel_x = -BLOCK_WIDTH * 0.5 + 0.5; // just in front of back wall
    Vec3::new(panel_x, panel_y, panel_z)
}

fn spawn_attention_panels(
    mut commands: Commands,
    assets: Res<AttentionAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut labels: EventWriter<SpawnLabel>,
) {
    let grid = MAX_SEQ_LEN;
    let half = (grid as f32 - 1.0) * 0.5;

    for layer in 0..N_LAYER {
        for head in 0..HEADS_PER_LAYER {
            let center = head_panel_center(layer, head);

            // Dark backdrop behind the cells so the panel reads as a framed
            // display even when all cells are near-zero.
            let backdrop_side = (grid as f32) * PANEL_CELL_SIZE + 0.25;
            let backdrop = meshes.add(Cuboid::new(0.08, backdrop_side, backdrop_side));
            commands.spawn(PbrBundle {
                mesh: backdrop,
                material: assets.panel_backdrop_mat.clone(),
                transform: Transform::from_xyz(center.x - 0.12, center.y, center.z),
                ..default()
            });

            // The 8×8 cell grid
            for q in 0..grid {
                for k in 0..grid {
                    let cy = center.y + (q as f32 - half) * PANEL_CELL_SIZE;
                    let cz = center.z + (k as f32 - half) * PANEL_CELL_SIZE;
                    commands.spawn((
                        PbrBundle {
                            mesh: assets.cell_mesh.clone(),
                            material: assets.head_mats[head][0].clone(),
                            transform: Transform::from_xyz(center.x, cy, cz)
                                .with_scale(Vec3::new(
                                    MIN_CELL_PROTRUDE,
                                    PANEL_CELL_SIZE * 0.86,
                                    PANEL_CELL_SIZE * 0.86,
                                )),
                            ..default()
                        },
                        AttentionCell {
                            layer,
                            head,
                            q,
                            k,
                            base_x: center.x,
                        },
                    ));
                }
            }

            // Per-head mini label — only for the first layer so we don't
            // spam 144 labels through the whole tower.
            if layer == 0 {
                labels.send(SpawnLabel {
                    text: format!("H{head}"),
                    position: center + Vec3::new(0.0, backdrop_side * 0.5 + 0.3, 0.0),
                    color: head_hue_color(head),
                    scale: 0.45,
                });
            }
        }
    }
}

fn spawn_qkv_pillars(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut labels: EventWriter<SpawnLabel>,
) {
    // Three colored pillars at each attention hall's +Z (entrance) side.
    let specs = [
        ("Q", Color::srgb(1.00, 0.35, 0.40), -6.0),
        ("K", Color::srgb(0.35, 1.00, 0.50), 0.0),
        ("V", Color::srgb(0.35, 0.55, 1.00), 6.0),
    ];

    for (name, col, x_off) in specs {
        let mat = materials.add(StandardMaterial {
            base_color: col,
            emissive: LinearRgba::from(col) * 2.4,
            perceptual_roughness: 0.3,
            metallic: 0.5,
            ..default()
        });
        let pillar_mesh = meshes.add(Cuboid::new(0.6, 3.8, 0.6));

        for layer in 0..N_LAYER {
            let y = attn_center_y(layer) - 2.0;
            commands.spawn(PbrBundle {
                mesh: pillar_mesh.clone(),
                material: mat.clone(),
                transform: Transform::from_xyz(x_off, y, BLOCK_DEPTH * 0.5 - 2.0),
                ..default()
            });

            // Label only layer 0 to avoid label spam
            if layer == 0 {
                labels.send(SpawnLabel {
                    text: name.to_string(),
                    position: Vec3::new(x_off, y + 2.8, BLOCK_DEPTH * 0.5 - 2.0),
                    color: col,
                    scale: 1.0,
                });
            }
        }
    }
}

fn update_attention_panels(
    state: NonSend<GptState>,
    mut assets: ResMut<AttentionAssets>,
    mut q: Query<(&AttentionCell, &mut Transform, &mut Handle<StandardMaterial>)>,
) {
    if state.version == assets.last_version {
        return;
    }
    assets.last_version = state.version;

    let n_tokens = state.n_tokens();
    let t = MAX_SEQ_LEN;
    let last_bucket = BRIGHTNESS_BUCKETS - 1;

    for (cell, mut tf, mut mat_handle) in q.iter_mut() {
        // Cells outside the real token range sit flat against the wall at
        // bucket 0 (near-black).
        if cell.q >= n_tokens || cell.k >= n_tokens {
            tf.scale.x = MIN_CELL_PROTRUDE;
            tf.translation.x = cell.base_x + MIN_CELL_PROTRUDE * 0.5;
            *mat_handle = assets.head_mats[cell.head][0].clone();
            continue;
        }

        let Some(layer_nodes) = state.model.layout.layers.get(cell.layer) else {
            continue;
        };
        let Some(ni) = layer_nodes.attention else { continue };
        let Some(values) = state.tile_values.get(ni).and_then(|v| v.as_ref()) else {
            continue;
        };

        // Attention tensor is [1, H, T, T], T == MAX_SEQ_LEN, flat row-major.
        let idx = cell.head * t * t + cell.q * t + cell.k;
        let Some(&w) = values.get(idx) else { continue };
        if !w.is_finite() {
            continue;
        }

        let mag = w.clamp(0.0, 1.0);
        // sqrt lifts low-but-non-zero weights so a 0.1 weight is visibly
        // brighter than a 0.0 weight (the softmax over causal masks means
        // many cells hover near 0.05–0.2 and we want those legible).
        let perceptual = mag.sqrt();
        let protrude = MIN_CELL_PROTRUDE + perceptual * (MAX_CELL_PROTRUDE - MIN_CELL_PROTRUDE);
        tf.scale.x = protrude;
        tf.translation.x = cell.base_x + protrude * 0.5;

        let bucket = ((perceptual * BRIGHTNESS_BUCKETS as f32) as usize).min(last_bucket);
        *mat_handle = assets.head_mats[cell.head][bucket].clone();
    }
}
