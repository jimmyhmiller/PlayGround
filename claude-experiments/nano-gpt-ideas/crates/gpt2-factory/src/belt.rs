//! Residual-stream belt — the main spine of the factory.
//!
//! Every "stage" of the residual stream (embedding output, post-attn-L0,
//! post-mlp-L0, …, post-mlp-L11) is rendered as a grid of cells:
//!   X axis → channel (subsampled from n_embd=768 to [`DISPLAY_CHANNELS`])
//!   Z axis → token position in the sequence
//!   Y axis → value magnitude (bar height)
//!   color → signed bucket from a precomputed gradient
//!
//! At startup we spawn all cells with a neutral idle material + minimum
//! height. When the background forward pass delivers data (tracked via
//! `GptState::version`), we update each cell's transform and material.
//!
//! This file also owns the Entry Hall and Exit Hall since they're the
//! endpoints of the belt.

use bevy::prelude::*;

use crate::config::*;
use crate::labels::SpawnLabel;
use crate::model::{GptState, StageKind};

pub struct BeltPlugin;

impl Plugin for BeltPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BeltAssets>()
            .add_systems(
                Startup,
                (
                    setup_belt_assets,
                    spawn_halls,
                    spawn_belt_cells,
                )
                    .chain(),
            )
            .add_systems(Update, update_belt_cells);
    }
}

/// How many channels to display per belt segment (of 768 in GPT-2 small).
pub const DISPLAY_CHANNELS: usize = 48;
/// Number of buckets in the signed color gradient.
pub const N_GRADIENT: usize = 17;

/// Physical extent of the cell grid per stage.
const GRID_X_EXTENT: f32 = 24.0;
const GRID_Z_EXTENT: f32 = 16.0;
/// Cap on how far a cell can grow vertically (in scene units).
const MAX_BAR_HEIGHT: f32 = 2.2;
/// Minimum cell height — what you see before data arrives.
const MIN_BAR_HEIGHT: f32 = 0.08;
/// Value range clamp: values above this in magnitude saturate the gradient.
const VALUE_CLIP: f32 = 6.0;

#[derive(Resource, Default)]
pub struct BeltAssets {
    pub cell_mesh: Handle<Mesh>,
    pub gradient: Vec<Handle<StandardMaterial>>,
    pub idle_mat: Handle<StandardMaterial>,
    pub last_version: u64,
}

#[derive(Component, Copy, Clone)]
pub struct BeltCell {
    pub stage: StageKind,
    pub token: usize,
    pub channel: usize,
    pub base_y: f32,
}

fn setup_belt_assets(
    mut assets: ResMut<BeltAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    assets.cell_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0));

    // Signed diverging gradient: cool-blue (negative) → near-white (zero)
    // → hot-orange (positive). Emissive scales with magnitude so big values
    // glow through bloom.
    for i in 0..N_GRADIENT {
        let t = i as f32 / (N_GRADIENT - 1) as f32; // 0..=1
        let signed = t * 2.0 - 1.0; // -1..=1
        let mag = signed.abs();

        let (r, g, b) = if signed < 0.0 {
            // blue side
            (
                0.15 + (1.0 - mag) * 0.3,
                0.35 + (1.0 - mag) * 0.4,
                0.9,
            )
        } else {
            // warm side
            (
                0.95,
                0.55 + (1.0 - mag) * 0.3,
                0.2 + (1.0 - mag) * 0.5,
            )
        };
        let col = Color::srgb(r, g, b);
        let emis_strength = 0.3 + mag * 2.4;
        assets.gradient.push(materials.add(StandardMaterial {
            base_color: col,
            emissive: LinearRgba::from(col) * emis_strength,
            perceptual_roughness: 0.35,
            metallic: 0.5,
            ..default()
        }));
    }

    assets.idle_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.18, 0.20, 0.28, 1.0),
        emissive: LinearRgba::rgb(0.02, 0.025, 0.05),
        perceptual_roughness: 0.85,
        metallic: 0.15,
        ..default()
    });
}

/// Build the full list of residual-stream stages from bottom to top.
fn all_stages() -> Vec<StageKind> {
    let mut v = Vec::with_capacity(1 + 2 * N_LAYER);
    v.push(StageKind::Embedding);
    for l in 0..N_LAYER {
        v.push(StageKind::PostAttn(l));
        v.push(StageKind::PostMlp(l));
    }
    v
}

/// Physical bottom-Y for a stage's cell grid.
fn stage_base_y(stage: StageKind) -> f32 {
    match stage {
        // Embedding sits in the gap between the ground and the first block.
        StageKind::Embedding => BLOCK_GAP * 0.4,
        StageKind::PostAttn(l) => attn_center_y(l) - MAX_BAR_HEIGHT * 0.5,
        StageKind::PostMlp(l) => mlp_center_y(l) - MAX_BAR_HEIGHT * 0.5,
    }
}

fn spawn_belt_cells(
    mut commands: Commands,
    assets: Res<BeltAssets>,
    state: NonSend<GptState>,
    mut labels: EventWriter<SpawnLabel>,
) {
    let n_tokens = state.n_tokens().max(1);
    let cell_x = GRID_X_EXTENT / DISPLAY_CHANNELS as f32;
    let cell_z = GRID_Z_EXTENT / n_tokens as f32;

    for stage in all_stages() {
        let base_y = stage_base_y(stage);
        for t in 0..n_tokens {
            // Tokens laid out along Z, centered on 0.
            let z = (t as f32 - (n_tokens as f32 - 1.0) * 0.5) * cell_z;
            for c in 0..DISPLAY_CHANNELS {
                let x = (c as f32 - (DISPLAY_CHANNELS as f32 - 1.0) * 0.5) * cell_x;
                commands.spawn((
                    PbrBundle {
                        mesh: assets.cell_mesh.clone(),
                        material: assets.idle_mat.clone(),
                        transform: Transform::from_xyz(
                            x,
                            base_y + MIN_BAR_HEIGHT * 0.5,
                            z,
                        )
                        .with_scale(Vec3::new(
                            cell_x * 0.86,
                            MIN_BAR_HEIGHT,
                            cell_z * 0.86,
                        )),
                        ..default()
                    },
                    BeltCell {
                        stage,
                        token: t,
                        channel: c,
                        base_y,
                    },
                ));
            }
        }

        // One label per stage, floating off to the +X side.
        let (label_text, color) = match stage {
            StageKind::Embedding => ("embedding".to_string(), Color::srgb(0.8, 0.9, 1.0)),
            StageKind::PostAttn(l) => {
                (format!("residual · post-attn L{l}"), Color::srgb(1.0, 0.75, 1.0))
            }
            StageKind::PostMlp(l) => {
                (format!("residual · post-mlp  L{l}"), Color::srgb(0.7, 1.0, 0.95))
            }
        };
        labels.send(SpawnLabel {
            text: label_text,
            position: Vec3::new(
                GRID_X_EXTENT * 0.5 + 1.5,
                base_y + MAX_BAR_HEIGHT * 0.5,
                -GRID_Z_EXTENT * 0.35,
            ),
            color,
            scale: 0.55,
        });
    }
}

fn update_belt_cells(
    state: NonSend<GptState>,
    mut assets: ResMut<BeltAssets>,
    mut q: Query<(&BeltCell, &mut Transform, &mut Handle<StandardMaterial>)>,
) {
    if state.version == assets.last_version {
        return;
    }
    assets.last_version = state.version;

    let n_embd = state.n_embd();
    let n_tokens = state.n_tokens();
    if n_tokens == 0 || n_embd == 0 {
        return;
    }
    let dim_stride = (n_embd / DISPLAY_CHANNELS).max(1);

    for (cell, mut tf, mut mat_handle) in q.iter_mut() {
        let Some(values) = state.get_residual(cell.stage) else {
            continue;
        };
        let dim = (cell.channel * dim_stride).min(n_embd - 1);
        // values are laid out as [1, T, D] row-major — token-major.
        let flat_idx = cell.token * n_embd + dim;
        let Some(&v) = values.get(flat_idx) else { continue };
        if !v.is_finite() {
            continue;
        }

        // Magnitude → bar height
        let mag = (v.abs() / VALUE_CLIP).clamp(0.0, 1.0);
        let bar = MIN_BAR_HEIGHT + mag * (MAX_BAR_HEIGHT - MIN_BAR_HEIGHT);
        tf.scale.y = bar;
        tf.translation.y = cell.base_y + bar * 0.5;

        // Signed value → gradient bucket
        let norm = ((v / VALUE_CLIP) * 0.5 + 0.5).clamp(0.0, 1.0);
        let idx = ((norm * N_GRADIENT as f32) as usize).min(N_GRADIENT - 1);
        *mat_handle = assets.gradient[idx].clone();
    }
}

// ----------------------------------------------------------------------------
// Entry + Exit halls (geometry only — no data binding yet)
// ----------------------------------------------------------------------------

#[derive(Component)]
pub struct EntryHall;

#[derive(Component)]
pub struct ExitHall;

fn spawn_halls(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut labels: EventWriter<SpawnLabel>,
) {
    // Entry Hall — embedding library wall at -Z end
    {
        // Approach ramp up from the ground
        let ramp_len = HALL_LENGTH;
        let ramp_mesh = meshes.add(Cuboid::new(GRID_X_EXTENT + 2.0, 0.3, ramp_len));
        let ramp_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.22, 0.3),
            perceptual_roughness: 0.6,
            metallic: 0.3,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: ramp_mesh,
                material: ramp_mat,
                transform: Transform::from_xyz(
                    0.0,
                    0.3,
                    -(BLOCK_DEPTH * 0.5 + ramp_len * 0.5),
                ),
                ..default()
            },
            EntryHall,
        ));

        // Embedding library wall
        let wall_w = BLOCK_WIDTH * 1.2;
        let wall_h = 18.0;
        let wall_mesh = meshes.add(Cuboid::new(wall_w, wall_h, 0.4));
        let wall_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.12, 0.14, 0.22),
            emissive: LinearRgba::rgb(0.05, 0.08, 0.18),
            perceptual_roughness: 0.7,
            metallic: 0.3,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: wall_mesh,
                material: wall_mat,
                transform: Transform::from_xyz(
                    0.0,
                    wall_h * 0.5,
                    -(BLOCK_DEPTH * 0.5 + HALL_LENGTH),
                ),
                ..default()
            },
            EntryHall,
        ));

        // Row pointers (placeholder visual for the 50k vocab rows)
        let n_strips = 32;
        for i in 0..n_strips {
            let t = i as f32 / (n_strips - 1) as f32;
            let strip_mesh = meshes.add(Cuboid::new(0.2, wall_h * 0.9, 0.08));
            let strip_color = Color::srgb(
                0.3 + 0.5 * (i as f32 * 0.7).sin().abs(),
                0.4 + 0.5 * (i as f32 * 1.3).cos().abs(),
                0.85,
            );
            let strip_mat = materials.add(StandardMaterial {
                base_color: strip_color,
                emissive: LinearRgba::from(strip_color) * 2.5,
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh: strip_mesh,
                    material: strip_mat,
                    transform: Transform::from_xyz(
                        (t - 0.5) * wall_w * 0.9,
                        wall_h * 0.5,
                        -(BLOCK_DEPTH * 0.5 + HALL_LENGTH) + 0.25,
                    ),
                    ..default()
                },
                EntryHall,
            ));
        }

        labels.send(SpawnLabel {
            text: "EMBEDDING LIBRARY".into(),
            position: Vec3::new(0.0, wall_h + 1.5, -(BLOCK_DEPTH * 0.5 + HALL_LENGTH)),
            color: Color::srgb(0.7, 0.85, 1.0),
            scale: 1.4,
        });
        labels.send(SpawnLabel {
            text: "ENTRY".into(),
            position: Vec3::new(0.0, 3.0, -(BLOCK_DEPTH * 0.5 + HALL_LENGTH * 0.5)),
            color: Color::srgb(0.9, 0.95, 0.7),
            scale: 1.1,
        });
    }

    // Exit Hall — unembedding wall above the top of the tower
    {
        let top_y = tower_total_height();
        let wall_h = 18.0;

        let wall_w = BLOCK_WIDTH * 1.2;
        let wall_mesh = meshes.add(Cuboid::new(wall_w, wall_h, 0.4));
        let wall_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.20, 0.14, 0.22),
            emissive: LinearRgba::rgb(0.15, 0.05, 0.15),
            perceptual_roughness: 0.7,
            metallic: 0.3,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: wall_mesh,
                material: wall_mat,
                transform: Transform::from_xyz(
                    0.0,
                    top_y + wall_h * 0.5 + 2.0,
                    BLOCK_DEPTH * 0.5 + HALL_LENGTH * 0.4,
                ),
                ..default()
            },
            ExitHall,
        ));

        let ramp_len = HALL_LENGTH * 0.5;
        let ramp_mesh = meshes.add(Cuboid::new(GRID_X_EXTENT + 2.0, 0.3, ramp_len));
        let ramp_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.22, 0.18, 0.28),
            perceptual_roughness: 0.6,
            metallic: 0.3,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: ramp_mesh,
                material: ramp_mat,
                transform: Transform::from_xyz(
                    0.0,
                    top_y + 1.0,
                    BLOCK_DEPTH * 0.5 + ramp_len * 0.5,
                ),
                ..default()
            },
            ExitHall,
        ));

        labels.send(SpawnLabel {
            text: "UNEMBEDDING".into(),
            position: Vec3::new(
                0.0,
                top_y + wall_h + 3.5,
                BLOCK_DEPTH * 0.5 + HALL_LENGTH * 0.4,
            ),
            color: Color::srgb(1.0, 0.8, 0.9),
            scale: 1.4,
        });
        labels.send(SpawnLabel {
            text: "EXIT — PREDICTED TOKEN".into(),
            position: Vec3::new(
                0.0,
                top_y + 4.0,
                BLOCK_DEPTH * 0.5 + HALL_LENGTH * 0.2,
            ),
            color: Color::srgb(0.95, 0.9, 0.6),
            scale: 1.0,
        });
    }
}
