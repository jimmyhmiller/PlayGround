//! Dot Product Station — a physical machine that computes Q·K in front of
//! you. Placed inside layer 0's attention hall.
//!
//! Animation cycle (~2.5 seconds per key):
//!   phase 0.00 – 0.20  →  K bars snap in for the current key token
//!   phase 0.20 – 0.55  →  Product bars grow (per-dimension Q·K)
//!   phase 0.55 – 0.85  →  Accumulator bar sums products to the scaled score
//!   phase 0.85 – 1.00  →  Score readout snaps to the final value, pause
//!
//! Q bars are always visible (they depend only on the query token). K, product,
//! accumulator, and score change each cycle as `current_k` advances.
//!
//! Data is extracted from the `NodeRole::QKV` tile which has shape [1, T, 3D]
//! laid out per token as [Q_dims | K_dims | V_dims]. For head `h`:
//!     q = qkv[t*3D + 0*D + h*d_head .. + d_head]
//!     k = qkv[t*3D + 1*D + h*d_head .. + d_head]
//!
//! The 64-dim vectors are subsampled to 16 visible bars for readability; the
//! displayed score still uses the full 64-dim dot product so the number is
//! accurate, only the bar-level visualization is subsampled.

use bevy::prelude::*;

use crate::labels::SpawnLabel;
use crate::model::GptState;

pub struct DotStationPlugin;

impl Plugin for DotStationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DotStation::default())
            .add_systems(
                Startup,
                (spawn_station, spawn_status_ui, spawn_teaching_panel),
            )
            .add_systems(
                Update,
                (
                    station_keys,
                    advance_station,
                    update_status_ui,
                    update_teaching_panel,
                ),
            );
    }
}

#[derive(Resource)]
pub struct DotStation {
    pub layer: usize,
    pub head: usize,
    pub query_token: usize,
    pub current_k: usize,
    pub cycle_time: f32,
    pub cycle_length: f32,
    pub enabled: bool,
    /// Last computed full (64-dim) scaled dot product. Displayed in the UI.
    pub last_score: f32,
    pub last_raw: f32,
}

impl Default for DotStation {
    fn default() -> Self {
        Self {
            layer: 0,
            head: 0,
            query_token: 0,
            current_k: 0,
            cycle_time: 0.0,
            cycle_length: 4.0,
            enabled: true,
            last_score: 0.0,
            last_raw: 0.0,
        }
    }
}

/// Remove the BPE prefix markers so token strings are readable in text.
fn clean_token(s: &str) -> String {
    s.replace('Ġ', "·").replace('Ċ', "↵")
}

// ---------------- Geometry ----------------
const STATION_CX: f32 = 0.0;
const STATION_CZ: f32 = 0.0;

const BAR_COUNT: usize = 16;
const BAR_SPACING: f32 = 0.85;
const BAR_THICKNESS: f32 = 0.55;

/// Y baseline of each row.
const Q_ROW_Y: f32 = 6.8;
const K_ROW_Y: f32 = 5.2;
const PROD_ROW_Y: f32 = 3.4;

/// X offset for the accumulator and score bars (right of the main rows).
const ACCUM_X: f32 = 9.0;
const SCORE_X: f32 = 11.4;

const Q_SCALE: f32 = 1.8;
const K_SCALE: f32 = 1.8;
const PROD_SCALE: f32 = 4.0;
const ACCUM_VIS_SCALE: f32 = 0.35;
const MIN_BAR_H: f32 = 0.04;
const MAX_BAR_H: f32 = 1.6;
const MAX_ACCUM_H: f32 = 5.5;

#[derive(Component)]
struct QBar(usize);
#[derive(Component)]
struct KBar(usize);
#[derive(Component)]
struct ProductBar(usize);
#[derive(Component)]
struct AccumBar;
#[derive(Component)]
struct ScoreBar;

#[derive(Resource)]
struct StationMats {
    _q: Handle<StandardMaterial>,
    _k: Handle<StandardMaterial>,
    prod_pos: Handle<StandardMaterial>,
    prod_neg: Handle<StandardMaterial>,
    accum_pos: Handle<StandardMaterial>,
    accum_neg: Handle<StandardMaterial>,
}

fn spawn_station(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut labels: EventWriter<SpawnLabel>,
) {
    let bar_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0));

    let q_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.5, 1.0),
        emissive: LinearRgba::rgb(0.9, 1.8, 4.0),
        perceptual_roughness: 0.4,
        metallic: 0.0,
        ..default()
    });
    let k_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 1.0, 0.5),
        emissive: LinearRgba::rgb(0.7, 3.2, 1.6),
        perceptual_roughness: 0.4,
        metallic: 0.0,
        ..default()
    });
    let prod_pos = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.82, 0.25),
        emissive: LinearRgba::rgb(3.2, 2.5, 0.55),
        perceptual_roughness: 0.35,
        metallic: 0.0,
        ..default()
    });
    let prod_neg = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.28, 0.45),
        emissive: LinearRgba::rgb(3.5, 0.7, 1.1),
        perceptual_roughness: 0.35,
        metallic: 0.0,
        ..default()
    });
    let accum_pos = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.95, 1.0),
        emissive: LinearRgba::rgb(3.2, 3.4, 4.5),
        perceptual_roughness: 0.35,
        metallic: 0.0,
        ..default()
    });
    let accum_neg = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.5, 0.7),
        emissive: LinearRgba::rgb(3.8, 1.1, 1.6),
        perceptual_roughness: 0.35,
        metallic: 0.0,
        ..default()
    });
    let housing_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.09, 0.10, 0.14),
        perceptual_roughness: 0.85,
        metallic: 0.05,
        ..default()
    });

    // Housing base slab the machine sits on
    let housing_w = BAR_COUNT as f32 * BAR_SPACING + 6.0;
    let housing_d = 3.6;
    let housing_mesh = meshes.add(Cuboid::new(housing_w, 0.4, housing_d));
    commands.spawn(PbrBundle {
        mesh: housing_mesh,
        material: housing_mat.clone(),
        transform: Transform::from_xyz(STATION_CX + 2.4, 2.4, STATION_CZ),
        ..default()
    });

    // Row backdrops (thin plates behind each bar row so they read as shelves)
    for (name_y, _) in [(Q_ROW_Y, "Q"), (K_ROW_Y, "K"), (PROD_ROW_Y, "P")] {
        let backdrop = meshes.add(Cuboid::new(housing_w - 8.0, 0.15, 1.2));
        commands.spawn(PbrBundle {
            mesh: backdrop,
            material: housing_mat.clone(),
            transform: Transform::from_xyz(STATION_CX - 1.0, name_y - 0.15, STATION_CZ - 0.7),
            ..default()
        });
    }

    // Bars: Q row, K row, Product row — 16 columns
    let half = (BAR_COUNT as f32 - 1.0) * 0.5;
    for i in 0..BAR_COUNT {
        let x = STATION_CX + (i as f32 - half) * BAR_SPACING - 1.0;

        commands.spawn((
            PbrBundle {
                mesh: bar_mesh.clone(),
                material: q_mat.clone(),
                transform: Transform::from_xyz(x, Q_ROW_Y, STATION_CZ).with_scale(Vec3::new(
                    BAR_THICKNESS,
                    MIN_BAR_H,
                    BAR_THICKNESS,
                )),
                ..default()
            },
            QBar(i),
        ));

        commands.spawn((
            PbrBundle {
                mesh: bar_mesh.clone(),
                material: k_mat.clone(),
                transform: Transform::from_xyz(x, K_ROW_Y, STATION_CZ).with_scale(Vec3::new(
                    BAR_THICKNESS,
                    MIN_BAR_H,
                    BAR_THICKNESS,
                )),
                ..default()
            },
            KBar(i),
        ));

        commands.spawn((
            PbrBundle {
                mesh: bar_mesh.clone(),
                material: prod_pos.clone(),
                transform: Transform::from_xyz(x, PROD_ROW_Y, STATION_CZ).with_scale(Vec3::new(
                    BAR_THICKNESS,
                    MIN_BAR_H,
                    BAR_THICKNESS,
                )),
                ..default()
            },
            ProductBar(i),
        ));
    }

    // Accumulator (vertical bar right of the product row)
    commands.spawn((
        PbrBundle {
            mesh: bar_mesh.clone(),
            material: accum_pos.clone(),
            transform: Transform::from_xyz(STATION_CX + ACCUM_X, PROD_ROW_Y, STATION_CZ)
                .with_scale(Vec3::new(1.1, MIN_BAR_H, 1.1)),
            ..default()
        },
        AccumBar,
    ));

    // Score readout (fatter bar further right)
    commands.spawn((
        PbrBundle {
            mesh: bar_mesh.clone(),
            material: accum_pos.clone(),
            transform: Transform::from_xyz(STATION_CX + SCORE_X, PROD_ROW_Y, STATION_CZ)
                .with_scale(Vec3::new(1.4, MIN_BAR_H, 1.4)),
            ..default()
        },
        ScoreBar,
    ));

    // Labels
    let left_label_x = STATION_CX + (-half - 1.5) * BAR_SPACING - 1.0;
    labels.send(SpawnLabel {
        text: "Q".into(),
        position: Vec3::new(left_label_x, Q_ROW_Y + 0.3, STATION_CZ),
        color: Color::srgb(0.5, 0.72, 1.0),
        scale: 1.1,
    });
    labels.send(SpawnLabel {
        text: "K".into(),
        position: Vec3::new(left_label_x, K_ROW_Y + 0.3, STATION_CZ),
        color: Color::srgb(0.5, 1.0, 0.65),
        scale: 1.1,
    });
    labels.send(SpawnLabel {
        text: "Q × K".into(),
        position: Vec3::new(left_label_x, PROD_ROW_Y + 0.3, STATION_CZ),
        color: Color::srgb(1.0, 0.85, 0.4),
        scale: 0.95,
    });
    labels.send(SpawnLabel {
        text: "Σ / √d".into(),
        position: Vec3::new(STATION_CX + ACCUM_X, 8.2, STATION_CZ),
        color: Color::srgb(0.85, 0.9, 1.0),
        scale: 0.95,
    });
    labels.send(SpawnLabel {
        text: "SCORE".into(),
        position: Vec3::new(STATION_CX + SCORE_X, 8.2, STATION_CZ),
        color: Color::srgb(1.0, 0.95, 0.5),
        scale: 0.95,
    });
    labels.send(SpawnLabel {
        text: "DOT PRODUCT · L0 H0".into(),
        position: Vec3::new(STATION_CX + 2.0, 9.2, STATION_CZ),
        color: Color::srgb(0.95, 0.95, 1.0),
        scale: 1.3,
    });

    commands.insert_resource(StationMats {
        _q: q_mat,
        _k: k_mat,
        prod_pos,
        prod_neg,
        accum_pos,
        accum_neg,
    });
}

/// Pull head-sliced Q and K vectors for the given (layer, head, q_tok, k_tok)
/// out of the QKV tile. Returns None if the tile isn't computed yet.
fn get_qk_vectors(
    state: &GptState,
    layer: usize,
    head: usize,
    q_token: usize,
    k_token: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let qkv_ni = state.model.layout.layers.get(layer)?.qkv?;
    let qkv = state.tile_values.get(qkv_ni)?.as_ref()?;
    let d = state.model.config.n_embd;
    let n_head = state.model.config.n_head;
    if d % n_head != 0 {
        return None;
    }
    let d_head = d / n_head;
    let t_stride = 3 * d;
    let q_off = q_token * t_stride + head * d_head;
    let k_off = q_token.saturating_mul(0) + k_token * t_stride + d + head * d_head;
    if q_off + d_head > qkv.len() || k_off + d_head > qkv.len() {
        return None;
    }
    Some((
        qkv[q_off..q_off + d_head].to_vec(),
        qkv[k_off..k_off + d_head].to_vec(),
    ))
}

fn subsample(v: &[f32], count: usize) -> Vec<f32> {
    if v.is_empty() || count == 0 {
        return vec![];
    }
    let stride = (v.len() / count).max(1);
    (0..count).map(|i| v[(i * stride).min(v.len() - 1)]).collect()
}

fn shape_bar(tf: &mut Transform, base_y: f32, height: f32) {
    let h = height.max(MIN_BAR_H);
    tf.scale.y = h;
    tf.translation.y = base_y + h * 0.5;
}

#[allow(clippy::too_many_arguments)]
fn advance_station(
    time: Res<Time>,
    mut station: ResMut<DotStation>,
    gpt: NonSend<GptState>,
    mats: Res<StationMats>,
    mut sets: ParamSet<(
        Query<(&QBar, &mut Transform)>,
        Query<(&KBar, &mut Transform)>,
        Query<(&ProductBar, &mut Transform, &mut Handle<StandardMaterial>)>,
        Query<(&mut Transform, &mut Handle<StandardMaterial>), With<AccumBar>>,
        Query<(&mut Transform, &mut Handle<StandardMaterial>), With<ScoreBar>>,
    )>,
) {
    if !station.enabled {
        return;
    }
    let n_tokens = gpt.n_tokens();
    if n_tokens == 0 {
        return;
    }
    // Query token defaults to the last prompt token (most interesting:
    // this is the one the model is using to predict the next token).
    let q_tok = (n_tokens - 1).min(station.query_token.max(0));
    station.query_token = q_tok;

    station.cycle_time += time.delta_seconds();
    if station.cycle_time > station.cycle_length {
        station.cycle_time = 0.0;
        station.current_k = (station.current_k + 1) % n_tokens;
    }
    let phase = (station.cycle_time / station.cycle_length).clamp(0.0, 1.0);
    let k_tok = station.current_k;

    let Some((q_vec, k_vec)) =
        get_qk_vectors(&gpt, station.layer, station.head, q_tok, k_tok)
    else {
        return;
    };

    let q_sub = subsample(&q_vec, BAR_COUNT);
    let k_sub = subsample(&k_vec, BAR_COUNT);

    // Q bars — always visible (they depend only on the query token).
    for (bar, mut tf) in sets.p0().iter_mut() {
        let v = q_sub.get(bar.0).copied().unwrap_or(0.0);
        let h = (v.abs() * Q_SCALE).min(MAX_BAR_H);
        shape_bar(&mut tf, Q_ROW_Y, h);
    }

    // K bars — fade in during the first 20% of the cycle.
    let k_reveal = (phase / 0.20).clamp(0.0, 1.0);
    for (bar, mut tf) in sets.p1().iter_mut() {
        let v = k_sub.get(bar.0).copied().unwrap_or(0.0);
        let h_target = (v.abs() * K_SCALE).min(MAX_BAR_H);
        shape_bar(&mut tf, K_ROW_Y, h_target * k_reveal);
    }

    // Product bars — reveal between phase 0.20 and 0.55.
    let p_reveal = ((phase - 0.20) / 0.35).clamp(0.0, 1.0);
    let products_sub: Vec<f32> = q_sub
        .iter()
        .zip(k_sub.iter())
        .map(|(q, k)| q * k)
        .collect();
    for (bar, mut tf, mut mat_handle) in sets.p2().iter_mut() {
        let v = products_sub.get(bar.0).copied().unwrap_or(0.0);
        let h_target = (v.abs() * PROD_SCALE).min(MAX_BAR_H);
        shape_bar(&mut tf, PROD_ROW_Y, h_target * p_reveal);
        *mat_handle = if v >= 0.0 {
            mats.prod_pos.clone()
        } else {
            mats.prod_neg.clone()
        };
    }

    // Accumulator — reveals between phase 0.55 and 0.85, using the TRUE
    // full-dimensional scaled dot product (not the subsampled one). The
    // bars above are a visualization but the number is exact.
    let a_reveal = ((phase - 0.55) / 0.30).clamp(0.0, 1.0);
    let raw_dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(q, k)| q * k).sum();
    let scaled = raw_dot / (q_vec.len() as f32).sqrt();
    station.last_raw = raw_dot;
    station.last_score = scaled;

    let accum_h_target = (scaled.abs() * ACCUM_VIS_SCALE).min(MAX_ACCUM_H);
    for (mut tf, mut mat_handle) in sets.p3().iter_mut() {
        shape_bar(&mut tf, PROD_ROW_Y, accum_h_target * a_reveal);
        *mat_handle = if scaled >= 0.0 {
            mats.accum_pos.clone()
        } else {
            mats.accum_neg.clone()
        };
    }

    // Score readout — snaps in at phase 0.85.
    let s_reveal = ((phase - 0.85) / 0.15).clamp(0.0, 1.0);
    for (mut tf, mut mat_handle) in sets.p4().iter_mut() {
        shape_bar(&mut tf, PROD_ROW_Y, accum_h_target * s_reveal);
        *mat_handle = if scaled >= 0.0 {
            mats.accum_pos.clone()
        } else {
            mats.accum_neg.clone()
        };
    }
}

// ---------------- Status UI ----------------

#[derive(Component)]
struct StationStatusText;

fn spawn_status_ui(mut commands: Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                bottom: Val::Px(20.0),
                left: Val::Percent(22.0),
                right: Val::Percent(22.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                align_items: AlignItems::Center,
                ..default()
            },
            background_color: BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            ..default()
        })
        .with_children(|panel| {
            panel.spawn(TextBundle::from_section(
                "DOT PRODUCT STATION",
                TextStyle {
                    font_size: 18.0,
                    color: Color::srgb(0.95, 0.95, 1.0),
                    ..default()
                },
            ));
            panel.spawn((
                TextBundle::from_section(
                    "…",
                    TextStyle {
                        font_size: 16.0,
                        color: Color::srgb(0.85, 0.92, 1.0),
                        ..default()
                    },
                ),
                StationStatusText,
            ));
        });
}

fn update_status_ui(
    station: Res<DotStation>,
    mut q: Query<&mut Text, With<StationStatusText>>,
) {
    let Ok(mut text) = q.get_single_mut() else { return };
    let state_str = if station.enabled { "running" } else { "PAUSED" };
    text.sections[0].value = format!(
        "L{} · H{} · q={} · k={}   raw Q·K = {:+7.2}   /√d = {:+6.2}   [{state_str}]",
        station.layer,
        station.head,
        station.query_token,
        station.current_k,
        station.last_raw,
        station.last_score,
    );
}

// ---------------- Teaching panel (the thing that actually explains) --

#[derive(Component)]
struct TeachingPanelText;

fn spawn_teaching_panel(mut commands: Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(20.0),
                left: Val::Px(20.0),
                width: Val::Px(520.0),
                padding: UiRect::all(Val::Px(16.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            background_color: BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.62)),
            ..default()
        })
        .with_children(|panel| {
            panel.spawn((
                TextBundle::from_section(
                    "",
                    TextStyle {
                        font_size: 15.0,
                        color: Color::srgb(0.88, 0.92, 1.0),
                        ..default()
                    },
                ),
                TeachingPanelText,
            ));
        });
}

fn update_teaching_panel(
    station: Res<DotStation>,
    gpt: NonSend<GptState>,
    mut q: Query<&mut Text, With<TeachingPanelText>>,
) {
    let Ok(mut text) = q.get_single_mut() else { return };

    let q_tok = station.query_token;
    let k_tok = station.current_k;

    let q_str = gpt
        .token_strings
        .get(q_tok)
        .map(|s| clean_token(s))
        .unwrap_or_else(|| "?".into());
    let k_str = gpt
        .token_strings
        .get(k_tok)
        .map(|s| clean_token(s))
        .unwrap_or_else(|| "?".into());

    let body = match get_qk_vectors(&gpt, station.layer, station.head, q_tok, k_tok) {
        None => {
            "Waiting for forward pass to produce QKV activations...".to_string()
        }
        Some((q_vec, k_vec)) => {
            let d_head = q_vec.len();
            let sqrt_d = (d_head as f32).sqrt();
            let raw: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
            let scaled = raw / sqrt_d;

            // Real softmax probability from the attention tile.
            let prob = gpt
                .model
                .layout
                .layers
                .get(station.layer)
                .and_then(|l| l.attention)
                .and_then(|ni| gpt.tile_values.get(ni))
                .and_then(|v| v.as_ref())
                .and_then(|data| {
                    let t = gpt2_viz::gpt2::MAX_SEQ_LEN;
                    data.get(station.head * t * t + q_tok * t + k_tok).copied()
                });

            let mut s = String::new();
            s.push_str("HOW ONE ATTENTION SCORE IS COMPUTED\n");
            s.push_str("=============================================\n");
            s.push_str(&format!("layer {}   head {}\n", station.layer, station.head));
            s.push_str(&format!("query token: \"{q_str}\"  (pos {q_tok})\n"));
            s.push_str(&format!("key   token: \"{k_str}\"  (pos {k_tok})\n"));
            s.push_str("\n");
            s.push_str(&format!(
                "Each token has been projected into a {d_head}-dim\n"
            ));
            s.push_str("query vector Q and key vector K (learned\n");
            s.push_str("linear projections of the residual stream).\n\n");
            s.push_str("The raw attention score between this query\n");
            s.push_str("and this key is their dot product, divided\n");
            s.push_str(&format!("by √{d_head} so the values stay well-scaled:\n\n"));
            s.push_str(&format!("  score = (Q · K) / √{d_head}\n\n"));
            s.push_str("Per-dimension contributions:\n");
            for i in 0..8.min(d_head) {
                let qv = q_vec[i];
                let kv = k_vec[i];
                let p = qv * kv;
                s.push_str(&format!(
                    "  i={i:<2}  {qv:+8.4}  *  {kv:+8.4}  =  {p:+9.5}\n"
                ));
            }
            if d_head > 8 {
                s.push_str(&format!("  ...  {} more dimensions ...\n", d_head - 8));
            }
            s.push_str("\n");
            s.push_str(&format!("Sum of all {d_head} products:  {raw:+10.4}\n"));
            s.push_str(&format!(
                "Divide by √{d_head} = {sqrt_d:.2}:     {scaled:+10.4}\n\n"
            ));

            if let Some(p) = prob {
                s.push_str("After softmax over all 8 key positions:\n");
                s.push_str(&format!("  attention = {:.4}\n", p));
                s.push_str(&format!(
                    "  → \"{q_str}\" pays {:.1}% attention to \"{k_str}\"\n",
                    p * 100.0
                ));
            } else {
                s.push_str("(softmax not yet computed)\n");
            }
            s.push_str("\n");
            s.push_str("──────────────────────────\n");
            s.push_str("[F] pause   [,] prev k   [.] next k\n");
            s
        }
    };

    text.sections[0].value = body;
}

// ---------------- Controls ----------------

fn station_keys(
    keys: Res<ButtonInput<KeyCode>>,
    gpt: NonSend<GptState>,
    mut station: ResMut<DotStation>,
) {
    if keys.just_pressed(KeyCode::KeyF) {
        station.enabled = !station.enabled;
        station.cycle_time = 0.0;
    }
    let n_tokens = gpt.n_tokens().max(1);
    if keys.just_pressed(KeyCode::Period) {
        station.current_k = (station.current_k + 1) % n_tokens;
        station.cycle_time = 0.0;
    }
    if keys.just_pressed(KeyCode::Comma) {
        station.current_k = (station.current_k + n_tokens - 1) % n_tokens;
        station.cycle_time = 0.0;
    }
}
