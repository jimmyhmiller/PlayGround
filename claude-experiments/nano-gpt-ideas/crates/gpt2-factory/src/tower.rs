//! Procedurally spawn the 12-floor tower.
//!
//! Each floor contains an Attention Hall (lower 2/3) and MLP Chamber (upper 1/3),
//! separated from the next floor by a thin slab. The building is "open" — no outer
//! walls along +/- X so you can see in from the side — but has floor/ceiling slabs,
//! two side rails, and back wall panels to give it a physical shell.

use bevy::prelude::*;

use crate::config::*;
use crate::labels::SpawnLabel;

pub struct TowerPlugin;

impl Plugin for TowerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_tower);
    }
}

#[derive(Component)]
pub struct TowerPart;

#[derive(Component)]
#[allow(dead_code)]
pub struct BlockRoom {
    pub layer: usize,
    pub kind: RoomKind,
}

#[derive(Clone, Copy, PartialEq)]
pub enum RoomKind {
    Attention,
    Mlp,
}

fn spawn_tower(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut label_events: EventWriter<SpawnLabel>,
) {
    // --- Shared materials ---
    let slab_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.10, 0.11, 0.14),
        perceptual_roughness: 0.85,
        metallic: 0.15,
        ..default()
    });
    let rail_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.20, 0.26),
        perceptual_roughness: 0.55,
        metallic: 0.5,
        ..default()
    });
    let back_wall_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.07, 0.08, 0.12),
        perceptual_roughness: 0.9,
        metallic: 0.05,
        ..default()
    });
    let attn_floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.11, 0.08, 0.15),
        emissive: LinearRgba::rgb(0.02, 0.01, 0.05),
        perceptual_roughness: 0.8,
        metallic: 0.25,
        ..default()
    });
    let mlp_floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.12, 0.14),
        emissive: LinearRgba::rgb(0.01, 0.03, 0.04),
        perceptual_roughness: 0.8,
        metallic: 0.25,
        ..default()
    });

    // --- Ground plane (dark foundation under everything) ---
    {
        let w = BLOCK_WIDTH * 2.5;
        let d = HALL_LENGTH * 3.0 + tower_total_height() * 0.0;
        let ground_mesh = meshes.add(Cuboid::new(w, 0.4, d));
        let ground_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.04, 0.05, 0.07),
            perceptual_roughness: 1.0,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: ground_mesh,
                material: ground_mat,
                transform: Transform::from_xyz(0.0, -0.2, 0.0),
                ..default()
            },
            TowerPart,
        ));
    }

    // --- Slabs between floors (and cap top) ---
    for i in 0..=N_LAYER {
        let y = floor_y(i);
        let slab = meshes.add(Cuboid::new(BLOCK_WIDTH, SLAB_THICKNESS, BLOCK_DEPTH));
        commands.spawn((
            PbrBundle {
                mesh: slab,
                material: slab_mat.clone(),
                transform: Transform::from_xyz(0.0, y, 0.0),
                ..default()
            },
            TowerPart,
        ));
    }

    // --- Per-block rooms: attention floor-plate, mlp floor-plate, back wall, side rails ---
    for layer in 0..N_LAYER {
        let y_base = floor_y(layer) + SLAB_THICKNESS * 0.5;

        // Attention hall floor plate (a second, colored floor just above the slab)
        let attn_y = y_base + 0.05;
        let attn_mesh = meshes.add(Cuboid::new(
            BLOCK_WIDTH - 1.0,
            0.08,
            BLOCK_DEPTH - 1.0,
        ));
        commands.spawn((
            PbrBundle {
                mesh: attn_mesh,
                material: attn_floor_mat.clone(),
                transform: Transform::from_xyz(0.0, attn_y, 0.0),
                ..default()
            },
            BlockRoom { layer, kind: RoomKind::Attention },
            TowerPart,
        ));

        // MLP chamber floor plate (sits on top of attention hall height)
        let mlp_y = floor_y(layer) + BLOCK_GAP + ATTN_HEIGHT;
        let mlp_plate = meshes.add(Cuboid::new(
            BLOCK_WIDTH - 4.0,
            0.08,
            BLOCK_DEPTH - 4.0,
        ));
        commands.spawn((
            PbrBundle {
                mesh: mlp_plate,
                material: mlp_floor_mat.clone(),
                transform: Transform::from_xyz(0.0, mlp_y, 0.0),
                ..default()
            },
            BlockRoom { layer, kind: RoomKind::Mlp },
            TowerPart,
        ));

        // Back wall (panel behind each block on -X side)
        let back_mesh = meshes.add(Cuboid::new(
            0.3,
            BLOCK_HEIGHT - 0.5,
            BLOCK_DEPTH - 1.0,
        ));
        let back_y = floor_y(layer) + BLOCK_GAP + BLOCK_HEIGHT * 0.5;
        commands.spawn((
            PbrBundle {
                mesh: back_mesh,
                material: back_wall_mat.clone(),
                transform: Transform::from_xyz(-BLOCK_WIDTH * 0.5 + 0.15, back_y, 0.0),
                ..default()
            },
            TowerPart,
        ));

        // Side rails (thin posts along the front-left / front-right of each block)
        for &x_off in &[-BLOCK_WIDTH * 0.5 + 0.4, BLOCK_WIDTH * 0.5 - 0.4] {
            for &z_off in &[-BLOCK_DEPTH * 0.5 + 0.4, BLOCK_DEPTH * 0.5 - 0.4] {
                let post = meshes.add(Cuboid::new(0.25, BLOCK_HEIGHT - 0.6, 0.25));
                commands.spawn((
                    PbrBundle {
                        mesh: post,
                        material: rail_mat.clone(),
                        transform: Transform::from_xyz(x_off, back_y, z_off),
                        ..default()
                    },
                    TowerPart,
                ));
            }
        }

        // Labels: floating text above each room
        label_events.send(SpawnLabel {
            text: format!("ATTENTION  L{layer}"),
            position: Vec3::new(
                -BLOCK_WIDTH * 0.5 + 3.0,
                attn_center_y(layer) + ATTN_HEIGHT * 0.35,
                -BLOCK_DEPTH * 0.5 + 1.0,
            ),
            color: Color::srgb(0.9, 0.6, 1.0),
            scale: 0.9,
        });
        label_events.send(SpawnLabel {
            text: format!("MLP  L{layer}"),
            position: Vec3::new(
                -BLOCK_WIDTH * 0.5 + 3.0,
                mlp_center_y(layer) + MLP_HEIGHT * 0.35,
                -BLOCK_DEPTH * 0.5 + 1.0,
            ),
            color: Color::srgb(0.5, 0.95, 0.95),
            scale: 0.7,
        });
    }

    // Top-level building label
    label_events.send(SpawnLabel {
        text: "GPT-2".into(),
        position: Vec3::new(0.0, tower_total_height() + 6.0, 0.0),
        color: Color::srgb(1.0, 0.95, 0.7),
        scale: 3.0,
    });
}
