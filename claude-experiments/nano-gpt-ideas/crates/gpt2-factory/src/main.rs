//! GPT-2 Factory — walkable 3D visualization of a transformer.

mod attention;
mod belt;
mod camera;
mod config;
mod dotstation;
mod labels;
mod model;
mod prediction;
mod pulse;
mod tower;

use bevy::prelude::*;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "The quick brown fox jumps over the lazy dog".to_string()
    };

    // Locate gpt2_weights/ at workspace root (same rule as gpt2-viz).
    let weights_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("gpt2_weights");
    if !weights_dir.join("manifest.json").exists() {
        eprintln!(
            "ERROR: gpt2_weights/ not found at {:?}.\n\
             Run the export script in the existing gpt2-viz workflow first.",
            weights_dir
        );
        std::process::exit(1);
    }

    let tokenizer = tokenizers::Tokenizer::from_file(weights_dir.join("tokenizer/tokenizer.json"))
        .unwrap_or_else(|e| {
            eprintln!("tokenizer load failed: {e}");
            std::process::exit(1);
        });

    let encoding = tokenizer.encode(prompt.as_str(), false).unwrap();
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let token_strings: Vec<String> = encoding
        .get_tokens()
        .iter()
        .map(|s| s.to_string())
        .collect();

    eprintln!("Prompt: {prompt:?}");
    eprintln!("Tokens ({}): {token_strings:?}", token_ids.len());

    let mut model = gpt2_viz::gpt2::Gpt2Model::load(&weights_dir);
    let n_nodes = model.node_infos.len();

    // Kick off the forward pass on a background thread before Bevy starts.
    let initial_ids: Vec<u32> = token_ids
        .iter()
        .take(gpt2_viz::gpt2::MAX_SEQ_LEN)
        .copied()
        .collect();
    model.launch_async(&initial_ids);

    let state = model::GptState {
        model,
        tokenizer,
        token_ids,
        token_strings,
        tile_values: vec![None; n_nodes],
        logits: None,
        computing: true,
        version: 0,
    };

    App::new()
        .insert_non_send_resource(state)
        .insert_resource(ClearColor(Color::srgb(0.015, 0.018, 0.028)))
        .insert_resource(AmbientLight {
            color: Color::srgb(0.55, 0.6, 0.85),
            brightness: 80.0,
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "GPT-2 Factory".into(),
                resolution: (1600., 1000.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(camera::FpsCameraPlugin)
        .add_plugins(model::ModelPlugin)
        .add_plugins(tower::TowerPlugin)
        .add_plugins(belt::BeltPlugin)
        .add_plugins(attention::AttentionPlugin)
        .add_plugins(dotstation::DotStationPlugin)
        .add_plugins(pulse::PulsePlugin)
        .add_plugins(prediction::PredictionPlugin)
        .add_plugins(labels::LabelPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, toggle_cursor_grab)
        .run();
}

fn setup_scene(mut commands: Commands) {
    // Key light — warm overhead
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.85),
            illuminance: 7500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(40.0, 80.0, 30.0)
            .looking_at(Vec3::new(0.0, 20.0, 0.0), Vec3::Y),
        ..default()
    });

    // Fill light — cool, from the side
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::srgb(0.4, 0.5, 0.9),
            illuminance: 2500.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(-50.0, 30.0, -40.0)
            .looking_at(Vec3::new(0.0, 20.0, 0.0), Vec3::Y),
        ..default()
    });
}

/// ESC releases the cursor; click to grab again.
fn toggle_cursor_grab(
    mut windows: Query<&mut Window>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
) {
    let Ok(mut win) = windows.get_single_mut() else { return };
    if keys.just_pressed(KeyCode::Escape) {
        win.cursor.visible = true;
        win.cursor.grab_mode = bevy::window::CursorGrabMode::None;
    }
    if mouse.just_pressed(MouseButton::Left) && win.cursor.visible {
        win.cursor.visible = false;
        win.cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
    }
}
