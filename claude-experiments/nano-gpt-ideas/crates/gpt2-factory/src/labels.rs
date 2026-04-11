//! Floating labels that track world-space anchors.
//!
//! Each label spawns an invisible world-space anchor and a UI `TextBundle`.
//! Every frame, we project the anchor through the 3D camera and move the UI
//! text to the resulting screen position.

use bevy::prelude::*;
use bevy::render::view::RenderLayers;

pub struct LabelPlugin;

impl Plugin for LabelPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_event::<SpawnLabel>()
            .add_systems(Startup, setup_label_camera)
            .add_systems(Update, (handle_spawn_events, update_label_positions));
    }
}

/// Fired by other plugins (like `tower`) to request a world-space label.
#[derive(Event, Clone)]
pub struct SpawnLabel {
    pub text: String,
    pub position: Vec3,
    pub color: Color,
    pub scale: f32,
}

/// Marker placed on the world-space transform we track.
#[derive(Component)]
pub struct WorldLabelAnchor {
    pub offset_y: f32,
}

/// Marker on the UI Text that tracks a world anchor.
#[derive(Component)]
pub struct TrackedLabel {
    pub anchor: Entity,
}

fn setup_label_camera(_commands: Commands) {
    // We reuse the main 3D camera for projection. The UI Text lives on the
    // default UI layer and we update its `Style` per-frame to follow an anchor.
}

fn handle_spawn_events(
    mut commands: Commands,
    mut events: EventReader<SpawnLabel>,
) {
    for ev in events.read() {
        // World-space anchor (invisible)
        let anchor = commands.spawn((
            SpatialBundle {
                transform: Transform::from_translation(ev.position),
                ..default()
            },
            WorldLabelAnchor { offset_y: 0.0 },
        )).id();

        // UI text entity that follows the anchor's projected screen position
        let font_size = (18.0 * ev.scale).clamp(12.0, 64.0);
        commands.spawn((
            TextBundle {
                text: Text::from_section(
                    ev.text.clone(),
                    TextStyle {
                        font_size,
                        color: ev.color,
                        ..default()
                    },
                ).with_justify(JustifyText::Center),
                style: Style {
                    position_type: PositionType::Absolute,
                    left: Val::Px(0.0),
                    top: Val::Px(0.0),
                    ..default()
                },
                ..default()
            },
            TrackedLabel { anchor },
            RenderLayers::default(),
        ));
    }
}

fn update_label_positions(
    cameras: Query<(&Camera, &GlobalTransform)>,
    anchors: Query<(&GlobalTransform, &WorldLabelAnchor)>,
    mut labels: Query<(&mut Style, &mut Visibility, &TrackedLabel)>,
) {
    let Ok((camera, cam_tf)) = cameras.get_single() else { return };
    for (mut style, mut vis, tracked) in labels.iter_mut() {
        let Ok((anchor_tf, anchor)) = anchors.get(tracked.anchor) else { continue };
        let mut world = anchor_tf.translation();
        world.y += anchor.offset_y;
        let Some(screen) = camera.world_to_viewport(cam_tf, world) else {
            *vis = Visibility::Hidden;
            continue;
        };
        *vis = Visibility::Inherited;
        // Text is anchored top-left in UI space; shift so the label is centered
        // roughly above the anchor.
        style.left = Val::Px(screen.x - 60.0);
        style.top = Val::Px(screen.y - 10.0);
    }
}
