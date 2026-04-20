use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, (pan_camera, zoom_camera));
    }
}

#[derive(Component)]
pub struct MainCamera;

fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d, MainCamera));
}

fn pan_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion_evr: MessageReader<bevy::input::mouse::MouseMotion>,
    mut q: Query<(&mut Transform, &Projection), With<MainCamera>>,
) {
    // Middle-mouse or right-mouse drag pans the canvas.
    if !(buttons.pressed(MouseButton::Middle) || buttons.pressed(MouseButton::Right)) {
        motion_evr.clear();
        return;
    }
    let Ok((mut tf, proj)) = q.single_mut() else { return };
    let scale = match proj {
        Projection::Orthographic(o) => o.scale,
        _ => 1.0,
    };
    for ev in motion_evr.read() {
        tf.translation.x -= ev.delta.x * scale;
        tf.translation.y += ev.delta.y * scale;
    }
}

fn zoom_camera(
    mut wheel_evr: MessageReader<MouseWheel>,
    mut q: Query<&mut Projection, With<MainCamera>>,
    panes: Query<&Interaction, With<crate::ui::ScrollPane>>,
) {
    // If the pointer is over a scrollable UI panel, the wheel is
    // theirs — let `ui::scroll_panes_on_wheel` consume it and don't
    // zoom the canvas. Checks hover/pressed so grabbing the
    // scrollbar still forwards the event correctly.
    let over_pane = panes.iter().any(|i| {
        matches!(i, Interaction::Hovered | Interaction::Pressed)
    });
    if over_pane {
        wheel_evr.clear();
        return;
    }
    let Ok(mut proj) = q.single_mut() else { return };
    let Projection::Orthographic(ortho) = proj.as_mut() else { return };
    for ev in wheel_evr.read() {
        let step = match ev.unit {
            MouseScrollUnit::Line => ev.y * 0.1,
            MouseScrollUnit::Pixel => ev.y * 0.005,
        };
        ortho.scale = (ortho.scale * (1.0 - step)).clamp(0.2, 5.0);
    }
}
