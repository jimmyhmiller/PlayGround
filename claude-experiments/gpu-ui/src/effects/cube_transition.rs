use glam::{Quat, Vec3};
use crate::scene::SceneNode;

/// Parameters for a 3D cube scene transition.
/// Each face of the cube holds a surface. Rotating the cube transitions between them.
#[derive(Clone, Debug)]
pub struct CubeTransitionParams {
    /// Which face (0-3) we're transitioning from
    pub from_face: u32,
    /// Which face we're transitioning to
    pub to_face: u32,
    /// Progress 0.0 to 1.0
    pub progress: f32,
    /// Whether to show reflections on the "floor" beneath the cube
    pub show_reflection: bool,
}

/// Apply cube rotation to a node's transform based on transition progress.
pub fn apply_cube_transition(node: &mut SceneNode, params: &CubeTransitionParams, _time: f32) {
    let from_angle = params.from_face as f32 * std::f32::consts::FRAC_PI_2;
    let to_angle = params.to_face as f32 * std::f32::consts::FRAC_PI_2;
    let angle = from_angle + (to_angle - from_angle) * ease_in_out(params.progress);

    node.transform.rotation = Quat::from_rotation_y(angle);
}

fn ease_in_out(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
    }
}
