use glam::{Mat4, Quat, Vec3};
use super::{Surface, SurfaceId};
use crate::effects::Effect;

/// Transform in 3D space.
#[derive(Clone, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// A node in the scene graph. Each node has a transform and optionally holds a surface.
/// Nodes can have children, forming a tree. Effects are attached at the node level.
#[derive(Clone, Debug)]
pub struct SceneNode {
    pub transform: Transform,
    pub surface: Option<Surface>,
    pub effects: Vec<Effect>,
    pub children: Vec<SceneNode>,
    pub visible: bool,
}

impl SceneNode {
    pub fn empty() -> Self {
        Self {
            transform: Transform::default(),
            surface: None,
            effects: Vec::new(),
            children: Vec::new(),
            visible: true,
        }
    }

    pub fn with_surface(surface: Surface) -> Self {
        Self {
            surface: Some(surface),
            ..Self::empty()
        }
    }

    pub fn at(mut self, position: Vec3) -> Self {
        self.transform.translation = position;
        self
    }

    pub fn rotated(mut self, rotation: Quat) -> Self {
        self.transform.rotation = rotation;
        self
    }

    pub fn scaled(mut self, scale: Vec3) -> Self {
        self.transform.scale = scale;
        self
    }

    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.effects.push(effect);
        self
    }

    pub fn with_child(mut self, child: SceneNode) -> Self {
        self.children.push(child);
        self
    }
}

/// A point light in the scene.
#[derive(Clone, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

/// The complete scene description. This is the boundary between
/// "what to draw" and "how to draw it."
#[derive(Clone, Debug)]
pub struct Scene {
    pub root: SceneNode,
    pub lights: Vec<PointLight>,
    pub ambient_light: Vec3,
    pub camera_position: Vec3,
    pub camera_target: Vec3,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            root: SceneNode::empty(),
            lights: vec![PointLight {
                position: Vec3::new(2.0, 5.0, 3.0),
                color: Vec3::ONE,
                intensity: 1.5,
                radius: 20.0,
            }],
            ambient_light: Vec3::splat(0.15),
            camera_position: Vec3::new(0.0, 0.0, 5.0),
            camera_target: Vec3::ZERO,
        }
    }
}

impl Scene {
    /// Collect all surfaces with their world-space transforms by walking the tree.
    pub fn collect_surfaces(&self) -> Vec<(Mat4, &Surface)> {
        let mut result = Vec::new();
        Self::walk(&self.root, Mat4::IDENTITY, &mut result);
        result
    }

    fn walk<'a>(node: &'a SceneNode, parent_transform: Mat4, out: &mut Vec<(Mat4, &'a Surface)>) {
        if !node.visible {
            return;
        }
        let world = parent_transform * node.transform.to_matrix();
        if let Some(ref surface) = node.surface {
            out.push((world, surface));
        }
        for child in &node.children {
            Self::walk(child, world, out);
        }
    }

    pub fn find_surface_mut(&mut self, id: SurfaceId) -> Option<&mut Surface> {
        Self::find_in_node(&mut self.root, id)
    }

    fn find_in_node(node: &mut SceneNode, id: SurfaceId) -> Option<&mut Surface> {
        if let Some(ref mut s) = node.surface {
            if s.id == id {
                return node.surface.as_mut();
            }
        }
        for child in &mut node.children {
            if let Some(s) = Self::find_in_node(child, id) {
                return Some(s);
            }
        }
        None
    }
}
