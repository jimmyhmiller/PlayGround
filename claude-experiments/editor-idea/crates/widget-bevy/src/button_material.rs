//! Widget button material — rounded-rect SDF with optional border and
//! soft drop shadow. One material per Button element rendered.
//!
//! All look properties come from theme tokens (via `WidgetPalette`),
//! so a single preset switch retones every button across every widget.

use bevy::asset::{embedded_path, AssetPath};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

pub struct WidgetButtonMaterialPlugin;

impl Plugin for WidgetButtonMaterialPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "button_material.wgsl");
        app.add_plugins(Material2dPlugin::<WidgetButtonMaterial>::default())
            .add_systems(Startup, init_button_mesh);
    }
}

/// Shared unit quad mesh — every button reuses it and scales via its
/// `Transform`. One mesh, many materials.
#[derive(Resource, Clone)]
pub struct WidgetButtonMesh(pub Handle<Mesh>);

fn init_button_mesh(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let handle = meshes.add(Rectangle::new(1.0, 1.0));
    commands.insert_resource(WidgetButtonMesh(handle));
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct WidgetButtonMaterial {
    #[uniform(0)]
    pub params: ButtonParams,
}

impl Material2d for WidgetButtonMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path(
            AssetPath::from_path_buf(embedded_path!("button_material.wgsl"))
                .with_source("embedded"),
        )
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct ButtonParams {
    /// Mesh extent in pixels (button + 2 × shadow_blur on each axis).
    pub mesh_size: Vec2,
    /// The clickable button rect inside the mesh.
    pub button_size: Vec2,
    pub corner_radius: f32,
    pub border_width: f32,
    pub bg: Vec4,
    pub border: Vec4,
    /// `(r, g, b, base_alpha)` — shadow at the rect edge.
    pub shadow_color: Vec4,
    pub shadow_blur: f32,
    pub shadow_offset_y: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}
