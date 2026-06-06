//! Host-side runtime for Glaze shader layers (Stage 3b).
//!
//! A widget sends a `Style.shader` carrying the WGSL fragment body the `glaze`
//! compiler produced. We wrap it in the canonical `GlazeUniforms` block, add it
//! to `Assets<Shader>` (cached by content hash), and run it on a quad at the
//! element's rect via a `Material2d` whose per-instance shader handle is pinned
//! in `specialize()` (same trick as `style_bevy::DynamicMaterial`).
//!
//! `time`/`dt` are bumped every frame by `tick_glaze_materials`, so animation is
//! independent of the (event-driven) widget content rebuild.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey, Material2dPlugin};

/// Canonical per-frame inputs a Glaze shader may read. Field order matches the
/// WGSL struct in [`assemble_wgsl`]; `encase` (via `ShaderType`) inserts the
/// same std140 padding the WGSL side does.
#[derive(Clone, Copy, ShaderType)]
pub struct GlazeUniforms {
    pub time: f32,
    pub dt: f32,
    pub hover: f32,
    pub focus: f32,
    pub press: f32,
    /// element corner radius (px) — the assembled shader masks its output to a
    /// rounded-rect of this radius so shaders don't overpaint rounded corners.
    pub radius: f32,
    pub mouse: Vec2,
    pub size: Vec2,
    pub resolution: Vec2,
}

impl Default for GlazeUniforms {
    fn default() -> Self {
        GlazeUniforms {
            time: 0.0,
            dt: 0.0,
            hover: 1.0,
            focus: 0.0,
            press: 0.0,
            radius: 0.0,
            mouse: Vec2::ZERO,
            size: Vec2::splat(1.0),
            resolution: Vec2::splat(1.0),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
#[bind_group_data(GlazeMaterialKey)]
pub struct GlazeMaterial {
    #[uniform(0)]
    pub u: GlazeUniforms,
    /// The per-instance fragment shader. `specialize` pins it; the pipeline
    /// cache keys on the handle via [`GlazeMaterialKey`].
    pub fragment: Handle<Shader>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct GlazeMaterialKey {
    fragment: Handle<Shader>,
}

impl From<&GlazeMaterial> for GlazeMaterialKey {
    fn from(m: &GlazeMaterial) -> Self {
        Self { fragment: m.fragment.clone() }
    }
}

impl Material2d for GlazeMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader = key.bind_group_data.fragment.clone();
        }
        Ok(())
    }
}

/// Caches generated `Shader` assets by the hash of their WGSL body, so a widget
/// re-render (which rebuilds the Element tree) reuses the compiled shader rather
/// than adding a fresh asset every time.
#[derive(Resource, Default)]
pub struct GlazeShaderCache {
    by_hash: HashMap<u64, Handle<Shader>>,
}

impl GlazeShaderCache {
    /// Get-or-create the `Shader` handle for a fragment body.
    pub fn handle_for(
        &mut self,
        body: &str,
        shaders: &mut Assets<Shader>,
    ) -> Handle<Shader> {
        let mut h = DefaultHasher::new();
        body.hash(&mut h);
        let key = h.finish();
        self.by_hash
            .entry(key)
            .or_insert_with(|| shaders.add(Shader::from_wgsl(assemble_wgsl(body), "glaze://generated.wgsl")))
            .clone()
    }
}

/// Wrap a compiler-produced fragment body in a complete mesh2d fragment shader
/// with the canonical uniform block.
pub fn assemble_wgsl(body: &str) -> String {
    format!(
        "#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n\
         \n\
         struct GlazeUniforms {{\n\
         \x20   time: f32,\n\x20   dt: f32,\n\x20   hover: f32,\n\x20   focus: f32,\n\
         \x20   press: f32,\n\x20   radius: f32,\n\x20   mouse: vec2<f32>,\n\x20   size: vec2<f32>,\n\x20   resolution: vec2<f32>,\n\
         }};\n\
         @group(#{{MATERIAL_BIND_GROUP}}) @binding(0) var<uniform> u: GlazeUniforms;\n\
         \n\
         // the compiler-produced body, as a callable so the fragment can mask it\n\
         fn glaze_body(in: VertexOutput) -> vec4<f32> {{\n{body}}}\n\
         \n\
         @fragment\n\
         fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {{\n\
         \x20   var col = glaze_body(in);\n\
         \x20   // clip to the element's rounded-rect (so shaders don't square off corners)\n\
         \x20   let p = (in.uv - vec2<f32>(0.5, 0.5)) * u.size;\n\
         \x20   let h = u.size * 0.5;\n\
         \x20   let r = min(u.radius, min(h.x, h.y));\n\
         \x20   let q = abs(p) - h + vec2<f32>(r, r);\n\
         \x20   let d = length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;\n\
         \x20   col.a = col.a * smoothstep(0.75, -0.75, d);\n\
         \x20   return col;\n\
         }}\n"
    )
}

pub struct GlazeMaterialPlugin;

impl Plugin for GlazeMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Material2dPlugin::<GlazeMaterial>::default())
            .init_resource::<GlazeShaderCache>()
            .add_systems(Update, tick_glaze_materials);
    }
}

/// Drive `time`/`dt` on every live Glaze material each frame, decoupled from the
/// (event-driven) widget content rebuild so shaders animate continuously.
fn tick_glaze_materials(time: Res<Time>, mut mats: ResMut<Assets<GlazeMaterial>>) {
    let t = time.elapsed_secs();
    let dt = time.delta_secs();
    for (_, m) in mats.iter_mut() {
        m.u.time = t;
        m.u.dt = dt;
    }
}
