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
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey, Material2dPlugin};

/// Content-local bounds and owning element for one live Glaze shader layer.
/// This stays on the material entity across frames, allowing interaction
/// uniforms to update without rebuilding the widget tree.
#[derive(Component, Clone, Debug)]
pub struct GlazeInteractionTarget {
    pub pane: Entity,
    pub element_id: Option<String>,
    pub rect: Rect,
}

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
            hover: 0.0,
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
        Self {
            fragment: m.fragment.clone(),
        }
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
    pub fn handle_for(&mut self, body: &str, shaders: &mut Assets<Shader>) -> Handle<Shader> {
        let mut h = DefaultHasher::new();
        body.hash(&mut h);
        let key = h.finish();
        self.by_hash
            .entry(key)
            .or_insert_with(|| {
                shaders.add(Shader::from_wgsl(
                    assemble_wgsl(body),
                    "glaze://generated.wgsl",
                ))
            })
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
            .add_systems(Update, update_glaze_materials);
    }
}

fn ease(current: f32, target: f32, dt: f32) -> f32 {
    let amount = 1.0 - (-14.0 * dt.max(0.0)).exp();
    current + (target - current) * amount
}

fn normalized_mouse(rect: Rect, point: Vec2) -> Vec2 {
    let size = rect.size().max(Vec2::splat(f32::EPSILON));
    (point - rect.min) / size
}

/// Drive clocks and interaction uniforms independently of the event-driven
/// widget rebuild.
fn update_glaze_materials(
    time: Res<Time>,
    windows: Query<&Window>,
    viewport: Option<Res<pane_bevy::PaneViewport>>,
    buttons: Res<ButtonInput<bevy::input::mouse::MouseButton>>,
    panes: Query<
        (
            Entity,
            &pane_bevy::PaneRect,
            Option<&Visibility>,
            Option<&crate::WidgetScroll>,
            Option<&crate::WidgetInputFocus>,
        ),
        With<pane_bevy::PaneTag>,
    >,
    layers: Query<(
        Entity,
        &GlazeInteractionTarget,
        &MeshMaterial2d<GlazeMaterial>,
    )>,
    mut mats: ResMut<Assets<GlazeMaterial>>,
    mut pressed_layers: Local<HashSet<Entity>>,
) {
    let t = time.elapsed_secs();
    let dt = time.delta_secs();
    let cursor_canvas = windows
        .single()
        .ok()
        .and_then(Window::cursor_position)
        .zip(viewport.as_deref())
        .map(|(pt, viewport)| viewport.window_to_canvas(pt));
    let candidates: Vec<(Entity, pane_bevy::PaneRect)> = panes
        .iter()
        .filter(|(_, _, vis, _, _)| !matches!(vis, Some(Visibility::Hidden)))
        .map(|(pane, rect, _, _, _)| (pane, *rect))
        .collect();
    let topmost = cursor_canvas
        .and_then(|pt| pane_bevy::topmost_pane_at(pt, &candidates).map(|pane| (pane, pt)));
    let left_down = buttons.pressed(bevy::input::mouse::MouseButton::Left);
    if buttons.just_released(bevy::input::mouse::MouseButton::Left) || !left_down {
        pressed_layers.clear();
    }

    // Standalone users such as `glaze_gallery` do not carry widget interaction
    // metadata, but their animated shaders still need clocks.
    for (_, material) in mats.iter_mut() {
        material.u.time = t;
        material.u.dt = dt;
    }

    for (entity, target, handle) in &layers {
        let Some(m) = mats.get_mut(&handle.0) else {
            continue;
        };

        let pane_state = panes.get(target.pane).ok();
        let pointer = match (topmost, pane_state.as_ref()) {
            (Some((pane, pt)), Some((_, rect, _, scroll, _))) if pane == target.pane => {
                let mut local = pane_bevy::pt_to_content_local(pt, rect);
                local.y += scroll.map(|s| s.y).unwrap_or(0.0);
                Some(local)
            }
            _ => None,
        };
        let hovered = pointer.is_some_and(|pt| target.rect.contains(pt));
        if buttons.just_pressed(bevy::input::mouse::MouseButton::Left) && hovered {
            pressed_layers.insert(entity);
        }
        let focused = pane_state
            .and_then(|(_, _, _, _, focus)| focus)
            .is_some_and(|focus| target.element_id.as_deref() == Some(focus.id.as_str()));

        m.u.hover = ease(m.u.hover, hovered as u8 as f32, dt);
        m.u.focus = ease(m.u.focus, focused as u8 as f32, dt);
        m.u.press = ease(
            m.u.press,
            (left_down && pressed_layers.contains(&entity)) as u8 as f32,
            dt,
        );
        if let Some(pt) = pointer {
            m.u.mouse = normalized_mouse(target.rect, pt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interaction_uniforms_default_inactive() {
        let uniforms = GlazeUniforms::default();
        assert_eq!(uniforms.hover, 0.0);
        assert_eq!(uniforms.focus, 0.0);
        assert_eq!(uniforms.press, 0.0);
        assert_eq!(uniforms.mouse, Vec2::ZERO);
    }

    #[test]
    fn interaction_easing_moves_toward_target() {
        let entered = ease(0.0, 1.0, 1.0 / 60.0);
        let left = ease(entered, 0.0, 1.0 / 60.0);
        assert!(entered > 0.0 && entered < 1.0);
        assert!(left >= 0.0 && left < entered);
    }

    #[test]
    fn mouse_is_normalized_within_element_bounds() {
        let rect = Rect::new(10.0, 20.0, 110.0, 70.0);
        assert_eq!(
            normalized_mouse(rect, Vec2::new(60.0, 45.0)),
            Vec2::splat(0.5)
        );
    }
}
