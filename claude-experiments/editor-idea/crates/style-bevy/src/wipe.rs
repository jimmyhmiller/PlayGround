//! Per-project wipe mask — the texture that persists "fingerprint"
//! smears in the dust.
//!
//! Each known project gets its own R8-equivalent image (we use
//! `Rgba8Unorm` for portability and only use the red channel). When
//! that project is active and the mouse moves, we paint a soft brush
//! into the image along the segment between the previous and current
//! cursor positions. The dust shader samples this mask via UV and
//! attenuates dust output by the sampled value — wiped regions show
//! no dust.
//!
//! The mask is **never reset** while the app is running. Switching
//! projects swaps in that project's mask (creating a fresh blank one
//! the first time a project becomes active). On-disk persistence is
//! not yet implemented; the mask lives in RAM only and resets on app
//! restart.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::dev::DevOverrides;
use crate::shader::{tick_project, tick_windex, ActiveProject, ShaderDataRegistry, WindexAnim, WIPE_MASK_SIZE};
use crate::state::ProjectStyleState;
use crate::theme::{tokens, Theme, TokenValue};

/// Handles for each project's wipe mask, indexed by project id.
#[derive(Resource, Default)]
pub struct WipeMasks {
    pub by_project: HashMap<u64, Handle<Image>>,
}

/// Last cursor position painted, for segment interpolation. Tracking
/// it inside a resource (vs `Local`) so the painter system can be
/// chained with other systems if needed later.
#[derive(Resource, Default)]
pub struct LastCursor(pub Option<Vec2>);

pub struct WipePlugin;

impl Plugin for WipePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WipeMasks>()
            .init_resource::<LastCursor>()
            .add_systems(
                Update,
                (
                    // Windex trigger: start the temporary animation;
                    // does NOT touch the persistent mask itself —
                    // that happens at the end via reset_on_windex.
                    start_windex_anim,
                    // Reset-on-complete runs AFTER tick_windex so it
                    // can observe the just_completed flag this frame.
                    reset_on_windex_complete.after(tick_windex),
                    // Painting reads the just-computed dust value out
                    // of `ShaderDataRegistry`, so it must run after
                    // the project provider has populated it.
                    paint_mouse_into_mask.after(tick_project),
                ),
            );
    }
}

/// When the Windex animation completes, reset the active project's
/// dust state to "fresh":
///   - Clear all per-project dev overrides (so the live timer takes
///     over again).
///   - Reset `last_focus_at` to now so the dust timer starts from 0.
///   - Wipe the persistent mask (no smears carry over).
///
/// The Windex animation itself plays before this fires; from the
/// user's perspective: spray sweeps across, dust appears wiped, then
/// when the spray fades the underlying state has actually been reset
/// so dust doesn't immediately reappear at its prior level.
fn reset_on_windex_complete(
    mut anim: ResMut<WindexAnim>,
    active: Res<ActiveProject>,
    mut overrides: ResMut<DevOverrides>,
    mut state: ResMut<ProjectStyleState>,
    masks: Res<WipeMasks>,
    mut images: ResMut<Assets<Image>>,
) {
    if !anim.just_completed {
        return;
    }
    anim.just_completed = false;
    let Some(pid) = active.0 else { return };

    // Drop this project's dev overrides (dust/edit/age forced values).
    overrides.per_project.remove(&pid);
    // Reset focus timer so live dust_seconds is ~0.
    state.note_focus(pid);
    // Clear the persistent wipe mask (R channel everywhere = 0 again).
    if let Some(handle) = masks.by_project.get(&pid)
        && let Some(image) = images.get_mut(handle)
        && let Some(data) = image.data.as_mut()
    {
        for chunk in data.chunks_mut(4) {
            chunk[0] = 0;
        }
    }
}

/// Consume `DevOverrides.pending_windex`: kick off a transient
/// animation (driven entirely from `default_background.wgsl`). Does
/// NOT modify the persistent wipe mask, so when the animation ends
/// the canvas returns to whatever wipe state it was in before.
fn start_windex_anim(
    time: Res<Time>,
    mut overrides: ResMut<DevOverrides>,
    mut anim: ResMut<WindexAnim>,
) {
    if !overrides.pending_windex {
        return;
    }
    overrides.pending_windex = false;
    anim.started_at = Some(time.elapsed_secs());
}

/// Get-or-create the active project's wipe mask, then paint a stroke
/// from the last cursor position to the current one. The stroke is a
/// soft circle (cosine falloff) at every interpolated step along the
/// segment, taking max() with whatever's already there.
pub fn paint_mouse_into_mask(
    active: Res<ActiveProject>,
    mut masks: ResMut<WipeMasks>,
    mut last: ResMut<LastCursor>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    registry: Res<ShaderDataRegistry>,
    theme: Res<Theme>,
    panes: Query<&pane_bevy::PaneRect, With<pane_bevy::PaneTag>>,
) {
    let Some(pid) = active.0 else {
        last.0 = None;
        return;
    };
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        last.0 = None;
        return;
    };
    let win_size = Vec2::new(window.width().max(1.0), window.height().max(1.0));

    let prev = last.0;
    last.0 = Some(cursor);

    // If the cursor is over a pane, skip painting. The panes sit
    // BELOW the dust overlay, but moving over them shouldn't leave
    // wipe trails — otherwise closing/moving a pane reveals "drawings"
    // where the user happened to drag past its rect.
    let over_pane = panes
        .iter()
        .any(|rect| point_in_rect(cursor, rect.pos, rect.size));
    if over_pane {
        return;
    }

    // Pull live-tunable thresholds out of the theme so they can be
    // tweaked from `theme.rhai` without a rebuild. Falls back to the
    // default-theme values if either token is missing.
    let gate_secs = match theme.get(tokens::WIPE_DUST_GATE_SECS) {
        Some(TokenValue::F32(v)) => v,
        _ => 60.0,
    };
    let brush_radius = match theme.get(tokens::WIPE_BRUSH_RADIUS_PX) {
        Some(TokenValue::F32(v)) => v,
        _ => 80.0,
    };
    let dust_intensity = match theme.get(tokens::DUST_INTENSITY) {
        Some(TokenValue::F32(v)) => v,
        _ => 1.0,
    };
    // If the project has dust turned off entirely, painting wipe
    // makes no sense — there's nothing on screen to clean.
    if dust_intensity <= 0.0 {
        return;
    }

    // Gate on current dust. If there's nothing visible to wipe,
    // update `last_cursor` (so the next segment doesn't stretch from
    // wherever we were a minute ago) but don't actually paint. We
    // read the just-computed dust value from `ShaderDataRegistry`,
    // which has already had any dev-override applied. The threshold
    // is in dust_seconds, which corresponds to the shader's
    // dust_amount curve — at `gate_secs = 600` (10 min) the shader
    // shows roughly `sqrt(600/86400) ≈ 0.083` dust intensity, the
    // floor of "visible enough to wipe".
    if registry.project.dust_seconds <= gate_secs {
        return;
    }

    // No prior cursor means this is the first frame the user moved
    // *while dust is on*. Treat this frame's stamp as a single dot
    // rather than a segment from somewhere stale.
    let prev = prev.unwrap_or(cursor);

    // Get-or-create the mask for this project.
    let handle = masks
        .by_project
        .entry(pid)
        .or_insert_with(|| images.add(blank_mask()))
        .clone();
    let Some(image) = images.get_mut(&handle) else { return };

    // Logical-window pixel coords -> mask pixel coords. Mask is
    // UV-mapped over the whole window, so a (0.5, 0.5) cursor lands
    // at the mask's center regardless of window aspect.
    let to_mask = |p: Vec2| -> Vec2 {
        let uv = p / win_size;
        Vec2::new(uv.x * WIPE_MASK_SIZE as f32, uv.y * WIPE_MASK_SIZE as f32)
    };
    let p0 = to_mask(prev);
    let p1 = to_mask(cursor);

    // Stamp the brush at every ~half-brush-radius step so a fast
    // cursor still produces a continuous line. `steps` is capped to
    // avoid the pathological case where the cursor "teleports".
    let segment_px = p0.distance(p1);
    let step_px = brush_radius * 0.4;
    let steps = ((segment_px / step_px).ceil() as usize).clamp(1, 64);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        stamp_brush(image, p0.lerp(p1, t), brush_radius);
    }
}

fn point_in_rect(p: Vec2, pos: Vec2, size: Vec2) -> bool {
    p.x >= pos.x && p.x <= pos.x + size.x && p.y >= pos.y && p.y <= pos.y + size.y
}

pub fn blank_mask() -> Image {
    let bytes = vec![0u8; (WIPE_MASK_SIZE * WIPE_MASK_SIZE * 4) as usize];
    Image::new(
        Extent3d {
            width: WIPE_MASK_SIZE,
            height: WIPE_MASK_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    )
}

/// Soft cosine-falloff brush. Each pixel within `radius` is bumped to
/// at least `(cos(d/r * π/2))²` * 255 — i.e. a smooth dome with no
/// hard edge. Uses max() rather than additive blending so repeated
/// strokes over the same spot eventually saturate at 255 instead of
/// blowing out.
fn stamp_brush(image: &mut Image, center: Vec2, radius: f32) {
    let w = WIPE_MASK_SIZE as i32;
    let h = WIPE_MASK_SIZE as i32;
    let Some(data) = image.data.as_mut() else { return };

    let r = radius.ceil() as i32;
    let cx = center.x.round() as i32;
    let cy = center.y.round() as i32;
    let x_lo = (cx - r).max(0);
    let x_hi = (cx + r).min(w - 1);
    let y_lo = (cy - r).max(0);
    let y_hi = (cy + r).min(h - 1);

    for y in y_lo..=y_hi {
        let row = (y * w) as usize * 4;
        for x in x_lo..=x_hi {
            let dx = x as f32 - center.x;
            let dy = y as f32 - center.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > radius {
                continue;
            }
            let t = (dist / radius).clamp(0.0, 1.0);
            // Cosine ease — full center, soft edge.
            let falloff = (1.0 - t * t).max(0.0);
            let stroke = (falloff * 255.0) as u8;
            let idx = row + (x as usize) * 4;
            // Only the R channel is sampled by the shader.
            if data[idx] < stroke {
                data[idx] = stroke;
            }
        }
    }
}
