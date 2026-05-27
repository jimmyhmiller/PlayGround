//! Per-pane render cameras.
//!
//! Each pane owns a sibling `Camera2d` whose viewport is restricted to
//! the pane's content area in window-physical pixels. Combined with a
//! `RenderLayers::layer(N)` filter matching the pane's `PaneLayer`, the
//! camera sees only that pane's content_root subtree and the GPU
//! refuses to write fragments outside the viewport. This is what
//! finally makes pane clipping a property of the renderer instead of a
//! per-kind discipline.
//!
//! Stage 2 (this module) spawns the cameras and keeps their viewports
//! in sync with `PaneRect`. The cameras render *nothing* until Stage 3
//! propagates `RenderLayers` to descendants of `content_root` — at
//! which point clipping just starts working, with no kind-side
//! changes.
//!
//! Lifecycle:
//! - On `Added<PaneLayer>` (pane spawn), spawn a camera linked back to
//!   the pane via `PaneCameraOf(pane_entity)`.
//! - Every frame in `PostUpdate`, sync viewport + order from
//!   `PaneRect` (handles drag, resize, focus/raise).
//! - When the pane closes, the integration point in
//!   `apply_pending_pane_actions` despawns the camera and returns the
//!   layer id to the allocator.

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, Viewport};
use bevy::prelude::*;

use crate::layers::PaneLayer;
use crate::{PaneRect, PaneTag};

/// Optional rectangular sub-region of the window where pane cameras
/// are allowed to render. The host sets this each frame so that
/// non-pane chrome (sidebar, top menu, status bar) sits visually
/// *on top of* the pane canvas — by clipping the per-pane camera
/// viewports to a region that excludes the chrome, panes simply
/// can't draw over those areas. Coordinates are window logical px,
/// top-left origin. `None` (or the resource missing) means "full
/// window", matching the pre-existing behavior.
#[derive(Resource, Copy, Clone, Debug, Default)]
pub struct PaneCanvasRegion {
    pub min: Vec2,
    pub max: Vec2,
    /// If false, the region is ignored and the full window is used.
    pub active: bool,
}

/// Marks a camera as belonging to a specific pane entity. The
/// `apply_pending_pane_actions` close handler uses this to find and
/// despawn the camera when the pane closes.
#[derive(Component, Copy, Clone, Debug)]
pub struct PaneCameraOf(pub Entity);

/// Spawn a per-pane camera the first frame after a pane gains a
/// `PaneLayer`. The camera renders only the pane's layer (nothing
/// today, the pane's content_root subtree once Stage 3 propagation
/// flips on) and writes only inside the pane's content rect.
///
/// We do this in a system instead of at `spawn_pane` time because the
/// initial viewport needs the window scale factor and `spawn_pane` is
/// called from contexts where window access is awkward. One-frame
/// delay before the camera exists is harmless: it has nothing to
/// render anyway until Stage 3.
pub fn spawn_pane_cameras(
    new_panes: Query<(Entity, &PaneLayer, &PaneRect), Added<PaneLayer>>,
    windows: Query<&Window>,
    region: Option<Res<PaneCanvasRegion>>,
    mut commands: Commands,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let region = region.as_deref().copied();
    for (pane_entity, layer, rect) in &new_panes {
        let cam_setup = pane_camera_setup(rect, window, region);
        let order = pane_camera_order(rect);
        commands.spawn((
            Camera2d,
            Camera {
                order,
                viewport: Some(cam_setup.viewport),
                // Don't clear — the pane camera overlays the main
                // camera's chrome render. Clearing would wipe the
                // chrome under the pane.
                clear_color: ClearColorConfig::None,
                ..default()
            },
            // The camera transform must match the (potentially
            // clamped) viewport's world center, not the pane's
            // unclipped center — otherwise content shifts when the
            // pane is dragged partly off-screen. See
            // `pane_camera_setup`.
            Transform::from_xyz(cam_setup.cam_center.x, cam_setup.cam_center.y, 0.0),
            RenderLayers::layer(layer.0),
            PaneCameraOf(pane_entity),
        ));
    }
}

/// Keep each pane camera's viewport, order, and transform in sync with
/// its pane's `PaneRect`. Runs every frame; cheap (a few field writes
/// per pane).
pub fn sync_pane_cameras(
    panes: Query<&PaneRect, With<PaneTag>>,
    mut cameras: Query<(&PaneCameraOf, &mut Camera, &mut Transform)>,
    windows: Query<&Window>,
    region: Option<Res<PaneCanvasRegion>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let region = region.as_deref().copied();
    for (owner, mut cam, mut xform) in &mut cameras {
        let Ok(rect) = panes.get(owner.0) else {
            // Pane is gone but the camera still exists for one frame —
            // close handler will reap it. Skip.
            continue;
        };
        let setup = pane_camera_setup(rect, window, region);
        let new_order = pane_camera_order(rect);

        let needs_viewport_update = match &cam.viewport {
            Some(v) => {
                v.physical_position != setup.viewport.physical_position
                    || v.physical_size != setup.viewport.physical_size
            }
            None => true,
        };
        if needs_viewport_update {
            cam.viewport = Some(setup.viewport);
        }
        if cam.order != new_order {
            cam.order = new_order;
        }
        if xform.translation.x != setup.cam_center.x
            || xform.translation.y != setup.cam_center.y
        {
            xform.translation.x = setup.cam_center.x;
            xform.translation.y = setup.cam_center.y;
        }
    }
}

/// Map a pane's `z` (f32) to a camera `order` (isize). Higher z draws
/// over lower z, mirroring the existing world-space transform ordering.
/// Multiplied by 100 to preserve sub-integer z; +1 keeps every pane
/// strictly above the main camera (order 0).
fn pane_camera_order(rect: &PaneRect) -> isize {
    ((rect.z.max(0.0) * 100.0) as isize) + 1
}

/// What the pane camera needs derived from PaneRect + window.
struct PaneCameraSetup {
    viewport: Viewport,
    /// World-space center of the (possibly clamped) viewport. The
    /// camera transform must sit here so the camera's view (a
    /// `viewport_logical × viewport_logical` window centered on this
    /// point) lines up with the screen pixels covered by the viewport
    /// — including the case where the viewport had to be clamped
    /// because the pane runs off the right or bottom of the window.
    cam_center: Vec2,
}

/// Compute the viewport AND camera-center together so they describe
/// the same screen region — derive both from the same clamped
/// logical rect to keep them consistent.
///
/// The viewport covers the FULL pane rect (chrome + content) clamped
/// to the window. The camera's world transform is at the world
/// coordinates of the clamped rect's center — that way pane content
/// at any world position maps to the same screen pixel whether
/// drawn by the main camera or the pane camera, even when the pane
/// extends past the window edge.
fn pane_camera_setup(
    rect: &PaneRect,
    window: &Window,
    region: Option<PaneCanvasRegion>,
) -> PaneCameraSetup {
    let win_w_logical = window.width();
    let win_h_logical = window.height();
    let scale = window.scale_factor();

    // Step 1: pane rect in window logical pixels (top-left origin).
    let pane_left = rect.pos.x;
    let pane_top = rect.pos.y;
    let pane_right = pane_left + rect.size.x;
    let pane_bottom = pane_top + rect.size.y;

    // Region bounds: the host's canvas area (full window if no
    // `PaneCanvasRegion` is set or `active == false`). Pane cameras
    // never render outside this region, so the host's non-pane chrome
    // (sidebar, top bar, etc.) sits visually on top of the canvas.
    let (region_left, region_top, region_right, region_bottom) = match region {
        Some(r) if r.active => (
            r.min.x.max(0.0).min(win_w_logical),
            r.min.y.max(0.0).min(win_h_logical),
            r.max.x.max(0.0).min(win_w_logical),
            r.max.y.max(0.0).min(win_h_logical),
        ),
        _ => (0.0, 0.0, win_w_logical, win_h_logical),
    };

    // Step 2: clamp to the visible region. A pane dragged partly outside
    // the region contributes only its on-region slice to the viewport;
    // the rest just isn't rendered.
    let vis_left = pane_left.clamp(region_left, region_right);
    let vis_top = pane_top.clamp(region_top, region_bottom);
    let vis_right = pane_right.clamp(region_left, region_right);
    let vis_bottom = pane_bottom.clamp(region_top, region_bottom);
    let vis_w = (vis_right - vis_left).max(0.0);
    let vis_h = (vis_bottom - vis_top).max(0.0);

    // Step 3: viewport in physical pixels. wgpu refuses a scissor
    // whose position equals the render-target dimension (must be
    // strictly inside), so clamp the position into [0, target-1] and
    // shrink width/height accordingly. This matters when a pane has
    // been dragged so far off the right/bottom that the visible
    // slice is empty.
    let win_phys_w = (win_w_logical * scale) as u32;
    let win_phys_h = (win_h_logical * scale) as u32;
    let mut phys_x = (vis_left * scale) as u32;
    let mut phys_y = (vis_top * scale) as u32;
    let mut phys_w = (vis_w * scale) as u32;
    let mut phys_h = (vis_h * scale) as u32;
    if phys_x >= win_phys_w {
        phys_x = win_phys_w.saturating_sub(1);
    }
    if phys_y >= win_phys_h {
        phys_y = win_phys_h.saturating_sub(1);
    }
    if phys_x + phys_w > win_phys_w {
        phys_w = win_phys_w - phys_x;
    }
    if phys_y + phys_h > win_phys_h {
        phys_h = win_phys_h - phys_y;
    }
    phys_w = phys_w.max(1);
    phys_h = phys_h.max(1);

    let viewport = Viewport {
        physical_position: UVec2::new(phys_x, phys_y),
        physical_size: UVec2::new(phys_w, phys_h),
        depth: 0.0..1.0,
    };

    // Step 4: camera center in world coordinates. World origin is at
    // the window center (y-up). The CLAMPED viewport's logical center
    // on screen is `((vis_left + vis_right) * 0.5, (vis_top +
    // vis_bottom) * 0.5)`. Map to world by subtracting window center.
    let vis_cx_logical = (vis_left + vis_right) * 0.5;
    let vis_cy_logical = (vis_top + vis_bottom) * 0.5;
    let cam_center = Vec2::new(
        vis_cx_logical - win_w_logical * 0.5,
        win_h_logical * 0.5 - vis_cy_logical,
    );

    PaneCameraSetup {
        viewport,
        cam_center,
    }
}

/// Propagate `RenderLayers::layer(N)` to everything under each pane's
/// content_root so the per-pane camera (and only it) renders that
/// pane's content.
///
/// Two passes per frame:
/// 1. For each newly-spawned pane (Added<PaneLayer>): walk the
///    content_root subtree and stamp the pane's layer on every
///    descendant that doesn't already have a RenderLayers component.
/// 2. For each newly-added child anywhere (Added<ChildOf>): walk up
///    its parent chain looking for an ancestor with `PaneLayer`. If
///    found, stamp the child and its descendants with that pane's
///    layer.
///
/// Pass 2 catches incremental additions kinds make at runtime (the
/// terminal grid pool growing, editor selection highlights, etc.)
/// without requiring those kinds to know about pane layers at all.
///
/// One-frame delay: a kind's children spawned this frame won't have
/// the layer until next frame's PreUpdate. The result is one frame
/// where the content briefly renders via the main camera (unclipped).
/// We accept that as the tradeoff for not blocking spawn on
/// propagation.
pub fn propagate_render_layers(
    new_layers: Query<(&PaneLayer, &crate::PaneChrome), Added<PaneLayer>>,
    new_children: Query<(Entity, &ChildOf), Added<ChildOf>>,
    pane_layers: Query<&PaneLayer>,
    layers_q: Query<&RenderLayers>,
    children_q: Query<&Children>,
    parents_q: Query<&ChildOf>,
    mut commands: Commands,
) {
    // Pass 1: brand-new panes — walk their whole content_root subtree.
    for (pane_layer, chrome) in &new_layers {
        stamp_subtree(
            chrome.content_root,
            pane_layer.0,
            &layers_q,
            &children_q,
            &mut commands,
        );
    }

    // Pass 2: any new child anywhere — inherit from nearest PaneLayer
    // ancestor. Walks UP via ChildOf (PaneLayer is on the pane entity)
    // rather than reading the parent's RenderLayers, because commands
    // we issued earlier this frame haven't applied yet, so the
    // parent's RenderLayers query lookup might miss them.
    for (child, _) in &new_children {
        if layers_q.get(child).is_ok() {
            continue;
        }
        let Some(layer_n) =
            ancestor_pane_layer(child, &pane_layers, &parents_q)
        else {
            continue;
        };
        stamp_subtree(child, layer_n, &layers_q, &children_q, &mut commands);
    }
}

/// Walk up the ChildOf chain from `entity` looking for an ancestor
/// that carries `PaneLayer`. Returns the layer id, or `None` if the
/// chain hits a parentless entity first.
fn ancestor_pane_layer(
    mut entity: Entity,
    pane_layers: &Query<&PaneLayer>,
    parents_q: &Query<&ChildOf>,
) -> Option<usize> {
    // Bound the walk so a pathological cycle (shouldn't happen, but)
    // can't hang the system.
    for _ in 0..256 {
        if let Ok(pl) = pane_layers.get(entity) {
            return Some(pl.0);
        }
        let Ok(parent) = parents_q.get(entity) else {
            return None;
        };
        entity = parent.0;
    }
    None
}

/// Insert `RenderLayers::layer(layer_n)` on `root` and every
/// descendant that doesn't already have a RenderLayers component.
/// Pre-existing RenderLayers are respected — kinds that want
/// something on layer 0 (chrome-like) can opt out by setting their
/// own.
fn stamp_subtree(
    root: Entity,
    layer_n: usize,
    layers_q: &Query<&RenderLayers>,
    children_q: &Query<&Children>,
    commands: &mut Commands,
) {
    let target = RenderLayers::layer(layer_n);
    let mut stack: Vec<Entity> = vec![root];
    while let Some(e) = stack.pop() {
        if layers_q.get(e).is_err() {
            commands.entity(e).insert(target.clone());
        }
        if let Ok(children) = children_q.get(e) {
            stack.extend(children.iter());
        }
    }
}
