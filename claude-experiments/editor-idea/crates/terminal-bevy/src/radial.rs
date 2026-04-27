//! Radial (pie / marking) menu.
//!
//! Right-click on the canvas opens the menu at the cursor. Items are
//! drawn as **pie wedges** arranged around a central dead-zone, with
//! item 0 always at 12 o'clock and the rest spaced clockwise. Hit
//! detection is by **angle from the menu center**, not by point-in-rect:
//! as long as the cursor is past the dead-zone radius, whichever wedge
//! its angle falls within is the "hovered" item — even far outside the
//! visible disc. That's what makes a real radial menu fast: targets are
//! effectively infinite-extent in the radial direction (Fitts's law),
//! and once you've learned "Terminal = up" you can flick the mouse in
//! that direction without looking.
//!
//! Click anywhere in the dead-zone (or off the menu entirely) to
//! cancel. Esc also cancels.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::asset::RenderAssetUsages;
use bevy::input::keyboard::KeyboardInput;
use bevy::mesh::{Indices, Mesh2d, PrimitiveTopology};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::sprite_render::{ColorMaterial, MeshMaterial2d};
use bevy::text::LineHeight;

use pane_bevy::{InputConsumed, PaneRegistry};

use crate::projects::{NewPaneRequest, PendingActions, Projects, Sidebar};
use crate::MonoFont;

const RADIAL_Z: f32 = 600.0;

/// Radius of the central cancel zone (no wedges, click-to-close).
const INNER_R: f32 = 36.0;
/// Outer radius of the visible wedges. Hit-test extends past this so
/// over-flick still selects.
const OUTER_R: f32 = 132.0;
/// Hit-test tolerance past the visible outer edge.
const HIT_OUTER_R: f32 = 220.0;
/// A small angular gap between adjacent wedges so they read as
/// separate sectors.
const WEDGE_GAP_RAD: f32 = 0.04;

const COLOR_BACKDROP: Color = Color::srgba(0.0, 0.0, 0.0, 0.32);
const COLOR_WEDGE: Color = Color::srgb(0.135, 0.143, 0.162);
const COLOR_WEDGE_HOVER: Color = Color::srgb(0.22, 0.36, 0.58);
const COLOR_DEADZONE: Color = Color::srgb(0.082, 0.087, 0.100);
const COLOR_DEADZONE_RING: Color = Color::srgb(0.235, 0.250, 0.275);
const COLOR_LABEL: Color = Color::srgb(0.84, 0.86, 0.90);
const COLOR_LABEL_HOVER: Color = Color::srgb(0.97, 0.98, 1.0);
const COLOR_ICON: Color = Color::srgb(0.94, 0.95, 0.97);

#[derive(Clone, Debug)]
pub struct RadialItem {
    pub icon: &'static str,
    pub label: &'static str,
    /// Pane kind to spawn when this item is picked.
    pub kind: &'static str,
}

/// Snapshot the registry's radial-eligible kinds in a stable order
/// (alphabetical by kind id) so item indices stay consistent for the
/// duration of one open menu.
fn collect_radial_items(registry: &PaneRegistry) -> Vec<RadialItem> {
    let mut items: Vec<RadialItem> = registry
        .iter()
        .filter_map(|spec| {
            spec.radial_icon.map(|icon| RadialItem {
                icon,
                label: spec.display_name,
                kind: spec.kind,
            })
        })
        .collect();
    items.sort_by_key(|i| i.kind);
    items
}

#[derive(Resource, Default)]
pub struct RadialMenu {
    /// Window-space cursor pos at menu open. None = closed.
    pub center: Option<Vec2>,
    /// Currently-hovered wedge index (None = in dead-zone or off-menu).
    pub hovered: Option<usize>,
    /// Items snapshotted at open time. Their indices stay consistent
    /// for the lifetime of one open menu, even if the registry changes
    /// (e.g. a new pane plugin is added).
    pub items: Vec<RadialItem>,
}

impl RadialMenu {
    fn item_count(&self) -> usize {
        self.items.len()
    }
}

#[derive(Component)]
struct RadialEntity;

pub struct RadialPlugin;

impl Plugin for RadialPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RadialMenu::default())
            .add_systems(
                Update,
                (radial_open_close, radial_hover, radial_render).chain(),
            );
    }
}

// ---------- Wedge geometry ----------

/// Center angle of wedge `idx` for an N-item menu, in window-y-down
/// space. Item 0 is at 12 o'clock (-PI/2), items go clockwise.
fn wedge_center_angle(idx: usize, n: usize) -> f32 {
    -FRAC_PI_2 + TAU * (idx as f32) / (n.max(1) as f32)
}

/// Half-angular-width of each wedge — fills the full ring with a small
/// gap between neighbours.
fn wedge_half_width(n: usize) -> f32 {
    (TAU / n.max(1) as f32) * 0.5 - WEDGE_GAP_RAD * 0.5
}

/// Wrap an angle to (-PI, PI].
fn wrap_pi(mut x: f32) -> f32 {
    while x > PI {
        x -= TAU;
    }
    while x <= -PI {
        x += TAU;
    }
    x
}

/// Convert a cursor offset (cursor - menu_center, window-y-down) to a
/// hovered wedge index. Returns `None` for the dead-zone or far-off.
fn hit_test(local: Vec2, n: usize) -> Option<usize> {
    let dist = local.length();
    if dist < INNER_R || dist > HIT_OUTER_R {
        return None;
    }
    let angle = local.y.atan2(local.x);
    let half = wedge_half_width(n);
    for i in 0..n {
        if wrap_pi(angle - wedge_center_angle(i, n)).abs() <= half {
            return Some(i);
        }
    }
    None
}

// ---------- Input ----------

fn radial_open_close(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut keys: MessageReader<KeyboardInput>,
    sidebar: Res<Sidebar>,
    mut menu: ResMut<RadialMenu>,
    mut consumed: ResMut<InputConsumed>,
    projects: Res<Projects>,
    registry: Res<PaneRegistry>,
    mut pending: ResMut<PendingActions>,
) {
    let Ok(window) = windows.single() else {
        return;
    };

    let mut esc = false;
    for ev in keys.read() {
        if ev.state.is_pressed() && matches!(ev.key_code, KeyCode::Escape) {
            esc = true;
        }
    }
    if esc && menu.center.is_some() {
        menu.center = None;
        menu.hovered = None;
        menu.items.clear();
        return;
    }

    if buttons.just_pressed(MouseButton::Right) {
        if let Some(pt) = window.cursor_position()
            && pt.x >= sidebar.width
        {
            menu.center = Some(pt);
            menu.items = collect_radial_items(&registry);
            menu.hovered = None;
        }
        return;
    }

    if menu.center.is_some() && buttons.just_pressed(MouseButton::Left) {
        if let Some(idx) = menu.hovered {
            if let Some(item) = menu.items.get(idx).cloned() {
                dispatch_pick(&item, menu.center.unwrap(), &projects, &mut pending);
            }
        }
        menu.center = None;
        menu.hovered = None;
        menu.items.clear();
        consumed.0 = true;
    }
}

fn radial_hover(windows: Query<&Window>, mut menu: ResMut<RadialMenu>) {
    let Some(center) = menu.center else { return };
    let Ok(window) = windows.single() else { return };
    let Some(pt) = window.cursor_position() else { return };
    let local = pt - center;
    let n = menu.item_count();
    let new_hover = hit_test(local, n);
    if menu.hovered != new_hover {
        menu.hovered = new_hover;
    }
}

fn dispatch_pick(
    item: &RadialItem,
    center: Vec2,
    projects: &Projects,
    pending: &mut PendingActions,
) {
    let Some(active) = projects.active else {
        return;
    };
    let origin = Vec2::new(center.x - 24.0, center.y - 10.0);
    pending.new_panes.push(NewPaneRequest {
        kind: item.kind,
        project_id: active,
        origin: Some(origin),
        config: serde_json::Value::Null,
    });
}

// ---------- Render ----------

#[derive(Default)]
struct LastRender {
    open: bool,
    hovered: Option<usize>,
    center: Option<Vec2>,
    item_count: usize,
}

fn radial_render(
    mut commands: Commands,
    menu: Res<RadialMenu>,
    windows: Query<&Window>,
    font: Res<MonoFont>,
    existing: Query<Entity, With<RadialEntity>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut last: Local<LastRender>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let win_w = window.width();
    let win_h = window.height();

    let want_open = menu.center.is_some();
    let already_open = existing.iter().next().is_some();
    let signature_changed = last.open != want_open
        || last.hovered != menu.hovered
        || last.center != menu.center
        || last.item_count != menu.item_count();
    if !signature_changed && !(want_open && !already_open) {
        return;
    }

    for e in &existing {
        commands.entity(e).despawn();
    }
    last.open = want_open;
    last.hovered = menu.hovered;
    last.center = menu.center;
    last.item_count = menu.item_count();

    let Some(center) = menu.center else {
        return;
    };
    let center_world = Vec2::new(center.x - win_w * 0.5, win_h * 0.5 - center.y);

    // Modal backdrop.
    commands.spawn((
        RadialEntity,
        Sprite {
            color: COLOR_BACKDROP,
            custom_size: Some(Vec2::new(win_w, win_h)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(-win_w * 0.5, win_h * 0.5, RADIAL_Z - 0.5),
    ));

    let n = menu.item_count().max(1);
    let half_width = wedge_half_width(n);

    // Wedges + their labels.
    for (i, item) in menu.items.iter().take(n).enumerate() {
        let center_angle = wedge_center_angle(i, n);
        // World rotation: window-y-down angle → world-y-up angle is just
        // its negation. We rotate the unit-aligned wedge mesh by this.
        let world_angle = -center_angle;

        let mesh = meshes.add(build_wedge_mesh(half_width, INNER_R, OUTER_R));
        let hovered = menu.hovered == Some(i);
        let mat = materials.add(ColorMaterial::from_color(if hovered {
            COLOR_WEDGE_HOVER
        } else {
            COLOR_WEDGE
        }));
        commands.spawn((
            RadialEntity,
            Mesh2d(mesh),
            MeshMaterial2d(mat),
            Transform {
                translation: Vec3::new(center_world.x, center_world.y, RADIAL_Z + 0.10),
                rotation: Quat::from_rotation_z(world_angle),
                ..default()
            },
        ));

        // Label position: along the wedge bisector, at the wedge's
        // radial midpoint. Computed in window space then mirrored to
        // world space (y flips).
        let label_r = (INNER_R + OUTER_R) * 0.5;
        let win_off = Vec2::new(center_angle.cos(), center_angle.sin()) * label_r;
        let world_off = Vec2::new(win_off.x, -win_off.y);
        let label_color = if hovered {
            COLOR_LABEL_HOVER
        } else {
            COLOR_LABEL
        };
        commands.spawn((
            RadialEntity,
            Text2d::new(item.icon),
            TextFont {
                font: font.0.clone(),
                font_size: 18.0,
                ..default()
            },
            LineHeight::Px(18.0),
            TextColor(COLOR_ICON),
            Anchor::CENTER,
            Transform::from_xyz(
                center_world.x + world_off.x,
                center_world.y + world_off.y + 7.0,
                RADIAL_Z + 0.30,
            ),
        ));
        commands.spawn((
            RadialEntity,
            Text2d::new(item.label),
            TextFont {
                font: font.0.clone(),
                font_size: 11.0,
                ..default()
            },
            LineHeight::Px(11.0),
            TextColor(label_color),
            Anchor::CENTER,
            Transform::from_xyz(
                center_world.x + world_off.x,
                center_world.y + world_off.y - 12.0,
                RADIAL_Z + 0.30,
            ),
        ));
    }

    // Dead-zone disc — `half_angle = PI` makes a full revolution.
    let disc_mesh = meshes.add(build_wedge_mesh(PI, 0.0, INNER_R));
    let disc_mat = materials.add(ColorMaterial::from_color(COLOR_DEADZONE));
    commands.spawn((
        RadialEntity,
        Mesh2d(disc_mesh),
        MeshMaterial2d(disc_mat),
        Transform::from_xyz(center_world.x, center_world.y, RADIAL_Z + 0.20),
    ));

    // Hairline ring around the dead-zone — gives the cancel zone a
    // visible boundary.
    let ring_mesh = meshes.add(build_wedge_mesh(PI, INNER_R - 0.5, INNER_R + 0.5));
    let ring_mat = materials.add(ColorMaterial::from_color(COLOR_DEADZONE_RING));
    commands.spawn((
        RadialEntity,
        Mesh2d(ring_mesh),
        MeshMaterial2d(ring_mat),
        Transform::from_xyz(center_world.x, center_world.y, RADIAL_Z + 0.21),
    ));
}

/// Build a wedge mesh centered on angle 0, sweeping from -half_angle to
/// +half_angle, between inner_r and outer_r. Pass `half_angle = PI` and
/// `inner_r = 0` for a filled disc; `half_angle = PI` with non-zero
/// inner/outer for a closed ring.
fn build_wedge_mesh(half_angle: f32, inner_r: f32, outer_r: f32) -> Mesh {
    // ~one segment per ~6° of arc, minimum 8.
    let segments = (((half_angle * 2.0) / (PI / 32.0)).ceil() as usize).max(8);
    let n = segments + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / segments as f32;
        let theta = -half_angle + 2.0 * half_angle * t;
        let (s, c) = theta.sin_cos();
        positions.push([c * inner_r, s * inner_r, 0.0]);
        positions.push([c * outer_r, s * outer_r, 0.0]);
        normals.push([0.0, 0.0, 1.0]);
        normals.push([0.0, 0.0, 1.0]);
        uvs.push([t, 0.0]);
        uvs.push([t, 1.0]);
    }
    let mut indices: Vec<u32> = Vec::with_capacity(segments * 6);
    for i in 0..segments {
        let base = (i * 2) as u32;
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 3);
        indices.push(base);
        indices.push(base + 3);
        indices.push(base + 2);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
