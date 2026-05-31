//! Per-pane right-click context menu.
//!
//! Right-click that lands on a pane opens a small vertical list of
//! actions (Pin/Unpin, Close). Right-click that misses every pane
//! falls through to the radial spawn menu in [`crate::radial`]. The
//! menu consumes [`InputConsumed`] on open and on item-pick so the
//! pane-mouse handler and the radial open-handler don't also act on
//! the same press/release.
//!
//! The menu is rendered as a couple of sprites + Text2d entities on a
//! dedicated z above the radial backdrop so it always sits on top.

use bevy::camera::visibility::RenderLayers;
use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::LineHeight;

use pane_bevy::{
    topmost_pane_at, InputConsumed, PaneRect, PaneTag, PanePinned, PendingPaneActions,
};

use crate::projects::{Projects, Sidebar};
use crate::MonoFont;

/// Above the radial menu's RADIAL_Z (=600) so a context menu opened on
/// a pane never sits behind a wedge.
const MENU_Z: f32 = 700.0;

const ROW_H: f32 = 24.0;
const ROW_PAD_X: f32 = 12.0;
const MENU_W: f32 = 140.0;
const MENU_PAD_Y: f32 = 4.0;
const FONT_SIZE: f32 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextAction {
    Pin,
    Unpin,
    Close,
}

impl ContextAction {
    fn label(self) -> &'static str {
        match self {
            ContextAction::Pin => "Pin to background",
            ContextAction::Unpin => "Unpin",
            ContextAction::Close => "Close",
        }
    }
}

#[derive(Resource, Default)]
pub struct ContextMenu {
    /// Window-space top-left of the menu (None = closed).
    pub origin: Option<Vec2>,
    pub target: Option<Entity>,
    pub items: Vec<ContextAction>,
    pub hovered: Option<usize>,
}

impl ContextMenu {
    fn close(&mut self) {
        self.origin = None;
        self.target = None;
        self.items.clear();
        self.hovered = None;
    }
}

#[derive(Component)]
struct ContextMenuEntity;

pub struct ContextMenuPlugin;

impl Plugin for ContextMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ContextMenu>().add_systems(
            Update,
            (
                // MUST run before radial::radial_open_close so it can
                // set `InputConsumed` on right-click-on-pane and the
                // radial sees that flag and stays closed.
                context_open_close.before(crate::radial::radial_open_close),
                context_hover,
                context_render,
            )
                .chain(),
        );
    }
}

fn context_open_close(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut keys: MessageReader<KeyboardInput>,
    sidebar: Res<Sidebar>,
    viewport: Res<pane_bevy::PaneViewport>,
    mut menu: ResMut<ContextMenu>,
    mut consumed: ResMut<InputConsumed>,
    panes: Query<(Entity, &PaneRect, &Visibility, Has<PanePinned>), With<PaneTag>>,
    mut pending: ResMut<PendingPaneActions>,
    _projects: Res<Projects>,
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
    if esc && menu.origin.is_some() {
        menu.close();
        return;
    }

    if buttons.just_pressed(MouseButton::Right) {
        // Close any previously open menu before considering a re-open.
        let was_open = menu.origin.is_some();
        if was_open {
            menu.close();
        }
        let Some(pt) = window.cursor_position() else {
            return;
        };
        if pt.x < sidebar.width {
            return;
        }
        // PaneRect lives in canvas-space; convert the cursor into the
        // same frame before hit-testing, otherwise panning/zooming the
        // canvas makes the radial menu open on top of visible panes.
        let pt_canvas = viewport.window_to_canvas(pt);
        // Only consider visible panes; include pinned so the user can
        // right-click them to unpin.
        let visible: Vec<(Entity, PaneRect, bool)> = panes
            .iter()
            .filter(|(_, _, vis, _)| !matches!(vis, Visibility::Hidden))
            .map(|(e, r, _, pinned)| (e, *r, pinned))
            .collect();
        // First try to hit an unpinned pane (they sit on top); fall
        // back to pinned. Reuses topmost_pane_at's z-aware hit-test.
        let unpinned_rects: Vec<(Entity, PaneRect)> = visible
            .iter()
            .filter(|(_, _, pinned)| !pinned)
            .map(|(e, r, _)| (*e, *r))
            .collect();
        let target = topmost_pane_at(pt_canvas, &unpinned_rects).or_else(|| {
            let pinned_rects: Vec<(Entity, PaneRect)> = visible
                .iter()
                .filter(|(_, _, pinned)| *pinned)
                .map(|(e, r, _)| (*e, *r))
                .collect();
            topmost_pane_at(pt_canvas, &pinned_rects)
        });
        let Some(target) = target else {
            // Miss every pane — let the radial menu handle the click.
            return;
        };
        let is_pinned = visible
            .iter()
            .find(|(e, _, _)| *e == target)
            .map(|(_, _, p)| *p)
            .unwrap_or(false);
        let items = if is_pinned {
            vec![ContextAction::Unpin, ContextAction::Close]
        } else {
            vec![ContextAction::Pin, ContextAction::Close]
        };
        menu.origin = Some(pt);
        menu.target = Some(target);
        menu.items = items;
        menu.hovered = None;
        // Suppress the radial open + pane left-click for this frame.
        consumed.0 = true;
        return;
    }

    if menu.origin.is_some() && buttons.just_pressed(MouseButton::Left) {
        let pick = menu.hovered.and_then(|i| menu.items.get(i).copied());
        let target = menu.target;
        menu.close();
        // Click on the menu itself counts as "consumed" so the pane
        // beneath doesn't focus / drag on the same release.
        consumed.0 = true;
        match (pick, target) {
            (Some(ContextAction::Pin), Some(e)) => pending.pin.push(e),
            (Some(ContextAction::Unpin), Some(e)) => pending.unpin.push(e),
            (Some(ContextAction::Close), Some(e)) => pending.close.push(e),
            _ => {}
        }
    }
}

fn context_hover(windows: Query<&Window>, mut menu: ResMut<ContextMenu>) {
    let Some(origin) = menu.origin else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    let menu_h = menu.items.len() as f32 * ROW_H + 2.0 * MENU_PAD_Y;
    let in_menu = pt.x >= origin.x
        && pt.x <= origin.x + MENU_W
        && pt.y >= origin.y
        && pt.y <= origin.y + menu_h;
    let new_hover = if !in_menu {
        None
    } else {
        let local_y = pt.y - origin.y - MENU_PAD_Y;
        let idx = (local_y / ROW_H).floor() as i32;
        if idx < 0 || idx as usize >= menu.items.len() {
            None
        } else {
            Some(idx as usize)
        }
    };
    if menu.hovered != new_hover {
        menu.hovered = new_hover;
    }
}

#[derive(Default)]
struct LastRender {
    open: bool,
    hovered: Option<usize>,
    origin: Option<Vec2>,
    item_count: usize,
}

fn context_render(
    mut commands: Commands,
    menu: Res<ContextMenu>,
    windows: Query<&Window>,
    font: Res<MonoFont>,
    theme: Res<style_bevy::Theme>,
    existing: Query<Entity, With<ContextMenuEntity>>,
    mut last: Local<LastRender>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let win_w = window.width();
    let win_h = window.height();

    let want_open = menu.origin.is_some();
    let already_open = existing.iter().next().is_some();
    let sig_changed = last.open != want_open
        || last.hovered != menu.hovered
        || last.origin != menu.origin
        || last.item_count != menu.items.len()
        || theme.is_changed();
    if !sig_changed && !(want_open && !already_open) {
        return;
    }
    for e in &existing {
        commands.entity(e).despawn();
    }
    last.open = want_open;
    last.hovered = menu.hovered;
    last.origin = menu.origin;
    last.item_count = menu.items.len();

    let Some(origin) = menu.origin else {
        return;
    };

    use style_bevy::tokens as t;
    let c = |id| Color::LinearRgba(theme.color(id));
    let bg = c(t::PANE_BG);
    let row_hover = c(t::SIDEBAR_ROW_ACTIVE_BG);
    let text = c(t::FG);
    let text_hover = c(t::FG);
    let border = c(t::CHROME_DIVIDER);

    let menu_h = menu.items.len() as f32 * ROW_H + 2.0 * MENU_PAD_Y;

    // Window-space (top-left, y-down) → world-space (center, y-up).
    let to_world = |p: Vec2| Vec2::new(p.x - win_w * 0.5, win_h * 0.5 - p.y);

    let menu_world_tl = to_world(origin);
    let overlay = RenderLayers::layer(crate::MENU_OVERLAY_LAYER);

    // Border / drop sprite (1px ring via slightly-larger sprite behind).
    commands.spawn((
        ContextMenuEntity,
        Sprite {
            color: border,
            custom_size: Some(Vec2::new(MENU_W + 2.0, menu_h + 2.0)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(menu_world_tl.x - 1.0, menu_world_tl.y + 1.0, MENU_Z),
        overlay.clone(),
    ));

    // Background.
    commands.spawn((
        ContextMenuEntity,
        Sprite {
            color: bg,
            custom_size: Some(Vec2::new(MENU_W, menu_h)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(menu_world_tl.x, menu_world_tl.y, MENU_Z + 0.10),
        overlay.clone(),
    ));

    for (i, action) in menu.items.iter().enumerate() {
        let row_top_window = origin + Vec2::new(0.0, MENU_PAD_Y + (i as f32) * ROW_H);
        let row_world_tl = to_world(row_top_window);
        let hovered = menu.hovered == Some(i);
        if hovered {
            commands.spawn((
                ContextMenuEntity,
                Sprite {
                    color: row_hover,
                    custom_size: Some(Vec2::new(MENU_W, ROW_H)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(row_world_tl.x, row_world_tl.y, MENU_Z + 0.20),
                overlay.clone(),
            ));
        }
        let label_color = if hovered { text_hover } else { text };
        commands.spawn((
            ContextMenuEntity,
            Text2d::new(action.label()),
            TextFont {
                font: font.0.clone(),
                font_size: FONT_SIZE,
                ..default()
            },
            LineHeight::Px(ROW_H),
            TextColor(label_color),
            Anchor::CENTER_LEFT,
            Transform::from_xyz(
                row_world_tl.x + ROW_PAD_X,
                row_world_tl.y - ROW_H * 0.5,
                MENU_Z + 0.30,
            ),
            overlay.clone(),
        ));
    }
}
