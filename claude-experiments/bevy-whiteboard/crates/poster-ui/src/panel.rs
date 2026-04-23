//! Right-side (or left-side, or …) poster panel primitives.
//!
//! A complete panel is composed by the consumer from a root node plus three
//! direct children — header, scrollable body, and footer — built with the
//! helpers here. Typical wiring:
//!
//! ```ignore
//! let panel = spawn_panel_root(&mut commands, &theme, right_sidebar(264.0));
//! commands.entity(panel).with_children(|p| {
//!     panel_header(p, &theme, "LIVING", "WHITEBOARD");
//!     panel_scroll_body(p, &theme, |body| {
//!         section(body, &theme, "Tools");
//!         tool_button(body, &theme, "↖", "Select", ToolBtn(Tool::Select));
//!         tool_button(body, &theme, "↝", "Connect", ToolBtn(Tool::Connect));
//!
//!         section(body, &theme, "Data Palette");
//!         swatch_row(body, &theme, |row| {
//!             for i in 0..DATA_SLOT_COUNT {
//!                 swatch(row, &theme, theme.data[i], ColorSwatch(i));
//!             }
//!         });
//!     });
//!     panel_footer(p, &theme, |f| {
//!         flat_action(f, &theme, "Clear", ActionBtn::Clear);
//!         flat_action(f, &theme, "Theme", ActionBtn::NextTheme);
//!     });
//! });
//! ```
//!
//! The `ToolBtn` / `ColorSwatch` / `ActionBtn` marker components are
//! consumer-defined: the library doesn't know about your domain, so it hangs
//! the marker onto the spawned entity and leaves interaction handling to you.
//! See [`apply_tool_button_style`] for a helper that paints hover / active
//! states consistently — call it from your own sync system each frame.
//!
//! The re-skin system for theme swaps lives on [`PanelReskinPlugin`] and
//! handles every marker component exported from this module. If you add your
//! own themed surfaces, write a small system gated on `theme.is_changed()`
//! that updates their colours — the pattern is the same throughout.

use crate::theme::Theme;
use crate::typography::{Bold, Mono, caps_spaced};
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;

pub struct PanelReskinPlugin;

impl Plugin for PanelReskinPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, re_skin_panel);
    }
}

// ────────────────────────────────────────────────────────────
// Re-skin marker components
// ────────────────────────────────────────────────────────────

/// Panel body background surface. Painted `theme.paper_alt` + `theme.ink` border.
#[derive(Component)]
pub struct PanelBg;

/// Dark header strip at the top of a panel. Painted `theme.ink`.
#[derive(Component)]
pub struct HeaderBg;

/// Footer strip at the bottom of a panel. Painted `theme.paper`.
#[derive(Component)]
pub struct FooterBg;

/// Upper line of the panel's header title (paper-colored).
#[derive(Component)]
pub struct HeaderTitle1;

/// Lower line of the panel's header title (accent-colored).
#[derive(Component)]
pub struct HeaderTitle2;

/// Section-divider heading text. Painted `theme.ink_soft`.
#[derive(Component)]
pub struct SectionTitle;

/// Marker on a `BorderColor` that should track `theme.rule`. Used on the
/// under-line beneath a section heading.
#[derive(Component)]
pub struct RuleBorder;

// ────────────────────────────────────────────────────────────
// Layout helpers
// ────────────────────────────────────────────────────────────

/// Node-layout preset: full-height sidebar attached to the right edge, with
/// symmetric 20px insets. Pass into [`spawn_panel_root`].
pub fn right_sidebar(width: f32) -> Node {
    Node {
        position_type: PositionType::Absolute,
        right: Val::Px(20.0),
        top: Val::Px(20.0),
        bottom: Val::Px(20.0),
        width: Val::Px(width),
        flex_direction: FlexDirection::Column,
        ..default()
    }
}

/// Full-height sidebar attached to the left edge.
pub fn left_sidebar(width: f32) -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: Val::Px(20.0),
        top: Val::Px(20.0),
        bottom: Val::Px(20.0),
        width: Val::Px(width),
        flex_direction: FlexDirection::Column,
        ..default()
    }
}

/// Spawn the panel root entity — the outer container that holds the header,
/// body, and footer as children. Hand the consumer-chosen layout in as a
/// `Node`; we overlay the bordered-pill styling on top.
pub fn spawn_panel_root(
    commands: &mut Commands,
    theme: &Theme,
    layout: Node,
) -> Entity {
    let mut layout = layout;
    layout.border = UiRect::all(Val::Px(1.5));
    layout.border_radius = BorderRadius::all(Val::Px(10.0));
    layout.overflow = Overflow::clip();
    commands
        .spawn((
            layout,
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.ink),
            // Interaction::None lets the scroll-pane router detect cursor-in-
            // bounds via the panel's own hit-testing. Buttons inside take
            // precedence on click, this only matters for hover / scroll.
            Interaction::None,
            PanelBg,
        ))
        .id()
}

// ────────────────────────────────────────────────────────────
// Header / Body / Footer
// ────────────────────────────────────────────────────────────

/// Dark header strip with a two-line title: the top line in `theme.paper`,
/// the bottom in `theme.accent`. Caps are auto-tracked via [`caps_spaced`].
pub fn panel_header(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    title_top: &str,
    title_bottom: &str,
) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect {
                    left: Val::Px(16.0),
                    right: Val::Px(16.0),
                    top: Val::Px(16.0),
                    bottom: Val::Px(14.0),
                },
                flex_direction: FlexDirection::Column,
                border: UiRect::bottom(Val::Px(1.5)),
                border_radius: BorderRadius {
                    top_left: Val::Px(10.0),
                    top_right: Val::Px(10.0),
                    bottom_left: Val::Px(0.0),
                    bottom_right: Val::Px(0.0),
                },
                ..default()
            },
            BackgroundColor(theme.ink),
            BorderColor::all(theme.ink),
            HeaderBg,
        ))
        .with_children(|h| {
            h.spawn((
                Text::new(caps_spaced(title_top)),
                TextFont { font_size: 24.0, ..default() },
                TextColor(theme.paper),
                Bold,
                HeaderTitle1,
            ));
            h.spawn((
                Text::new(caps_spaced(title_bottom)),
                TextFont { font_size: 24.0, ..default() },
                TextColor(theme.accent),
                Bold,
                HeaderTitle2,
            ));
        });
}

/// Scrollable panel body. Pass a closure that populates the body with
/// sections, tool buttons, swatch rows, etc. The body itself is a vertical
/// flex column with mousewheel scrolling (via [`crate::scroll::ScrollPane`]).
pub fn panel_scroll_body(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    body_fn: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            Node {
                flex_grow: 1.0,
                // `min_height: 0` overrides flexbox's default
                // `min-size: auto`, which otherwise refuses to shrink
                // the body below its content size — that prevents
                // the body from ever *having* overflow, so the
                // scroll container never scrolls. With this set, the
                // body claims only the space flex allots it and the
                // rest becomes scroll overflow.
                min_height: Val::Px(0.0),
                width: Val::Percent(100.0),
                padding: UiRect::all(Val::Px(12.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            Interaction::None,
            ScrollPosition::default(),
            crate::scroll::ScrollPane,
            PanelBodyBg,
        ))
        .with_children(body_fn);
}

/// Marker for scrollable panel body (re-skinned to `theme.paper_alt`).
#[derive(Component)]
pub struct PanelBodyBg;

/// Light footer strip rounded at the bottom. Pass a closure to fill it — the
/// typical shape is a horizontal row of [`flat_action`] buttons.
pub fn panel_footer(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    footer_fn: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect::all(Val::Px(10.0)),
                column_gap: Val::Px(6.0),
                border: UiRect::top(Val::Px(1.5)),
                border_radius: BorderRadius {
                    top_left: Val::Px(0.0),
                    top_right: Val::Px(0.0),
                    bottom_left: Val::Px(10.0),
                    bottom_right: Val::Px(10.0),
                },
                ..default()
            },
            BackgroundColor(theme.paper),
            BorderColor::all(theme.ink),
            FooterBg,
        ))
        .with_children(footer_fn);
}

// ────────────────────────────────────────────────────────────
// Body primitives
// ────────────────────────────────────────────────────────────

/// Section-divider heading: a tracked-caps label with a thin underline.
pub fn section(parent: &mut ChildSpawnerCommands, theme: &Theme, label: &str) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect {
                    top: Val::Px(12.0),
                    bottom: Val::Px(6.0),
                    left: Val::Px(4.0),
                    right: Val::Px(4.0),
                },
                margin: UiRect::bottom(Val::Px(6.0)),
                border: UiRect::bottom(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(theme.rule),
            RuleBorder,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(caps_spaced(label)),
                TextFont { font_size: 10.0, ..default() },
                TextColor(theme.ink_soft),
                Bold,
                SectionTitle,
            ));
        });
}

/// Two-column grid wrapper; tool buttons compose naturally inside. Use this
/// to group related tools (emitters, processors, terminals, …).
pub fn tool_grid(
    parent: &mut ChildSpawnerCommands,
    body_fn: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            display: Display::Grid,
            grid_template_columns: vec![
                RepeatedGridTrack::flex(1, 1.0),
                RepeatedGridTrack::flex(1, 1.0),
            ],
            column_gap: Val::Px(4.0),
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(body_fn);
}

/// A rounded-rect button with a leading glyph + label. The `extra` bundle is
/// whatever marker / data components the consumer wants attached (typically a
/// `ToolBtn(Tool)`-style newtype so their sync system can hit-test).
///
/// Style is inert at spawn: add [`apply_tool_button_style`] to your per-frame
/// sync system to paint hover / active states.
pub fn tool_button(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    glyph: &str,
    label: &str,
    extra: impl Bundle,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(36.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::FlexStart,
                padding: UiRect::horizontal(Val::Px(10.0)),
                column_gap: Val::Px(8.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(theme.rule),
            extra,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(glyph.to_string()),
                TextFont { font_size: 15.0, ..default() },
                TextColor(theme.ink),
            ));
            b.spawn((
                Text::new(caps_spaced(label)),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Bold,
            ));
        });
}

/// Paint a tool button's hover / active state. Call from your own sync system
/// each frame, resolving `is_active` from whatever "current tool" resource
/// you maintain. Also recolours descendant text (glyph + label) to match the
/// active/hovered/resting foreground.
pub fn apply_tool_button_style(
    theme: &Theme,
    interaction: &Interaction,
    is_active: bool,
    bg: &mut BackgroundColor,
    border: &mut BorderColor,
) -> Color {
    let hovered = matches!(interaction, Interaction::Hovered | Interaction::Pressed);
    let (bg_c, border_c, text_c) = if is_active {
        (theme.ink, theme.ink, theme.paper)
    } else if hovered {
        (theme.paper, theme.ink, theme.ink)
    } else {
        (Color::NONE, theme.rule, theme.ink)
    };
    bg.0 = bg_c;
    *border = BorderColor::all(border_c);
    text_c
}

/// Horizontal row of colour swatches. Use inside a scroll body to build a
/// "Data Palette" row; fill with [`swatch`] calls.
pub fn swatch_row(
    parent: &mut ChildSpawnerCommands,
    body_fn: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            margin: UiRect::vertical(Val::Px(6.0)),
            column_gap: Val::Px(6.0),
            ..default()
        })
        .with_children(body_fn);
}

/// One colour swatch — a flex-grow cell that paints the given colour. The
/// `extra` bundle typically carries a `ColorSwatch(index)` marker so your
/// click handler can resolve which swatch was pressed.
pub fn swatch(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    color: Color,
    extra: impl Bundle,
) {
    parent.spawn((
        Button,
        Node {
            flex_grow: 1.0,
            height: Val::Px(28.0),
            border: UiRect::all(Val::Px(1.5)),
            border_radius: BorderRadius::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(color),
        BorderColor::all(theme.ink),
        extra,
    ));
}

/// Footer button: rectangular, ink-bordered, caps label. Meant for actions
/// like "Clear" or "Theme".
pub fn flat_action(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    extra: impl Bundle,
) {
    parent
        .spawn((
            Button,
            Node {
                flex_grow: 1.0,
                padding: UiRect::vertical(Val::Px(8.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(theme.ink),
            extra,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(caps_spaced(label)),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Bold,
            ));
        });
}

/// Small square-ish icon button for row actions (move-up, delete ×, etc.).
pub fn icon_button(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    glyph: &str,
    extra: impl Bundle,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(20.0),
                height: Val::Px(20.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(theme.rule),
            extra,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(glyph.to_string()),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Bold,
            ));
        });
}

/// `label · control` row with consistent spacing and styling. Used for
/// inspector rows, slider rows, kv rows. `control` supplies the right-hand
/// widget; `actions` hangs optional trailing buttons (move-up, delete ×, …).
pub fn row(parent: &mut ChildSpawnerCommands, theme: &Theme, label: &str, value: &str) {
    row_with_actions(parent, theme, label, value, |_| {});
}

/// Same as [`row`] but with a closure for trailing action widgets.
pub fn row_with_actions<F>(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    value: &str,
    actions: F,
) where
    F: FnOnce(&mut ChildSpawnerCommands),
{
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(5.0)),
            column_gap: Val::Px(6.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|r| {
            r.spawn((
                Text::new(caps_spaced(label)),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                Bold,
            ));
            r.spawn((
                Text::new(value.to_string()),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Bold,
                Mono,
            ));
            actions(r);
        });
}

// ────────────────────────────────────────────────────────────
// Hit-test helper
// ────────────────────────────────────────────────────────────

/// Returns `true` if the pointer is currently over any UI node with an
/// `Interaction` component — i.e. a canvas click at the same instant is
/// really a UI click and should be ignored by your world-space handlers.
///
/// Typical use inside a canvas click handler:
/// ```ignore
/// fn drop_on_canvas(ui: Query<&Interaction>, ...) {
///     if poster_ui::pointer_over_ui(&ui) { return; }
///     // ... spawn a node in world space
/// }
/// ```
pub fn pointer_over_ui(ui: &Query<&Interaction>) -> bool {
    ui.iter()
        .any(|i| matches!(i, Interaction::Hovered | Interaction::Pressed))
}

// ────────────────────────────────────────────────────────────
// Formatting helpers
// ────────────────────────────────────────────────────────────

pub const NS_PER_US: u64 = 1_000;
pub const NS_PER_MS: u64 = 1_000_000;
pub const NS_PER_S: u64 = 1_000_000_000;

/// Human-friendly duration label (`"500ms"`, `"1.2s"`, `"25us"`, …). Use in
/// row controls and canvas labels alike so units stay consistent.
pub fn fmt_duration(ns: u64) -> String {
    if ns >= NS_PER_S {
        format!("{:.1}s", ns as f64 / NS_PER_S as f64)
    } else if ns >= NS_PER_MS {
        format!("{}ms", ns / NS_PER_MS)
    } else if ns >= NS_PER_US {
        format!("{}us", ns / NS_PER_US)
    } else {
        format!("{}ns", ns)
    }
}

// ────────────────────────────────────────────────────────────
// Re-skin system
// ────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn re_skin_panel(
    theme: Res<Theme>,
    mut panel_bg: Query<
        &mut BackgroundColor,
        (
            With<PanelBg>,
            Without<HeaderBg>,
            Without<FooterBg>,
            Without<PanelBodyBg>,
        ),
    >,
    mut panel_border: Query<&mut BorderColor, With<PanelBg>>,
    mut body_bg: Query<
        &mut BackgroundColor,
        (With<PanelBodyBg>, Without<PanelBg>, Without<HeaderBg>, Without<FooterBg>),
    >,
    mut header_bg: Query<
        &mut BackgroundColor,
        (With<HeaderBg>, Without<PanelBg>, Without<FooterBg>, Without<PanelBodyBg>),
    >,
    mut header_border: Query<&mut BorderColor, (With<HeaderBg>, Without<PanelBg>, Without<RuleBorder>)>,
    mut footer_bg: Query<
        &mut BackgroundColor,
        (With<FooterBg>, Without<PanelBg>, Without<HeaderBg>, Without<PanelBodyBg>),
    >,
    mut footer_border: Query<&mut BorderColor, (With<FooterBg>, Without<PanelBg>, Without<HeaderBg>, Without<RuleBorder>)>,
    mut rule_borders: Query<&mut BorderColor, (With<RuleBorder>, Without<PanelBg>, Without<HeaderBg>, Without<FooterBg>)>,
    mut h1: Query<&mut TextColor, (With<HeaderTitle1>, Without<HeaderTitle2>, Without<SectionTitle>)>,
    mut h2: Query<&mut TextColor, (With<HeaderTitle2>, Without<HeaderTitle1>, Without<SectionTitle>)>,
    mut sec: Query<&mut TextColor, (With<SectionTitle>, Without<HeaderTitle1>, Without<HeaderTitle2>)>,
) {
    if !theme.is_changed() {
        return;
    }
    for mut bg in panel_bg.iter_mut() { bg.0 = theme.paper_alt; }
    for mut b in panel_border.iter_mut() { *b = BorderColor::all(theme.ink); }
    for mut bg in body_bg.iter_mut() { bg.0 = theme.paper_alt; }
    for mut bg in header_bg.iter_mut() { bg.0 = theme.ink; }
    for mut b in header_border.iter_mut() { *b = BorderColor::all(theme.ink); }
    for mut bg in footer_bg.iter_mut() { bg.0 = theme.paper; }
    for mut b in footer_border.iter_mut() { *b = BorderColor::all(theme.ink); }
    for mut b in rule_borders.iter_mut() { *b = BorderColor::all(theme.rule); }
    for mut t in h1.iter_mut() { t.0 = theme.paper; }
    for mut t in h2.iter_mut() { t.0 = theme.accent; }
    for mut t in sec.iter_mut() { t.0 = theme.ink_soft; }
}
