//! Right-side poster panel: tool picker, data-color swatches, action footer.
//! All visual choices come from [`Theme`] via [`re_skin_palette`] so swapping
//! the theme re-paints every button, swatch, and divider in place — there is
//! no per-theme spawn pass.

use crate::bridge::Bold;
use crate::inspector::InspectorRoot;
use crate::theme::{Theme, DATA_SLOT_COUNT};
use crate::tool::{ActiveColor, ActiveTool, Tool};
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;

pub struct PalettePlugin;

impl Plugin for PalettePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PaletteDrag>()
            .add_systems(Startup, spawn_palette)
            .add_systems(
                Update,
                (
                    handle_tool_buttons,
                    handle_color_buttons,
                    handle_action_buttons,
                    sync_active_color_to_theme,
                    re_skin_palette,
                    sync_tool_button_visuals,
                    sync_color_button_visuals,
                    sync_user_preset_buttons,
                    start_primitive_drag,
                    update_primitive_drag_ghost,
                    end_primitive_drag,
                ),
            );
    }
}

/// Active drag of a primitive tile from the palette. Ghost follows
/// the cursor; release over an opened node appends the primitive.
#[derive(Resource, Default)]
pub struct PaletteDrag {
    pub active: Option<crate::tool::PrimitiveKind>,
    pub ghost: Option<Entity>,
}

/// Marker on the Node that holds the user-presets buttons.
/// `sync_user_preset_buttons` despawns and respawns its children
/// whenever `PresetLibrary` changes so the section reflects the
/// latest library state.
#[derive(Component)]
pub struct UserPresetsContainer;

fn sync_user_preset_buttons(
    library: Res<crate::ui::PresetLibrary>,
    theme: Res<Theme>,
    containers: Query<(Entity, Option<&Children>), With<UserPresetsContainer>>,
    mut commands: Commands,
) {
    if !library.is_changed() {
        return;
    }
    for (entity, children) in containers.iter() {
        if let Some(children) = children {
            for c in children.iter() {
                commands.entity(c).despawn();
            }
        }
        commands.entity(entity).with_children(|c| {
            for (i, preset) in library.user.iter().enumerate() {
                spawn_tool_button(c, Tool::UserPreset(i), &preset.label, &theme);
            }
        });
    }
}

/// Marker for the palette root node — used to detect "pointer over UI" so we
/// don't place nodes when clicking the palette itself.
#[derive(Component)]
pub struct PaletteRoot;

#[derive(Component)]
struct ToolButton(Tool);

/// Index into `Theme.data` rather than a baked color, so swapping themes
/// repaints the swatches without forgetting the user's selection.
#[derive(Component)]
struct ColorSwatch(usize);

/// One of the bottom-row footer actions.
#[derive(Component, Clone, Copy)]
enum ActionButton {
    Clear,
    NextTheme,
}

/// Tag the parts of the panel chrome whose color comes from the theme so the
/// re-skin system can find them. `PanelBg` paints the panel body, `HeaderBg`
/// the dark header strip, `HeaderTitle` the bottom (accent-colored) line of
/// the title, etc.
#[derive(Component)]
struct PanelBg;
#[derive(Component)]
struct HeaderBg;
#[derive(Component)]
struct HeaderTitle1;
#[derive(Component)]
struct HeaderTitle2;
#[derive(Component)]
struct SectionTitle;
#[derive(Component)]
struct FooterBg;

/// Bevy 0.18 doesn't ship a `letter-spacing` text style, so we approximate
/// the iso50 design's wide-tracked caps by inserting a hair space (`U+200A`)
/// between glyphs. Better than monospaced and visually close enough for
/// header/section labels at small point sizes.
pub fn caps_spaced(s: &str) -> String {
    let upper = s.to_uppercase();
    let mut out = String::with_capacity(upper.len() * 3);
    for (i, ch) in upper.chars().enumerate() {
        if i > 0 {
            out.push('\u{2009}'); // thin space
        }
        out.push(ch);
    }
    out
}

fn spawn_palette(mut commands: Commands, theme: Res<Theme>) {
    let panel_root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(20.0),
                top: Val::Px(20.0),
                bottom: Val::Px(20.0),
                width: Val::Px(264.0),
                flex_direction: FlexDirection::Column,
                border: UiRect::all(Val::Px(1.5)),
                border_radius: BorderRadius::all(Val::Px(10.0)),
                overflow: Overflow::clip(),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.ink),
            Interaction::None,
            PaletteRoot,
            PanelBg,
        ))
        .id();

    commands.entity(panel_root).with_children(|p| {
        // ── Poster header bar ──────────────────────────────────────────────
        p.spawn((
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
                // Rounded only at the top so the header pill matches the
                // panel's outer corners but seams flush with the body below.
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
                Text::new(caps_spaced("LIVING")),
                TextFont {
                    font_size: 24.0,
                    ..default()
                },
                TextColor(theme.paper),
                Bold,
                HeaderTitle1,
            ));
            h.spawn((
                Text::new(caps_spaced("WHITEBOARD")),
                TextFont {
                    font_size: 24.0,
                    ..default()
                },
                TextColor(theme.accent),
                Bold,
                HeaderTitle2,
            ));
        });

        // ── Scrollable body ────────────────────────────────────────────────
        // Body is a vertically-scrolling column. Interaction::None is
        // added so the `scroll_panels_on_wheel` system can detect hover
        // and route mousewheel events to this panel.
        p.spawn((
            Node {
                flex_grow: 1.0,
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
            crate::ui::ScrollPane,
        ))
        .with_children(|body| {
            section(body, "Tools", &theme);
            tool_group(body, &theme, &[
                (Tool::Select, "Select"),
                (Tool::Edge, "Connect"),
            ]);

            section(body, "Emitters", &theme);
            tool_group(body, &theme, &[
                (Tool::Generator, "Generator"),
                (Tool::Client, "Client"),
            ]);

            section(body, "Processors", &theme);
            tool_group(body, &theme, &[
                (Tool::Worker, "Worker"),
                (Tool::Router, "Router"),
                (Tool::Queue, "Queue"),
            ]);

            section(body, "Workflows", &theme);
            tool_group(body, &theme, &[
                (Tool::Steps, "Steps"),
            ]);

            // Step library — primitive instructions. Clicking a
            // tile, then clicking inside an opened node, appends
            // that primitive to the node's program.
            section(body, "Primitives", &theme);
            body.spawn(Node {
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
            .with_children(|g| {
                for kind in crate::tool::PrimitiveKind::all() {
                    spawn_tool_button(g, Tool::Primitive(kind), kind.label(), &theme);
                }
            });

            // "My Presets" — dynamic section, rebuilt by
            // `sync_user_preset_buttons` whenever PresetLibrary
            // changes. We spawn the heading + empty container here
            // so the layout is stable from frame 0.
            section(body, "My Presets", &theme);
            body.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    ..default()
                },
                UserPresetsContainer,
            ));

            section(body, "Terminals", &theme);
            tool_group(body, &theme, &[
                (Tool::Sink, "Sink"),
                (Tool::Probe, "Probe"),
            ]);

            section(body, "Data Palette", &theme);
            body.spawn(Node {
                width: Val::Percent(100.0),
                margin: UiRect::vertical(Val::Px(6.0)),
                column_gap: Val::Px(6.0),
                ..default()
            })
            .with_children(|row| {
                for i in 0..DATA_SLOT_COUNT {
                    row.spawn((
                        Button,
                        Node {
                            flex_grow: 1.0,
                            height: Val::Px(28.0),
                            border: UiRect::all(Val::Px(1.5)),
                            border_radius: BorderRadius::all(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(theme.data[i]),
                        BorderColor::all(theme.ink),
                        ColorSwatch(i),
                    ));
                }
            });

            // Inspector mount — body is rebuilt by `inspector.rs` whenever
            // selection changes. Hidden by default.
            body.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    margin: UiRect::top(Val::Px(4.0)),
                    display: Display::None,
                    ..default()
                },
                InspectorRoot,
            ));
        });

        // ── Footer actions ────────────────────────────────────────────────
        p.spawn((
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
        .with_children(|f| {
            flat_action(f, &theme, "Clear", ActionButton::Clear);
            flat_action(f, &theme, "Theme", ActionButton::NextTheme);
        });
    });
}

fn section(parent: &mut ChildSpawnerCommands, label: &str, theme: &Theme) {
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
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(caps_spaced(label)),
                TextFont {
                    font_size: 10.0,
                    ..default()
                },
                TextColor(theme.ink_soft),
                Bold,
                SectionTitle,
            ));
        });
}

fn tool_group(parent: &mut ChildSpawnerCommands, theme: &Theme, items: &[(Tool, &str)]) {
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
        .with_children(|g| {
            for (tool, label) in items {
                spawn_tool_button(g, *tool, label, theme);
            }
        });
}

/// Map each tool to a Unicode glyph that approximates its design icon. We
/// avoid emoji (color-painted, ignores TextColor); these are all geometric
/// shapes from the BMP that Jost / Avenir Next render in line with their
/// regular weight.
fn tool_glyph(tool: Tool) -> &'static str {
    match tool {
        Tool::Select => "↖",
        Tool::Edge => "↝",
        Tool::Generator => "◉",
        Tool::Client => "☻",
        Tool::Worker => "▣",
        Tool::Router => "⊕",
        Tool::Queue => "▦",
        Tool::Sink => "▽",
        Tool::Probe => "◎",
        Tool::Steps => "≡",
        Tool::UserPreset(_) => "★",
        Tool::Primitive(_) => "∙",
    }
}

fn spawn_tool_button(
    parent: &mut ChildSpawnerCommands,
    tool: Tool,
    label: &str,
    theme: &Theme,
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
            ToolButton(tool),
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(tool_glyph(tool)),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(theme.ink),
            ));
            b.spawn((
                Text::new(caps_spaced(label)),
                TextFont {
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.ink),
                Bold,
            ));
        });
}

fn flat_action(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    action: ActionButton,
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
            action,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(caps_spaced(label)),
                TextFont {
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.ink),
                Bold,
            ));
        });
}

// ---- Interaction systems -------------------------------------------------

fn handle_tool_buttons(
    mut q: Query<(&Interaction, &ToolButton), (Changed<Interaction>, With<Button>)>,
    mut active: ResMut<ActiveTool>,
) {
    for (interaction, btn) in q.iter_mut() {
        if *interaction == Interaction::Pressed {
            // Primitive tiles are drag-only — pressing them starts
            // a drag via `start_primitive_drag`, not a tool switch.
            // Leaves the user's existing tool active so they can
            // return to, e.g., Select after dropping.
            if matches!(btn.0, Tool::Primitive(_)) {
                continue;
            }
            active.0 = btn.0;
        }
    }
}

fn handle_color_buttons(
    q: Query<(&Interaction, &ColorSwatch), (Changed<Interaction>, With<Button>)>,
    mut active: ResMut<ActiveColor>,
    theme: Res<Theme>,
) {
    for (interaction, swatch) in q.iter() {
        if *interaction == Interaction::Pressed {
            active.0 = theme.data[swatch.0];
        }
    }
}

fn handle_action_buttons(
    q: Query<(&Interaction, &ActionButton), (Changed<Interaction>, With<Button>)>,
    mut theme: ResMut<Theme>,
) {
    for (interaction, action) in q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match action {
            ActionButton::Clear => {
                // Wired in a later phase — the current sim has no top-level
                // "clear all" yet. Leaving the button hot so the wiring is
                // obvious when it lands.
            }
            ActionButton::NextTheme => {
                *theme = theme.next();
            }
        }
    }
}

/// Keep `ActiveColor` pointing at slot 0 of whatever the live theme is, so
/// freshly-placed nodes pick up theme-appropriate hues immediately after a
/// theme swap. The user can still override with the swatches.
fn sync_active_color_to_theme(theme: Res<Theme>, mut active: ResMut<ActiveColor>) {
    if theme.is_changed() {
        active.0 = theme.data[0];
    }
}

/// Repaint every theme-dependent UI surface in place when the Theme resource
/// changes. We deliberately re-skin instead of despawn/respawn so user state
/// (active tool, hover state, focus) survives the swap.
#[allow(clippy::too_many_arguments)]
fn re_skin_palette(
    theme: Res<Theme>,
    mut panel_bg: Query<&mut BackgroundColor, (With<PanelBg>, Without<HeaderBg>, Without<FooterBg>)>,
    mut panel_borders: Query<&mut BorderColor, With<PanelBg>>,
    mut header_bg: Query<&mut BackgroundColor, (With<HeaderBg>, Without<PanelBg>, Without<FooterBg>)>,
    mut footer_bg: Query<&mut BackgroundColor, (With<FooterBg>, Without<PanelBg>, Without<HeaderBg>)>,
    mut h1: Query<&mut TextColor, (With<HeaderTitle1>, Without<HeaderTitle2>, Without<SectionTitle>)>,
    mut h2: Query<&mut TextColor, (With<HeaderTitle2>, Without<HeaderTitle1>, Without<SectionTitle>)>,
    mut sec: Query<&mut TextColor, (With<SectionTitle>, Without<HeaderTitle1>, Without<HeaderTitle2>)>,
    mut swatches: Query<(&ColorSwatch, &mut BackgroundColor), (Without<PanelBg>, Without<HeaderBg>, Without<FooterBg>)>,
) {
    if !theme.is_changed() {
        return;
    }
    for mut bg in panel_bg.iter_mut() {
        bg.0 = theme.paper_alt;
    }
    for mut border in panel_borders.iter_mut() {
        *border = BorderColor::all(theme.ink);
    }
    for mut bg in header_bg.iter_mut() {
        bg.0 = theme.ink;
    }
    for mut bg in footer_bg.iter_mut() {
        bg.0 = theme.paper;
    }
    for mut t in h1.iter_mut() {
        t.0 = theme.paper;
    }
    for mut t in h2.iter_mut() {
        t.0 = theme.accent;
    }
    for mut t in sec.iter_mut() {
        t.0 = theme.ink_soft;
    }
    for (swatch, mut bg) in swatches.iter_mut() {
        bg.0 = theme.data[swatch.0];
    }
}

fn sync_tool_button_visuals(
    theme: Res<Theme>,
    active: Res<ActiveTool>,
    mut q: Query<(
        Entity,
        &Interaction,
        &ToolButton,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut TextColor>,
) {
    for (entity, interaction, btn, mut bg, mut border) in q.iter_mut() {
        let is_active = active.0 == btn.0;
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

        for child in children_q.iter_descendants(entity) {
            if let Ok(mut tc) = text_q.get_mut(child) {
                tc.0 = text_c;
            }
        }
    }
}

fn sync_color_button_visuals(
    theme: Res<Theme>,
    active: Res<ActiveColor>,
    mut q: Query<(&ColorSwatch, &mut BorderColor), Without<ToolButton>>,
) {
    for (swatch, mut border) in q.iter_mut() {
        let is_active = color_approx_eq(theme.data[swatch.0], active.0);
        *border = if is_active {
            BorderColor::all(theme.ink)
        } else {
            BorderColor::all(theme.rule)
        };
    }
}

fn color_approx_eq(a: Color, b: Color) -> bool {
    let a = a.to_srgba();
    let b = b.to_srgba();
    (a.red - b.red).abs() < 1e-3
        && (a.green - b.green).abs() < 1e-3
        && (a.blue - b.blue).abs() < 1e-3
}

/// Returns true if the mouse is currently over any UI element (any node whose
/// Interaction is Hovered/Pressed). Used to prevent canvas clicks from firing
/// when the user clicks the palette.
pub fn pointer_over_ui(ui: &Query<&Interaction>) -> bool {
    ui.iter()
        .any(|i| matches!(i, Interaction::Hovered | Interaction::Pressed))
}

/// Marker component on the floating "ghost" tile entity spawned
/// while a palette primitive is being dragged. Its position is
/// updated each frame to track the cursor.
#[derive(Component)]
struct PaletteGhost;

/// Start a palette drag when the user presses on a primitive tile.
/// Suppresses `handle_tool_buttons` so the active tool doesn't
/// change — drag is a one-shot drop gesture, not a tool selector.
fn start_primitive_drag(
    interactions: Query<(&Interaction, &ToolButton), Changed<Interaction>>,
    windows: Query<&Window>,
    theme: Res<Theme>,
    mut drag: ResMut<PaletteDrag>,
    mut commands: Commands,
) {
    for (interaction, btn) in interactions.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Tool::Primitive(kind) = btn.0 else { continue };
        if drag.active.is_some() {
            continue;
        }
        drag.active = Some(kind);
        let Ok(win) = windows.single() else { continue };
        let cursor = win.cursor_position().unwrap_or_default();
        let ghost = commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(cursor.x - 40.0),
                    top: Val::Px(cursor.y - 12.0),
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                    border: UiRect::all(Val::Px(1.5)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(theme.paper),
                BorderColor::all(theme.ink),
                PaletteGhost,
                // High z so ghost floats above the palette itself.
                ZIndex(1000),
            ))
            .with_children(|g| {
                g.spawn((
                    Text::new(kind.label().to_string()),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(theme.ink),
                    Bold,
                ));
            })
            .id();
        drag.ghost = Some(ghost);
    }
}

/// Each frame while a drag is active, move the ghost to follow
/// the cursor.
fn update_primitive_drag_ghost(
    drag: Res<PaletteDrag>,
    windows: Query<&Window>,
    mut ghosts: Query<&mut Node, With<PaletteGhost>>,
) {
    if drag.active.is_none() {
        return;
    }
    let Some(entity) = drag.ghost else { return };
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    if let Ok(mut node) = ghosts.get_mut(entity) {
        node.left = Val::Px(cursor.x - 40.0);
        node.top = Val::Px(cursor.y - 12.0);
    }
}

/// On mouseup during a palette drag, drop the primitive onto the
/// opened node under the cursor (or cancel if not over one).
/// Always despawns the ghost and clears drag state.
fn end_primitive_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cams: Query<
        (&Camera, &GlobalTransform),
        With<crate::camera::MainCamera>,
    >,
    nodes_q: Query<(
        &Transform,
        &crate::nodes::SimNode,
        &crate::bridge::SimNodeRef,
        Option<&crate::nodes::Opened>,
    )>,
    active_color: Res<ActiveColor>,
    mut drag: ResMut<PaletteDrag>,
    mut sim_res: ResMut<crate::bridge::SimResource>,
    mut commands: Commands,
) {
    if !mouse.just_released(MouseButton::Left) {
        return;
    }
    let Some(kind) = drag.active.take() else { return };
    if let Some(g) = drag.ghost.take() {
        commands.entity(g).despawn();
    }

    // Translate cursor into world space and look for an opened-ish
    // node under it. If found, append the primitive.
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };
    let sc = crate::bridge::bevy_to_sim_color(active_color.0);
    for (tf, sn, nref, opened) in nodes_q.iter() {
        let half = sn.size / 2.0;
        let c = tf.translation.truncate();
        if (world.x - c.x).abs() > half.x || (world.y - c.y).abs() > half.y {
            continue;
        }
        let is_openable =
            sn.kind == crate::nodes::NodeKind::Steps || opened.is_some();
        if !is_openable {
            return;
        }
        sim_res
            .0
            .push_instruction(nref.0, kind.default_instruction(sc));
        return;
    }
}
