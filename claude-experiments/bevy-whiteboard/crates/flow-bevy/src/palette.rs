//! Right-side tool palette panel + keyboard hotkeys.
//!
//! The panel is assembled from `poster-ui` primitives — this module owns only
//! the wiring: which tools appear in which section, the click-routing
//! markers, and the keyboard shortcuts. Visual styling / theming belongs to
//! poster-ui.
//!
//! Hotkeys (mirror the panel):
//!   g — Generator     e — toggle Connect
//!   c — Client        space — play/pause
//!   w — Worker        [ / ]  — slow/fast SIM
//!   r — Router        - / =  — slow/fast VISUAL (independent of sim)
//!   q — Queue         .      — step one event
//!   s — Sink          esc    — Select

use bevy::prelude::*;
use flow::Event;
use poster_ui::{
    Theme,
    apply_tool_button_style,
    flat_action,
    panel_footer,
    panel_header,
    panel_scroll_body,
    right_sidebar,
    section,
    spawn_panel_root,
    swatch,
    swatch_row,
    tool_button,
    tool_grid,
    DATA_SLOT_COUNT,
};

use crate::bridge::SimClock;
use crate::examples::{Example, LoadExample};
use crate::gadgets::Kind;
use crate::sim_driver::{SimCommand, SimDriverRes};
use crate::tool::{ActiveSlot, ActiveTool, Tool};

pub struct PalettePlugin;
impl Plugin for PalettePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_palette)
            .add_systems(
                Update,
                (
                    handle_hotkeys,
                    handle_tool_buttons,
                    handle_swatch_buttons,
                    handle_action_buttons,
                    sync_tool_button_visuals,
                    sync_swatch_visuals,
                ),
            );
    }
}

// ────────────────────────────────────────────────────────────
// Markers (domain wiring for poster-ui builders)
// ────────────────────────────────────────────────────────────

/// Marker on every palette tool button. The inner `Tool` lets a `Changed<
/// Interaction>` handler set `ActiveTool`, and integration tests can query
/// `Query<(Entity, &ToolBtn)>` to find the button they want to click.
#[derive(Component, Clone, Copy)]
pub struct ToolBtn(pub Tool);

/// Marker on each colour swatch in the Data Palette row. Carries the slot
/// index so click handlers can resolve it against `Theme::data`.
#[derive(Component, Clone, Copy)]
pub struct ColorSwatch(pub usize);

/// Marker on each footer action button (Clear / Theme).
#[derive(Component, Clone, Copy)]
pub enum ActionBtn {
    Clear,
    NextTheme,
}

// ────────────────────────────────────────────────────────────
// Spawn
// ────────────────────────────────────────────────────────────

fn spawn_palette(mut commands: Commands, theme: Res<Theme>) {
    let panel = spawn_panel_root(&mut commands, &theme, right_sidebar(264.0));

    commands.entity(panel).with_children(|p| {
        panel_header(p, &theme, "FLOW", "WHITEBOARD");

        panel_scroll_body(p, &theme, |body| {
            section(body, &theme, "Tools");
            tool_grid(body, |g| {
                tool_button(g, &theme, "↖", "Select", ToolBtn(Tool::Select));
                tool_button(g, &theme, "↝", "Connect", ToolBtn(Tool::Connect));
            });

            section(body, &theme, "Emitters");
            tool_grid(body, |g| {
                tool_button(g, &theme, Kind::Generator.glyph(),     "Generator",     ToolBtn(Tool::Drop(Kind::Generator)));
                tool_button(g, &theme, Kind::Client.glyph(),        "Client",        ToolBtn(Tool::Drop(Kind::Client)));
                tool_button(g, &theme, Kind::BackoffClient.glyph(), "BackoffClient", ToolBtn(Tool::Drop(Kind::BackoffClient)));
            });

            section(body, &theme, "Processors");
            tool_grid(body, |g| {
                tool_button(g, &theme, Kind::Worker.glyph(), "Worker", ToolBtn(Tool::Drop(Kind::Worker)));
                tool_button(g, &theme, Kind::Router.glyph(), "Router", ToolBtn(Tool::Drop(Kind::Router)));
                tool_button(g, &theme, Kind::Queue.glyph(),  "Queue",  ToolBtn(Tool::Drop(Kind::Queue)));
            });

            section(body, &theme, "Terminals");
            tool_grid(body, |g| {
                tool_button(g, &theme, Kind::Sink.glyph(), "Sink", ToolBtn(Tool::Drop(Kind::Sink)));
                tool_button(g, &theme, "◎", "Probe", ToolBtn(Tool::Probe));
            });

            section(body, &theme, "Data Palette");
            swatch_row(body, |row| {
                for i in 0..DATA_SLOT_COUNT {
                    swatch(row, &theme, theme.data[i], ColorSwatch(i));
                }
            });

            // Examples moved to the hover-reveal dropdown in the
            // top-left corner (see `crate::examples_menu`). Hotkeys
            // 1..4 below still fire `LoadExample` directly.

            // Inspector mount: the inspector module repopulates this
            // node's children whenever the selection changes. Hidden by
            // default (no selection → no contents).
            body.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    margin: UiRect::top(Val::Px(8.0)),
                    ..default()
                },
                crate::inspector::InspectorMount,
            ));

        });

        panel_footer(p, &theme, |f| {
            flat_action(f, &theme, "Clear", ActionBtn::Clear);
            flat_action(f, &theme, "Theme", ActionBtn::NextTheme);
        });
    });

}

// ────────────────────────────────────────────────────────────
// Interaction
// ────────────────────────────────────────────────────────────

fn handle_hotkeys(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut active: ResMut<ActiveTool>,
    mut clock: ResMut<SimClock>,
    mut driver: ResMut<SimDriverRes>,
    mut timeline: ResMut<crate::edges::VisualTimelineRes>,
    mut load: bevy::ecs::message::MessageWriter<LoadExample>,
) {
    // Digit keys map to Example::ALL in order. Matches the glyph shown
    // on each Examples-section button, so the palette doubles as a
    // hotkey hint.
    let digit_keys = [
        KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4,
        KeyCode::Digit5, KeyCode::Digit6, KeyCode::Digit7, KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    for (i, kc) in digit_keys.iter().enumerate() {
        if i >= Example::ALL.len() { break; }
        if keys.just_pressed(*kc) {
            load.write(LoadExample(Example::ALL[i]));
        }
    }
    if keys.just_pressed(KeyCode::KeyG) { active.0 = Tool::Drop(Kind::Generator); }
    if keys.just_pressed(KeyCode::KeyC) { active.0 = Tool::Drop(Kind::Client); }
    if keys.just_pressed(KeyCode::KeyB) { active.0 = Tool::Drop(Kind::BackoffClient); }
    if keys.just_pressed(KeyCode::KeyW) { active.0 = Tool::Drop(Kind::Worker); }
    if keys.just_pressed(KeyCode::KeyR) { active.0 = Tool::Drop(Kind::Router); }
    if keys.just_pressed(KeyCode::KeyQ) { active.0 = Tool::Drop(Kind::Queue); }
    if keys.just_pressed(KeyCode::KeyS) { active.0 = Tool::Drop(Kind::Sink); }
    if keys.just_pressed(KeyCode::KeyP) { active.0 = Tool::Probe; }
    if keys.just_pressed(KeyCode::KeyE) {
        active.0 = match active.0 {
            Tool::Connect => Tool::Select,
            _ => Tool::Connect,
        };
    }
    if keys.just_pressed(KeyCode::Escape) { active.0 = Tool::Select; }
    if keys.just_pressed(KeyCode::Space) { clock.paused = !clock.paused; }
    if keys.just_pressed(KeyCode::BracketLeft)  { clock.multiplier *= 0.5; }
    if keys.just_pressed(KeyCode::BracketRight) { clock.multiplier *= 2.0; }
    // Visual scale — decoupled from sim speed. `-` halves (packets
    // flash quicker); `=` doubles (packets linger longer). Under F12
    // `set_k` only affects future ingestions; already-in-flight
    // packets complete their existing windows naturally.
    if keys.just_pressed(KeyCode::Minus) {
        let k = timeline.0.k * 0.5;
        timeline.0.set_k(k);
    }
    if keys.just_pressed(KeyCode::Equal) {
        let k = timeline.0.k * 2.0;
        timeline.0.set_k(k);
    }
    // Silence unused-param warnings when neither key was pressed.
    let _ = (&time, &driver);
    if keys.just_pressed(KeyCode::Period) {
        driver.0.send_command(SimCommand::new(|sim| step_to_visible(sim)));
    }
    let _ = &mut clock;
}

fn handle_tool_buttons(
    q: Query<(&Interaction, &ToolBtn), (Changed<Interaction>, With<Button>)>,
    mut active: ResMut<ActiveTool>,
) {
    for (interaction, btn) in q.iter() {
        if *interaction == Interaction::Pressed {
            active.0 = btn.0;
        }
    }
}

fn handle_swatch_buttons(
    q: Query<(&Interaction, &ColorSwatch), (Changed<Interaction>, With<Button>)>,
    mut active: ResMut<ActiveSlot>,
) {
    for (interaction, swatch) in q.iter() {
        if *interaction == Interaction::Pressed {
            active.0 = swatch.0;
        }
    }
}

/// Paint the active swatch with an accent-coloured border; others stay at
/// the default ink border. Gives visual feedback for the selected slot.
fn sync_swatch_visuals(
    theme: Res<Theme>,
    active: Res<ActiveSlot>,
    mut q: Query<(&ColorSwatch, &mut BorderColor), With<Button>>,
) {
    for (swatch, mut border) in q.iter_mut() {
        *border = if swatch.0 == active.0 {
            BorderColor::all(theme.accent)
        } else {
            BorderColor::all(theme.ink)
        };
    }
}

fn handle_action_buttons(
    q: Query<(&Interaction, &ActionBtn), (Changed<Interaction>, With<Button>)>,
    mut theme: ResMut<Theme>,
    mut load: bevy::ecs::message::MessageWriter<LoadExample>,
) {
    for (interaction, action) in q.iter() {
        if *interaction != Interaction::Pressed { continue; }
        match action {
            ActionBtn::Clear => {
                // "Clear" is conceptually "load an empty scenario." We
                // don't have a variant for that — load the default
                // instead, which is the thing a user most likely wants
                // to return to after experimenting.
                load.write(LoadExample(Example::ThreeLaneFanout));
            }
            ActionBtn::NextTheme => {
                *theme = theme.next();
            }
        }
    }
}

// handle_example_buttons moved to crate::examples_menu along with the
// ExampleBtn marker.

/// Paint hover / active state onto each tool button each frame. Delegates to
/// `poster_ui::apply_tool_button_style` for the colour choices, then stamps
/// the text-colour onto descendant glyph + label entities.
fn sync_tool_button_visuals(
    theme: Res<Theme>,
    active: Res<ActiveTool>,
    mut q: Query<(Entity, &Interaction, &ToolBtn, &mut BackgroundColor, &mut BorderColor)>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut TextColor>,
) {
    for (entity, interaction, btn, mut bg, mut border) in q.iter_mut() {
        let is_active = active.0 == btn.0;
        let text_c = apply_tool_button_style(&theme, interaction, is_active, &mut bg, &mut border);
        for child in children_q.iter_descendants(entity) {
            if let Ok(mut tc) = text_q.get_mut(child) {
                tc.0 = text_c;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
// `.` step: advance the sim until something visible happens
// ────────────────────────────────────────────────────────────

pub fn step_to_visible(sim: &mut flow::Sim) {
    const MAX_ITERS: usize = 256;
    for _ in 0..MAX_ITERS {
        let Some(next_t) = sim.next_event_time_ns() else { return; };
        let before = sim.log.total_recorded;
        let target = next_t.max(sim.now_ns.saturating_add(1));
        sim.run_until(target);
        let produced = (sim.log.total_recorded - before) as usize;
        if produced == 0 { return; }
        let visible = sim.log.events.iter().rev().take(produced).any(|ev| match ev {
            Event::PacketEmitted { from, to, .. } => from != to,
            Event::NodeSpawned { .. } | Event::NodeDespawned { .. } => true,
            _ => false,
        });
        if visible { return; }
    }
}
