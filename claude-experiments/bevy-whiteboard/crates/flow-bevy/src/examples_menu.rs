//! Hover-reveal dropdown in the top-left corner that lists the
//! pre-built example scenarios. Clicking one fires a `LoadExample`
//! message; the canvas wipes and rebuilds.
//!
//! UX:
//!   - A small always-visible trigger pill sits 20px in from the
//!     top-left edge.
//!   - Hovering it (or any button in the dropdown) makes the
//!     dropdown visible.
//!   - A short grace period (~150ms) keeps the dropdown up when
//!     the cursor briefly crosses a gap — avoids flicker as the
//!     user moves from trigger → dropdown row.
//!
//! The dropdown is a sibling of the trigger, absolutely positioned
//! just beneath it, so toggling its visibility doesn't affect any
//! other layout.

use bevy::prelude::*;
use poster_ui::{Bold, Theme, caps_spaced};

use crate::examples::{Example, LoadExample};

/// Milliseconds the dropdown stays visible after the last time
/// either the trigger or any dropdown button was hovered. Keeps
/// the menu up long enough to move the cursor between elements
/// without closing on the gap between them.
const HOVER_GRACE_MS: f32 = 150.0;

pub struct ExamplesMenuPlugin;
impl Plugin for ExamplesMenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DropdownState>()
            .add_systems(Startup, spawn_examples_menu)
            .add_systems(
                Update,
                (
                    track_hover,
                    update_dropdown_visibility,
                    update_trigger_visibility,
                    handle_example_buttons,
                    style_example_buttons,
                ),
            );
    }
}

// ────────────────────────────────────────────────────────────
// Markers
// ────────────────────────────────────────────────────────────

#[derive(Component)]
struct ExamplesTrigger;

/// Marker on the `Text` child of the trigger so the
/// `update_trigger_visibility` system can dim/restore its alpha
/// alongside the trigger background.
#[derive(Component)]
struct ExamplesTriggerLabel;

#[derive(Component)]
struct ExamplesDropdown;

#[derive(Component, Clone, Copy)]
struct ExampleBtn(Example);

#[derive(Resource, Default)]
struct DropdownState {
    /// Real-time seconds at the last observed hover on trigger or
    /// any dropdown button. Compared against the grace threshold.
    last_hover_secs: f32,
    /// Whether we're currently showing the dropdown. Tracked so we
    /// only flip `Visibility` when the desired state actually
    /// changes (Visibility::set_if_neq isn't available on Bevy 0.18's
    /// Visibility enum).
    open: bool,
}

// ────────────────────────────────────────────────────────────
// Spawn
// ────────────────────────────────────────────────────────────

fn spawn_examples_menu(mut commands: Commands, theme: Res<Theme>) {
    // Trigger pill at top-left. Absolutely positioned so it floats
    // over the canvas and doesn't compete with anything for layout.
    commands.spawn((
        Button,
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(20.0),
            top: Val::Px(20.0),
            height: Val::Px(28.0),
            padding: UiRect::horizontal(Val::Px(12.0)),
            align_items: AlignItems::Center,
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            ..default()
        },
        // Start fully transparent: the user only wants the trigger
        // to surface when their cursor is on it. The
        // `update_trigger_visibility` system swaps these to
        // `theme.paper_alt` / `theme.ink` once a hover is detected.
        BackgroundColor(Color::NONE),
        BorderColor::all(Color::NONE),
        ZIndex(500),
        ExamplesTrigger,
    ))
    .with_children(|p| {
        p.spawn((
            Text::new(caps_spaced("Examples")),
            TextFont { font_size: 11.0, ..default() },
            // Same idea — text is invisible by default; the
            // visibility system fades it in on hover.
            TextColor(Color::NONE),
            Bold,
            ExamplesTriggerLabel,
        ));
    });

    // Dropdown panel, positioned right under the trigger. Starts
    // hidden. We align left-edge to 20px like the trigger and push
    // down to sit just below it (28px trigger height + 20px top + 4px
    // gap = 52px).
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(20.0),
            top: Val::Px(52.0),
            width: Val::Px(220.0),
            padding: UiRect::all(Val::Px(6.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(2.0),
            border: UiRect::all(Val::Px(1.5)),
            border_radius: BorderRadius::all(Val::Px(8.0)),
            ..default()
        },
        BackgroundColor(theme.paper_alt),
        BorderColor::all(theme.ink),
        Visibility::Hidden,
        ZIndex(500),
        ExamplesDropdown,
    ))
    .with_children(|p| {
        for (i, ex) in Example::ALL.iter().enumerate() {
            let key = format!("{}", i + 1);
            spawn_example_row(p, &theme, &key, *ex);
        }
    });
}

fn spawn_example_row(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    hotkey: &str,
    ex: Example,
) {
    parent.spawn((
        Button,
        Node {
            width: Val::Percent(100.0),
            height: Val::Px(30.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::FlexStart,
            padding: UiRect::horizontal(Val::Px(10.0)),
            column_gap: Val::Px(10.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(5.0)),
            ..default()
        },
        BackgroundColor(Color::NONE),
        BorderColor::all(theme.rule),
        ExampleBtn(ex),
    ))
    .with_children(|b| {
        b.spawn((
            Text::new(hotkey.to_string()),
            TextFont { font_size: 13.0, ..default() },
            TextColor(theme.ink_soft),
            Bold,
        ));
        b.spawn((
            Text::new(caps_spaced(ex.label())),
            TextFont { font_size: 11.0, ..default() },
            TextColor(theme.ink),
            Bold,
        ));
    });
}

// ────────────────────────────────────────────────────────────
// Hover tracking + visibility
// ────────────────────────────────────────────────────────────

/// If the trigger OR any example button is being hovered/pressed,
/// stamp the current time into `DropdownState.last_hover_secs`.
/// The visibility system consults this against the grace window.
fn track_hover(
    time: Res<Time>,
    mut state: ResMut<DropdownState>,
    trigger_q: Query<&Interaction, With<ExamplesTrigger>>,
    dropdown_btn_q: Query<&Interaction, With<ExampleBtn>>,
) {
    let hot = |i: &Interaction| matches!(i, Interaction::Hovered | Interaction::Pressed);
    let hovered =
        trigger_q.iter().any(hot) || dropdown_btn_q.iter().any(hot);
    if hovered {
        state.last_hover_secs = time.elapsed_secs();
    }
}

fn update_dropdown_visibility(
    time: Res<Time>,
    mut state: ResMut<DropdownState>,
    mut q: Query<&mut Visibility, With<ExamplesDropdown>>,
) {
    let elapsed = time.elapsed_secs();
    let within_grace = (elapsed - state.last_hover_secs) * 1000.0 < HOVER_GRACE_MS;
    let should_open = within_grace;
    if should_open == state.open { return; }
    state.open = should_open;
    for mut v in q.iter_mut() {
        *v = if should_open { Visibility::Visible } else { Visibility::Hidden };
    }
}

// ────────────────────────────────────────────────────────────
// Click handling + visuals
// ────────────────────────────────────────────────────────────

fn handle_example_buttons(
    q: Query<(&Interaction, &ExampleBtn), (Changed<Interaction>, With<Button>)>,
    mut load: bevy::ecs::message::MessageWriter<LoadExample>,
) {
    for (interaction, btn) in q.iter() {
        if *interaction == Interaction::Pressed {
            load.write(LoadExample(btn.0));
        }
    }
}

fn style_example_buttons(
    theme: Res<Theme>,
    mut q: Query<(&Interaction, &mut BackgroundColor, &mut BorderColor), With<ExampleBtn>>,
) {
    for (interaction, mut bg, mut border) in q.iter_mut() {
        let hovered = matches!(interaction, Interaction::Hovered | Interaction::Pressed);
        bg.0 = if hovered { theme.paper } else { Color::NONE };
        *border = BorderColor::all(if hovered { theme.ink } else { theme.rule });
    }
}

/// Toggle the trigger pill's visual look so it's invisible by
/// default and only surfaces when the cursor is on it (or while
/// the dropdown is already open). The trigger's `Node` and `Button`
/// stay live so hover detection still works — only the colors flip.
fn update_trigger_visibility(
    state: Res<DropdownState>,
    theme: Res<Theme>,
    trigger_q: Query<&Interaction, With<ExamplesTrigger>>,
    mut trigger_style: Query<
        (&mut BackgroundColor, &mut BorderColor),
        With<ExamplesTrigger>,
    >,
    mut trigger_text: Query<&mut TextColor, With<ExamplesTriggerLabel>>,
) {
    let trigger_hot = trigger_q
        .iter()
        .any(|i| matches!(i, Interaction::Hovered | Interaction::Pressed));
    let visible = trigger_hot || state.open;
    for (mut bg, mut border) in trigger_style.iter_mut() {
        bg.0 = if visible { theme.paper_alt } else { Color::NONE };
        *border = BorderColor::all(if visible { theme.ink } else { Color::NONE });
    }
    for mut tc in trigger_text.iter_mut() {
        tc.0 = if visible { theme.ink } else { Color::NONE };
    }
}
