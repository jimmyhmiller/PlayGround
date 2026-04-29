//! Horizontal slider primitive.
//!
//! Spawns a [`Slider`] entity holding the current value plus min/max bounds.
//! Visuals: a paper-on-ink track with an accent-filled bar whose width
//! tracks the value, a label on top-left, and a numeric readout on
//! top-right (with a consumer-chosen unit suffix).
//!
//! Interaction: click anywhere on the track to set the value; hold and
//! drag to continuously update. The drag pins to the slider that got the
//! initial `Pressed` so cursor-off-edge doesn't stall at the endpoint.
//!
//! Consumer wiring:
//! ```ignore
//! #[derive(Component)]
//! struct RateSlider { node: NodeId }
//!
//! spawn_slider(parent, &theme, "Rate", 0.0, 20.0, 10.0, "/s", RateSlider { node });
//! ```
//!
//! A consumer system reacts to `Changed<Slider>` filtered by its marker
//! and pushes the new value into sim state.

use crate::theme::Theme;
use crate::typography::{Bold, Mono, caps_spaced};
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

pub struct SliderPlugin;

impl Plugin for SliderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SliderDrag>().add_systems(
            Update,
            (
                start_slider_drag,
                continue_slider_drag,
                sync_slider_fill,
                sync_slider_value_text,
            ),
        );
    }
}

/// Horizontal slider. `value` lives in `[min, max]` inclusive. Consumers
/// react to `Changed<Slider>` to push the new value into their domain.
///
/// `step > 0` snaps `value` to the nearest multiple of `step` during
/// drag (offset from `min`), so e.g. `min = 2, step = 1` produces a
/// slider that lands only on `2, 3, 4, …` regardless of cursor
/// precision. `step = 0` (the default) keeps the slider continuous.
/// Display formatting follows the same rule as before: clean
/// integers render without a decimal.
#[derive(Component, Debug, Clone)]
pub struct Slider {
    pub value: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

impl Slider {
    pub fn fraction(&self) -> f32 {
        if self.max <= self.min { 0.0 }
        else { ((self.value - self.min) / (self.max - self.min)).clamp(0.0, 1.0) }
    }
}

/// Marker on the accent-coloured fill child of a slider. Stores the owning
/// slider entity directly so the sync system is a straight lookup.
#[derive(Component)]
struct SliderFill {
    slider: Entity,
}

/// Marker on the numeric readout `Text`. Back-references the slider so the
/// text can live anywhere in the layout tree.
#[derive(Component)]
struct SliderValueText {
    slider: Entity,
    unit: &'static str,
}

/// Pins the active slider across frames so cursor-off-track doesn't stall
/// the drag. Released on mouse-up.
#[derive(Resource, Default)]
struct SliderDrag {
    entity: Option<Entity>,
}

/// Build a complete slider row: header (`label ↘ value`) above a clickable
/// track with fill. `unit` is suffixed onto the numeric readout. `extra` is
/// the consumer's marker bundle, attached to the [`Slider`] entity so
/// their `Changed<Slider>` sync system can resolve which slider changed.
///
/// Implementation note: we spawn the track *first* to capture its entity
/// id, then spawn the header with that id wired into the value-text
/// component. `FlexDirection::ColumnReverse` on the row container puts
/// the header visually on top despite being spawned second.
pub fn spawn_slider(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    min: f32,
    max: f32,
    initial: f32,
    unit: &'static str,
    extra: impl Bundle,
) {
    spawn_slider_with_step(parent, theme, label, min, max, /*step=*/ 0.0, initial, unit, extra);
}

/// `spawn_slider` plus a `step` knob — see [`Slider`]. Spawns
/// pass-through to the same widget; the only difference is the
/// `step` field on the `Slider` component. `step = 0.0` is identical
/// to the continuous form.
pub fn spawn_slider_with_step(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    min: f32,
    max: f32,
    step: f32,
    initial: f32,
    unit: &'static str,
    extra: impl Bundle,
) {
    let value = if step > 0.0 {
        snap_to_step(initial.clamp(min, max), min, step)
    } else {
        initial.clamp(min, max)
    };
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(6.0)),
            flex_direction: FlexDirection::ColumnReverse,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|row| {
            // Track first — capture its id so the header's value-text and
            // the track's fill-child can both reference it.
            let track_entity = row
                .spawn((
                    Button,
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(14.0),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(4.0)),
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    BackgroundColor(theme.paper),
                    BorderColor::all(theme.ink),
                    Slider { min, max, value, step },
                    extra,
                ))
                .id();

            // Fill is a child of the track, tagged with its slider's id.
            row.commands().entity(track_entity).with_children(|t| {
                t.spawn((
                    Node {
                        width: Val::Percent(100.0 * fraction(value, min, max)),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    BackgroundColor(theme.accent),
                    SliderFill { slider: track_entity },
                ));
            });

            // Header (label + value readout). Lives *below* the track in
            // spawn order but flex-column-reverse puts it on top.
            row.spawn(Node {
                width: Val::Percent(100.0),
                justify_content: JustifyContent::SpaceBetween,
                ..default()
            })
            .with_children(|head| {
                head.spawn((
                    Text::new(caps_spaced(label)),
                    TextFont { font_size: 9.0, ..default() },
                    TextColor(theme.ink_soft),
                    Bold,
                ));
                head.spawn((
                    Text::new(format_value(value, unit)),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(theme.ink),
                    Bold,
                    Mono,
                    SliderValueText { slider: track_entity, unit },
                ));
            });
        });
}

fn fraction(v: f32, min: f32, max: f32) -> f32 {
    if max <= min { 0.0 } else { ((v - min) / (max - min)).clamp(0.0, 1.0) }
}

fn format_value(v: f32, unit: &str) -> String {
    if v.abs() >= 10.0 || v.fract().abs() < 0.05 {
        format!("{:.0}{}", v, unit)
    } else {
        format!("{:.1}{}", v, unit)
    }
}

// ──────────────────────────────────────────────────────────────
// Interaction systems
// ──────────────────────────────────────────────────────────────

fn start_slider_drag(
    sliders: Query<(Entity, &Interaction), (Changed<Interaction>, With<Slider>)>,
    mut drag: ResMut<SliderDrag>,
) {
    for (entity, interaction) in sliders.iter() {
        if *interaction == Interaction::Pressed {
            drag.entity = Some(entity);
        }
    }
}

fn continue_slider_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut drag: ResMut<SliderDrag>,
    mut sliders: Query<(&bevy::ui::ComputedNode, &bevy::ui::UiGlobalTransform, &mut Slider)>,
) {
    if !mouse.pressed(MouseButton::Left) {
        drag.entity = None;
        return;
    }
    let Some(entity) = drag.entity else { return };
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((computed, xform, mut slider)) = sliders.get_mut(entity) else {
        drag.entity = None;
        return;
    };
    let size = computed.size;
    if size.x <= 0.0 { return; }
    // ComputedNode + UiGlobalTransform are physical pixels; cursor is
    // logical. Scale cursor before comparing so Retina doesn't pin the
    // slider at zero.
    let scale = win.scale_factor();
    let cursor_px = cursor.x * scale;
    let center_x = xform.translation.x;
    let left = center_x - size.x * 0.5;
    let frac = ((cursor_px - left) / size.x).clamp(0.0, 1.0);
    let mut new_val = slider.min + frac * (slider.max - slider.min);
    if slider.step > 0.0 {
        new_val = snap_to_step(new_val, slider.min, slider.step).clamp(slider.min, slider.max);
    }
    if (new_val - slider.value).abs() < 1e-4 { return; }
    slider.value = new_val;
}

/// Snap `v` to the nearest multiple of `step`, anchored at zero. So
/// `step = 1` lands the slider on integer values regardless of where
/// `min` falls — `min = 0.1, step = 1` still snaps `1.0` to `1.0`,
/// not `1.1`. Caller is responsible for clamping back to `[min, max]`
/// after the snap if the rounded value drifted out of range.
fn snap_to_step(v: f32, _min: f32, step: f32) -> f32 {
    if step <= 0.0 { return v; }
    (v / step).round() * step
}

fn sync_slider_fill(
    sliders: Query<&Slider, Changed<Slider>>,
    mut fills: Query<(&mut Node, &SliderFill)>,
) {
    for (mut node, fill) in fills.iter_mut() {
        let Ok(slider) = sliders.get(fill.slider) else { continue };
        node.width = Val::Percent(100.0 * slider.fraction());
    }
}

fn sync_slider_value_text(
    sliders: Query<&Slider, Changed<Slider>>,
    mut texts: Query<(&mut Text, &SliderValueText)>,
) {
    for (mut text, svt) in texts.iter_mut() {
        let Ok(slider) = sliders.get(svt.slider) else { continue };
        let new = format_value(slider.value, svt.unit);
        if text.0 != new { text.0 = new; }
    }
}
