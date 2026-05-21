//! Single-line text input widget.
//!
//! Backed by `editor-core`'s `EditorState` so caret movement,
//! selection, undo/redo, word-jumps, and clipboard ops are all the same
//! commands the multi-line editor uses. The widget is a Bevy entity
//! with a `TextInput` component and two child entities (text + caret).
//! Hosts spawn one per editable field, attach it under whatever
//! transform they want, and route a `request_focus(entity)` call when
//! the user clicks it.
//!
//! Keyboard handling reads `FocusedTextInput` — a single global resource
//! pointing at the one input that owns keystrokes this frame. Multiple
//! inputs in the same window are fine: only one is focused at a time.
//!
//! Submit (Enter) and Cancel (Esc) don't mutate the buffer — they
//! emit `TextInputEvent::Submit` / `Cancel` and clear focus. Hosts
//! react to those events to commit/cancel their own edit state.

use std::time::Duration;

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::LineHeight;
use editor_core::commands::{
    cursor_char_left, cursor_char_right, cursor_line_end, cursor_line_start, cursor_word_left,
    cursor_word_right, delete_char_backward, delete_char_forward, select_all, select_char_left,
    select_char_right, select_line_end, select_line_start, select_word_left, select_word_right,
};
use editor_core::history::{redo, undo};
use editor_core::selection::Selection;
use editor_core::state::EditorState;
use editor_core::transaction::{Change, Transaction};
use ropey::Rope;

const CARET_BLINK_PERIOD: Duration = Duration::from_millis(1100);

// ---------- Components ----------

#[derive(Component)]
pub struct TextInput {
    pub state: EditorState,
    pub style: TextInputStyle,
    /// Width of the input in pixels — used for click hit-testing.
    pub width: f32,
}

impl TextInput {
    pub fn text(&self) -> String {
        self.state.doc.to_string()
    }

    /// Replace the buffer with `s` and place the caret at the end.
    /// Used by hosts that want transactional edit/cancel semantics:
    /// stash the saved value, let the user edit, then either commit or
    /// reset the buffer with this.
    pub fn set_text(&mut self, s: &str) {
        let s = s.replace(['\n', '\r'], "");
        self.state = EditorState::new(Rope::from_str(&s), Selection::cursor(s.chars().count()));
    }
}

#[derive(Clone)]
pub struct TextInputStyle {
    pub font: Handle<Font>,
    pub font_size: f32,
    pub line_height: f32,
    /// Cell advance in pixels — required for caret/click positioning.
    /// For the editor/terminal monospace fonts this is the same value
    /// they measure once at startup.
    pub cell_width: f32,
    pub color_idle: Color,
    pub color_focused: Color,
    pub color_caret: Color,
    pub color_selection: Color,
}

/// Marker for the singularly-focused text input. The keyboard system
/// only mutates the input pointed to by `FocusedTextInput.0`.
#[derive(Component)]
pub struct TextInputFocused;

#[derive(Component)]
pub struct TextInputView {
    pub text_entity: Entity,
    pub caret_entity: Entity,
    /// Pool of selection-rect sprites; despawned + respawned on
    /// selection change.
    pub selection_entities: Vec<Entity>,
}

// ---------- Resources / events ----------

#[derive(Resource, Default)]
pub struct FocusedTextInput(pub Option<Entity>);

/// The most-recently-focused text input. Survives blur (Enter/Esc /
/// click-outside) so we can re-focus it on the next keystroke without
/// requiring the user to click back into the field. Cleared only by
/// `focus_text_input` setting a new target.
#[derive(Resource, Default)]
pub struct LastFocusedTextInput(pub Option<Entity>);

#[derive(Message, Clone, Copy, Debug)]
pub enum TextInputEvent {
    Changed { entity: Entity },
    Submit { entity: Entity },
    Cancel { entity: Entity },
}

// ---------- Plugin ----------

pub struct TextInputPlugin;

impl Plugin for TextInputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FocusedTextInput>()
            .init_resource::<LastFocusedTextInput>()
            .add_message::<TextInputEvent>()
            .add_systems(
                Update,
                (
                    auto_refocus_last_text_input,
                    text_input_keyboard,
                    text_input_render_text,
                    text_input_render_caret,
                    text_input_render_selection,
                )
                    .chain(),
            );
    }
}

// ---------- Spawn ----------

/// Spawn a single-line text input under `parent`. Caller positions the
/// returned entity wherever they want via `Transform`. Returns the
/// `TextInput` entity (not its text/caret children).
pub fn spawn_text_input(
    commands: &mut Commands,
    parent: Entity,
    initial: &str,
    style: TextInputStyle,
    width: f32,
    transform: Transform,
) -> Entity {
    let initial = strip_newlines(initial);
    let state = EditorState::new(Rope::from_str(&initial), Selection::cursor(initial.chars().count()));

    let input = commands
        .spawn((
            ChildOf(parent),
            transform,
            Visibility::default(),
        ))
        .id();

    let text_entity = commands
        .spawn((
            ChildOf(input),
            Text2d::new(initial),
            TextFont {
                font: style.font.clone(),
                font_size: style.font_size,
                ..default()
            },
            LineHeight::Px(style.line_height),
            TextColor(style.color_idle),
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.1),
        ))
        .id();

    let caret_entity = commands
        .spawn((
            ChildOf(input),
            Sprite {
                color: style.color_caret,
                custom_size: Some(Vec2::new(2.0, style.line_height)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.3),
            Visibility::Hidden,
        ))
        .id();

    commands.entity(input).insert((
        TextInput {
            state,
            style,
            width,
        },
        TextInputView {
            text_entity,
            caret_entity,
            selection_entities: Vec::new(),
        },
    ));

    input
}

fn strip_newlines(s: &str) -> String {
    s.replace(['\n', '\r'], "")
}

// ---------- Public helpers ----------

/// Set focus to one input (or `None` to blur all). Atomically removes
/// the `TextInputFocused` marker from any other entity.
///
/// The current `focused.0` (if any non-None) is mirrored into
/// `LastFocusedTextInput` before being overwritten — `Some(next)` makes
/// `next` the new last-focus, and a `None` (blur) leaves the previous
/// last-focus intact so a subsequent keystroke can refocus the same
/// field the user was just typing into.
pub fn focus_text_input(
    commands: &mut Commands,
    focused: &mut FocusedTextInput,
    inputs: impl IntoIterator<Item = Entity>,
    target: Option<Entity>,
) {
    if focused.0 == target {
        return;
    }
    if let Some(prev) = focused.0 {
        commands.entity(prev).remove::<TextInputFocused>();
    }
    if let Some(next) = target {
        commands.entity(next).insert(TextInputFocused);
    }
    focused.0 = target;
    let _ = inputs; // reserved for future "blur others" sweep if needed
}


/// Compute the column at a given local-x position. Used by hosts to
/// turn a click on a text-input into a caret position.
pub fn col_at_x(local_x: f32, cell_width: f32) -> usize {
    if cell_width <= 0.0 {
        return 0;
    }
    let col = (local_x / cell_width + 0.5).floor();
    if col <= 0.0 { 0 } else { col as usize }
}

/// Move the caret of `input` to the column corresponding to `local_x`.
/// Clamps to the document length.
pub fn click_to_caret(input: &mut TextInput, local_x: f32) {
    let col = col_at_x(local_x, input.style.cell_width);
    let n = input.state.doc.len_chars();
    let pos = col.min(n);
    let tr = Transaction::new().select(Selection::cursor(pos));
    input.state = input.state.apply(&tr);
}

// ---------- Auto-refocus ----------

/// Keeps `LastFocusedTextInput` in sync with `FocusedTextInput`, then
/// restores focus to the most-recent input when the user starts typing
/// with nothing focused — as long as the focused *pane* is still the
/// one that owns that input, so clicking a terminal or editor doesn't
/// silently redirect typing into an old run-button field.
fn auto_refocus_last_text_input(
    mut commands: Commands,
    mut focused: ResMut<FocusedTextInput>,
    mut last: ResMut<LastFocusedTextInput>,
    focused_pane: Res<crate::FocusedPane>,
    events: MessageReader<KeyboardInput>,
    inputs_q: Query<(), With<TextInput>>,
    child_of_q: Query<&ChildOf>,
    pane_q: Query<(), With<crate::PaneTag>>,
) {
    // Mirror the current focus into "last" while it's set.
    if let Some(cur) = focused.0 {
        last.0 = Some(cur);
    }

    if focused.0.is_some() {
        return;
    }
    let Some(target) = last.0 else { return };
    if inputs_q.get(target).is_err() {
        // The remembered input was despawned (e.g. its pane was closed).
        last.0 = None;
        return;
    }

    // Only restore if at least one printable key is being pressed this
    // frame. Pure modifier presses, arrow keys, etc. shouldn't yank
    // focus back to a field the user blurred deliberately.
    if !any_text_press(events) {
        return;
    }

    // Only refocus if the focused pane still owns the target input —
    // otherwise the user clicked a different pane and we'd be stealing
    // its keystrokes.
    let Some(owner) = focused_pane.0 else { return };
    if ancestor_pane(target, &child_of_q, &pane_q) != Some(owner) {
        return;
    }

    commands.entity(target).insert(TextInputFocused);
    focused.0 = Some(target);
}

fn any_text_press(mut events: MessageReader<KeyboardInput>) -> bool {
    use bevy::input::keyboard::Key;
    events.read().any(|ev| {
        if !ev.state.is_pressed() {
            return false;
        }
        matches!(&ev.logical_key, Key::Character(_) | Key::Space)
    })
}

fn ancestor_pane(
    mut entity: Entity,
    child_of_q: &Query<&ChildOf>,
    pane_q: &Query<(), With<crate::PaneTag>>,
) -> Option<Entity> {
    for _ in 0..32 {
        if pane_q.get(entity).is_ok() {
            return Some(entity);
        }
        match child_of_q.get(entity) {
            Ok(p) => entity = p.parent(),
            Err(_) => return None,
        }
    }
    None
}

// ---------- Keyboard ----------

#[allow(clippy::too_many_arguments)]
fn text_input_keyboard(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    focused: Res<FocusedTextInput>,
    mut inputs: Query<&mut TextInput, With<TextInputFocused>>,
    mut out: MessageWriter<TextInputEvent>,
) {
    let Some(target) = focused.0 else {
        events.read().for_each(|_| {});
        return;
    };
    let Ok(mut ti) = inputs.get_mut(target) else {
        events.read().for_each(|_| {});
        return;
    };

    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);
    let alt = mods.pressed(KeyCode::AltLeft) || mods.pressed(KeyCode::AltRight);
    let meta = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let mod_word = alt || ctrl;
    let mod_doc = meta || ctrl;

    let mut changed = false;
    let mut submit = false;
    let mut cancel = false;

    for ev in events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        match ev.key_code {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                submit = true;
                continue;
            }
            KeyCode::Escape => {
                cancel = true;
                continue;
            }
            KeyCode::ArrowLeft => {
                let cmd = match (shift, mod_word) {
                    (true, true) => select_word_left,
                    (true, false) => select_char_left,
                    (false, true) => cursor_word_left,
                    (false, false) => cursor_char_left,
                };
                if apply(&mut ti.state, cmd) {
                    changed = true;
                }
                continue;
            }
            KeyCode::ArrowRight => {
                let cmd = match (shift, mod_word) {
                    (true, true) => select_word_right,
                    (true, false) => select_char_right,
                    (false, true) => cursor_word_right,
                    (false, false) => cursor_char_right,
                };
                if apply(&mut ti.state, cmd) {
                    changed = true;
                }
                continue;
            }
            KeyCode::Home => {
                let cmd = if shift { select_line_start } else { cursor_line_start };
                if apply(&mut ti.state, cmd) {
                    changed = true;
                }
                continue;
            }
            KeyCode::End => {
                let cmd = if shift { select_line_end } else { cursor_line_end };
                if apply(&mut ti.state, cmd) {
                    changed = true;
                }
                continue;
            }
            KeyCode::Backspace => {
                if delete_or_remove_selection(&mut ti.state, delete_char_backward) {
                    changed = true;
                }
                continue;
            }
            KeyCode::Delete => {
                if delete_or_remove_selection(&mut ti.state, delete_char_forward) {
                    changed = true;
                }
                continue;
            }
            KeyCode::KeyA if mod_doc => {
                if apply(&mut ti.state, select_all) {
                    changed = true;
                }
                continue;
            }
            KeyCode::KeyZ if mod_doc => {
                let new = if shift { redo(&ti.state) } else { undo(&ti.state) };
                if let Some(s) = new {
                    ti.state = s;
                    changed = true;
                }
                continue;
            }
            KeyCode::KeyC if mod_doc => {
                copy_selection(&ti.state);
                continue;
            }
            KeyCode::KeyX if mod_doc => {
                copy_selection(&ti.state);
                if delete_or_remove_selection(&mut ti.state, |_| None) {
                    changed = true;
                }
                continue;
            }
            KeyCode::KeyV if mod_doc => {
                if paste(&mut ti.state) {
                    changed = true;
                }
                continue;
            }
            _ => {}
        }

        // Cmd/ctrl/alt-modified printable keys aren't text input.
        if mod_doc || alt {
            continue;
        }
        let text: Option<String> = match &ev.logical_key {
            Key::Character(s) => Some(strip_newlines(s)),
            Key::Space => Some(" ".into()),
            _ => None,
        };
        if let Some(text) = text.filter(|t| !t.is_empty()) {
            insert_str(&mut ti.state, &text);
            changed = true;
        }
    }

    if changed {
        out.write(TextInputEvent::Changed { entity: target });
    }
    if submit {
        out.write(TextInputEvent::Submit { entity: target });
    }
    if cancel {
        out.write(TextInputEvent::Cancel { entity: target });
    }
}

fn apply(state: &mut EditorState, cmd: fn(&EditorState) -> Option<Transaction>) -> bool {
    if let Some(tr) = cmd(state) {
        *state = state.apply(&tr);
        true
    } else {
        false
    }
}

/// Backspace/Delete with selection-aware fallback: if there's a
/// selection, delete it; otherwise use the supplied char-level cmd.
fn delete_or_remove_selection(
    state: &mut EditorState,
    fallback: fn(&EditorState) -> Option<Transaction>,
) -> bool {
    let r = state.selection.primary_range();
    if r.from() != r.to() {
        let tr = Transaction::new()
            .change(Change::new(r.from(), r.to(), String::new()))
            .select(Selection::cursor(r.from()));
        *state = state.apply_with_history(&tr);
        return true;
    }
    if let Some(tr) = fallback(state) {
        *state = state.apply_with_history(&tr);
        true
    } else {
        false
    }
}

fn insert_str(state: &mut EditorState, text: &str) {
    let r = state.selection.primary_range();
    let new_pos = r.from() + text.chars().count();
    let tr = Transaction::new()
        .change(Change::new(r.from(), r.to(), text.to_string()))
        .select(Selection::cursor(new_pos));
    *state = state.apply_with_history(&tr);
}

fn copy_selection(state: &EditorState) {
    let r = state.selection.primary_range();
    if r.from() == r.to() {
        return;
    }
    let s = state.doc.slice(r.from()..r.to()).to_string();
    if let Ok(mut cb) = arboard::Clipboard::new() {
        let _ = cb.set_text(s);
    }
}

fn paste(state: &mut EditorState) -> bool {
    let Ok(mut cb) = arboard::Clipboard::new() else {
        return false;
    };
    let Ok(text) = cb.get_text() else {
        return false;
    };
    let text = strip_newlines(&text);
    if text.is_empty() {
        return false;
    }
    insert_str(state, &text);
    true
}

// ---------- Render ----------

fn text_input_render_text(
    inputs: Query<
        (&TextInput, &TextInputView, Option<&TextInputFocused>),
        Or<(Changed<TextInput>, Added<TextInputFocused>)>,
    >,
    mut text_q: Query<&mut Text2d>,
    mut color_q: Query<&mut TextColor>,
) {
    for (ti, view, focused) in &inputs {
        let want = ti.text();
        if let Ok(mut t) = text_q.get_mut(view.text_entity) {
            if t.0 != want {
                t.0 = want;
            }
        }
        if let Ok(mut c) = color_q.get_mut(view.text_entity) {
            c.0 = if focused.is_some() {
                ti.style.color_focused
            } else {
                ti.style.color_idle
            };
        }
    }
}

fn text_input_render_caret(
    time: Res<Time>,
    inputs: Query<(&TextInput, &TextInputView, Option<&TextInputFocused>)>,
    mut t_q: Query<&mut Transform>,
    mut vis_q: Query<&mut Visibility>,
) {
    let elapsed = time.elapsed_secs_f64();
    let blink_on = (elapsed / CARET_BLINK_PERIOD.as_secs_f64()).fract() < 0.55;
    for (ti, view, focused) in &inputs {
        let visible = focused.is_some() && blink_on;
        if let Ok(mut v) = vis_q.get_mut(view.caret_entity) {
            *v = if visible { Visibility::Inherited } else { Visibility::Hidden };
        }
        if focused.is_some() {
            let head = ti.state.selection.primary_range().head;
            let col = head.min(ti.state.doc.len_chars());
            let x = col as f32 * ti.style.cell_width;
            if let Ok(mut t) = t_q.get_mut(view.caret_entity) {
                t.translation.x = x;
                t.translation.y = 0.0;
                t.translation.z = 0.3;
            }
        }
    }
}

fn text_input_render_selection(
    mut commands: Commands,
    mut inputs: Query<
        (Entity, &TextInput, &mut TextInputView),
        Or<(Changed<TextInput>, Added<TextInputFocused>)>,
    >,
) {
    for (input_entity, ti, mut view) in &mut inputs {
        for e in view.selection_entities.drain(..) {
            commands.entity(e).despawn();
        }
        let r = ti.state.selection.primary_range();
        if r.from() == r.to() {
            continue;
        }
        let cw = ti.style.cell_width;
        let x = r.from() as f32 * cw;
        let w = ((r.to() - r.from()) as f32 * cw).max(1.0);
        let e = commands
            .spawn((
                ChildOf(input_entity),
                Sprite {
                    color: ti.style.color_selection,
                    custom_size: Some(Vec2::new(w, ti.style.line_height)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(x, 0.0, 0.05),
            ))
            .id();
        view.selection_entities.push(e);
    }
}
