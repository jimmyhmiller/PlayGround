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
use editor_core::selection::{Range, Selection};
use editor_core::state::EditorState;
use editor_core::transaction::{Change, Transaction};
use ropey::Rope;

const CARET_BLINK_PERIOD: Duration = Duration::from_millis(1100);

// ---------- Components ----------

#[derive(Component)]
pub struct TextInput {
    pub state: EditorState,
    pub style: TextInputStyle,
    /// Width of the input in pixels — used for click hit-testing and,
    /// for multiline inputs, for word-wrap.
    pub width: f32,
    /// When true the buffer keeps `\n`, the text renders word-wrapped
    /// across visual lines, and Cmd/Shift+Enter inserts a newline
    /// (plain Enter still submits). When false the input is the classic
    /// single-line field: newlines are stripped on every entry path.
    pub multiline: bool,
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
        let s = if self.multiline {
            s.replace('\r', "")
        } else {
            s.replace(['\n', '\r'], "")
        };
        self.state = EditorState::new(Rope::from_str(&s), Selection::cursor(s.chars().count()));
    }

    /// Max columns that fit in `width` at the monospace advance. Returns
    /// `None` for single-line inputs or degenerate widths (treat as
    /// "no wrap").
    fn wrap_cols(&self) -> Option<usize> {
        if !self.multiline {
            return None;
        }
        let cw = self.style.cell_width;
        if cw <= 0.0 || self.width <= cw {
            return None;
        }
        Some((self.width / cw).floor() as usize)
    }

    /// Lay the document out into visual (post-wrap) lines. Single-line
    /// inputs always return exactly one line spanning the whole doc.
    fn visual_lines(&self) -> Vec<VisualLine> {
        let s = self.text();
        wrap_visual_lines(&s, self.wrap_cols())
    }

    /// Visual `(line, col)` of a char position under the current wrap.
    fn caret_line_col(&self, pos: usize) -> (usize, usize) {
        let lines = self.visual_lines();
        for (i, l) in lines.iter().enumerate() {
            if pos < l.next_start {
                return (i, pos.saturating_sub(l.start));
            }
        }
        // pos at (or past) end of doc → last line, last column.
        let last = lines.len().saturating_sub(1);
        let col = pos.saturating_sub(lines[last].start);
        (last, col)
    }

    /// Char position for a visual `(line, col)`, clamping col to the
    /// line's displayed length and line to the document.
    fn pos_at_line_col(&self, line: usize, col: usize) -> usize {
        let lines = self.visual_lines();
        if lines.is_empty() {
            return 0;
        }
        let line = line.min(lines.len() - 1);
        let l = &lines[line];
        let max_col = l.end_display - l.start;
        l.start + col.min(max_col)
    }
}

/// One visual line after word-wrap. Char indices into the document.
/// `[start, end_display)` are the chars actually drawn on this line;
/// `next_start` is where the following visual line begins — equal to
/// `end_display` for a char-break, or one past it when a separator
/// char (an explicit `\n` or the space a soft-wrap broke on) is
/// consumed and not drawn.
#[derive(Clone, Copy, Debug)]
struct VisualLine {
    start: usize,
    end_display: usize,
    next_start: usize,
}

/// Greedy monospace word-wrap producing visual-line char ranges.
/// `max_cols == None` means no wrapping (single-line, or degenerate
/// width): explicit `\n` still split lines. Mirrors the column
/// arithmetic of the issues pane's `wrap_line_count` so the editing
/// layout matches the non-editing display.
fn wrap_visual_lines(text: &str, max_cols: Option<usize>) -> Vec<VisualLine> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    // Split into hard segments on '\n' (the newline char is consumed).
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = 0usize;
    for (i, &ch) in chars.iter().enumerate() {
        if ch == '\n' {
            segments.push((seg_start, i));
            seg_start = i + 1;
        }
    }
    segments.push((seg_start, n));

    let mut lines: Vec<VisualLine> = Vec::new();
    let seg_count = segments.len();
    for (si, (hs, he)) in segments.into_iter().enumerate() {
        // next_start after this hard segment: one past the '\n' for all
        // but the final segment (which ends at the doc end).
        let after_seg = if si + 1 < seg_count { he + 1 } else { he };

        let max = match max_cols {
            Some(m) if m > 0 => m,
            _ => {
                lines.push(VisualLine {
                    start: hs,
                    end_display: he,
                    next_start: after_seg,
                });
                continue;
            }
        };
        if he == hs {
            lines.push(VisualLine {
                start: hs,
                end_display: he,
                next_start: after_seg,
            });
            continue;
        }

        // Word ranges within [hs, he), split on single spaces (empty
        // words model runs of spaces, matching `str::split(' ')`).
        let mut words: Vec<(usize, usize)> = Vec::new();
        let mut ws = hs;
        for i in hs..he {
            if chars[i] == ' ' {
                words.push((ws, i));
                ws = i + 1;
            }
        }
        words.push((ws, he));

        let mut line_start = hs;
        let mut col = 0usize;
        let mut first_word = true;
        for (wi, (wstart, wend)) in words.iter().copied().enumerate() {
            let wlen = wend - wstart;
            if wlen == 0 {
                // A space delimiter (empty word). Costs one column
                // unless it would overflow, in which case wrap on it.
                if !first_word {
                    if col + 1 > max {
                        // Break: the space at wstart-1 is consumed.
                        lines.push(VisualLine {
                            start: line_start,
                            end_display: wstart.saturating_sub(1),
                            next_start: wstart,
                        });
                        line_start = wstart;
                        col = 0;
                    } else {
                        col += 1;
                    }
                }
                first_word = false;
                continue;
            }
            let needed = if first_word { wlen } else { wlen + 1 };
            if col + needed <= max {
                col += needed;
            } else if wlen <= max {
                // Wrap whole word to a fresh line; the joining space at
                // wstart-1 is the consumed break point.
                let brk = if wi > 0 { wstart - 1 } else { wstart };
                lines.push(VisualLine {
                    start: line_start,
                    end_display: brk,
                    next_start: wstart,
                });
                line_start = wstart;
                col = wlen;
            } else {
                // Word longer than the line: char-break it.
                if col > 0 {
                    let brk = if wi > 0 { wstart - 1 } else { wstart };
                    lines.push(VisualLine {
                        start: line_start,
                        end_display: brk,
                        next_start: wstart,
                    });
                    line_start = wstart;
                }
                let mut cur = wstart;
                let mut remaining = wlen;
                while remaining > max {
                    lines.push(VisualLine {
                        start: line_start,
                        end_display: cur + max,
                        next_start: cur + max,
                    });
                    cur += max;
                    line_start = cur;
                    remaining -= max;
                }
                col = remaining;
            }
            first_word = false;
        }
        // Flush the trailing partial line of this segment.
        lines.push(VisualLine {
            start: line_start,
            end_display: he,
            next_start: after_seg,
        });
    }

    if lines.is_empty() {
        lines.push(VisualLine {
            start: 0,
            end_display: 0,
            next_start: 0,
        });
    }
    lines
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
    spawn_text_input_inner(commands, parent, initial, style, width, transform, false)
}

/// Spawn a multiline text input: keeps `\n`, word-wraps the rendered
/// text to `width`, and inserts a newline on Cmd/Shift+Enter (plain
/// Enter still submits). Otherwise identical to [`spawn_text_input`].
pub fn spawn_text_input_multiline(
    commands: &mut Commands,
    parent: Entity,
    initial: &str,
    style: TextInputStyle,
    width: f32,
    transform: Transform,
) -> Entity {
    spawn_text_input_inner(commands, parent, initial, style, width, transform, true)
}

#[allow(clippy::too_many_arguments)]
fn spawn_text_input_inner(
    commands: &mut Commands,
    parent: Entity,
    initial: &str,
    style: TextInputStyle,
    width: f32,
    transform: Transform,
    multiline: bool,
) -> Entity {
    let initial = if multiline {
        initial.replace('\r', "")
    } else {
        strip_newlines(initial)
    };
    let state = EditorState::new(Rope::from_str(&initial), Selection::cursor(initial.chars().count()));

    let input = commands
        .spawn((
            ChildOf(parent),
            transform,
            Visibility::default(),
        ))
        .id();

    // We feed the text entity our own pre-wrapped text (visual lines
    // joined with '\n'), so the renderer must NOT word-wrap again.
    let initial_display = wrap_display_text(&initial, {
        let cw = style.cell_width;
        if multiline && cw > 0.0 && width > cw {
            Some((width / cw).floor() as usize)
        } else {
            None
        }
    });
    let text_entity = commands
        .spawn((
            ChildOf(input),
            Text2d::new(initial_display),
            bevy::text::TextLayout {
                linebreak: bevy::text::LineBreak::NoWrap,
                ..default()
            },
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
            multiline,
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

/// The string to hand to the `Text2d` (whose `LineBreak` is `NoWrap`):
/// the document re-broken at our own wrap points so what's drawn lines
/// up exactly with the caret/selection math.
fn wrap_display_text(text: &str, max_cols: Option<usize>) -> String {
    let chars: Vec<char> = text.chars().collect();
    let lines = wrap_visual_lines(text, max_cols);
    let mut out = String::new();
    for (i, l) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.extend(&chars[l.start..l.end_display]);
    }
    out
}

#[cfg(test)]
mod wrap_tests {
    use super::*;

    fn lines(text: &str, max: usize) -> Vec<(usize, usize, usize)> {
        wrap_visual_lines(text, Some(max))
            .into_iter()
            .map(|l| (l.start, l.end_display, l.next_start))
            .collect()
    }

    #[test]
    fn no_wrap_keeps_one_line() {
        assert_eq!(
            wrap_visual_lines("hello world", None)
                .iter()
                .map(|l| (l.start, l.end_display))
                .collect::<Vec<_>>(),
            vec![(0, 11)]
        );
    }

    #[test]
    fn hard_newline_splits_and_consumes() {
        // "ab\ncd": newline at idx 2 is consumed (not displayed).
        assert_eq!(lines("ab\ncd", 80), vec![(0, 2, 3), (3, 5, 5)]);
        assert_eq!(wrap_display_text("ab\ncd", Some(80)), "ab\ncd");
    }

    #[test]
    fn empty_line_between_newlines() {
        // "a\n\nb": middle empty visual line.
        assert_eq!(lines("a\n\nb", 80), vec![(0, 1, 2), (2, 2, 3), (3, 4, 4)]);
        assert_eq!(wrap_display_text("a\n\nb", Some(80)), "a\n\nb");
    }

    #[test]
    fn soft_wrap_drops_break_space() {
        // "aa bbbb" at width 3 → "aa" / "bbb" / "b"; the space at idx 2
        // is consumed by the wrap (not drawn).
        assert_eq!(lines("aa bbbb", 3), vec![(0, 2, 3), (3, 6, 6), (6, 7, 7)]);
        assert_eq!(wrap_display_text("aa bbbb", Some(3)), "aa\nbbb\nb");
    }

    #[test]
    fn word_wrap_keeps_words_whole() {
        // "foo bar baz" at width 7 → "foo bar" / "baz".
        assert_eq!(wrap_display_text("foo bar baz", Some(7)), "foo bar\nbaz");
    }

    #[test]
    fn empty_doc_is_one_empty_line() {
        assert_eq!(lines("", 80), vec![(0, 0, 0)]);
        assert_eq!(wrap_display_text("", Some(80)), "");
    }
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
    owner: Res<crate::KeyboardOwner>,
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
    // A text-entry modal (command palette, rename) owns the keyboard —
    // don't let a printable keystroke yank focus back into a field.
    if owner.is_modal() {
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
    owner: Res<crate::KeyboardOwner>,
    child_of_q: Query<&ChildOf>,
    pane_q: Query<(), With<crate::PaneTag>>,
    mut inputs: Query<&mut TextInput, With<TextInputFocused>>,
    mut out: MessageWriter<TextInputEvent>,
) {
    let Some(target) = focused.0 else {
        events.read().for_each(|_| {});
        return;
    };
    // A field can keep `FocusedTextInput` set while the user clicks away
    // to another pane. Only consume keys if the keyboard owner allows this
    // field's pane — otherwise the focused pane's handler owns them (this
    // is the fix for typing landing in a field AND a terminal at once).
    if let Some(pane) = ancestor_pane(target, &child_of_q, &pane_q) {
        if !owner.allows_pane(pane) {
            events.read().for_each(|_| {});
            return;
        }
    }
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
                // Multiline: Cmd/Shift+Enter inserts a newline; plain
                // Enter still submits. Single-line: always submit.
                if ti.multiline && (meta || shift) {
                    insert_str(&mut ti.state, "\n");
                    changed = true;
                } else {
                    submit = true;
                }
                continue;
            }
            KeyCode::Escape => {
                cancel = true;
                continue;
            }
            KeyCode::ArrowUp | KeyCode::ArrowDown if ti.multiline => {
                let (line, col) = ti.caret_line_col(ti.state.selection.primary_range().head);
                let target_line = if ev.key_code == KeyCode::ArrowUp {
                    line.saturating_sub(1)
                } else {
                    line + 1
                };
                let pos = ti.pos_at_line_col(target_line, col);
                let sel = if shift {
                    Selection::new(
                        vec![Range::new(ti.state.selection.primary_range().anchor, pos)],
                        0,
                    )
                } else {
                    Selection::cursor(pos)
                };
                ti.state = ti.state.apply(&Transaction::new().select(sel));
                changed = true;
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
            // Cmd/Ctrl+V (no Shift) = paste. Cmd+Shift+V is reserved for
            // app-global shortcuts (profiler vsync toggle, etc.) and must
            // not leak in here as a paste.
            KeyCode::KeyV if mod_doc && !shift => {
                let multiline = ti.multiline;
                if paste(&mut ti.state, multiline) {
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

fn paste(state: &mut EditorState, multiline: bool) -> bool {
    let Ok(mut cb) = arboard::Clipboard::new() else {
        return false;
    };
    let Ok(text) = cb.get_text() else {
        return false;
    };
    let text = if multiline {
        text.replace('\r', "")
    } else {
        strip_newlines(&text)
    };
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
        let want = if ti.multiline {
            wrap_display_text(&ti.text(), ti.wrap_cols())
        } else {
            ti.text()
        };
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
            let head = ti.state.selection.primary_range().head.min(ti.state.doc.len_chars());
            let (line, col) = if ti.multiline {
                ti.caret_line_col(head)
            } else {
                (0, head)
            };
            let x = col as f32 * ti.style.cell_width;
            let y = -(line as f32 * ti.style.line_height);
            if let Ok(mut t) = t_q.get_mut(view.caret_entity) {
                t.translation.x = x;
                t.translation.y = y;
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
        let (from, to) = (r.from(), r.to());
        let cw = ti.style.cell_width;
        let lh = ti.style.line_height;
        let color = ti.style.color_selection;

        // One rect per visual line the selection touches. For
        // single-line inputs there is exactly one line, so this reduces
        // to the previous single-rect behavior.
        let lines = ti.visual_lines();
        for (li, l) in lines.iter().enumerate() {
            let lo = from.max(l.start);
            let hi = to.min(l.end_display);
            if hi <= lo {
                continue;
            }
            let x = (lo - l.start) as f32 * cw;
            let w = ((hi - lo) as f32 * cw).max(1.0);
            let y = -(li as f32 * lh);
            let e = commands
                .spawn((
                    ChildOf(input_entity),
                    Sprite {
                        color,
                        custom_size: Some(Vec2::new(w, lh)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(x, y, 0.05),
                ))
                .id();
            view.selection_entities.push(e);
        }
    }
}
