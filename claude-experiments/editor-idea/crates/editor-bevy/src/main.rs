use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::text::{ComputedTextBlock, TextLayoutInfo};
use editor_core::commands::{
    cursor_char_left, cursor_char_right, cursor_doc_end, cursor_doc_start, cursor_line_down,
    cursor_line_end, cursor_line_start, cursor_line_up, cursor_word_left, cursor_word_right,
    delete_char_backward, delete_char_forward, indent_more, insert_newline_and_indent,
    select_all, select_char_left, select_char_right, select_doc_end, select_doc_start,
    select_line_down, select_line_end, select_line_start, select_line_up, select_word_left,
    select_word_right,
};
use editor_core::history::{redo, undo};
use editor_core::selection::Selection;
use editor_core::state::EditorState;
use editor_core::transaction::{Change, Transaction};

// Font metrics: hard-coded for now (real impl would query the layout).
const FONT_SIZE: f32 = 16.0;
const CHAR_WIDTH: f32 = 9.6; // works OK for the default Fira Sans fallback at 16pt
// Matches bevy's default `RelativeToFont(1.2)` line height.
const LINE_HEIGHT: f32 = FONT_SIZE * 1.2;
const MARGIN: f32 = 16.0;

#[derive(Resource)]
struct EditorRes(EditorState);

#[derive(Component)]
struct DocText;

#[derive(Component)]
struct CursorBar;

fn main() {
    let initial = "fn main() {\n    println!(\"hello, editor\");\n}\n";
    App::new()
        .insert_resource(EditorRes(
            EditorState::new(ropey::Rope::from_str(initial), Selection::cursor(0))
                .with_indent_unit("    "),
        ))
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "editor".into(),
                resolution: (900u32, 600u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (handle_input, sync_text, sync_cursor).chain(),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    // Root container so absolute positioning is anchored to the window.
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            position_type: PositionType::Absolute,
            ..default()
        })
        .with_children(|root| {
            root.spawn((
                DocText,
                Text::new(""),
                TextFont {
                    font_size: FONT_SIZE,
                    ..default()
                },
                TextColor(Color::srgb(0.92, 0.92, 0.94)),
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(MARGIN),
                    left: Val::Px(MARGIN),
                    ..default()
                },
            ));
            // `sync_cursor` reads the laid-out `ComputedTextBlock` to
            // find each line's actual y, so we let bevy default the
            // line height (`RelativeToFont(1.2)`) rather than pinning it.

            root.spawn((
                CursorBar,
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(MARGIN),
                    left: Val::Px(MARGIN),
                    width: Val::Px(2.0),
                    height: Val::Px(LINE_HEIGHT),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.55, 0.85, 1.0)),
            ));
        });
}

fn sync_text(state: Res<EditorRes>, mut q: Query<&mut Text, With<DocText>>) {
    if let Ok(mut text) = q.single_mut() {
        let s = state.0.doc.to_string();
        if text.0 != s {
            text.0 = s;
        }
    }
}

/// Place the cursor by reading the actual laid-out text rather than
/// guessing line heights from constants. `ComputedTextBlock::buffer()`
/// exposes cosmic-text's layout runs — each run carries `line_i`,
/// `line_top`, and `line_height` in physical pixels. We divide by
/// `TextLayoutInfo.scale_factor` to get logical UI pixels.
///
/// Cosmic-text emits a run for *every* line, including empty ones
/// (see cosmic-text's shape.rs: "create a visual line for empty lines").
/// It also appends a trailing phantom empty line when the text ends in
/// a newline — matching ropey's `char_to_line` for a cursor past the
/// final `\n`. So `line_i` lines up 1:1 with the ropey line index.
fn sync_cursor(
    state: Res<EditorRes>,
    text_q: Query<(&ComputedTextBlock, &TextLayoutInfo), With<DocText>>,
    mut cursor_q: Query<&mut Node, With<CursorBar>>,
) {
    let Ok(mut node) = cursor_q.single_mut() else { return };
    let head = state.0.selection.primary_range().head;
    let line = state.0.doc.char_to_line(head);
    let col = head - state.0.doc.line_to_char(line);

    let top_px = text_q
        .single()
        .ok()
        .and_then(|(block, layout_info)| {
            let scale = layout_info.scale_factor.max(1.0);
            let mut matched: Option<f32> = None;
            let mut tail: Option<(usize, f32, f32)> = None; // (line_i, line_top, line_height)
            for run in block.buffer().layout_runs() {
                if run.line_i == line {
                    matched = Some(run.line_top / scale);
                    break;
                }
                tail = Some((run.line_i, run.line_top, run.line_height));
            }
            matched.or_else(|| {
                // Cursor sits on a line past the last emitted run (can
                // happen transiently when state updates before layout
                // re-runs). Extrapolate from the last run's geometry.
                let (last_i, last_top, last_height) = tail?;
                let below = (line - last_i) as f32;
                Some((last_top + below * last_height) / scale)
            })
        })
        .unwrap_or_else(|| line as f32 * LINE_HEIGHT);

    node.left = Val::Px(MARGIN + col as f32 * CHAR_WIDTH);
    node.top = Val::Px(MARGIN + top_px);
}

fn handle_input(
    mut keys: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<EditorRes>,
) {
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);
    let alt = mods.pressed(KeyCode::AltLeft) || mods.pressed(KeyCode::AltRight);
    let meta = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let mod_word = alt || ctrl; // Mac word-jump is alt; Windows/Linux is ctrl
    let mod_doc = meta || ctrl;

    for ev in keys.read() {
        if !ev.state.is_pressed() {
            continue;
        }

        // Built-in editing commands first (arrows, backspace, etc.).
        let cmd_result = match ev.key_code {
            KeyCode::ArrowLeft => Some(if shift {
                if mod_word { run(&state.0, select_word_left) } else { run(&state.0, select_char_left) }
            } else if mod_word {
                run(&state.0, cursor_word_left)
            } else {
                run(&state.0, cursor_char_left)
            }),
            KeyCode::ArrowRight => Some(if shift {
                if mod_word { run(&state.0, select_word_right) } else { run(&state.0, select_char_right) }
            } else if mod_word {
                run(&state.0, cursor_word_right)
            } else {
                run(&state.0, cursor_char_right)
            }),
            KeyCode::ArrowUp => Some(if shift {
                run(&state.0, select_line_up)
            } else {
                run(&state.0, cursor_line_up)
            }),
            KeyCode::ArrowDown => Some(if shift {
                run(&state.0, select_line_down)
            } else {
                run(&state.0, cursor_line_down)
            }),
            KeyCode::Home => Some(if shift {
                if mod_doc { run(&state.0, select_doc_start) } else { run(&state.0, select_line_start) }
            } else if mod_doc {
                run(&state.0, cursor_doc_start)
            } else {
                run(&state.0, cursor_line_start)
            }),
            KeyCode::End => Some(if shift {
                if mod_doc { run(&state.0, select_doc_end) } else { run(&state.0, select_line_end) }
            } else if mod_doc {
                run(&state.0, cursor_doc_end)
            } else {
                run(&state.0, cursor_line_end)
            }),
            KeyCode::Backspace => Some(run_history(&state.0, delete_char_backward)),
            KeyCode::Delete => Some(run_history(&state.0, delete_char_forward)),
            KeyCode::Enter | KeyCode::NumpadEnter => {
                Some(run_history(&state.0, insert_newline_and_indent))
            }
            KeyCode::Tab => Some(run_history(&state.0, indent_more)),
            // Cmd/Ctrl combos
            KeyCode::KeyA if mod_doc => Some(run(&state.0, select_all)),
            KeyCode::KeyZ if mod_doc => Some(if shift {
                redo(&state.0).map(|new| (new, true))
            } else {
                undo(&state.0).map(|new| (new, true))
            }),
            _ => None,
        };

        if let Some(Some((new_state, _))) = cmd_result {
            state.0 = new_state;
            continue;
        }
        if let Some(None) = cmd_result {
            // Command didn't apply; consume the key without falling through
            // to character insertion (so e.g. Cmd+S doesn't insert "s").
            continue;
        }

        // Fall back: if the key produced a typeable character and no modifier
        // (other than shift) is held, insert it.
        if mod_doc || alt {
            continue;
        }
        if let Key::Character(s) = &ev.logical_key {
            // Use the first char only (most keys produce one); paste is a
            // separate path we don't yet handle.
            let text: String = s.chars().take(1).collect();
            if text.is_empty() {
                continue;
            }
            let tr = Transaction::new()
                .change(Change::new(
                    state.0.selection.primary_range().from(),
                    state.0.selection.primary_range().to(),
                    text.clone(),
                ))
                .select(Selection::cursor(
                    state.0.selection.primary_range().from() + text.chars().count(),
                ));
            state.0 = state.0.apply_with_history(&tr);
        }
    }
}

/// Run a non-history command, returning the new state and whether anything happened.
fn run(
    state: &EditorState,
    cmd: fn(&EditorState) -> Option<Transaction>,
) -> Option<(EditorState, bool)> {
    cmd(state).map(|tr| (state.apply(&tr), true))
}

/// Same but applies via `apply_with_history` so the change is undoable.
fn run_history(
    state: &EditorState,
    cmd: fn(&EditorState) -> Option<Transaction>,
) -> Option<(EditorState, bool)> {
    cmd(state).map(|tr| (state.apply_with_history(&tr), true))
}
