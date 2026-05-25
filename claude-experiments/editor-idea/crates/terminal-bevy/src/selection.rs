//! Click-drag text selection inside a terminal + system-clipboard copy
//! and paste.
//!
//! The selection itself (anchor + head cell coords, plus overlay pool)
//! lives on `TerminalSelection` in `lib.rs`; this module owns the
//! systems that visualise it and the keyboard shortcuts that move text
//! between the terminal and the OS clipboard.
//!
//! Selection is line-flow: the highlighted region is one or more
//! per-row strips. For copy we read the same range out of the worker's
//! snapshot, trim trailing whitespace per row, and join with newlines.

use std::sync::Mutex;

use arboard::Clipboard;
use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use pane_bevy::{FocusedPane, PaneChrome, PaneKindMarker, PaneTag};

use crate::worker::{SnapCell, WorkerMsg};
use crate::{
    MonoMetrics, TermGrid, TerminalSelection, TerminalStore, LINE_HEIGHT, PANE_KIND,
};

// Selection color now driven by theme tokens::SELECTION (look up
// in `render_selection_overlays`).

/// Wraps the lazily-initialised system clipboard. `arboard::Clipboard`
/// is `!Send` on some platforms, but `Mutex` lets us treat it as a
/// resource — only the one system that touches it ever locks.
#[derive(Resource)]
pub struct ClipboardState {
    pub clipboard: Mutex<Option<Clipboard>>,
}

impl Default for ClipboardState {
    fn default() -> Self {
        Self {
            clipboard: Mutex::new(Clipboard::new().ok()),
        }
    }
}

pub struct SelectionPlugin;

impl Plugin for SelectionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClipboardState::default())
            .add_systems(Update, (render_selection_overlays, copy_paste_keys));
    }
}

/// Rebuild the per-row overlay sprites for each terminal whose
/// selection has changed since the last frame.
fn render_selection_overlays(
    mut commands: Commands,
    metrics: Res<MonoMetrics>,
    theme: Res<style_bevy::Theme>,
    mut q: Query<
        (&PaneChrome, &TermGrid, &mut TerminalSelection, &PaneKindMarker),
        With<PaneTag>,
    >,
) {
    let sel_color = Color::LinearRgba(theme.color(style_bevy::tokens::SELECTION));
    for (chrome, grid, mut sel, kind) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }
        let active = sel.is_active();
        let cols = grid.cols as i32;
        let rows = grid.rows as i32;

        // Tear down whatever's there. Cheap: at most `rows` entities,
        // and only when the selection is being rebuilt.
        for &e in &sel.overlays {
            if let Ok(mut ec) = commands.get_entity(e) {
                ec.despawn();
            }
        }
        sel.overlays.clear();

        if !active || cols == 0 || rows == 0 {
            continue;
        }

        let Some((start, end)) = sel.normalised() else {
            continue;
        };

        let cell_w = metrics.cell_width;

        // Walk the visible rows of the selection range, building one
        // strip per row. Strips for off-grid rows are skipped — the
        // selection state still remembers them so the user can drag
        // back into view, but we don't render anything outside the grid.
        let visible_first = start.1.max(0);
        let visible_last = end.1.min(rows - 1);
        for row in visible_first..=visible_last {
            let (col_start, col_end) = if start.1 == end.1 {
                // Single-row selection.
                (start.0.min(end.0), start.0.max(end.0))
            } else if row == start.1 {
                // First row: from start col to end of row.
                (start.0, cols - 1)
            } else if row == end.1 {
                // Last row: from start of row to end col.
                (0, end.0)
            } else {
                // Middle row: full span.
                (0, cols - 1)
            };
            let col_start = col_start.max(0);
            let col_end = col_end.min(cols - 1);
            if col_end < col_start {
                continue;
            }
            let span_cells = (col_end - col_start + 1) as f32;
            let x = col_start as f32 * cell_w;
            let y = -(row as f32) * LINE_HEIGHT;
            let entity = commands
                .spawn((
                    ChildOf(chrome.content_root),
                    Sprite {
                        color: sel_color,
                        custom_size: Some(Vec2::new(span_cells * cell_w, LINE_HEIGHT)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(x, y, 0.4),
                ))
                .id();
            sel.overlays.push(entity);
        }
    }
}

/// Cmd+C copies the focused terminal's selection to the clipboard;
/// Cmd+V writes the clipboard contents to the focused terminal's pty.
fn copy_paste_keys(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    focused: Res<FocusedPane>,
    store: Res<TerminalStore>,
    sels: Query<&TerminalSelection>,
    kinds: Query<&PaneKindMarker>,
    clip: Res<ClipboardState>,
) {
    // Skip unless the focused pane is a terminal.
    let target_is_terminal = focused
        .0
        .and_then(|e| kinds.get(e).ok())
        .is_some_and(|k| k.0 == PANE_KIND);
    if !target_is_terminal {
        events.read().for_each(|_| {});
        return;
    }
    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    if !cmd {
        // Drain so we don't carry the events into a future Cmd-down frame.
        events.read().for_each(|_| {});
        return;
    }
    for ev in events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        match ev.key_code {
            KeyCode::KeyC => {
                let Some(target) = focused.0 else { continue };
                let Ok(sel) = sels.get(target) else { continue };
                if !sel.is_active() {
                    continue;
                }
                let Some(data) = store.map.get(&target) else { continue };
                let text = {
                    let g = data.worker.snapshot.lock().expect("snapshot lock");
                    extract_selection_text(&g.cells, g.cols as i32, g.rows as i32, sel)
                };
                if let Ok(mut guard) = clip.clipboard.lock()
                    && let Some(c) = guard.as_mut()
                {
                    let _ = c.set_text(text);
                }
            }
            KeyCode::KeyV => {
                let Some(target) = focused.0 else { continue };
                let Some(data) = store.map.get(&target) else { continue };
                let text = clip
                    .clipboard
                    .lock()
                    .ok()
                    .and_then(|mut g| g.as_mut().and_then(|c| c.get_text().ok()));
                if let Some(text) = text
                    && !text.is_empty()
                {
                    data.worker.send(WorkerMsg::Paste(text));
                }
            }
            _ => {}
        }
    }
}

fn extract_selection_text(
    cells: &[SnapCell],
    cols: i32,
    rows: i32,
    sel: &TerminalSelection,
) -> String {
    let Some((start, end)) = sel.normalised() else {
        return String::new();
    };
    let mut out = String::new();
    let visible_first = start.1.max(0);
    let visible_last = end.1.min(rows - 1);
    for row in visible_first..=visible_last {
        let (col_start, col_end) = if start.1 == end.1 {
            (start.0.min(end.0), start.0.max(end.0))
        } else if row == start.1 {
            (start.0, cols - 1)
        } else if row == end.1 {
            (0, end.0)
        } else {
            (0, cols - 1)
        };
        let col_start = col_start.max(0);
        let col_end = col_end.min(cols - 1);
        if col_end < col_start {
            if row != visible_last {
                out.push('\n');
            }
            continue;
        }
        let row_base = (row as usize) * (cols as usize);
        let mut line = String::new();
        for c in col_start..=col_end {
            let idx = row_base + c as usize;
            if let Some(cell) = cells.get(idx) {
                line.push(cell.ch);
            }
        }
        // Trim trailing whitespace — terminals pad rows with spaces and
        // copying a screen full of those is annoying. Leading spaces
        // (indentation) are preserved.
        let trimmed = line.trim_end();
        out.push_str(trimmed);
        if row != visible_last {
            out.push('\n');
        }
    }
    out
}
