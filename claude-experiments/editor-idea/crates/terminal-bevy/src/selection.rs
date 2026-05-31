//! Click-drag text selection inside a terminal + system-clipboard copy
//! and paste.
//!
//! The selection itself (anchor + head cell coords, plus overlay pool)
//! lives on `TerminalSelection` in `lib.rs`; this module owns the
//! systems that visualise it and the keyboard shortcuts that move text
//! between the terminal and the OS clipboard.
//!
//! Selection is line-flow: the highlighted region is one or more
//! per-row strips. For copy we ask the worker to read the range out of
//! its libghostty `Terminal` (which owns the full scrollback, not just
//! the visible snapshot), trim trailing whitespace per row, and join
//! with newlines — so a selection spanning content scrolled off-screen
//! copies in full. See `WorkerMsg::ExtractText`.

use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Duration;

use arboard::Clipboard;
use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use pane_bevy::{FocusedPane, PaneChrome, PaneKindMarker, PaneTag};

use crate::worker::WorkerMsg;
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
    store: Res<TerminalStore>,
    mut q: Query<
        (Entity, &PaneChrome, &TermGrid, &mut TerminalSelection, &PaneKindMarker),
        With<PaneTag>,
    >,
) {
    let sel_color = Color::LinearRgba(theme.color(style_bevy::tokens::SELECTION));
    for (entity, chrome, grid, mut sel, kind) in &mut q {
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

        // Selection rows are absolute (scrollback-anchored). Convert to
        // the current viewport row by subtracting this terminal's
        // viewport_offset; the rest of the math is identical to the
        // pre-scrolling code. Rows that fell outside [0, rows-1] are
        // simply skipped — state preserved, just not drawn.
        let offset: i64 = store
            .map
            .get(&entity)
            .map(|d| d.worker.snapshot.lock().expect("snapshot lock").viewport_offset as i64)
            .unwrap_or(0);
        let start_view = start.1 - offset;
        let end_view = end.1 - offset;

        let cell_w = metrics.cell_width;

        let visible_first = start_view.max(0);
        let visible_last = end_view.min((rows - 1) as i64);
        if visible_first > visible_last {
            continue;
        }
        for row in visible_first..=visible_last {
            let (col_start, col_end) = if start_view == end_view {
                // Single-row selection.
                (start.0.min(end.0), start.0.max(end.0))
            } else if row == start_view {
                // First row: from start col to end of row.
                (start.0, cols - 1)
            } else if row == end_view {
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
                let Some((start, end)) = sel.normalised() else {
                    continue;
                };
                let Some(data) = store.map.get(&target) else { continue };
                // Read the selected text off the worker's `Terminal`,
                // which owns the full scrollback — the visible snapshot
                // only holds the on-screen grid, so a selection that
                // spans content scrolled off-screen would otherwise copy
                // truncated. Round-trip via a oneshot reply channel; the
                // worker wakes immediately on the wake pipe, so the
                // bounded wait is effectively instant.
                let (reply_tx, reply_rx) = mpsc::channel();
                data.worker.send(WorkerMsg::ExtractText {
                    start,
                    end,
                    reply: reply_tx,
                });
                let Ok(text) = reply_rx.recv_timeout(Duration::from_millis(500)) else {
                    continue;
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

