//! Undo/redo history. Simpler than CM6's: no time-based coalescing, no
//! `isolateHistory` markers, no `addToHistory: false` annotation. Each call
//! to `apply_with_history` records one event. `apply` (without history)
//! still exists for transactions that shouldn't be undoable (e.g. external
//! collaborator edits, the undo command itself).

use crate::selection::Selection;
use crate::state::EditorState;
use crate::transaction::{Change, Transaction};

#[derive(Debug, Clone, Default)]
pub struct History {
    pub undo: Vec<HistoryEvent>,
    pub redo: Vec<HistoryEvent>,
    /// Selection-history: the BEFORE selection from each apply that changed
    /// the selection (whether or not it changed the doc). `undo_selection`
    /// pops from here; `redo_selection` puts entries on `selection_redo`.
    pub selection_undo: Vec<crate::selection::Selection>,
    pub selection_redo: Vec<crate::selection::Selection>,
}

#[derive(Debug, Clone)]
pub struct HistoryEvent {
    pub forward: Vec<Change>,
    pub inverse: Vec<Change>,
    pub selection_before: Selection,
    pub selection_after: Selection,
}

impl EditorState {
    /// Like `apply`, but records the change on the undo stack and clears redo.
    /// If the new event is "joinable" with the current top of the undo stack
    /// (consecutive single-char-style insertions / deletions at adjacent
    /// positions), it merges instead of pushing — matches CM6's default
    /// `newGroupDelay` coalescing without time tracking.
    pub fn apply_with_history(&self, tr: &Transaction) -> EditorState {
        self.apply_with_history_inner(tr, /*allow_join=*/ true)
    }

    /// Like `apply_with_history` but always pushes a new undo event even if
    /// it would otherwise coalesce. Mirrors CM6's `isolateHistory.full`.
    pub fn apply_with_history_isolated(&self, tr: &Transaction) -> EditorState {
        self.apply_with_history_inner(tr, /*allow_join=*/ false)
    }

    fn apply_with_history_inner(&self, tr: &Transaction, allow_join: bool) -> EditorState {
        let inverse_changes = tr.invert(&self.doc).changes;
        let mut new_state = self.apply(tr);
        // Track selection change for undo_selection.
        if new_state.selection != self.selection {
            new_state
                .history
                .selection_undo
                .push(self.selection.clone());
            new_state.history.selection_redo.clear();
        }
        // Selection-only transactions don't go on the content undo stack.
        if tr.changes.is_empty() {
            return new_state;
        }
        let new_event = HistoryEvent {
            forward: tr.changes.clone(),
            inverse: inverse_changes,
            selection_before: self.selection.clone(),
            selection_after: new_state.selection.clone(),
        };
        if allow_join {
            if let Some(top) = new_state.history.undo.last_mut() {
                if can_join(top, &new_event) {
                    join_into(top, &new_event);
                    new_state.history.redo.clear();
                    return new_state;
                }
            }
        }
        new_state.history.undo.push(new_event);
        new_state.history.redo.clear();
        new_state
    }
}

fn can_join(prev: &HistoryEvent, new: &HistoryEvent) -> bool {
    if prev.forward.len() != 1 || new.forward.len() != 1 {
        return false;
    }
    let pf = &prev.forward[0];
    let nf = &new.forward[0];
    let pf_insert = pf.from == pf.to && !pf.insert.is_empty();
    let nf_insert = nf.from == nf.to && !nf.insert.is_empty();
    let pf_delete = pf.from < pf.to && pf.insert.is_empty();
    let nf_delete = nf.from < nf.to && nf.insert.is_empty();
    if pf_insert && nf_insert {
        // Forward typing: previous insert ends where new insert begins.
        let prev_end = pf.from + pf.insert.chars().count();
        nf.from == prev_end
    } else if pf_delete && nf_delete {
        // Backspace chain: previous delete starts where new delete ends.
        nf.to == pf.from
    } else {
        false
    }
}

fn join_into(top: &mut HistoryEvent, new: &HistoryEvent) {
    let pf = &top.forward[0];
    let nf = &new.forward[0];
    if pf.from == pf.to {
        // Two pure insertions; concatenate text.
        let combined_text = format!("{}{}", pf.insert, nf.insert);
        let combined_len = combined_text.chars().count();
        let from = pf.from;
        top.forward = vec![Change::new(from, from, combined_text)];
        top.inverse = vec![Change::delete(from, from + combined_len)];
    } else {
        // Two pure deletions; combined deleted text is new_deleted + prev_deleted.
        let new_inv_text = new.inverse[0].insert.clone();
        let prev_inv_text = top.inverse[0].insert.clone();
        let combined_deleted = format!("{}{}", new_inv_text, prev_inv_text);
        top.forward = vec![Change::delete(nf.from, pf.to)];
        top.inverse = vec![Change::insert(nf.from, combined_deleted)];
    }
    top.selection_after = new.selection_after.clone();
}

pub fn undo(state: &EditorState) -> Option<EditorState> {
    let event = state.history.undo.last()?.clone();
    let tr = Transaction {
        changes: event.inverse.clone(),
        selection: Some(event.selection_before.clone()),
    };
    let mut new_state = state.apply(&tr);
    new_state.history.undo.pop();
    new_state.history.redo.push(event);
    Some(new_state)
}

pub fn redo(state: &EditorState) -> Option<EditorState> {
    let event = state.history.redo.last()?.clone();
    let tr = Transaction {
        changes: event.forward.clone(),
        selection: Some(event.selection_after.clone()),
    };
    let mut new_state = state.apply(&tr);
    new_state.history.redo.pop();
    new_state.history.undo.push(event);
    Some(new_state)
}

pub fn undo_depth(state: &EditorState) -> usize {
    state.history.undo.len()
}

pub fn redo_depth(state: &EditorState) -> usize {
    state.history.redo.len()
}

/// Pop the most recent selection change off the selection-undo stack and
/// restore the prior selection. Doesn't touch document content.
pub fn undo_selection(state: &EditorState) -> Option<EditorState> {
    let prev = state.history.selection_undo.last().cloned()?;
    let mut new_state = state.clone();
    new_state.history.selection_undo.pop();
    new_state.history.selection_redo.push(state.selection.clone());
    new_state.selection = prev;
    Some(new_state)
}

pub fn redo_selection(state: &EditorState) -> Option<EditorState> {
    let next = state.history.selection_redo.last().cloned()?;
    let mut new_state = state.clone();
    new_state.history.selection_redo.pop();
    new_state.history.selection_undo.push(state.selection.clone());
    new_state.selection = next;
    Some(new_state)
}
