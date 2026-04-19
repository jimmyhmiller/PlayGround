//! Subset of CodeMirror 6 `commands/test/test-history.ts`.
//! No event coalescing implemented — each `apply_with_history` is its own
//! undoable event. Tests that depend on `newGroupDelay`-style merging are
//! adapted (one undo per type) or skipped.

use editor_core::history::{redo, redo_depth, redo_selection, undo, undo_depth, undo_selection};
use editor_core::selection::Selection;
use editor_core::state::EditorState;
use editor_core::transaction::{Change, Transaction};

fn mk_state(doc: &str) -> EditorState {
    EditorState::from_str(doc)
}

fn type_at(state: &EditorState, text: &str, at: usize) -> EditorState {
    let tr = Transaction::new()
        .change(Change::insert(at, text))
        .select(Selection::cursor(at + text.chars().count()));
    state.apply_with_history(&tr)
}

fn type_end(state: &EditorState, text: &str) -> EditorState {
    let at = state.doc.len_chars();
    type_at(state, text, at)
}

#[test]
fn allows_to_undo_a_change() {
    let state = mk_state("");
    let state = type_end(&state, "newtext");
    let state = undo(&state).expect("undo applies");
    assert_eq!(state.doc.to_string(), "");
}

#[test]
fn allows_to_redo_a_change() {
    let state = mk_state("");
    let state = type_end(&state, "newtext");
    let state = undo(&state).expect("undo applies");
    let state = redo(&state).expect("redo applies");
    assert_eq!(state.doc.to_string(), "newtext");
}

#[test]
fn tracks_multiple_levels_of_history() {
    // Direct port of CM6 — `type` calls at adjacent positions coalesce.
    let state = mk_state("one");
    let state = type_end(&state, "new"); // event 1: insert "new" at end
    let state = type_end(&state, "text"); // joins event 1 → event 1 = insert "newtext"
    let state = type_at(&state, "some", 0); // event 2: insert "some" at 0
    assert_eq!(state.doc.to_string(), "someonenewtext");
    let state = undo(&state).expect("undo 1");
    assert_eq!(state.doc.to_string(), "onenewtext");
    let state = undo(&state).expect("undo 2");
    assert_eq!(state.doc.to_string(), "one");
    let state = redo(&state).expect("redo 1");
    assert_eq!(state.doc.to_string(), "onenewtext");
    let state = redo(&state).expect("redo 2");
    assert_eq!(state.doc.to_string(), "someonenewtext");
    let state = undo(&state).expect("undo 3");
    assert_eq!(state.doc.to_string(), "onenewtext");
}

#[test]
fn allows_to_undo_nearby_changes_in_one_change() {
    let state = mk_state("");
    let state = type_end(&state, "new");
    let state = type_end(&state, "text");
    let state = undo(&state).expect("undo");
    assert_eq!(state.doc.to_string(), "");
}

#[test]
fn allows_to_redo_nearby_changes_in_one_change() {
    let state = mk_state("");
    let state = type_end(&state, "new");
    let state = type_end(&state, "text");
    let state = undo(&state).expect("undo");
    let state = redo(&state).expect("redo");
    assert_eq!(state.doc.to_string(), "newtext");
}

#[test]
fn isolated_apply_skips_join() {
    let state = mk_state("");
    let tr1 = Transaction::new()
        .change(Change::insert(0, "a"))
        .select(Selection::cursor(1));
    let state = state.apply_with_history_isolated(&tr1);
    let tr2 = Transaction::new()
        .change(Change::insert(1, "b"))
        .select(Selection::cursor(2));
    let state = state.apply_with_history_isolated(&tr2);
    assert_eq!(undo_depth(&state), 2);
    let state = undo(&state).expect("undo 1");
    assert_eq!(state.doc.to_string(), "a");
    let state = undo(&state).expect("undo 2");
    assert_eq!(state.doc.to_string(), "");
}

#[test]
fn puts_cursor_after_change_on_redo() {
    // Mirrors CM6: insert "!" at position 3 with explicit cursor at 4,
    // then move cursor elsewhere, undo, redo, expect cursor back at 4.
    let state = EditorState::new(
        ropey::Rope::from_str("one\n\ntwo"),
        Selection::cursor(0),
    );
    // Insert "!" at 3 with explicit selection at 4.
    let tr = Transaction::new()
        .change(Change::insert(3, "!"))
        .select(Selection::cursor(4));
    let state = state.apply_with_history(&tr);

    // Move the cursor to end (no history change — pure selection isn't an
    // edit). For this MVP, pure selection updates aren't part of history;
    // we just bypass with apply.
    let move_tr = Transaction::new().select(Selection::cursor(state.doc.len_chars()));
    let state = state.apply(&move_tr);

    let state = undo(&state).expect("undo");
    let state = redo(&state).expect("redo");
    assert_eq!(state.selection.main().head, 4);
}

#[test]
fn undo_at_empty_history_returns_none() {
    let state = mk_state("hello");
    assert!(undo(&state).is_none());
}

#[test]
fn redo_at_empty_history_returns_none() {
    let state = mk_state("hello");
    let state = type_end(&state, "!");
    let state = undo(&state).expect("undo");
    let state = redo(&state).expect("redo");
    assert!(redo(&state).is_none());
}

#[test]
fn new_change_after_undo_clears_redo_stack() {
    // Use type_at(..., 0) to alternate insertion sites and avoid coalescing.
    let state = mk_state("");
    let state = type_at(&state, "a", 0); // event 1: "a"
    let state = type_at(&state, "b", 0); // event 2: prepend "b" → "ba" (not adjacent)
    assert_eq!(undo_depth(&state), 2);
    let state = undo(&state).expect("undo b");
    assert_eq!(redo_depth(&state), 1);
    let state = type_at(&state, "c", 0); // event 3 (clears redo)
    assert_eq!(redo_depth(&state), 0);
    assert_eq!(undo_depth(&state), 2);
    assert_eq!(state.doc.to_string(), "ca");
}

#[test]
fn undo_redo_round_trip_doc_invariant() {
    let state = mk_state("hello");
    let state = type_end(&state, " world");
    let original_doc = "hello world".to_string();
    let undone = undo(&state).expect("undo");
    assert_eq!(undone.doc.to_string(), "hello");
    let redone = redo(&undone).expect("redo");
    assert_eq!(redone.doc.to_string(), original_doc);
}

#[test]
fn restores_selection_on_undo_redo_undo() {
    // Port of CM6's "restores selection on undo-redo-undo".
    // Set up: doc = "1\n2\n3" (5 chars), cursor at 5 (end), then insert "."
    // making doc "1\n2\n3." with cursor at 6, then move cursor elsewhere,
    // then undo / redo / undo and verify selection bounces back correctly.
    let state = mk_state("1\n2\n3");
    let move_to = |st: &EditorState, p: usize| {
        st.apply(&Transaction::new().select(Selection::cursor(p)))
    };

    let state = move_to(&state, 5);
    let tr = Transaction::new()
        .change(Change::insert(5, "."))
        .select(Selection::cursor(6));
    let state = state.apply_with_history(&tr);
    let state = move_to(&state, 0);

    let state = undo(&state).expect("undo");
    assert_eq!(state.selection.main().head, 5);

    let state = move_to(&state, 0);
    let state = redo(&state).expect("redo");
    assert_eq!(state.selection.main().head, 6);

    let state = move_to(&state, 0);
    let state = undo(&state).expect("undo");
    assert_eq!(state.selection.main().head, 5);
}

#[test]
fn undo_selection_steps_back_through_selection_changes() {
    let state = mk_state("hello world");
    let move_to = |st: &EditorState, p: usize| {
        st.apply_with_history(&Transaction::new().select(Selection::cursor(p)))
    };
    let state = move_to(&state, 5); // sel: 0 -> 5
    let state = move_to(&state, 3); // sel: 5 -> 3
    let state = move_to(&state, 7); // sel: 3 -> 7
    assert_eq!(state.selection.main().head, 7);

    let state = undo_selection(&state).expect("undo sel 1");
    assert_eq!(state.selection.main().head, 3);
    let state = undo_selection(&state).expect("undo sel 2");
    assert_eq!(state.selection.main().head, 5);
    let state = undo_selection(&state).expect("undo sel 3");
    assert_eq!(state.selection.main().head, 0);
    assert!(undo_selection(&state).is_none());

    let state = redo_selection(&state).expect("redo sel");
    assert_eq!(state.selection.main().head, 5);
}

#[test]
fn undo_selection_doesnt_touch_doc() {
    let state = mk_state("hello");
    let state = type_end(&state, " world");
    let move_state = state.apply_with_history(
        &Transaction::new().select(Selection::cursor(0)),
    );
    assert_eq!(move_state.doc.to_string(), "hello world");
    let undone = undo_selection(&move_state).expect("undo sel");
    // Doc unchanged.
    assert_eq!(undone.doc.to_string(), "hello world");
    // Selection rolled back to where it was after the type.
    assert_eq!(undone.selection.main().head, 11);
}

#[test]
fn deep_undo_redo_chain() {
    // Use isolated apply so each char is its own event regardless of joining.
    let mut state = mk_state("");
    let chunks = ["a", "b", "c", "d", "e"];
    for (i, c) in chunks.iter().enumerate() {
        let tr = Transaction::new()
            .change(Change::insert(i, c.to_string()))
            .select(Selection::cursor(i + 1));
        state = state.apply_with_history_isolated(&tr);
    }
    assert_eq!(state.doc.to_string(), "abcde");
    assert_eq!(undo_depth(&state), 5);
    for _ in 0..5 {
        state = undo(&state).expect("undo");
    }
    assert_eq!(state.doc.to_string(), "");
    assert_eq!(redo_depth(&state), 5);
    for _ in 0..5 {
        state = redo(&state).expect("redo");
    }
    assert_eq!(state.doc.to_string(), "abcde");
}
