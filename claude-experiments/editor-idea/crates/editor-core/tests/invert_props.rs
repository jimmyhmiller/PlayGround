//! Property tests for `Transaction::invert`.
//! The core invariant: `apply(invert(t, d), apply(t, d)) == d`.

use editor_core::selection::Selection;
use editor_core::state::EditorState;
use editor_core::transaction::{Change, Transaction};
use proptest::prelude::*;
use ropey::Rope;

/// Generate a doc and a list of non-overlapping changes against it.
/// Each change is `(from, to, insert_text)` with `from <= to <= doc.len_chars()`.
fn arb_doc_and_changes() -> impl Strategy<Value = (String, Vec<(usize, usize, String)>)> {
    "[a-z\n ]{0,80}".prop_flat_map(|doc| {
        let len = doc.chars().count();
        let edits = prop::collection::vec(
            (0usize..=len.max(1), 0usize..=len.max(1), "[a-zA-Z\n ]{0,8}"),
            0..=6,
        );
        (Just(doc), edits).prop_map(move |(d, raw)| {
            // Clamp + normalize each (a, b) into (from, to) within doc bounds,
            // sort ascending, drop any that overlap a previous one.
            let len = d.chars().count();
            let mut spans: Vec<(usize, usize, String)> = raw
                .into_iter()
                .map(|(a, b, ins)| {
                    let a = a.min(len);
                    let b = b.min(len);
                    (a.min(b), a.max(b), ins)
                })
                .collect();
            spans.sort_by_key(|s| s.0);
            let mut clean: Vec<(usize, usize, String)> = Vec::new();
            let mut last_end = 0;
            for (f, t, ins) in spans {
                if f >= last_end {
                    clean.push((f, t, ins));
                    last_end = t;
                }
            }
            (d, clean)
        })
    })
}

fn build_tx(spans: &[(usize, usize, String)]) -> Transaction {
    Transaction::new().changes(
        spans
            .iter()
            .map(|(f, t, ins)| Change::new(*f, *t, ins.clone())),
    )
}

fn apply(rope: &Rope, tr: &Transaction) -> Rope {
    let state = EditorState::new(rope.clone(), Selection::cursor(0));
    state.apply(tr).doc
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn invert_round_trip((doc, spans) in arb_doc_and_changes()) {
        let rope = Rope::from_str(&doc);
        let tr = build_tx(&spans);
        let new_doc = apply(&rope, &tr);
        let inv = tr.invert(&rope);
        let back = apply(&new_doc, &inv);
        prop_assert_eq!(back.to_string(), doc);
    }

    #[test]
    fn invert_invert_is_apply((doc, spans) in arb_doc_and_changes()) {
        // invert(invert(t, d), apply(t, d)) should produce a transaction that,
        // applied to d, yields apply(t, d).
        let rope = Rope::from_str(&doc);
        let tr = build_tx(&spans);
        let new_doc = apply(&rope, &tr);
        let inv = tr.invert(&rope);
        let inv_inv = inv.invert(&new_doc);
        let result = apply(&rope, &inv_inv);
        prop_assert_eq!(result.to_string(), new_doc.to_string());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn compose_equals_sequential_apply(
        (doc, a_spans) in arb_doc_and_changes(),
        b_spec in prop::collection::vec((0usize..=80, 0usize..=80, "[a-z\n ]{0,5}"), 0..=4),
    ) {
        let rope = Rope::from_str(&doc);
        let a = build_tx(&a_spans);
        let mid = apply(&rope, &a);
        // Build b's changes against mid, normalize the same way.
        let len = mid.len_chars();
        let mut spans: Vec<(usize, usize, String)> = b_spec
            .into_iter()
            .map(|(x, y, ins)| {
                let x = x.min(len);
                let y = y.min(len);
                (x.min(y), x.max(y), ins)
            })
            .collect();
        spans.sort_by_key(|s| s.0);
        let mut clean: Vec<(usize, usize, String)> = Vec::new();
        let mut last_end = 0;
        for (f, t, ins) in spans {
            if f >= last_end {
                clean.push((f, t, ins));
                last_end = t;
            }
        }
        let b = build_tx(&clean);
        let composed = a.compose(&rope, &b);
        let by_compose = apply(&rope, &composed);
        let by_sequence = apply(&mid, &b);
        prop_assert_eq!(by_compose.to_string(), by_sequence.to_string());
    }

    #[test]
    fn compose_with_empty_is_identity(
        (doc, spans) in arb_doc_and_changes(),
    ) {
        let rope = Rope::from_str(&doc);
        let a = build_tx(&spans);
        let empty = Transaction::new();
        let composed_left = empty.compose(&rope, &a);
        let composed_right = a.compose(&rope, &empty);
        prop_assert_eq!(
            apply(&rope, &composed_left).to_string(),
            apply(&rope, &a).to_string()
        );
        prop_assert_eq!(
            apply(&rope, &composed_right).to_string(),
            apply(&rope, &a).to_string()
        );
    }
}

#[test]
fn invert_handcrafted_pure_insert() {
    let doc = Rope::from_str("hello");
    let tr = Transaction::new().change(Change::insert(5, " world"));
    let new = apply(&doc, &tr);
    assert_eq!(new.to_string(), "hello world");
    let inv = tr.invert(&doc);
    let back = apply(&new, &inv);
    assert_eq!(back.to_string(), "hello");
}

#[test]
fn invert_handcrafted_pure_delete() {
    let doc = Rope::from_str("hello world");
    let tr = Transaction::new().change(Change::delete(5, 11));
    let new = apply(&doc, &tr);
    assert_eq!(new.to_string(), "hello");
    let inv = tr.invert(&doc);
    let back = apply(&new, &inv);
    assert_eq!(back.to_string(), "hello world");
}

#[test]
fn invert_handcrafted_replace_and_shift() {
    // Two non-overlapping changes that change the doc length differently.
    let doc = Rope::from_str("aaaa....bbbb");
    let tr = Transaction::new()
        .change(Change::new(0, 4, "X")) // shrink: 4 chars -> 1
        .change(Change::new(8, 12, "YYYYY")); // grow: 4 chars -> 5
    let new = apply(&doc, &tr);
    assert_eq!(new.to_string(), "X....YYYYY");
    let inv = tr.invert(&doc);
    let back = apply(&new, &inv);
    assert_eq!(back.to_string(), "aaaa....bbbb");
}
