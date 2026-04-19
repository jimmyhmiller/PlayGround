//! Subset of CodeMirror 6 `state/test/test-text.ts` ports.
//! Many CM6 Text tests exercise CM6's own balanced-tree internals (rebalance
//! depth, node boundaries) — those are skipped because we use ropey, which
//! has its own internal structure. Tests that exercise *behaviour* of
//! replace/append/slice/eq are ported as exercises of our Transaction layer
//! over ropey.

use editor_core::selection::Selection;
use editor_core::state::EditorState;
use editor_core::transaction::{Change, Transaction};
use ropey::Rope;

fn doc_of(lines: &[&str]) -> Rope {
    Rope::from_str(&lines.join("\n"))
}

fn replace(doc: &Rope, from: usize, to: usize, with: &str) -> Rope {
    let state = EditorState::new(doc.clone(), Selection::cursor(0));
    let tr = Transaction::new().change(Change::new(from, to, with.to_string()));
    state.apply(&tr).doc
}

#[test]
fn handles_basic_replacement() {
    let doc = doc_of(&["one", "two", "three"]);
    let after = replace(&doc, 2, 5, "foo\nbar");
    assert_eq!(after.to_string(), "onfoo\nbarwo\nthree");
}

#[test]
fn can_append_documents() {
    let doc = doc_of(&["one", "two", "three"]);
    let after = replace(&doc, doc.len_chars(), doc.len_chars(), "!\nok");
    assert_eq!(after.to_string(), "one\ntwo\nthree!\nok");
}

#[test]
fn can_handle_deleting_entire_document() {
    let doc = doc_of(&["one", "two", "three"]);
    let after = replace(&doc, 0, doc.len_chars(), "");
    assert_eq!(after.to_string(), "");
}

#[test]
fn can_handle_deleting_at_start() {
    let doc = Rope::from_str("abcdefghij");
    let after = replace(&doc, 0, 3, "");
    assert_eq!(after.to_string(), "defghij");
}

#[test]
fn can_handle_deleting_at_end() {
    let doc = Rope::from_str("abcdefghij");
    let after = replace(&doc, 7, 10, "");
    assert_eq!(after.to_string(), "abcdefg");
}

#[test]
fn can_insert_at_node_boundaries() {
    // CM6 had a specific test for inserting on internal tree boundaries; for
    // ropey we just verify mid-doc inserts compose normally.
    let doc = Rope::from_str("0123456789");
    let after = replace(&doc, 5, 5, "XYZ");
    assert_eq!(after.to_string(), "01234XYZ56789");
}

#[test]
fn can_build_up_doc_by_repeated_appending() {
    let mut doc = Rope::new();
    let mut text = String::new();
    for i in 1..200 {
        let add = format!("newtext{i} ");
        doc = replace(&doc, doc.len_chars(), doc.len_chars(), &add);
        text.push_str(&add);
    }
    assert_eq!(doc.to_string(), text);
}

#[test]
fn maintains_content_during_random_editing() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    fn rng(seed: &mut u64) -> u64 {
        let mut h = DefaultHasher::new();
        seed.hash(&mut h);
        *seed = h.finish();
        *seed
    }
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut text: String = (0..200).map(|_| ('a' as u8 + (rng(&mut seed) % 26) as u8) as char).collect();
    let mut doc = Rope::from_str(&text);
    for _ in 0..100 {
        let len = doc.len_chars();
        if len == 0 {
            break;
        }
        let ins_pos = (rng(&mut seed) as usize) % len;
        let ins_char = ('A' as u8 + (rng(&mut seed) % 26) as u8) as char;
        let ins_str = ins_char.to_string();
        text.insert(ins_pos, ins_char);
        doc = replace(&doc, ins_pos, ins_pos, &ins_str);
        let new_len = doc.len_chars();
        let del_from = (rng(&mut seed) as usize) % new_len;
        let del_to = (del_from + (rng(&mut seed) as usize) % 20).min(new_len);
        text.replace_range(del_from..del_to, "");
        doc = replace(&doc, del_from, del_to, "");
        assert_eq!(doc.to_string(), text);
    }
}

#[test]
fn returns_correct_strings_for_slice() {
    // CM6 builds a 4-digit-zero-padded integer doc and slices random ranges.
    let mut text = String::new();
    for i in 0..100 {
        text.push_str(&format!("{:04}", i));
        text.push('\n');
    }
    let doc = Rope::from_str(&text);
    // A few deterministic slices.
    for (from, to) in [(0, 5), (10, 50), (200, 250), (0, doc.len_chars()), (100, 100)] {
        assert_eq!(doc.slice(from..to).to_string(), text[from..to]);
    }
}

#[test]
fn rope_equality_is_content_based() {
    let a = doc_of(&["foo", "bar"]);
    let b = doc_of(&["foo", "bar"]);
    assert_eq!(a, b);
    let c = doc_of(&["foo", "baz"]);
    assert_ne!(a, c);
}
