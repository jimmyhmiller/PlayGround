use crate::selection::Selection;

/// A single replacement: replace `from..to` (char indices into the doc the
/// transaction was built against) with `insert`.
///
/// All `from`/`to` in a `Transaction` refer to the *original* doc — sibling
/// changes do not shift each other. Mirrors CodeMirror 6's `ChangeSpec` model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub from: usize,
    pub to: usize,
    pub insert: String,
}

impl Change {
    pub fn new(from: usize, to: usize, insert: impl Into<String>) -> Self {
        assert!(from <= to);
        Self { from, to, insert: insert.into() }
    }

    pub fn insert(at: usize, text: impl Into<String>) -> Self {
        Self::new(at, at, text)
    }

    pub fn delete(from: usize, to: usize) -> Self {
        Self::new(from, to, "")
    }

    pub fn delta(&self) -> isize {
        self.insert.chars().count() as isize - (self.to - self.from) as isize
    }
}

/// Map a position through a set of sibling changes.
/// `assoc < 0` biases the position to before any inserted text at `pos`,
/// `assoc >= 0` biases after.
pub fn map_pos(changes: &[Change], pos: usize, assoc: i32) -> usize {
    let mut shift: isize = 0;
    for c in changes {
        if c.to < pos || (c.to == pos && assoc >= 0 && c.from != c.to) {
            shift += c.delta();
        } else if c.from < pos || (c.from == pos && assoc >= 0) {
            let inserted = c.insert.chars().count() as isize;
            return (c.from as isize + shift + inserted) as usize;
        } else {
            break;
        }
    }
    (pos as isize + shift) as usize
}

#[derive(Debug, Clone, Default)]
pub struct Transaction {
    pub changes: Vec<Change>,
    pub selection: Option<Selection>,
}

impl Transaction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn change(mut self, c: Change) -> Self {
        self.changes.push(c);
        self
    }

    pub fn changes(mut self, cs: impl IntoIterator<Item = Change>) -> Self {
        self.changes.extend(cs);
        self
    }

    pub fn select(mut self, sel: Selection) -> Self {
        self.selection = Some(sel);
        self
    }

    /// Sort changes ascending by `from`. Required by `apply` and `map_pos`.
    pub fn normalize(&mut self) {
        self.changes.sort_by_key(|c| c.from);
        // Sibling changes must not overlap.
        for w in self.changes.windows(2) {
            assert!(w[0].to <= w[1].from, "overlapping changes in transaction");
        }
    }

    /// Compose this transaction with `other` against the original `doc`.
    /// `other` is interpreted as acting on the doc *after* `self` is applied.
    /// Result invariant: applying the returned transaction to `doc` yields
    /// the same final doc as applying `self` then `other`.
    ///
    /// This implementation diffs the start and end docs and returns a single
    /// minimal `Change`. It loses the granularity of the intermediate
    /// changes but preserves the application result. Selection is dropped.
    pub fn compose(&self, doc: &ropey::Rope, other: &Transaction) -> Transaction {
        use crate::selection::Selection;
        use crate::state::EditorState;
        let mid = EditorState::new(doc.clone(), Selection::cursor(0))
            .apply(self)
            .doc;
        let end = EditorState::new(mid, Selection::cursor(0))
            .apply(other)
            .doc;
        let original: String = doc.to_string();
        let final_str: String = end.to_string();
        let original_chars: Vec<char> = original.chars().collect();
        let final_chars: Vec<char> = final_str.chars().collect();
        let common_prefix = original_chars
            .iter()
            .zip(final_chars.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let max_suffix = original_chars.len().min(final_chars.len()) - common_prefix;
        let common_suffix = original_chars
            .iter()
            .rev()
            .zip(final_chars.iter().rev())
            .take_while(|(a, b)| a == b)
            .take(max_suffix)
            .count();
        let from = common_prefix;
        let to = original_chars.len() - common_suffix;
        let insert: String = final_chars
            [common_prefix..final_chars.len() - common_suffix]
            .iter()
            .collect();
        let mut tx = Transaction::new();
        if !(from == to && insert.is_empty()) {
            tx = tx.change(Change::new(from, to, insert));
        }
        tx
    }

    /// Build the inverse: a transaction that, applied to `apply(self, doc)`,
    /// yields `doc`. Property: `apply(self.invert(d), apply(self, d)) == d`.
    /// `doc` must be the same doc this transaction was built against.
    pub fn invert(&self, doc: &ropey::Rope) -> Transaction {
        let mut sorted = self.changes.clone();
        sorted.sort_by_key(|c| c.from);
        let mut inverse = Vec::with_capacity(sorted.len());
        let mut shift: isize = 0;
        for c in &sorted {
            let removed: String = doc.slice(c.from..c.to).to_string();
            let insert_chars = c.insert.chars().count();
            let new_from = (c.from as isize + shift) as usize;
            let new_to = new_from + insert_chars;
            inverse.push(Change::new(new_from, new_to, removed));
            shift += insert_chars as isize - (c.to - c.from) as isize;
        }
        Transaction { changes: inverse, selection: None }
    }
}
