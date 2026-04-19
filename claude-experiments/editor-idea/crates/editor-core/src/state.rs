use ropey::Rope;

use crate::comment::CommentTokens;
use crate::history::History;
use crate::indent::IndentRules;
use crate::selection::{Range, Selection};
use crate::transaction::{map_pos, Transaction};

#[derive(Debug, Clone)]
pub struct EditorState {
    pub doc: Rope,
    pub selection: Selection,
    pub indent_unit: String,
    pub comment_tokens: CommentTokens,
    pub indent_rules: IndentRules,
    pub history: History,
}

impl EditorState {
    pub fn new(doc: impl Into<Rope>, selection: Selection) -> Self {
        Self {
            doc: doc.into(),
            selection,
            indent_unit: "  ".into(),
            comment_tokens: CommentTokens::default(),
            indent_rules: IndentRules::default(),
            history: History::default(),
        }
    }

    pub fn with_indent_unit(mut self, unit: impl Into<String>) -> Self {
        self.indent_unit = unit.into();
        self
    }

    pub fn from_str(text: &str) -> Self {
        Self::new(Rope::from_str(text), Selection::cursor(0))
    }

    pub fn with_comment_tokens(mut self, tokens: CommentTokens) -> Self {
        self.comment_tokens = tokens;
        self
    }

    pub fn apply(&self, tr: &Transaction) -> EditorState {
        let mut tr = tr.clone();
        tr.normalize();

        let mut doc = self.doc.clone();
        // Apply changes in reverse so earlier positions stay valid.
        for c in tr.changes.iter().rev() {
            if c.from != c.to {
                doc.remove(c.from..c.to);
            }
            if !c.insert.is_empty() {
                doc.insert(c.from, &c.insert);
            }
        }

        let selection = tr.selection.unwrap_or_else(|| {
            // Default position mapping (matches CM6): non-empty ranges
            // contain insertions at their boundaries (from with assoc=-1,
            // to with assoc=+1); empty cursors stick to the right of any
            // insertion at their position (both with assoc=+1).
            let ranges = self
                .selection
                .ranges
                .iter()
                .map(|r| {
                    // CM6 default: empty cursors stick after insertions
                    // (assoc=+1). For non-empty ranges, an insertion at the
                    // boundary stays *outside* the range, so the from edge
                    // sticks right (assoc=+1) and the to edge sticks left
                    // (assoc=-1). Backward ranges flip those.
                    let (anchor_assoc, head_assoc) = if r.is_empty() {
                        (1, 1)
                    } else if r.anchor < r.head {
                        (1, -1)
                    } else {
                        (-1, 1)
                    };
                    Range::new(
                        map_pos(&tr.changes, r.anchor, anchor_assoc),
                        map_pos(&tr.changes, r.head, head_assoc),
                    )
                })
                .collect();
            Selection::new(ranges, self.selection.primary)
        });

        let selection = selection.map(&doc);
        EditorState {
            doc,
            selection,
            indent_unit: self.indent_unit.clone(),
            comment_tokens: self.comment_tokens.clone(),
            indent_rules: self.indent_rules.clone(),
            history: self.history.clone(),
        }
    }
}
