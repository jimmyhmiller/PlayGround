use crate::transform::*;
use crate::types::*;
use serde::{Deserialize, Serialize};

/// Represents the differences between two documents A and B,
/// tracked from their common agreement A&B.
///
/// ```text
/// A ←─aₙ─ ... ←─a₁─ A&B ─b₁─→ ... ─bₘ─→ B
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Differences {
    /// Edits from agreement to A (stored in order: a₁ is first, aₙ is last)
    pub a_diffs: Vec<Edit>,
    /// Edits from agreement to B (stored in order: b₁ is first, bₘ is last)
    pub b_diffs: Vec<Edit>,
}

/// Result of translating an edit through the difference sequences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResult {
    /// The translated edit (δ on the B side)
    pub delta: Edit,
    /// Updated a differences
    pub a_diffs: Vec<Edit>,
    /// Updated b differences
    pub b_diffs: Vec<Edit>,
}

/// Translate an edit `epsilon` (applied to A) through the differences.
///
/// 1. Retract epsilon backwards through all a_diffs (aₙ, ..., a₁)
/// 2. Project the result forwards through all b_diffs (b₁, ..., bₘ)
///
/// Returns None if any retraction is impossible (dependency).
pub fn translate(
    epsilon: &Edit,
    a_diffs: &[Edit],
    b_diffs: &[Edit],
) -> Option<TranslateResult> {
    let mut current = epsilon.clone();
    let mut new_a_diffs = a_diffs.to_vec();

    // Retract through a_diffs in reverse order
    for i in (0..new_a_diffs.len()).rev() {
        let result = retract(&current, &new_a_diffs[i])?;
        current = result.edit;
        new_a_diffs[i] = result.adjust;
    }

    // Now current is the edit at the agreement level (ε₀)
    let mut new_b_diffs = b_diffs.to_vec();

    // Project through b_diffs in forward order
    for i in 0..new_b_diffs.len() {
        let result = project(&current, &new_b_diffs[i]);
        current = result.edit;
        new_b_diffs[i] = result.adjust;
    }

    Some(TranslateResult {
        delta: current,
        a_diffs: new_a_diffs,
        b_diffs: new_b_diffs,
    })
}

impl Differences {
    pub fn new() -> Self {
        Differences {
            a_diffs: vec![],
            b_diffs: vec![],
        }
    }

    /// Apply an edit to document A and update the differences.
    /// Returns the translated edit on B (delta), or None if retraction fails
    /// (but the edit is STILL appended — dependent edits are valid, they just
    /// can't be independently migrated until their dependency is migrated first).
    pub fn edit_a(&mut self, epsilon: &Edit) -> Option<Edit> {
        match translate(epsilon, &self.a_diffs, &self.b_diffs) {
            Some(result) if result.delta.is_id() => {
                // Edit was absorbed into the agreement
                self.a_diffs = result.a_diffs;
                self.b_diffs = result.b_diffs;
                Some(result.delta)
            }
            Some(result) => {
                // Edit increases differences: append
                self.a_diffs.push(epsilon.clone());
                Some(result.delta)
            }
            None => {
                // Translation failed (dependency). The edit still happened on A,
                // so it must be tracked. Append it — it depends on an earlier
                // edit and must be migrated after that dependency.
                self.a_diffs.push(epsilon.clone());
                None
            }
        }
    }

    /// Apply an edit to document B and update the differences.
    pub fn edit_b(&mut self, epsilon: &Edit) -> Option<Edit> {
        match translate(epsilon, &self.b_diffs, &self.a_diffs) {
            Some(result) if result.delta.is_id() => {
                self.b_diffs = result.a_diffs;
                self.a_diffs = result.b_diffs;
                Some(result.delta)
            }
            Some(result) => {
                self.b_diffs.push(epsilon.clone());
                Some(result.delta)
            }
            None => {
                self.b_diffs.push(epsilon.clone());
                None
            }
        }
    }

    /// Migrate the first (earliest) difference of A to B.
    ///
    /// This is the simplest form: a_diffs[0] is already at the agreement level,
    /// so we just project it through b_diffs. No retraction needed.
    /// The remaining a_diffs [a_1..a_n] stay valid because they're relative
    /// to the new agreement (= old agreement + a_0).
    pub fn migrate_first_a_to_b(&mut self) -> Option<Edit> {
        if self.a_diffs.is_empty() {
            return None;
        }

        // Remove the first diff (it's at the agreement level)
        let edit = self.a_diffs.remove(0);

        // Project through b_diffs to get delta
        let mut delta = edit;
        let mut new_b_diffs = self.b_diffs.clone();

        for i in 0..new_b_diffs.len() {
            let result = project(&delta, &new_b_diffs[i]);
            delta = result.edit;
            new_b_diffs[i] = result.adjust;
        }

        // Always update b_diffs after migration. Unlike edit_a (where the paper
        // says to keep diffs unchanged when delta != Id), migration REMOVES an
        // edit from a_diffs and APPLIES it to B. The b_diffs must be adjusted
        // to account for the migrated edit's effect on the index space.
        self.b_diffs = new_b_diffs;

        Some(delta)
    }

    /// Migrate the first (earliest) difference of B to A.
    /// Symmetric to migrate_first_a_to_b.
    pub fn migrate_first_b_to_a(&mut self) -> Option<Edit> {
        if self.b_diffs.is_empty() {
            return None;
        }

        let edit = self.b_diffs.remove(0);

        let mut delta = edit;
        let mut new_a_diffs = self.a_diffs.clone();

        for i in 0..new_a_diffs.len() {
            let result = project(&delta, &new_a_diffs[i]);
            delta = result.edit;
            new_a_diffs[i] = result.adjust;
        }

        self.a_diffs = new_a_diffs;

        Some(delta)
    }

    /// Migrate a_diffs[idx] to B.
    /// For idx > 0, retract through preceding diffs to reach agreement level first.
    pub fn migrate_a_to_b(&mut self, idx: usize) -> Option<Edit> {
        if idx >= self.a_diffs.len() {
            return None;
        }

        if idx == 0 {
            return self.migrate_first_a_to_b();
        }

        // For non-first diffs: retract the edit through preceding diffs
        // to reach the agreement level, then project through b_diffs.
        let edit = self.a_diffs.remove(idx);

        // Retract through preceding diffs (in reverse) to reach agreement
        let mut current = edit.clone();
        let mut i = idx;
        while i > 0 {
            i -= 1;
            if let Some(result) = retract(&current, &self.a_diffs[i]) {
                current = result.edit;
                self.a_diffs[i] = result.adjust;
            } else {
                // Dependency: can't migrate. Put the edit back.
                self.a_diffs.insert(idx, edit);
                return None;
            }
        }

        // current is now at agreement level. Project through b_diffs.
        let mut delta = current;
        let mut new_b_diffs = self.b_diffs.clone();
        for j in 0..new_b_diffs.len() {
            let result = project(&delta, &new_b_diffs[j]);
            delta = result.edit;
            new_b_diffs[j] = result.adjust;
        }

        if delta.is_id() {
            self.b_diffs = new_b_diffs;
        }

        Some(delta)
    }

    /// Check if migrating a_diffs[i] would conflict with any b_diff.
    pub fn conflicts_a(&self, idx: usize) -> Vec<usize> {
        if idx >= self.a_diffs.len() {
            return vec![];
        }
        let edit = &self.a_diffs[idx];
        self.b_diffs.iter().enumerate()
            .filter(|(_, b)| edits_conflict(edit, b))
            .map(|(j, _)| j)
            .collect()
    }

    /// List all conflicts between a_diffs and b_diffs.
    pub fn all_conflicts(&self) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        for (i, a) in self.a_diffs.iter().enumerate() {
            for (j, b) in self.b_diffs.iter().enumerate() {
                if edits_conflict(a, b) {
                    conflicts.push(Conflict {
                        from_edit: a.clone(),
                        to_edit: b.clone(),
                        from_idx: i,
                        to_idx: j,
                    });
                }
            }
        }
        conflicts
    }
}

/// Check if two edits conflict (target the same location with the same kind).
/// Per the paper (p. 7): "conflicts that do occur are definite, and are
/// resolved just by choosing which one wins."
pub fn edits_conflict(a: &Edit, b: &Edit) -> bool {
    // Point edits (Conv, Rename, Set) conflict with same kind at same index
    let a_idx = point_idx(a);
    let b_idx = point_idx(b);
    if let (Some(ai), Some(bi)) = (a_idx, b_idx) {
        if ai == bi && std::mem::discriminant(a) == std::mem::discriminant(b) {
            return true;
        }
    }
    // Move conflicts
    match (a, b) {
        (Edit::Move { i: ai, .. }, Edit::Move { i: bi, .. }) => ai == bi,
        // Point edit at Move target
        (_, Edit::Move { i: mi, .. }) if a_idx == Some(*mi) => true,
        (Edit::Move { i: mi, .. }, _) if b_idx == Some(*mi) => true,
        _ => false,
    }
}

fn point_idx(edit: &Edit) -> Option<usize> {
    match edit {
        Edit::Conv { idx, .. } | Edit::Rename { idx, .. } | Edit::Set { idx, .. } => Some(*idx),
        _ => None,
    }
}

/// A conflict between two edits from different branches.
#[derive(Debug, Clone)]
pub struct Conflict {
    /// The edit from the "from" side
    pub from_edit: Edit,
    /// The edit from the "to" side that it conflicts with
    pub to_edit: Edit,
    /// Index of the from_edit in its diff list
    pub from_idx: usize,
    /// Index of the to_edit in its diff list
    pub to_idx: usize,
}

/// Strategy for resolving conflicts during merge.
pub trait ConflictResolver {
    /// Given a conflict, return which side wins: true = "from" wins, false = "to" wins.
    fn resolve(&mut self, conflict: &Conflict) -> bool;
}

/// Default: "from" always wins (the side being migrated takes priority).
pub struct FromWins;

impl ConflictResolver for FromWins {
    fn resolve(&mut self, _conflict: &Conflict) -> bool {
        true
    }
}

/// "to" always wins (the receiving side keeps its edits).
pub struct ToWins;

impl ConflictResolver for ToWins {
    fn resolve(&mut self, _conflict: &Conflict) -> bool {
        false
    }
}

/// Collect conflicts without resolving — returns them for manual resolution.
pub struct CollectConflicts {
    pub conflicts: Vec<Conflict>,
}

impl CollectConflicts {
    pub fn new() -> Self {
        CollectConflicts { conflicts: Vec::new() }
    }
}

impl Default for CollectConflicts {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictResolver for CollectConflicts {
    fn resolve(&mut self, conflict: &Conflict) -> bool {
        self.conflicts.push(conflict.clone());
        true // default to from wins, but caller can inspect conflicts
    }
}

impl Default for Differences {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply::*;

    #[test]
    fn test_translate_absorbs_same_edit() {
        // A and B both start from (num). B changes to (str).
        // Then A also changes to (str). The edit should be absorbed.
        let mut diffs = Differences::new();

        // B does Conv[0, str]
        let b_edit = Edit::Conv { idx: 0, ty: AtomicType::Str };
        diffs.edit_b(&b_edit);

        // A does Conv[0, str] (same thing)
        let a_edit = Edit::Conv { idx: 0, ty: AtomicType::Str };
        let delta = diffs.edit_a(&a_edit).unwrap();

        // Should be absorbed (delta = Id)
        assert_eq!(delta, Edit::Id);
        // No more differences
        assert!(diffs.a_diffs.is_empty() || diffs.a_diffs.iter().all(|e| e.is_id()));
        assert!(diffs.b_diffs.is_empty() || diffs.b_diffs.iter().all(|e| e.is_id()));
    }

    #[test]
    fn test_basic_migration() {
        let doc_a = Document::from_types(&[AtomicType::Num]);
        let doc_b = Document::from_types(&[AtomicType::Num]);

        let mut diffs = Differences::new();

        // A converts to str
        let a_edit = Edit::Conv { idx: 0, ty: AtomicType::Str };
        let _doc_a = apply(&doc_a, &a_edit).unwrap();
        diffs.edit_a(&a_edit);

        assert_eq!(diffs.a_diffs.len(), 1);

        // Migrate A's edit to B
        let delta = diffs.migrate_a_to_b(0).unwrap();
        let _doc_b = apply(&doc_b, &delta).unwrap();

        // A's diffs should be empty now
        assert!(diffs.a_diffs.is_empty());
    }
}
