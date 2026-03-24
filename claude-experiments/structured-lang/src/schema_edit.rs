//! Name-based OT for schema operations.
//!
//! The paper's positional OT operates on ordered sequences (identified by index).
//! Schemas are unordered maps (identified by name). This module extends the
//! formalism to operate on names instead of positions.
//!
//! The algebraic structure is the same: project/retract satisfy commutativity
//! (post ∘ diff = adjust ∘ pre). But the rules are simpler because name
//! equality replaces index equality — no shifting, no Move.

use serde::{Deserialize, Serialize};
use crate::types::{AtomicType, Value};

/// A schema edit operates on a named field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaEdit {
    /// No-op.
    Id,
    /// Add a new field with a name and type.
    AddField { name: String, ty: AtomicType },
    /// Remove a field by name (tombstone).
    RemoveField { name: String },
    /// Convert a field's type.
    ConvertField { name: String, ty: AtomicType },
    /// Rename a field.
    RenameField { old_name: String, new_name: String },
    /// Set a field's value on a specific row.
    SetField { row: u64, field: String, value: Value },
}

impl SchemaEdit {
    pub fn is_id(&self) -> bool {
        matches!(self, SchemaEdit::Id)
    }

    /// The field name this edit targets (for conflict detection).
    pub fn target_name(&self) -> Option<&str> {
        match self {
            SchemaEdit::Id => None,
            SchemaEdit::AddField { name, .. } => Some(name),
            SchemaEdit::RemoveField { name } => Some(name),
            SchemaEdit::ConvertField { name, .. } => Some(name),
            SchemaEdit::RenameField { old_name, .. } => Some(old_name),
            SchemaEdit::SetField { field, .. } => Some(field),
        }
    }

    /// The row this edit targets (only Some for SetField).
    pub fn target_row(&self) -> Option<u64> {
        match self {
            SchemaEdit::SetField { row, .. } => Some(*row),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaTransformResult {
    pub edit: SchemaEdit,
    pub adjust: SchemaEdit,
}

/// Project: given `pre` (edit on left) and `diff` (edit on top), compute
/// `post` (right) and `adjust` (bottom) satisfying: post ∘ diff = adjust ∘ pre.
///
/// Rules:
/// - Equal edits cancel: (Id, Id)
/// - Same target name, same edit kind: conflict (pre wins, adjust → Id)
/// - Same target name, different kind: depends on interaction
/// - Different target names: independent (both pass through)
/// - Rename changes target names for subsequent edits
pub fn schema_project(pre: &SchemaEdit, diff: &SchemaEdit) -> SchemaTransformResult {
    // Equal edits cancel
    if pre == diff && !pre.is_id() {
        return SchemaTransformResult {
            edit: SchemaEdit::Id,
            adjust: SchemaEdit::Id,
        };
    }

    // Id is a fixpoint
    if pre.is_id() {
        return SchemaTransformResult {
            edit: SchemaEdit::Id,
            adjust: diff.clone(),
        };
    }
    if diff.is_id() {
        return SchemaTransformResult {
            edit: pre.clone(),
            adjust: SchemaEdit::Id,
        };
    }

    // Rename collision: RenameField's new_name clashes with another edit's target.
    // e.g., RenameField("id"→"email") vs AddField("email") — conflict, pre wins.
    if let SchemaEdit::RenameField { new_name, .. } = pre {
        if let Some(diff_name) = diff.target_name() {
            if new_name == diff_name && !matches!(diff, SchemaEdit::RenameField { .. }) {
                return SchemaTransformResult {
                    edit: pre.clone(),
                    adjust: SchemaEdit::Id,
                };
            }
        }
    }
    if let SchemaEdit::RenameField { new_name, .. } = diff {
        if let Some(pre_name) = pre.target_name() {
            if new_name == pre_name && !matches!(pre, SchemaEdit::RenameField { .. }) {
                return SchemaTransformResult {
                    edit: pre.clone(),
                    adjust: diff.clone(),
                };
            }
        }
    }

    // Rename interactions: if one side renames a field that the other targets,
    // the other edit follows the rename. Check this BEFORE same-name logic.
    if let SchemaEdit::RenameField { old_name, new_name } = pre {
        if diff.target_name() == Some(old_name.as_str()) && !matches!(diff, SchemaEdit::RenameField { .. }) {
            let adjusted_diff = rename_target(diff, new_name);
            return SchemaTransformResult {
                edit: pre.clone(),
                adjust: adjusted_diff,
            };
        }
    }
    if let SchemaEdit::RenameField { old_name, new_name } = diff {
        if pre.target_name() == Some(old_name.as_str()) && !matches!(pre, SchemaEdit::RenameField { .. }) {
            let adjusted_pre = rename_target(pre, new_name);
            return SchemaTransformResult {
                edit: adjusted_pre,
                adjust: diff.clone(),
            };
        }
    }

    let pre_name = pre.target_name();
    let diff_name = diff.target_name();

    // Different target names: independent
    if pre_name != diff_name {
        return SchemaTransformResult {
            edit: pre.clone(),
            adjust: diff.clone(),
        };
    }

    // SetField: same field name but different rows are independent
    if let (SchemaEdit::SetField { row: r1, .. }, SchemaEdit::SetField { row: r2, .. }) = (pre, diff) {
        if r1 != r2 {
            return SchemaTransformResult {
                edit: pre.clone(),
                adjust: diff.clone(),
            };
        }
    }

    // Same target name — interaction depends on edit kinds
    match (pre, diff) {
        // AddField vs AddField: same name
        (SchemaEdit::AddField { ty: t1, .. }, SchemaEdit::AddField { ty: t2, .. }) => {
            if t1 == t2 {
                // Same name, same type: equal intent, cancel
                SchemaTransformResult {
                    edit: SchemaEdit::Id,
                    adjust: SchemaEdit::Id,
                }
            } else {
                // Same name, different type: conflict, pre wins.
                // Produce AddField (not ConvertField) so the target's existing
                // column gets its type and default value overwritten.
                SchemaTransformResult {
                    edit: pre.clone(),
                    adjust: SchemaEdit::Id,
                }
            }
        }

        // ConvertField vs ConvertField: same name, conflict
        (SchemaEdit::ConvertField { .. }, SchemaEdit::ConvertField { .. }) => {
            SchemaTransformResult {
                edit: pre.clone(),
                adjust: SchemaEdit::Id,
            }
        }

        // SetField vs SetField: same field, conflict
        (SchemaEdit::SetField { .. }, SchemaEdit::SetField { .. }) => {
            SchemaTransformResult {
                edit: pre.clone(),
                adjust: SchemaEdit::Id,
            }
        }

        // RenameField vs RenameField: same source, conflict — pre wins.
        // The delta must rename from diff's new_name to pre's new_name,
        // because diff has already been applied to the target state.
        (SchemaEdit::RenameField { new_name: pre_new, .. },
         SchemaEdit::RenameField { new_name: diff_new, .. }) => {
            SchemaTransformResult {
                edit: SchemaEdit::RenameField {
                    old_name: diff_new.clone(),
                    new_name: pre_new.clone(),
                },
                adjust: SchemaEdit::Id,
            }
        }

        // RemoveField vs RemoveField: same name, cancel
        (SchemaEdit::RemoveField { .. }, SchemaEdit::RemoveField { .. }) => {
            SchemaTransformResult {
                edit: SchemaEdit::Id,
                adjust: SchemaEdit::Id,
            }
        }

        // AddField vs RemoveField (same name): conflict — pre wins
        (SchemaEdit::AddField { .. }, SchemaEdit::RemoveField { .. }) |
        (SchemaEdit::RemoveField { .. }, SchemaEdit::AddField { .. }) => {
            SchemaTransformResult {
                edit: pre.clone(),
                adjust: SchemaEdit::Id,
            }
        }

        // Different kinds at same name: independent (orthogonal concerns)
        // e.g., ConvertField("age") vs SetField("age") — type and value are independent
        // e.g., RenameField("age"→"years") vs ConvertField("age") — follows rename
        _ => {
            SchemaTransformResult {
                edit: pre.clone(),
                adjust: diff.clone(),
            }
        }
    }
}

/// Retract: given `post` and `diff`, find `pre` and `adjust`.
/// For name-based OT, retract is symmetric to project in most cases.
pub fn schema_retract(post: &SchemaEdit, diff: &SchemaEdit) -> Option<SchemaTransformResult> {
    // Id fixpoints
    if post.is_id() {
        return Some(SchemaTransformResult {
            edit: SchemaEdit::Id,
            adjust: diff.clone(),
        });
    }
    if diff.is_id() {
        return Some(SchemaTransformResult {
            edit: post.clone(),
            adjust: SchemaEdit::Id,
        });
    }

    // For name-based OT, retract follows the same structure as project.
    // The commutativity constraint is the same: post ∘ diff = adjust ∘ pre.
    // Since there's no index shifting, the rules are symmetric.
    let result = schema_project(post, diff);

    // Dependency: AddField can't be retracted through RemoveField of same name
    if let (SchemaEdit::AddField { .. } | SchemaEdit::SetField { .. } | SchemaEdit::ConvertField { .. },
            SchemaEdit::RemoveField { .. }) = (post, diff) {
        if post.target_name() == diff.target_name() {
            return None; // dependency
        }
    }

    Some(result)
}

/// Adjust a schema edit's target to follow a rename.
fn rename_target(edit: &SchemaEdit, new_name: &str) -> SchemaEdit {
    match edit {
        SchemaEdit::AddField { ty, .. } => SchemaEdit::AddField { name: new_name.to_string(), ty: *ty },
        SchemaEdit::RemoveField { .. } => SchemaEdit::RemoveField { name: new_name.to_string() },
        SchemaEdit::ConvertField { ty, .. } => SchemaEdit::ConvertField { name: new_name.to_string(), ty: *ty },
        SchemaEdit::SetField { row, value, .. } => SchemaEdit::SetField { row: *row, field: new_name.to_string(), value: value.clone() },
        SchemaEdit::RenameField { new_name: nn, .. } => SchemaEdit::RenameField { old_name: new_name.to_string(), new_name: nn.clone() },
        SchemaEdit::Id => SchemaEdit::Id,
    }
}

/// Check if two schema edits conflict (target the same field with the same kind).
pub fn schema_edits_conflict(a: &SchemaEdit, b: &SchemaEdit) -> bool {
    if a.is_id() || b.is_id() {
        return false;
    }
    let a_name = a.target_name();
    let b_name = b.target_name();
    if a_name.is_none() || b_name.is_none() || a_name != b_name {
        return false;
    }
    // SetField: same field name but different rows are NOT a conflict
    if let (SchemaEdit::SetField { row: r1, .. }, SchemaEdit::SetField { row: r2, .. }) = (a, b) {
        return r1 == r2;
    }
    // Same target name — conflict if same kind of edit
    matches!(
        (a, b),
        (SchemaEdit::SetField { .. }, SchemaEdit::SetField { .. })
        | (SchemaEdit::ConvertField { .. }, SchemaEdit::ConvertField { .. })
        | (SchemaEdit::RenameField { .. }, SchemaEdit::RenameField { .. })
        | (SchemaEdit::AddField { .. }, SchemaEdit::AddField { .. })
        | (SchemaEdit::RemoveField { .. }, SchemaEdit::RemoveField { .. })
    )
}

// ── SchemaDifferences: same structure as Differences but for SchemaEdit ──

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaDifferences {
    pub a_diffs: Vec<SchemaEdit>,
    pub b_diffs: Vec<SchemaEdit>,
}

/// Translate a schema edit through the diff sequences.
pub fn schema_translate(
    epsilon: &SchemaEdit,
    a_diffs: &[SchemaEdit],
    b_diffs: &[SchemaEdit],
) -> Option<SchemaTranslateResult> {
    let mut current = epsilon.clone();
    let mut new_a_diffs = a_diffs.to_vec();

    // Retract through a_diffs in reverse order
    for i in (0..new_a_diffs.len()).rev() {
        let result = schema_retract(&current, &new_a_diffs[i])?;
        current = result.edit;
        new_a_diffs[i] = result.adjust;
    }

    let mut new_b_diffs = b_diffs.to_vec();

    // Project through b_diffs in forward order
    for i in 0..new_b_diffs.len() {
        let result = schema_project(&current, &new_b_diffs[i]);
        current = result.edit;
        new_b_diffs[i] = result.adjust;
    }

    Some(SchemaTranslateResult {
        delta: current,
        a_diffs: new_a_diffs,
        b_diffs: new_b_diffs,
    })
}

#[derive(Debug, Clone)]
pub struct SchemaTranslateResult {
    pub delta: SchemaEdit,
    pub a_diffs: Vec<SchemaEdit>,
    pub b_diffs: Vec<SchemaEdit>,
}

/// A conflict between two schema edits targeting the same field.
#[derive(Debug, Clone)]
pub struct SchemaConflict {
    pub from_edit: SchemaEdit,
    pub to_edit: SchemaEdit,
    pub from_idx: usize,
    pub to_idx: usize,
}

impl SchemaDifferences {
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect all conflicts: pairs of edits targeting the same field name
    /// with conflicting kinds (both SetField, both ConvertField, etc.)
    pub fn all_conflicts(&self) -> Vec<SchemaConflict> {
        let mut conflicts = Vec::new();
        for (i, a) in self.a_diffs.iter().enumerate() {
            for (j, b) in self.b_diffs.iter().enumerate() {
                if schema_edits_conflict(a, b) {
                    conflicts.push(SchemaConflict {
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

    pub fn edit_a(&mut self, epsilon: &SchemaEdit) -> Option<SchemaEdit> {
        match schema_translate(epsilon, &self.a_diffs, &self.b_diffs) {
            Some(result) if result.delta.is_id() => {
                self.a_diffs = result.a_diffs;
                self.b_diffs = result.b_diffs;
                Some(result.delta)
            }
            Some(result) => {
                self.a_diffs.push(epsilon.clone());
                Some(result.delta)
            }
            None => {
                self.a_diffs.push(epsilon.clone());
                None
            }
        }
    }

    pub fn edit_b(&mut self, epsilon: &SchemaEdit) -> Option<SchemaEdit> {
        match schema_translate(epsilon, &self.b_diffs, &self.a_diffs) {
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

    pub fn migrate_first_a_to_b(&mut self) -> Option<SchemaEdit> {
        if self.a_diffs.is_empty() {
            return None;
        }
        let edit = self.a_diffs.remove(0);
        let mut delta = edit;
        let mut new_b_diffs = self.b_diffs.clone();
        for i in 0..new_b_diffs.len() {
            let result = schema_project(&delta, &new_b_diffs[i]);
            delta = result.edit;
            new_b_diffs[i] = result.adjust;
        }
        self.b_diffs = new_b_diffs;
        Some(delta)
    }

    pub fn migrate_first_b_to_a(&mut self) -> Option<SchemaEdit> {
        if self.b_diffs.is_empty() {
            return None;
        }
        let edit = self.b_diffs.remove(0);
        let mut delta = edit;
        let mut new_a_diffs = self.a_diffs.clone();
        for i in 0..new_a_diffs.len() {
            let result = schema_project(&delta, &new_a_diffs[i]);
            delta = result.edit;
            new_a_diffs[i] = result.adjust;
        }
        self.a_diffs = new_a_diffs;
        Some(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_add_cancels() {
        let a = SchemaEdit::AddField { name: "email".into(), ty: AtomicType::Str };
        let b = SchemaEdit::AddField { name: "email".into(), ty: AtomicType::Str };
        let r = schema_project(&a, &b);
        assert_eq!(r.edit, SchemaEdit::Id);
        assert_eq!(r.adjust, SchemaEdit::Id);
    }

    #[test]
    fn different_adds_independent() {
        let a = SchemaEdit::AddField { name: "email".into(), ty: AtomicType::Str };
        let b = SchemaEdit::AddField { name: "phone".into(), ty: AtomicType::Num };
        let r = schema_project(&a, &b);
        assert_eq!(r.edit, a);
        assert_eq!(r.adjust, b);
    }

    #[test]
    fn same_name_diff_type_conflict() {
        let a = SchemaEdit::AddField { name: "x".into(), ty: AtomicType::Str };
        let b = SchemaEdit::AddField { name: "x".into(), ty: AtomicType::Num };
        let r = schema_project(&a, &b);
        // Pre wins: x becomes Str (via AddField overwrite)
        assert_eq!(r.edit, SchemaEdit::AddField { name: "x".into(), ty: AtomicType::Str });
        assert_eq!(r.adjust, SchemaEdit::Id);
    }

    #[test]
    fn set_conflict_pre_wins() {
        let a = SchemaEdit::SetField { row: 0, field: "name".into(), value: Value::Str("Alice".into()) };
        let b = SchemaEdit::SetField { row: 0, field: "name".into(), value: Value::Str("Bob".into()) };
        let r = schema_project(&a, &b);
        assert_eq!(r.edit, a);
        assert_eq!(r.adjust, SchemaEdit::Id);
    }

    #[test]
    fn merge_adds_from_both_sides() {
        let mut diffs = SchemaDifferences::new();

        // A adds email
        diffs.edit_a(&SchemaEdit::AddField { name: "email".into(), ty: AtomicType::Str });
        // B adds phone
        diffs.edit_b(&SchemaEdit::AddField { name: "phone".into(), ty: AtomicType::Num });

        assert_eq!(diffs.a_diffs.len(), 1);
        assert_eq!(diffs.b_diffs.len(), 1);

        // Migrate A → B
        let delta = diffs.migrate_first_a_to_b().unwrap();
        assert_eq!(delta, SchemaEdit::AddField { name: "email".into(), ty: AtomicType::Str });

        // Migrate B → A
        let delta = diffs.migrate_first_b_to_a().unwrap();
        assert_eq!(delta, SchemaEdit::AddField { name: "phone".into(), ty: AtomicType::Num });
    }

    #[test]
    fn same_add_absorbed_in_diffs() {
        let mut diffs = SchemaDifferences::new();

        // Both sides add "email" with same type
        diffs.edit_a(&SchemaEdit::AddField { name: "email".into(), ty: AtomicType::Str });
        diffs.edit_b(&SchemaEdit::AddField { name: "email".into(), ty: AtomicType::Str });

        // The second edit should be absorbed (same intent)
        // After both edits, there should be no remaining diffs
        // (both sides agree they added "email")
        assert!(diffs.a_diffs.is_empty() || diffs.b_diffs.is_empty(),
            "same-name add should be absorbed: a={:?}, b={:?}", diffs.a_diffs, diffs.b_diffs);
    }

    #[test]
    fn rename_followed_by_set() {
        let pre = SchemaEdit::RenameField { old_name: "name".into(), new_name: "full_name".into() };
        let diff = SchemaEdit::SetField { row: 0, field: "name".into(), value: Value::Str("Alice".into()) };
        let r = schema_project(&pre, &diff);
        // Set should follow the rename
        assert_eq!(r.adjust, SchemaEdit::SetField { row: 0, field: "full_name".into(), value: Value::Str("Alice".into()) });
    }
}
