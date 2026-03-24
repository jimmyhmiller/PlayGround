//! Chaos test: randomly generate entire histories with arbitrary numbers of
//! branches, forks, schema edits, value edits, and merges. After all merges
//! complete, verify that all branches are IDENTICAL.
//!
//! This is the ultimate property test — it proves convergence for any
//! possible sequence of operations across any topology.

use proptest::prelude::*;
use structured_lang::database::Database;
use structured_lang::types::*;

fn snapshot(
    db: &Database,
    branch: u64,
    table: &str,
) -> Option<(Vec<(String, String)>, Vec<Vec<(String, serde_json::Value)>>)> {
    let schema = db.get_table_view(branch, table).ok()?;
    let rows = db.list_rows(branch, table).ok()?;
    let row_data: Vec<Vec<(String, serde_json::Value)>> =
        rows.into_iter().map(|r| r.fields).collect();
    Some((schema.columns, row_data))
}

/// An action that can be performed on the database.
#[derive(Debug, Clone)]
enum Action {
    Fork { parent_idx: usize, name: String },
    AddColumn { branch_idx: usize, name: String, ty: AtomicType },
    RemoveColumn { branch_idx: usize, name: String },
    RenameColumn { branch_idx: usize, old: String, new: String },
    ConvertColumn { branch_idx: usize, name: String, ty: AtomicType },
    SetField { branch_idx: usize, row_idx: usize, field: String, value: String },
}

fn arb_type() -> impl Strategy<Value = AtomicType> {
    prop_oneof![Just(AtomicType::Str), Just(AtomicType::Num), Just(AtomicType::Bool)]
}

fn arb_field_name() -> impl Strategy<Value = String> {
    "[a-z]{2,4}"
}

fn arb_action() -> impl Strategy<Value = Action> {
    prop_oneof![
        // Fork from a random existing branch
        3 => (0..20usize, "[a-z]{2,5}")
            .prop_map(|(parent_idx, name)| Action::Fork { parent_idx, name }),
        // Add column
        5 => (0..20usize, arb_field_name(), arb_type())
            .prop_map(|(branch_idx, name, ty)| Action::AddColumn { branch_idx, name, ty }),
        // Remove column
        2 => (0..20usize, arb_field_name())
            .prop_map(|(branch_idx, name)| Action::RemoveColumn { branch_idx, name }),
        // Rename column
        2 => (0..20usize, arb_field_name(), arb_field_name())
            .prop_map(|(branch_idx, old, new)| Action::RenameColumn { branch_idx, old, new }),
        // Convert column type
        2 => (0..20usize, arb_field_name(), arb_type())
            .prop_map(|(branch_idx, name, ty)| Action::ConvertColumn { branch_idx, name, ty }),
        // Set field value
        4 => (0..20usize, 0..5usize, arb_field_name(), "[a-z]{1,8}")
            .prop_map(|(branch_idx, row_idx, field, value)| Action::SetField { branch_idx, row_idx, field, value }),
    ]
}

/// Execute an action, ignoring errors (the action might reference nonexistent
/// branches, columns, or rows — that's fine, we just skip it).
fn execute(db: &mut Database, branches: &mut Vec<u64>, rows: &[u64], action: &Action) {
    match action {
        Action::Fork { parent_idx, name } => {
            if let Some(&parent) = branches.get(*parent_idx % branches.len().max(1)) {
                if let Ok(id) = db.fork_branch(parent, name) {
                    branches.push(id);
                }
            }
        }
        Action::AddColumn { branch_idx, name, ty } => {
            if let Some(&branch) = branches.get(*branch_idx % branches.len().max(1)) {
                let _ = db.add_column(branch, "t", name, *ty);
            }
        }
        Action::RemoveColumn { branch_idx, name } => {
            if let Some(&branch) = branches.get(*branch_idx % branches.len().max(1)) {
                let _ = db.remove_column(branch, "t", name);
            }
        }
        Action::RenameColumn { branch_idx, old, new } => {
            if let Some(&branch) = branches.get(*branch_idx % branches.len().max(1)) {
                let _ = db.rename_column(branch, "t", old, new);
            }
        }
        Action::ConvertColumn { branch_idx, name, ty } => {
            if let Some(&branch) = branches.get(*branch_idx % branches.len().max(1)) {
                let _ = db.convert_column(branch, "t", name, *ty);
            }
        }
        Action::SetField { branch_idx, row_idx, field, value } => {
            if let Some(&branch) = branches.get(*branch_idx % branches.len().max(1)) {
                if let Some(&row) = rows.get(*row_idx % rows.len().max(1)) {
                    let _ = db.set_field(branch, "t", row, field, Value::Str(value.clone()));
                }
            }
        }
    }
}

/// Merge all branches together until convergence.
/// Does multiple rounds of pairwise merges between all tracked pairs.
fn merge_all_pairs(db: &mut Database, branches: &[u64]) {
    for _ in 0..3 {
        for i in 0..branches.len() {
            for j in 0..branches.len() {
                if i != j {
                    let _ = db.merge_all(branches[i], branches[j]);
                }
            }
        }
    }
}

// ============================================================================
// THE CHAOS TEST
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80000))]

    #[test]
    fn chaos_random_history_converges(
        actions in prop::collection::vec(arb_action(), 3..15),
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![
            ("id", AtomicType::Num),
            ("name", AtomicType::Str),
        ]).unwrap();
        let r0 = db.insert_row(main, "t", vec![
            ("id", Value::Num(1.0)),
            ("name", Value::Str("Alice".into())),
        ]).unwrap();
        let r1 = db.insert_row(main, "t", vec![
            ("id", Value::Num(2.0)),
            ("name", Value::Str("Bob".into())),
        ]).unwrap();

        let mut branches = vec![main];
        let rows = vec![r0, r1];

        // Execute random actions
        for action in &actions {
            execute(&mut db, &mut branches, &rows, action);
        }

        // Skip if only one branch (nothing to merge)
        if branches.len() < 2 {
            return Ok(());
        }

        // Ensure all pairs are tracked
        for i in 0..branches.len() {
            for j in (i+1)..branches.len() {
                let _ = db.diff_branches(branches[i], branches[j]);
            }
        }

        // Merge everything
        merge_all_pairs(&mut db, &branches);

        // PROPERTY: all branches must be IDENTICAL
        let reference = snapshot(&db, branches[0], "t");
        for &branch in &branches[1..] {
            let current = snapshot(&db, branch, "t");
            match (&reference, &current) {
                (Some(ref_snap), Some(cur_snap)) => {
                    prop_assert_eq!(&ref_snap.0, &cur_snap.0,
                        "branches {} and {} have different columns after full merge.\n  {}: {:?}\n  {}: {:?}",
                        branches[0], branch, branches[0], ref_snap.0, branch, cur_snap.0);
                    prop_assert_eq!(&ref_snap.1, &cur_snap.1,
                        "branches {} and {} have different row data after full merge.\n  {}: {:?}\n  {}: {:?}",
                        branches[0], branch, branches[0], ref_snap.1, branch, cur_snap.1);
                }
                (None, None) => {} // both missing table, ok
                _ => {
                    prop_assert!(false,
                        "branch {} and {} disagree on table existence", branches[0], branch);
                }
            }
        }
    }
}

// ============================================================================
// LONGER CHAOS: more actions, more branches
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20000))]

    #[test]
    fn chaos_long_history_converges(
        actions in prop::collection::vec(arb_action(), 10..30),
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![
            ("x", AtomicType::Str),
            ("y", AtomicType::Num),
            ("z", AtomicType::Bool),
        ]).unwrap();
        let r0 = db.insert_row(main, "t", vec![
            ("x", Value::Str("hello".into())),
            ("y", Value::Num(42.0)),
            ("z", Value::Bool(true)),
        ]).unwrap();

        let mut branches = vec![main];
        let rows = vec![r0];

        for action in &actions {
            execute(&mut db, &mut branches, &rows, action);
        }

        if branches.len() < 2 {
            return Ok(());
        }

        for i in 0..branches.len() {
            for j in (i+1)..branches.len() {
                let _ = db.diff_branches(branches[i], branches[j]);
            }
        }

        merge_all_pairs(&mut db, &branches);

        let reference = snapshot(&db, branches[0], "t");
        for &branch in &branches[1..] {
            let current = snapshot(&db, branch, "t");
            if let (Some(ref_snap), Some(cur_snap)) = (&reference, &current) {
                prop_assert_eq!(&ref_snap.0, &cur_snap.0,
                    "columns differ between branches {} and {} (of {} total branches)",
                    branches[0], branch, branches.len());
                prop_assert_eq!(&ref_snap.1, &cur_snap.1,
                    "rows differ between branches {} and {} (of {} total branches)",
                    branches[0], branch, branches.len());
            }
        }
    }
}
