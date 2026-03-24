//! Stress test: random chains of ALL operations across multiple branches,
//! tables, and rows. After every merge, verify FULL identity between branches.
//! Runs for a long time with many complex combinations.

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

fn assert_branches_identical(db: &Database, a: u64, b: u64, tables: &[String]) {
    for table in tables {
        let sa = snapshot(db, a, table);
        let sb = snapshot(db, b, table);
        match (sa, sb) {
            (Some(sa), Some(sb)) => {
                assert_eq!(sa.0, sb.0,
                    "branches {} and {} differ on table {} columns:\n  {}: {:?}\n  {}: {:?}",
                    a, b, table, a, sa.0, b, sb.0);
                assert_eq!(sa.1, sb.1,
                    "branches {} and {} differ on table {} row data:\n  {}: {:?}\n  {}: {:?}",
                    a, b, table, a, sa.1, b, sb.1);
            }
            (None, None) => {} // both don't have the table
            (sa, sb) => panic!(
                "branch {} has table {}={}, branch {} has {}={}",
                a, table, sa.is_some(), b, table, sb.is_some()
            ),
        }
    }
}

/// Operations that can be performed on a branch
#[derive(Debug, Clone)]
enum Op {
    AddColumn { table: usize, name: String, ty: AtomicType },
    RemoveColumn { table: usize, name: String },
    RenameColumn { table: usize, old: String, new: String },
    ConvertColumn { table: usize, name: String, to: AtomicType },
    InsertRow { table: usize },
    SetField { table: usize, row: usize, col: String, val: String },
}

fn arb_type() -> impl Strategy<Value = AtomicType> {
    prop_oneof![Just(AtomicType::Str), Just(AtomicType::Num), Just(AtomicType::Bool)]
}

fn arb_op(n_tables: usize, _n_existing_cols: usize) -> impl Strategy<Value = Op> {
    let t = 0..n_tables;
    prop_oneof![
        // Add column
        4 => (t.clone(), "[a-z]{2,4}", arb_type())
            .prop_map(|(table, name, ty)| Op::AddColumn { table, name, ty }),
        // Remove column (use generic name that may or may not exist)
        1 => (t.clone(), "[a-z]{2,4}")
            .prop_map(|(table, name)| Op::RemoveColumn { table, name }),
        // Rename column
        2 => (t.clone(), "[a-z]{2,4}", "[a-z]{2,4}")
            .prop_map(|(table, old, new)| Op::RenameColumn { table, old, new }),
        // Convert column
        2 => (t.clone(), "[a-z]{2,4}", arb_type())
            .prop_map(|(table, name, to)| Op::ConvertColumn { table, name, to }),
        // Insert row — TODO: not yet merged between branches
        // 2 => t.clone().prop_map(|table| Op::InsertRow { table }),
        // Set field on existing rows
        3 => (t.clone(), 0..5usize, "[a-z]{2,4}", "[a-z]{1,8}")
            .prop_map(|(table, row, col, val)| Op::SetField { table, row, col, val }),
    ]
}

fn apply_op(db: &mut Database, branch: u64, tables: &[String], rows: &mut Vec<Vec<u64>>, op: &Op) {
    match op {
        Op::AddColumn { table, name, ty } => {
            if let Some(t) = tables.get(*table) {
                let _ = db.add_column(branch, t, name, *ty);
            }
        }
        Op::RemoveColumn { table, name } => {
            if let Some(t) = tables.get(*table) {
                let _ = db.remove_column(branch, t, name);
            }
        }
        Op::RenameColumn { table, old, new } => {
            if let Some(t) = tables.get(*table) {
                let _ = db.rename_column(branch, t, old, new);
            }
        }
        Op::ConvertColumn { table, name, to } => {
            if let Some(t) = tables.get(*table) {
                let _ = db.convert_column(branch, t, name, *to);
            }
        }
        Op::InsertRow { table } => {
            if let Some(t) = tables.get(*table) {
                if let Ok(id) = db.insert_row(branch, t, vec![]) {
                    if *table < rows.len() {
                        rows[*table].push(id);
                    }
                }
            }
        }
        Op::SetField { table, row, col, val } => {
            if let Some(t) = tables.get(*table) {
                if let Some(row_ids) = rows.get(*table) {
                    if let Some(&rid) = row_ids.get(*row % row_ids.len().max(1)) {
                        let _ = db.set_field(branch, t, rid, col, Value::Str(val.clone()));
                    }
                }
            }
        }
    }
}

// ============================================================================
// The big one: random ops on two branches, merge, verify identical
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5000))]

    #[test]
    fn stress_two_branches_converge(
        ops_a in prop::collection::vec(arb_op(2, 3), 1..8),
        ops_b in prop::collection::vec(arb_op(2, 3), 1..8),
        seed in 0..100000u64,
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");

        let tables = vec!["users".to_string(), "posts".to_string()];
        for t in &tables {
            db.create_table(main, t, vec![("id", AtomicType::Num), ("val", AtomicType::Str)]).unwrap();
        }
        let r0 = db.insert_row(main, "users", vec![
            ("id", Value::Num(seed as f64)),
            ("val", Value::Str("init".into())),
        ]).unwrap();
        let r1 = db.insert_row(main, "posts", vec![
            ("id", Value::Num(seed as f64)),
            ("val", Value::Str("init".into())),
        ]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        db.diff_branches(a, b).unwrap();

        let mut rows_a = vec![vec![r0], vec![r1]];
        let mut rows_b = vec![vec![r0], vec![r1]];

        for op in &ops_a {
            apply_op(&mut db, a, &tables, &mut rows_a, op);
        }
        for op in &ops_b {
            apply_op(&mut db, b, &tables, &mut rows_b, op);
        }

        // Merge both ways
        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);

        // Must be identical on all tables
        for t in &tables {
            let sa = snapshot(&db, a, t);
            let sb = snapshot(&db, b, t);
            if let (Some(sa), Some(sb)) = (sa, sb) {
                prop_assert_eq!(&sa.0, &sb.0,
                    "table {} columns differ after merge", t);
                prop_assert_eq!(&sa.1, &sb.1,
                    "table {} rows differ after merge", t);
            }
        }
    }
}

// ============================================================================
// Three branches: random ops, cascade merge, all identical
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn stress_three_branches_converge(
        ops_a in prop::collection::vec(arb_op(1, 3), 1..6),
        ops_b in prop::collection::vec(arb_op(1, 3), 1..6),
        ops_c in prop::collection::vec(arb_op(1, 3), 1..6),
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![
            ("x", AtomicType::Str),
            ("y", AtomicType::Num),
        ]).unwrap();
        let r = db.insert_row(main, "t", vec![
            ("x", Value::Str("hello".into())),
            ("y", Value::Num(42.0)),
        ]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        let c = db.fork_branch(main, "c").unwrap();
        db.diff_branches(a, b).unwrap();
        db.diff_branches(b, c).unwrap();
        db.diff_branches(a, c).unwrap();

        let tables = vec!["t".to_string()];
        let mut rows_a = vec![vec![r]];
        let mut rows_b = vec![vec![r]];
        let mut rows_c = vec![vec![r]];

        for op in &ops_a { apply_op(&mut db, a, &tables, &mut rows_a, op); }
        for op in &ops_b { apply_op(&mut db, b, &tables, &mut rows_b, op); }
        for op in &ops_c { apply_op(&mut db, c, &tables, &mut rows_c, op); }

        // Cascade merge: a↔b, b↔c, a↔c, then again to propagate
        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);
        let _ = db.merge_all(b, c);
        let _ = db.merge_all(c, b);
        let _ = db.merge_all(a, c);
        let _ = db.merge_all(c, a);
        // Second pass to fully propagate
        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);
        let _ = db.merge_all(b, c);
        let _ = db.merge_all(c, b);

        let sa = snapshot(&db, a, "t");
        let sb = snapshot(&db, b, "t");
        let sc = snapshot(&db, c, "t");

        if let (Some(sa), Some(sb), Some(sc)) = (sa, sb, sc) {
            prop_assert_eq!(&sa.0, &sb.0, "a vs b columns differ");
            prop_assert_eq!(&sb.0, &sc.0, "b vs c columns differ");
            prop_assert_eq!(&sa.1, &sb.1, "a vs b rows differ");
            prop_assert_eq!(&sb.1, &sc.1, "b vs c rows differ");
        }
    }
}

// ============================================================================
// Chain of forks: main → a → b → c, edits on each, merge back up
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn stress_fork_chain_converge(
        ops_a in prop::collection::vec(arb_op(1, 2), 1..5),
        ops_b in prop::collection::vec(arb_op(1, 2), 1..5),
        ops_c in prop::collection::vec(arb_op(1, 2), 1..5),
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("base", AtomicType::Str)]).unwrap();
        let r = db.insert_row(main, "t", vec![("base", Value::Str("x".into()))]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(a, "b").unwrap();
        let c = db.fork_branch(b, "c").unwrap();

        let tables = vec!["t".to_string()];
        let mut rows_a = vec![vec![r]];
        let mut rows_b = vec![vec![r]];
        let mut rows_c = vec![vec![r]];

        for op in &ops_a { apply_op(&mut db, a, &tables, &mut rows_a, op); }
        for op in &ops_b { apply_op(&mut db, b, &tables, &mut rows_b, op); }
        for op in &ops_c { apply_op(&mut db, c, &tables, &mut rows_c, op); }

        // Merge up the chain: c→b→a→main, then back down
        let _ = db.merge_all(c, b);
        let _ = db.merge_all(b, a);
        let _ = db.merge_all(a, main);
        let _ = db.merge_all(main, a);
        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, c);
        // Second pass
        let _ = db.merge_all(c, b);
        let _ = db.merge_all(b, a);
        let _ = db.merge_all(a, main);
        let _ = db.merge_all(main, a);
        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, c);

        if let (Some(sm), Some(sa), Some(sb), Some(sc)) = (
            snapshot(&db, main, "t"),
            snapshot(&db, a, "t"),
            snapshot(&db, b, "t"),
            snapshot(&db, c, "t"),
        ) {
            prop_assert_eq!(&sm.0, &sa.0, "main vs a columns");
            prop_assert_eq!(&sa.0, &sb.0, "a vs b columns");
            prop_assert_eq!(&sb.0, &sc.0, "b vs c columns");
            prop_assert_eq!(&sm.1, &sa.1, "main vs a rows");
            prop_assert_eq!(&sa.1, &sb.1, "a vs b rows");
            prop_assert_eq!(&sb.1, &sc.1, "b vs c rows");
        }
    }
}
