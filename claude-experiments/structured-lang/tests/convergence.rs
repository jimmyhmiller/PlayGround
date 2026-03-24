//! Property tests that prove FULL convergence: after bidirectional merge,
//! two branches must be IDENTICAL — same columns in same order, same types,
//! same values on every row. Not just "same set of column names."

use proptest::prelude::*;
use structured_lang::database::Database;
use structured_lang::types::*;

/// Get a complete snapshot of a branch's table: ordered columns with types,
/// and all row data in order.
fn snapshot(
    db: &Database,
    branch: u64,
    table: &str,
) -> (Vec<(String, String)>, Vec<Vec<(String, serde_json::Value)>>) {
    let schema = db.get_table_view(branch, table).unwrap();
    let rows = db.list_rows(branch, table).unwrap();
    let row_data: Vec<Vec<(String, serde_json::Value)>> =
        rows.into_iter().map(|r| r.fields).collect();
    (schema.columns, row_data)
}

/// Assert two snapshots are identical — same columns in same order,
/// same types, same row values.
fn assert_identical(
    label: &str,
    a: &(Vec<(String, String)>, Vec<Vec<(String, serde_json::Value)>>),
    b: &(Vec<(String, String)>, Vec<Vec<(String, serde_json::Value)>>),
) {
    assert_eq!(a.0, b.0,
        "{}: columns differ.\n  left:  {:?}\n  right: {:?}", label, a.0, b.0);
    assert_eq!(a.1.len(), b.1.len(),
        "{}: row count differs ({} vs {})", label, a.1.len(), b.1.len());
    for (i, (ra, rb)) in a.1.iter().zip(b.1.iter()).enumerate() {
        assert_eq!(ra, rb,
            "{}: row {} differs.\n  left:  {:?}\n  right: {:?}", label, i, ra, rb);
    }
}

// ============================================================================
// Basic: fork, edit one side, merge, must be identical
// ============================================================================

#[test]
fn fork_edit_merge_identical() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("x", AtomicType::Str)]).unwrap();
    db.insert_row(main, "t", vec![("x", Value::Str("hello".into()))]).unwrap();

    let dev = db.fork_branch(main, "dev").unwrap();
    db.add_column(dev, "t", "y", AtomicType::Num).unwrap();
    db.set_field(dev, "t", 0, "y", Value::Num(42.0)).unwrap();

    db.merge_all(dev, main).unwrap();

    let s_main = snapshot(&db, main, "t");
    let s_dev = snapshot(&db, dev, "t");
    assert_identical("after merge dev→main", &s_main, &s_dev);
}

// ============================================================================
// Bidirectional: both sides edit, merge both ways, must be identical
// ============================================================================

#[test]
fn bidirectional_merge_identical() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("name", AtomicType::Str)]).unwrap();
    db.insert_row(main, "t", vec![("name", Value::Str("Alice".into()))]).unwrap();

    let a = db.fork_branch(main, "a").unwrap();
    let b = db.fork_branch(main, "b").unwrap();
    db.diff_branches(a, b).unwrap();

    db.add_column(a, "t", "email", AtomicType::Str).unwrap();
    db.set_field(a, "t", 0, "email", Value::Str("alice@a.com".into())).unwrap();

    db.add_column(b, "t", "dept", AtomicType::Str).unwrap();
    db.set_field(b, "t", 0, "dept", Value::Str("Engineering".into())).unwrap();

    db.merge_all(a, b).unwrap();
    db.merge_all(b, a).unwrap();

    let sa = snapshot(&db, a, "t");
    let sb = snapshot(&db, b, "t");
    assert_identical("after full bidirectional merge", &sa, &sb);
}

// ============================================================================
// Three-way: main → a, main → b, edits on both, merge to main
// ============================================================================

#[test]
fn three_way_merge_to_parent_identical() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("base", AtomicType::Str)]).unwrap();
    db.insert_row(main, "t", vec![("base", Value::Str("x".into()))]).unwrap();

    let a = db.fork_branch(main, "a").unwrap();
    let b = db.fork_branch(main, "b").unwrap();

    db.add_column(a, "t", "from_a", AtomicType::Num).unwrap();
    db.add_column(b, "t", "from_b", AtomicType::Bool).unwrap();

    // Merge both into main
    db.merge_all(a, main).unwrap();
    db.merge_all(b, main).unwrap();

    // Now merge main back to both children
    db.merge_all(main, a).unwrap();
    db.merge_all(main, b).unwrap();

    let sm = snapshot(&db, main, "t");
    let sa = snapshot(&db, a, "t");
    let sb = snapshot(&db, b, "t");
    assert_identical("main vs a", &sm, &sa);
    assert_identical("main vs b", &sm, &sb);
}

// ============================================================================
// Property: random schema edits, full merge, IDENTICAL snapshots
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn random_schema_edits_converge_identical(
        n_a in 1..4usize,
        n_b in 1..4usize,
        seed in 0..10000u64,
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("base", AtomicType::Str)]).unwrap();
        db.insert_row(main, "t", vec![("base", Value::Str("x".into()))]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        db.diff_branches(a, b).unwrap();

        for i in 0..n_a {
            let name = format!("a_{}_{}", seed, i);
            let _ = db.add_column(a, "t", &name, AtomicType::Str);
        }
        for i in 0..n_b {
            let name = format!("b_{}_{}", seed, i);
            let _ = db.add_column(b, "t", &name, AtomicType::Str);
        }

        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);

        let sa = snapshot(&db, a, "t");
        let sb = snapshot(&db, b, "t");

        // Columns must be identical — same names, same order, same types
        prop_assert_eq!(&sa.0, &sb.0,
            "columns must be identical after full merge.\n  a: {:?}\n  b: {:?}", sa.0, sb.0);
        // Row data must be identical
        prop_assert_eq!(&sa.1, &sb.1,
            "row data must be identical after full merge");
    }

    #[test]
    fn random_schema_and_value_edits_converge_identical(
        n_a in 1..3usize,
        n_b in 1..3usize,
        seed in 0..10000u64,
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("name", AtomicType::Str), ("val", AtomicType::Num)]).unwrap();
        let row = db.insert_row(main, "t", vec![
            ("name", Value::Str("test".into())),
            ("val", Value::Num(0.0)),
        ]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        db.diff_branches(a, b).unwrap();

        // A: add columns and set values
        for i in 0..n_a {
            let name = format!("a_{}_{}", seed, i);
            if db.add_column(a, "t", &name, AtomicType::Str).is_ok() {
                let _ = db.set_field(a, "t", row, &name, Value::Str(format!("val_a_{}", i)));
            }
        }

        // B: add columns and set values
        for i in 0..n_b {
            let name = format!("b_{}_{}", seed, i);
            if db.add_column(b, "t", &name, AtomicType::Str).is_ok() {
                let _ = db.set_field(b, "t", row, &name, Value::Str(format!("val_b_{}", i)));
            }
        }

        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);

        let sa = snapshot(&db, a, "t");
        let sb = snapshot(&db, b, "t");

        prop_assert_eq!(&sa.0, &sb.0,
            "columns must be identical.\n  a: {:?}\n  b: {:?}", sa.0, sb.0);
        prop_assert_eq!(&sa.1, &sb.1,
            "rows must be identical.\n  a: {:?}\n  b: {:?}", sa.1, sb.1);
    }

    #[test]
    fn random_mixed_ops_converge_identical(
        ops_a in prop::collection::vec(0..5u8, 1..4),
        ops_b in prop::collection::vec(0..5u8, 1..4),
        seed in 0..10000u64,
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![
            ("name", AtomicType::Str),
            ("age", AtomicType::Num),
        ]).unwrap();
        let row = db.insert_row(main, "t", vec![
            ("name", Value::Str("Alice".into())),
            ("age", Value::Num(30.0)),
        ]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();
        db.diff_branches(a, b).unwrap();

        // Apply random ops to each branch
        for (i, op) in ops_a.iter().enumerate() {
            let col = format!("ac_{}_{}", seed, i);
            match op % 5 {
                0 => { let _ = db.add_column(a, "t", &col, AtomicType::Str); }
                1 => { let _ = db.add_column(a, "t", &col, AtomicType::Num); }
                2 => { let _ = db.add_column(a, "t", &col, AtomicType::Bool); }
                3 => { let _ = db.set_field(a, "t", row, "name", Value::Str(format!("a_{}", i))); }
                4 => { let _ = db.rename_column(a, "t", "age", &format!("age_a_{}", seed)); }
                _ => {}
            }
        }
        for (i, op) in ops_b.iter().enumerate() {
            let col = format!("bc_{}_{}", seed, i);
            match op % 5 {
                0 => { let _ = db.add_column(b, "t", &col, AtomicType::Str); }
                1 => { let _ = db.add_column(b, "t", &col, AtomicType::Num); }
                2 => { let _ = db.add_column(b, "t", &col, AtomicType::Bool); }
                3 => { let _ = db.set_field(b, "t", row, "name", Value::Str(format!("b_{}", i))); }
                4 => { let _ = db.rename_column(b, "t", "age", &format!("age_b_{}", seed)); }
                _ => {}
            }
        }

        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);

        let sa = snapshot(&db, a, "t");
        let sb = snapshot(&db, b, "t");

        prop_assert_eq!(&sa.0, &sb.0,
            "columns must be identical after mixed ops merge.\n  a: {:?}\n  b: {:?}", sa.0, sb.0);
        prop_assert_eq!(&sa.1, &sb.1,
            "rows must be identical after mixed ops merge.\n  a: {:?}\n  b: {:?}", sa.1, sb.1);
    }
}
