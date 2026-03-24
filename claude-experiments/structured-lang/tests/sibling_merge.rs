//! Property tests for sibling branch merges.
//! Two branches forked from the same parent should be mergeable.

use proptest::prelude::*;
use structured_lang::database::Database;
use structured_lang::types::*;

/// The exact failing scenario: fork two siblings, make edits on each,
/// merge between them. Edits should propagate.
#[test]
fn sibling_branches_can_merge_schema() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("x", AtomicType::Str)]).unwrap();
    db.insert_row(main, "t", vec![("x", Value::Str("a".into()))]).unwrap();

    let a = db.fork_branch(main, "a").unwrap();
    let b = db.fork_branch(main, "b").unwrap();

    // A adds column
    db.add_column(a, "t", "col_a", AtomicType::Num).unwrap();

    // B adds column
    db.add_column(b, "t", "col_b", AtomicType::Str).unwrap();

    // Merge A → B
    let applied = db.merge_all(a, b).unwrap();
    assert!(!applied.is_empty(), "sibling merge should apply edits");

    let sb = db.get_table_view(b, "t").unwrap();
    let cols: Vec<&str> = sb.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"col_a"), "B should have col_a after merge: {:?}", cols);
    assert!(cols.contains(&"col_b"), "B should still have col_b: {:?}", cols);
    // Should NOT have duplicates
    assert_eq!(cols.iter().filter(|c| **c == "col_a").count(), 1,
        "col_a should appear exactly once: {:?}", cols);
}

#[test]
fn sibling_branches_can_merge_values() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("name", AtomicType::Str)]).unwrap();
    let row = db.insert_row(main, "t", vec![("name", Value::Str("init".into()))]).unwrap();

    let a = db.fork_branch(main, "a").unwrap();
    let b = db.fork_branch(main, "b").unwrap();

    // A sets value
    db.set_field(a, "t", row, "name", Value::Str("from_a".into())).unwrap();

    // Merge A → B
    let _ = db.merge_all(a, b);

    let view = db.get_row(b, "t", row).unwrap();
    let fields: std::collections::HashMap<String, serde_json::Value> =
        view.fields.into_iter().collect();
    assert_eq!(fields.get("name"), Some(&serde_json::json!("from_a")),
        "B should have A's value after sibling merge");
}

#[test]
fn sibling_rename_merges_not_duplicates() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("name", AtomicType::Str), ("age", AtomicType::Num)]).unwrap();
    db.insert_row(main, "t", vec![
        ("name", Value::Str("Alice".into())),
        ("age", Value::Num(30.0)),
    ]).unwrap();

    let a = db.fork_branch(main, "a").unwrap();
    let b = db.fork_branch(main, "b").unwrap();

    // A renames name → full_name
    db.rename_column(a, "t", "name", "full_name").unwrap();

    // Merge A → B
    let _ = db.merge_all(a, b);

    let sb = db.get_table_view(b, "t").unwrap();
    let cols: Vec<&str> = sb.columns.iter().map(|(n, _)| n.as_str()).collect();
    // B should have full_name, NOT both name and full_name
    assert!(cols.contains(&"full_name"), "B should have full_name: {:?}", cols);
    assert!(!cols.contains(&"name"), "B should NOT have old 'name' column: {:?}", cols);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn sibling_schema_edits_converge(
        n_a in 1..3usize,
        n_b in 1..3usize,
        seed in 0..1000u64,
    ) {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "t", vec![("base", AtomicType::Str)]).unwrap();
        db.insert_row(main, "t", vec![("base", Value::Str("x".into()))]).unwrap();

        let a = db.fork_branch(main, "a").unwrap();
        let b = db.fork_branch(main, "b").unwrap();

        for i in 0..n_a {
            let name = format!("a_{}{}", seed, i);
            let _ = db.add_column(a, "t", &name, AtomicType::Str);
        }
        for i in 0..n_b {
            let name = format!("b_{}{}", seed, i);
            let _ = db.add_column(b, "t", &name, AtomicType::Str);
        }

        let _ = db.merge_all(a, b);
        let _ = db.merge_all(b, a);

        let sa = db.get_table_view(a, "t").unwrap();
        let sb = db.get_table_view(b, "t").unwrap();
        let mut ca: Vec<String> = sa.columns.iter().map(|(n, _)| n.clone()).collect();
        let mut cb: Vec<String> = sb.columns.iter().map(|(n, _)| n.clone()).collect();
        ca.sort();
        cb.sort();
        prop_assert_eq!(&ca, &cb, "siblings should converge after full merge");
    }
}
