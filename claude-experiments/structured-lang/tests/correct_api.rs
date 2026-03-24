//! Tests that the API is correct by construction:
//! - No manual diff_branches needed (auto-tracked on fork)
//! - New rows after fork get tracked
//! - Schema + value edits compose correctly through merge

use structured_lang::database::Database;
use structured_lang::types::*;

#[test]
fn fork_auto_tracks_no_manual_diff_needed() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("x", AtomicType::Num)]).unwrap();
    db.insert_row(main, "t", vec![("x", Value::Num(1.0))]).unwrap();

    // Fork — tracking starts automatically
    let dev = db.fork_branch(main, "dev").unwrap();

    // Make changes WITHOUT calling diff_branches
    db.add_column(dev, "t", "y", AtomicType::Str).unwrap();

    // Merge should work — edits were tracked from the fork point
    let applied = db.merge_all(dev, main).unwrap();
    assert!(!applied.is_empty(), "merge should have applied edits without manual diff_branches");

    let schema = db.get_table_view(main, "t").unwrap();
    let cols: Vec<&str> = schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"y"), "main should have column 'y' after merge: {:?}", cols);
}

#[test]
fn new_row_after_fork_gets_value_tracking() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("name", AtomicType::Str)]).unwrap();

    let dev = db.fork_branch(main, "dev").unwrap();

    // Insert a NEW row on dev (after fork)
    let row = db.insert_row(dev, "t", vec![("name", Value::Str("Alice".into()))]).unwrap();

    // Set a value on the new row
    db.set_field(dev, "t", row, "name", Value::Str("Alice Smith".into())).unwrap();

    // The Set edit should be tracked even though the row didn't exist at fork time
    let diffs = db.get_diffs(dev, main);
    let total: usize = diffs.iter().map(|(_, a, _)| a).sum();
    assert!(total > 0, "edits on new row should be tracked: {:?}", diffs);
}

#[test]
fn add_column_set_value_then_merge() {
    // This is the exact bug the user hit in the GUI:
    // Add a column on main, set a value on a row, fork to dev,
    // merge main→dev. Dev gets the column but NOT the value.
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "users", vec![("name", AtomicType::Str)]).unwrap();
    let alice = db.insert_row(main, "users", vec![
        ("name", Value::Str("Alice".into())),
    ]).unwrap();

    let dev = db.fork_branch(main, "dev").unwrap();

    // Main: add column + set value
    db.add_column(main, "users", "thing", AtomicType::Str).unwrap();
    db.set_field(main, "users", alice, "thing", Value::Str("Hello".into())).unwrap();

    // Verify main has it
    let row = db.get_row(main, "users", alice).unwrap();
    let fields: std::collections::HashMap<String, serde_json::Value> =
        row.fields.into_iter().collect();
    assert_eq!(fields.get("thing"), Some(&serde_json::json!("Hello")));

    // Merge main → dev
    let applied = db.merge_all(main, dev).unwrap();
    assert!(!applied.is_empty(), "should have merged edits");

    // Dev should have the column
    let schema = db.get_table_view(dev, "users").unwrap();
    let cols: Vec<&str> = schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"thing"), "dev should have 'thing' column: {:?}", cols);

    // Dev should have the VALUE too — not just the default
    let row = db.get_row(dev, "users", alice).unwrap();
    let fields: std::collections::HashMap<String, serde_json::Value> =
        row.fields.into_iter().collect();
    assert_eq!(fields.get("thing"), Some(&serde_json::json!("Hello")),
        "dev should have 'Hello' value after merge, not default. Got: {:?}", fields);
}

#[test]
fn schema_and_value_edits_compose_through_merge() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "users", vec![
        ("name", AtomicType::Str),
        ("age", AtomicType::Num),
    ]).unwrap();
    let alice = db.insert_row(main, "users", vec![
        ("name", Value::Str("Alice".into())),
        ("age", Value::Num(30.0)),
    ]).unwrap();

    let dev = db.fork_branch(main, "dev").unwrap();

    // Dev: add column AND set a value
    db.add_column(dev, "users", "email", AtomicType::Str).unwrap();
    db.set_field(dev, "users", alice, "email", Value::Str("alice@test.com".into())).unwrap();

    // Merge to main
    let applied = db.merge_all(dev, main).unwrap();
    assert!(!applied.is_empty(), "should have merged edits: {:?}", applied);

    // Main should have the column AND the value
    let schema = db.get_table_view(main, "users").unwrap();
    let cols: Vec<&str> = schema.columns.iter().map(|(n, _)| n.as_str()).collect();
    assert!(cols.contains(&"email"), "main should have email column");

    let row = db.get_row(main, "users", alice).unwrap();
    let fields: std::collections::HashMap<String, serde_json::Value> =
        row.fields.into_iter().collect();
    assert_eq!(fields.get("email"), Some(&serde_json::json!("alice@test.com")),
        "main should have Alice's email value after merge");
}

#[test]
fn different_rows_different_branches_no_conflict() {
    let mut db = Database::new();
    let main = db.create_branch("main");
    db.create_table(main, "t", vec![("val", AtomicType::Num)]).unwrap();
    let r1 = db.insert_row(main, "t", vec![("val", Value::Num(1.0))]).unwrap();
    let r2 = db.insert_row(main, "t", vec![("val", Value::Num(2.0))]).unwrap();

    let a = db.fork_branch(main, "a").unwrap();
    let b = db.fork_branch(main, "b").unwrap();
    // Need to track a↔b (they're both children of main, not of each other)
    db.diff_branches(a, b).unwrap();

    db.set_field(a, "t", r1, "val", Value::Num(10.0)).unwrap();
    db.set_field(b, "t", r2, "val", Value::Num(20.0)).unwrap();

    // No conflicts — different rows
    let conflicts = db.get_conflicts(a, b);
    let value_conflicts: Vec<_> = conflicts.iter()
        .filter(|(loc, _)| loc.contains("row:"))
        .collect();
    assert!(value_conflicts.is_empty(),
        "different rows should not conflict: {:?}", value_conflicts);

    // Merge both ways
    db.merge_all(a, b).unwrap();
    db.merge_all(b, a).unwrap();

    // Both branches should have both values
    let r1_a = db.get_row(a, "t", r1).unwrap();
    let r1_fields: std::collections::HashMap<String, serde_json::Value> =
        r1_a.fields.into_iter().collect();
    assert_eq!(r1_fields.get("val"), Some(&serde_json::json!(10.0)));

    let r2_b = db.get_row(b, "t", r2).unwrap();
    let r2_fields: std::collections::HashMap<String, serde_json::Value> =
        r2_b.fields.into_iter().collect();
    assert_eq!(r2_fields.get("val"), Some(&serde_json::json!(20.0)));
}
